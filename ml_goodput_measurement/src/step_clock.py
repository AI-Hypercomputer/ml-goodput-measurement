"""Step boundaries from accelerator completion events (see README)."""

import collections
import datetime
import logging
import time
from typing import Callable, Deque, Dict, List, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)

_MS_PER_S = 1000.0


class TimingEvent(Protocol):
  """What the clock needs from `torch.cuda.Event` / `torch.tpu.Event`."""

  def record(self) -> None:
    ...

  def query(self) -> bool:
    ...

  def elapsed_time(self, end: 'TimingEvent') -> float:
    """Milliseconds from this event's completion to `end`'s (may be negative)."""


class StepStartRecorder(Protocol):
  """What the clock needs from `GoodputRecorder`."""

  def record_step_start_time(
      self, step: int, start_time: Optional[datetime.datetime] = None
  ) -> None:
    ...


class StepClock:
  """Records the accelerator finish time of each step as the next step's start.

  Per step, on all ranks: `on_step_end(step)` after issuing the step. On the
  lead rank only: `poll()` once per step, `on_reference_point()` at job start
  and whenever the device is known to be idle (e.g. after a logging step's
  metrics sync), `corrected_start()` after a save or eval, and `flush()` at
  teardown. The trainer records step 1's host start itself; the clock records
  the starts of steps 2..N.

  Any backend error disables the clock after one warning. Pending steps then
  get host-time starts so the step log has no gap, and the trainer should
  record host starts itself while `disabled` is True.
  """

  def __init__(
      self,
      recorder: Optional[StepStartRecorder],
      event_factory: Callable[[], TimingEvent],
      *,
      is_lead: bool = True,
      wall_clock: Callable[[], float] = time.time,
      max_completions: int = 1024,
  ):
    """Initializes the clock.

    Args:
      recorder: Receives the step starts; None on non-lead ranks.
      event_factory: Returns a fresh timing event, e.g.
        `lambda: torch.tpu.Event(enable_timing=True)`.
      is_lead: Whether this rank polls events and records step starts.
      wall_clock: Returns the current time in epoch seconds.
      max_completions: How many recent completion times to keep for
        `completion_time` and `corrected_start`.
    """
    self._recorder = recorder
    self._event_factory = event_factory
    self._is_lead = is_lead
    self._wall_clock = wall_clock
    self._max_completions = max_completions
    self._pending: Deque[Tuple[int, TimingEvent, bool]] = collections.deque()
    self._anchor: Optional[Tuple[TimingEvent, float]] = None
    self._pending_anchor: Optional[Tuple[TimingEvent, float]] = None
    self._completions: Dict[int, float] = {}
    self._last_recorded_start: Optional[float] = None
    self._disabled = False

  def on_reference_point(self) -> None:
    """Pairs a new event with the wall clock; call while the device is idle.

    If the device turns out to be busy, the reference is placed at the first
    `poll` that sees the event complete (error bounded by one poll interval).
    """
    if self._disabled:
      return
    try:
      event = self._event_factory()
      event.record()
      if not self._is_lead:
        return
      wall = self._wall_clock()
      if event.query():
        self._anchor = (event, wall)
        self._pending_anchor = None
      else:
        self._pending_anchor = (event, wall)
    except Exception as e:  # pylint: disable=broad-exception-caught
      self._disable('record reference event', e)

  def on_step_end(self, step: int, *, last_step: bool = False) -> None:
    """Records the end-of-step event. The last step records no step N+1."""
    if self._disabled:
      return
    try:
      event = self._event_factory()
      event.record()
    except Exception as e:  # pylint: disable=broad-exception-caught
      self._disable('record step event', e)
      return
    if self._is_lead:
      self._pending.append((step, event, last_step))

  def poll(self) -> List[Tuple[int, float]]:
    """Records completed steps as the next step's start, without blocking.

    Returns the `(step, start_time)` pairs recorded by this call.
    """
    if self._disabled or not self._is_lead:
      return []
    recorded: List[Tuple[int, float]] = []
    try:
      self._maybe_adopt_anchor()
      if self._anchor is None:
        return recorded
      anchor_event, anchor_wall = self._anchor
      while self._pending:
        step, event, last_step = self._pending[0]
        if not event.query():
          break
        self._pending.popleft()
        done = anchor_wall + anchor_event.elapsed_time(event) / _MS_PER_S
        self._remember_completion(step, done)
        if not last_step:
          recorded.append((step + 1, self._record_start(step + 1, done)))
    except Exception as e:  # pylint: disable=broad-exception-caught
      self._disable('poll', e)
    return recorded

  def flush(self, timeout_s: float = 10.0) -> List[Tuple[int, float]]:
    """Polls until no events are pending or `timeout_s` passes (teardown)."""
    recorded: List[Tuple[int, float]] = []
    if self._disabled or not self._is_lead:
      return recorded
    deadline = self._wall_clock() + timeout_s
    while True:
      recorded.extend(self.poll())
      if not self._pending or self._disabled:
        return recorded
      if self._wall_clock() >= deadline:
        logger.warning(
            'StepClock.flush: %d step event(s) still pending after %.1fs.',
            len(self._pending),
            timeout_s,
        )
        return recorded
      time.sleep(0.001)

  def completion_time(self, step: int) -> Optional[float]:
    """Accelerator finish time of `step` in epoch seconds, if known yet."""
    if step not in self._completions:
      self.poll()
    return self._completions.get(step)

  def corrected_start(self, host_start: float, last_issued_step: int) -> float:
    """`max(host_start, completion_time(last_issued_step))` for a save or eval.

    Returns `host_start` unchanged when the completion is not known.
    """
    done = self.completion_time(last_issued_step)
    if done is None:
      return host_start
    return max(host_start, done)

  @property
  def pending(self) -> int:
    """Step events not yet seen complete (lead rank)."""
    return len(self._pending)

  @property
  def disabled(self) -> bool:
    """True after a backend error; every method is a no-op from then on."""
    return self._disabled

  def _maybe_adopt_anchor(self) -> None:
    if self._pending_anchor is None:
      return
    event, recorded_at = self._pending_anchor
    if event.query():
      observed_at = self._wall_clock()
      logger.debug(
          'StepClock: reference recorded at %.3f seen complete at %.3f.',
          recorded_at,
          observed_at,
      )
      self._anchor = (event, observed_at)
      self._pending_anchor = None

  def _remember_completion(self, step: int, done: float) -> None:
    self._completions[step] = done
    while len(self._completions) > self._max_completions:
      self._completions.pop(next(iter(self._completions)))

  def _record_start(self, step: int, start: float) -> float:
    """Records `start` for `step`, never earlier than the previous start."""
    if self._last_recorded_start is not None:
      start = max(start, self._last_recorded_start)
    self._last_recorded_start = start
    if self._recorder is not None:
      self._recorder.record_step_start_time(
          step, datetime.datetime.fromtimestamp(start, tz=datetime.timezone.utc)
      )
    return start

  def _disable(self, what: str, error: Exception) -> None:
    self._disabled = True
    logger.warning(
        'StepClock: failed to %s (%s: %s); falling back to host step starts.',
        what,
        type(error).__name__,
        error,
    )
    pending, self._pending = list(self._pending), collections.deque()
    try:
      now = self._wall_clock()
      for step, _, last_step in pending:
        if not last_step:
          self._record_start(step + 1, now)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.warning('StepClock: could not backfill pending steps: %s', e)
