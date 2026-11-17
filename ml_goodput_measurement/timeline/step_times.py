"""Step-time helpers shared by the timeline builder, metrics and renderer.

The step time of step n is the time from its start to the start of step n + 1.
A workload's baseline (steady-state) step time is the lower median of them.
Elastic workloads also log their slice counts: their baseline only counts the
steps that ran with all slices active, and a step with fewer active slices is
expected to take what the steps with the same slice counts took; see Baseline.
"""

import bisect
import collections
import dataclasses
from typing import Optional, Sequence

# A delta between consecutive steps that is this long or longer is not a step
# time: it spans an interruption, a slow recovery or a stall.
MAX_STEP_DELTA_S = 600.0

# A step with fewer active slices only has an expected time when at least this
# many steps ran with the same slice counts; see Baseline.
MIN_STEPS_FOR_EXPECTED = 5

# Elastic slice count reports, (timestamp, active, available, total), by time;
# see event_parser.ParsedLogs.elastic_slice_counts.
SliceCounts = Sequence[tuple[float, int, int, int]]


@dataclasses.dataclass(frozen=True)
class StepDelta:
  """The time from the start of a step to the start of the next one."""

  step: int  # The earlier step.
  start: float  # Its start time, Unix seconds.
  delta_s: float  # Time to the start of step + 1.

  @property
  def end(self) -> float:
    """The start of step + 1."""
    return self.start + self.delta_s


def consecutive_step_deltas(
    steps: Sequence[tuple[int, float]],
) -> list[StepDelta]:
  """Returns the step time of every step that is followed by the next step.

  A pair only counts when the step numbers are consecutive and the delta is
  positive and below MAX_STEP_DELTA_S, which leaves out repeated steps after a
  rollback and gaps across interruptions.

  Args:
    steps: (step number, start time) pairs, by time.

  Returns:
    One StepDelta per qualifying pair, in the given order.
  """
  deltas = []
  for (s_prev, t_prev), (s_curr, t_curr) in zip(steps, steps[1:]):
    d = t_curr - t_prev
    if s_curr == s_prev + 1 and 0 < d < MAX_STEP_DELTA_S:
      deltas.append(StepDelta(step=s_prev, start=t_prev, delta_s=d))
  return deltas


def lower_median(values: Sequence[float]) -> float:
  """The lower median of values, or 0.0 if there are none."""
  if not values:
    return 0.0
  return sorted(values)[(len(values) - 1) // 2]


class Baseline:
  """A workload's steady-state step time, and the expected time of any step.

  Without slice counts (McJAX and non-elastic workloads), the baseline is the
  lower median of all step times and every step is expected to take it. With
  slice counts, the baseline is the lower median of the steps that ran with
  all slices active, or of all steps if none did. A step with fewer active
  slices is expected to take the lower median of the steps that ran with the
  same (active, total) slices throughout. That time is measured rather than
  scaled from the baseline because workloads differ: with a fixed per-device
  batch, fewer slices keep the step time (and shrink the global batch); with a
  fixed global batch, the step time grows by total / active. With fewer than
  MIN_STEPS_FOR_EXPECTED such steps, the expected time is unknown. A step's
  slices are the fewest active slices during it; a step that starts before the
  first slice count report is expected to take the baseline.

  Attributes:
    step_time_s: The baseline step time, or 0.0 if no two consecutive steps
      are known.
  """

  def __init__(
      self,
      steps: Sequence[tuple[int, float]],
      slice_counts: SliceCounts = (),
  ):
    """Computes the baseline and the expected step times.

    Args:
      steps: (step number, start time) pairs, by time.
      slice_counts: The elastic slice count reports, if any.
    """
    self._counts = list(slice_counts)
    self._times = [c[0] for c in self._counts]
    deltas = consecutive_step_deltas(steps)
    # The step times of the steps that ran with the same slices throughout.
    by_slices = collections.defaultdict(list)
    for d in deltas:
      slices = self._steady_slices(d.start, d.end)
      if slices:
        by_slices[slices].append(d.delta_s)
    full = [
        t
        for (active, total), times in by_slices.items()
        if active == total
        for t in times
    ]
    self.step_time_s = lower_median(full or [d.delta_s for d in deltas])
    # The expected step time with fewer active slices, by (active, total).
    self._expected = {
        slices: lower_median(times)
        for slices, times in by_slices.items()
        if slices[0] < slices[1] and len(times) >= MIN_STEPS_FOR_EXPECTED
    }

  def _in_effect(
      self, start: float, end: float
  ) -> Sequence[tuple[float, int, int, int]]:
    """The slice count reports in effect at some time from start to end.

    Args:
      start: Unix seconds.
      end: Unix seconds, >= start.

    Returns:
      The last report at or before start, then the reports before end; none
      if no report is at or before start.
    """
    i = bisect.bisect_right(self._times, start)
    if not i:
      return []
    return self._counts[i - 1 : max(i, bisect.bisect_left(self._times, end))]

  def _steady_slices(
      self, start: float, end: float
  ) -> Optional[tuple[int, int]]:
    """(active, total) slices if they stayed the same from start to end.

    Args:
      start: Unix seconds.
      end: Unix seconds, >= start.

    Returns:
      The counts, or None when they are unknown, changed or had no active
      slice.
    """
    counts = {
        (active, total) for _, active, _, total in self._in_effect(start, end)
    }
    if len(counts) != 1:
      return None
    ((active, total),) = counts
    return (active, total) if active > 0 else None

  def full_capacity(self, start: float, end: float) -> bool:
    """Whether the slice counts say that all slices were active throughout."""
    slices = self._steady_slices(start, end)
    return slices is not None and slices[0] == slices[1]

  def slices(self, start: float, end: float) -> Optional[tuple[int, int]]:
    """(active, total) slices for the fewest active slices from start to end.

    Reports without active slices (while slices are replaced) are left out.

    Args:
      start: Unix seconds.
      end: Unix seconds, >= start.

    Returns:
      The counts, or None when they are unknown.
    """
    counts = [c for c in self._in_effect(start, end) if c[1] > 0]
    if not counts:
      return None
    _, active, _, total = min(counts, key=lambda c: c[1])
    return active, total

  def expected_s(self, start: float, end: float) -> Optional[float]:
    """The expected time of a step from start to end; see the class doc.

    Args:
      start: Unix seconds.
      end: Unix seconds, >= start.

    Returns:
      The baseline, or for a step with fewer active slices the step time
      measured with them; None if fewer than MIN_STEPS_FOR_EXPECTED steps ran
      with them.
    """
    slices = self.slices(start, end)
    if slices is None or slices[0] >= slices[1]:
      return self.step_time_s
    return self._expected.get(slices)


def baseline_step_time(
    steps: Sequence[tuple[int, float]], slice_counts: SliceCounts = ()
) -> float:
  """The steady-state step time, or 0.0 if it is unknown; see Baseline."""
  return Baseline(steps, slice_counts).step_time_s


def overlapping(
    intervals: Sequence[tuple[float, float]], deltas: Sequence[StepDelta]
) -> list[Optional[tuple[float, float]]]:
  """For every step, an interval that overlaps it, e.g. a checkpoint save.

  Args:
    intervals: (start, end) pairs, in any order.
    deltas: The steps.

  Returns:
    One entry per step: the latest-starting interval that overlaps it, or
    None.
  """
  items = sorted(intervals)
  starts = [s for s, _ in items]
  max_ends = []
  for _, e in items:
    max_ends.append(max(e, max_ends[-1]) if max_ends else e)
  hits: list[Optional[tuple[float, float]]] = []
  for d in deltas:
    # The intervals that start before the step ends; one of them overlaps the
    # step if the latest end among them is after the step starts.
    i = bisect.bisect_left(starts, d.end)
    if not i or max_ends[i - 1] <= d.start:
      hits.append(None)
      continue
    i -= 1
    while items[i][1] <= d.start:
      i -= 1
    hits.append(items[i])
  return hits
