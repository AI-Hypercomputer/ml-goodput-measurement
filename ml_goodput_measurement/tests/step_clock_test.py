"""Tests for the device-event step clock."""

import datetime
from unittest import mock

from absl.testing import absltest
from cloud_goodput.ml_goodput_measurement.src import step_clock


class _FakeDevice:
  """A device timeline: events complete when `now` passes their finish time."""

  def __init__(self, skew=0.0):
    self.now = 0.0  # wall clock, epoch seconds
    self.next_finish = None  # finish time assigned to the next recorded event
    self.fail_record = False
    # Device clock runs (1 + skew) times as fast as the wall clock, so
    # elapsed_time between distant events accumulates error.
    self.skew = skew

  def event(self):
    return _FakeEvent(self)


class _FakeEvent:

  def __init__(self, device):
    self._device = device
    self.finish = None  # device completion time (epoch seconds)

  def record(self):
    if self._device.fail_record:
      raise RuntimeError('backend down')
    self.finish = self._device.next_finish
    if self.finish is None:
      # Nothing queued: completes immediately (device idle).
      self.finish = self._device.now

  def query(self):
    return self.finish is not None and self._device.now >= self.finish

  def elapsed_time(self, end):
    return (end.finish - self.finish) * (1.0 + self._device.skew) * 1000.0


class _FakeRecorder:

  def __init__(self):
    self.starts = []  # (step, epoch seconds)

  def record_step_start_time(self, step, start_time=None):
    assert start_time is not None and start_time.tzinfo is not None
    self.starts.append((step, start_time.timestamp()))


class StepClockTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.device = _FakeDevice()
    self.recorder = _FakeRecorder()
    self.clock = step_clock.StepClock(
        self.recorder, self.device.event, wall_clock=lambda: self.device.now
    )

  def _issue(self, step, finish, last_step=False):
    """Issues `step` whose device work finishes at `finish`."""
    self.device.next_finish = finish
    self.clock.on_step_end(step, last_step=last_step)
    self.device.next_finish = None

  def test_completions_become_next_step_starts(self):
    self.device.now = 100.0
    self.clock.on_reference_point()
    self._issue(1, finish=110.0)
    self._issue(2, finish=112.0)
    self.device.now = 113.0
    recorded = self.clock.poll()
    self.assertEqual(recorded, [(2, 110.0), (3, 112.0)])
    self.assertEqual(self.recorder.starts, [(2, 110.0), (3, 112.0)])
    self.assertEqual(self.clock.completion_time(1), 110.0)

  def test_host_run_ahead_records_in_order_as_events_complete(self):
    self.device.now = 0.0
    self.clock.on_reference_point()
    for step in range(1, 6):  # host issues 5 steps, device needs 1s each
      self._issue(step, finish=float(step))
    self.device.now = 2.5
    self.assertEqual(self.clock.poll(), [(2, 1.0), (3, 2.0)])
    self.assertEqual(self.clock.pending, 3)
    self.device.now = 10.0
    self.assertEqual(self.clock.poll(), [(4, 3.0), (5, 4.0), (6, 5.0)])
    self.assertEqual(self.clock.pending, 0)

  def test_sawtooth_host_timestamps_become_even_device_steps(self):
    # Host issues a step every 22 ms; the device needs 100 ms per step. Host
    # timestamps would give 22 ms steps plus one long step per sync; the clock
    # must give 100 ms steps within 1%.
    self.device.now = 1000.0
    self.clock.on_reference_point()
    for step in range(1, 51):
      self.device.now = 1000.0 + 0.022 * step
      self._issue(step, finish=1000.0 + 0.1 * step)
      self.clock.poll()
    self.device.now = 1010.0
    self.clock.poll()
    starts = [t for _, t in self.recorder.starts]
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    self.assertLen(starts, 50)
    for gap in gaps:
      self.assertAlmostEqual(gap, 0.1, delta=0.001)

  def test_reference_point_taken_while_busy_is_placed_at_first_complete_poll(self):
    self.device.now = 0.0
    self.device.next_finish = 5.0  # device busy: the anchor completes at 5s
    self.clock.on_reference_point()
    self.device.next_finish = None
    self._issue(1, finish=6.0)
    self.device.now = 4.0
    self.assertEqual(self.clock.poll(), [])  # no usable reference yet
    self.device.now = 5.0  # a poll at the completion time sees it exactly
    self.assertEqual(self.clock.poll(), [])
    self.device.now = 7.0
    self.assertEqual(self.clock.poll(), [(2, 6.0)])

  def test_reference_point_while_idle_uses_record_time(self):
    self.device.now = 10.0
    self.clock.on_reference_point()  # idle: completes immediately
    self._issue(1, finish=10.5)
    self.device.now = 12.0  # polled late; the record time is still exact
    self.assertEqual(self.clock.poll(), [(2, 10.5)])

  def test_later_reference_point_bounds_device_clock_drift(self):
    self.device.skew = 0.01  # device clock 1% fast
    self.device.now = 0.0
    self.clock.on_reference_point()
    self._issue(1, finish=1.0)
    self.device.now = 1.0
    self.clock.poll()
    # 50s later, relative to the first reference the error is 0.5s; a new
    # reference point taken while idle resets it to 1% of one step.
    self.device.now = 50.0
    self.clock.on_reference_point()
    self._issue(2, finish=51.0)
    self.device.now = 52.0
    ((step, start),) = self.clock.poll()
    self.assertEqual(step, 3)
    self.assertAlmostEqual(start, 51.0, delta=0.02)

  def test_last_step_records_no_phantom_step(self):
    self.device.now = 0.0
    self.clock.on_reference_point()
    self._issue(9, finish=1.0)
    self._issue(10, finish=2.0, last_step=True)
    self.device.now = 3.0
    self.assertEqual(self.clock.poll(), [(10, 1.0)])
    self.assertEqual(self.clock.completion_time(10), 2.0)
    self.assertEqual([s for s, _ in self.recorder.starts], [10])

  def test_corrected_start_moves_save_past_queued_compute(self):
    self.device.now = 0.0
    self.clock.on_reference_point()
    self._issue(5, finish=8.0)  # device still busy when the host starts saving
    self.device.now = 9.0  # the save (which waits for the device) has returned
    self.assertEqual(self.clock.corrected_start(host_start=3.0, last_issued_step=5), 8.0)
    # Host started the save after the device was already done: unchanged.
    self.assertEqual(self.clock.corrected_start(host_start=8.5, last_issued_step=5), 8.5)
    # Unknown step: unchanged.
    self.assertEqual(self.clock.corrected_start(host_start=3.0, last_issued_step=99), 3.0)

  def test_starts_never_go_backwards(self):
    self.device.now = 0.0
    self.clock.on_reference_point()
    self._issue(1, finish=2.0)
    self._issue(2, finish=1.9)  # stamp noise: earlier than the previous step
    self.device.now = 3.0
    self.assertEqual(self.clock.poll(), [(2, 2.0), (3, 2.0)])

  def test_non_lead_rank_records_events_but_no_starts(self):
    recorder = _FakeRecorder()
    clock = step_clock.StepClock(
        recorder, self.device.event, is_lead=False, wall_clock=lambda: 0.0
    )
    clock.on_reference_point()
    clock.on_step_end(1)
    self.assertEqual(clock.poll(), [])
    self.assertEqual(clock.pending, 0)
    self.assertEqual(recorder.starts, [])

  def test_backend_error_disables_clock_and_backfills_pending_steps(self):
    self.device.now = 0.0
    self.clock.on_reference_point()
    self._issue(1, finish=10.0)
    self._issue(2, finish=11.0)
    self.device.now = 5.0
    self.device.fail_record = True
    with mock.patch.object(step_clock.logger, 'warning') as warning:
      self.clock.on_step_end(3)  # raises inside; must not propagate
    self.assertTrue(self.clock.disabled)
    warning.assert_called()
    # Steps 1 and 2 were pending: their successors get host-time starts so the
    # step sequence stays contiguous; step 3's successor is the trainer's job.
    self.assertEqual(self.recorder.starts, [(2, 5.0), (3, 5.0)])
    self.assertEqual(self.clock.poll(), [])
    self.assertEqual(self.clock.corrected_start(1.0, 1), 1.0)

  def test_flush_waits_for_pending_events_and_times_out(self):
    self.device.now = 0.0
    self.clock.on_reference_point()
    self._issue(1, finish=0.002)
    self._issue(2, finish=100.0)

    real_sleep = step_clock.time.sleep

    def advance(seconds):
      self.device.now += 0.01
      real_sleep(0)

    with mock.patch.object(step_clock.time, 'sleep', side_effect=advance):
      recorded = self.clock.flush(timeout_s=0.05)
    self.assertEqual(recorded, [(2, 0.002)])
    self.assertEqual(self.clock.pending, 1)

  def test_recorded_times_are_utc_datetimes(self):
    self.device.now = 1700000000.0
    self.clock.on_reference_point()
    self._issue(1, finish=1700000001.5)
    self.device.now = 1700000002.0
    with mock.patch.object(
        self.recorder,
        'record_step_start_time',
        wraps=self.recorder.record_step_start_time,
    ) as record:
      self.clock.poll()
    (call,) = record.call_args_list
    step, start_time = call.args
    self.assertEqual(step, 2)
    self.assertEqual(start_time.tzinfo, datetime.timezone.utc)
    self.assertAlmostEqual(start_time.timestamp(), 1700000001.5)


if __name__ == '__main__':
  absltest.main()
