"""Goodput tests to validate Recorder, Calculator and Logger classes."""

import dataclasses
from dataclasses import asdict
import datetime
import random
import time
from typing import Optional

from ml_goodput_measurement.src import goodput
from ml_goodput_measurement.src.goodput_utils import BadputType
from ml_goodput_measurement.src.goodput_utils import compute_ideal_step_time, get_timestamp_from_log_entry

from google3.testing.pybase import googletest


# Fake job timeline information for test purposes.
_TEST_JOB_START_TIME = datetime.datetime(
    year=2024, month=1, day=1, hour=1, minute=0, second=0, microsecond=0
)
_TEST_PROGRAM_STARTUP_TIME = datetime.timedelta(seconds=5)
_TEST_TPU_INIT_TIME = datetime.timedelta(seconds=1)
_TEST_TRAINING_PREPARATION_TIME = datetime.timedelta(seconds=2)
_TEST_DATA_LOADING_TIME = datetime.timedelta(seconds=2)
_TEST_STEP_START_TIME = _TEST_JOB_START_TIME + _TEST_PROGRAM_STARTUP_TIME
_TEST_TOTAL_STEPS = 5
_TEST_STEP_TIME = datetime.timedelta(seconds=3)
_TEST_JOB_END_TIME = _TEST_STEP_START_TIME + _TEST_STEP_TIME * _TEST_TOTAL_STEPS
# Badput time included in the first step time after start and restart.
_TEST_FIRST_STEP_EXTRA_TIME = datetime.timedelta(seconds=5)
# Anomalous large step times
_TEST_ANOMALOUS_STEP_TIME = datetime.timedelta(seconds=30)


class MockCloudLogger:

  def __init__(self, job_name, logger_name):
    self.job_name = job_name
    self.logger_name = logger_name
    self.entries = []

  def write_cloud_logging_entry(self, entry):
    timestamp = get_timestamp_from_log_entry(entry)
    if timestamp is not None:
      self.entries.append((timestamp, entry))

  def read_cloud_logging_entries(self, start_time=None, end_time=None):
    return [
        entry
        for timestamp, entry in self.entries
        if (start_time is None or timestamp >= start_time)
        and (end_time is None or timestamp <= end_time)
    ]


@dataclasses.dataclass
class MockSaveStepStatistics:
  """Attributes for save step statistics.

  Attributes:
    step: The step number.
    event_type: The event type.
    checkpoint_manager_blocking_start_time: The start time of checkpoint manager
      blocking section.
    directory: The directory of the checkpoint.
    reached_preemption: Whether the event reached preemption.
    preemption_received_at: The time when preemption was received.
    wait_for_prev_start_time: The start time of waiting for previous checkpoint.
    checkpointer_blocking_start_time: The start time of blocking time introduced
      by checkpointer.
    get_old_steps_start_time: The start time of getting old steps.
    synchronous: Whether the event is synchronous.
    wait_for_prev_duration_secs: The duration of waiting for previous
      checkpoint.
    checkpointer_blocking_duration_secs: The duration of blocking time
      introduced by checkpointer.
    get_old_steps_duration_secs: The duration of getting old steps.
    checkpoint_manager_blocking_duration_secs: The duration of checkpoint
      manager blocking section.
  """

  step: Optional[int] = None
  event_type: Optional[str] = 'save'
  directory: Optional[str] = None
  reached_preemption: Optional[bool] = False
  preemption_received_at: Optional[float] = None
  synchronous: Optional[bool] = False
  wait_for_prev_start_time: Optional[float] = None
  wait_for_prev_duration_secs: Optional[float] = None
  checkpointer_blocking_start_time: Optional[float] = None
  checkpointer_blocking_duration_secs: Optional[float] = None
  get_old_steps_start_time: Optional[float] = None
  get_old_steps_duration_secs: Optional[float] = None
  checkpoint_manager_blocking_start_time: Optional[float] = None
  checkpoint_manager_blocking_duration_secs: Optional[float] = None


@dataclasses.dataclass
class MockRestoreStepStatistics:
  """Attributes for restore step statistics.

  Attributes:
    step: The step number.
    event_type: The event type.
    directory: The directory of the checkpoint.
    checkpointer_start_time: The start time of restoring the checkpoint, while
      using the checkpointer.
    checkpointer_duration_secs: The total duration for restoring the checkpoint,
      while using the checkpointer.
    checkpoint_manager_start_time: The start time for restoring the checkpoint,
      while using the checkpoint manager.
    checkpoint_manager_duration_secs: The total duration for restoring the
      checkpoint, while using the checkpoint manager.
  """

  step: Optional[int] = None
  event_type: Optional[str] = 'restore'
  directory: Optional[str] = None
  checkpointer_start_time: Optional[float] = None
  checkpointer_duration_secs: Optional[float] = None
  checkpoint_manager_start_time: Optional[float] = None
  checkpoint_manager_duration_secs: Optional[float] = None


class GoodputTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    self.job_name = 'test-run'
    self.logger_name = 'test-log'
    self.mock_cloud_logger = MockCloudLogger(self.job_name, self.logger_name)
    self.goodput_recorder = goodput.GoodputRecorder(
        self.job_name,
        self.logger_name,
        True,
        self.mock_cloud_logger,
    )
    self.goodput_calculator = goodput.GoodputCalculator(
        self.job_name, self.logger_name, self.mock_cloud_logger
    )

  def _mock_sample_program(self):
    # Record job start time of the job: use a fake timestamp
    self.goodput_recorder.record_job_start_time(_TEST_JOB_START_TIME)

    # Mock _TEST_TOTAL_STEPS steps of training
    step_start_time = _TEST_STEP_START_TIME
    for step in range(_TEST_TOTAL_STEPS):
      # Record step time
      self.goodput_recorder.record_step_start_time(step, step_start_time)
      step_start_time += _TEST_STEP_TIME

    # Record job end time
    self.goodput_recorder.record_job_end_time(_TEST_JOB_END_TIME)

  def test_goodput_recorder(self):
    """Test function to validate goodput recorder and logger."""
    # Emulate job run timeline.
    self._mock_sample_program()

    # Ensure read returns the right number of entries.
    validate_entries = self.mock_cloud_logger.read_cloud_logging_entries()
    # There should be one entry for each of the 5 steps, one job start
    # and one job end entry.
    self.assertLen(validate_entries, _TEST_TOTAL_STEPS + 2)
    # Ensure payload contains the expected information.
    for entry_payload in validate_entries:
      self.assertIn(goodput._JOB_NAME, entry_payload)
      self.assertEqual(entry_payload[goodput._JOB_NAME], self.job_name)
      if goodput._JOB_START_TIME in entry_payload:
        self.assertEqual(
            entry_payload[goodput._JOB_START_TIME],
            _TEST_JOB_START_TIME.timestamp(),
        )
      if goodput._JOB_END_TIME in entry_payload:
        self.assertEqual(
            entry_payload[goodput._JOB_END_TIME],
            _TEST_JOB_END_TIME.timestamp(),
        )
      if goodput._STEP_START_TIME in entry_payload:
        step_count = entry_payload[goodput._STEP_COUNT]
        expected_start_start_time = (
            _TEST_STEP_START_TIME + _TEST_STEP_TIME * step_count
        )
        self.assertEqual(
            entry_payload[goodput._STEP_START_TIME],
            expected_start_start_time.timestamp(),
        )

  def test_goodput_calculator(self):
    """Test function to validate goodput calculator."""
    # Emulate job run timeline.
    self._mock_sample_program()
    # Get the computed Goodput from the library and compare with expected
    # result.
    computed_goodput, _, total_steps = self.goodput_calculator.get_job_goodput()
    expected_goodput = (
        (_TEST_STEP_TIME * _TEST_TOTAL_STEPS)
        / (_TEST_JOB_END_TIME - _TEST_JOB_START_TIME)
        * 100
    )
    self.assertEqual(computed_goodput, expected_goodput)
    self.assertEqual(total_steps, _TEST_TOTAL_STEPS - 1)

  def test_goodput_with_startup_badput(self):
    """Test function to validate goodput with startup badput."""

    job_start_time = datetime.datetime.utcnow()
    self.goodput_recorder.record_job_start_time(job_start_time)

    # Mock _TEST_TOTAL_STEPS steps of training
    step_start_time = job_start_time + _TEST_PROGRAM_STARTUP_TIME

    # All steps but first progress with average step time.
    for step in range(_TEST_TOTAL_STEPS):
      # Record step time
      self.goodput_recorder.record_step_start_time(step, step_start_time)
      step_start_time += _TEST_STEP_TIME
      # Introduce startup badput during the first step
      if step == 0:
        step_start_time += _TEST_FIRST_STEP_EXTRA_TIME

    total_time = (
        _TEST_PROGRAM_STARTUP_TIME
        + _TEST_STEP_TIME * _TEST_TOTAL_STEPS
        + _TEST_FIRST_STEP_EXTRA_TIME
    )
    job_end_time = job_start_time + total_time
    self.goodput_recorder.record_job_end_time(job_end_time)

    # Get the computed Goodput from the library and compare with expected
    # result.

    computed_goodput, _, _ = self.goodput_calculator.get_job_goodput()
    expected_goodput = (
        (_TEST_TOTAL_STEPS * _TEST_STEP_TIME.total_seconds())
        / total_time.total_seconds()
        * 100
    )

    self.assertAlmostEqual(computed_goodput, expected_goodput, delta=0.1)


class GoodputDisruptionCompleteRestartTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    self.job_name = 'test-run'
    self.logger_name = 'test-log'
    self.mock_cloud_logger = MockCloudLogger(self.job_name, self.logger_name)
    self.goodput_recorder = goodput.GoodputRecorder(
        self.job_name,
        self.logger_name,
        True,
        self.mock_cloud_logger,
    )
    self.goodput_calculator = goodput.GoodputCalculator(
        self.job_name, self.logger_name, self.mock_cloud_logger
    )

  def test_goodput_calculator(self):
    """Test function to validate goodput calculator."""
    # It is not ideal to use non-deterministic timestamps in unit tests, but
    # testing this complex scenario using deterministic timestamps is not
    # straightforward.
    # TODO(xfgu): Refactor this test.
    job_start_time = datetime.datetime.utcnow()
    self.goodput_recorder.record_job_start_time(job_start_time)

    # Mock _TEST_TOTAL_STEPS steps of training
    step_start_time = job_start_time + _TEST_PROGRAM_STARTUP_TIME
    for step in range(_TEST_TOTAL_STEPS):
      # Record step time
      self.goodput_recorder.record_step_start_time(step, step_start_time)
      step_start_time += _TEST_STEP_TIME

    # Simulate a 30-second disruption.
    disruption_time = datetime.timedelta(seconds=30)
    job_start_time = step_start_time + disruption_time
    self.goodput_recorder.record_job_start_time(job_start_time)
    step_start_time = job_start_time + _TEST_PROGRAM_STARTUP_TIME

    steps_before_query = _TEST_TOTAL_STEPS - 2
    for step in range(steps_before_query):
      self.goodput_recorder.record_step_start_time(step, step_start_time)
      step_start_time += _TEST_STEP_TIME

    # Get the computed Goodput from the library and compare with expected
    # result.

    # The time from when the job first started to when the last step start was
    # logged.
    total_time = (
        _TEST_PROGRAM_STARTUP_TIME
        + _TEST_STEP_TIME * _TEST_TOTAL_STEPS
        + disruption_time
        + _TEST_PROGRAM_STARTUP_TIME
        + (steps_before_query - 1) * _TEST_STEP_TIME
    )
    seconds_before_query = 2
    query_time = total_time.total_seconds() + seconds_before_query

    time.sleep(query_time)
    computed_goodput, _, _ = self.goodput_calculator.get_job_goodput()
    expected_goodput = (
        (
            (steps_before_query - 1) * _TEST_STEP_TIME.total_seconds()
            + seconds_before_query
        )
        / query_time
        * 100
    )

    self.assertAlmostEqual(computed_goodput, expected_goodput, delta=0.1)


class GoodputDisruptionPartialRestartTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    self.job_name = 'test-run'
    self.logger_name = 'test-log'
    self.mock_cloud_logger = MockCloudLogger(self.job_name, self.logger_name)
    self.goodput_recorder = goodput.GoodputRecorder(
        self.job_name,
        self.logger_name,
        True,
        self.mock_cloud_logger,
    )
    self.goodput_calculator = goodput.GoodputCalculator(
        self.job_name, self.logger_name, self.mock_cloud_logger
    )

  def test_goodput_calculator(self):
    """Test function to validate goodput calculator."""
    # It is not ideal to use non-deterministic timestamps in unit tests, but
    # testing this complex scenario using deterministic timestamps is not
    # straightforward.
    # TODO(xfgu): Refactor this test.
    job_start_time = datetime.datetime.utcnow()
    self.goodput_recorder.record_job_start_time(job_start_time)

    # Mock _TEST_TOTAL_STEPS steps of training
    step_start_time = job_start_time + _TEST_PROGRAM_STARTUP_TIME
    for step in range(_TEST_TOTAL_STEPS):
      # Record step time
      self.goodput_recorder.record_step_start_time(step, step_start_time)
      step_start_time += _TEST_STEP_TIME

    # Simulate a 30-second disruption.
    disruption_time = datetime.timedelta(seconds=30)
    job_start_time = step_start_time + disruption_time
    self.goodput_recorder.record_job_start_time(job_start_time)
    step_start_time = job_start_time + _TEST_PROGRAM_STARTUP_TIME

    restart_from_step = 2
    for step in range(restart_from_step, _TEST_TOTAL_STEPS):
      self.goodput_recorder.record_step_start_time(step, step_start_time)
      step_start_time += _TEST_STEP_TIME

    # Get the computed Goodput from the library and compare with expected
    # result.

    # The time from when the job first started to when the last step start was
    # logged.
    total_time = (
        _TEST_PROGRAM_STARTUP_TIME
        + _TEST_STEP_TIME * _TEST_TOTAL_STEPS
        + disruption_time
        + _TEST_PROGRAM_STARTUP_TIME
        + (_TEST_TOTAL_STEPS - restart_from_step) * _TEST_STEP_TIME
    )
    seconds_before_query = 2
    query_time = total_time.total_seconds() + seconds_before_query

    time.sleep(query_time)
    computed_goodput, _, _ = self.goodput_calculator.get_job_goodput()
    expected_goodput = (
        (
            _TEST_TOTAL_STEPS * _TEST_STEP_TIME.total_seconds()
            + seconds_before_query
        )
        / query_time
        * 100
    )

    self.assertAlmostEqual(computed_goodput, expected_goodput, delta=0.1)

  def test_goodput_with_startup_badput(self):
    """Test function to validate goodput with startup badput."""

    job_start_time = datetime.datetime.utcnow()
    self.goodput_recorder.record_job_start_time(job_start_time)

    # Mock _TEST_TOTAL_STEPS steps of training
    step_start_time = job_start_time + _TEST_PROGRAM_STARTUP_TIME

    # All steps but first progress with average step time.
    for step in range(0, _TEST_TOTAL_STEPS):
      # Record step time
      self.goodput_recorder.record_step_start_time(step, step_start_time)
      step_start_time += _TEST_STEP_TIME
      # Introduce startup badput during the first step
      if step == 0:
        step_start_time += _TEST_FIRST_STEP_EXTRA_TIME

    # Simulate a 30-second disruption.
    disruption_time = datetime.timedelta(seconds=30)
    job_start_time = step_start_time + disruption_time
    self.goodput_recorder.record_job_start_time(job_start_time)
    step_start_time = job_start_time + _TEST_PROGRAM_STARTUP_TIME

    restart_from_step = 2
    # All steps but first progress with average step time.
    for step in range(restart_from_step, _TEST_TOTAL_STEPS):
      self.goodput_recorder.record_step_start_time(step, step_start_time)
      step_start_time += _TEST_STEP_TIME
      # Introduce badput during the first step after restart
      if step == restart_from_step:
        step_start_time += _TEST_FIRST_STEP_EXTRA_TIME

    # Get the computed Goodput from the library and compare with expected
    # result.

    # The time from when the job first started to when the last step start was
    # logged.
    total_time = (
        _TEST_PROGRAM_STARTUP_TIME
        + _TEST_STEP_TIME * _TEST_TOTAL_STEPS
        + _TEST_FIRST_STEP_EXTRA_TIME
        + disruption_time
        + _TEST_PROGRAM_STARTUP_TIME
        + (_TEST_TOTAL_STEPS - restart_from_step) * _TEST_STEP_TIME
        + _TEST_FIRST_STEP_EXTRA_TIME
    )
    seconds_before_query = 2
    query_time = total_time.total_seconds() + seconds_before_query

    time.sleep(query_time)
    computed_goodput, _, _ = self.goodput_calculator.get_job_goodput()
    expected_goodput = (
        (
            _TEST_TOTAL_STEPS * _TEST_STEP_TIME.total_seconds()
            + seconds_before_query
        )
        / query_time
        * 100
    )

    self.assertAlmostEqual(computed_goodput, expected_goodput, delta=0.1)


class GoodputPathwaysTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    self.job_name = 'test-run'
    self.logger_name = 'test-log'
    self.mock_cloud_logger = MockCloudLogger(self.job_name, self.logger_name)
    self.goodput_recorder = goodput.GoodputRecorder(
        self.job_name,
        self.logger_name,
        True,
        self.mock_cloud_logger,
    )
    self.goodput_calculator = goodput.GoodputCalculator(
        self.job_name, self.logger_name, self.mock_cloud_logger, True
    )

  def test_goodput_with_anomalous_steps_single_disruption(self):
    """Test function to validate goodput with anomalous step times due to a single disruption."""
    # This test simulates _TEST_TOTAL_STEPS training steps and a single
    # disruption during the job's run time as follows:
    # [0, 1, 2, Handled Disruption, 3, 4]
    # The handled disruption will manifest as anomalously large step times.

    job_start_time = datetime.datetime.utcnow()
    self.goodput_recorder.record_job_start_time(job_start_time)

    # Mock some program startup time before the training steps
    step_start_time = job_start_time + _TEST_PROGRAM_STARTUP_TIME

    # First few steps progress with normal step time.
    for step in range(_TEST_TOTAL_STEPS - 3):
      # Record step time
      self.goodput_recorder.record_step_start_time(step, step_start_time)
      step_start_time += _TEST_STEP_TIME

    # Introduce an anomalously large step time due to a disruption.
    self.goodput_recorder.record_step_start_time(
        _TEST_TOTAL_STEPS - 3, step_start_time
    )
    step_start_time += _TEST_ANOMALOUS_STEP_TIME + _TEST_STEP_TIME

    # Remaining steps progress with normal step time.
    for step in range(_TEST_TOTAL_STEPS - 2, _TEST_TOTAL_STEPS):
      # Record step time
      self.goodput_recorder.record_step_start_time(step, step_start_time)
      step_start_time += _TEST_STEP_TIME

    job_end_time = (
        job_start_time
        + _TEST_PROGRAM_STARTUP_TIME
        + _TEST_ANOMALOUS_STEP_TIME
        + _TEST_STEP_TIME * _TEST_TOTAL_STEPS
    )
    self.goodput_recorder.record_job_end_time(job_end_time)

    # The time from when the job first started to when the last step start was
    # logged.
    total_time = (
        _TEST_PROGRAM_STARTUP_TIME
        + _TEST_STEP_TIME * _TEST_TOTAL_STEPS
        + _TEST_ANOMALOUS_STEP_TIME
    )

    computed_goodput, _, _ = self.goodput_calculator.get_job_goodput()
    expected_goodput = (
        (_TEST_TOTAL_STEPS * _TEST_STEP_TIME.total_seconds())
        / total_time.total_seconds()
        * 100
    )

    self.assertAlmostEqual(computed_goodput, expected_goodput, delta=0.1)

  def test_goodput_with_anomalous_steps_multiple_disruptions(self):
    """Test function to validate goodput with anomalous step times due to multiple disruptions."""

    # This test simulates _TEST_TOTAL_STEPS * 2 training steps and multiple
    # disruptions during the job's run time as follows:
    # [0, 1, 2, Handled Disruption, 3, 4, 5, 6, 7 Handled Disruption, 8, 9]
    # The handled disruptions will manifest as anomalously large step times.

    job_start_time = datetime.datetime.utcnow()
    self.goodput_recorder.record_job_start_time(job_start_time)

    # Mock some program startup time before the training steps
    step_start_time = job_start_time + _TEST_PROGRAM_STARTUP_TIME

    # First few steps progress with normal step time.
    for step in range(_TEST_TOTAL_STEPS - 3):
      # Record step time
      self.goodput_recorder.record_step_start_time(step, step_start_time)
      step_start_time += _TEST_STEP_TIME

    # Introduce an anomalously large step time due to a disruption.
    self.goodput_recorder.record_step_start_time(
        _TEST_TOTAL_STEPS - 3, step_start_time
    )
    step_start_time += _TEST_ANOMALOUS_STEP_TIME + _TEST_STEP_TIME

    # A few more steps progress with normal step time.
    for step in range(_TEST_TOTAL_STEPS - 2, _TEST_TOTAL_STEPS + 2):
      # Record step time
      self.goodput_recorder.record_step_start_time(step, step_start_time)
      step_start_time += _TEST_STEP_TIME

    # Introduce an anomalously large step time due to a second disruption.
    self.goodput_recorder.record_step_start_time(
        _TEST_TOTAL_STEPS + 2, step_start_time
    )
    step_start_time += _TEST_ANOMALOUS_STEP_TIME + _TEST_STEP_TIME

    # Remaining steps progress with normal step time.
    for step in range(_TEST_TOTAL_STEPS + 3, _TEST_TOTAL_STEPS * 2):
      # Record step time
      self.goodput_recorder.record_step_start_time(step, step_start_time)
      step_start_time += _TEST_STEP_TIME

    job_end_time = (
        job_start_time
        + _TEST_PROGRAM_STARTUP_TIME
        + _TEST_ANOMALOUS_STEP_TIME * 2
        + _TEST_STEP_TIME * _TEST_TOTAL_STEPS * 2
    )
    self.goodput_recorder.record_job_end_time(job_end_time)

    # The time from when the job first started to when the last step start was
    # logged.
    total_time = (
        _TEST_PROGRAM_STARTUP_TIME
        + _TEST_STEP_TIME * _TEST_TOTAL_STEPS * 2
        + _TEST_ANOMALOUS_STEP_TIME * 2
    )

    computed_goodput, _, _ = self.goodput_calculator.get_job_goodput()
    expected_goodput = (
        (2 * _TEST_TOTAL_STEPS * _TEST_STEP_TIME.total_seconds())
        / total_time.total_seconds()
        * 100
    )

    self.assertAlmostEqual(computed_goodput, expected_goodput, delta=0.1)


class BadputTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    self.job_name = 'test-run'
    self.logger_name = 'test-log'
    self.mock_cloud_logger = MockCloudLogger(self.job_name, self.logger_name)
    self.goodput_recorder = goodput.GoodputRecorder(
        self.job_name,
        self.logger_name,
        True,
        self.mock_cloud_logger,
    )
    self.goodput_calculator = goodput.GoodputCalculator(
        self.job_name, self.logger_name, self.mock_cloud_logger
    )

  def test_tpu_init_recorder(self):
    """Test function to validate goodput recorder for TPU init."""
    # Record TPU init
    self.goodput_recorder.record_tpu_init_start_time(_TEST_JOB_START_TIME)
    self.goodput_recorder.record_tpu_init_end_time(
        _TEST_JOB_START_TIME + _TEST_TPU_INIT_TIME
    )

    # Ensure read returns the right number of entries.
    validate_entries = self.mock_cloud_logger.read_cloud_logging_entries()
    self.assertLen(validate_entries, 2)
    # Ensure payload contains the expected information.
    for entry_payload in validate_entries:
      self.assertIn(goodput._JOB_NAME, entry_payload)
      self.assertEqual(entry_payload[goodput._JOB_NAME], self.job_name)
      if goodput._TPU_INIT_START_TIME in entry_payload:
        self.assertEqual(
            entry_payload[goodput._TPU_INIT_START_TIME],
            _TEST_JOB_START_TIME.timestamp(),
        )
      if goodput._TPU_INIT_END_TIME in entry_payload:
        self.assertEqual(
            entry_payload[goodput._TPU_INIT_END_TIME],
            (_TEST_JOB_START_TIME + _TEST_TPU_INIT_TIME).timestamp(),
        )

  def test_training_prep_recorder(self):
    """Test function to validate goodput recorder for training preparation."""
    # Record training preparation time.
    training_prep_start_time = _TEST_JOB_START_TIME + _TEST_TPU_INIT_TIME
    training_prep_end_time = (
        _TEST_JOB_START_TIME
        + _TEST_TPU_INIT_TIME
        + _TEST_TRAINING_PREPARATION_TIME
    )
    self.goodput_recorder.record_training_preparation_start_time(
        training_prep_start_time
    )
    self.goodput_recorder.record_training_preparation_end_time(
        training_prep_end_time
    )

    # Ensure read returns the right number of entries.
    validate_entries = self.mock_cloud_logger.read_cloud_logging_entries()
    self.assertLen(validate_entries, 2)
    # Ensure payload contains the expected information.
    for entry_payload in validate_entries:
      self.assertIn(goodput._JOB_NAME, entry_payload)
      self.assertEqual(entry_payload[goodput._JOB_NAME], self.job_name)
      if goodput._TRAINING_PREPARATION_START_TIME in entry_payload:
        self.assertEqual(
            entry_payload[goodput._TRAINING_PREPARATION_START_TIME],
            training_prep_start_time.timestamp(),
        )
      if goodput._TRAINING_PREPARATION_END_TIME in entry_payload:
        self.assertEqual(
            entry_payload[goodput._TRAINING_PREPARATION_END_TIME],
            training_prep_end_time.timestamp(),
        )

  def test_training_prep_recorder_no_timestamps(self):
    """Test function to validate goodput recorder for training preparation with no timestamps."""
    # Record training preparation time.
    expected_start_time = datetime.datetime.utcnow()
    self.goodput_recorder.record_training_preparation_start_time(None)
    time.sleep(_TEST_TRAINING_PREPARATION_TIME.total_seconds())
    expected_end_time = datetime.datetime.utcnow()
    self.goodput_recorder.record_training_preparation_end_time(None)

    # Ensure read returns the right number of entries.
    validate_entries = self.mock_cloud_logger.read_cloud_logging_entries()
    self.assertLen(validate_entries, 2)
    # Ensure payload contains the expected information.
    for entry_payload in validate_entries:
      self.assertIn(goodput._JOB_NAME, entry_payload)
      self.assertEqual(entry_payload[goodput._JOB_NAME], self.job_name)
      if goodput._TRAINING_PREPARATION_START_TIME in entry_payload:
        self.assertAlmostEqual(
            entry_payload[goodput._TRAINING_PREPARATION_START_TIME],
            expected_start_time.timestamp(),
            delta=0.1,
        )

      if goodput._TRAINING_PREPARATION_END_TIME in entry_payload:
        self.assertAlmostEqual(
            entry_payload[goodput._TRAINING_PREPARATION_END_TIME],
            expected_end_time.timestamp(),
            delta=0.1,
        )

  def test_data_loading_recorder(self):
    """Test function to validate goodput recorder for data loading."""
    # Record data loading time.
    data_loading_start_time = (
        _TEST_JOB_START_TIME
        + _TEST_TPU_INIT_TIME
        + _TEST_TRAINING_PREPARATION_TIME
    )
    data_loading_end_time = (
        _TEST_JOB_START_TIME
        + _TEST_TPU_INIT_TIME
        + _TEST_TRAINING_PREPARATION_TIME
        + _TEST_DATA_LOADING_TIME
    )
    self.goodput_recorder.record_data_loading_start_time(
        data_loading_start_time
    )
    self.goodput_recorder.record_data_loading_end_time(data_loading_end_time)

    # Ensure read returns the right number of entries.
    validate_entries = self.mock_cloud_logger.read_cloud_logging_entries()
    self.assertLen(validate_entries, 2)
    # Ensure payload contains the expected information.
    for entry_payload in validate_entries:
      self.assertIn(goodput._JOB_NAME, entry_payload)
      self.assertEqual(entry_payload[goodput._JOB_NAME], self.job_name)
      if goodput._DATA_LOADING_START_TIME in entry_payload:
        self.assertEqual(
            entry_payload[goodput._DATA_LOADING_START_TIME],
            data_loading_start_time.timestamp(),
        )
      if goodput._DATA_LOADING_END_TIME in entry_payload:
        self.assertEqual(
            entry_payload[goodput._DATA_LOADING_END_TIME],
            data_loading_end_time.timestamp(),
        )

  def test_data_loading_recorder_no_timestamps(self):
    """Test function to validate goodput recorder for data loading."""
    # Record data loading time.
    expected_start_time = datetime.datetime.utcnow()
    self.goodput_recorder.record_data_loading_start_time(None)
    time.sleep(_TEST_DATA_LOADING_TIME.total_seconds())
    expected_end_time = datetime.datetime.utcnow()
    self.goodput_recorder.record_data_loading_end_time(None)

    # Ensure read returns the right number of entries.
    validate_entries = self.mock_cloud_logger.read_cloud_logging_entries()
    self.assertLen(validate_entries, 2)
    # Ensure payload contains the expected information.
    for entry_payload in validate_entries:
      self.assertIn(goodput._JOB_NAME, entry_payload)
      self.assertEqual(entry_payload[goodput._JOB_NAME], self.job_name)
      if goodput._DATA_LOADING_START_TIME in entry_payload:
        self.assertAlmostEqual(
            entry_payload[goodput._DATA_LOADING_START_TIME],
            expected_start_time.timestamp(),
            delta=0.1,
        )
      if goodput._DATA_LOADING_END_TIME in entry_payload:
        self.assertAlmostEqual(
            entry_payload[goodput._DATA_LOADING_END_TIME],
            expected_end_time.timestamp(),
            delta=0.1,
        )

  def test_badput_calculator_tpu_initialization(self):
    """Test function to validate computation of badput due to TPU initialization."""

    job_start_time = datetime.datetime.now(datetime.timezone.utc)
    self.goodput_recorder.record_job_start_time(job_start_time)

    # Mock TPU initialization.
    self.goodput_recorder.record_tpu_init_start_time(job_start_time)
    self.goodput_recorder.record_tpu_init_end_time(
        job_start_time + _TEST_TPU_INIT_TIME
    )

    # Mock _TEST_TOTAL_STEPS steps of training with built-in badput
    # due to program startup.
    step_start_time = (
        job_start_time + _TEST_TPU_INIT_TIME + _TEST_PROGRAM_STARTUP_TIME
    )
    for step in range(_TEST_TOTAL_STEPS):
      # Record step time.
      self.goodput_recorder.record_step_start_time(step, step_start_time)
      step_start_time += _TEST_STEP_TIME

    total_time = (
        _TEST_TPU_INIT_TIME
        + _TEST_PROGRAM_STARTUP_TIME
        + _TEST_STEP_TIME * _TEST_TOTAL_STEPS
    )
    job_end_time = job_start_time + total_time
    self.goodput_recorder.record_job_end_time(job_end_time)

    expected_badput_due_to_tpu_initialization = (
        (_TEST_TPU_INIT_TIME.total_seconds()) / total_time.total_seconds() * 100
    )
    _, computed_badput_breakdown, _ = self.goodput_calculator.get_job_goodput(
        include_badput_breakdown=True
    )
    self.assertNotEmpty(computed_badput_breakdown)
    self.assertIn(BadputType.TPU_INITIALIZATION, computed_badput_breakdown)
    self.assertAlmostEqual(
        computed_badput_breakdown[BadputType.TPU_INITIALIZATION],
        expected_badput_due_to_tpu_initialization,
        delta=0.1,
    )

  def test_badput_calculator_training_preparation(self):
    """Test function to validate computation of badput due to training preparation."""

    job_start_time = datetime.datetime.now(datetime.timezone.utc)
    self.goodput_recorder.record_job_start_time(job_start_time)

    # Mock TPU initialization.
    self.goodput_recorder.record_tpu_init_start_time(job_start_time)
    self.goodput_recorder.record_tpu_init_end_time(
        job_start_time + _TEST_TPU_INIT_TIME
    )
    # Mock training preparation.
    self.goodput_recorder.record_training_preparation_start_time(
        job_start_time + _TEST_TPU_INIT_TIME
    )
    self.goodput_recorder.record_training_preparation_end_time(
        job_start_time + _TEST_TPU_INIT_TIME + _TEST_TRAINING_PREPARATION_TIME
    )

    # Mock training.
    step_start_time = (
        job_start_time + _TEST_TPU_INIT_TIME + _TEST_TRAINING_PREPARATION_TIME
    )
    for step in range(_TEST_TOTAL_STEPS):
      # Record step time.
      self.goodput_recorder.record_step_start_time(step, step_start_time)
      step_start_time += _TEST_STEP_TIME

    total_time = (
        _TEST_TPU_INIT_TIME
        + _TEST_TRAINING_PREPARATION_TIME
        + _TEST_STEP_TIME * _TEST_TOTAL_STEPS
    )
    job_end_time = job_start_time + total_time
    self.goodput_recorder.record_job_end_time(job_end_time)

    # Compute Badput with selection.
    _, computed_badput_breakdown, _ = self.goodput_calculator.get_job_goodput(
        include_badput_breakdown=True
    )
    expected_badput_due_to_training_preparation = (
        (_TEST_TRAINING_PREPARATION_TIME.total_seconds())
        / total_time.total_seconds()
        * 100
    )

    self.assertNotEmpty(computed_badput_breakdown)
    self.assertIn(BadputType.TRAINING_PREP, computed_badput_breakdown)
    self.assertAlmostEqual(
        computed_badput_breakdown[BadputType.TRAINING_PREP],
        expected_badput_due_to_training_preparation,
        delta=0.1,
    )

  def test_badput_calculator_data_loading(self):
    """Test function to validate computation of badput due to data loading."""

    job_start_time = datetime.datetime.now(datetime.timezone.utc)
    self.goodput_recorder.record_job_start_time(job_start_time)

    # Mock TPU initialization.
    self.goodput_recorder.record_tpu_init_start_time(job_start_time)
    self.goodput_recorder.record_tpu_init_end_time(
        job_start_time + _TEST_TPU_INIT_TIME
    )
    # Mock training preparation.
    self.goodput_recorder.record_training_preparation_start_time(
        job_start_time + _TEST_TPU_INIT_TIME
    )
    self.goodput_recorder.record_training_preparation_end_time(
        job_start_time + _TEST_TPU_INIT_TIME + _TEST_TRAINING_PREPARATION_TIME
    )
    # Mock data loading.
    self.goodput_recorder.record_data_loading_start_time(
        job_start_time + _TEST_TPU_INIT_TIME + _TEST_TRAINING_PREPARATION_TIME
    )
    self.goodput_recorder.record_data_loading_end_time(
        job_start_time
        + _TEST_TPU_INIT_TIME
        + _TEST_TRAINING_PREPARATION_TIME
        + _TEST_DATA_LOADING_TIME
    )

    # Mock training.
    step_start_time = (
        job_start_time
        + _TEST_TPU_INIT_TIME
        + _TEST_TRAINING_PREPARATION_TIME
        + _TEST_DATA_LOADING_TIME
    )
    for step in range(_TEST_TOTAL_STEPS):
      # Record step time.
      self.goodput_recorder.record_step_start_time(step, step_start_time)
      step_start_time += _TEST_STEP_TIME

    total_time = (
        _TEST_TPU_INIT_TIME
        + _TEST_TRAINING_PREPARATION_TIME
        + _TEST_DATA_LOADING_TIME
        + _TEST_STEP_TIME * _TEST_TOTAL_STEPS
    )
    job_end_time = job_start_time + total_time
    self.goodput_recorder.record_job_end_time(job_end_time)

    # Compute Badput with selection.
    _, computed_badput_breakdown, _ = self.goodput_calculator.get_job_goodput(
        include_badput_breakdown=True
    )
    expected_badput_due_to_data_loading = (
        (_TEST_DATA_LOADING_TIME.total_seconds())
        / total_time.total_seconds()
        * 100
    )

    self.assertNotEmpty(computed_badput_breakdown)
    self.assertIn(BadputType.DATA_LOADING, computed_badput_breakdown)
    self.assertAlmostEqual(
        computed_badput_breakdown[BadputType.DATA_LOADING],
        expected_badput_due_to_data_loading,
        delta=0.1,
    )

  def test_badput_calculator_program_startup(self):
    """Test function to validate computation of badput due to program startup."""

    job_start_time = datetime.datetime.now(datetime.timezone.utc)
    self.goodput_recorder.record_job_start_time(job_start_time)

    # Mock TPU initialization.
    self.goodput_recorder.record_tpu_init_start_time(job_start_time)
    self.goodput_recorder.record_tpu_init_end_time(
        job_start_time + _TEST_TPU_INIT_TIME
    )
    # Mock training preparation.
    self.goodput_recorder.record_training_preparation_start_time(
        job_start_time + _TEST_TPU_INIT_TIME
    )
    self.goodput_recorder.record_training_preparation_end_time(
        job_start_time + _TEST_TPU_INIT_TIME + _TEST_TRAINING_PREPARATION_TIME
    )
    # Mock data loading.
    self.goodput_recorder.record_data_loading_start_time(
        job_start_time + _TEST_TPU_INIT_TIME + _TEST_TRAINING_PREPARATION_TIME
    )
    self.goodput_recorder.record_data_loading_end_time(
        job_start_time
        + _TEST_TPU_INIT_TIME
        + _TEST_TRAINING_PREPARATION_TIME
        + _TEST_DATA_LOADING_TIME
    )

    # Mock training.
    step_start_time = (
        job_start_time
        + _TEST_TPU_INIT_TIME
        + _TEST_TRAINING_PREPARATION_TIME
        + _TEST_DATA_LOADING_TIME
    )
    # All steps but first progress with average step time.
    for step in range(_TEST_TOTAL_STEPS):
      # Record step time
      self.goodput_recorder.record_step_start_time(step, step_start_time)
      step_start_time += _TEST_STEP_TIME
      # Add startup badput during the first step
      if step == 0:
        step_start_time += _TEST_FIRST_STEP_EXTRA_TIME

    total_time = (
        _TEST_TPU_INIT_TIME
        + _TEST_TRAINING_PREPARATION_TIME
        + _TEST_DATA_LOADING_TIME
        + _TEST_FIRST_STEP_EXTRA_TIME
        + _TEST_STEP_TIME * _TEST_TOTAL_STEPS
    )
    job_end_time = job_start_time + total_time
    self.goodput_recorder.record_job_end_time(job_end_time)

    # Compute Badput.
    _, computed_badput_breakdown, _ = self.goodput_calculator.get_job_goodput(
        include_badput_breakdown=True
    )
    expected_badput_due_to_program_startup = (
        (_TEST_FIRST_STEP_EXTRA_TIME.total_seconds())
        / total_time.total_seconds()
        * 100
    )

    self.assertNotEmpty(computed_badput_breakdown)
    self.assertIn(BadputType.PROGRAM_STARTUP, computed_badput_breakdown)
    self.assertAlmostEqual(
        computed_badput_breakdown[BadputType.PROGRAM_STARTUP],
        expected_badput_due_to_program_startup,
        delta=0.1,
    )

  def test_badput_calculator_program_startup_with_disruptions(self):
    """Validate computation of badput due to program startup after a disruption."""

    job_start_time = datetime.datetime.now(datetime.timezone.utc)
    self.goodput_recorder.record_job_start_time(job_start_time)

    # Mock TPU initialization.
    self.goodput_recorder.record_tpu_init_start_time(job_start_time)
    self.goodput_recorder.record_tpu_init_end_time(
        job_start_time + _TEST_TPU_INIT_TIME
    )
    # Mock training preparation.
    self.goodput_recorder.record_training_preparation_start_time(
        job_start_time + _TEST_TPU_INIT_TIME
    )
    self.goodput_recorder.record_training_preparation_end_time(
        job_start_time + _TEST_TPU_INIT_TIME + _TEST_TRAINING_PREPARATION_TIME
    )
    # Mock data loading.
    self.goodput_recorder.record_data_loading_start_time(
        job_start_time + _TEST_TPU_INIT_TIME + _TEST_TRAINING_PREPARATION_TIME
    )
    self.goodput_recorder.record_data_loading_end_time(
        job_start_time
        + _TEST_TPU_INIT_TIME
        + _TEST_TRAINING_PREPARATION_TIME
        + _TEST_DATA_LOADING_TIME
    )

    # Mock training.
    step_start_time = (
        job_start_time
        + _TEST_TPU_INIT_TIME
        + _TEST_TRAINING_PREPARATION_TIME
        + _TEST_DATA_LOADING_TIME
    )
    # All steps but first progress with average step time.
    for step in range(_TEST_TOTAL_STEPS):
      # Record step time
      self.goodput_recorder.record_step_start_time(step, step_start_time)
      step_start_time += _TEST_STEP_TIME
      # Add startup badput during the first step
      if step == 0:
        step_start_time += _TEST_FIRST_STEP_EXTRA_TIME

    # Simulate a 30-second disruption.
    disruption_time = datetime.timedelta(seconds=30)
    job_restart_time = step_start_time + disruption_time
    self.goodput_recorder.record_job_start_time(job_restart_time)
    step_start_time = (
        job_restart_time
        + _TEST_TPU_INIT_TIME
        + _TEST_TRAINING_PREPARATION_TIME
        + _TEST_DATA_LOADING_TIME
    )

    restart_from_step = 2
    # All steps but first progress with average step time.
    for step in range(restart_from_step, _TEST_TOTAL_STEPS):
      self.goodput_recorder.record_step_start_time(step, step_start_time)
      step_start_time += _TEST_STEP_TIME
      if step == restart_from_step:
        step_start_time += _TEST_FIRST_STEP_EXTRA_TIME

    total_time = (
        _TEST_TPU_INIT_TIME
        + _TEST_TRAINING_PREPARATION_TIME
        + _TEST_DATA_LOADING_TIME
        + _TEST_FIRST_STEP_EXTRA_TIME
        + _TEST_STEP_TIME * _TEST_TOTAL_STEPS
        + disruption_time
        + _TEST_TPU_INIT_TIME
        + _TEST_TRAINING_PREPARATION_TIME
        + _TEST_DATA_LOADING_TIME
        + _TEST_FIRST_STEP_EXTRA_TIME
        + (_TEST_TOTAL_STEPS - restart_from_step) * _TEST_STEP_TIME
    )

    job_end_time = job_start_time + total_time
    self.goodput_recorder.record_job_end_time(job_end_time)

    # Compute Badput.
    _, computed_badput_breakdown, _ = self.goodput_calculator.get_job_goodput(
        include_badput_breakdown=True
    )
    expected_badput_due_to_program_startup = (
        ((_TEST_FIRST_STEP_EXTRA_TIME * 2).total_seconds())
        / total_time.total_seconds()
        * 100
    )

    self.assertNotEmpty(computed_badput_breakdown)
    self.assertIn(BadputType.PROGRAM_STARTUP, computed_badput_breakdown)
    self.assertAlmostEqual(
        computed_badput_breakdown[BadputType.PROGRAM_STARTUP],
        expected_badput_due_to_program_startup,
        delta=0.1,
    )

  def test_badput_calculator_wasted_progress_and_disruptions(self):
    """Validate computation of badput due to wasted progress and disruptions."""

    job_start_time = datetime.datetime.now(datetime.timezone.utc)
    self.goodput_recorder.record_job_start_time(job_start_time)

    # Mock TPU initialization.
    self.goodput_recorder.record_tpu_init_start_time(job_start_time)
    self.goodput_recorder.record_tpu_init_end_time(
        job_start_time + _TEST_TPU_INIT_TIME
    )
    # Mock training preparation.
    self.goodput_recorder.record_training_preparation_start_time(
        job_start_time + _TEST_TPU_INIT_TIME
    )
    self.goodput_recorder.record_training_preparation_end_time(
        job_start_time + _TEST_TPU_INIT_TIME + _TEST_TRAINING_PREPARATION_TIME
    )
    # Mock data loading.
    self.goodput_recorder.record_data_loading_start_time(
        job_start_time + _TEST_TPU_INIT_TIME + _TEST_TRAINING_PREPARATION_TIME
    )
    self.goodput_recorder.record_data_loading_end_time(
        job_start_time
        + _TEST_TPU_INIT_TIME
        + _TEST_TRAINING_PREPARATION_TIME
        + _TEST_DATA_LOADING_TIME
    )

    # Mock training.
    step_start_time = (
        job_start_time
        + _TEST_TPU_INIT_TIME
        + _TEST_TRAINING_PREPARATION_TIME
        + _TEST_DATA_LOADING_TIME
    )
    # All steps but first progress with average step time.
    for step in range(_TEST_TOTAL_STEPS):
      # Record step time
      self.goodput_recorder.record_step_start_time(step, step_start_time)
      step_start_time += _TEST_STEP_TIME
      # Add startup badput during the first step
      if step == 0:
        step_start_time += _TEST_FIRST_STEP_EXTRA_TIME

    # Simulate a 30-second disruption.
    disruption_time = datetime.timedelta(seconds=30)
    job_restart_time = step_start_time + disruption_time
    self.goodput_recorder.record_job_start_time(job_restart_time)
    step_start_time = (
        job_restart_time
        + _TEST_TPU_INIT_TIME
        + _TEST_TRAINING_PREPARATION_TIME
        + _TEST_DATA_LOADING_TIME
    )

    restart_from_step = 2
    # All steps but first progress with average step time.
    for step in range(restart_from_step, _TEST_TOTAL_STEPS):
      self.goodput_recorder.record_step_start_time(step, step_start_time)
      step_start_time += _TEST_STEP_TIME
      if step == restart_from_step:
        step_start_time += _TEST_FIRST_STEP_EXTRA_TIME

    total_time = (
        _TEST_TPU_INIT_TIME
        + _TEST_TRAINING_PREPARATION_TIME
        + _TEST_DATA_LOADING_TIME
        + _TEST_FIRST_STEP_EXTRA_TIME
        + _TEST_STEP_TIME * _TEST_TOTAL_STEPS
        + disruption_time
        + _TEST_TPU_INIT_TIME
        + _TEST_TRAINING_PREPARATION_TIME
        + _TEST_DATA_LOADING_TIME
        + _TEST_FIRST_STEP_EXTRA_TIME
        + (_TEST_TOTAL_STEPS - restart_from_step) * _TEST_STEP_TIME
    )

    job_end_time = job_start_time + total_time
    self.goodput_recorder.record_job_end_time(job_end_time)

    # Compute Badput.
    _, computed_badput_breakdown, _ = self.goodput_calculator.get_job_goodput(
        include_badput_breakdown=True
    )
    wasted_progress_and_disruption_time = (
        disruption_time
        + (_TEST_TOTAL_STEPS - restart_from_step) * _TEST_STEP_TIME
    )
    expected_badput_due_to_disruptions = (
        (wasted_progress_and_disruption_time.total_seconds())
        / total_time.total_seconds()
        * 100
    )

    self.assertNotEmpty(computed_badput_breakdown)
    self.assertIn(
        BadputType.WASTED_PROGRESS_FROM_DISRUPTION,
        computed_badput_breakdown,
    )
    self.assertAlmostEqual(
        computed_badput_breakdown[BadputType.WASTED_PROGRESS_FROM_DISRUPTION],
        expected_badput_due_to_disruptions,
        delta=0.1,
    )

  def test_badput_calculator_unknown_badput(self):
    """Test function to validate unknown badput bucket."""

    job_start_time = datetime.datetime.now(datetime.timezone.utc)
    self.goodput_recorder.record_job_start_time(job_start_time)

    # Mock TPU initialization.
    self.goodput_recorder.record_tpu_init_start_time(job_start_time)
    self.goodput_recorder.record_tpu_init_end_time(
        job_start_time + _TEST_TPU_INIT_TIME
    )

    # Mock _TEST_TOTAL_STEPS steps of training with built-in badput
    # due to program startup.
    step_start_time = (
        job_start_time + _TEST_TPU_INIT_TIME + _TEST_PROGRAM_STARTUP_TIME
    )
    for step in range(_TEST_TOTAL_STEPS):
      # Record step time.
      self.goodput_recorder.record_step_start_time(step, step_start_time)
      step_start_time += _TEST_STEP_TIME

    unknown_badput_time = datetime.timedelta(seconds=5)
    total_time = (
        _TEST_TPU_INIT_TIME
        + _TEST_PROGRAM_STARTUP_TIME
        + _TEST_STEP_TIME * _TEST_TOTAL_STEPS
        + unknown_badput_time
    )
    job_end_time = job_start_time + total_time
    self.goodput_recorder.record_job_end_time(job_end_time)

    _, computed_badput_breakdown, _ = self.goodput_calculator.get_job_goodput(
        include_badput_breakdown=True
    )
    self.assertNotEmpty(computed_badput_breakdown)
    self.assertIn(BadputType.OTHER, computed_badput_breakdown)

    expected_badput_due_to_unknown = (
        (unknown_badput_time.total_seconds()) / total_time.total_seconds() * 100
    )
    self.assertAlmostEqual(
        computed_badput_breakdown[BadputType.OTHER],
        expected_badput_due_to_unknown,
        delta=0.1,
    )

  def test_badput_calculator_checkpoint_badput(self):
    """Validate computation of badput due to checkpoint manager time."""

    job_start_time = datetime.datetime.now(datetime.timezone.utc)
    self.goodput_recorder.record_job_start_time(job_start_time)

    # Mock TPU initialization.
    self.goodput_recorder.record_tpu_init_start_time(job_start_time)
    self.goodput_recorder.record_tpu_init_end_time(
        job_start_time + _TEST_TPU_INIT_TIME
    )
    # Mock training preparation.
    self.goodput_recorder.record_training_preparation_start_time(
        job_start_time + _TEST_TPU_INIT_TIME
    )
    self.goodput_recorder.record_training_preparation_end_time(
        job_start_time + _TEST_TPU_INIT_TIME + _TEST_TRAINING_PREPARATION_TIME
    )
    # Mock data loading.
    self.goodput_recorder.record_data_loading_start_time(
        job_start_time + _TEST_TPU_INIT_TIME + _TEST_TRAINING_PREPARATION_TIME
    )
    self.goodput_recorder.record_data_loading_end_time(
        job_start_time
        + _TEST_TPU_INIT_TIME
        + _TEST_TRAINING_PREPARATION_TIME
        + _TEST_DATA_LOADING_TIME
    )

    # Mock training.
    step_start_time = (
        job_start_time
        + _TEST_TPU_INIT_TIME
        + _TEST_TRAINING_PREPARATION_TIME
        + _TEST_DATA_LOADING_TIME
    )
    # All steps but first progress with average step time.
    for step in range(_TEST_TOTAL_STEPS):
      # Record step time
      self.goodput_recorder.record_step_start_time(step, step_start_time)
      step_start_time += _TEST_STEP_TIME
      # Add startup badput during the first step
      if step == 0:
        step_start_time += _TEST_FIRST_STEP_EXTRA_TIME

    # Mock a save operation.
    save_stats = MockSaveStepStatistics(
        step=1,
        event_type='save',
        directory='gs://bucket/path',
        wait_for_prev_start_time=10.0,
        wait_for_prev_duration_secs=1.0,
        checkpointer_blocking_start_time=12.0,
        checkpointer_blocking_duration_secs=2.0,
        get_old_steps_start_time=13.0,
        get_old_steps_duration_secs=3.0,
        checkpoint_manager_blocking_start_time=10.0,
        checkpoint_manager_blocking_duration_secs=6.0,
        reached_preemption=True,
        preemption_received_at=10.0,
        synchronous=True,
    )
    self.mock_cloud_logger.write_cloud_logging_entry(asdict(save_stats))

    # Simulate a 30-second disruption.
    disruption_time = datetime.timedelta(seconds=30)
    job_restart_time = step_start_time + disruption_time
    self.goodput_recorder.record_job_start_time(job_restart_time)
    step_start_time = (
        job_restart_time
        + _TEST_TPU_INIT_TIME
        + _TEST_TRAINING_PREPARATION_TIME
        + _TEST_DATA_LOADING_TIME
    )

    restart_from_step = 2
    # All steps but first progress with average step time.
    for step in range(restart_from_step, _TEST_TOTAL_STEPS):
      self.goodput_recorder.record_step_start_time(step, step_start_time)
      step_start_time += _TEST_STEP_TIME
      if step == restart_from_step:
        step_start_time += _TEST_FIRST_STEP_EXTRA_TIME

    total_time = (
        _TEST_TPU_INIT_TIME
        + _TEST_TRAINING_PREPARATION_TIME
        + _TEST_DATA_LOADING_TIME
        + _TEST_FIRST_STEP_EXTRA_TIME
        + _TEST_STEP_TIME * _TEST_TOTAL_STEPS
        + disruption_time
        + _TEST_TPU_INIT_TIME
        + _TEST_TRAINING_PREPARATION_TIME
        + _TEST_DATA_LOADING_TIME
        + _TEST_FIRST_STEP_EXTRA_TIME
        + (_TEST_TOTAL_STEPS - restart_from_step) * _TEST_STEP_TIME
    )
    restore_stats = MockRestoreStepStatistics(
        step=1,
        event_type='restore',
        directory='gs://bucket/path',
        checkpointer_start_time=10.0,
        checkpointer_duration_secs=2.0,
        checkpoint_manager_start_time=10.0,
        checkpoint_manager_duration_secs=2.0,
    )
    self.mock_cloud_logger.write_cloud_logging_entry(asdict(restore_stats))

    job_end_time = job_start_time + total_time
    self.goodput_recorder.record_job_end_time(job_end_time)

    # Compute Badput.
    _, computed_badput_breakdown, _ = self.goodput_calculator.get_job_goodput(
        include_badput_breakdown=True
    )
    wasted_progress_and_disruption_time = (
        disruption_time
        + (_TEST_TOTAL_STEPS - restart_from_step) * _TEST_STEP_TIME
    )
    expected_badput_due_to_disruptions = (
        (wasted_progress_and_disruption_time.total_seconds())
        / total_time.total_seconds()
        * 100
    )

    self.assertNotEmpty(computed_badput_breakdown)
    self.assertIn(
        BadputType.WASTED_PROGRESS_FROM_DISRUPTION,
        computed_badput_breakdown,
    )
    self.assertAlmostEqual(
        computed_badput_breakdown[BadputType.WASTED_PROGRESS_FROM_DISRUPTION],
        expected_badput_due_to_disruptions,
        delta=0.1,
    )
    self.assertIn(
        BadputType.UNPRODUCTIVE_CHECKPOINT_SAVE_TIME, computed_badput_breakdown
    )

    self.assertIn(
        BadputType.UNPRODUCTIVE_CHECKPOINT_RESTORE_TIME,
        computed_badput_breakdown,
    )

    expect_badput_due_to_checkpointing_save = (
        (save_stats.checkpoint_manager_blocking_duration_secs)
        / total_time.total_seconds()
        * 100
    )

    expect_badput_due_to_checkpointing_restore = (
        (restore_stats.checkpoint_manager_duration_secs)
        / total_time.total_seconds()
        * 100
    )

    self.assertEqual(
        computed_badput_breakdown[BadputType.UNPRODUCTIVE_CHECKPOINT_SAVE_TIME],
        expect_badput_due_to_checkpointing_save,
    )

    self.assertEqual(
        computed_badput_breakdown[
            BadputType.UNPRODUCTIVE_CHECKPOINT_RESTORE_TIME
        ],
        expect_badput_due_to_checkpointing_restore,
    )

  def test_goodput_badput_with_interval_query(self):
    """Validate computation of goodput and badput with interval query."""

    job_start_time = datetime.datetime.now(datetime.timezone.utc)
    self.goodput_recorder.record_job_start_time(job_start_time)

    # Mock TPU initialization.
    self.goodput_recorder.record_tpu_init_start_time(job_start_time)
    self.goodput_recorder.record_tpu_init_end_time(
        job_start_time + _TEST_TPU_INIT_TIME
    )
    # Mock training preparation.
    self.goodput_recorder.record_training_preparation_start_time(
        job_start_time + _TEST_TPU_INIT_TIME
    )
    self.goodput_recorder.record_training_preparation_end_time(
        job_start_time + _TEST_TPU_INIT_TIME + _TEST_TRAINING_PREPARATION_TIME
    )
    # Mock data loading.
    self.goodput_recorder.record_data_loading_start_time(
        job_start_time + _TEST_TPU_INIT_TIME + _TEST_TRAINING_PREPARATION_TIME
    )
    self.goodput_recorder.record_data_loading_end_time(
        job_start_time
        + _TEST_TPU_INIT_TIME
        + _TEST_TRAINING_PREPARATION_TIME
        + _TEST_DATA_LOADING_TIME
    )

    # Mock training.
    step_start_time = (
        job_start_time
        + _TEST_TPU_INIT_TIME
        + _TEST_TRAINING_PREPARATION_TIME
        + _TEST_DATA_LOADING_TIME
    )
    # All steps but first progress with average step time.
    for step in range(_TEST_TOTAL_STEPS):
      # Record step time
      self.goodput_recorder.record_step_start_time(step, step_start_time)
      step_start_time += _TEST_STEP_TIME
      # Add startup badput during the first step
      if step == 0:
        step_start_time += _TEST_FIRST_STEP_EXTRA_TIME

    intermediate_job_end_time = step_start_time

    # Simulate a 30-second disruption.
    disruption_time = datetime.timedelta(seconds=30)
    job_restart_time = step_start_time + disruption_time
    self.goodput_recorder.record_job_start_time(job_restart_time)
    step_start_time = (
        job_restart_time
        + _TEST_TPU_INIT_TIME
        + _TEST_TRAINING_PREPARATION_TIME
        + _TEST_DATA_LOADING_TIME
    )

    restart_from_step = 2
    # All steps but first progress with average step time.
    for step in range(restart_from_step, _TEST_TOTAL_STEPS):
      self.goodput_recorder.record_step_start_time(step, step_start_time)
      step_start_time += _TEST_STEP_TIME
      if step == restart_from_step:
        step_start_time += _TEST_FIRST_STEP_EXTRA_TIME

    total_time = (
        _TEST_TPU_INIT_TIME
        + _TEST_TRAINING_PREPARATION_TIME
        + _TEST_DATA_LOADING_TIME
        + _TEST_FIRST_STEP_EXTRA_TIME
        + _TEST_STEP_TIME * _TEST_TOTAL_STEPS
        + disruption_time
        + _TEST_TPU_INIT_TIME
        + _TEST_TRAINING_PREPARATION_TIME
        + _TEST_DATA_LOADING_TIME
        + _TEST_FIRST_STEP_EXTRA_TIME
        + (_TEST_TOTAL_STEPS - restart_from_step) * _TEST_STEP_TIME
    )

    job_end_time = job_start_time + total_time
    self.goodput_recorder.record_job_end_time(job_end_time)

    # Compute Goodput and Badput with the interval query API.
    (
        computed_goodput,
        computed_badput_breakdown,
        last_step,
        total_job_time,
        number_of_disruptions,
    ) = self.goodput_calculator.get_job_goodput_interval(
        job_start_time, job_end_time
    )

    productive_time = _TEST_STEP_TIME * _TEST_TOTAL_STEPS
    expected_goodput = (
        (productive_time.total_seconds()) / total_time.total_seconds() * 100
    )
    wasted_progress_and_disruption_time = (
        disruption_time
        + (_TEST_TOTAL_STEPS - restart_from_step) * _TEST_STEP_TIME
    )
    expected_badput_due_to_disruptions = (
        (wasted_progress_and_disruption_time.total_seconds())
        / total_time.total_seconds()
        * 100
    )

    # Validate last step
    self.assertEqual(last_step, _TEST_TOTAL_STEPS - 1)
    # Validate total job time
    self.assertEqual(total_job_time, total_time.total_seconds())
    # Validate number of disruptions
    self.assertEqual(number_of_disruptions, 1)
    # Validate Goodput
    self.assertAlmostEqual(computed_goodput, expected_goodput, delta=0.1)
    # Validate Badput
    self.assertNotEmpty(computed_badput_breakdown)
    self.assertIn(
        BadputType.WASTED_PROGRESS_FROM_DISRUPTION,
        computed_badput_breakdown,
    )
    self.assertAlmostEqual(
        computed_badput_breakdown[BadputType.WASTED_PROGRESS_FROM_DISRUPTION],
        expected_badput_due_to_disruptions,
        delta=0.1,
    )

    # Update the interval to exclude the disruption and validate new values.
    (
        computed_goodput,
        computed_badput_breakdown,
        last_step,
        total_job_time,
        number_of_disruptions,
    ) = self.goodput_calculator.get_job_goodput_interval(
        job_start_time, intermediate_job_end_time
    )

    productive_time = _TEST_STEP_TIME * (_TEST_TOTAL_STEPS - 1)
    expected_intermediate_total_time = (
        _TEST_TPU_INIT_TIME
        + _TEST_TRAINING_PREPARATION_TIME
        + _TEST_DATA_LOADING_TIME
        + _TEST_FIRST_STEP_EXTRA_TIME
        + _TEST_STEP_TIME * (_TEST_TOTAL_STEPS - 1)
    )
    expected_goodput = (
        (productive_time.total_seconds())
        / expected_intermediate_total_time.total_seconds()
        * 100
    )

    # Validate last step
    self.assertEqual(last_step, _TEST_TOTAL_STEPS - 1)
    # Validate total job time
    self.assertEqual(
        total_job_time, expected_intermediate_total_time.total_seconds()
    )
    # There should be no disruptions in the interval.
    self.assertEqual(number_of_disruptions, 0)
    # Validate Goodput
    self.assertAlmostEqual(computed_goodput, expected_goodput, delta=0.1)
    # Validate Badput
    self.assertNotEmpty(computed_badput_breakdown)
    self.assertIn(
        BadputType.WASTED_PROGRESS_FROM_DISRUPTION,
        computed_badput_breakdown,
    )
    self.assertEqual(
        computed_badput_breakdown[BadputType.WASTED_PROGRESS_FROM_DISRUPTION], 0
    )

  def _generate_step_start_times(self, number_of_steps: int, start_time):
    """Generate a list of n non-decreasing datetime objects."""
    max_step_seconds = 600
    step_start_times = [start_time]
    for _ in range(1, number_of_steps):
      increment = random.randint(1, max_step_seconds)
      new_time = step_start_times[-1] + datetime.timedelta(seconds=increment)
      step_start_times.append(new_time)
    return step_start_times

  def test_get_step_deviation(self):
    """Test function to validate step deviation computation."""
    job_start_time = datetime.datetime.utcnow()
    self.goodput_recorder.record_job_start_time(job_start_time)
    # Generate a list of 100 step start times with random step times.
    step_count = 0
    max_steps = 100
    test_step_start_times = self._generate_step_start_times(
        number_of_steps=max_steps, start_time=job_start_time
    )

    # Record step start times.
    for step_start_time in test_step_start_times:
      self.goodput_recorder.record_step_start_time(step_count, step_start_time)
      step_count += 1

    job_end_time = test_step_start_times[-1] + datetime.timedelta(seconds=10)
    self.goodput_recorder.record_job_end_time(job_end_time)

    step_times = self.goodput_calculator._get_step_times()
    ideal_step_time = compute_ideal_step_time(
        step_times=list(step_times.values()), previous_ideal_step_time=None
    )
    computed_step_deviations = self.goodput_calculator.get_step_deviation()
    expected_step_deviations = {
        step_count: abs(step_time - ideal_step_time)
        for step_count, step_time in step_times.items()
    }
    for step_count, expected_deviation in expected_step_deviations.items():
      computed_deviation = computed_step_deviations[step_count]
      self.assertAlmostEqual(
          expected_deviation,
          computed_deviation,
          delta=0.1,
      )


if __name__ == '__main__':
  googletest.main()
