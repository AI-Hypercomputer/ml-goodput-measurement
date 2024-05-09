"""Goodput tests to validate Recorder, Calculator and Logger classes."""

import datetime
import time

from cloud_tpu_goodput.ml_goodput_measurement.src import goodput

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
    self.entries.append(entry)

  def read_cloud_logging_entries(self):
    return self.entries


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
    computed_goodput, _ = self.goodput_calculator.get_job_goodput()
    expected_goodput = (
        (_TEST_STEP_TIME * _TEST_TOTAL_STEPS)
        / (_TEST_JOB_END_TIME - _TEST_JOB_START_TIME)
        * 100
    )
    self.assertEqual(computed_goodput, expected_goodput)

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

    computed_goodput, _ = self.goodput_calculator.get_job_goodput()
    expected_goodput = (
        (
            _TEST_TOTAL_STEPS * _TEST_STEP_TIME.total_seconds()
        )
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
    computed_goodput, _ = self.goodput_calculator.get_job_goodput()
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
    computed_goodput, _ = self.goodput_calculator.get_job_goodput()
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
    computed_goodput, _ = self.goodput_calculator.get_job_goodput()
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

    computed_goodput, _ = self.goodput_calculator.get_job_goodput()
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

    computed_goodput, _ = self.goodput_calculator.get_job_goodput()
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
    computed_badput_breakdown = (
        self.goodput_calculator.get_job_badput_breakdown()
    )
    self.assertNotEmpty(computed_badput_breakdown)
    self.assertIn(
        goodput.BadputType.TPU_INITIALIZATION, computed_badput_breakdown
    )
    self.assertAlmostEqual(
        computed_badput_breakdown[goodput.BadputType.TPU_INITIALIZATION],
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
    computed_badput_breakdown = (
        self.goodput_calculator.get_job_badput_breakdown()
    )
    expected_badput_due_to_training_preparation = (
        (_TEST_TRAINING_PREPARATION_TIME.total_seconds())
        / total_time.total_seconds()
        * 100
    )

    self.assertNotEmpty(computed_badput_breakdown)
    self.assertIn(goodput.BadputType.TRAINING_PREP, computed_badput_breakdown)
    self.assertAlmostEqual(
        computed_badput_breakdown[goodput.BadputType.TRAINING_PREP],
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
    computed_badput_breakdown = (
        self.goodput_calculator.get_job_badput_breakdown()
    )
    expected_badput_due_to_data_loading = (
        (_TEST_DATA_LOADING_TIME.total_seconds())
        / total_time.total_seconds()
        * 100
    )

    self.assertNotEmpty(computed_badput_breakdown)
    self.assertIn(goodput.BadputType.DATA_LOADING, computed_badput_breakdown)
    self.assertAlmostEqual(
        computed_badput_breakdown[goodput.BadputType.DATA_LOADING],
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
    computed_badput_breakdown = (
        self.goodput_calculator.get_job_badput_breakdown()
    )
    expected_badput_due_to_program_startup = (
        (_TEST_FIRST_STEP_EXTRA_TIME.total_seconds())
        / total_time.total_seconds()
        * 100
    )

    self.assertNotEmpty(computed_badput_breakdown)
    self.assertIn(goodput.BadputType.PROGRAM_STARTUP, computed_badput_breakdown)
    self.assertAlmostEqual(
        computed_badput_breakdown[goodput.BadputType.PROGRAM_STARTUP],
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
    computed_badput_breakdown = (
        self.goodput_calculator.get_job_badput_breakdown()
    )
    expected_badput_due_to_program_startup = (
        ((_TEST_FIRST_STEP_EXTRA_TIME * 2).total_seconds())
        / total_time.total_seconds()
        * 100
    )

    self.assertNotEmpty(computed_badput_breakdown)
    self.assertIn(goodput.BadputType.PROGRAM_STARTUP, computed_badput_breakdown)
    self.assertAlmostEqual(
        computed_badput_breakdown[goodput.BadputType.PROGRAM_STARTUP],
        expected_badput_due_to_program_startup,
        delta=0.1,
    )


if __name__ == '__main__':
  googletest.main()
