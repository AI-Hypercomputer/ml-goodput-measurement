"""Goodput tests to validate Recorder, Calculator and Logger classes."""

import datetime

from cloud_tpu_goodput.ml_goodput_measurement.src import goodput

from google3.testing.pybase import googletest

# Fake job timeline information for test purposes.
_TEST_JOB_START_TIME = datetime.datetime(
    year=2024, month=1, day=1, hour=1, minute=0, second=0, microsecond=0
)
_TEST_PROGRAM_STARTUP_TIME = datetime.timedelta(seconds=5)
_TEST_STEP_START_TIME = _TEST_JOB_START_TIME + _TEST_PROGRAM_STARTUP_TIME
_TEST_TOTAL_STEPS = 5
_TEST_STEP_TIME_DELTA = datetime.timedelta(seconds=3)
_TEST_JOB_END_TIME = (
    _TEST_STEP_START_TIME + _TEST_STEP_TIME_DELTA * _TEST_TOTAL_STEPS
)


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
    self.logger_name = 'test_log'
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

    # Mock 5 _TEST_TOTAL_STEPS of training
    step_start_time = _TEST_STEP_START_TIME
    for step in range(_TEST_TOTAL_STEPS):
      # Record step time
      self.goodput_recorder.record_step_start_time(step, step_start_time)
      step_start_time += _TEST_STEP_TIME_DELTA

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
            _TEST_STEP_START_TIME + _TEST_STEP_TIME_DELTA * step_count
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
    computed_goodput = self.goodput_calculator.get_job_goodput()
    expected_goodput = (
        (_TEST_STEP_TIME_DELTA * _TEST_TOTAL_STEPS)
        / (_TEST_JOB_END_TIME - _TEST_JOB_START_TIME)
        * 100
    )
    self.assertEqual(computed_goodput, expected_goodput)


if __name__ == '__main__':
  googletest.main()
