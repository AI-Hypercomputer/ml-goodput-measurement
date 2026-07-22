"""Unit tests for ElasticGoodputRecorder."""

import datetime
from typing import Optional

from cloud_goodput.ml_goodput_measurement.src import goodput_elastic
from cloud_goodput.ml_goodput_measurement.src import goodput_utils

from google3.testing.pybase import googletest

get_timestamp_from_log_entry = goodput_utils.get_timestamp_from_log_entry


class MockCloudLogger:

  def __init__(self, job_name, logger_name):
    self.job_name = job_name
    self.logger_name = logger_name
    self.entries = []
    self.last_entry_info = None

  def write_cloud_logging_entry(self, entry):
    timestamp = get_timestamp_from_log_entry(entry)
    if timestamp is not None:
      self.entries.append((timestamp, entry))
      self.last_entry_info = (timestamp, timestamp)

  def read_cloud_logging_entries(
      self, start_time=None, end_time=None, last_entry_info=None
  ):

    def to_aware(dt):
      return (
          dt.replace(tzinfo=datetime.timezone.utc)
          if dt is not None and dt.tzinfo is None
          else dt
      )

    start_time = to_aware(start_time)
    end_time = to_aware(end_time)
    return [
        entry
        for timestamp, entry in self.entries
        if (start_time is None or to_aware(timestamp) > start_time)
        and (end_time is None or to_aware(timestamp) <= end_time)
    ], self.last_entry_info


class ElasticGoodputTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    self.job_name = f'test-run-{self.id().split(".")[-1]}'
    self.logger_name = 'test-log'
    self.mock_cloud_logger = MockCloudLogger(self.job_name, self.logger_name)
    self.goodput_recorder = goodput_elastic.ElasticGoodputRecorder(
        self.job_name,
        self.logger_name,
        True,
        self.mock_cloud_logger,
    )

  def test_record_elastic_slice_counts(self):
    active_slices = 2
    total_slices = 4
    available_slices = 2
    timestamp = datetime.datetime(
        year=2026,
        month=5,
        day=28,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
        tzinfo=datetime.timezone.utc,
    )

    self.goodput_recorder.record_elastic_slice_counts(
        active_slices,
        total_slices,
        available_slices,
        timestamp,
    )

    entries, _ = self.mock_cloud_logger.read_cloud_logging_entries()
    self.assertEqual(len(entries), 1)

    entry = entries[0]
    self.assertEqual(entry[goodput_elastic._JOB_NAME], self.job_name)
    self.assertEqual(entry[goodput_elastic._ACTIVE_SLICES], active_slices)
    self.assertEqual(
        entry[goodput_elastic._TOTAL_SLICES], total_slices
    )
    self.assertEqual(
        entry[goodput_elastic._AVAILABLE_SLICES], available_slices
    )
    self.assertEqual(
        entry[goodput_elastic._ELASTIC_SLICE_COUNTS_TIMESTAMP],
        timestamp.timestamp(),
    )

  def test_record_elastic_slice_counts_no_timestamp(self):
    active_slices = 2
    total_slices = 4
    available_slices = 2

    start_time = datetime.datetime.now(datetime.timezone.utc)
    self.goodput_recorder.record_elastic_slice_counts(
        active_slices,
        total_slices,
        available_slices,
    )
    end_time = datetime.datetime.now(datetime.timezone.utc)

    entries, _ = self.mock_cloud_logger.read_cloud_logging_entries()
    self.assertEqual(len(entries), 1)

    entry = entries[0]
    self.assertEqual(entry[goodput_elastic._JOB_NAME], self.job_name)
    self.assertEqual(entry[goodput_elastic._ACTIVE_SLICES], active_slices)
    self.assertEqual(
        entry[goodput_elastic._TOTAL_SLICES], total_slices
    )
    self.assertEqual(
        entry[goodput_elastic._AVAILABLE_SLICES], available_slices
    )
    entry_timestamp = entry[goodput_elastic._ELASTIC_SLICE_COUNTS_TIMESTAMP]
    self.assertTrue(
        start_time.timestamp() <= entry_timestamp <= end_time.timestamp()
    )

  def test_record_elastic_slice_counts_logging_disabled(self):
    active_slices = 2
    total_slices = 4
    available_slices = 2
    timestamp = datetime.datetime(
        year=2026,
        month=5,
        day=28,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
        tzinfo=datetime.timezone.utc,
    )

    recorder = goodput_elastic.ElasticGoodputRecorder(
        self.job_name,
        self.logger_name,
        False,
        self.mock_cloud_logger,
    )

    recorder.record_elastic_slice_counts(
        active_slices,
        total_slices,
        available_slices,
        timestamp,
    )

    entries, _ = self.mock_cloud_logger.read_cloud_logging_entries()
    self.assertEqual(len(entries), 0)


class ElasticGoodputCalculatorTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    self.job_name = f'test-run-{self.id().split(".")[-1]}'
    self.logger_name = 'test-log'
    self.mock_cloud_logger = MockCloudLogger(self.job_name, self.logger_name)
    self.goodput_recorder = goodput_elastic.ElasticGoodputRecorder(
        self.job_name,
        self.logger_name,
        True,
        self.mock_cloud_logger,
    )
    self.goodput_calculator = goodput_elastic.ElasticGoodputCalculator(
        self.job_name, self.logger_name, self.mock_cloud_logger
    )

  def test_compute_time_weighted_efficiency(self):

    start_time = datetime.datetime(
        2026, 5, 28, 0, 0, 0, tzinfo=datetime.timezone.utc
    )

    self.goodput_recorder.record_elastic_slice_counts(2, 4, 2, start_time)

    change_time = start_time + datetime.timedelta(hours=8)
    self.goodput_recorder.record_elastic_slice_counts(2, 4, 4, change_time)

    end_time = start_time + datetime.timedelta(hours=9)

    entries, _ = self.mock_cloud_logger.read_cloud_logging_entries()
    slice_records = self.goodput_calculator._extract_slice_count_entries(
        entries
    )

    stepping_eff, available_eff = (
        self.goodput_calculator._compute_time_weighted_efficiency(
            slice_records,
            start_time.timestamp(),
            end_time.timestamp(),
        )
    )

    self.assertAlmostEqual(stepping_eff, 0.50)
    self.assertAlmostEqual(
        available_eff, 5.0 / 9.0
    )  # (2/4*8 + 4/4*1)/9 = 5/9 = 0.5555...

  def test_get_current_productive_and_unproductive_time_elastic(self):
    job_start = datetime.datetime(
        2026, 5, 28, 0, 0, 0, tzinfo=datetime.timezone.utc
    )
    self.goodput_recorder.record_job_start_time(job_start)

    t1 = job_start + datetime.timedelta(seconds=10)
    self.goodput_recorder.record_elastic_wait_start_time('slice_down', t1)
    t2 = t1 + datetime.timedelta(seconds=10)
    self.goodput_recorder.record_elastic_wait_end_time('slice_down', t2)

    t3 = t2 + datetime.timedelta(seconds=10)
    self.goodput_recorder.record_elastic_wait_start_time('scale_up', t3)
    t4 = t3 + datetime.timedelta(seconds=15)
    self.goodput_recorder.record_elastic_wait_end_time('scale_up', t4)

    t5 = t4 + datetime.timedelta(seconds=10)
    self.goodput_recorder.record_elastic_reinit_start_time(t5)
    t6 = t5 + datetime.timedelta(seconds=20)
    self.goodput_recorder.record_elastic_reinit_end_time(t6)

    self.goodput_recorder.record_tpu_init_start_time(
        t5 + datetime.timedelta(seconds=2)
    )
    self.goodput_recorder.record_tpu_init_end_time(
        t5 + datetime.timedelta(seconds=7)
    )

    self.goodput_recorder.record_training_preparation_start_time(
        t5 + datetime.timedelta(seconds=10)
    )
    self.goodput_recorder.record_training_preparation_end_time(
        t5 + datetime.timedelta(seconds=15)
    )

    self.goodput_recorder.record_step_start_time(0, t6)
    self.goodput_recorder.record_step_start_time(
        1, t6 + datetime.timedelta(seconds=10)
    )

    self.goodput_calculator._fetch_new_entries(
        t6 + datetime.timedelta(seconds=10)
    )

    prod, unprod, _, _ = (
        self.goodput_calculator._get_current_productive_and_unproductive_time()
    )

    self.assertEqual(
        unprod.get(goodput_utils.BadputType.ELASTIC_SLICE_DOWN), 10.0
    )
    self.assertEqual(
        unprod.get(goodput_utils.BadputType.ELASTIC_SCALE_UP), 15.0
    )
    self.assertEqual(
        unprod.get(goodput_utils.BadputType.ELASTIC_REINITIALIZATION), 20.0
    )

    self.assertEqual(
        unprod.get(goodput_utils.BadputType.TPU_INITIALIZATION, 0.0), 0.0
    )
    self.assertEqual(
        unprod.get(goodput_utils.BadputType.TRAINING_PREP, 0.0), 0.0
    )

  def test_get_job_goodput_details_elastic(self):
    job_start = datetime.datetime(
        2026, 5, 28, 0, 0, 0, tzinfo=datetime.timezone.utc
    )
    self.goodput_recorder.record_job_start_time(job_start)
    self.goodput_recorder.record_elastic_slice_counts(2, 4, 2, job_start)

    t1 = job_start + datetime.timedelta(hours=8)
    self.goodput_recorder.record_elastic_slice_counts(2, 4, 4, t1)

    t2 = t1 + datetime.timedelta(hours=1)
    self.goodput_recorder.record_step_start_time(0, t2)

    t3 = t2 + datetime.timedelta(seconds=10)
    self.goodput_recorder.record_step_start_time(1, t3)

    job_end = t3 + datetime.timedelta(seconds=10)
    self.goodput_recorder.record_job_end_time(job_end)

    # Populate cache by calling get_job_goodput.
    self.goodput_calculator.get_job_goodput(
        include_badput_breakdown=True,
        configured_ideal_step_time=10.0,
    )

    details = self.goodput_calculator.get_job_goodput_details()

    self.assertIn('stepping_slice_efficiency', details)
    self.assertIn('available_slice_efficiency', details)

    self.assertAlmostEqual(details['stepping_slice_efficiency'], 0.5)

    # 8h * 0.5 + 1h20s * 1.0 / (9h20s)
    expected_avail_eff = (28800.0 * 0.5 + 3620.0 * 1.0) / 32420.0
    self.assertAlmostEqual(
        details['available_slice_efficiency'], expected_avail_eff
    )


if __name__ == '__main__':
  googletest.main()
