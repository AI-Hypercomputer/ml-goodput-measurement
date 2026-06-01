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
    self.job_name = 'test-run'
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


if __name__ == '__main__':
  googletest.main()
