"""Tests for the _CloudLogger class."""
import datetime
import logging
from unittest import mock

from absl.testing import absltest
from cloud_goodput.ml_goodput_measurement.src import goodput

_CloudLogger = goodput._CloudLogger


class CloudLoggerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.job_name = 'test-job'
    self.log_name = 'test-log'
    self.project_id = 'test-project-id'

  @mock.patch('google.cloud.logging.Client')
  def test_init_captures_project_id(self, mock_client_cls):
    """Test capture of project ID from the logging client."""
    mock_instance = mock_client_cls.return_value
    mock_instance.project = self.project_id

    logger = goodput._CloudLogger(self.job_name, self.log_name)

    self.assertEqual(logger.project_id, self.project_id)
    self.assertEqual(logger.log_name, self.log_name)

  @mock.patch('google.cloud.logging.Client')
  def test_get_filter_msg(self, mock_client_cls):
    """Verifies the filter message construction for fast reads."""
    mock_instance = mock_client_cls.return_value
    mock_instance.project = self.project_id

    logger = goodput._CloudLogger(self.job_name, self.log_name)

    filter_msg = logger._get_filter_msg(start_time=None, end_time=None)

    expected_full_log_name = f'projects/{self.project_id}/logs/{self.log_name}'
    self.assertIn(f'logName="{expected_full_log_name}"', filter_msg)
    self.assertIn(f'jsonPayload.job_name="{self.job_name}"', filter_msg)
    self.assertIn('severity=INFO', filter_msg)

  @mock.patch('google.cloud.logging.Client')
  def test_get_filter_msg_no_project(self, mock_client_cls):
    """Verifies safety: If project ID is missing, skip the optimization."""
    # Setup mock with no project ID.
    mock_instance = mock_client_cls.return_value
    mock_instance.project = None

    logger = goodput._CloudLogger(self.job_name, self.log_name)

    filter_msg = logger._get_filter_msg(start_time=None, end_time=None)

    self.assertNotIn('logName=', filter_msg)
    self.assertIn(f'jsonPayload.job_name="{self.job_name}"', filter_msg)

  @mock.patch('google.cloud.logging.Client')
  def test_read_passes_filter_to_client(self, mock_client_cls):
    """Verifies that the filter is sent to the GCP client for fast reads."""
    mock_client_instance = mock_client_cls.return_value
    mock_client_instance.project = self.project_id
    mock_gcp_logger = mock_client_instance.logger.return_value

    logger = goodput._CloudLogger(self.job_name, self.log_name)

    logger.read_cloud_logging_entries()

    mock_gcp_logger.list_entries.assert_called_once()

    _, kwargs = mock_gcp_logger.list_entries.call_args
    passed_filter = kwargs.get('filter_')
    self.assertIn(
        f'projects/{self.project_id}/logs/{self.log_name}', passed_filter
    )

  @mock.patch('google.cloud.logging.Client')
  def test_write_logs_entry(self, mock_client_cls):
    """Verifies entries are routed via the async CloudLoggingHandler."""
    mock_gcp_logger = mock_client_cls.return_value.logger.return_value

    mock_client_instance = mock_client_cls.return_value
    mock_client_instance.project = self.project_id

    logger = goodput._CloudLogger(self.job_name, self.log_name)

    entry = {'job_name': self.job_name, 'data': 123}
    with mock.patch.object(logger._async_handler, 'emit') as mock_emit:
      logger.write_cloud_logging_entry(entry)

      mock_emit.assert_called_once()
      record = mock_emit.call_args[0][0]
      self.assertEqual(record.msg, entry)
      self.assertEqual(record.levelno, logging.INFO)

    # Synchronous `Logger.log_struct` is no longer called inline; the
    # async transport commits via `client.logging_api.write_entries`.
    mock_gcp_logger.log_struct.assert_not_called()

  @mock.patch('google.cloud.logging.Client')
  def test_default_retention(self, mock_client_cls):
    """Verifies default retention period is applied to the filter."""
    mock_instance = mock_client_cls.return_value
    mock_instance.project = self.project_id

    logger = goodput._CloudLogger(self.job_name, self.log_name)
    end_time = datetime.datetime(
        2026, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc
    )
    expected_start = end_time - goodput._CLOUD_LOGGING_DEFAULT_RETENTION
    filter_msg = logger._get_filter_msg(start_time=None, end_time=end_time)
    self.assertIn(f'timestamp>"{expected_start.isoformat()}"', filter_msg)

  @mock.patch('google.cloud.logging.Client')
  def test_custom_retention(self, mock_client_cls):
    """Verifies custom retention period is applied to the filter."""
    mock_instance = mock_client_cls.return_value
    mock_instance.project = self.project_id

    custom_retention = datetime.timedelta(hours=1)
    logger = goodput._CloudLogger(
        self.job_name, self.log_name, max_logs_retention_period=custom_retention
    )
    end_time = datetime.datetime(
        2026, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc
    )
    expected_start = end_time - custom_retention
    filter_msg = logger._get_filter_msg(start_time=None, end_time=end_time)
    self.assertIn(f'timestamp>"{expected_start.isoformat()}"', filter_msg)

  @mock.patch('google.cloud.logging.Client')
  def test_calculator_passes_retention(self, mock_client_cls):
    """Verifies GoodputCalculator passes the retention down to _CloudLogger."""
    mock_instance = mock_client_cls.return_value
    mock_instance.project = self.project_id
    custom_retention = datetime.timedelta(days=7)
    calculator = goodput.GoodputCalculator(
        self.job_name, self.log_name, max_logs_retention_period=custom_retention
    )
    self.assertEqual(
        calculator._cloud_logger.retention_period, custom_retention
    )


class CloudLoggerAsyncWritesTest(absltest.TestCase):
  """Cloud Logging writes are dispatched via CloudLoggingHandler."""

  def setUp(self):
    super().setUp()
    self.job_name = 'test-job'
    self.log_name = 'test-log'

  @mock.patch('google.cloud.logging.Client')
  def test_handler_attached_with_propagate_false(self, mock_client_cls):
    mock_client_cls.return_value.project = 'p'

    logger = _CloudLogger(self.job_name, self.log_name)

    self.assertIsNotNone(logger._async_handler)
    self.assertIn(logger._async_handler, logger._async_logger.handlers)
    self.assertFalse(
        logger._async_logger.propagate,
        'must not propagate dict payloads to root handlers',
    )

  @mock.patch('google.cloud.logging.Client')
  def test_write_drops_entries_for_other_job(self, mock_client_cls):
    """write_cloud_logging_entry is a no-op for entries with a foreign job."""
    mock_client_cls.return_value.project = 'p'

    logger = _CloudLogger(self.job_name, self.log_name)
    with mock.patch.object(logger._async_handler, 'emit') as mock_emit:
      logger.write_cloud_logging_entry(
          {goodput._JOB_NAME: 'someone-else', 'data': 1}
      )
      logger.write_cloud_logging_entry(None)
      mock_emit.assert_not_called()

  @mock.patch('google.cloud.logging.Client')
  def test_flush_forwards_to_handler(self, mock_client_cls):
    mock_client_cls.return_value.project = 'p'
    logger = _CloudLogger(self.job_name, self.log_name)
    with mock.patch.object(logger._async_handler, 'flush') as mock_flush:
      logger.flush()
      mock_flush.assert_called_once()

  @mock.patch('google.cloud.logging.Client')
  def test_two_instances_same_log_name_do_not_share_handlers(
      self, mock_client_cls
  ):
    """Regression: each instance must own its own stdlib logger.

    `logging.getLogger(name)` returns a singleton per name, so naming the
    backing logger only by `log_name` would cause a second instance's
    `addHandler` to attach onto the first instance's logger — every entry
    would then be uploaded once per accumulated handler.
    """
    mock_client_cls.return_value.project = 'p'

    a = _CloudLogger(self.job_name, self.log_name)
    b = _CloudLogger(self.job_name, self.log_name)

    self.assertIsNot(a._async_logger, b._async_logger)
    self.assertEqual(a._async_logger.handlers, [a._async_handler])
    self.assertEqual(b._async_logger.handlers, [b._async_handler])

    # Writing on `b` must not fan out to `a`'s handler.
    entry = {goodput._JOB_NAME: self.job_name, 'data': 1}
    with mock.patch.object(a._async_handler, 'emit') as emit_a, \
        mock.patch.object(b._async_handler, 'emit') as emit_b:
      b.write_cloud_logging_entry(entry)
      emit_a.assert_not_called()
      emit_b.assert_called_once()


class GoodputRecorderFlushTest(absltest.TestCase):
  """`GoodputRecorder.flush` is the public entry point for explicit drains."""

  @mock.patch('google.cloud.logging.Client')
  def test_flush_forwards_to_cloud_logger(self, mock_client_cls):
    mock_client_cls.return_value.project = 'p'
    recorder = goodput.GoodputRecorder(
        'test-job', 'test-log', logging_enabled=True
    )
    with mock.patch.object(recorder._cloud_logger, 'flush') as mock_flush:
      recorder.flush()
      mock_flush.assert_called_once()

  def test_flush_is_noop_when_logging_disabled(self):
    recorder = goodput.GoodputRecorder(
        'test-job', 'test-log', logging_enabled=False
    )
    self.assertIsNone(recorder._cloud_logger)
    # Should not raise.
    recorder.flush()


if __name__ == '__main__':
  absltest.main()
