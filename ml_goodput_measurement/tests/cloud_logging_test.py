"""Tests for the _CloudLogger class."""
import datetime
import logging
import threading
import time
import types
from unittest import mock

from absl.testing import absltest
from cloud_goodput.ml_goodput_measurement.src import goodput

_CloudLogger = goodput._CloudLogger


def _log_entry(payload, timestamp, insert_id):
  return types.SimpleNamespace(
      payload=payload, timestamp=timestamp, insert_id=insert_id
  )


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


class CloudLoggerReadNewEntriesTest(absltest.TestCase):
  """`read_new_entries` re-checks an overlap window and de-duplicates by id."""

  def setUp(self):
    super().setUp()
    self.cursor_ts = datetime.datetime(
        2026, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc
    )
    self.end_time = self.cursor_ts + datetime.timedelta(seconds=60)

  def _make_logger(self, mock_client_cls, entries):
    mock_client_cls.return_value.project = 'test-project-id'
    logger = _CloudLogger('test-job', 'test-log')
    logger.logger.list_entries.return_value = entries
    return logger

  def _passed_filter(self, logger):
    return logger.logger.list_entries.call_args.kwargs['filter_']

  @mock.patch('google.cloud.logging.Client')
  def test_overlap_window_has_no_insert_id_tiebreak(self, mock_client_cls):
    logger = self._make_logger(mock_client_cls, [])

    logger.read_new_entries(
        (self.cursor_ts, 'id-9'), self.end_time, 300.0, set()
    )

    passed_filter = self._passed_filter(logger)
    window_start = self.cursor_ts - datetime.timedelta(seconds=300)
    self.assertIn(f'timestamp>"{window_start.isoformat()}"', passed_filter)
    self.assertIn(f'timestamp<="{self.end_time.isoformat()}"', passed_filter)
    self.assertNotIn('insertId', passed_filter)

  @mock.patch('google.cloud.logging.Client')
  def test_zero_overlap_keeps_insert_id_tiebreak(self, mock_client_cls):
    logger = self._make_logger(mock_client_cls, [])

    logger.read_new_entries((self.cursor_ts, 'id-9'), self.end_time, 0.0, set())

    passed_filter = self._passed_filter(logger)
    self.assertIn(f'timestamp > "{self.cursor_ts.isoformat()}"', passed_filter)
    self.assertIn('insertId > "id-9"', passed_filter)

  @mock.patch('google.cloud.logging.Client')
  def test_drops_known_ids_and_never_moves_cursor_backward(
      self, mock_client_cls
  ):
    late_ts = self.cursor_ts - datetime.timedelta(seconds=10)
    newer_ts = self.cursor_ts + datetime.timedelta(seconds=5)
    entries = [
        _log_entry({'late': 1}, late_ts, 'id-late'),
        _log_entry({'cursor': 2}, self.cursor_ts, 'id-9'),
        _log_entry({'known': 3}, self.cursor_ts, 'id-known'),
        _log_entry({'new': 4}, newer_ts, 'id-new'),
    ]
    logger = self._make_logger(mock_client_cls, entries)

    payloads, cursor, entry_ids = logger.read_new_entries(
        (self.cursor_ts, 'id-9'), self.end_time, 300.0, {'id-known'}
    )

    self.assertEqual(payloads, [{'late': 1}, {'new': 4}])
    self.assertEqual(cursor, (newer_ts, 'id-new'))
    self.assertEqual(entry_ids, [('id-late', late_ts), ('id-new', newer_ts)])

    # A read that only finds a late entry leaves the cursor where it was.
    logger.logger.list_entries.return_value = entries[:1]
    payloads, cursor, _ = logger.read_new_entries(
        (self.cursor_ts, 'id-9'), self.end_time, 300.0, set()
    )
    self.assertEqual(payloads, [{'late': 1}])
    self.assertEqual(cursor, (self.cursor_ts, 'id-9'))

  @mock.patch('google.cloud.logging.Client')
  def test_without_cursor_reads_retention_window(self, mock_client_cls):
    logger = self._make_logger(
        mock_client_cls,
        [_log_entry({'job_start_time': 1.0}, self.cursor_ts, 'id-1')],
    )

    payloads, cursor, entry_ids = logger.read_new_entries(
        None, self.end_time, 0.0, ()
    )

    passed_filter = self._passed_filter(logger)
    expected_start = self.end_time - goodput._CLOUD_LOGGING_DEFAULT_RETENTION
    self.assertIn(f'timestamp>"{expected_start.isoformat()}"', passed_filter)
    self.assertEqual(payloads, [{'job_start_time': 1.0}])
    self.assertEqual(cursor, (self.cursor_ts, 'id-1'))
    self.assertEqual(entry_ids, [('id-1', self.cursor_ts)])
    self.assertEqual(
        logger.job_start_time,
        datetime.datetime.fromtimestamp(1.0, tz=datetime.timezone.utc),
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
  def test_flush_with_timeout_returns_true_when_handler_completes(
      self, mock_client_cls
  ):
    mock_client_cls.return_value.project = 'p'
    logger = _CloudLogger(self.job_name, self.log_name)
    with mock.patch.object(logger._async_handler, 'flush') as mock_flush:
      self.assertTrue(logger.flush(timeout_s=5))
      mock_flush.assert_called_once()

  @mock.patch('google.cloud.logging.Client')
  def test_flush_with_timeout_gives_up_when_handler_blocks(
      self, mock_client_cls
  ):
    mock_client_cls.return_value.project = 'p'
    logger = _CloudLogger(self.job_name, self.log_name)
    release = threading.Event()
    with mock.patch.object(
        logger._async_handler, 'flush', side_effect=release.wait
    ):
      with self.assertLogs(goodput.logger, level='WARNING') as logs:
        start_time = time.monotonic()
        self.assertFalse(logger.flush(timeout_s=0.05))
        elapsed = time.monotonic() - start_time
    release.set()
    self.assertLess(elapsed, 1)
    self.assertLen(logs.output, 1)
    self.assertIn('0.05', logs.output[0])

  @mock.patch('google.cloud.logging.Client')
  def test_flush_without_timeout_blocks_until_handler_completes(
      self, mock_client_cls
  ):
    mock_client_cls.return_value.project = 'p'
    logger = _CloudLogger(self.job_name, self.log_name)
    release = threading.Event()
    results = []
    with mock.patch.object(
        logger._async_handler, 'flush', side_effect=release.wait
    ):
      flush_thread = threading.Thread(
          target=lambda: results.append(logger.flush())
      )
      flush_thread.start()
      flush_thread.join(timeout=0.2)
      self.assertTrue(flush_thread.is_alive())
      release.set()
      flush_thread.join(timeout=5)
    self.assertFalse(flush_thread.is_alive())
    self.assertEqual(results, [True])

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

  @mock.patch('google.cloud.logging.Client')
  def test_flush_forwards_timeout_and_result(self, mock_client_cls):
    mock_client_cls.return_value.project = 'p'
    recorder = goodput.GoodputRecorder(
        'test-job', 'test-log', logging_enabled=True
    )
    with mock.patch.object(
        recorder._cloud_logger, 'flush', return_value=False
    ) as mock_flush:
      self.assertFalse(recorder.flush(timeout_s=2.5))
      mock_flush.assert_called_once_with(timeout_s=2.5)

  def test_flush_is_noop_when_logging_disabled(self):
    recorder = goodput.GoodputRecorder(
        'test-job', 'test-log', logging_enabled=False
    )
    self.assertIsNone(recorder._cloud_logger)
    # Nothing to drain, so it reports success without raising.
    self.assertTrue(recorder.flush())
    self.assertTrue(recorder.flush(timeout_s=0.01))


if __name__ == '__main__':
  absltest.main()
