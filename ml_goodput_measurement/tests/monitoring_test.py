"""Tests to validate the monitoring module.

This module tests the GoodputMonitor class and its functionality, specifically
the uploading of step deviation, goodput and badput data to Tensorboard.
"""

import threading
import time
from unittest import mock

from absl.testing import absltest
from cloud_goodput.ml_goodput_measurement.src import gcp_metrics
from cloud_goodput.ml_goodput_measurement.src import goodput_elastic
from cloud_goodput.ml_goodput_measurement.src import goodput_utils
from cloud_goodput.ml_goodput_measurement.src import monitoring
from google.cloud import monitoring_v3

BadputType = goodput_utils.BadputType
GCPOptions = goodput_utils.GCPOptions
GoodputMonitor = monitoring.GoodputMonitor
GoodputType = goodput_utils.GoodputType
IntervalMetricType = goodput_utils.IntervalMetricType
MagicMock = mock.MagicMock
MetricType = goodput_utils.MetricType
ValueType = gcp_metrics.ValueType

patch = mock.patch
_TEST_UPLOAD_INTERVAL = 1


class GoodputMonitorTests(absltest.TestCase):
  """Tests for the GoodputMonitor class."""

  def setUp(self):
    super().setUp()
    self.job_name = 'test-run'
    self.logger_name = 'test-logger'
    self.tensorboard_dir = 'test-dir'

  def _create_timeseries(
      self, metric_type: str, labels: dict, value: float
  ) -> monitoring_v3.TimeSeries:
    ts = monitoring_v3.TimeSeries()
    ts.metric.type = metric_type
    ts.metric.labels.update(labels)
    ts.resource.type = 'compute.googleapis.com/Workload'
    ts.resource.labels.update({
        'location': 'test-location',
        'workload_id': 'test-run',
        'replica_id': 'test-replica-id',
    })
    ts.points.append(
        monitoring_v3.Point(
            value=monitoring_v3.TypedValue(double_value=value),
        )
    )
    return ts

  def _compare_calls_ignore_time_series(
      self, expected_call, actual_call
  ) -> bool:
    if (
        expected_call.args != actual_call.args
        or expected_call.kwargs.keys() != actual_call.kwargs.keys()
    ):
      return False

    for key, expected_value in expected_call.kwargs.items():
      actual_value = actual_call.kwargs[key]
      if key == 'time_series':
        continue
      if expected_value != actual_value:
        return False

    return True

  def _setup_mock_goodput_monitor(
      self, mock_logging_client, mock_summary_writer, mock_metric_service_client
  ) -> GoodputMonitor:
    mock_client = MagicMock()
    mock_metric_service_client.return_value = mock_client
    mock_logging_client.return_value = MagicMock()
    mock_summary_writer.return_value = MagicMock()

    gcp_options = GCPOptions(
        enable_gcp_goodput_metrics=True,
        project_id='test-project',
        location='test-location',
        acc_type='test-acc-type',
        replica_id='test-replica-id',
    )

    return GoodputMonitor(
        job_name='test-run',
        logger_name='test-logger',
        tensorboard_dir='/tmp',
        upload_interval=1,
        monitoring_enabled=True,
        gcp_options=gcp_options,
    )

  @patch('tensorboardX.writer.SummaryWriter')
  @patch('google.cloud.logging.Client')
  def test_goodput_monitor_init(self, mock_logger_client, mock_summary_writer):
    mock_summary_writer.return_value = MagicMock()
    mock_logger_client.return_value = MagicMock()
    goodput_monitor = GoodputMonitor(
        self.job_name,
        self.logger_name,
        self.tensorboard_dir,
        upload_interval=_TEST_UPLOAD_INTERVAL,
        monitoring_enabled=True,
    )
    # Objects should be initialized correctly.
    self.assertIsNotNone(goodput_monitor)
    self.assertIs(goodput_monitor._writer, mock_summary_writer.return_value)
    self.assertIsNotNone(goodput_monitor._goodput_calculator)

    # Process management events should be initialized correctly.
    self.assertIsNotNone(goodput_monitor._step_deviation_termination_event)
    self.assertFalse(goodput_monitor._step_deviation_termination_event.is_set())
    self.assertIsNone(goodput_monitor._step_deviation_process)
    self.assertIsNotNone(goodput_monitor._goodput_termination_event)
    self.assertFalse(goodput_monitor._goodput_termination_event.is_set())
    self.assertIsNone(goodput_monitor._goodput_process)

  @patch(
      'cloud_goodput.ml_goodput_measurement.src.monitoring.GoodputMonitor._final_goodput_query_and_upload'
  )
  @patch('multiprocessing.Event')
  @patch('multiprocessing.Process')
  @patch('tensorboardX.writer.SummaryWriter')
  @patch('google.cloud.logging.Client')
  def test_multiprocess_goodput_monitor_start_and_stop(
      self,
      mock_logger_client,
      mock_summary_writer,
      mock_process,
      mock_event,
      mock_final_goodput_upload,
  ):
    mock_process_instance = mock_process.return_value
    mock_process_instance.is_alive.return_value = True

    mock_goodput_termination_event = MagicMock()
    mock_goodput_final_flush_event = MagicMock()
    mock_step_deviation_termination_event = MagicMock()
    mock_step_deviation_final_flush_event = MagicMock()
    mock_rolling_window_termination_event = MagicMock()
    mock_rolling_window_final_flush_event = MagicMock()
    mock_event.side_effect = [
        mock_goodput_termination_event,
        mock_goodput_final_flush_event,
        mock_step_deviation_termination_event,
        mock_step_deviation_final_flush_event,
        mock_rolling_window_termination_event,
        mock_rolling_window_final_flush_event,
    ]

    mock_summary_writer.return_value = MagicMock()
    mock_logger_client.return_value = MagicMock()

    goodput_monitor = monitoring.GoodputMonitor(
        self.job_name,
        self.logger_name,
        self.tensorboard_dir,
        upload_interval=_TEST_UPLOAD_INTERVAL,
        monitoring_enabled=True,
    )

    goodput_monitor.start_goodput_uploader()
    mock_process.assert_called_once_with(
        target=monitoring._goodput_worker,
        args=(
            goodput_monitor._worker_config,
            mock_goodput_termination_event,
            mock_goodput_final_flush_event,
        ),
        daemon=True,
    )
    mock_process_instance.start.assert_called_once()
    mock_goodput_termination_event.clear.assert_called_once()
    mock_goodput_final_flush_event.clear.assert_called_once()
    goodput_monitor.stop_goodput_uploader()
    # The worker is asked to perform the final flush itself (using its warm
    # cache) before exiting.
    mock_goodput_final_flush_event.set.assert_called_once()
    mock_goodput_termination_event.set.assert_called_once()
    mock_process_instance.join.assert_any_call(timeout=10.0)

    mock_process_instance.terminate.assert_called_once()
    self.assertEqual(mock_process_instance.join.call_count, 2)
    # Since the worker process was running, the cold parent-side fallback
    # query should not run.
    mock_final_goodput_upload.assert_not_called()
    self.assertIsNone(goodput_monitor._goodput_process)

  @patch(
      'cloud_goodput.ml_goodput_measurement.src.monitoring.GoodputMonitor._final_goodput_query_and_upload'
  )
  @patch('multiprocessing.Event')
  @patch('multiprocessing.Process')
  @patch('tensorboardX.writer.SummaryWriter')
  @patch('google.cloud.logging.Client')
  def test_multiprocess_goodput_monitor_start_and_stop_skip_final_flush(
      self,
      mock_logger_client,
      mock_summary_writer,
      mock_process,
      mock_event,
      mock_final_goodput_upload,
  ):
    mock_process_instance = mock_process.return_value
    mock_process_instance.is_alive.return_value = True
    mock_event.side_effect = lambda: MagicMock()

    mock_summary_writer.return_value = MagicMock()
    mock_logger_client.return_value = MagicMock()

    goodput_monitor = monitoring.GoodputMonitor(
        self.job_name,
        self.logger_name,
        self.tensorboard_dir,
        upload_interval=_TEST_UPLOAD_INTERVAL,
        monitoring_enabled=True,
        skip_final_flush=True,
    )

    goodput_monitor.start_goodput_uploader()
    goodput_monitor.stop_goodput_uploader()
    goodput_monitor._goodput_final_flush_event.set.assert_not_called()
    mock_final_goodput_upload.assert_not_called()

  @patch(
      'cloud_goodput.ml_goodput_measurement.src.monitoring.GoodputMonitor._final_goodput_query_and_upload'
  )
  @patch('multiprocessing.Event')
  @patch('multiprocessing.Process')
  @patch('tensorboardX.writer.SummaryWriter')
  @patch('google.cloud.logging.Client')
  def test_multiprocess_goodput_monitor_stop_override_skip_final_flush(
      self,
      mock_logger_client,
      mock_summary_writer,
      mock_process,
      mock_event,
      mock_final_goodput_upload,
  ):
    mock_process_instance = mock_process.return_value
    mock_process_instance.is_alive.return_value = True
    mock_event.side_effect = lambda: MagicMock()

    mock_summary_writer.return_value = MagicMock()
    mock_logger_client.return_value = MagicMock()

    goodput_monitor = monitoring.GoodputMonitor(
        self.job_name,
        self.logger_name,
        self.tensorboard_dir,
        upload_interval=_TEST_UPLOAD_INTERVAL,
        monitoring_enabled=True,
        skip_final_flush=False,
    )

    goodput_monitor.start_goodput_uploader()
    goodput_monitor.stop_goodput_uploader(skip_final_flush=True)
    goodput_monitor._goodput_final_flush_event.set.assert_not_called()
    mock_final_goodput_upload.assert_not_called()

  @patch('tensorboardX.writer.SummaryWriter')
  @patch('google.cloud.logging.Client')
  def test_final_flush_timeout_defaults_to_process_termination_timeout(
      self, mock_logger_client, mock_summary_writer
  ):
    mock_summary_writer.return_value = MagicMock()
    mock_logger_client.return_value = MagicMock()
    goodput_monitor = monitoring.GoodputMonitor(
        self.job_name,
        self.logger_name,
        self.tensorboard_dir,
        upload_interval=_TEST_UPLOAD_INTERVAL,
        monitoring_enabled=True,
    )
    self.assertEqual(
        goodput_monitor._final_flush_timeout_seconds,
        monitoring._PROCESS_TERMINATION_TIMEOUT_SECONDS,
    )

  @patch('tensorboardX.writer.SummaryWriter')
  @patch('google.cloud.logging.Client')
  def test_worker_join_timeout_uses_final_flush_timeout_when_flushing(
      self, mock_logger_client, mock_summary_writer
  ):
    """When a final flush is requested, the join must wait long enough to cover the worker's warm flush, so it should use final_flush_timeout_seconds rather than the fixed process-termination timeout."""
    mock_summary_writer.return_value = MagicMock()
    mock_logger_client.return_value = MagicMock()
    goodput_monitor = monitoring.GoodputMonitor(
        self.job_name,
        self.logger_name,
        self.tensorboard_dir,
        upload_interval=_TEST_UPLOAD_INTERVAL,
        monitoring_enabled=True,
        final_flush_timeout_seconds=45,
    )
    self.assertEqual(
        goodput_monitor._worker_join_timeout(should_skip=False), 45
    )
    # With nothing to flush, there's no extra wait to budget for, so the
    # fixed process-termination timeout applies regardless of
    # final_flush_timeout_seconds.
    self.assertEqual(
        goodput_monitor._worker_join_timeout(should_skip=True),
        monitoring._PROCESS_TERMINATION_TIMEOUT_SECONDS,
    )

  @patch(
      'cloud_goodput.ml_goodput_measurement.src.monitoring.GoodputMonitor._final_goodput_query_and_upload'
  )
  @patch('multiprocessing.Event')
  @patch('multiprocessing.Process')
  @patch('tensorboardX.writer.SummaryWriter')
  @patch('google.cloud.logging.Client')
  def test_stop_goodput_uploader_joins_with_custom_final_flush_timeout(
      self,
      mock_logger_client,
      mock_summary_writer,
      mock_process,
      mock_event,
      mock_final_goodput_upload,
  ):
    mock_process_instance = mock_process.return_value
    mock_process_instance.is_alive.return_value = False
    mock_event.side_effect = lambda: MagicMock()
    mock_summary_writer.return_value = MagicMock()
    mock_logger_client.return_value = MagicMock()

    goodput_monitor = monitoring.GoodputMonitor(
        self.job_name,
        self.logger_name,
        self.tensorboard_dir,
        upload_interval=_TEST_UPLOAD_INTERVAL,
        monitoring_enabled=True,
        final_flush_timeout_seconds=45,
    )

    goodput_monitor.start_goodput_uploader()
    goodput_monitor.stop_goodput_uploader()

    mock_process_instance.join.assert_any_call(timeout=45)

  def test_run_with_timeout_none_blocks_until_done(self):
    func = MagicMock()
    monitoring._run_with_timeout(func, None, 'test op')
    func.assert_called_once()

  def test_run_with_timeout_completes_in_time(self):
    func = MagicMock()
    monitoring._run_with_timeout(func, 5, 'test op')
    func.assert_called_once()

  def test_run_with_timeout_abandons_slow_func(self):
    started = threading.Event()

    def slow_func():
      started.set()
      time.sleep(5)

    start_time = time.monotonic()
    monitoring._run_with_timeout(slow_func, 0.1, 'slow op')
    elapsed = time.monotonic() - start_time

    self.assertTrue(started.wait(timeout=1))
    self.assertLess(elapsed, 1)

  @patch(
      'cloud_goodput.ml_goodput_measurement.src.monitoring._query_and_upload_goodput_once'
  )
  @patch(
      'cloud_goodput.ml_goodput_measurement.src.monitoring._create_gcp_metrics_sender'
  )
  @patch(
      'cloud_goodput.ml_goodput_measurement.src.monitoring._create_tensorboard_writer'
  )
  @patch(
      'cloud_goodput.ml_goodput_measurement.src.monitoring._create_goodput_calculator'
  )
  def test_goodput_worker_performs_warm_final_flush_when_requested(
      self,
      mock_create_calculator,
      mock_create_writer,
      mock_create_metrics_sender,
      mock_query_and_upload_once,
  ):
    mock_create_writer.return_value = MagicMock()
    termination_event = threading.Event()
    termination_event.set()  # Loop body never runs.
    final_flush_event = threading.Event()
    final_flush_event.set()

    monitoring._goodput_worker(
        {'job_name': 'test-run', 'include_badput_breakdown': False, 'upload_interval': 1},
        termination_event,
        final_flush_event,
    )

    mock_query_and_upload_once.assert_called_once()

  @patch(
      'cloud_goodput.ml_goodput_measurement.src.monitoring._query_and_upload_goodput_once'
  )
  @patch(
      'cloud_goodput.ml_goodput_measurement.src.monitoring._create_gcp_metrics_sender'
  )
  @patch(
      'cloud_goodput.ml_goodput_measurement.src.monitoring._create_tensorboard_writer'
  )
  @patch(
      'cloud_goodput.ml_goodput_measurement.src.monitoring._create_goodput_calculator'
  )
  def test_goodput_worker_skips_final_flush_when_not_requested(
      self,
      mock_create_calculator,
      mock_create_writer,
      mock_create_metrics_sender,
      mock_query_and_upload_once,
  ):
    mock_create_writer.return_value = MagicMock()
    termination_event = threading.Event()
    termination_event.set()  # Loop body never runs.
    final_flush_event = threading.Event()  # Not set: skip_final_flush case.

    monitoring._goodput_worker(
        {'job_name': 'test-run', 'include_badput_breakdown': False, 'upload_interval': 1},
        termination_event,
        final_flush_event,
    )

    mock_query_and_upload_once.assert_not_called()

  @patch(
      'cloud_goodput.ml_goodput_measurement.src.monitoring._query_and_upload_goodput_once'
  )
  @patch(
      'cloud_goodput.ml_goodput_measurement.src.monitoring._create_gcp_metrics_sender'
  )
  @patch(
      'cloud_goodput.ml_goodput_measurement.src.monitoring._create_tensorboard_writer'
  )
  @patch(
      'cloud_goodput.ml_goodput_measurement.src.monitoring._create_goodput_calculator'
  )
  def test_goodput_worker_notices_termination_promptly_during_long_interval(
      self,
      mock_create_calculator,
      mock_create_writer,
      mock_create_metrics_sender,
      mock_query_and_upload_once,
  ):
    """A blind time.sleep(upload_interval) would make the worker miss termination_event until the interval elapses, risking forceful termination before the final flush ever runs. termination_event.wait(timeout=...) must wake up immediately instead."""
    mock_create_writer.return_value = MagicMock()
    termination_event = threading.Event()
    final_flush_event = threading.Event()
    final_flush_event.set()

    worker_thread = threading.Thread(
        target=monitoring._goodput_worker,
        args=(
            {
                'job_name': 'test-run',
                'include_badput_breakdown': False,
                # Deliberately much longer than any reasonable termination
                # timeout, to prove the worker doesn't just sleep through it.
                'upload_interval': 100,
            },
            termination_event,
            final_flush_event,
        ),
    )
    worker_thread.start()
    time.sleep(0.05)
    termination_event.set()
    worker_thread.join(timeout=1)

    self.assertFalse(worker_thread.is_alive())
    # Exactly one call: the final flush. No periodic cycle ever ran.
    mock_query_and_upload_once.assert_called_once()


  @patch(
      'cloud_goodput.ml_goodput_measurement.src.monitoring.GoodputMonitor._final_goodput_query_and_upload'
  )
  @patch('tensorboardX.writer.SummaryWriter')
  @patch('google.cloud.logging.Client')
  def test_goodput_monitor_stop_without_start_uses_timeout_bounded_cold_flush(
      self,
      mock_logger_client,
      mock_summary_writer,
      mock_final_goodput_upload,
  ):
    """If the uploader was never started, there's no warm worker cache to use, so stop_goodput_uploader falls back to a one-off query bounded by final_flush_timeout_seconds."""
    mock_summary_writer.return_value = MagicMock()
    mock_logger_client.return_value = MagicMock()
    mock_final_goodput_upload.side_effect = lambda: time.sleep(5)

    goodput_monitor = monitoring.GoodputMonitor(
        self.job_name,
        self.logger_name,
        self.tensorboard_dir,
        upload_interval=_TEST_UPLOAD_INTERVAL,
        monitoring_enabled=True,
        final_flush_timeout_seconds=0.1,
    )

    # start_goodput_uploader() is intentionally not called: no worker process
    # ever ran, so there is no warm cache for it to flush before exiting.
    start_time = time.monotonic()
    goodput_monitor.stop_goodput_uploader()
    elapsed = time.monotonic() - start_time

    mock_final_goodput_upload.assert_called_once()
    self.assertLess(elapsed, 1)

  @patch(
      'cloud_goodput.ml_goodput_measurement.src.monitoring.GoodputMonitor._write_goodput_to_tensorboard'
  )
  @patch('tensorboardX.writer.SummaryWriter')
  @patch('google.cloud.logging.Client')
  async def test_goodput_monitor_start_goodput_uploader_success(
      self, mock_logger_client, mock_summary_writer, mock_goodput_to_tensorboard
  ):
    mock_summary_writer.return_value = MagicMock()
    mock_goodput_to_tensorboard.return_value = MagicMock()
    mock_logger_client.return_value = MagicMock()
    goodput_monitor = monitoring.GoodputMonitor(
        self.job_name,
        self.logger_name,
        self.tensorboard_dir,
        upload_interval=_TEST_UPLOAD_INTERVAL,
        monitoring_enabled=True,
    )
    goodput_monitor.start_goodput_uploader()
    self.assertIsNotNone(goodput_monitor._goodput_process)
    self.assertFalse(goodput_monitor._goodput_termination_event.is_set())
    mock_goodput_to_tensorboard.assert_called_once()
    mock_summary_writer.return_value.add_scalar.assert_called_once()
    goodput_monitor.stop_goodput_uploader()
    self.assertIsNone(goodput_monitor._goodput_process)
    self.assertTrue(goodput_monitor._goodput_termination_event.is_set())

  @patch(
      'cloud_goodput.ml_goodput_measurement.src.monitoring.GoodputMonitor._write_goodput_to_tensorboard'
  )
  @patch('tensorboardX.writer.SummaryWriter')
  @patch('google.cloud.logging.Client')
  async def test_goodput_monitor_start_goodput_uploader_failure(
      self, mock_logger_client, mock_summary_writer, mock_goodput_to_tensorboard
  ):
    mock_logger_client.return_value = MagicMock()
    mock_summary_writer.return_value = MagicMock()
    mock_goodput_to_tensorboard.side_effect = ValueError('Test Error')
    goodput_monitor = monitoring.GoodputMonitor(
        self.job_name,
        self.logger_name,
        self.tensorboard_dir,
        upload_interval=_TEST_UPLOAD_INTERVAL,
        monitoring_enabled=True,
    )
    goodput_monitor.start_goodput_uploader()
    self.assertIsNotNone(goodput_monitor._goodput_process)
    self.assertTrue(goodput_monitor._goodput_process.is_alive())
    self.assertFalse(goodput_monitor._goodput_termination_event.is_set())
    mock_goodput_to_tensorboard.assert_called_once()
    with self.assertRaisesRegex(ValueError, 'Test Error'):
      goodput_monitor._query_and_upload_goodput()
    mock_summary_writer.return_value.add_scalar.assert_not_called()
    goodput_monitor.stop_goodput_uploader()
    self.assertIsNone(goodput_monitor._goodput_process)
    self.assertFalse(goodput_monitor._goodput_process.is_alive())
    self.assertTrue(goodput_monitor._goodput_termination_event.is_set())

  @patch(
      'cloud_goodput.ml_goodput_measurement.src.monitoring.GoodputMonitor._write_badput_to_tensorboard'
  )
  @patch('tensorboardX.writer.SummaryWriter')
  @patch('google.cloud.logging.Client')
  async def test_goodput_monitor_start_badput_uploader_success(
      self, mock_logger_client, mock_summary_writer, mock_badput_to_tensorboard
  ):
    mock_summary_writer.return_value = MagicMock()
    mock_badput_to_tensorboard.return_value = MagicMock()
    mock_logger_client.return_value = MagicMock()
    goodput_monitor = monitoring.GoodputMonitor(
        self.job_name,
        self.logger_name,
        self.tensorboard_dir,
        upload_interval=_TEST_UPLOAD_INTERVAL,
        monitoring_enabled=True,
        include_badput_breakdown=True,
    )

    goodput_monitor.start_goodput_uploader()
    self.assertIsNotNone(goodput_monitor._goodput_process)
    self.assertTrue(goodput_monitor._goodput_process.is_alive())
    self.assertFalse(goodput_monitor._goodput_termination_event.is_set())
    self.assertTrue(goodput_monitor._include_badput_breakdown)

    mock_badput_to_tensorboard.assert_called_once()
    mock_summary_writer.return_value.add_scalar.assert_called_once()

    goodput_monitor.stop_goodput_uploader()
    self.assertFalse(goodput_monitor._goodput_process.is_alive())
    self.assertIsNone(goodput_monitor._goodput_process)
    self.assertTrue(goodput_monitor._goodput_termination_event.is_set())

  @patch(
      'cloud_goodput.ml_goodput_measurement.src.monitoring.GoodputMonitor._write_step_deviation_to_tensorboard'
  )
  @patch('tensorboardX.writer.SummaryWriter')
  @patch('google.cloud.logging.Client')
  async def test_goodput_monitor_start_step_deviation_uploader_success(
      self,
      mock_logger_client,
      mock_summary_writer,
      mock_step_deviation_to_tensorboard,
  ):
    mock_logger_client.return_value = MagicMock()
    mock_summary_writer.return_value = MagicMock()
    mock_step_deviation_to_tensorboard.return_value = MagicMock()
    goodput_monitor = monitoring.GoodputMonitor(
        self.job_name,
        self.logger_name,
        self.tensorboard_dir,
        upload_interval=_TEST_UPLOAD_INTERVAL,
        monitoring_enabled=True,
        include_step_deviation=True,
    )
    goodput_monitor.start_step_deviation_uploader()
    self.assertTrue(goodput_monitor._step_deviation_process.is_alive())
    self.assertIsNotNone(goodput_monitor._step_deviation_process)
    self.assertFalse(goodput_monitor._step_deviation_termination_event.is_set())
    mock_step_deviation_to_tensorboard.assert_called_once()
    mock_summary_writer.return_value.add_scalar.assert_called_once()
    goodput_monitor.stop_step_deviation_uploader()
    self.assertFalse(goodput_monitor._step_deviation_process.is_alive())
    self.assertIsNone(goodput_monitor._step_deviation_process)
    self.assertTrue(goodput_monitor._step_deviation_termination_event.is_set())

  @patch(
      'cloud_goodput.ml_goodput_measurement.src.monitoring.GoodputMonitor._write_step_deviation_to_tensorboard'
  )
  @patch('tensorboardX.writer.SummaryWriter')
  @patch('google.cloud.logging.Client')
  async def test_goodput_monitor_start_step_deviation_uploader_failure(
      self,
      mock_logger_client,
      mock_summary_writer,
      mock_query_and_upload_step_deviation,
  ):
    mock_logger_client.return_value = MagicMock()
    mock_summary_writer.return_value = MagicMock()
    mock_query_and_upload_step_deviation.side_effect = ValueError('Test Error')
    goodput_monitor = monitoring.GoodputMonitor(
        self.job_name,
        self.logger_name,
        self.tensorboard_dir,
        upload_interval=_TEST_UPLOAD_INTERVAL,
        monitoring_enabled=True,
        include_step_deviation=True,
    )
    goodput_monitor.start_step_deviation_uploader()
    self.assertTrue(goodput_monitor._step_deviation_process.is_alive())
    self.assertIsNotNone(goodput_monitor._step_deviation_process)
    self.assertFalse(goodput_monitor._step_deviation_termination_event.is_set())
    mock_query_and_upload_step_deviation.assert_called_once()
    with self.assertRaisesRegex(ValueError, 'Test Error'):
      goodput_monitor._query_and_upload_step_deviation()
    mock_summary_writer.return_value.add_scalar.assert_not_called()
    goodput_monitor.stop_step_deviation_uploader()
    self.assertFalse(goodput_monitor._step_deviation_process.is_alive())
    self.assertIsNone(goodput_monitor._step_deviation_process)
    self.assertTrue(goodput_monitor._step_deviation_termination_event.is_set())

  @patch('google.cloud.monitoring_v3.MetricServiceClient')
  @patch('tensorboardX.writer.SummaryWriter')
  @patch('google.cloud.logging.Client')
  def test_send_goodput_metrics_to_gcp_success(
      self,
      mock_logging_client,
      mock_summary_writer,
      mock_metric_service_client,
  ):
    mock_client = MagicMock()
    mock_metric_service_client.return_value = mock_client
    mock_logging_client.return_value = MagicMock()
    mock_summary_writer.return_value = MagicMock()

    gcp_options = GCPOptions(
        enable_gcp_goodput_metrics=True,
        project_id='test-project',
        location='test-location',
        acc_type='test-acc-type',
        replica_id='test-replica-id',
        cluster_name='test-cluster-name',
    )

    goodput_monitor = GoodputMonitor(
        self.job_name,
        self.logger_name,
        self.tensorboard_dir,
        upload_interval=_TEST_UPLOAD_INTERVAL,
        monitoring_enabled=True,
        gcp_options=gcp_options,
    )

    # Mock the get_job_goodput_details to return test data
    goodput_monitor._goodput_calculator.get_job_goodput_details = MagicMock(
        return_value={
            MetricType.GOODPUT_TIME.value: {
                GoodputType.TOTAL: 10.0,
            },
            MetricType.BADPUT_TIME.value: {
                BadputType.TPU_INITIALIZATION: 2.0,
                BadputType.DATA_LOADING_SYNC: 1.0,
            },
            MetricType.DISRUPTION_COUNT.value: 0,
            MetricType.MAX_PRODUCTIVE_STEP.value: 2,
            MetricType.TOTAL_ELAPSED_TIME.value: 20.0,
            MetricType.STEP_TIME_DEVIATION.value: {
                0: 1.0,
                1: 1.0,
                2: 1.0,
            },
            MetricType.IDEAL_STEP_TIME.value: 1.0,
        }
    )

    details = goodput_monitor._goodput_calculator.get_job_goodput_details()
    monitoring._upload_goodput_metrics_to_gcm(
        goodput_monitor._metrics_sender,
        details,
        goodput_monitor._worker_config,
    )

    expected_calls = [
        mock.call.create_time_series(
            name='projects/test-project',
            time_series=[
                self._create_timeseries(
                    'compute.googleapis.com/workload/goodput_time',
                    {
                        'goodput_source': 'TOTAL',
                        'accelerator_type': 'test-acc-type',
                        'cluster_name': 'test-cluster-name',
                    },
                    10.0,
                )
            ],
        ),
        mock.call.create_time_series(
            name='projects/test-project',
            time_series=[
                self._create_timeseries(
                    'compute.googleapis.com/workload/badput_time',
                    {
                        'badput_source': 'TPU_INITIALIZATION',
                        'accelerator_type': 'test-acc-type',
                        'cluster_name': 'test-cluster-name',
                    },
                    2.0,
                )
            ],
        ),
        mock.call.create_time_series(
            name='projects/test-project',
            time_series=[
                self._create_timeseries(
                    'compute.googleapis.com/workload/badput_time',
                    {
                        'badput_source': 'DATA_LOADING_SYNC',
                        'accelerator_type': 'test-acc-type',
                        'cluster_name': 'test-cluster-name',
                    },
                    1.0,
                )
            ],
        ),
        mock.call.create_time_series(
            name='projects/test-project',
            time_series=[
                self._create_timeseries(
                    'compute.googleapis.com/workload/disruptions',
                    {
                        'accelerator_type': 'test-acc-type',
                        'window_type': 'CUMULATIVE',
                        'cluster_name': 'test-cluster-name',
                    },
                    0,
                )
            ],
        ),
        mock.call.create_time_series(
            name='projects/test-project',
            time_series=[
                self._create_timeseries(
                    'compute.googleapis.com/workload/max_productive_steps',
                    {
                        'accelerator_type': 'test-acc-type',
                        'cluster_name': 'test-cluster-name',
                    },
                    2,
                )
            ],
        ),
        mock.call.create_time_series(
            name='projects/test-project',
            time_series=[
                self._create_timeseries(
                    'compute.googleapis.com/workload/total_elapsed_time',
                    {
                        'accelerator_type': 'test-acc-type',
                        'window_type': 'CUMULATIVE',
                        'cluster_name': 'test-cluster-name',
                    },
                    20.0,
                )
            ],
        ),
        mock.call.create_time_series(
            name='projects/test-project',
            time_series=[
                self._create_timeseries(
                    'compute.googleapis.com/workload/step_time_deviation',
                    {
                        'accelerator_type': 'test-acc-type',
                        'cluster_name': 'test-cluster-name',
                    },
                    1.0,
                )
            ],
        ),
        mock.call.create_time_series(
            name='projects/test-project',
            time_series=[
                self._create_timeseries(
                    'compute.googleapis.com/workload/performance',
                    {
                        'accelerator_type': 'test-acc-type',
                        'cluster_name': 'test-cluster-name',
                    },
                    1.0,
                )
            ],
        ),
    ]

    actual_calls = mock_client.create_time_series.call_args_list

    # Verify each call individually
    for expected_call in expected_calls:
      self.assertTrue(
          any(
              self._compare_calls_ignore_time_series(expected_call, actual)
              for actual in actual_calls
          ),
          f'Expected call not found: {expected_call}',
      )

  @patch('google.cloud.monitoring_v3.MetricServiceClient')
  @patch('tensorboardX.writer.SummaryWriter')
  @patch('google.cloud.logging.Client')
  def test_send_goodput_metrics_to_gcp_exception(
      self,
      mock_logging_client,
      mock_summary_writer,
      mock_metric_service_client,
  ):
    mock_client = MagicMock()
    mock_client.create_time_series.side_effect = Exception('Test Exception')
    mock_metric_service_client.return_value = mock_client
    mock_logging_client.return_value = MagicMock()
    mock_summary_writer.return_value = MagicMock()

    gcp_options = GCPOptions(
        enable_gcp_goodput_metrics=True,
        project_id='test-project',
        location='test-location',
        acc_type='test-acc-type',
        replica_id='test-replica-id',
    )

    goodput_monitor = GoodputMonitor(
        self.job_name,
        self.logger_name,
        self.tensorboard_dir,
        upload_interval=_TEST_UPLOAD_INTERVAL,
        monitoring_enabled=True,
        gcp_options=gcp_options,
    )

    # Mock the get_job_goodput_details to return test data
    goodput_monitor._goodput_calculator.get_job_goodput_details = MagicMock(
        return_value={
            MetricType.GOODPUT_TIME.value: {
                GoodputType.TOTAL: 10.0,
            },
            MetricType.BADPUT_TIME.value: {
                BadputType.DATA_LOADING_SYNC: 2.0,
            },
            MetricType.DISRUPTION_COUNT.value: 0,
            MetricType.MAX_PRODUCTIVE_STEP.value: 2,
            MetricType.TOTAL_ELAPSED_TIME.value: 20.0,
            MetricType.STEP_TIME_DEVIATION.value: {
                0: 1.0,
                1: 1.0,
                2: 1.0,
            },
            MetricType.IDEAL_STEP_TIME.value: 1.0,
        }
    )

    details = goodput_monitor._goodput_calculator.get_job_goodput_details()
    monitoring._upload_goodput_metrics_to_gcm(
        goodput_monitor._metrics_sender,
        details,
        goodput_monitor._worker_config
    )

    # Verify that create_time_series was called, even if it raised an exception
    mock_client.create_time_series.assert_called_once()

  @patch('google.cloud.monitoring_v3.MetricServiceClient')
  @patch('tensorboardX.writer.SummaryWriter')
  @patch('google.cloud.logging.Client')
  def test_send_goodput_metrics_to_gcp_exclusion(
      self,
      mock_logging_client,
      mock_summary_writer,
      mock_metric_service_client
  ):
    mock_client = MagicMock()
    mock_metric_service_client.return_value = mock_client
    mock_logging_client.return_value = MagicMock()
    mock_summary_writer.return_value = MagicMock()

    gcp_options = GCPOptions(
        enable_gcp_goodput_metrics=True,
        project_id='test-project',
        location='test-location',
        acc_type='test-acc-type',
        replica_id='test-replica-id',
    )

    goodput_monitor = GoodputMonitor(
        self.job_name,
        self.logger_name,
        self.tensorboard_dir,
        upload_interval=_TEST_UPLOAD_INTERVAL,
        monitoring_enabled=True,
        gcp_options=gcp_options,
    )

    # Mock the get_job_goodput_details to return test data, including an
    # excluded type
    goodput_monitor._goodput_calculator.get_job_goodput_details = MagicMock(
        return_value={
            MetricType.GOODPUT_TIME.value: {
                GoodputType.TOTAL: 10.0,
            },
            MetricType.BADPUT_TIME.value: {
                BadputType.TPU_INITIALIZATION: 2.0,
                BadputType.DATA_LOADING_SYNC: 1.0,
                BadputType.DATA_LOADING_ASYNC: (
                    3.0
                ),  # DATA_LOADING_ASYNC is in ACTIVITY_EXCLUSION_LIST
            },
            MetricType.DISRUPTION_COUNT.value: 0,
            MetricType.MAX_PRODUCTIVE_STEP.value: 2,
            MetricType.TOTAL_ELAPSED_TIME.value: 20.0,
            MetricType.STEP_TIME_DEVIATION.value: {
                0: 1.0,
                1: 1.0,
                2: 1.0,
            },
            MetricType.IDEAL_STEP_TIME.value: 1.0,
        }
    )

    details = goodput_monitor._goodput_calculator.get_job_goodput_details()
    monitoring._upload_goodput_metrics_to_gcm(
        goodput_monitor._metrics_sender,
        details,
        goodput_monitor._worker_config
    )

    # Verify that create_time_series was called with the correct data,
    # excluding DATA_LOADING_ASYNC
    expected_calls = [
        mock.call.create_time_series(
            name='projects/test-project',
            time_series=[
                self._create_timeseries(
                    'compute.googleapis.com/workload/goodput_time',
                    {
                        'goodput_source': 'TOTAL',
                        'accelerator_type': 'test-acc-type',
                    },
                    10.0,
                )
            ],
        ),
        mock.call.create_time_series(
            name='projects/test-project',
            time_series=[
                self._create_timeseries(
                    'compute.googleapis.com/workload/badput_time',
                    {
                        'badput_source': 'TPU_INITIALIZATION',
                        'accelerator_type': 'test-acc-type',
                    },
                    2.0,
                )
            ],
        ),
        mock.call.create_time_series(
            name='projects/test-project',
            time_series=[
                self._create_timeseries(
                    'compute.googleapis.com/workload/badput_time',
                    {
                        'badput_source': 'DATA_LOADING_SYNC',
                        'accelerator_type': 'test-acc-type',
                    },
                    1.0,
                )
            ],
        ),
        mock.call.create_time_series(
            name='projects/test-project',
            time_series=[
                self._create_timeseries(
                    'compute.googleapis.com/workload/disruptions',
                    {
                        'accelerator_type': 'test-acc-type',
                        'window_type': 'CUMULATIVE',
                    },
                    0,
                )
            ],
        ),
        mock.call.create_time_series(
            name='projects/test-project',
            time_series=[
                self._create_timeseries(
                    'compute.googleapis.com/workload/max_productive_steps',
                    {
                        'accelerator_type': 'test-acc-type',
                    },
                    2,
                )
            ],
        ),
        mock.call.create_time_series(
            name='projects/test-project',
            time_series=[
                self._create_timeseries(
                    'compute.googleapis.com/workload/total_elapsed_time',
                    {
                        'accelerator_type': 'test-acc-type',
                        'window_type': 'CUMULATIVE',
                    },
                    20.0,
                )
            ],
        ),
        mock.call.create_time_series(
            name='projects/test-project',
            time_series=[
                self._create_timeseries(
                    'compute.googleapis.com/workload/step_time_deviation',
                    {
                        'accelerator_type': 'test-acc-type',
                    },
                    1.0,
                )
            ],
        ),
        mock.call.create_time_series(
            name='projects/test-project',
            time_series=[
                self._create_timeseries(
                    'compute.googleapis.com/workload/performance',
                    {
                        'accelerator_type': 'test-acc-type',
                    },
                    1.0,
                )
            ],
        ),
    ]

    actual_calls = mock_client.create_time_series.call_args_list

    # Verify each call individually
    for expected_call in expected_calls:
      self.assertTrue(
          any(
              self._compare_calls_ignore_time_series(expected_call, actual)
              for actual in actual_calls
          ),
          f'Expected call not found: {expected_call}',
      )
    # Verify unexpected calls are not made
    for actual_call in actual_calls:
      for ts in actual_call.kwargs.get('time_series', []):
        if (
            'badput_source' in ts.metric.labels
            and ts.metric.labels['badput_source'] == 'DATA_LOADING_ASYNC'
        ):
          self.fail(f'Unexpected call found: {ts}')

  @patch('google.cloud.monitoring_v3.MetricServiceClient')
  @patch('tensorboardX.writer.SummaryWriter')
  @patch('google.cloud.logging.Client')
  def test_send_interval_goodput_metrics_to_gcp(
      self,
      mock_logging_client,
      mock_summary_writer,
      mock_metric_service_client,
  ):
    mock_client = MagicMock()
    mock_metric_service_client.return_value = mock_client
    mock_logging_client.return_value = MagicMock()
    mock_summary_writer.return_value = MagicMock()

    gcp_options = GCPOptions(
        enable_gcp_goodput_metrics=True,
        project_id='test-project',
        location='test-location',
        acc_type='test-acc-type',
        replica_id='test-replica-id',
    )

    goodput_monitor = GoodputMonitor(
        self.job_name,
        self.logger_name,
        self.tensorboard_dir,
        upload_interval=_TEST_UPLOAD_INTERVAL,
        monitoring_enabled=True,
        gcp_options=gcp_options,
    )

    # Mock the get_job_goodput_details to return test data
    goodput_monitor._goodput_calculator.get_interval_metric_details = MagicMock(
        return_value={
            IntervalMetricType.INTERVAL_GOODPUT.value: {
                GoodputType.TOTAL: 90.0,
            },
            IntervalMetricType.INTERVAL_BADPUT.value: {
                BadputType.TPU_INITIALIZATION: 2.0,
                BadputType.DATA_LOADING_SYNC: 8.0,
            },
            IntervalMetricType.INTERVAL_SIZE.value: 100,
        }
    )

    details = goodput_monitor._goodput_calculator.get_interval_metric_details()
    monitoring._upload_interval_goodput_metrics_to_gcm(
        goodput_monitor._metrics_sender,
        details,
        goodput_monitor._worker_config
    )

    expected_calls = [
        mock.call.create_time_series(
            name='projects/test-project',
            time_series=[
                self._create_timeseries(
                    'compute.googleapis.com/workload/interval_goodput',
                    {
                        'goodput_source': 'TOTAL',
                        'accelerator_type': 'test-acc-type',
                        'rolling_window_size': '100',
                    },
                    90.0,
                )
            ],
        ),
        mock.call.create_time_series(
            name='projects/test-project',
            time_series=[
                self._create_timeseries(
                    'compute.googleapis.com/workload/interval_badput',
                    {
                        'badput_source': 'TPU_INITIALIZATION',
                        'accelerator_type': 'test-acc-type',
                        'rolling_window_size': '100',
                    },
                    2.0,
                )
            ],
        ),
        mock.call.create_time_series(
            name='projects/test-project',
            time_series=[
                self._create_timeseries(
                    'compute.googleapis.com/workload/interval_badput',
                    {
                        'badput_source': 'DATA_LOADING_SYNC',
                        'accelerator_type': 'test-acc-type',
                        'rolling_window_size': '100',
                    },
                    8.0,
                )
            ],
        ),
    ]

    actual_calls = mock_client.create_time_series.call_args_list

    # Verify each call individually
    for expected_call in expected_calls:
      self.assertTrue(
          any(
              self._compare_calls_ignore_time_series(expected_call, actual)
              for actual in actual_calls
          ),
          f'Expected call not found: {expected_call}',
      )

  @patch('google.cloud.monitoring_v3.MetricServiceClient')
  @patch('tensorboardX.writer.SummaryWriter')
  @patch('google.cloud.logging.Client')
  def test_send_goodput_metrics_custom_sync_events(
      self, mock_logging_client, mock_summary_writer, mock_metric_service_client
  ):
    mock_client = MagicMock()
    mock_metric_service_client.return_value = mock_client
    mock_logging_client.return_value = MagicMock()
    mock_summary_writer.return_value = MagicMock()

    gcp_options = GCPOptions(
        enable_gcp_goodput_metrics=True,
        project_id='test-project',
        location='test-location',
        acc_type='test-acc-type',
        replica_id='test-replica-id',
    )

    goodput_monitor = GoodputMonitor(
        self.job_name,
        self.logger_name,
        self.tensorboard_dir,
        upload_interval=_TEST_UPLOAD_INTERVAL,
        monitoring_enabled=True,
        gcp_options=gcp_options,
    )

    # Mock the get_job_goodput_details to return test data, including an
    # excluded type
    goodput_monitor._goodput_calculator.get_job_goodput_details = MagicMock(
        return_value={
            MetricType.GOODPUT_TIME.value: {
                GoodputType.TOTAL: 10.0,
            },
            MetricType.BADPUT_TIME.value: {
                BadputType.TPU_INITIALIZATION: 2.0,
                BadputType.DATA_LOADING_SYNC: 1.0,
                BadputType.CUSTOM_BADPUT_EVENTS: {
                    'EVAL_STEP': 3.0,
                    'SDC_COMPILATION': 4.0,
                },
            },
            MetricType.DISRUPTION_COUNT.value: 0,
            MetricType.MAX_PRODUCTIVE_STEP.value: 2,
            MetricType.TOTAL_ELAPSED_TIME.value: 20.0,
            MetricType.STEP_TIME_DEVIATION.value: {
                0: 1.0,
                1: 1.0,
                2: 1.0,
            },
            MetricType.IDEAL_STEP_TIME.value: 1.0,
        }
    )

    details = goodput_monitor._goodput_calculator.get_job_goodput_details()
    monitoring._upload_goodput_metrics_to_gcm(
        goodput_monitor._metrics_sender,
        details,
        goodput_monitor._worker_config
    )

    expected_calls = [
        mock.call.create_time_series(
            name='projects/test-project',
            time_series=[
                self._create_timeseries(
                    'compute.googleapis.com/workload/goodput_time',
                    {
                        'goodput_source': 'TOTAL',
                        'accelerator_type': 'test-acc-type',
                    },
                    10.0,
                )
            ],
        ),
        mock.call.create_time_series(
            name='projects/test-project',
            time_series=[
                self._create_timeseries(
                    'compute.googleapis.com/workload/badput_time',
                    {
                        'badput_source': 'TPU_INITIALIZATION',
                        'accelerator_type': 'test-acc-type',
                    },
                    2.0,
                )
            ],
        ),
        mock.call.create_time_series(
            name='projects/test-project',
            time_series=[
                self._create_timeseries(
                    'compute.googleapis.com/workload/badput_time',
                    {
                        'badput_source': 'DATA_LOADING_SYNC',
                        'accelerator_type': 'test-acc-type',
                    },
                    1.0,
                )
            ],
        ),
        mock.call.create_time_series(
            name='projects/test-project',
            time_series=[
                self._create_timeseries(
                    'compute.googleapis.com/workload/disruptions',
                    {
                        'accelerator_type': 'test-acc-type',
                        'window_type': 'CUMULATIVE',
                    },
                    0,
                )
            ],
        ),
        mock.call.create_time_series(
            name='projects/test-project',
            time_series=[
                self._create_timeseries(
                    'compute.googleapis.com/workload/max_productive_steps',
                    {
                        'accelerator_type': 'test-acc-type',
                    },
                    2,
                )
            ],
        ),
        mock.call.create_time_series(
            name='projects/test-project',
            time_series=[
                self._create_timeseries(
                    'compute.googleapis.com/workload/total_elapsed_time',
                    {
                        'accelerator_type': 'test-acc-type',
                        'window_type': 'CUMULATIVE',
                    },
                    20.0,
                )
            ],
        ),
        mock.call.create_time_series(
            name='projects/test-project',
            time_series=[
                self._create_timeseries(
                    'compute.googleapis.com/workload/step_time_deviation',
                    {
                        'accelerator_type': 'test-acc-type',
                    },
                    1.0,
                )
            ],
        ),
        mock.call.create_time_series(
            name='projects/test-project',
            time_series=[
                self._create_timeseries(
                    'compute.googleapis.com/workload/performance',
                    {
                        'accelerator_type': 'test-acc-type',
                    },
                    1.0,
                )
            ],
        ),
    ]

    actual_calls = mock_client.create_time_series.call_args_list

    # Verify each call individually
    for expected_call in expected_calls:
      self.assertTrue(
          any(
              self._compare_calls_ignore_time_series(expected_call, actual_call)
              for actual_call in actual_calls
          ),
          f'Expected call not found: {expected_call}',
      )

  @patch(
      'cloud_goodput.ml_goodput_measurement.src.monitoring.GoodputMonitor._final_interval_goodput_query_and_upload'
  )
  @patch(
      'cloud_goodput.ml_goodput_measurement.src.monitoring.GoodputMonitor._final_step_deviation_query_and_upload'
  )
  @patch(
      'cloud_goodput.ml_goodput_measurement.src.monitoring.GoodputMonitor._final_goodput_query_and_upload'
  )
  async def test_goodput_monitor_final_query_and_upload(
      self,
      mock_final_goodput_query_and_upload,
      mock_final_step_deviation_query_and_upload,
      mock_final_interval_goodput_query_and_upload,
  ):
    mock_final_goodput_query_and_upload.return_value = MagicMock()
    mock_final_step_deviation_query_and_upload.return_value = MagicMock()
    mock_final_interval_goodput_query_and_upload.return_value = MagicMock()
    goodput_monitor = monitoring.GoodputMonitor(
        self.job_name,
        self.logger_name,
        self.tensorboard_dir,
        upload_interval=_TEST_UPLOAD_INTERVAL,
        monitoring_enabled=True,
    )
    goodput_monitor.__del__()
    mock_final_goodput_query_and_upload.assert_called_once()
    mock_final_step_deviation_query_and_upload.assert_called_once()
    mock_final_interval_goodput_query_and_upload.assert_called_once()

  @patch('google.cloud.monitoring_v3.MetricServiceClient')
  @patch('tensorboardX.writer.SummaryWriter')
  @patch('google.cloud.logging.Client')
  def test_upload_goodput_metrics_includes_cluster_name(
      self,
      mock_logging_client,
      mock_summary_writer,
      mock_metric_service_client,
  ):
    """Verifies that cluster_name label is attached when configured."""
    mock_client = MagicMock()
    mock_metric_service_client.return_value = mock_client
    mock_logging_client.return_value = MagicMock()
    mock_summary_writer.return_value = MagicMock()

    gcp_options = GCPOptions(
        enable_gcp_goodput_metrics=True,
        project_id='test-project',
        location='test-location',
        acc_type='test-acc-type',
        replica_id='test-replica-id',
        cluster_name='test-cluster',
    )

    goodput_monitor = GoodputMonitor(
        self.job_name,
        self.logger_name,
        self.tensorboard_dir,
        upload_interval=_TEST_UPLOAD_INTERVAL,
        monitoring_enabled=True,
        gcp_options=gcp_options,
    )
    goodput_monitor._goodput_calculator.get_job_goodput_details = MagicMock(
        return_value={
            MetricType.GOODPUT_TIME.value: {
                GoodputType.TOTAL: 90.0,
            },
            MetricType.BADPUT_TIME.value: {},
            MetricType.DISRUPTION_COUNT.value: 0,
            MetricType.MAX_PRODUCTIVE_STEP.value: 0,
            MetricType.TOTAL_ELAPSED_TIME.value: 0.0,
            MetricType.STEP_TIME_DEVIATION.value: {},
            MetricType.IDEAL_STEP_TIME.value: 0.0,
        }
    )

    details = goodput_monitor._goodput_calculator.get_job_goodput_details()
    monitoring._upload_goodput_metrics_to_gcm(
        goodput_monitor._metrics_sender,
        details,
        goodput_monitor._worker_config,
    )

    mock_client.create_time_series.assert_called_once()
    call_kwargs = mock_client.create_time_series.call_args.kwargs
    actual_time_series_list = call_kwargs['time_series']

    goodput_ts = next(
        (
            ts
            for ts in actual_time_series_list
            if ts.metric.type == 'compute.googleapis.com/workload/goodput_time'
        ),
        None,
    )
    self.assertIsNotNone(
        goodput_ts, 'Goodput time metric not found in upload call'
    )
    self.assertIn('cluster_name', goodput_ts.metric.labels)
    self.assertEqual(goodput_ts.metric.labels['cluster_name'], 'test-cluster')


if __name__ == '__main__':
  absltest.main()
