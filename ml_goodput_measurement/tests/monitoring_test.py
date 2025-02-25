"""Tests to validate the monitoring module.

This module tests the GoodputMonitor class and its functionality, specifically
the uploading of step deviation, goodput and badput data to Tensorboard.
"""

from unittest import mock

from absl.testing import absltest
from ml_goodput_measurement.src import monitoring

GoodputMonitor = monitoring.GoodputMonitor
patch = mock.patch
MagicMock = mock.MagicMock

_TEST_UPLOAD_INTERVAL = 1


class GoodputMonitorTests(absltest.TestCase):
  """Tests for the GoodputMonitor class."""

  def setUp(self):
    super().setUp()
    self.job_name = 'test-run'
    self.logger_name = 'test-logger'
    self.tensorboard_dir = 'test-dir'

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

    # Thread events should be initialized correctly.
    self.assertIsNotNone(goodput_monitor._step_deviation_termination_event)
    self.assertFalse(goodput_monitor._step_deviation_termination_event.is_set())
    self.assertFalse(goodput_monitor._step_deviation_uploader_thread_running)
    self.assertIsNotNone(goodput_monitor._termination_event)
    self.assertFalse(goodput_monitor._termination_event.is_set())
    self.assertFalse(goodput_monitor._uploader_thread_running)

  @patch(
      'ml_goodput_measurement.src.monitoring.GoodputMonitor._write_goodput_to_tensorboard'
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
    self.assertTrue(goodput_monitor._uploader_thread_running)
    self.assertIsNotNone(goodput_monitor._goodput_upload_thread)
    self.assertFalse(goodput_monitor._termination_event.is_set())
    mock_goodput_to_tensorboard.assert_called_once()
    mock_summary_writer.return_value.add_scalar.assert_called_once()
    goodput_monitor.stop_goodput_uploader()
    self.assertFalse(goodput_monitor._uploader_thread_running)
    self.assertIsNone(goodput_monitor._goodput_upload_thread)
    self.assertTrue(goodput_monitor._termination_event.is_set())

  @patch(
      'ml_goodput_measurement.src.monitoring.GoodputMonitor._write_goodput_to_tensorboard'
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
    self.assertTrue(goodput_monitor._uploader_thread_running)
    self.assertIsNotNone(goodput_monitor._goodput_upload_thread)
    self.assertFalse(goodput_monitor._termination_event.is_set())
    mock_goodput_to_tensorboard.assert_called_once()
    with self.assertRaisesRegex(ValueError, 'Test Error'):
      goodput_monitor._query_and_upload_goodput()
    mock_summary_writer.return_value.add_scalar.assert_not_called()
    goodput_monitor.stop_goodput_uploader()
    self.assertFalse(goodput_monitor._uploader_thread_running)
    self.assertIsNone(goodput_monitor._goodput_upload_thread)
    self.assertTrue(goodput_monitor._termination_event.is_set())

  @patch(
      'ml_goodput_measurement.src.monitoring.GoodputMonitor._write_badput_to_tensorboard'
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
    self.assertTrue(goodput_monitor._uploader_thread_running)
    self.assertIsNotNone(goodput_monitor._goodput_upload_thread)
    self.assertFalse(goodput_monitor._termination_event.is_set())
    self.assertTrue(goodput_monitor._include_badput_breakdown)

    mock_badput_to_tensorboard.assert_called_once()
    mock_summary_writer.return_value.add_scalar.assert_called_once()

    goodput_monitor.stop_goodput_uploader()
    self.assertFalse(goodput_monitor._uploader_thread_running)
    self.assertIsNone(goodput_monitor._goodput_upload_thread)
    self.assertTrue(goodput_monitor._termination_event.is_set())

  @patch(
      'ml_goodput_measurement.src.monitoring.GoodputMonitor._write_step_deviation_to_tensorboard'
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
    self.assertTrue(goodput_monitor._step_deviation_uploader_thread_running)
    self.assertIsNotNone(goodput_monitor._step_deviation_upload_thread)
    self.assertFalse(goodput_monitor._step_deviation_termination_event.is_set())
    mock_step_deviation_to_tensorboard.assert_called_once()
    mock_summary_writer.return_value.add_scalar.assert_called_once()
    goodput_monitor.stop_step_deviation_uploader()
    self.assertFalse(goodput_monitor._step_deviation_uploader_thread_running)
    self.assertIsNone(goodput_monitor._step_deviation_upload_thread)
    self.assertTrue(goodput_monitor._step_deviation_termination_event.is_set())

  @patch(
      'ml_goodput_measurement.src.monitoring.GoodputMonitor._write_step_deviation_to_tensorboard'
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
    self.assertTrue(goodput_monitor._step_deviation_uploader_thread_running)
    self.assertIsNotNone(goodput_monitor._step_deviation_upload_thread)
    self.assertFalse(goodput_monitor._step_deviation_termination_event.is_set())
    mock_query_and_upload_step_deviation.assert_called_once()
    with self.assertRaisesRegex(ValueError, 'Test Error'):
      goodput_monitor._query_and_upload_step_deviation()
    mock_summary_writer.return_value.add_scalar.assert_not_called()
    goodput_monitor.stop_step_deviation_uploader()
    self.assertFalse(goodput_monitor._step_deviation_uploader_thread_running)
    self.assertIsNone(goodput_monitor._step_deviation_upload_thread)
    self.assertTrue(goodput_monitor._step_deviation_termination_event.is_set())

  @patch(
      'ml_goodput_measurement.src.monitoring.GoodputMonitor._send_goodput_metrics_to_gcp'
    )
  @patch(
      'ml_goodput_measurement.src.monitoring.GoodputMonitor._query_and_upload_goodput'
    )
  @patch('ml_goodput_measurement.src.monitoring.check_gcloud_auth')
  @patch('tensorboardX.writer.SummaryWriter')
  @patch('google.cloud.logging.Client')
  async def test_goodput_monitor_gcp_metrics_success(
      self,
      mock_logger_client,
      mock_summary_writer,
      mock_check_gcloud_auth,
      mock_query_and_upload_goodput,
      mock_send_goodput_metrics_to_gcp,
    ):
    mock_logger_client.return_value = MagicMock()
    mock_summary_writer.return_value = MagicMock()
    mock_query_and_upload_goodput.return_value = MagicMock()
    mock_send_goodput_metrics_to_gcp.return_value = MagicMock()
    mock_check_gcloud_auth.return_value = True
    goodput_monitor = GoodputMonitor(
        self.job_name,
        self.logger_name,
        self.tensorboard_dir,
        upload_interval=_TEST_UPLOAD_INTERVAL,
        monitoring_enabled=True,
        send_metrics_to_gcp=True,
        project_id=self.project_id,
        location=self.location,
        replica_id=self.replica_id,
        acc_type=self.acc_type,
    )
    self.goodput_monitor = goodput_monitor
    goodput_monitor.start_goodput_uploader()
    self.assertTrue(goodput_monitor._uploader_thread_running)
    self.assertIsNotNone(goodput_monitor._gcp_metrics_thread)
    self.assertFalse(goodput_monitor._termination_event.is_set())

    goodput_monitor._gcp_metrics_thread.join()

    mock_query_and_upload_goodput.assert_called_once()
    mock_send_goodput_metrics_to_gcp.assert_called_once()

  @patch(
      'ml_goodput_measurement.src.monitoring.GoodputMonitor._send_goodput_metrics_to_gcp'
  )
  @patch(
      'ml_goodput_measurement.src.monitoring.GoodputMonitor._query_and_upload_goodput'
  )
  @patch('ml_goodput_measurement.src.monitoring.check_gcloud_auth')
  @patch('tensorboardX.writer.SummaryWriter')
  @patch('google.cloud.logging.Client')
  async def test_goodput_monitor_gcp_metrics_failure(
      self,
      mock_logger_client,
      mock_summary_writer,
      mock_check_gcloud_auth,
      mock_query_and_upload_goodput,
      mock_send_goodput_metrics_to_gcp,
    ):
    mock_logger_client.return_value = MagicMock()
    mock_summary_writer.return_value = MagicMock()
    mock_query_and_upload_goodput.return_value = MagicMock()
    mock_send_goodput_metrics_to_gcp.side_effect = ValueError('Test Error')
    mock_check_gcloud_auth.return_value = True
    goodput_monitor = GoodputMonitor(
        self.job_name,
        self.logger_name,
        self.tensorboard_dir,
        upload_interval=_TEST_UPLOAD_INTERVAL,
        monitoring_enabled=True,
        send_metrics_to_gcp=True,
        project_id=self.project_id,
        location=self.location,
        replica_id=self.replica_id,
        acc_type=self.acc_type,
    )
    self.goodput_monitor = goodput_monitor
    goodput_monitor.start_goodput_uploader()
    self.assertTrue(goodput_monitor._uploader_thread_running)
    self.assertIsNotNone(goodput_monitor._gcp_metrics_thread)
    self.assertFalse(goodput_monitor._termination_event.is_set())

    with self.assertRaisesRegex(ValueError, 'Test Error'):
      goodput_monitor._gcp_metrics_thread.join()

    mock_send_goodput_metrics_to_gcp.assert_called()
    mock_query_and_upload_goodput.assert_called_once()


if __name__ == '__main__':
  absltest.main()
