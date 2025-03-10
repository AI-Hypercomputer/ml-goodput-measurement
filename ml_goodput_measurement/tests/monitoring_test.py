"""Tests to validate the monitoring module.

This module tests the GoodputMonitor class and its functionality, specifically
the uploading of step deviation, goodput and badput data to Tensorboard.
"""

from unittest import mock

from absl.testing import absltest
from ml_goodput_measurement.src import gcp_metrics
from ml_goodput_measurement.src import goodput_utils
from ml_goodput_measurement.src import monitoring

from google.cloud import monitoring_v3


GoodputMonitor = monitoring.GoodputMonitor
patch = mock.patch
MagicMock = mock.MagicMock
GoodputType = goodput_utils.GoodputType
BadputType = goodput_utils.BadputType
ValueType = gcp_metrics.ValueType
GCPOptions = goodput_utils.GCPOptions

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
            'goodput_time_dict': {
                GoodputType.TOTAL: 10.0,
            },
            'badput_time_dict': {
                BadputType.TPU_INITIALIZATION: 2.0,
                BadputType.DATA_LOADING: 1.0,
            },
        }
    )

    goodput_monitor._send_goodput_metrics_to_gcp()

    # Helper function to create TimeSeries
    def create_timeseries(metric_type, labels, value):
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

    # Helper function to compare calls. Ignore time series as they are
    # dynamically generated.
    def compare_calls_ignore_time_series(expected_call, actual_call):
      if (
          expected_call.args != actual_call.args
          or expected_call.kwargs.keys() != actual_call.kwargs.keys()
      ):
        return False

      for key, expected_value in expected_call.kwargs.items():
        actual_value = actual_call.kwargs[key]
        if key == 'time_series':
          # Ignore the TimeSeries objects
          continue
        else:
          if expected_value != actual_value:
            return False

      return True

    # Create TimeSeries objects
    goodput_ts = create_timeseries(
        'compute.googleapis.com/workload/goodput_time',
        {'goodput_source': 'TOTAL', 'accelerator_type': 'test-acc-type'},
        10.0,
    )
    tpu_init_ts = create_timeseries(
        'compute.googleapis.com/workload/badput_time',
        {
            'badput_source': 'TPU_INITIALIZATION',
            'accelerator_type': 'test-acc-type',
        },
        2.0,
    )
    data_loading_ts = create_timeseries(
        'compute.googleapis.com/workload/badput_time',
        {'badput_source': 'DATA_LOADING', 'accelerator_type': 'test-acc-type'},
        1.0,
    )

    # Verify that create_time_series was called with the correct data
    expected_calls = [
        mock.call.create_time_series(
            name='projects/test-project',
            time_series=[goodput_ts],
        ),
        mock.call.create_time_series(
            name='projects/test-project',
            time_series=[tpu_init_ts],
        ),
        mock.call.create_time_series(
            name='projects/test-project',
            time_series=[data_loading_ts],
        ),
    ]

    actual_calls = mock_client.create_time_series.call_args_list

    # Verify each call individually
    for expected_call in expected_calls:
      found = False
      for actual_call in actual_calls:
        if compare_calls_ignore_time_series(expected_call, actual_call):
          found = True
          break
      if not found:
        self.fail(f"Expected call not found: {expected_call}")

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
            'goodput_time_dict': {
                GoodputType.TOTAL: 10.0,
            },
            'badput_time_dict': {
                BadputType.DATA_LOADING: 2.0,
            },
        }
    )

    goodput_monitor._send_goodput_metrics_to_gcp()

    # Verify that create_time_series was called, even if it raised an exception
    mock_client.create_time_series.assert_called_once()

if __name__ == '__main__':
  absltest.main()
