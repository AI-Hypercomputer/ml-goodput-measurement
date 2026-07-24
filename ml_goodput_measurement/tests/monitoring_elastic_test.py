"""Tests for ElasticGoodputMonitor."""

from unittest import mock

from absl.testing import absltest
from cloud_goodput.ml_goodput_measurement.src import gcp_metrics
from cloud_goodput.ml_goodput_measurement.src import goodput_elastic
from cloud_goodput.ml_goodput_measurement.src import goodput_utils
from cloud_goodput.ml_goodput_measurement.src import monitoring
from cloud_goodput.ml_goodput_measurement.src import monitoring_elastic
from google.cloud import monitoring_v3

BadputType = goodput_utils.BadputType
GCPOptions = goodput_utils.GCPOptions
GoodputType = goodput_utils.GoodputType
IntervalMetricType = goodput_utils.IntervalMetricType
MagicMock = mock.MagicMock
MetricType = goodput_utils.MetricType
ValueType = gcp_metrics.ValueType

patch = mock.patch
_TEST_UPLOAD_INTERVAL = 1


class ElasticGoodputMonitorTests(absltest.TestCase):
  """Tests for the ElasticGoodputMonitor class."""

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

  def _compare_timeseries_ignore_time(self, expected_ts, actual_ts) -> bool:
    if expected_ts.metric.type != actual_ts.metric.type:
      return False
    if dict(expected_ts.metric.labels) != dict(actual_ts.metric.labels):
      return False
    if expected_ts.resource.type != actual_ts.resource.type:
      return False
    if dict(expected_ts.resource.labels) != dict(actual_ts.resource.labels):
      return False
    if len(expected_ts.points) != len(actual_ts.points):
      return False
    for p1, p2 in zip(expected_ts.points, actual_ts.points):
      if p1.value != p2.value:
        return False
    return True

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
        for exp_ts in expected_value:
          if not any(
              self._compare_timeseries_ignore_time(exp_ts, act_ts)
              for act_ts in actual_value
          ):
            return False
      elif expected_value != actual_value:
        return False

    return True

  def _setup_mock_elastic_goodput_monitor(
      self, mock_logging_client, mock_summary_writer, mock_metric_service_client
  ) -> monitoring_elastic.ElasticGoodputMonitor:
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

    return monitoring_elastic.ElasticGoodputMonitor(
        job_name='test-run',
        logger_name='test-logger',
        tensorboard_dir='/tmp',
        upload_interval=1,
        monitoring_enabled=True,
        gcp_options=gcp_options,
        include_slice_efficiency=True,
    )

  @patch('tensorboardX.writer.SummaryWriter')
  @patch('google.cloud.logging.Client')
  def test_elastic_goodput_monitor_init(
      self, mock_logger_client, mock_summary_writer
  ):
    mock_summary_writer.return_value = MagicMock()
    mock_logger_client.return_value = MagicMock()

    monitor = monitoring_elastic.ElasticGoodputMonitor(
        self.job_name,
        self.logger_name,
        self.tensorboard_dir,
        upload_interval=_TEST_UPLOAD_INTERVAL,
        monitoring_enabled=True,
        include_slice_efficiency=True,
    )

    self.assertIsNotNone(monitor)
    self.assertIsInstance(
        monitor._goodput_calculator,
        goodput_elastic.ElasticGoodputCalculator,
    )
    self.assertEqual(
        monitor._worker_config['calculator_class'],
        goodput_elastic.ElasticGoodputCalculator,
    )
    self.assertTrue(monitor._worker_config['include_slice_efficiency'])

  @patch('google.cloud.monitoring_v3.MetricServiceClient')
  @patch('tensorboardX.writer.SummaryWriter')
  @patch('google.cloud.logging.Client')
  def test_upload_goodput_metrics_to_gcm_elastic_badput(
      self,
      mock_logging_client,
      mock_summary_writer,
      mock_metric_service_client,
  ):
    monitor = self._setup_mock_elastic_goodput_monitor(
        mock_logging_client, mock_summary_writer, mock_metric_service_client
    )
    mock_client = mock_metric_service_client.return_value

    monitor._goodput_calculator.get_job_goodput_details = MagicMock(
        return_value={
            MetricType.GOODPUT_TIME.value: {
                GoodputType.TOTAL: 10.0,
            },
            MetricType.BADPUT_TIME.value: {
                BadputType.ELASTIC_SLICE_DOWN: 2.0,
                BadputType.ELASTIC_SCALE_UP: 3.0,
                BadputType.ELASTIC_REINITIALIZATION: 4.0,
            },
            MetricType.DISRUPTION_COUNT.value: 0,
            MetricType.MAX_PRODUCTIVE_STEP.value: 2,
            MetricType.TOTAL_ELAPSED_TIME.value: 20.0,
            MetricType.STEP_TIME_DEVIATION.value: {},
            MetricType.IDEAL_STEP_TIME.value: 1.0,
        }
    )

    details = monitor._goodput_calculator.get_job_goodput_details()
    monitoring._upload_goodput_metrics_to_gcm(
        monitor._metrics_sender,
        details,
        monitor._worker_config,
    )

    expected_calls = [
        mock.call.create_time_series(
            name='projects/test-project',
            time_series=[
                self._create_timeseries(
                    'compute.googleapis.com/workload/badput_time',
                    {
                        'badput_source': 'ELASTIC_SLICE_DOWN',
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
                        'badput_source': 'ELASTIC_SCALE_UP',
                        'accelerator_type': 'test-acc-type',
                    },
                    3.0,
                )
            ],
        ),
        mock.call.create_time_series(
            name='projects/test-project',
            time_series=[
                self._create_timeseries(
                    'compute.googleapis.com/workload/badput_time',
                    {
                        'badput_source': 'ELASTIC_REINITIALIZATION',
                        'accelerator_type': 'test-acc-type',
                    },
                    4.0,
                )
            ],
        ),
    ]

    actual_calls = mock_client.create_time_series.call_args_list

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
  def test_upload_goodput_metrics_to_gcm_slice_efficiency(
      self,
      mock_logging_client,
      mock_summary_writer,
      mock_metric_service_client,
  ):
    monitor = self._setup_mock_elastic_goodput_monitor(
        mock_logging_client, mock_summary_writer, mock_metric_service_client
    )
    mock_client = mock_metric_service_client.return_value

    monitor._goodput_calculator.get_job_goodput_details = MagicMock(
        return_value={
            MetricType.GOODPUT_TIME.value: {
                GoodputType.TOTAL: 10.0,
            },
            MetricType.BADPUT_TIME.value: {
                BadputType.ELASTIC_SLICE_DOWN: 2.0,
            },
            MetricType.DISRUPTION_COUNT.value: 0,
            MetricType.MAX_PRODUCTIVE_STEP.value: 2,
            MetricType.TOTAL_ELAPSED_TIME.value: 20.0,
            MetricType.STEP_TIME_DEVIATION.value: {},
            MetricType.IDEAL_STEP_TIME.value: 1.0,
            'stepping_slice_efficiency': 0.8,
            'available_slice_efficiency': 0.9,
        }
    )

    details = monitor._goodput_calculator.get_job_goodput_details()
    monitoring._upload_goodput_metrics_to_gcm(
        monitor._metrics_sender,
        details,
        monitor._worker_config,
    )

    expected_calls = [
        mock.call.create_time_series(
            name='projects/test-project',
            time_series=[
                self._create_timeseries(
                    'compute.googleapis.com/workload/stepping_slice_efficiency',
                    {
                        'accelerator_type': 'test-acc-type',
                        'window_type': 'CUMULATIVE',
                        'rolling_window_size': '0',
                    },
                    0.8,
                )
            ],
        ),
        mock.call.create_time_series(
            name='projects/test-project',
            time_series=[
                self._create_timeseries(
                    'compute.googleapis.com/workload/available_slice_efficiency',
                    {
                        'accelerator_type': 'test-acc-type',
                        'window_type': 'CUMULATIVE',
                        'rolling_window_size': '0',
                    },
                    0.9,
                )
            ],
        ),
    ]

    actual_calls = mock_client.create_time_series.call_args_list

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
  def test_upload_interval_goodput_metrics_to_gcm_slice_efficiency(
      self,
      mock_logging_client,
      mock_summary_writer,
      mock_metric_service_client,
  ):
    monitor = self._setup_mock_elastic_goodput_monitor(
        mock_logging_client, mock_summary_writer, mock_metric_service_client
    )
    mock_client = mock_metric_service_client.return_value

    interval_metric_details = {
        IntervalMetricType.INTERVAL_SIZE.value: 3600,
        'stepping_slice_efficiency': 0.85,
        'available_slice_efficiency': 0.95,
    }

    monitoring._upload_interval_goodput_metrics_to_gcm(
        monitor._metrics_sender,
        interval_metric_details,
        monitor._worker_config,
    )

    expected_calls = [
        mock.call.create_time_series(
            name='projects/test-project',
            time_series=[
                self._create_timeseries(
                    'compute.googleapis.com/workload/stepping_slice_efficiency',
                    {
                        'accelerator_type': 'test-acc-type',
                        'rolling_window_size': '3600',
                        'window_type': 'INTERVAL',
                    },
                    0.85,
                )
            ],
        ),
        mock.call.create_time_series(
            name='projects/test-project',
            time_series=[
                self._create_timeseries(
                    'compute.googleapis.com/workload/available_slice_efficiency',
                    {
                        'accelerator_type': 'test-acc-type',
                        'rolling_window_size': '3600',
                        'window_type': 'INTERVAL',
                    },
                    0.95,
                )
            ],
        ),
    ]

    actual_calls = mock_client.create_time_series.call_args_list

    for expected_call in expected_calls:
      self.assertTrue(
          any(
              self._compare_calls_ignore_time_series(expected_call, actual)
              for actual in actual_calls
          ),
          f'Expected call not found: {expected_call}',
      )

if __name__ == '__main__':
  absltest.main()
