"""Tests to validate the goodput utils."""
from unittest import mock
from absl.testing import absltest
from cloud_goodput.ml_goodput_measurement.src import goodput_utils
import requests


class GoodputUtilsTest(absltest.TestCase):

  def test_gcp_options_defaults(self):
    """Verifies default values for GCPOptions."""
    options = goodput_utils.GCPOptions()
    self.assertIsNone(options.project_id)
    self.assertIsNone(options.location)
    self.assertEqual(options.replica_id, '0')
    self.assertIsNone(options.acc_type)
    self.assertIsNone(options.cluster_name)
    self.assertTrue(options.enable_gcp_goodput_metrics)

  def test_gcp_options_initialization(self):
    """Verifies initialization of GCPOptions with values."""
    options = goodput_utils.GCPOptions(
        project_id='p1',
        location='l1',
        replica_id='1',
        acc_type='tpu',
        cluster_name='test-cluster',
        enable_gcp_goodput_metrics=False,
    )
    self.assertEqual(options.project_id, 'p1')
    self.assertEqual(options.cluster_name, 'test-cluster')

  @mock.patch.object(goodput_utils, 'get_gcp_metadata')
  def test_get_cluster_name_success(self, mock_get_metadata):
    """Tests successful retrieval of cluster name."""
    mock_get_metadata.return_value = 'test-cluster'

    result = goodput_utils.get_cluster_name()
    self.assertEqual(result, 'test-cluster')
    mock_get_metadata.assert_called_once_with(
        'instance', 'attributes/cluster-name'
    )

  @mock.patch.object(goodput_utils, 'get_gcp_metadata')
  def test_get_cluster_name_failure(self, mock_get_metadata):
    """Tests missing cluster name behavior."""
    mock_get_metadata.return_value = None
    result = goodput_utils.get_cluster_name()
    self.assertIsNone(result)

  @mock.patch('requests.Session')
  def test_get_gcp_metadata_success(self, mock_session_cls):
    """Tests metadata fetching success."""
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.text = 'test-value'
    mock_session = mock_session_cls.return_value
    mock_session.get.return_value = mock_response
    result = goodput_utils.get_gcp_metadata('instance', 'attributes/test-key')
    self.assertEqual(result, 'test-value')
    mock_session.get.assert_called_with(
        'http://metadata.google.internal/computeMetadata/v1/instance/attributes/test-key',
        headers={'Metadata-Flavor': 'Google'},
        timeout=5,
    )

  @mock.patch('requests.Session')
  def test_get_gcp_metadata_404(self, mock_session_cls):
    """Tests metadata fetching failure with 404."""
    mock_response = mock.Mock()
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        '404 Client Error'
    )

    mock_session = mock_session_cls.return_value
    mock_session.get.return_value = mock_response
    result = goodput_utils.get_gcp_metadata('instance', 'missing-key')
    self.assertIsNone(result)

  def test_compute_baseline_step_time(self):
    """Tests lower median baseline calculation for various step sequences."""
    self.assertEqual(goodput_utils.compute_baseline_step_time([]), 0.0)
    self.assertEqual(goodput_utils.compute_baseline_step_time([5.0]), 5.0)
    # 2 steps: lower median selects min
    self.assertEqual(
        goodput_utils.compute_baseline_step_time([10.0, 2.0]), 2.0
    )
    # 3 steps with warmup and delayed compilation
    self.assertEqual(
        goodput_utils.compute_baseline_step_time([1.0, 60.0, 2.0]), 2.0
    )
    # 6 steps with delayed compilation
    self.assertEqual(
        goodput_utils.compute_baseline_step_time([1.0, 60.0, 2.0, 2.0, 2.0, 2.0]),
        2.0,
    )

  def test_compute_ideal_step_time_min_step_time(self):
    """Steps shorter than min_step_time (1 s by default) are ignored."""
    sub_second_steps = [0.10, 0.11, 0.12, 0.10, 0.11, 0.9]
    self.assertIsNone(goodput_utils.compute_ideal_step_time(sub_second_steps))
    # With the floor disabled, the median + 3 * MAD filter drops the 0.9 s step.
    self.assertAlmostEqual(
        goodput_utils.compute_ideal_step_time(
            sub_second_steps, min_step_time=0.0
        ),
        0.108,
    )
    steps = [0.5, 2.0, 2.0, 2.0, 2.0]
    self.assertEqual(goodput_utils.compute_ideal_step_time(steps), 2.0)
    self.assertEqual(
        goodput_utils.compute_ideal_step_time(steps, min_step_time=1.0), 2.0
    )
    self.assertAlmostEqual(
        goodput_utils.compute_ideal_step_time(steps, min_step_time=0.0), 1.7
    )


if __name__ == '__main__':
  absltest.main()
