"""Goodput monitoring API.

This file contains all the utilities to monitor and upload goodput data of a
user workload to Tensorboard asynchronously.
"""

import logging
import os
import threading
import time

from ml_goodput_measurement.src.gcp_metrics import GCPMetrics
from ml_goodput_measurement.src.gcp_metrics import ValueType
from ml_goodput_measurement.src.goodput import GoodputCalculator
from ml_goodput_measurement.src.goodput_utils import BadputType
from ml_goodput_measurement.src.goodput_utils import check_gcloud_auth
from tensorboardX import writer


_TENSORBOARD_GCS_SUBDIR = 'goodput'
_TENSORBOARD_GOODPUT_LABEL = 'goodput'
_TENSORBOARD_BADPUT_LABEL = 'badput'
_TENSORBOARD_STEP_DEVIATION_LABEL = 'step_deviation'

logger = logging.getLogger(__name__)


class GoodputMonitor:
  """Queries and uploads goodput data to Tensorboard at a regular interval."""

  def __init__(
      self,
      job_name: str,
      logger_name: str,
      tensorboard_dir: str,
      upload_interval: int,
      monitoring_enabled: bool = False,
      pathway_enabled: bool = False,
      include_badput_breakdown=False,
      include_step_deviation=False,
      configured_ideal_step_time=None,
      step_deviation_interval_seconds=10,
      project_id: str = None,
      location: str = None,
      replica_id: str = None,
      acc_type: str = None,
      send_metrics_to_gcp: bool = False,
  ):
    """Initializes the GoodputMonitor.

    Args:
      job_name: The name of the job to monitor.
      logger_name: The name of the Google Cloud Logging logger to use.
      tensorboard_dir: The directory to write TensorBoard data to.
      upload_interval: The interval to upload data to TensorBoard and GCP
        Monitoring.
      monitoring_enabled: Whether to enable monitoring. If the application is
        interested in monitoring Goodput, it should set this value to True if
        monitoring from TPU worker 0 andthe application's configurations
        request Goodput monitoring.
      pathway_enabled: Whether the application is using Pathways.
      include_badput_breakdown: Whether to query and upload badput breakdown
        data to Tensorboard.
      include_step_deviation: Whether to query and upload step deviation data
        to Tensorboard.
      configured_ideal_step_time: The optional ideal step time configured by
        the user.
      step_deviation_interval_seconds: The interval to query step deviation
        data.
      project_id: The GCP project ID to send metrics to.
      location: The GCP location of the workload.
      replica_id: The GCP replica ID of the workload.
      acc_type: The accelerator type of the workload.
      send_metrics_to_gcp: Whether to send metrics to GCP Monitoring.
    """
    if not monitoring_enabled:
      logger.info(
          'Monitoring is disabled. Returning without initializing'
          ' GoodputMonitor.'
      )
      return

    # Common configurations.
    self._job_name = job_name
    self._logger_name = logger_name
    self._tensorboard_dir = os.path.join(
        tensorboard_dir, _TENSORBOARD_GCS_SUBDIR
    )
    # Goodput configurations.
    self._upload_interval = upload_interval
    self._include_badput_breakdown = include_badput_breakdown

    # Step deviation configurations.
    self._include_step_deviation = include_step_deviation
    self._step_deviation_interval_seconds = step_deviation_interval_seconds
    self._configured_ideal_step_time = configured_ideal_step_time

    # Initialize the GoodputCalculator.
    self._goodput_calculator = GoodputCalculator(
        job_name=self._job_name,
        logger_name=self._logger_name,
        using_pathways=pathway_enabled,
    )
    self._writer = writer.SummaryWriter(self._tensorboard_dir)

    # Goodput uploader flags to signal the daemon thread if it exists when to initate
    # shutdown and wait for termination.
    self._uploader_thread_running = False
    self._goodput_upload_thread = None
    self._termination_event = threading.Event()
    self._termination_event.clear()

    # Step deviation threading flags.
    self._step_deviation_uploader_thread_running = False
    self._step_deviation_upload_thread = None
    self._step_deviation_termination_event = threading.Event()
    self._step_deviation_termination_event.clear()

    # GCP Monitoring configurations.
    self._project_id = project_id
    self._location = location
    self._replica_id = replica_id
    self._acc_type = acc_type
    self._send_metrics_to_gcp = send_metrics_to_gcp
    self._metrics_sender = None
    self._gcp_metrics_thread = None
    if self._send_metrics_to_gcp:
      self._metrics_sender = GCPMetrics(project_id=self._project_id)

  def __del__(self):
    if self._uploader_thread_running:
      self.stop_goodput_uploader()
      self.stop_step_deviation_uploader()

  def _write_goodput_data_to_tensorboard(
      self,
      job_goodput: float,
      badput_breakdown: dict[BadputType, float],
      last_step: int,
  ):
    """Writes goodput and badput breakdown to Tensorboard."""
    self._write_goodput_to_tensorboard(job_goodput, last_step)
    if self._include_badput_breakdown:
      self._write_badput_to_tensorboard(badput_breakdown, last_step)

  def _write_goodput_to_tensorboard(self, job_goodput: float, last_step: int):
    if self._writer is not None:
      self._writer.add_scalar(
          _TENSORBOARD_GOODPUT_LABEL, job_goodput, last_step
      )
      self._writer.flush()

  def _write_badput_to_tensorboard(
      self, job_badput_breakdown: dict[BadputType, float], last_step: int
  ):
    def _get_badput_label(badput_type: BadputType) -> str:
      return str(badput_type.name).lower()

    if self._writer is not None:
      for badput_type, badput_percentage in job_badput_breakdown.items():
        label_suffix = _get_badput_label(badput_type)
        self._writer.add_scalar(
            _TENSORBOARD_BADPUT_LABEL + '/' + label_suffix,
            float(badput_percentage),
            last_step,
            display_name=label_suffix,
        )
      self._writer.flush()

  def _query_and_upload_goodput(self):
    """Queries and uploads goodput data to Tensorboard."""
    while not self._termination_event.is_set():
      time.sleep(self._upload_interval)
      try:
        job_goodput, job_badput_breakdown, last_step = (
            self._goodput_calculator.get_job_goodput(
                include_badput_breakdown=self._include_badput_breakdown
            )
        )
        self._write_goodput_data_to_tensorboard(job_goodput, job_badput_breakdown, last_step)
      except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(
            'Error while querying and uploading goodput to Tensorboard. This'
            ' will not impact the workload. Error: %s',
            e,
        )

  def _send_goodput_metrics_to_gcp(self):
    """Sends goodput and badput metrics to GCP Monitoring."""
    while not self._termination_event.is_set():
      time.sleep(self._upload_interval)
      try:
        goodput_details = self._goodput_calculator.get_job_goodput_details()
        goodput_metrics = []
        for goodput_type, time_value in goodput_details['goodput_time'].items():
          goodput_metrics.append({
              'metric_type': 'compute.googleapis.com/workload/goodput_time',
              'value': time_value,
              'value_type': ValueType.DOUBLE,
              'metric_labels': {
                  'goodput_source': goodput_type.name,
                  'accelerator_type': self._acc_type,
              },
              'resource_type': 'compute.googleapis.com/Workload',
              'resource_labels': {
                  'location': self._location,
                  'workload_id': self._job_name,
                  'replica_id': self._replica_id,
              },
          })
        for badput_type, time_value in goodput_details['badput_time'].items():
          goodput_metrics.append({
              'metric_type': 'compute.googleapis.com/workload/badput_time',
              'value': time_value,
              'value_type': ValueType.DOUBLE,
              'metric_labels': {
                  'badput_source': badput_type.name,
                  'accelerator_type': self._acc_type,
              },
              'resource_type': 'compute.googleapis.com/Workload',
              'resource_labels': {
                  'location': self._location,
                  'workload_id': self._job_name,
                  'replica_id': self._replica_id,
              },
          })
        if self._metrics_sender:
          self._metrics_sender.send_metrics(goodput_metrics)
      except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(
            'Error while sending goodput metrics to GCP Monitoring. This'
            ' will not impact the workload. Error: %s',
            e,
        )

  def start_goodput_uploader(self):
    """Starts the goodput uploader thread."""
    if self._uploader_thread_running:
      raise RuntimeError('Goodput uploader thread is already running.')

    self._termination_event.clear()
    self._goodput_upload_thread = threading.Thread(
        target=self._query_and_upload_goodput, daemon=True
    )
    logger.info(
        'Starting goodput query and uploader thread in the background for job:'
        ' %s and logger: %s',
        self._job_name,
        self._logger_name,
    )

    self._goodput_upload_thread.start()

    # Start the GCP metrics sender thread
    if self._metrics_sender:
      check_gcloud_auth(self._project_id)
      self._gcp_metrics_thread = threading.Thread(
          target=self._send_goodput_metrics_to_gcp, daemon=True
      )
      logger.info(
          'Starting GCP metrics sender thread in the background for job:'
          ' %s and logger: %s',
          self._job_name,
          self._logger_name,
      )
      self._gcp_metrics_thread.start()

    self._uploader_thread_running = True

  def stop_goodput_uploader(self):
    """Stops the goodput uploader thread."""
    if not self._uploader_thread_running:
      raise RuntimeError('Goodput uploader thread is not running.')

    self._termination_event.set()
    if self._goodput_upload_thread is not None:
      logger.info('Waiting for goodput query and uploader thread to complete.')
      self._goodput_upload_thread.join()
      self._goodput_upload_thread = None
    logger.info(
        'Goodput query and uploader thread stopped. No more goodput data will'
        ' be uploaded to Tensorboard.'
    )

    # Stop the GCP metrics sender thread
    if self._gcp_metrics_thread is not None:
      logger.info('Waiting for GCP metrics sender thread to complete.')
      self._gcp_metrics_thread.join()
      self._gcp_metrics_thread = None
      logger.info('GCP metrics sender thread stopped.')

    self._uploader_thread_running = False

  def _write_step_deviation_to_tensorboard(
      self, step_deviation: dict[int, float]
  ):
    if self._writer is not None:
      for step_count, step_deviation in step_deviation.items():
        self._writer.add_scalar(
            _TENSORBOARD_STEP_DEVIATION_LABEL,
            float(step_deviation),
            step_count,
        )
      self._writer.flush()

  def _query_and_upload_step_deviation(self):
    """Queries and uploads step deviation data to Tensorboard."""
    while not self._step_deviation_termination_event.is_set():
      time.sleep(self._step_deviation_interval_seconds)
      try:
        step_deviation = self._goodput_calculator.get_step_deviation(
            self._configured_ideal_step_time
        )
      except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(
            'Error while querying step deviation to Tensorboard.'
            ' This will not impact the workload. Error: %s',
            e,
        )
        continue
      try:
        self._write_step_deviation_to_tensorboard(step_deviation)
      except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(
            'Error while writing step deviation to Tensorboard.'
            ' This will not impact the workload. Error: %s',
            e,
        )

  def start_step_deviation_uploader(self):
    """Starts the step deviation uploader thread."""
    if not self._include_step_deviation:
      logger.info(
          'Step deviation monitoring is disabled. Returning without initializing'
          ' step deviation uploader thread.'
      )
      return

    if self._step_deviation_uploader_thread_running:
      raise RuntimeError('Step deviation uploader thread is already running.')

    self._step_deviation_termination_event.clear()
    self._step_deviation_upload_thread = threading.Thread(
        target=self._query_and_upload_step_deviation, daemon=True
    )
    logger.info(
        'Starting step deviation query and uploader thread in the background'
        ' for job: %s and logger: %s',
        self._job_name,
        self._logger_name,
    )
    self._step_deviation_upload_thread.start()
    self._step_deviation_uploader_thread_running = True

  def stop_step_deviation_uploader(self):
    """Stops the step deviation uploader thread."""
    if not self._step_deviation_uploader_thread_running:
      raise RuntimeError('Step deviation uploader thread is not running.')

    self._step_deviation_termination_event.set()
    if self._step_deviation_upload_thread is not None:
      logger.info(
          'Waiting for step deviation query and uploader thread to complete.'
      )
      self._step_deviation_upload_thread.join()
    logger.info(
        'Step deviation query and uploader thread stopped. No more step'
        ' deviation data will be uploaded to Tensorboard.'
    )
    self._step_deviation_uploader_thread_running = False
