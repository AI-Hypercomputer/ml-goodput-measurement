"""Goodput monitoring API.

This file contains all the utilities to monitor and upload goodput data of a
user workload to Tensorboard asynchronously.
"""

import datetime
import logging
import math
import os
import threading
import time

from cloud_goodput.ml_goodput_measurement.src import gcp_metrics
from cloud_goodput.ml_goodput_measurement.src import goodput
from cloud_goodput.ml_goodput_measurement.src import goodput_utils
from tensorboardX import writer

BadputType = goodput_utils.BadputType
GCPOptions = goodput_utils.GCPOptions
GCPMetrics = gcp_metrics.GCPMetrics
GoodputCalculator = goodput.GoodputCalculator
IntervalMetricType = goodput_utils.IntervalMetricType
IntervalWorkloadMetricDetails = goodput_utils.IntervalWorkloadMetricDetails
MetricType = goodput_utils.MetricType
MonitoringWindowType = goodput_utils.MonitoringWindowType
ValueType = gcp_metrics.ValueType
UnproductiveTimeDict = goodput.UnproductiveTimeDict
WorkloadMetricDetails = goodput_utils.WorkloadMetricDetails

ACTIVITY_EXCLUSION_LIST = goodput_utils.ACTIVITY_EXCLUSION_LIST
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
      gcp_options: GCPOptions = GCPOptions(),
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
      gcp_options: The options for Google Cloud Monitoring.
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

    # Goodput uploader flags to signal the daemon thread if it exists when to
    # initate shutdown and wait for termination.
    self._goodput_uploader_thread_running = False
    self._goodput_upload_thread = None
    self._termination_event = threading.Event()
    self._termination_event.clear()

    # Step deviation threading flags.
    self._step_deviation_uploader_thread_running = False
    self._step_deviation_upload_thread = None
    self._step_deviation_termination_event = threading.Event()
    self._step_deviation_termination_event.clear()

    # Google Cloud Monitoring configurations.
    self._gcp_options = gcp_options
    self._metrics_sender = None

    # If step deviation is not included, disable GCP step deviation metrics.
    if not self._include_step_deviation:
      self._gcp_options.enable_gcp_step_deviation_metrics = False

    if (
        self._gcp_options.enable_gcp_goodput_metrics
        or self._gcp_options.enable_gcp_step_deviation_metrics
    ):
      if not self._gcp_options.project_id:
        self._gcp_options.project_id = goodput_utils.get_gcp_project_id()
      if not self._gcp_options.location:
        self._gcp_options.location = goodput_utils.get_node_zone()
      if not self._gcp_options.acc_type:
        self._gcp_options.acc_type = goodput_utils.get_accelerator_type()
      if self._gcp_options.project_id and self._gcp_options.location:
        self._metrics_sender = GCPMetrics(
            project_id=self._gcp_options.project_id
        )
      else:
        self._gcp_options.enable_gcp_goodput_metrics = False
        self._gcp_options.enable_gcp_step_deviation_metrics = False
        logger.warning(
            'Project ID or location is not set. Google Cloud Monitoring will not be'
            ' enabled.'
        )
      # Goodput interval uploader flags.
      self._interval_uploader_thread_running = False
      self._interval_goodput_upload_thread = None
      self._interval_termination_event = threading.Event()
      self._interval_termination_event.clear()
      self._rolling_windows = []

  def __del__(self):
    try:
      self.stop_goodput_uploader()
      self.stop_step_deviation_uploader()
      self.stop_rolling_window_goodput_uploader()

    except Exception:  # pylint: disable=broad-exception-caught
      pass

  def _log_tensorboard_scalars(
      self,
      label_prefix: str,
      data: dict[str, float | dict[str, float]],
      step: int,
  ):
    """Logs scalar values (flat or nested) to TensorBoard under a label prefix."""
    if self._writer is None:
      return

    for data_type, data_value in data.items():
      if isinstance(data_value, dict):
        for subtype, subval in data_value.items():
          full_label = f'{label_prefix}/{data_type}/{subtype}'.lower()
          self._writer.add_scalar(
              full_label, float(subval), step, display_name=subtype.lower()
          )
      else:
        full_label = f'{label_prefix}/{data_type.lower()}'
        self._writer.add_scalar(
            full_label, float(data_value), step, display_name=data_type.lower()
        )

    self._writer.flush()

  def _upload_goodput_metrics_to_tensorboard(
      self,
      job_goodput: float,
      badput_breakdown: UnproductiveTimeDict,
      last_step: int,
  ):
    """Writes goodput and badput breakdown to Tensorboard."""
    try:
      self._write_goodput_to_tensorboard(job_goodput, last_step)
      if self._include_badput_breakdown:
        self._write_badput_to_tensorboard(badput_breakdown, last_step)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.error(
          'Error while writing goodput and badput data to Tensorboard. This'
          ' will not impact the workload. Error: %s',
          e,
      )

  def _write_goodput_to_tensorboard(self, job_goodput: float, last_step: int):
    self._log_tensorboard_scalars(
        _TENSORBOARD_GOODPUT_LABEL,
        {_TENSORBOARD_GOODPUT_LABEL: job_goodput},
        last_step,
    )

  def _write_badput_to_tensorboard(
      self,
      job_badput_breakdown: UnproductiveTimeDict,
      last_step: int,
  ):
    """Writes badput breakdown to TensorBoard."""
    flattened_badput: dict[str, float | dict[str, float]] = {}

    for badput_type, badput_value in job_badput_breakdown.items():
      if isinstance(badput_value, dict):
        flattened_badput[badput_type.name.lower()] = {
            subtype.lower(): value for subtype, value in badput_value.items()
        }
      else:
        flattened_badput[badput_type.name.lower()] = badput_value

    self._log_tensorboard_scalars(
        _TENSORBOARD_BADPUT_LABEL,
        flattened_badput,
        last_step,
    )

  def _flatten_badput_dict(
      self,
      badput_time_dict: UnproductiveTimeDict,
  ) -> list[tuple[str, float]]:
    """Flattens nested badput types into (label, value) pairs for export."""
    flat_badput = []
    for badput_type, val in badput_time_dict.items():
      if isinstance(val, dict):
        for subtype, subval in val.items():
          flat_badput.append((f'{badput_type.name}.{subtype.upper()}', subval))
      else:
        flat_badput.append((badput_type.name, val))
    return flat_badput

  def _upload_goodput_metrics_to_gcm(
      self, goodput_details: WorkloadMetricDetails
  ):
    """Sends goodput and badput metrics to GCM."""
    try:
      gcm_metrics = []

      # Populate goodput time metrics.
      for goodput_type, time_value in goodput_details[
          MetricType.GOODPUT_TIME.value
      ].items():
        if goodput_type.name in ACTIVITY_EXCLUSION_LIST:
          continue
        gcm_metrics.append({
            'metric_type': 'compute.googleapis.com/workload/goodput_time',
            'value': time_value,
            'value_type': ValueType.DOUBLE,
            'metric_labels': {
                'goodput_source': goodput_type.name,
                'accelerator_type': self._gcp_options.acc_type,
            },
            'resource_type': 'compute.googleapis.com/Workload',
            'resource_labels': {
                'location': self._gcp_options.location,
                'workload_id': self._job_name,
                'replica_id': self._gcp_options.replica_id,
            },
        })

      # Populate badput time metrics.
      for badput_label, time_value in self._flatten_badput_dict(
          goodput_details[MetricType.BADPUT_TIME.value]
      ):
        if badput_label in ACTIVITY_EXCLUSION_LIST:
          continue
        gcm_metrics.append({
            'metric_type': 'compute.googleapis.com/workload/badput_time',
            'value': time_value,
            'value_type': ValueType.DOUBLE,
            'metric_labels': {
                'badput_source': badput_label,
                'accelerator_type': self._gcp_options.acc_type,
            },
            'resource_type': 'compute.googleapis.com/Workload',
            'resource_labels': {
                'location': self._gcp_options.location,
                'workload_id': self._job_name,
                'replica_id': self._gcp_options.replica_id,
            },
        })

      # Populate disruption metrics.
      gcm_metrics.append({
          'metric_type': 'compute.googleapis.com/workload/disruptions',
          'value': goodput_details[MetricType.DISRUPTION_COUNT.value],
          'value_type': ValueType.INT,
          'metric_labels': {
              'accelerator_type': self._gcp_options.acc_type,
              'window_type': MonitoringWindowType.CUMULATIVE.value,
          },
          'resource_type': 'compute.googleapis.com/Workload',
          'resource_labels': {
              'location': self._gcp_options.location,
              'workload_id': self._job_name,
              'replica_id': self._gcp_options.replica_id,
          },
      })

      # Populate max productive step metrics.
      gcm_metrics.append({
          'metric_type': 'compute.googleapis.com/workload/max_productive_steps',
          'value': goodput_details[MetricType.MAX_PRODUCTIVE_STEP.value],
          'value_type': ValueType.INT,
          'metric_labels': {
              'accelerator_type': self._gcp_options.acc_type,
          },
          'resource_type': 'compute.googleapis.com/Workload',
          'resource_labels': {
              'location': self._gcp_options.location,
              'workload_id': self._job_name,
              'replica_id': self._gcp_options.replica_id,
          },
      })

      # Populate step time deviation metrics.
      step_time_deviations = goodput_details[
          MetricType.STEP_TIME_DEVIATION.value
      ]
      if step_time_deviations:
        step_time_deviation_from_baseline = (
            goodput_utils.compute_step_deviation_from_baseline(
                step_time_deviations
            )
        )
        gcm_metrics.append({
            'metric_type': (
                'compute.googleapis.com/workload/step_time_deviation'
            ),
            'value': step_time_deviation_from_baseline,
            'value_type': ValueType.DOUBLE,
            'metric_labels': {
                'accelerator_type': self._gcp_options.acc_type,
            },
            'resource_type': 'compute.googleapis.com/Workload',
            'resource_labels': {
                'location': self._gcp_options.location,
                'workload_id': self._job_name,
                'replica_id': self._gcp_options.replica_id,
            },
        })

      # Populate total elapsed time metrics.
      gcm_metrics.append({
          'metric_type': 'compute.googleapis.com/workload/total_elapsed_time',
          'value': goodput_details[MetricType.TOTAL_ELAPSED_TIME.value],
          'value_type': ValueType.DOUBLE,
          'metric_labels': {
              'accelerator_type': self._gcp_options.acc_type,
              'window_type': MonitoringWindowType.CUMULATIVE.value,
          },
          'resource_type': 'compute.googleapis.com/Workload',
          'resource_labels': {
              'location': self._gcp_options.location,
              'workload_id': self._job_name,
              'replica_id': self._gcp_options.replica_id,
          },
      })

      # Populate ideal step time metrics.
      gcm_metrics.append({
          'metric_type': 'compute.googleapis.com/workload/performance',
          'value': goodput_details[MetricType.IDEAL_STEP_TIME.value],
          'value_type': ValueType.DOUBLE,
          'resource_type': 'compute.googleapis.com/Workload',
          'resource_labels': {
              'location': self._gcp_options.location,
              'workload_id': self._job_name,
              'replica_id': self._gcp_options.replica_id,
          },
      })

      # Send metrics to Google Cloud Monitoring.
      if self._metrics_sender and gcm_metrics:
        self._metrics_sender.send_metrics(gcm_metrics)

    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.error(
          'Error while sending goodput metrics to GCM. This'
          ' will not impact the workload. Error: %s',
          e,
      )

  def _query_and_upload_goodput(self):
    """Queries and uploads goodput data to Tensorboard."""
    while not self._termination_event.is_set():
      time.sleep(self._upload_interval)
      # Query metrics and update the cache.
      try:
        job_goodput, job_badput_breakdown, last_step = (
            self._goodput_calculator.get_job_goodput(
                include_badput_breakdown=self._include_badput_breakdown
            )
        )
      except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning(
            'Error while querying goodput. Skipping this cycle. Error: %s', e
        )
        continue

      try:
        # Upload metrics to Tensorboard.
        self._upload_goodput_metrics_to_tensorboard(
            job_goodput, job_badput_breakdown, last_step
        )
      except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning(
            'Could not upload goodput metrics to Tensorboard. Skipping'
            ' this cycle. Error: %s',
            e,
        )

      try:
        # Upload metrics to Google Cloud Monitoring.
        if self._gcp_options.enable_gcp_goodput_metrics:
          self._upload_goodput_metrics_to_gcm(
              self._goodput_calculator.get_job_goodput_details()
          )
      except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning(
            'Could not upload goodput metrics to Google Cloud Monitoring.'
            ' Skipping this cycle. Error: %s',
            e,
        )

  def _final_goodput_query_and_upload(self):
    """Performs final cumulative goodput query and uploads data to Tensorboard & GCM."""
    time.sleep(self._upload_interval)
    try:
      job_goodput, job_badput_breakdown, last_step = (
          self._goodput_calculator.get_job_goodput(
              include_badput_breakdown=self._include_badput_breakdown
          )
      )
      self._upload_goodput_metrics_to_tensorboard(
          job_goodput, job_badput_breakdown, last_step
      )
      if self._gcp_options.enable_gcp_goodput_metrics:
        self._upload_goodput_metrics_to_gcm(
            self._goodput_calculator.get_job_goodput_details()
        )
      logger.info(
          'Final goodput query and upload for job: %s and logger: %s completed'
          ' with total goodput: %.2f%%, last step: %d',
          self._job_name,
          self._logger_name,
          job_goodput,
          last_step,
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.error(
          'Error while performing final goodput query and upload for job: %s'
          ' and logger: %s. This will not impact the workload. Error: %s',
          self._job_name,
          self._logger_name,
          e,
      )

  def start_goodput_uploader(self):
    """Starts the goodput uploader thread."""
    if self._goodput_uploader_thread_running:
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
    self._goodput_uploader_thread_running = True

  def stop_goodput_uploader(self):
    """Stops the cumulative goodput uploader thread and performs a final cumulative goodput upload."""
    if not self._goodput_uploader_thread_running:
      raise RuntimeError('Cumulative goodput uploader thread is not running.')

    self._termination_event.set()
    if self._goodput_upload_thread is not None:
      logger.info(
          'Waiting for cumulative goodput query and uploader thread to'
          ' complete.'
      )
      self._goodput_upload_thread.join()
      self._goodput_upload_thread = None
    logger.info(
        'Cumulative goodput query and uploader thread stopped. No more goodput'
        ' data will be uploaded to Tensorboard or GCM.'
    )
    self._goodput_uploader_thread_running = False
    # Final goodput query and upload.
    self._final_goodput_query_and_upload()

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

  def _send_step_deviation_metric_to_gcp(self, step_deviations):
    """Sends step deviation metric to GCM."""
    try:
      if not step_deviations:
        logger.warning(
            'Step deviation is empty. This will not impact the workload.'
        )
        return
      avg_step_deviation = sum(step_deviations.values()) / len(step_deviations)

      if math.isnan(avg_step_deviation):
        logger.warning(
            'Step deviation is NaN. This will not impact the workload.'
        )
        return

      perf_metric = [{
          'metric_type': 'compute.googleapis.com/workload/performance',
          'value': avg_step_deviation,
          'value_type': ValueType.DOUBLE,
          'resource_type': 'compute.googleapis.com/Workload',
          'resource_labels': {
              'location': self._gcp_options.location,
              'workload_id': self._job_name,
              'replica_id': self._gcp_options.replica_id,
          },
      }]
      if self._metrics_sender:
        self._metrics_sender.send_metrics(perf_metric)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.error(
          'Error while sending step deviation to GCM.'
          ' This will not impact the workload. Error: %s',
          e,
      )

  def _query_and_upload_step_deviation_to_tensorboard_and_gcp(self):
    """Queries and uploads step deviation data to Tensorboard and GCM."""
    try:
      step_deviation = self._goodput_calculator.get_step_deviation(
          self._configured_ideal_step_time
      )
      self._write_step_deviation_to_tensorboard(step_deviation)
      if self._gcp_options.enable_gcp_step_deviation_metrics:
        self._send_step_deviation_metric_to_gcp(step_deviation)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.error(
          'Error while querying and uploading step deviation to Tensorboard.'
          ' This will not impact the workload. Error: %s',
          e,
      )

  def _query_and_upload_step_deviation(self):
    """Queries and uploads step deviation data to Tensorboard."""
    while not self._step_deviation_termination_event.is_set():
      time.sleep(self._step_deviation_interval_seconds)
      self._query_and_upload_step_deviation_to_tensorboard_and_gcp()

  def _final_step_deviation_query_and_upload(self):
    """Performs final step deviation query and uploads data to Tensorboard & GCM."""
    time.sleep(self._step_deviation_interval_seconds)
    try:
      step_deviation = self._goodput_calculator.get_step_deviation(
          self._configured_ideal_step_time
      )
      self._write_step_deviation_to_tensorboard(step_deviation)
      if self._gcp_options.enable_gcp_step_deviation_metrics:
        self._send_step_deviation_metric_to_gcp(step_deviation)
      logger.info(
          'Final step deviation query and upload for job: %s and logger: %s'
          ' completed',
          self._job_name,
          self._logger_name,
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.error(
          'Error while performing final step deviation query and upload for'
          ' job: %s and logger: %s. This will not impact the workload. Error:'
          ' %s',
          self._job_name,
          self._logger_name,
          e,
      )

  def start_step_deviation_uploader(self):
    """Starts the step deviation uploader thread."""
    if not self._include_step_deviation:
      logger.info(
          'Step deviation monitoring is disabled. Returning without'
          ' initializing step deviation uploader thread.'
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
        ' deviation data will be uploaded to Tensorboard or GCM.'
    )
    self._step_deviation_uploader_thread_running = False
    # Final step deviation query and upload.
    self._final_step_deviation_query_and_upload()

  def _final_rolling_window_goodput_query_and_upload(self):
    """Performs final rolling window goodput query and uploads data to GCM for all rolling windows."""
    time.sleep(self._upload_interval)
    try:
      now = datetime.datetime.now(datetime.timezone.utc)

      # Perform the final upload for each rolling window.
      for window_size in self._rolling_windows:
        window_end = now
        window_start = now - datetime.timedelta(seconds=window_size)
        window_start = window_start.replace(tzinfo=datetime.timezone.utc)

        # Get rolling window metrics for the current window size.
        rolling_window_metric_details = (
            self._goodput_calculator.get_interval_metric_details(
                window_start, window_end
            )
        )

        # Upload the metrics to GCM.
        self._upload_interval_goodput_metrics_to_gcm(
            rolling_window_metric_details
        )

      logger.info(
          'Final rolling window goodput query and upload for job: %s and'
          ' logger: %s completed.',
          self._job_name,
          self._logger_name,
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.error(
          'Error while performing final rolling window goodput query and upload'
          ' for job: %s and logger: %s. This will not impact the workload.'
          ' Error: %s',
          self._job_name,
          self._logger_name,
          e,
      )

  def _upload_interval_goodput_metrics_to_gcm(
      self,
      interval_metric_details: IntervalWorkloadMetricDetails,
  ):
    """Uploads interval goodput metrics to GCM."""
    try:
      gcm_metrics = []
      window_size = interval_metric_details[
          IntervalMetricType.INTERVAL_SIZE.value
      ]

      # Populate Interval Goodput.
      for goodput_type, goodput_value in interval_metric_details[
          IntervalMetricType.INTERVAL_GOODPUT.value
      ].items():
        if goodput_type.name in ACTIVITY_EXCLUSION_LIST:
          continue
        gcm_metrics.append({
            'metric_type': 'compute.googleapis.com/workload/interval_goodput',
            'value': goodput_value,
            'value_type': ValueType.DOUBLE,
            'metric_labels': {
                'goodput_source': goodput_type.name,
                'accelerator_type': self._gcp_options.acc_type,
                'rolling_window_size': str(window_size),
            },
            'resource_type': 'compute.googleapis.com/Workload',
            'resource_labels': {
                'location': self._gcp_options.location,
                'workload_id': self._job_name,
                'replica_id': self._gcp_options.replica_id,
            },
        })

        # Populate Interval Badput.
        for badput_type, badput_value in self._flatten_badput_dict(
            interval_metric_details[IntervalMetricType.INTERVAL_BADPUT.value]
        ):
          if badput_type in ACTIVITY_EXCLUSION_LIST:
            continue
          gcm_metrics.append({
              'metric_type': 'compute.googleapis.com/workload/interval_badput',
              'value': badput_value,
              'value_type': ValueType.DOUBLE,
              'metric_labels': {
                  'badput_source': badput_type,
                  'accelerator_type': self._gcp_options.acc_type,
                  'rolling_window_size': str(window_size),
              },
              'resource_type': 'compute.googleapis.com/Workload',
              'resource_labels': {
                  'location': self._gcp_options.location,
                  'workload_id': self._job_name,
                  'replica_id': self._gcp_options.replica_id,
              },
          })

      if self._metrics_sender:
        self._metrics_sender.send_metrics(gcm_metrics)

    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.error(
          'Error while uploading interval goodput metrics to GCM. This will'
          ' not impact the workload. Error: %s',
          e,
      )

  def _query_and_upload_rolling_window_goodput(self):
    """Queries and uploads rolling window goodput to GCM."""
    while not self._interval_termination_event.is_set():
      time.sleep(self._upload_interval)
      if not self._gcp_options.enable_gcp_goodput_metrics:
        continue

      try:
        now = datetime.datetime.now(datetime.timezone.utc)
        for window_size in self._rolling_windows:
          window_end = now
          window_start = now - datetime.timedelta(seconds=window_size)
          window_start = window_start.replace(tzinfo=datetime.timezone.utc)
          interval_metric_details = (
              self._goodput_calculator.get_interval_metric_details(
                  window_start, window_end
              )
          )
          self._upload_interval_goodput_metrics_to_gcm(interval_metric_details)
      except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning(
            'Error while querying and uploading rolling window goodput to GCM.'
            'Skipping this cycle. This will not impact the workload. Error: %s',
            e,
        )

  def start_rolling_window_goodput_uploader(
      self, rolling_windows_seconds: list[int]
  ):
    """Starts the goodput uploader thread for user-specified interval windows."""
    if self._interval_uploader_thread_running:
      raise RuntimeError('Goodput interval uploader thread is already running.')

    self._interval_termination_event.clear()
    self._rolling_windows = rolling_windows_seconds
    self._interval_goodput_upload_thread = threading.Thread(
        target=self._query_and_upload_rolling_window_goodput,
        daemon=True,
    )
    logger.info(
        'Starting rolling window goodput query and uploader thread in the'
        ' background for job: %s and logger: %s',
        self._job_name,
        self._logger_name,
    )
    self._interval_goodput_upload_thread.start()
    self._interval_uploader_thread_running = True

  def stop_rolling_window_goodput_uploader(self):
    """Stops the rolling window goodput uploader thread and performs a final rolling window goodput upload."""
    if not self._interval_uploader_thread_running:
      raise RuntimeError(
          'Rolling window goodput uploader thread is not running.'
      )

    self._interval_termination_event.set()
    if self._interval_goodput_upload_thread is not None:
      logger.info(
          'Waiting for rolling window goodput query and uploader thread to'
          ' complete.'
      )
      self._interval_goodput_upload_thread.join()
      self._interval_goodput_upload_thread = None
    logger.info(
        'Rolling window goodput query and uploader thread stopped. No more'
        ' rolling window goodput data will be uploaded to GCM.'
    )

    self._interval_uploader_thread_running = False

    # Perform the final rolling window goodput query and upload
    self._final_rolling_window_goodput_query_and_upload()
