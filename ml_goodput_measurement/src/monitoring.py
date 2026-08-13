"""Goodput monitoring API.

This file contains all the utilities to monitor and upload goodput data of a
user workload to Tensorboard and Google Cloud Monitoring asynchronously.
"""

import datetime
import logging
import math
import multiprocessing
from multiprocessing import synchronize
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
_PROCESS_TERMINATION_TIMEOUT_SECONDS = 10

logger = logging.getLogger(__name__)


def _query_and_upload_goodput_once(
    calculator: GoodputCalculator,
    summary_writer: writer.SummaryWriter,
    metrics_sender: GCPMetrics | None,
    config: dict,
    pid: int,
) -> None:
  """Performs a single cumulative goodput query and upload cycle.

  Used both by the periodic loop in _goodput_worker and for the final,
  warm-cache flush performed just before the worker exits.
  """
  try:
    job_goodput, job_badput, last_step = calculator.get_job_goodput(
        include_badput_breakdown=config['include_badput_breakdown']
    )
  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.warning(
        '[PID: %s] Error while querying goodput for job %s. Skipping this'
        ' cycle. Error: %s',
        pid,
        config['job_name'],
        e,
    )
    return

  try:
    _upload_goodput_metrics_to_tensorboard(
        summary_writer,
        job_goodput,
        job_badput,
        last_step,
        config['include_badput_breakdown'],
    )
  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.warning(
        '[PID: %s] Could not upload goodput metrics to Tensorboard for job'
        ' %s. Error: %s',
        pid,
        config['job_name'],
        e,
    )

  try:
    if config['gcp_options'].enable_gcp_goodput_metrics and metrics_sender:
      # Final attempt: get details from the calculator's state and upload
      details = calculator.get_job_goodput_details()
      _upload_goodput_metrics_to_gcm(metrics_sender, details, config)
  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.warning(
        '[PID: %s] Could not get details or upload metrics to Google Cloud'
        ' Monitoring for job %s. Error: %s',
        pid,
        config['job_name'],
        e,
    )


def _goodput_worker(
    config: dict,
    termination_event: synchronize.Event,
    final_flush_event: synchronize.Event,
):
  """Worker process for querying and uploading cumulative goodput."""
  pid = os.getpid()
  logger.info(
      '[PID: %s] Goodput worker process started for job: %s',
      pid,
      config['job_name'],
  )

  calculator = _create_goodput_calculator(config)
  summary_writer = _create_tensorboard_writer(config)
  metrics_sender = _create_gcp_metrics_sender(config)

  last_gcs_sync_time = time.monotonic()
  gcs_sync_interval = config.get('gcs_sync_interval_seconds', 3600)

  while not termination_event.wait(timeout=config['upload_interval']):
    _query_and_upload_goodput_once(
        calculator, summary_writer, metrics_sender, config, pid
    )

    # Periodic GCS backup
    now = time.monotonic()
    if now - last_gcs_sync_time >= gcs_sync_interval:
      try:
        logger.info('[PID: %s] Syncing timeline cache to GCS...', pid)
        calculator.sync_to_gcs()
        last_gcs_sync_time = now
      except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning('[PID: %s] Failed to sync timeline to GCS: %s', pid, e)

  if final_flush_event.is_set():
    # The calculator's cache is already warm from the loop above, so this
    # final query only fetches entries written since the last cycle.
    logger.info(
        '[PID: %s] Performing final goodput query and upload for job: %s',
        pid,
        config['job_name'],
    )
    _query_and_upload_goodput_once(
        calculator, summary_writer, metrics_sender, config, pid
    )

  summary_writer.close()
  logger.info(
      '[PID: %s] Goodput worker process for job %s stopped.',
      pid,
      config['job_name'],
  )


def _query_and_upload_step_deviation_once(
    calculator: GoodputCalculator,
    summary_writer: writer.SummaryWriter,
    metrics_sender: GCPMetrics | None,
    config: dict,
    pid: int,
) -> None:
  """Performs a single step deviation query and upload cycle.

  Used both by the periodic loop in _step_deviation_worker and for the
  final, warm-cache flush performed just before the worker exits.
  """
  try:
    step_dev = calculator.get_step_deviation(
        config['configured_ideal_step_time']
    )
  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.warning(
        '[PID: %s] Error getting step deviation for job %s. Skipping this'
        ' cycle. Error: %s',
        pid,
        config['job_name'],
        e,
    )
    return
  try:
    _write_step_deviation_to_tensorboard(summary_writer, step_dev)
  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.warning(
        '[PID: %s] Could not write step deviation to Tensorboard for job %s.'
        ' Error: %s',
        pid,
        config['job_name'],
        e,
    )
  try:
    if (
        config['gcp_options'].enable_gcp_step_deviation_metrics
        and metrics_sender
    ):
      _send_step_deviation_metric_to_gcp(metrics_sender, step_dev, config)
  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.warning(
        '[PID: %s] Could not send step deviation metric to GCP for job %s.'
        ' Error: %s',
        pid,
        config['job_name'],
        e,
    )


def _step_deviation_worker(
    config: dict,
    termination_event: synchronize.Event,
    final_flush_event: synchronize.Event,
):
  """Worker process for querying and uploading step deviation."""
  pid = os.getpid()
  logger.info(
      '[PID: %s] Step deviation worker process started for job: %s',
      pid,
      config['job_name'],
  )

  calculator = _create_goodput_calculator(config)
  summary_writer = _create_tensorboard_writer(config)
  metrics_sender = _create_gcp_metrics_sender(config)

  while not termination_event.wait(
      timeout=config['step_deviation_interval_seconds']
  ):
    _query_and_upload_step_deviation_once(
        calculator, summary_writer, metrics_sender, config, pid
    )

  if final_flush_event.is_set():
    logger.info(
        '[PID: %s] Performing final step deviation query and upload for'
        ' job: %s',
        pid,
        config['job_name'],
    )
    _query_and_upload_step_deviation_once(
        calculator, summary_writer, metrics_sender, config, pid
    )

  summary_writer.close()
  logger.info(
      '[PID: %s] Step deviation worker process for job %s stopped.',
      pid,
      config['job_name'],
  )


def _rolling_window_worker(
    config: dict,
    termination_event: synchronize.Event,
    final_flush_event: synchronize.Event,
):
  """Worker process for querying and uploading rolling window goodput."""
  pid = os.getpid()
  logger.info(
      '[PID: %s] Rolling window worker process started for job: %s',
      pid,
      config['job_name'],
  )
  calculator = _create_goodput_calculator(config)
  metrics_sender = _create_gcp_metrics_sender(config)

  while not termination_event.wait(timeout=config['upload_interval']):
    now = datetime.datetime.now(datetime.timezone.utc)
    for window_size in config['rolling_windows']:
      try:
        window_start = now - datetime.timedelta(seconds=window_size)
        window_start = window_start.replace(tzinfo=datetime.timezone.utc)
        interval_metric_details = calculator.get_interval_metric_details(
            window_start, now
        )
        _upload_interval_goodput_metrics_to_gcm(
            metrics_sender, interval_metric_details, config  # pyrefly: ignore[bad-argument-type]
        )
      except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning(
            '[PID: %s] Error in rolling window (size: %ss) for job %s.'
            ' Error: %s',
            pid,
            window_size,
            config['job_name'],
            e,
        )

  if final_flush_event.is_set():
    logger.info(
        '[PID: %s] Performing final rolling window goodput query and upload'
        ' for job: %s',
        pid,
        config['job_name'],
    )
    now = datetime.datetime.now(datetime.timezone.utc)
    for window_size in config['rolling_windows']:
      try:
        window_start = now - datetime.timedelta(seconds=window_size)
        window_start = window_start.replace(tzinfo=datetime.timezone.utc)
        interval_metric_details = calculator.get_interval_metric_details(
            window_start, now
        )
        _upload_interval_goodput_metrics_to_gcm(
            metrics_sender, interval_metric_details, config  # pyrefly: ignore[bad-argument-type]
        )
      except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning(
            '[PID: %s] Error in rolling window (size: %ss) for job %s.'
            ' Error: %s',
            pid,
            window_size,
            config['job_name'],
            e,
        )

  logger.info(
      '[PID: %s] Rolling window worker process for job %s stopped.',
      pid,
      config['job_name'],
  )


def _run_with_timeout(
    func,
    timeout_seconds: float | None,
    description: str,
) -> None:
  """Runs func, abandoning it if it does not complete within timeout_seconds.

  If timeout_seconds is None, func is run synchronously with no timeout,
  preserving the default blocking behavior. Otherwise func is run in a daemon
  thread so that a slow or hung network call (e.g. fetching log entries or
  uploading metrics) cannot block workload shutdown past timeout_seconds.

  Args:
    func: A zero-argument callable to run.
    timeout_seconds: The maximum time to wait for func to complete, or None
      to wait indefinitely.
    description: A human-readable description of func, used for logging if
      it is abandoned.
  """
  if timeout_seconds is None:
    func()
    return
  thread = threading.Thread(target=func, daemon=True)
  thread.start()
  thread.join(timeout=timeout_seconds)
  if thread.is_alive():
    logger.warning(
        '%s did not complete within %s seconds and will be abandoned to'
        ' avoid blocking workload shutdown. The job may be missing its'
        ' final goodput metrics upload.',
        description,
        timeout_seconds,
    )


def _create_goodput_calculator(config: dict) -> GoodputCalculator:
  """Creates a GoodputCalculator instance from the shared config."""
  calculator_class = config.get('calculator_class', GoodputCalculator)
  return calculator_class(
      job_name=config['job_name'],
      logger_name=config['logger_name'],
      using_pathways=config['pathway_enabled'],
      gcs_path=config.get('tensorboard_dir'),
      cache_dir=config.get('cache_dir', '/tmp'),
      max_logs_retention_period=config.get('max_logs_retention_period'),
  )


def _create_gcp_metrics_sender(config: dict) -> GCPMetrics | None:
  """Creates a GCPMetrics instance or None based on the config."""
  return (
      GCPMetrics(project_id=config['gcp_options'].project_id)
      if config['gcp_options'].project_id
      else None
  )


def _create_tensorboard_writer(config: dict) -> writer.SummaryWriter:
  """Creates a SummaryWriter instance from the shared config."""
  return writer.SummaryWriter(config['tensorboard_dir'])


def _log_tensorboard_scalars(
    summary_writer: writer.SummaryWriter,
    label_prefix: str,
    data: dict[str, float | dict[str, float]],
    step: int,
):
  """Logs scalar values to TensorBoard."""
  for data_type, data_value in data.items():
    if isinstance(data_value, dict):
      for subtype, subval in data_value.items():
        full_label = f'{label_prefix}/{data_type}/{subtype}'.lower()
        summary_writer.add_scalar(
            full_label, float(subval), step, display_name=subtype.lower()
        )
    else:
      full_label = f'{label_prefix}/{data_type.lower()}'
      summary_writer.add_scalar(
          full_label, float(data_value), step, display_name=data_type.lower()
      )
  summary_writer.flush()


def _upload_goodput_metrics_to_tensorboard(
    summary_writer: writer.SummaryWriter,
    job_goodput: float,
    badput_breakdown: UnproductiveTimeDict,
    last_step: int,
    include_badput_breakdown: bool,
):
  """Writes goodput and badput breakdown to Tensorboard."""
  try:
    _log_tensorboard_scalars(
        summary_writer,
        _TENSORBOARD_GOODPUT_LABEL,
        {_TENSORBOARD_GOODPUT_LABEL: job_goodput},
        last_step,
    )
    if include_badput_breakdown:
      flattened_badput: dict[str, float | dict[str, float]] = {
          bt.name.lower(): (
              {st.lower(): v for st, v in bv.items()}
              if isinstance(bv, dict)
              else bv
          )
          for bt, bv in badput_breakdown.items()
      }
      _log_tensorboard_scalars(
          summary_writer, _TENSORBOARD_BADPUT_LABEL, flattened_badput, last_step
      )
  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.error('Error writing to Tensorboard: %s', e)


def _flatten_badput_dict(
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
    metrics_sender: GCPMetrics,
    goodput_details: WorkloadMetricDetails,
    config: dict,
):
  """Sends goodput and badput metrics to GCM.

  This is a standalone function designed to be run in a separate process.

  Args:
    metrics_sender: An initialized GCPMetrics client instance.
    goodput_details: A dictionary containing the detailed metrics to upload.
    config: A dictionary containing all necessary configuration.
  """
  try:
    gcm_metrics = []
    # Use the config dictionary instead of self-based attributes
    gcp_options = config['gcp_options']
    job_name = config['job_name']
    optional_labels = {}
    cluster_name = getattr(gcp_options, 'cluster_name', None)
    if cluster_name:
      optional_labels['cluster_name'] = cluster_name

    def _build_labels(labels_dict):
      labels = labels_dict.copy()
      labels.update(optional_labels)
      return labels

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
          'metric_labels': _build_labels({
              'goodput_source': goodput_type.name,
              'accelerator_type': gcp_options.acc_type,
          }),
          'resource_type': 'compute.googleapis.com/Workload',
          'resource_labels': {
              'location': gcp_options.location,
              'workload_id': job_name,
              'replica_id': gcp_options.replica_id,
          },
      })

    # Populate badput time metrics.
    for badput_label, time_value in _flatten_badput_dict(
        goodput_details[MetricType.BADPUT_TIME.value]
    ):
      if badput_label in ACTIVITY_EXCLUSION_LIST:
        continue
      gcm_metrics.append({
          'metric_type': 'compute.googleapis.com/workload/badput_time',
          'value': time_value,
          'value_type': ValueType.DOUBLE,
          'metric_labels': _build_labels({
              'badput_source': badput_label,
              'accelerator_type': gcp_options.acc_type,
          }),
          'resource_type': 'compute.googleapis.com/Workload',
          'resource_labels': {
              'location': gcp_options.location,
              'workload_id': job_name,
              'replica_id': gcp_options.replica_id,
          },
      })

    # Populate disruption metrics.
    gcm_metrics.append({
        'metric_type': 'compute.googleapis.com/workload/disruptions',
        'value': goodput_details[MetricType.DISRUPTION_COUNT.value],
        'value_type': ValueType.INT,
        'metric_labels': _build_labels({
            'accelerator_type': gcp_options.acc_type,
            'window_type': MonitoringWindowType.CUMULATIVE.value,
        }),
        'resource_type': 'compute.googleapis.com/Workload',
        'resource_labels': {
            'location': gcp_options.location,
            'workload_id': job_name,
            'replica_id': gcp_options.replica_id,
        },
    })

    # Populate max productive step metrics.
    gcm_metrics.append({
        'metric_type': 'compute.googleapis.com/workload/max_productive_steps',
        'value': goodput_details[MetricType.MAX_PRODUCTIVE_STEP.value],
        'value_type': ValueType.INT,
        'metric_labels': _build_labels(
            {'accelerator_type': gcp_options.acc_type}
        ),
        'resource_type': 'compute.googleapis.com/Workload',
        'resource_labels': {
            'location': gcp_options.location,
            'workload_id': job_name,
            'replica_id': gcp_options.replica_id,
        },
    })

    # Populate step time deviation metrics.
    step_time_deviations = goodput_details.get(
        MetricType.STEP_TIME_DEVIATION.value
    )
    if step_time_deviations:
      step_time_deviation_from_baseline = (
          goodput_utils.compute_step_deviation_from_baseline(
              step_time_deviations
          )
      )
      gcm_metrics.append({
          'metric_type': 'compute.googleapis.com/workload/step_time_deviation',
          'value': step_time_deviation_from_baseline,
          'value_type': ValueType.DOUBLE,
          'metric_labels': _build_labels(
              {'accelerator_type': gcp_options.acc_type}
          ),
          'resource_type': 'compute.googleapis.com/Workload',
          'resource_labels': {
              'location': gcp_options.location,
              'workload_id': job_name,
              'replica_id': gcp_options.replica_id,
          },
      })

    # Populate total elapsed time metrics.
    gcm_metrics.append({
        'metric_type': 'compute.googleapis.com/workload/total_elapsed_time',
        'value': goodput_details[MetricType.TOTAL_ELAPSED_TIME.value],
        'value_type': ValueType.DOUBLE,
        'metric_labels': _build_labels({
            'accelerator_type': gcp_options.acc_type,
            'window_type': MonitoringWindowType.CUMULATIVE.value,
        }),
        'resource_type': 'compute.googleapis.com/Workload',
        'resource_labels': {
            'location': gcp_options.location,
            'workload_id': job_name,
            'replica_id': gcp_options.replica_id,
        },
    })

    # Populate ideal step time metrics.
    gcm_metrics.append({
        'metric_type': 'compute.googleapis.com/workload/performance',
        'value': goodput_details[MetricType.IDEAL_STEP_TIME.value],
        'value_type': ValueType.DOUBLE,
        'resource_type': 'compute.googleapis.com/Workload',
        'resource_labels': {
            'location': gcp_options.location,
            'workload_id': job_name,
            'replica_id': gcp_options.replica_id,
        },
    })

    # Populate slice efficiency metrics (cumulative).
    slice_efficiency_metrics = []
    if config.get('include_slice_efficiency', False):
      for metric_type, key in (
          (
              'compute.googleapis.com/workload/stepping_slice_efficiency',
              'stepping_slice_efficiency',
          ),
          (
              'compute.googleapis.com/workload/available_slice_efficiency',
              'available_slice_efficiency',
          ),
      ):
        value = goodput_details.get(key)
        if value is not None:
          slice_efficiency_metrics.append({
              'metric_type': metric_type,
              'value': value,
              'value_type': ValueType.DOUBLE,
              'metric_labels': _build_labels({
                  'accelerator_type': gcp_options.acc_type,
                  'window_type': 'CUMULATIVE',
                  'rolling_window_size': '0',
              }),
              'resource_type': 'compute.googleapis.com/Workload',
              'resource_labels': {
                  'location': gcp_options.location,
                  'workload_id': job_name,
                  'replica_id': gcp_options.replica_id,
              },
          })
    # TODO(b/519328677): Enable after metrics rollout
    gcm_metrics.extend(slice_efficiency_metrics)

    # Send metrics to Google Cloud Monitoring.
    if metrics_sender and gcm_metrics:
      log_context = {
          'job_name': config.get('job_name', 'unknown-job'),
          'pid': os.getpid(),
          'worker': multiprocessing.current_process().name,
          'metrics_type': 'cumulative',
      }
      metrics_sender.send_metrics(gcm_metrics, context=log_context)

  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.error(
        'Error while sending goodput metrics to GCM. This'
        ' will not impact the workload. Error: %s',
        e,
    )


def _write_step_deviation_to_tensorboard(
    summary_writer: writer.SummaryWriter, step_deviation: dict[int, float]
):
  """Writes step deviation to Tensorboard."""
  try:
    if summary_writer is not None:
      for step_count, deviation_value in step_deviation.items():
        summary_writer.add_scalar(
            _TENSORBOARD_STEP_DEVIATION_LABEL,
            float(deviation_value),
            step_count,
        )
      summary_writer.flush()
  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.error('Error writing step deviation to Tensorboard: %s', e)


def _send_step_deviation_metric_to_gcp(
    metrics_sender: GCPMetrics, step_deviations: dict, config: dict
):
  """Sends step deviation metric to GCM."""
  try:
    if not step_deviations:
      return
    avg_step_deviation = sum(step_deviations.values()) / len(step_deviations)
    if math.isnan(avg_step_deviation):
      return

    gcp_options = config['gcp_options']
    perf_metric = [{
        'metric_type': 'compute.googleapis.com/workload/performance',
        'value': avg_step_deviation,
        'value_type': ValueType.DOUBLE,
        'resource_type': 'compute.googleapis.com/Workload',
        'resource_labels': {
            'location': gcp_options.location,
            'workload_id': config['job_name'],
            'replica_id': gcp_options.replica_id,
        },
    }]
    if metrics_sender:
      log_context = {
          'job_name': config.get('job_name', 'unknown-job'),
          'pid': os.getpid(),
          'worker': multiprocessing.current_process().name,
          'metrics_type': 'step-time-deviation',
      }
      metrics_sender.send_metrics(perf_metric, context=log_context)
  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.error('Error sending step deviation to GCM: %s', e)


def _upload_interval_goodput_metrics_to_gcm(
    metrics_sender: GCPMetrics,
    interval_metric_details: IntervalWorkloadMetricDetails,
    config: dict,
):
  """Uploads interval goodput metrics to GCM.

  This is a standalone function designed to be run in a separate process.

  Args:
    metrics_sender: An initialized GCPMetrics client instance.
    interval_metric_details: A dictionary containing the detailed interval
      metrics to upload.
    config: A dictionary containing all necessary configuration.
  """
  try:
    gcm_metrics = []
    gcp_options = config['gcp_options']
    job_name = config['job_name']
    window_size = interval_metric_details.get(
        IntervalMetricType.INTERVAL_SIZE.value
    )
    if not window_size:
      logger.warning(
          'Interval size not found in metric details. Skipping upload.'
      )
      return

    optional_labels = {}
    cluster_name = getattr(gcp_options, 'cluster_name', None)
    if cluster_name:
      optional_labels['cluster_name'] = cluster_name

    def _build_labels(labels_dict):
      labels = labels_dict.copy()
      labels.update(optional_labels)
      return labels

    # Populate Interval Goodput.
    interval_goodput_data = interval_metric_details.get(
        IntervalMetricType.INTERVAL_GOODPUT.value, {}
    )
    if isinstance(interval_goodput_data, dict):
      for goodput_type, goodput_value in interval_goodput_data.items():
        if goodput_type.name in ACTIVITY_EXCLUSION_LIST:
          continue
        gcm_metrics.append({
            'metric_type': 'compute.googleapis.com/workload/interval_goodput',
            'value': goodput_value,
            'value_type': ValueType.DOUBLE,
            'metric_labels': _build_labels({
                'goodput_source': goodput_type.name,
                'accelerator_type': gcp_options.acc_type,
                'rolling_window_size': str(window_size),
            }),
            'resource_type': 'compute.googleapis.com/Workload',
            'resource_labels': {
                'location': gcp_options.location,
                'workload_id': job_name,
                'replica_id': gcp_options.replica_id,
            },
        })

    # Populate Interval Badput.
    interval_badput_data = interval_metric_details.get(
        IntervalMetricType.INTERVAL_BADPUT.value, {}
    )
    for badput_type, badput_value in _flatten_badput_dict(interval_badput_data):
      if badput_type in ACTIVITY_EXCLUSION_LIST:
        continue
      gcm_metrics.append({
          'metric_type': 'compute.googleapis.com/workload/interval_badput',
          'value': badput_value,
          'value_type': ValueType.DOUBLE,
          'metric_labels': _build_labels({
              'badput_source': badput_type,
              'accelerator_type': gcp_options.acc_type,
              'rolling_window_size': str(window_size),
          }),
          'resource_type': 'compute.googleapis.com/Workload',
          'resource_labels': {
              'location': gcp_options.location,
              'workload_id': job_name,
              'replica_id': gcp_options.replica_id,
          },
      })

    # Populate slice efficiency metrics (interval).
    slice_efficiency_metrics = []
    if config.get('include_slice_efficiency', False):
      for metric_type, key in (
          (
              'compute.googleapis.com/workload/stepping_slice_efficiency',
              'stepping_slice_efficiency',
          ),
          (
              'compute.googleapis.com/workload/available_slice_efficiency',
              'available_slice_efficiency',
          ),
      ):
        value = interval_metric_details.get(key)
        if value is not None:
          slice_efficiency_metrics.append({
              'metric_type': metric_type,
              'value': value,
              'value_type': ValueType.DOUBLE,
              'metric_labels': _build_labels({
                  'accelerator_type': gcp_options.acc_type,
                  'rolling_window_size': str(window_size),
                  'window_type': 'INTERVAL',
              }),
              'resource_type': 'compute.googleapis.com/Workload',
              'resource_labels': {
                  'location': gcp_options.location,
                  'workload_id': job_name,
                  'replica_id': gcp_options.replica_id,
              },
          })
    # TODO(b/519328677): Enable after metrics rollout
    gcm_metrics.extend(slice_efficiency_metrics)

    if metrics_sender and gcm_metrics:
      log_context = {
          'job_name': config.get('job_name', 'unknown-job'),
          'pid': os.getpid(),
          'worker': multiprocessing.current_process().name,
          'metrics_type': 'rolling-window',
          'window_size': str(window_size),
      }
      metrics_sender.send_metrics(gcm_metrics, context=log_context)

  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.error(
        'Error while uploading interval goodput metrics to GCM. This will'
        ' not impact the workload. Error: %s',
        e,
    )


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
      skip_final_flush: bool = False,
      final_flush_timeout_seconds: (
          float | None
      ) = _PROCESS_TERMINATION_TIMEOUT_SECONDS,
      gcs_sync_interval_seconds: int = 3600,
      cache_dir: str = '/tmp',
      max_logs_retention_period: datetime.timedelta | None = None,
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
        monitoring from TPU worker 0 and the application's configurations
        request Goodput monitoring.
      pathway_enabled: Whether the application is using Pathways.
      include_badput_breakdown: Whether to query and upload badput breakdown
        data to Tensorboard.
      include_step_deviation: Whether to query and upload step deviation data to
        Tensorboard.
      configured_ideal_step_time: The optional ideal step time configured by the
        user.
      step_deviation_interval_seconds: The interval to query step deviation
        data.
      gcp_options: The options for Google Cloud Monitoring.
      skip_final_flush: Whether to skip the final goodput metrics flush upon
        termination.
      final_flush_timeout_seconds: The maximum time to wait for the final
        goodput metrics flush (query plus Tensorboard/GCM upload) to complete
        upon termination. If an uploader process is running, this is used as the
        join timeout so the worker has time to perform the flush itself using
        its warm cache before exiting; if no uploader process was ever started,
        it instead bounds a one-off cold query run by the caller. Defaults to
        _PROCESS_TERMINATION_TIMEOUT_SECONDS. If the flush does not complete
        within this timeout, it is abandoned (and the worker, if any, is
        terminated) so that it does not block workload shutdown (e.g. behind a
        multi-host termination barrier). Pass None to block until the flush
        completes (the legacy behavior). Has no effect if skip_final_flush is
        True.
      gcs_sync_interval_seconds: The interval to sync the cache to GCS.
      cache_dir: Local directory to store the cache files.
      max_logs_retention_period: The maximum retention period for Cloud Logging
        logs. If not set, defaults to 7 days.
    """
    if not monitoring_enabled:
      logger.info(
          'Monitoring is disabled. GoodputMonitor will not be initialized.'
      )
      self._initialized = False
      return
    self._initialized = True
    self._skip_final_flush = skip_final_flush
    self._final_flush_timeout_seconds = final_flush_timeout_seconds

    # Environment variable overrides for testing
    env_sync_interval = os.environ.get('ML_GOODPUT_GCS_SYNC_INTERVAL_SECONDS')
    if env_sync_interval is not None:
      try:
        gcs_sync_interval_seconds = int(env_sync_interval)
        logger.info(
            'Overriding GCS sync interval to %d seconds from environment.',
            gcs_sync_interval_seconds,
        )
      except ValueError:
        logger.warning(
            'Invalid ML_GOODPUT_GCS_SYNC_INTERVAL_SECONDS: %s',
            env_sync_interval,
        )

    env_cache_dir = os.environ.get('ML_GOODPUT_CACHE_DIR')
    if env_cache_dir is not None:
      cache_dir = env_cache_dir
      logger.info(
          'Overriding cache directory to %s from environment.', cache_dir
      )

    self._goodput_calculator = GoodputCalculator(
        job_name=job_name,
        logger_name=logger_name,
        using_pathways=pathway_enabled,
        gcs_path=tensorboard_dir,
        cache_dir=cache_dir,
        max_logs_retention_period=max_logs_retention_period,
    )
    tensorboard_path = os.path.join(tensorboard_dir, _TENSORBOARD_GCS_SUBDIR)
    self._writer = writer.SummaryWriter(tensorboard_path)

    self._metrics_sender = None
    if (
        gcp_options.enable_gcp_goodput_metrics
        or gcp_options.enable_gcp_step_deviation_metrics
    ):
      if not gcp_options.project_id:
        gcp_options.project_id = goodput_utils.get_gcp_project_id()
      if not gcp_options.location:
        gcp_options.location = goodput_utils.get_node_zone()
      if not gcp_options.acc_type:
        gcp_options.acc_type = goodput_utils.get_accelerator_type()
      if gcp_options.project_id and gcp_options.location:
        self._metrics_sender = GCPMetrics(project_id=gcp_options.project_id)
      else:
        logger.warning(
            'Project ID or location could not be determined. Disabling GCP'
            ' Monitoring.'
        )
        gcp_options.enable_gcp_goodput_metrics = False
        gcp_options.enable_gcp_step_deviation_metrics = False

    self._worker_config = {
        'job_name': job_name,
        'logger_name': logger_name,
        'tensorboard_dir': os.path.join(
            tensorboard_dir, _TENSORBOARD_GCS_SUBDIR
        ),
        'upload_interval': upload_interval,
        'pathway_enabled': pathway_enabled,
        'include_badput_breakdown': include_badput_breakdown,
        'include_step_deviation': include_step_deviation,
        'configured_ideal_step_time': configured_ideal_step_time,
        'step_deviation_interval_seconds': step_deviation_interval_seconds,
        'gcp_options': gcp_options,
        'rolling_windows': [],
        'calculator_class': GoodputCalculator,
        'gcs_sync_interval_seconds': gcs_sync_interval_seconds,
        'cache_dir': cache_dir,
        'max_logs_retention_period': max_logs_retention_period,
    }

    # Process management attributes
    self._goodput_process = None
    self._goodput_termination_event = multiprocessing.Event()
    self._goodput_final_flush_event = multiprocessing.Event()

    self._step_deviation_process = None
    self._step_deviation_termination_event = multiprocessing.Event()
    self._step_deviation_final_flush_event = multiprocessing.Event()

    self._rolling_window_process = None
    self._rolling_window_termination_event = multiprocessing.Event()
    self._rolling_window_final_flush_event = multiprocessing.Event()

  def __del__(self):
    if not getattr(self, '_initialized', False):
      return

    try:
      goodput_process = getattr(self, '_goodput_process', None)
      if goodput_process and goodput_process.is_alive():
        self.stop_goodput_uploader()
      step_deviation_process = getattr(self, '_step_deviation_process', None)
      if step_deviation_process and step_deviation_process.is_alive():
        self.stop_step_deviation_uploader()
      rolling_window_process = getattr(self, '_rolling_window_process', None)
      if rolling_window_process and rolling_window_process.is_alive():
        self.stop_rolling_window_goodput_uploader()
    except Exception:  # pylint: disable=broad-except
      pass

  def _worker_join_timeout(self, should_skip: bool) -> float | None:
    """Returns the timeout to use when joining an uploader process.

    When a final flush is requested, the worker performs it just before
    exiting (see _goodput_worker et al.), so the join must wait long enough
    to cover that query and upload too. final_flush_timeout_seconds is the
    customer-facing knob for that combined wait. Otherwise (skip_final_flush),
    there is no flush work to wait for, so the fixed process-termination
    timeout applies.
    """
    if should_skip:
      return _PROCESS_TERMINATION_TIMEOUT_SECONDS
    return getattr(
        self, '_final_flush_timeout_seconds', _PROCESS_TERMINATION_TIMEOUT_SECONDS
    )

  def start_goodput_uploader(self):
    """Starts the goodput uploader process."""
    if not self._initialized:
      return
    if self._goodput_process and self._goodput_process.is_alive():
      logger.warning(
          'Cumulative goodput uploader process (PID: %s) is already running.',
          self._goodput_process.pid,
      )
      return
    self._goodput_termination_event.clear()
    self._goodput_final_flush_event.clear()
    self._goodput_process = multiprocessing.Process(
        target=_goodput_worker,
        args=(
            self._worker_config,
            self._goodput_termination_event,
            self._goodput_final_flush_event,
        ),
        daemon=True,
    )
    self._goodput_process.start()
    logger.info(
        'Cumulative goodput monitoring process started for job: %s (PID: %s)',
        self._worker_config['job_name'],
        self._goodput_process.pid,
    )

  def stop_goodput_uploader(self, skip_final_flush: bool | None = None):
    """Stops the cumulative goodput uploader process and performs a final upload."""
    if not self._initialized:
      return

    should_skip = (
        skip_final_flush
        if skip_final_flush is not None
        else getattr(self, '_skip_final_flush', False)
    )
    worker_was_running = self._goodput_process is not None

    if self._goodput_process:
      pid = self._goodput_process.pid
      logger.info('Shutting down cumulative goodput process (PID: %s)', pid)
      # Perform final query and upload using its own warm cache before it exits.
      if not should_skip:
        self._goodput_final_flush_event.set()
      self._goodput_termination_event.set()
      self._goodput_process.join(
          timeout=self._worker_join_timeout(should_skip)
      )
      if self._goodput_process.is_alive():
        logger.warning(
            'Cumulative goodput process (PID: %s) did not exit gracefully.'
            ' Terminating forcefully.',
            pid,
        )
        self._goodput_process.terminate()
        self._goodput_process.join()

      exit_code = self._goodput_process.exitcode
      if exit_code:
        logger.warning(
            'Goodput process (PID: %s) terminated abnormally or was forcefully'
            ' terminated. Exit Code: %s ',
            pid,
            exit_code,
        )
    self._goodput_process = None
    if not should_skip and not worker_was_running:
      # No uploader process was ever started, so there is no warm cache to
      # draw from. Fall back to a one-off query, bounded by
      # final_flush_timeout_seconds.
      _run_with_timeout(
          self._final_goodput_query_and_upload,
          getattr(self, '_final_flush_timeout_seconds', None),
          f'Final goodput query and upload for job: {self._worker_config["job_name"]}',
      )

  def _final_goodput_query_and_upload(self):
    """Performs final cumulative goodput query and uploads data to Tensorboard & GCM."""
    try:
      calculator = self._goodput_calculator
      job_goodput, job_badput, last_step = calculator.get_job_goodput(
          include_badput_breakdown=self._worker_config[
              'include_badput_breakdown'
          ]
      )
      with _create_tensorboard_writer(self._worker_config) as summary_writer:
        _upload_goodput_metrics_to_tensorboard(
            summary_writer,
            job_goodput,
            job_badput,
            last_step,
            self._worker_config['include_badput_breakdown'],
        )
      metrics_sender = self._metrics_sender
      if (
          self._worker_config['gcp_options'].enable_gcp_goodput_metrics
          and metrics_sender
      ):
        details = calculator.get_job_goodput_details()
        _upload_goodput_metrics_to_gcm(
            metrics_sender, details, self._worker_config
        )
      logger.info(
          'Final goodput query and upload for job: %s and logger: %s completed'
          ' with total goodput: %.2f%%, last step: %d',
          self._worker_config['job_name'],
          self._worker_config['logger_name'],
          job_goodput,
          last_step,
      )
    except Exception as e:  # pylint: disable=broad-except
      logger.error(
          'Error while performing final goodput query and upload for job: %s'
          ' and logger: %s. This will not impact the workload. Error: %s',
          self._worker_config['job_name'],
          self._worker_config['logger_name'],
          e,
      )

  def start_step_deviation_uploader(self):
    """Starts the step deviation uploader process."""
    if (
        not self._initialized
        or not self._worker_config['include_step_deviation']
    ):
      logger.info(
          'Step deviation monitoring is disabled. Returning without'
          ' initializing step deviation uploader thread.'
      )
      return
    if self._step_deviation_process and self._step_deviation_process.is_alive():
      logger.warning(
          'Step deviation uploader process (PID: %s) is already running.',
          self._step_deviation_process.pid,
      )
      return
    self._step_deviation_termination_event.clear()
    self._step_deviation_final_flush_event.clear()
    self._step_deviation_process = multiprocessing.Process(
        target=_step_deviation_worker,
        args=(
            self._worker_config,
            self._step_deviation_termination_event,
            self._step_deviation_final_flush_event,
        ),
        daemon=True,
    )
    self._step_deviation_process.start()
    logger.info(
        'Step deviation process started for job: %s (PID: %s)',
        self._worker_config['job_name'],
        self._step_deviation_process.pid,
    )

  def stop_step_deviation_uploader(self, skip_final_flush: bool | None = None):
    """Stops the step deviation uploader process and performs a final upload."""
    if not self._initialized:
      return

    should_skip = (
        skip_final_flush
        if skip_final_flush is not None
        else getattr(self, '_skip_final_flush', False)
    )
    worker_was_running = self._step_deviation_process is not None

    if self._step_deviation_process:
      pid = self._step_deviation_process.pid
      logger.info('Shutting down step deviation process (PID: %s)', pid)

      if not should_skip:
        self._step_deviation_final_flush_event.set()
      self._step_deviation_termination_event.set()
      self._step_deviation_process.join(
          timeout=self._worker_join_timeout(should_skip)
      )

      if self._step_deviation_process.is_alive():
        logger.warning(
            'Step deviation process (PID: %s) did not exit gracefully.'
            ' Terminating forcefully.',
            pid,
        )
        self._step_deviation_process.terminate()
        self._step_deviation_process.join()

      exit_code = self._step_deviation_process.exitcode
      if exit_code:
        logger.warning(
            'Step deviation process (PID: %s) terminated abnormally. Exit'
            ' Code: %s',
            pid,
            exit_code,
        )

    self._step_deviation_process = None
    if not should_skip and not worker_was_running:
      _run_with_timeout(
          self._final_step_deviation_query_and_upload,
          getattr(self, '_final_flush_timeout_seconds', None),
          f'Final step deviation query and upload for job: {self._worker_config["job_name"]}',
      )

  def _final_step_deviation_query_and_upload(self):
    """Performs a final step deviation query and upload."""
    try:
      calculator = self._goodput_calculator
      step_dev = calculator.get_step_deviation(
          self._worker_config['configured_ideal_step_time']
      )
      with _create_tensorboard_writer(self._worker_config) as summary_writer:
        _write_step_deviation_to_tensorboard(summary_writer, step_dev)

      metrics_sender = self._metrics_sender
      if (
          self._worker_config['gcp_options'].enable_gcp_step_deviation_metrics
          and metrics_sender
      ):
        _send_step_deviation_metric_to_gcp(
            metrics_sender, step_dev, self._worker_config
        )
    except Exception as e:  # pylint: disable=broad-except
      logger.error(
          'Error while performing final step deviation query and upload for'
          ' job: %s and logger: %s. This will not impact the workload. Error:'
          ' %s',
          self._worker_config['job_name'],
          self._worker_config['logger_name'],
          e,
      )

  def start_rolling_window_goodput_uploader(
      self, rolling_windows_seconds: list[int]
  ):
    """Starts the rolling window goodput uploader process."""
    if not self._initialized:
      return
    if self._rolling_window_process and self._rolling_window_process.is_alive():
      logger.warning(
          'Rolling window uploader process (PID: %s) is already running.',
          self._rolling_window_process.pid,
      )
      return
    self._worker_config['rolling_windows'] = rolling_windows_seconds
    self._rolling_window_termination_event.clear()
    self._rolling_window_final_flush_event.clear()
    self._rolling_window_process = multiprocessing.Process(
        target=_rolling_window_worker,
        args=(
            self._worker_config,
            self._rolling_window_termination_event,
            self._rolling_window_final_flush_event,
        ),
        daemon=True,
    )
    self._rolling_window_process.start()
    logger.info(
        'Rolling window goodput uploader process started for job: %s (PID: %s)',
        self._worker_config['job_name'],
        self._rolling_window_process.pid,
    )

  def stop_rolling_window_goodput_uploader(
      self, skip_final_flush: bool | None = None
  ):
    """Stops the rolling window goodput uploader process and performs a final upload."""
    if not self._initialized:
      return

    should_skip = (
        skip_final_flush
        if skip_final_flush is not None
        else getattr(self, '_skip_final_flush', False)
    )
    worker_was_running = self._rolling_window_process is not None

    if self._rolling_window_process:
      pid = self._rolling_window_process.pid
      logger.info('Shutting down rolling window process (PID: %s)', pid)

      if not should_skip:
        self._rolling_window_final_flush_event.set()
      self._rolling_window_termination_event.set()
      self._rolling_window_process.join(
          timeout=self._worker_join_timeout(should_skip)
      )

      if self._rolling_window_process.is_alive():
        logger.warning(
            'Rolling window process (PID: %s) did not exit gracefully.'
            ' Terminating forcefully.',
            pid,
        )
        self._rolling_window_process.terminate()
        self._rolling_window_process.join()

      exit_code = self._rolling_window_process.exitcode
      if exit_code:
        logger.warning(
            'Rolling window process (PID: %s) terminated abnormally. Exit'
            ' Code: %s',
            pid,
            exit_code,
        )

    self._rolling_window_process = None
    if not should_skip and not worker_was_running:
      _run_with_timeout(
          self._final_rolling_window_goodput_query_and_upload,
          getattr(self, '_final_flush_timeout_seconds', None),
          f'Final rolling window goodput query and upload for job: {self._worker_config["job_name"]}',
      )

  def _final_rolling_window_goodput_query_and_upload(self):
    """Performs a finalrolling window goodput query and upload."""
    try:
      calculator = self._goodput_calculator
      metrics_sender = self._metrics_sender

      if (
          not self._worker_config['gcp_options'].enable_gcp_goodput_metrics
          or not metrics_sender
      ):
        return

      now = datetime.datetime.now(datetime.timezone.utc)
      for window_size in self._worker_config['rolling_windows']:
        window_end = now
        window_start = now - datetime.timedelta(seconds=window_size)
        window_start = window_start.replace(tzinfo=datetime.timezone.utc)
        rolling_window_metric_details = calculator.get_interval_metric_details(
            window_start, window_end
        )
        _upload_interval_goodput_metrics_to_gcm(
            metrics_sender, rolling_window_metric_details, self._worker_config
        )
      logger.info(
          'Final rolling window goodput query and upload for job: %s and'
          ' logger: %s completed',
          self._worker_config['job_name'],
          self._worker_config['logger_name'],
      )
    except Exception as e:  # pylint: disable=broad-except
      logger.error(
          'Error while performing final rolling window goodput query and upload'
          ' for job: %s and logger: %s. This will not impact the workload.'
          ' Error: %s',
          self._worker_config['job_name'],
          self._worker_config['logger_name'],
          e,
      )
