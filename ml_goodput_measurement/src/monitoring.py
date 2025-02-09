"""Goodput monitoring API.

This file contains all the utilities to monitor and upload goodput data of a
user workload to Tensorboard asynchronously.
"""

import logging
import os
import threading
import time

from cloud_goodput.ml_goodput_measurement.src.goodput import GoodputCalculator
from cloud_goodput.ml_goodput_measurement.src.goodput_utils import BadputType
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
  ):
    """Initializes the GoodputMonitor.

    Args:
        job_name: The name of the job to monitor.
        logger_name: The name of the Google Cloud Logging logger to use.
        tensorboard_dir: The directory to write TensorBoard data to.
        upload_interval: The interval to upload data to TensorBoard.
        monitoring_enabled: Whether to enable monitoring. If the application is
          interested in monitoring Goodput, it should set this value to True if
          monitoring from TPU worker 0 andthe application's configurations
          request Goodput monitoring.
        pathway_enabled: Whether the application is using Pathways.
        include_badput_breakdown: Whether to query and upload badput breakdown
          data to Tensorboard.
        include_step_deviation: Whether to query and upload step deviation data
          to Tensorboard.
        step_deviation_interval_seconds: The interval to query step deviation
          data.
        configured_ideal_step_time: The optionalideal step time configured by the user.
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
    self._uploader_thread_running = True

  def stop_goodput_uploader(self):
    """Stops the goodput uploader thread."""
    if not self._uploader_thread_running:
      raise RuntimeError('Goodput uploader thread is not running.')

    self._termination_event.set()
    if self._goodput_upload_thread is not None:
      logger.info('Waiting for goodput query and uploader thread to complete.')
      self._goodput_upload_thread.join()
    logger.info(
        'Goodput query and uploader thread stopped. No more goodput data will'
        ' be uploaded to Tensorboard.'
    )
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
