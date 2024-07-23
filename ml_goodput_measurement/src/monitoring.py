"""Goodput monitoring API.

This file contains all the utilities to monitor and upload goodput data of a
user workload to Tensorboard asynchronously.
"""

import datetime
import logging
import threading
import time

from cloud_tpu_goodput.ml_goodput_measurement.src.goodput import GoodputCalculator
from tensorboardX import writer

_TENSORBOARD_METRIC_LABEL = 'goodput'

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
    """
    if not monitoring_enabled:
      logger.info(
          'Monitoring is disabled. Returning without initializing'
          ' GoodputMonitor.'
      )
      return

    self._job_name = job_name
    self._logger_name = logger_name
    self._tensorboard_dir = tensorboard_dir
    self._upload_interval = upload_interval
    self._goodput_calculator = GoodputCalculator(
        job_name=self._job_name,
        logger_name=self._logger_name,
    )
    self._writer = writer.SummaryWriter(self._tensorboard_dir)

    # Flag to signal the daemon thread if it exists when to initate
    # shutdown and wait for termination.
    self._uploader_thread_running = False
    self._goodput_upload_thread = None
    self._termination_event = threading.Event()
    self._termination_event.clear()

  def __del__(self):
    if self._uploader_thread_running:
      self.stop_goodput_uploader()

  def _write_to_tensorboard(self, job_goodput: float, last_step: int):
    if self._writer is not None:
      self._writer.add_scalar(
          _TENSORBOARD_METRIC_LABEL, job_goodput, last_step
      )
      self._writer.flush()

  def _query_and_upload_goodput(self):
    """Queries and uploads goodput data to Tensorboard."""
    while not self._termination_event.is_set():
      time.sleep(self._upload_interval)
      try:
        job_goodput, _, last_step = self._goodput_calculator.get_job_goodput()
        self._write_to_tensorboard(job_goodput, last_step)
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
