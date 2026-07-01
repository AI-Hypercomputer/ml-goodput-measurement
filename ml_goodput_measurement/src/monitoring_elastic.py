"""Elastic Goodput monitoring API."""

from cloud_goodput.ml_goodput_measurement.src import goodput_elastic
from cloud_goodput.ml_goodput_measurement.src import goodput_utils
from cloud_goodput.ml_goodput_measurement.src import monitoring

GCPOptions = goodput_utils.GCPOptions


class ElasticGoodputMonitor(monitoring.GoodputMonitor):
  """GoodputMonitor for elastic training jobs.

  Uses ElasticGoodputCalculator to produce elastic badput breakdown
  (ELASTIC_SLICE_DOWN, ELASTIC_SCALE_UP, ELASTIC_REINITIALIZATION) and
  optionally slice efficiency metrics.
  """

  def __init__(
      self,
      job_name: str,
      logger_name: str,
      tensorboard_dir: str,
      upload_interval: int,
      monitoring_enabled: bool = False,
      include_badput_breakdown: bool = False,
      include_step_deviation: bool = False,
      include_slice_efficiency: bool = False,
      configured_ideal_step_time=None,
      step_deviation_interval_seconds: int = 10,
      gcp_options: GCPOptions = GCPOptions(),
      skip_final_flush: bool = False,
      final_flush_timeout_seconds: (
          float | None
      ) = monitoring._PROCESS_TERMINATION_TIMEOUT_SECONDS,
  ):
    super().__init__(
        job_name=job_name,
        logger_name=logger_name,
        tensorboard_dir=tensorboard_dir,
        upload_interval=upload_interval,
        monitoring_enabled=monitoring_enabled,
        pathway_enabled=True,  # elastic always uses Pathways
        include_badput_breakdown=include_badput_breakdown,
        include_step_deviation=include_step_deviation,
        configured_ideal_step_time=configured_ideal_step_time,
        step_deviation_interval_seconds=step_deviation_interval_seconds,
        gcp_options=gcp_options,
        skip_final_flush=skip_final_flush,
        final_flush_timeout_seconds=final_flush_timeout_seconds,
    )
    if self._initialized:
      # Replace base calculator with elastic-aware one.
      self._goodput_calculator = goodput_elastic.ElasticGoodputCalculator(
          job_name=job_name,
          logger_name=logger_name,
          using_pathways=True,
      )
      self._worker_config['calculator_class'] = (
          goodput_elastic.ElasticGoodputCalculator
      )
      self._worker_config['include_slice_efficiency'] = include_slice_efficiency
