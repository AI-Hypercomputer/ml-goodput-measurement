"""""Goodput Elasticimplementations."""

import datetime
from typing import Optional

from cloud_goodput.ml_goodput_measurement.src import goodput

_JOB_NAME = 'job_name'
_ACTIVE_SLICES = 'active_slices'
_TOTAL_SLICES = 'total_slices'
_AVAILABLE_SLICES = 'available_slices'
_ELASTIC_SLICE_COUNTS_TIMESTAMP = 'elastic_slice_counts_timestamp'

GoodputRecorder = goodput.GoodputRecorder


class ElasticGoodputRecorder(GoodputRecorder):
  """The Goodput recorder (child class) for elastic."""

  def record_elastic_slice_counts(
      self,
      active_slices: int,
      total_slices: int,
      available_slices: int,
      timestamp: Optional[datetime.datetime] = None,
  ) -> None:
    if self._cloud_logger is None:
      return
    if timestamp is None:
      timestamp = datetime.datetime.now(datetime.timezone.utc)
    self._cloud_logger.write_cloud_logging_entry({
        _JOB_NAME: self.job_name,
        _ACTIVE_SLICES: int(active_slices),
        _TOTAL_SLICES: int(total_slices),
        _AVAILABLE_SLICES: int(available_slices),
        _ELASTIC_SLICE_COUNTS_TIMESTAMP: timestamp.timestamp(),
    })
