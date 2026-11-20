# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Parses goodput log entries into the events of the timeline."""

from collections import defaultdict
from dataclasses import dataclass, field
import math
from typing import Any

JOB_START = 'job_start_time'
JOB_END = 'job_end_time'
STEP_COUNT = 'step_count'
STEP_START = 'step_start_time'
TPU_INIT_START = 'tpu_init_start_time'
TPU_INIT_END = 'tpu_init_end_time'
TRAINING_PREP_START = 'training_prep_start_time'
TRAINING_PREP_END = 'training_prep_end_time'
DATA_LOADING_START = 'data_loading_start_time'
DATA_LOADING_END = 'data_loading_end_time'
CUSTOM_TYPE = 'custom_badput_event_type'
CUSTOM_START = 'custom_badput_event_start_time'
CUSTOM_END = 'custom_badput_event_end_time'

# Checkpoint keys
CKPT_RESTORE_START = 'checkpoint_manager_start_time'
CKPT_RESTORE_DUR = 'checkpoint_manager_duration_secs'
CKPT_SAVE_START = 'checkpoint_manager_blocking_start_time'
CKPT_SAVE_DUR = 'checkpoint_manager_blocking_duration_secs'
EXPLICIT_CKPT_RESTORE_START = 'checkpoint_restore_start_time'
EXPLICIT_CKPT_RESTORE_END = 'checkpoint_restore_end_time'

# Explicit sub-phase & disruption keys
JOB_DISRUPTION = 'job_disruption_time'
DATA_ITERATOR_INIT_START = 'data_iterator_init_start_time'
DATA_ITERATOR_INIT_END = 'data_iterator_init_end_time'
COMPILATION_START = 'compilation_start_time'
COMPILATION_END = 'compilation_end_time'

# Custom badput types (case-insensitive) parsed like the explicit
# checkpoint_restore_* and data_iterator_init_* entries, e.g. from AXLearn.
# A blocking snapshot save counts as phase 6 like the snapshot restore.
SNAPSHOT_RESTORE_CUSTOM_TYPE = 'SNAPSHOT_RESTORE'
SNAPSHOT_SAVE_CUSTOM_TYPE = 'SNAPSHOT_SAVE'
DATA_ITERATOR_INIT_CUSTOM_TYPE = 'DATA_ITERATOR_INIT'

# Elastic keys
ELASTIC_WAIT_TYPE = 'elastic_wait_event_type'
ELASTIC_WAIT_START = 'elastic_wait_start_time'
ELASTIC_WAIT_END = 'elastic_wait_end_time'
ELASTIC_REINIT_START = 'elastic_reinit_start_time'
ELASTIC_REINIT_END = 'elastic_reinit_end_time'
ELASTIC_SLICE_COUNTS_TS = 'elastic_slice_counts_timestamp'
ACTIVE_SLICES = 'active_slices'
AVAILABLE_SLICES = 'available_slices'
TOTAL_SLICES = 'total_slices'


@dataclass
class ParsedLogs:
  """Events extracted from the workload's goodput log entries."""

  job_starts: list[float]
  job_ends: list[float]
  tpu_init: list[tuple[float, float]]
  training_prep: list[tuple[float, float]]
  data_loading: list[tuple[float, float]]
  steps: list[tuple[int, float]]
  checkpoint_save: list[tuple[float, float]]
  checkpoint_restore: list[tuple[float, float]]
  elastic_scale_up: list[tuple[float, float]]
  elastic_slice_down: list[tuple[float, float]]
  elastic_reinitialization: list[tuple[float, float]]
  elastic_slice_counts: list[tuple[float, int, int, int]] = field(
      default_factory=list
  )
  job_disruptions: list[float] = field(default_factory=list)
  data_iterator_init: list[tuple[float, float]] = field(default_factory=list)
  compilation: list[tuple[float, float]] = field(default_factory=list)
  # Subsets of checkpoint_restore from SNAPSHOT_RESTORE / SNAPSHOT_SAVE events,
  # labeled apart.
  snapshot_restore: list[tuple[float, float]] = field(default_factory=list)
  snapshot_save: list[tuple[float, float]] = field(default_factory=list)
  custom: dict[str, list[tuple[float, float]]] = field(default_factory=dict)


def safe_float(val: Any) -> float | None:
  try:
    f = float(val)
    if math.isnan(f):
      return None
    return f
  except (ValueError, TypeError):
    return None


def parse_entries(entries: list[dict]) -> ParsedLogs:
  """Route each log entry to its event bucket statefully and chronologically."""
  job_starts, job_ends = [], []
  job_disruptions: list[float] = []
  tpu_init: list[tuple[float, float]] = []
  training_prep: list[tuple[float, float]] = []
  data_loading: list[tuple[float, float]] = []
  data_iterator_init: list[tuple[float, float]] = []
  compilation: list[tuple[float, float]] = []
  steps: list[tuple[int, float]] = []
  checkpoint_save: list[tuple[float, float]] = []
  checkpoint_restore: list[tuple[float, float]] = []
  snapshot_restore: list[tuple[float, float]] = []
  snapshot_save: list[tuple[float, float]] = []
  elastic_scale_up: list[tuple[float, float]] = []
  elastic_slice_down: list[tuple[float, float]] = []
  elastic_reinitialization: list[tuple[float, float]] = []
  elastic_slice_counts: list[tuple[float, int, int, int]] = []
  custom: dict[str, list[tuple[float, float]]] = defaultdict(list)

  active_tpu_start = None
  active_prep_start = None
  active_dl_start = None
  active_data_iter_start = None
  active_ckpt_restore_start = None
  active_compile_start = None
  active_custom_starts = {}
  active_elastic_wait_starts = {}
  active_elastic_reinit_start = None

  # Assumes entries are sorted chronologically by timestamp (default from
  # Logging API)
  for e in entries:
    if JOB_START in e:
      val = safe_float(e[JOB_START])
      if val is not None:
        job_starts.append(val)
    elif JOB_END in e:
      val = safe_float(e[JOB_END])
      if val is not None:
        job_ends.append(val)
    elif JOB_DISRUPTION in e:
      val = safe_float(e[JOB_DISRUPTION])
      if val is not None:
        job_disruptions.append(val)

    elif TPU_INIT_START in e:
      active_tpu_start = safe_float(e[TPU_INIT_START])
    elif TPU_INIT_END in e:
      val = safe_float(e[TPU_INIT_END])
      if val is not None and active_tpu_start is not None:
        if active_tpu_start <= val:
          tpu_init.append((active_tpu_start, val))
        active_tpu_start = None

    elif TRAINING_PREP_START in e:
      active_prep_start = safe_float(e[TRAINING_PREP_START])
    elif TRAINING_PREP_END in e:
      val = safe_float(e[TRAINING_PREP_END])
      if val is not None and active_prep_start is not None:
        if active_prep_start <= val:
          training_prep.append((active_prep_start, val))
        active_prep_start = None

    elif DATA_ITERATOR_INIT_START in e:
      active_data_iter_start = safe_float(e[DATA_ITERATOR_INIT_START])
    elif DATA_ITERATOR_INIT_END in e:
      val = safe_float(e[DATA_ITERATOR_INIT_END])
      if val is not None and active_data_iter_start is not None:
        if active_data_iter_start <= val:
          data_iterator_init.append((active_data_iter_start, val))
        active_data_iter_start = None

    elif EXPLICIT_CKPT_RESTORE_START in e:
      active_ckpt_restore_start = safe_float(e[EXPLICIT_CKPT_RESTORE_START])
    elif EXPLICIT_CKPT_RESTORE_END in e:
      val = safe_float(e[EXPLICIT_CKPT_RESTORE_END])
      if val is not None and active_ckpt_restore_start is not None:
        if active_ckpt_restore_start <= val:
          checkpoint_restore.append((active_ckpt_restore_start, val))
        active_ckpt_restore_start = None

    elif COMPILATION_START in e:
      active_compile_start = safe_float(e[COMPILATION_START])
    elif COMPILATION_END in e:
      val = safe_float(e[COMPILATION_END])
      if val is not None and active_compile_start is not None:
        if active_compile_start <= val:
          compilation.append((active_compile_start, val))
        active_compile_start = None

    elif DATA_LOADING_START in e:
      active_dl_start = safe_float(e[DATA_LOADING_START])
    elif DATA_LOADING_END in e:
      val = safe_float(e[DATA_LOADING_END])
      if val is not None and active_dl_start is not None:
        if active_dl_start <= val:
          data_loading.append((active_dl_start, val))
        active_dl_start = None

    elif STEP_START in e:
      val = safe_float(e[STEP_START])
      try:
        raw_step = e.get(STEP_COUNT, e.get('current_step'))
        if raw_step is not None and val is not None:
          step_cnt = int(raw_step)
          steps.append((step_cnt, val))
      except (ValueError, TypeError, KeyError):
        pass

    elif CKPT_RESTORE_START in e and CKPT_RESTORE_DUR in e:
      st = safe_float(e[CKPT_RESTORE_START])
      dur = safe_float(e[CKPT_RESTORE_DUR])
      if st is not None and dur is not None:
        if dur >= 0:
          checkpoint_restore.append((st, st + dur))

    elif CKPT_SAVE_START in e and CKPT_SAVE_DUR in e:
      st = safe_float(e[CKPT_SAVE_START])
      dur = safe_float(e[CKPT_SAVE_DUR])
      if st is not None and dur is not None:
        if dur >= 0:
          checkpoint_save.append((st, st + dur))

    elif ELASTIC_WAIT_START in e:
      wtype = e.get(ELASTIC_WAIT_TYPE, 'unknown')
      active_elastic_wait_starts[wtype] = safe_float(e[ELASTIC_WAIT_START])
    elif ELASTIC_WAIT_END in e:
      wtype = e.get(ELASTIC_WAIT_TYPE, 'unknown')
      val = safe_float(e[ELASTIC_WAIT_END])
      if val is not None and active_elastic_wait_starts.get(wtype) is not None:
        if active_elastic_wait_starts[wtype] <= val:
          interval = (active_elastic_wait_starts[wtype], val)
          if wtype == 'elastic_scale_up':
            elastic_scale_up.append(interval)
          elif wtype == 'elastic_slice_down':
            elastic_slice_down.append(interval)
          else:
            custom[wtype].append(interval)
        del active_elastic_wait_starts[wtype]

    elif ELASTIC_REINIT_START in e:
      active_elastic_reinit_start = safe_float(e[ELASTIC_REINIT_START])
    elif ELASTIC_REINIT_END in e:
      val = safe_float(e[ELASTIC_REINIT_END])
      if val is not None and active_elastic_reinit_start is not None:
        if active_elastic_reinit_start <= val:
          elastic_reinitialization.append((active_elastic_reinit_start, val))
        active_elastic_reinit_start = None

    elif ELASTIC_SLICE_COUNTS_TS in e:
      ts = safe_float(e[ELASTIC_SLICE_COUNTS_TS])
      active = int(e.get(ACTIVE_SLICES, 0))
      avail = int(e.get(AVAILABLE_SLICES, 0))
      total = int(e.get(TOTAL_SLICES, 1))
      if ts is not None:
        elastic_slice_counts.append((ts, active, avail, total))

    elif CUSTOM_START in e:
      ctype = e.get(CUSTOM_TYPE, 'unknown')
      active_custom_starts[ctype] = safe_float(e[CUSTOM_START])
    elif CUSTOM_END in e:
      ctype = e.get(CUSTOM_TYPE, 'unknown')
      val = safe_float(e[CUSTOM_END])
      if val is not None and active_custom_starts.get(ctype) is not None:
        if active_custom_starts[ctype] <= val:
          interval = (active_custom_starts[ctype], val)
          if ctype == 'elastic_scale_up':
            elastic_scale_up.append(interval)
          elif ctype == 'elastic_slice_down':
            elastic_slice_down.append(interval)
          elif ctype == 'elastic_reinitialization':
            elastic_reinitialization.append(interval)
          elif str(ctype).upper() == SNAPSHOT_RESTORE_CUSTOM_TYPE:
            checkpoint_restore.append(interval)
            snapshot_restore.append(interval)
          elif str(ctype).upper() == SNAPSHOT_SAVE_CUSTOM_TYPE:
            checkpoint_restore.append(interval)
            snapshot_save.append(interval)
          elif str(ctype).upper() == DATA_ITERATOR_INIT_CUSTOM_TYPE:
            data_iterator_init.append(interval)
          else:
            custom[ctype].append(interval)
        del active_custom_starts[ctype]

  # Ensure final structures are sorted
  tpu_init.sort()
  training_prep.sort()
  data_loading.sort()
  data_iterator_init.sort()
  compilation.sort()
  checkpoint_save.sort()
  checkpoint_restore.sort()
  snapshot_restore.sort()
  snapshot_save.sort()
  elastic_scale_up.sort()
  elastic_slice_down.sort()
  elastic_reinitialization.sort()
  elastic_slice_counts.sort(key=lambda x: x[0])
  for k in custom:
    custom[k].sort()

  return ParsedLogs(
      job_starts=sorted(job_starts),
      job_ends=sorted(job_ends),
      job_disruptions=sorted(job_disruptions),
      tpu_init=tpu_init,
      training_prep=training_prep,
      data_loading=data_loading,
      data_iterator_init=data_iterator_init,
      compilation=compilation,
      steps=sorted(steps, key=lambda x: x[1]),
      checkpoint_save=checkpoint_save,
      checkpoint_restore=checkpoint_restore,
      snapshot_restore=snapshot_restore,
      snapshot_save=snapshot_save,
      elastic_scale_up=elastic_scale_up,
      elastic_slice_down=elastic_slice_down,
      elastic_reinitialization=elastic_reinitialization,
      elastic_slice_counts=elastic_slice_counts,
      custom=dict(custom),
  )
