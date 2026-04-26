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

from collections import defaultdict
from dataclasses import dataclass, field

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


@dataclass
class ParsedLogs:
  """Events extracted from the workload's goodput log entries."""

  job_starts: list[float]
  job_ends: list[float]
  tpu_init: list[tuple[float, float]]
  training_prep: list[tuple[float, float]]
  data_loading: list[tuple[float, float]]
  steps: dict[int, float]
  custom: dict[str, list[tuple[float, float]]] = field(default_factory=dict)


def parse_entries(entries: list[dict]) -> ParsedLogs:
  """Route each log entry to its event bucket by payload key."""
  job_starts, job_ends = [], []
  tpu_starts, tpu_ends = [], []
  prep_starts, prep_ends = [], []
  dl_starts, dl_ends = [], []
  steps: dict[int, float] = {}
  custom_starts: dict[str, list[float]] = defaultdict(list)
  custom_ends: dict[str, list[float]] = defaultdict(list)

  for e in entries:
    if JOB_START in e:
      job_starts.append(float(e[JOB_START]))
    elif JOB_END in e:
      job_ends.append(float(e[JOB_END]))
    elif TPU_INIT_START in e:
      tpu_starts.append(float(e[TPU_INIT_START]))
    elif TPU_INIT_END in e:
      tpu_ends.append(float(e[TPU_INIT_END]))
    elif TRAINING_PREP_START in e:
      prep_starts.append(float(e[TRAINING_PREP_START]))
    elif TRAINING_PREP_END in e:
      prep_ends.append(float(e[TRAINING_PREP_END]))
    elif DATA_LOADING_START in e:
      dl_starts.append(float(e[DATA_LOADING_START]))
    elif DATA_LOADING_END in e:
      dl_ends.append(float(e[DATA_LOADING_END]))
    elif STEP_START in e:
      steps[int(e[STEP_COUNT])] = float(e[STEP_START])
    elif CUSTOM_START in e:
      custom_starts[e.get(CUSTOM_TYPE, 'unknown')].append(
          float(e[CUSTOM_START])
      )
    elif CUSTOM_END in e:
      custom_ends[e.get(CUSTOM_TYPE, 'unknown')].append(float(e[CUSTOM_END]))

  def _pair(starts: list[float], ends: list[float]) -> list[tuple[float, float]]:
    return list(zip(sorted(starts), sorted(ends)))

  return ParsedLogs(
      job_starts=sorted(job_starts),
      job_ends=sorted(job_ends),
      tpu_init=_pair(tpu_starts, tpu_ends),
      training_prep=_pair(prep_starts, prep_ends),
      data_loading=_pair(dl_starts, dl_ends),
      steps=steps,
      custom={
          k: _pair(v, custom_ends.get(k, [])) for k, v in custom_starts.items()
      },
  )
