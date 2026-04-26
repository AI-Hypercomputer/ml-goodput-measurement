# Copyright 2024 Google LLC
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

from dataclasses import dataclass
from enum import Enum


class EventType(str, Enum):

  STEP_BLOCK = 'step_block'
  STARTUP = 'startup'
  TPU_INIT = 'tpu_init'
  TRAINING_PREP = 'training_prep'
  DATA_LOADING = 'data_loading'
  INTERRUPTION = 'interruption'
  CUSTOM = 'custom'


@dataclass(frozen=True)
class SegmentStyle:
  color: str
  row_label: str
  draw_order: int


EVENT_STYLE_MAP: dict[EventType, SegmentStyle] = {
    EventType.STEP_BLOCK:    SegmentStyle('#2ecc71', 'Productive Steps', 0),
    EventType.STARTUP:       SegmentStyle('#bdc3c7', 'Startup / Recovery', 1),
    EventType.TPU_INIT:      SegmentStyle('#f39c12', 'Device Init (TPU)', 2),
    EventType.TRAINING_PREP: SegmentStyle('#e67e22', 'Training Prep', 3),
    EventType.DATA_LOADING:  SegmentStyle('#d35400', 'Data Loading', 4),
    EventType.INTERRUPTION:  SegmentStyle('#e74c3c', 'Interruption', 5),
    EventType.CUSTOM:        SegmentStyle('#9b59b6', 'Custom Badput', 6),
}

ROW_ORDER: list[str] = [
    s.row_label
    for s in sorted(EVENT_STYLE_MAP.values(), key=lambda s: s.draw_order)
]
