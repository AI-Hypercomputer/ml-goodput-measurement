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

"""Event types and segment styles of the goodput timeline."""

from dataclasses import dataclass
from enum import Enum


class EventType(str, Enum):

  # Stepping container: the span of a run's productive steps, which holds
  # PRODUCTIVE_STEPS and the JIT_COMPILE overhead of its first steps, the way
  # WASTED_PROGRESS holds LOST_PROGRESS and JIT_COMPILE.
  STEP_BLOCK = 'step_block'
  PRODUCTIVE_STEPS = 'productive_steps'
  WASTED_PROGRESS = 'wasted_progress'
  STARTUP = 'startup'
  TPU_INIT = 'tpu_init'
  TRAINING_PREP = 'training_prep'
  DATA_LOADING = 'data_loading'
  INTERRUPTION = 'interruption'
  CUSTOM = 'custom'
  CHECKPOINT_SAVE = 'checkpoint_save'
  CHECKPOINT_RESTORE = 'checkpoint_restore'
  SNAPSHOT_SAVE = 'snapshot_save'
  ELASTIC_SCALE_UP = 'elastic_scale_up'
  ELASTIC_SLICE_DOWN = 'elastic_slice_down'
  ELASTIC_REINIT = 'elastic_reinit'
  AOT_COMPILE = 'aot_compile'
  JIT_COMPILE = 'jit_compile'
  LOST_PROGRESS = 'lost_progress'


@dataclass(frozen=True)
class SegmentStyle:
  color: str
  row_label: str
  draw_order: int


EVENT_STYLE_MAP: dict[EventType, SegmentStyle] = {
    EventType.STEP_BLOCK: SegmentStyle(
        '#a7f3d0', 'Stepping (Productive Steps + JIT compile overhead)', 0
    ),
    EventType.PRODUCTIVE_STEPS: SegmentStyle(
        '#10b981', 'Productive Steps (excl. JIT compile)', 1
    ),
    EventType.WASTED_PROGRESS: SegmentStyle(
        '#fde68a',
        'Wasted Progress (Lost + JIT compile during wasted progress)',
        2,
    ),
    EventType.LOST_PROGRESS: SegmentStyle(
        '#f59e0b', 'Lost Progress (Repeated Steps, excl. JIT compile)', 3
    ),
    EventType.STARTUP: SegmentStyle('#cbd5e1', 'Startup / Recovery', 4),
    EventType.TPU_INIT: SegmentStyle('#f97316', 'Device Init (TPU)', 5),
    EventType.TRAINING_PREP: SegmentStyle('#6366f1', 'Training Prep', 6),
    EventType.DATA_LOADING: SegmentStyle('#ec4899', 'Data Loading', 7),
    EventType.INTERRUPTION: SegmentStyle('#ef4444', 'Interruption', 8),
    EventType.CUSTOM: SegmentStyle('#a855f7', 'Custom Badput', 9),
    EventType.CHECKPOINT_SAVE: SegmentStyle('#1e293b', 'Checkpoint Save', 10),
    EventType.CHECKPOINT_RESTORE: SegmentStyle(
        '#14b8a6', 'Checkpoint / Snapshot Restore', 11
    ),
    EventType.SNAPSHOT_SAVE: SegmentStyle('#84cc16', 'Snapshot Save', 12),
    EventType.ELASTIC_SCALE_UP: SegmentStyle('#3b82f6', 'Elastic Scale Up', 13),
    EventType.ELASTIC_SLICE_DOWN: SegmentStyle(
        '#dc2626', 'Elastic Slice Down', 14
    ),
    EventType.ELASTIC_REINIT: SegmentStyle(
        '#06b6d4', 'Elastic Reinitialization', 15
    ),
    EventType.AOT_COMPILE: SegmentStyle('#8b5cf6', 'AOT Compilation', 16),
    EventType.JIT_COMPILE: SegmentStyle('#d946ef', 'JIT Compile Overhead', 17),
}


ROW_ORDER: list[str] = [
    s.row_label
    for s in sorted(EVENT_STYLE_MAP.values(), key=lambda s: s.draw_order)
]
