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

"""Writes synthetic goodput logs and a compare config for example screenshots.

All names, times and durations are made up. Uses only the standard library;
run it with any python3. It writes the logs and compare_config.json next to
this file, replacing the checked-in ones.
"""

import datetime
import json
import os
import random

OUT = os.path.dirname(os.path.abspath(__file__))
T0 = datetime.datetime(2026, 1, 15, 9, 0, tzinfo=datetime.timezone.utc).timestamp()
STEP_S = 9.0
JIT_EXTRA = (46.0, 19.0, 8.0)
CKPT_DIR = 'gs://example-bucket/checkpoints'


class Log:
  """Builds a list of goodput log payloads in time order."""

  def __init__(self, seed):
    self.entries = []
    self.rng = random.Random(seed)
    self.t = T0

  def add(self, **kw):
    self.entries.append(kw)

  def interval(self, start_key, end_key, start, dur, **extra):
    self.add(**{start_key: start}, **extra)
    self.add(**{end_key: start + dur}, **extra)
    return start + dur

  def startup(self, cluster_s, tpu_s, data_iter_s, compile_s, first_batch_s,
              restore_s=0.0, restore_kind='checkpoint', restore_step=0):
    """Job start (or reinit) phases; returns the time the first step starts."""
    t = self.t + cluster_s
    t = self.interval('tpu_init_start_time', 'tpu_init_end_time', t, tpu_s)
    prep_start = t
    self.add(training_prep_start_time=prep_start)
    t += data_iter_s
    if restore_s:
      if restore_kind == 'snapshot':
        self.add(custom_badput_event_start_time=t,
                 custom_badput_event_type='SNAPSHOT_RESTORE')
        self.add(custom_badput_event_end_time=t + restore_s,
                 custom_badput_event_type='SNAPSHOT_RESTORE')
      else:
        self.add(event_type='restore', step=restore_step, directory=CKPT_DIR,
                 checkpoint_manager_start_time=t,
                 checkpoint_manager_duration_secs=restore_s,
                 checkpointer_start_time=t,
                 checkpointer_duration_secs=restore_s,
                 broadcast_duration_secs=None)
      t += restore_s
    t = self.interval('compilation_start_time', 'compilation_end_time', t,
                      compile_s)
    self.add(training_prep_end_time=t)
    if first_batch_s:
      t = self.interval('data_loading_start_time', 'data_loading_end_time', t,
                        first_batch_s)
    self.t = t
    return t

  def steps(self, first, last, ckpt_every=0, ckpt_block_s=0.0, evals=(),
            eval_s=0.0):
    """Steps first..last; returns the time the last step started."""
    t = self.t
    last_start = t
    for i, step in enumerate(range(first, last + 1)):
      self.add(step_count=step, step_start_time=round(t, 3))
      last_start = t
      dur = STEP_S * (1 + self.rng.uniform(-0.02, 0.02))
      if i < len(JIT_EXTRA):
        dur += JIT_EXTRA[i] * (1 + self.rng.uniform(-0.1, 0.1))
      if ckpt_every and step % ckpt_every == 0 and i >= len(JIT_EXTRA):
        s = round(t + 1.0, 3)
        self.add(event_type='save', step=step, directory=CKPT_DIR,
                 synchronous=True,
                 checkpoint_manager_blocking_start_time=s,
                 checkpoint_manager_blocking_duration_secs=ckpt_block_s,
                 wait_for_prev_start_time=None,
                 wait_for_prev_duration_secs=0.0,
                 get_old_steps_start_time=None,
                 get_old_steps_duration_secs=0.0,
                 checkpointer_blocking_start_time=s,
                 checkpointer_blocking_duration_secs=ckpt_block_s)
        dur += ckpt_block_s
      if step in evals:
        s = t + STEP_S
        self.add(custom_badput_event_start_time=s,
                 custom_badput_event_type='EVAL_STEP')
        self.add(custom_badput_event_end_time=s + eval_s,
                 custom_badput_event_type='EVAL_STEP')
        dur += eval_s
      t += dur
    self.t = t
    return last_start

  def slice_counts(self, t, active, total=4):
    self.add(elastic_slice_counts_timestamp=t, active_slices=active,
             available_slices=active, total_slices=total)


def full_restart():
  """Three runs, two full restarts, checkpoints every 250 steps."""
  log = Log(seed=1)
  log.add(job_start_time=T0)
  log.startup(cluster_s=20, tpu_s=120, data_iter_s=60, compile_s=150,
              first_batch_s=30)
  incidents = [
      # (last step, ckpt step, fault, slice wait, cluster, tpu, data iter,
      #  restore, compile, first batch)
      (560, 500, 70, 380, 55, 125, 20, 140, 110, 30),
      (1040, 1000, 95, 540, 45, 120, 25, 160, 100, 25),
  ]
  first = 0
  for (last, ckpt, fault, wait, cluster, tpu, dit, restore, comp,
       fb) in incidents:
    last_start = log.steps(first, last, ckpt_every=250, ckpt_block_s=24.0,
                           evals=(300,) if first == 0 else (), eval_s=150.0)
    t_crash = last_start + STEP_S
    log.add(job_disruption_time=t_crash + fault)
    log.t = t_crash + fault + wait
    log.add(job_start_time=log.t)
    log.startup(cluster_s=cluster, tpu_s=tpu, data_iter_s=dit,
                compile_s=comp, first_batch_s=fb, restore_s=restore,
                restore_step=ckpt)
    first = ckpt
  last_start = log.steps(first, 1500, ckpt_every=250, ckpt_block_s=24.0)
  log.add(job_end_time=last_start + STEP_S + 20)
  return log.entries


def elastic(seed, incidents, last_step, ckpt_every, restore_kind):
  """In-place elastic recoveries: wait for a replacement slice, reinit."""
  log = Log(seed=seed)
  log.add(job_start_time=T0)
  log.slice_counts(T0, 4)
  log.startup(cluster_s=20, tpu_s=120, data_iter_s=60, compile_s=150,
              first_batch_s=30)
  first = 0
  for (last, resume, fault, wait, tpu, dit, restore, comp, fb) in incidents:
    last_start = log.steps(first, last, ckpt_every=ckpt_every,
                           ckpt_block_s=20.0 if ckpt_every else 0.0)
    t_crash = last_start + STEP_S
    log.slice_counts(t_crash + fault, 3)
    log.interval('elastic_wait_start_time', 'elastic_wait_end_time',
                 t_crash + fault, wait,
                 elastic_wait_event_type='elastic_slice_down')
    reinit_start = t_crash + fault + wait
    log.add(elastic_reinit_start_time=reinit_start)
    log.t = reinit_start
    resume_t = log.startup(cluster_s=5, tpu_s=tpu, data_iter_s=dit,
                           compile_s=comp, first_batch_s=0.0,
                           restore_s=restore, restore_kind=restore_kind,
                           restore_step=resume)
    log.add(elastic_reinit_end_time=resume_t)
    log.t = log.interval('data_loading_start_time', 'data_loading_end_time',
                         resume_t, fb)
    log.slice_counts(log.t, 4)
    first = resume
  last_start = log.steps(first, last_step, ckpt_every=ckpt_every,
                         ckpt_block_s=20.0 if ckpt_every else 0.0)
  log.add(job_end_time=last_start + STEP_S + 20)
  return log.entries


def main():
  os.makedirs(OUT, exist_ok=True)
  workloads = {
      'full-restart-1': full_restart(),
      'elastic-gcs-1': elastic(
          seed=2,
          incidents=[
              # (last, resume, fault, wait, tpu, data iter, restore,
              #  compile, first batch)
              (450, 375, 40, 150, 70, 20, 150, 120, 25),
              (940, 875, 35, 120, 65, 25, 140, 110, 25),
              (1390, 1375, 45, 180, 75, 20, 160, 125, 30),
          ],
          last_step=1700, ckpt_every=125, restore_kind='checkpoint'),
      'elastic-snapshot-1': elastic(
          seed=3,
          incidents=[
              (520, 505, 30, 140, 60, 15, 25, 70, 15),
              (1010, 990, 25, 110, 55, 15, 20, 65, 15),
              (1480, 1460, 35, 160, 60, 15, 30, 75, 15),
          ],
          last_step=1800, ckpt_every=0, restore_kind='snapshot'),
  }
  for name, entries in workloads.items():
    with open(os.path.join(OUT, f'{name}.json'), 'w') as f:
      json.dump(entries, f)
    print(name, len(entries), 'entries')

  config = {
      'title': 'Recovery strategies (synthetic data)',
      'subtitle': 'Example workloads; all names and numbers are made up.',
      'groups': [{'name': 'example-model \u00b7 4x64',
                  'description': 'Synthetic data, 256 chips'}],
      'workloads': [
          {'workload_id': 'full-restart-1', 'label': 'Full restart',
           'model': 'example-model', 'shape': '4x64', 'baseline': True,
           'entries_path': 'full-restart-1.json'},
          {'workload_id': 'elastic-gcs-1',
           'label': 'Elastic, restore from checkpoint',
           'model': 'example-model', 'shape': '4x64',
           'entries_path': 'elastic-gcs-1.json'},
          {'workload_id': 'elastic-snapshot-1',
           'label': 'Elastic, in-memory snapshot',
           'model': 'example-model', 'shape': '4x64',
           'entries_path': 'elastic-snapshot-1.json'},
      ],
  }
  with open(os.path.join(OUT, 'compare_config.json'), 'w') as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
  print('config written')


if __name__ == '__main__':
  main()
