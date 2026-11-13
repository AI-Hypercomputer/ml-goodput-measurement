<!--
 Copyright 2023 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->
# ML Goodput Measurement

![PyPI](https://img.shields.io/pypi/v/ml-goodput-measurement) ![Python](https://img.shields.io/pypi/pyversions/ml-goodput-measurement) ![License](https://img.shields.io/badge/license-Apache--2.0-blue)

Goodput is the share of a training workload's wall-clock time spent on
preserved, productive work: training steps that completed and were not lost to
a restart or rollback. Badput is the remainder, attributed to its cause:
accelerator initialization, program startup and compilation, data loading,
checkpoint save and restore, evaluation and other custom events, disruptions
and the progress they wasted. It is a workload-level metric: one number for the
whole job, across all hosts and restarts, computed from the job's own
timeline. Goodput says how much accelerator time became training progress;
the Badput breakdown says what to fix.

`ml-goodput-measurement` records the timestamps that define these intervals
from any Python training loop (JAX, PyTorch, or any other framework), stores
them in Google Cloud Logging, and computes Goodput, the Badput breakdown and
step-time deviation on demand or continuously, with export to TensorBoard and
Google Cloud Monitoring for dashboards and alerting.

## Install

```bash
pip install ml-goodput-measurement
```

Requires a Google Cloud project with the Cloud Logging API enabled; see
[Prerequisites](https://github.com/AI-Hypercomputer/ml-goodput-measurement/blob/main/docs/prerequisites.md).

## Quick start

```python
from ml_goodput_measurement import goodput

# Record from exactly one process per job (the lead process).
recorder = goodput.GoodputRecorder(
    job_name=run_name, logger_name=f"goodput_{run_name}", logging_enabled=is_lead
)
recorder.record_job_start_time()
for step in range(num_steps):
  recorder.record_step_start_time(step)
  train_step()
recorder.record_job_end_time()

# Compute from anywhere with access to the project's logs.
calculator = goodput.GoodputCalculator(
    job_name=run_name, logger_name=f"goodput_{run_name}"
)
goodput_pct, badput, last_step = calculator.get_job_goodput(
    include_badput_breakdown=True
)
```

A 100-step run with checkpoints every 25 steps reports, for example:

| | % of wall-clock |
|---|---|
| Goodput | 52.5 |
| PROGRAM_STARTUP | 20.0 |
| UNPRODUCTIVE_CHECKPOINT_SAVE_TIME | 11.7 |
| OTHER | 8.0 |
| TRAINING_PREP | 4.3 |
| DATA_LOADING_SYNC | 3.5 |

Recording the optional intervals (initialization, data loading, checkpointing,
custom events such as evaluation) attributes more of the Badput;
`GoodputMonitor` uploads the numbers while the job runs. For asynchronous
backends such as PyTorch on TPU, `StepClock` takes step boundaries from
accelerator events.

## How it works

- `GoodputRecorder` runs in the training process on the lead host and writes
  timestamped entries to Cloud Logging; nothing blocks the training loop.
- `GoodputCalculator` reads the job's entries from anywhere with access to the
  project, keeps a local cache, and computes Goodput, the Badput breakdown and
  step-time deviation.
- `GoodputMonitor` runs the calculator on a schedule and uploads the results to
  TensorBoard and Google Cloud Monitoring.

## Guides

- [Recording intervals](https://github.com/AI-Hypercomputer/ml-goodput-measurement/blob/main/docs/recording.md): job, steps, initialization, data loading, checkpointing, custom events
- [Badput types](https://github.com/AI-Hypercomputer/ml-goodput-measurement/blob/main/docs/badput.md): what each bucket means and how it is computed
- [Monitoring](https://github.com/AI-Hypercomputer/ml-goodput-measurement/blob/main/docs/monitoring.md): continuous upload, TensorBoard, Cloud Monitoring dashboards and alerts
- [PyTorch and asynchronous backends](https://github.com/AI-Hypercomputer/ml-goodput-measurement/blob/main/docs/pytorch.md): `StepClock`
- [Elastic training](https://github.com/AI-Hypercomputer/ml-goodput-measurement/blob/main/docs/elastic.md): Pathways, slice-down and scale-up
- [Troubleshooting](https://github.com/AI-Hypercomputer/ml-goodput-measurement/blob/main/docs/troubleshooting.md)

## Integrations

- [MaxText](https://github.com/AI-Hypercomputer/maxtext) (JAX): reference
  integration

<!-- NEWS START -->
## Releases

- 0.2.5: monitor no longer freezes after a restart on the same host;
  late-visible log entries are read; `StepClock`, bounded final flush,
  `min_step_time`
- 0.2.4: elastic restart and checkpoint calculation fixes; custom badput events
  in the last step are counted
- 0.2.3: GCS cache sync for faster cold-start reads
<!-- NEWS END -->

Full history in
[CHANGELOG.md](https://github.com/AI-Hypercomputer/ml-goodput-measurement/blob/main/CHANGELOG.md).

## Contributing and support

See [CONTRIBUTING.md](https://github.com/AI-Hypercomputer/ml-goodput-measurement/blob/main/CONTRIBUTING.md).
Questions and bug reports: [GitHub issues](https://github.com/AI-Hypercomputer/ml-goodput-measurement/issues).
Apache-2.0 licensed.
