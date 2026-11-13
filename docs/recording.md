# Recording intervals

`GoodputRecorder` writes the timestamps that define a job's intervals to
Cloud Logging from the training process. This page covers the logger
name, the recorder object and each record call.

## Import

 To use this package, import the `goodput` module:

 ```python
 from ml_goodput_measurement import goodput
 from ml_goodput_measurement import monitoring
 ```

## Define the name of the Google Cloud Logging logger

 Create a run-specific logger name where Cloud Logging entries can be written to
 and read from.

 > **IMPORTANT:** Please use a unique `run_name` for each individual experiment
 or workload that you intend to monitor separately. If you unintentionally reuse
 `run_name` or `goodput_logger_name` in the same storage bucket of a GCP
 project, your cumulative Goodput metrics may be inaccurately taking previous
 runs into account.

 For example:

 ```python
 goodput_logger_name = f'goodput_{config.run_name}' # Here run_name is unique.
 ```

## Create a `GoodputRecorder` object

 Next, create a recorder object with the following parameters:

 1. `job_name`: The full run name of the job.
 2. `logger_name`: The name of the Cloud Logging logger object (created in the
    previous step).
 3. `logging_enabled`: Whether or not this process has Cloud Logging enabled.

  > **_NOTE:_** For a multi-worker setup, please ensure that only one worker
   writes the logs to avoid the duplication. In JAX the check could be
   `jax.process_index() == 0`; in PyTorch `torch.distributed.get_rank() == 0`.

  > **_NOTE:_** `logging_enabled` defaults to `False` and Goodput computations
  cannot be completed if no logs are ever written.

 For example:

 ```python
 goodput_recorder = goodput.GoodputRecorder(
  job_name=config.run_name,
  logger_name=goodput_logger_name,
  logging_enabled=(jax.process_index() == 0)
  )
 ```

  > **_NOTE:_** JAX initialization should be complete before this call.

## Record Data with `GoodputRecorder`

### Record Job Start and End Time

 Use the recorder object to record the job's overall start and end time.

 For example:

 ```python
 def main(argv: Sequence[str]) -> None:
 # Initialize configs…
 goodput_recorder.record_job_start_time(datetime.datetime.now())
 # Device Initialization and device scanning…
 # Set up other things for the main training loop…
 # Main training loop
 train_loop(config)
 goodput_recorder.record_job_end_time(datetime.datetime.now())
 ```

### Record Step Time

 Use the recorder object to record a step's start time using
 `record_step_start_time(step_count)`:

For example:

 ```python
 def train_loop(config, state=None):
 # Set up mesh, model, state, checkpoint manager…

 # Initialize functional train arguments and model parameters…

 # Define the compilation

 for step in np.arange(start_step, config.steps):
   goodput_recorder.record_step_start_time(step)
   # Training step…

 return state
 ```

For asynchronous backends such as PyTorch on TPU or GPU, take the step start
from accelerator events instead; see
[Record Step Time from Accelerator Events](pytorch.md#record-step-time-from-accelerator-events-pytorch).

### Record Device Initialization, Training Preparation and Data Loading Time

  - Use the recorder object to record Device Initialization time using
      `record_tpu_init_start_time` and `record_tpu_init_end_time`.
  - Use the recorder object to record Training Preparation time using
      `record_training_preparation_start_time` and
      `record_training_preparation_end_time`.
  - Use the recorder object to record Data Loading time using
      `record_data_loading_start_time` and `record_data_loading_end_time`.

  For example:

  ```python
  def train_loop(config, state=None):
  goodput_recorder.record_tpu_init_start_time()
  # Set up mesh, model, state, checkpoint manager…
  goodput_recorder.record_tpu_init_end_time()
  goodput_recorder.record_training_preparation_start_time()
  # Set up training set, initialize functional train args and model parameters…
  # Define the compilation
  # Set up any metrics collectors
  goodput_recorder.record_training_preparation_end_time()

  for step in np.arange(start_step, config.steps):
    goodput_recorder.record_data_loading_start_time()
    example_batch = load_next_batch(data_iterator, example_batch, config)
    goodput_recorder.record_data_loading_end_time()
    goodput_recorder.record_step_start_time(step)
    # Training step…

  return state
  ```

### Record Custom Badput Events (e.g., Evaluation, SDC Checks)

- Use the recorder object to record the **start** of a custom badput event using
  `record_custom_badput_event_start_time(custom_badput_event_type='your_event_name')`.
- Use the recorder object to record the **end** of a custom badput event using
  `record_custom_badput_event_end_time(custom_badput_event_type='your_event_name')`.

Use these APIs when you want to account for time spent on operations that
block the training loop and use accelerator resources, do not contribute to
productive training and occur while training is in progress — such as step
evaluations, SDC checks, or re-compilations.

For example:

```python
def train_loop(config, state=None):
  goodput_recorder.record_training_preparation_start_time()
  # Initialize training config, setup model, load checkpoint...
  goodput_recorder.record_training_preparation_end_time()

  for step in range(config.steps):
    goodput_recorder.record_data_loading_start_time()
    batch = load_batch(train_data)
    goodput_recorder.record_data_loading_end_time()

    goodput_recorder.record_step_start_time(step)
    # Run training step...
    run_train_step(step, state)

    if step % config.eval_interval == 0:
      # Record a custom badput event for evaluation
      goodput_recorder.record_custom_badput_event_start_time(
          custom_badput_event_type="eval_step")
      run_step_evaluation(model, val_data)
      goodput_recorder.record_custom_badput_event_end_time(
          custom_badput_event_type="eval_step")

    if step % config.sdc_check_interval == 0:
      # Record a custom badput event for SDC check
      goodput_recorder.record_custom_badput_event_start_time(
          custom_badput_event_type="sdc_check")
      run_sdc_check(state)
      goodput_recorder.record_custom_badput_event_end_time(
          custom_badput_event_type="sdc_check")

  return state
```

> **_NOTE:_** The `custom_badput_event_type` string should be descriptive and
consistent (e.g., "eval_step", "sdc_check"), to ensure accurate aggregation and
reporting in badput breakdowns.

### Record Checkpoint Save and Restore Time (non-Orbax checkpointers)

Orbax writes checkpoint save and restore statistics to the goodput log itself.
If you checkpoint with another framework (for example PyTorch Distributed
Checkpoint), record them with:

- `record_checkpoint_save_stats(...)` once per saving step, with the time the
  training loop was blocked (and optionally its breakdown).
- `record_checkpoint_restore_stats(...)` once per restore.

They are reported as `UNPRODUCTIVE_CHECKPOINT_SAVE_TIME` and
`UNPRODUCTIVE_CHECKPOINT_RESTORE_TIME`. Timestamps are UTC POSIX seconds; write
at most one save entry per step, because entries for the same step are averaged.
Both methods raise `ValueError` on a non-positive duration, so skip the call
when nothing blocked.

```python
start = time.time()
checkpointer.save(step)  # blocks the training loop
goodput_recorder.record_checkpoint_save_stats(
    step=step,
    directory=checkpoint_dir,  # a gs:// path counts as a persistent save
    checkpoint_manager_blocking_start_time=start,
    checkpoint_manager_blocking_duration_secs=time.time() - start,
    synchronous=True,
)
```

See also: [Badput types](badput.md),
[PyTorch and asynchronous backends](pytorch.md),
[Troubleshooting](troubleshooting.md).
