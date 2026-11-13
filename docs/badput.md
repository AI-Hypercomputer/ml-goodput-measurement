# Badput types

`GoodputCalculator` reads a job's entries and computes Goodput and the
Badput breakdown. This page covers the calculator, each Badput type and
interval queries.

## Retrieve Goodput with `GoodputCalculator`

In order to retrieve the Goodput of a job run, all you need to do is instantiate
a `GoodputCalculator` object with the job's run name and the Cloud Logging
logger name used to record data for that job run. Then call the
`get_job_goodput` API to get the computed Goodput for the job run.

It is recommended to make the `get_job_goodput` calls for a job run from an
instance that runs elsewhere from your training machine.

### Create a `GoodputCalculator` object

Create the calculator object:

```python
goodput_logger_name = f'goodput_{config.run_name}' # You can choose your own logger name.
goodput_calculator = goodput.GoodputCalculator(job_name=config.run_name, logger_name=goodput_logger_name)
```

If you want to enable Pathways, turn on the `using_pathways` flag:

```python
goodput_logger_name = f'goodput_{config.run_name}' # You can choose your own logger name.
goodput_calculator = goodput.GoodputCalculator(job_name=config.run_name, logger_name=goodput_logger_name, using_pathways=True)
```

### Retrieve Goodput

Finally, call the `get_job_goodput` API to retrieve Goodput for the entire job
run. This API takes an optional parameter `include_badput_breakdown`. which
defaults to `False`.

The returned result is a tuple of the job’s Goodput at query-time, a dictionary
mapping various sources of Badput and their corresponding percentages and the
last recorded step. If `include_badput_breakdown` is not set, an empty
dictionary for Badput is returned.

If you are only interested in Goodput:

```python
total_goodput, _, _ = goodput_calculator.get_job_goodput()
print(f"Total job goodput: {total_goodput:.2f}%")
```

### Retrieve Badput Breakdown

Badput breakdown is dictionary representation of various sources of Badput
mapped to its corresponding value. Badput is the percentage of time spent by the
job doing work that is not training to the total lifetime of the job. This
includes time spent doing device initialization, training preparation, program
startup, checkpoint loading, compilation or re-compilation, data loading,
checkpoint saving, custom badput events, wasted progress and time lost due to
disruptions.

If you are interested in retrieving Badput Breakdown along with Goodput:

```python
goodput, badput_breakdown, last_step = goodput_calculator.get_job_goodput(include_badput_breakdown=True)
print(f"Last step recorded: {last_step}")
print(f"Goodput: {goodput:.2f}%")
print(f"Badput due to TPU initialization: {badput_breakdown[goodput.BadputType.TPU_INITIALIZATION]:.2f}%")
print(f"Badput due to training preparation: {badput_breakdown[goodput.BadputType.TRAINING_PREP]:.2f}%")
print(f"Badput due to program startup: {badput_breakdown[goodput.BadputType.PROGRAM_STARTUP]:.2f}%")
print(f"Badput due to data loading: {badput_breakdown[goodput.BadputType.DATA_LOADING_SYNC]:.2f}%")
print(f"Badput due to wasted progress from disruption: {badput_breakdown[goodput.BadputType.WASTED_PROGRESS_FROM_DISRUPTION]:.2f}%")
print(f"Badput due to infrastructure recovery from disruption: {badput_breakdown[goodput.BadputType.INFRASTRUCTURE_RECOVERY_FROM_DISRUPTION]:.2f}%")
print(f"Badput due to checkpoint save: {badput_breakdown[goodput.BadputType.UNPRODUCTIVE_CHECKPOINT_SAVE_TIME]:.2f}%")
print(f"Badput due to checkpoint restore: {badput_breakdown[goodput.BadputType.UNPRODUCTIVE_CHECKPOINT_RESTORE_TIME]:.2f}%")
print(f"Badput due to step evaluation: {badput_breakdown[goodput.BadputType.CUSTOM_BADPUT_EVENTS].get('EVAL_STEP', 0.0):.2f}%")
print(f"Badput due to SDC checks: {badput_breakdown[goodput.BadputType.CUSTOM_BADPUT_EVENTS].get('SDC_CHECK', 0.0):.2f}%")
print(f"Badput from unknown source: {badput_breakdown[goodput.BadputType.OTHER]:.2f}%")
```

## Supported Badput Types

Following Badput Breakdown buckets are supported by the library at this time:

```python
# Supported Badput Types
class BadputType(enum.Enum):
  """The type of Badput."""

  TPU_INITIALIZATION = 1
  TRAINING_PREP = 2
  PROGRAM_STARTUP = 3
  DATA_LOADING_SYNC = 4
  DATA_LOADING_ASYNC = 5
  UNPRODUCTIVE_CHECKPOINT_SAVE_TIME = 6
  UNPRODUCTIVE_CHECKPOINT_RESTORE_TIME = 7
  WASTED_PROGRESS_FROM_DISRUPTION = 8
  INFRASTRUCTURE_RECOVERY_FROM_DISRUPTION = 9
  CUSTOM_BADPUT_EVENTS = 10
  OTHER = 11
```

## Badput Breakdown Details

 - Accelerator Initialization Time (TPU_INITIALIZATION)

  This is the time spent on device discovery, slice initialization,
  device driver re-initialization and reset, security setup, initialization of
  pre-mapped buffers and more.

 - Training Preparation Time (TRAINING_PREP)

  This is the time spent on the creation of checkpoint managers, checkpoint
  loading, running mesh and model optimizers and more.

 - Program Startup Time (PROGRAM_STARTUP)

  This is the time spent on framework specific function transformations
  (such as JAX tracing), compilation tasks, runtime initialization etc.

 - Data Loading Time (DATA_LOADING_SYNC)

  This is the time spent on loading each batch of data for the training at a
  step to continue. This should be a small contribution to Badput if parallel
  data loading is used.

  Async data loading is accumulated overlapping with training steps and is
  non-blocking, therefore is not unproductive time. The time spent on overlapped
  data loading is stored in BadputType.DATA_LOADING_ASYNC, but does **not**
  affect overall Goodput of the workload.

 - Checkpointing Time (UNPRODUCTIVE_CHECKPOINT_SAVE_TIME,
   UNPRODUCTIVE_CHECKPOINT_RESTORE_TIME)

  This is the time spent on saving a checkpoint and restoring a checkpoint.

  Depending on the type of checkpointing technology used by the program, there
  could be unproductive time while saving a checkpoint. When checkpointing is
  synchronous, the save operation will block training progress until it is
  complete.

  During asynchronous checkpointing, the model parameters or weights have to be
  transferred from the device memory to the host memory which is a blocking
  operation on the device. After the transfer, the device can proceed with model
  training while the CPU saves the checkpoint to storage in the background. The
  first blocking operation contributes to unproductive checkpoint time.

  If auto checkpointing is used, the checkpoint save operation is initiated upon
  detection of a planned disruption signal. The save operation in type of
  checkpointing is synchronous resulting in time lost to Badput.

  > **_NOTE:_** With Orbax, this Badput is reported automatically once the
     Orbax structured logger is configured. For other checkpointers, record it
     with the checkpoint recorder API (instructions in
     [Record Checkpoint Save and Restore Time (non-Orbax checkpointers)](recording.md#record-checkpoint-save-and-restore-time-non-orbax-checkpointers)).

  > **_NOTE:_** Do **NOT** record the same checkpoint operation twice: use
     either Orbax's structured checkpoint logger **OR** the checkpoint recorder
     API, and do not also wrap it in a Custom Badput event.

 - Wasted Progress due to Disruption (WASTED_PROGRESS_FROM_DISRUPTION)

 Based on checkpointing frequency, a disruption may result in time lost in the
 form of wasted progress, i.e. time that was spent on productive training but
 lost after restart as well as time lost for the infrastructure to restart the
 workload.

 - Infrastructure Recovery Time due to Disruption
   (INFRASTRUCTURE_RECOVERY_FROM_DISRUPTION)

 This is the time taken by the infrastructure to restart the workload after a
 disruption. The root-cause of the disruption could be anything (application
 layer, infrastructure layer, hardware layer).

  When there is a disruption, Badput is expected to accumulate in
  each of the following buckets after restart:

  - Accelerator Initialization
  - Training Preparation
  - Program Startup
  - Wasted Progress due to Disruption
  - Infrastructure Recovery Time

 - Custom Badput Events (CUSTOM_BADPUT_EVENTS)

 Your application can optionally use record and monitor badput from custom
 synchronous (blocking training) and overlapping (between training steps)
 events. These events are are generally used for useful non-training activity on
 the accelerator while training is in progress such as performing SDC checks
 or evaluations.

## Interval Query Goodput and Badput

If you are interested in retrieving Goodput and Badput of the workload within a
specific window of time, the `GoodputCalculator` exposes the
`get_job_goodput_interval` API which computes metrics between the start and end
of this window.

This API also returns the last step recorded for the job. the total job time in
this window and the number of disruptions within the interval window.

> **_IMPORTANT:_** **Use this API if** you know the exact window of time within
 the workload's total run time that you are interested in.

> **_IMPORTANT:_** **Do NOT use this API if** your workload has been manually
 disrupted.

> **_IMPORTANT:_** **Do NOT use this API if** you have accidentally reused a
 previous `run_name`.

```python
# Example usage
start_time_str = "2024-12-16 1:05:00"
start_time_utc = convert_pst_to_utc(start_time_str)
end_time_str = "2024-12-17 2:00:00"
end_time_utc = convert_pst_to_utc(end_time_str)
current_goodput, badput_breakdown, last_step, total_time, disruptions = goodput_calculator.get_job_goodput_interval(start_time_utc, end_time_utc)
```

See also: [Recording intervals](recording.md), [Monitoring](monitoring.md),
[Elastic Goodput](elastic.md).
