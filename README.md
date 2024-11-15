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

## Overview

 ML Goodput Measurement is a library intended to be used with Cloud accelerators
 to log necessary information and query a job's Goodput and Badput Breakdown. It
 can be pip installed to import its modules, and retrieve information about a
 training job's overall productive Goodput and sources of Badput. The package
 exposes API interfaces to log useful information from the user application and
 query Goodput for the job run, gain insight into the productivity of ML
 workloads and utilization of compute resources.

 The package also exposes Goodput Monitoring APIs which allow asynchronous query
 and export of the job's Goodput to Tensorboard with configurable upload interval.

## Components


 The ML Goodput Measurement library consists of the following main components: 

  - `GoodputRecorder`

  - `GoodputCalculator`
  - `GoodputMonitor`


 The `GoodputRecorder`
 exposes APIs to the client to export key timestamps while a training job makes
 progress, namely APIs that allow logging of productive step time and total job
 run time. The library will serialize and store this data in Google Cloud
 Logging.

 The `GoodputCalculator` exposes APIs to compute Goodput based on the
 recorded data. Cloud Logging handles its internal operations asynchronously.
 The recommended way to compute Goodput is to run an analysis program separate
 from the training application, either on a CPU instance or on the users'
 development machine.

 The `GoodputMonitor` exposes APIs to query and upload goodput data to
 Tensorboard asynchronously. It does this by instantiating a `GoodputCaluclator`
 under the hood.

## Installation

 To install the ML Goodput Measurement package, run the following command on the VM:

 ```bash
 pip install ml-goodput-measurement
 ```

## Usage

The usage of this package requires the setup of a Google Cloud project with
billing enabled to properly use Google Cloud Logging. If you don't have a Google
Cloud project, or if you don't have billing enabled for your Google Cloud
project, then do the following:

1. In the Google Cloud console, on the project selector page,
 [select or create a Google Cloud project](https://cloud.google.com/resource-manager/docs/creating-managing-projects).

2. Make sure that billing is enabled for your Google Cloud project. Instructions can be found [here](https://cloud.google.com/billing/docs/how-to/verify-billing-enabled#console)

3. [Enable](https://console.cloud.google.com/flows/enableapi?apiid=logging.googleapis.com&_ga=2.27841276.1571868865.1726250448-123998259.1726107009) the Cloud Logging API.

To run your training on Cloud accelerator, set up the environment by following
instructions [here](https://cloud.google.com/tpu/docs/setup-gcp-account).

To learn more about Google Cloud Logging, visit this [page](https://cloud.google.com/logging/docs).


### Import

 To use this package, import the `goodput` module:


 ```python
 from ml_goodput_measurement import goodput
 from ml_goodput_measurement import monitoring
 ```

### Define the name of the Google Cloud Logging logger.

 Create a run-specific logger name where Cloud Logging entries can be written to and read from.

 For example:

 ```python
 goodput_logger_name = f'goodput_{config.run_name}'
 ```

### Create a `GoodputRecorder` object

 Next, create a recorder object with the following parameters:

 1. `job_name`: The full run name of the job.
 2. `logger_name`: The name of the Cloud Logging logger object (created in the previous step).
 3. `logging_enabled`: Whether or not this process has Cloud Logging enabled.

 
 
  > **_NOTE:_** For a multi-worker setup, please ensure that only one worker
   writes the logs to avoid the duplication. In JAX, for example, the check 
   could be `if jax.process_index() == 0`


  > **_NOTE:_** `logging_enabled` defaults to `False` and Goodput computations cannot be completed if no logs are ever written.

 For example:


 ```python
 goodput_recorder = goodput.GoodputRecorder(job_name=config.run_name, logger_name=goodput_logger_name, logging_enabled=(jax.process_index() == 0))
 ```


### Record Data with `GoodputRecorder`

#### Record Job Start and End Time

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


#### Record Step Time

 Use the recorder object to record a step's start time using `record_step_start_time(step_count)`:

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

#### Record Device Initialization, Training Preparation and Data Loading Time

  - Use the recorder object to record Device Initialization time using `record_tpu_init_start_time` and `record_tpu_init_end_time`.
  - Use the recorder object to record Training Preparation time using `record_training_preparation_start_time` and `record_training_preparation_end_time`.
  - Use the recorder object to record Data Loading time using `record_data_loading_start_time` and `record_data_loading_end_time`.


  For example:

  ```python
  def train_loop(config, state=None):
  goodput_recorder.record_tpu_init_start_time()
  # Set up mesh, model, state, checkpoint manager…
  goodput_recorder.record_tpu_init_end_time()
  goodput_recorder.record_training_preparation_start_time()
  # Set up training set, initialize functional train arguments and model parameters…
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

### Retrieve Goodput with `GoodputCalculator`

In order to retrieve the Goodput of a job run, all you need to do is instantiate
a `GoodputCalculator` object with the job's run name and the Cloud Logging
logger name used to record data for that job run. Then call the `get_job_goodput`
API to get the computed Goodput for the job run. 

It is recommended to make the `get_job_goodput` calls for a job run from an
instance that runs elsewhere from your training machine.


#### Create a `GoodputCalculator` object

Create the calculator object:

```python
goodput_logger_name = f'goodput_{config.run_name}' # You can choose your own logger name.
goodput_calculator = goodput.GoodputCalculator(job_name=config.run_name, logger_name=goodput_logger_name)
```

#### Retrieve Goodput

Finally, call the `get_job_goodput` API to retrieve Goodput for the entire job run. This API takes an optional parameter `include_badput_breakdown`. which defaults to `False`.

The returned result is a tuple of the job’s Goodput at query-time, a dictionary mapping various sources of Badput and their corresponding percentages and the last recorded step. If `include_badput_breakdown` is not set, an empty dictionary for Badput is returned.


If you are only interested in Goodput:

```python
total_goodput, _, _ = goodput_calculator.get_job_goodput()
print(f"Total job goodput: {total_goodput:.2f}%")
```

#### Retrieve Badput Breakdown

Badput breakdown is dictionary representation of various sources of Badput mapped to its corresponding value. 
Badput is the percentage of time spent by the job doing work that is not training to the total lifetime of the job. This includes time spent doing Device initialization, training preparation, checkpoint loading, compilation or re-compilation, data loading, checkpoint saving and time lost due to disruptions.

Following Badput Breakdown buckets are supported by the library at this time:

```python
# Supported Badput Types
class BadputType(enum.Enum):
 """The type of Badput."""
 TPU_INITIALIZATION = 1
 TRAINING_PREP = 2
 PROGRAM_STARTUP = 3
 DATA_LOADING = 4
 UNPRODUCTIVE_CHECKPOINTING = 5
 WASTED_PROGRESS_FROM_DISRUPTION = 6
 OTHER = 7
```

If you are interested in retrieving Badput Breakdown along with Goodput:

```python
goodput, badput_breakdown, last_step = goodput_calculator.get_job_goodput(include_badput_breakdown=True)
print(f"Last step recorded: {last_step}")
print(f"Goodput: {goodput:.2f}%")
print(f"Badput due to TPU initialization: {badput_breakdown[goodput.BadputType.TPU_INITIALIZATION]:.2f}%")
print(f"Badput due to training preparation: {badput_breakdown[goodput.BadputType.TRAINING_PREP]:.2f}%")
print(f"Badput due to program startup: {badput_breakdown[goodput.BadputType.PROGRAM_STARTUP]:.2f}%")
print(f"Badput due to data loading: {badput_breakdown[goodput.BadputType.DATA_LOADING]:.2f}%")
print(f"Badput due to disruption and wasted progress: {badput_breakdown[goodput.BadputType.WASTED_PROGRESS_FROM_DISRUPTION]:.2f}%")
```

### Monitor Goodput with `GoodputMonitor`

In order to monitor the Goodput of a job run on Tensorboard, all you need to do
is instantiate a `GoodputMonitor` object with the job's run name, cloud logger 
name and Goodput monitoring configurations (as described below). Then call the 
`start_goodput_uploader` API to asynchronously query and upload measured Goodput
to the specified Tensorboard directory. 


#### Create a `GoodputMonitor` object

Create a `GoodputMonitor` object with the following parameters:

 1. `job_name`: The full run name of the job.
 2. `logger_name`: The name of the Cloud Logging logger object (created in the previous step).
 3. `tensorboard_dir`: The directory to write TensorBoard data to.
 4. `upload_interval`: The time interval at which to query and upload data to TensorBoard.
 5. `monitoring_enabled`: Whether or not monitoring is enabled. If the application is
      interested in monitoring Goodput, it should set this value to True. Only one worker 
      should enable monitoring.
 6. `include_badput_breakdown`: Whether to query and upload badput breakdown data to Tensorboard.

> **_NOTE:_** Please ensure that only **one** worker enables monitoring of Goodput.
   In JAX, for example, the check could be `if jax.process_index() == 0`

For example:

```python
goodput_logger_name = f'goodput_{config.run_name}' # You can choose your own logger name.
goodput_monitoring_enabled = config.monitor_goodput and jax.process_index() == 0 # Check for configs whether or not the enable monitoring.

goodput_monitor = monitoring.GoodputMonitor(
      job_name=config.run_name,
      logger_name=logger_name,
      tensorboard_dir=config.tensorboard_dir,
      upload_interval=config.goodput_upload_interval_seconds,
      monitoring_enabled=True,
      include_badput_breakdown=True,
    )
```

#### Start asynchronous "query and upload" of Goodput

Call the `start_goodput_uploader` API to spin off a thread which continuously queries and uploads Goodput.

```python
goodput_monitor.start_goodput_uploader()
```

