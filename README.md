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

 ML Goodput Measurement is a library intended to be used with Cloud TPU to log the
 necessary information and query a job's Goodput. It can be pip installed to
 import its modules, and retrieve information about a training job's overall
 productive Goodput. The package exposes API interfaces to log useful
 information from the user application and query Goodput for the job run, gain
 insight into the productivity of ML workloads and utilization of compute
 resources.

## Components


 The ML Goodput Measurement library consists of two main components: 
 the `GoodputRecorder` and the `GoodputCalculator`. The `GoodputRecorder`
 exposes APIs to the client to export key timestamps while a training job makes
 progress, namely APIs that allow logging of productive step time and total job
 run time. The library will serialize and store this data in Google Cloud
 Logging. The `GoodputCalculator` exposes APIs to compute Goodput based on the
 recorded data. Cloud Logging handles its internal operations asynchronously.
 The recommended way to compute Goodput is to run an analysis program separate
 from the training application, either on a CPU instance or on the users'
 development machine.

## Installation

 To install the ML Goodput Measurement package, run the following command on TPU VM:

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

To run your training on Cloud TPU, set up the Cloud TPU environment by following
instructions [here](https://cloud.google.com/tpu/docs/setup-gcp-account).

To learn more about Google Cloud Logging, visit this [page](https://cloud.google.com/logging/docs).


### Import

 To use this package, import the `goodput` module:


 ```python
 from ml_goodput_measurement import goodput
 ```

### Define the name of the Google Cloud Logging logger bucket

 Create a run-specific logger bucket where Cloud Logging entries can be written and read from.

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
 # TPU Initialization and device scanning…
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

Finally, call the `get_job_goodput` API to retrieve Goodput for the entire job run.

```python
total_goodput = goodput_calculator.get_job_goodput()
print(f"Total job goodput: {total_goodput:.2f}%")
```

