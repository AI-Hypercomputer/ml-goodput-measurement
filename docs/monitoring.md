# Monitoring

`GoodputMonitor` runs the calculator on a schedule and uploads the results
to TensorBoard and Google Cloud Monitoring. This page covers its options,
the cumulative and rolling-window uploaders and the exported metrics.

## Monitor Goodput with `GoodputMonitor`

In order to monitor the Goodput of a job run on Tensorboard, all you need to do
is instantiate a `GoodputMonitor` object with the job's run name, cloud logger
name and Goodput monitoring configurations (as described below). Then call the
`start_goodput_uploader` API to asynchronously query and upload measured Goodput
to the specified Tensorboard directory.

### Create a `GoodputMonitor` object

Create a `GoodputMonitor` object with the following parameters:

 1. `job_name`: The full run name of the job.
 2. `logger_name`: The name of the Cloud Logging logger object (created in the
    previous step).
 3. `tensorboard_dir`: The directory to write TensorBoard data to.
 4. `upload_interval`: The time interval at which to query and upload data to
    TensorBoard.
 5. `monitoring_enabled`: Whether or not monitoring is enabled.
        If the application is interested in monitoring Goodput, it should set
        this value to True. Only one worker should enable monitoring.
  6. `include_badput_breakdown`: Whether to query and upload badput breakdown
        data to Tensorboard.
  7. `skip_final_flush`: Whether to skip the final goodput metrics flush upon
        termination. Defaults to `False`. Setting this to `True` is useful to
        avoid blocking job termination, as the final flush can take longer
        due to Cloud Logging reads.
  8. `max_logs_retention_period`: The maximum time period (in seconds) to query
        logs for. Defaults to 7 days (`7 * 24 * 60 * 60`). Reducing this window
        prevents query timeouts on large or long-running training runs.
  9. `wait_for_job_end_seconds`: How long the final flush may wait (polling once
        per second) for the `job_end_time` entry to become readable from Cloud
        Logging before its last upload, so the last step is credited. Defaults
        to `0` (no wait); must fit inside `final_flush_timeout_seconds`.
  10. `min_step_time`: The shortest step time (in seconds) taken into account
        when computing the ideal step time. Defaults to `1.0`, which ignores
        spuriously short step gaps; set it to `0.0` for workloads with
        sub-second steps, otherwise no ideal step time or step deviation is
        reported for them.

> **_NOTE:_** Please ensure that only **one** worker enables monitoring of
   Goodput. In JAX the check could be `jax.process_index() == 0`; in PyTorch
   `torch.distributed.get_rank() == 0`.

`GoodputMonitor` and `GoodputCalculator` also accept `read_overlap_seconds`
(default `300`) and `read_lag_seconds` (default `0`): each incremental Cloud
Logging read starts that many seconds before the last entry read (entries are
de-duplicated by `insertId`) and stops that many seconds before the query time,
so entries that become visible late are still picked up. The environment
variables `ML_GOODPUT_READ_OVERLAP_SECONDS` and `ML_GOODPUT_READ_LAG_SECONDS`
override them.

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
      max_logs_retention_period=7 * 24 * 60 * 60, # Optional: override default 7 days retention
    )
```

If you want to enable Pathways, also pass `pathway_enabled=True`.

## Monitor Cumulative Goodput Metrics

### Start Asynchronous "Query and Upload" of Goodput

Call the `start_goodput_uploader` API to launch a background process which
continuously queries and uploads cumulative Goodput metrics to Tensorboard
& Google Cloud Monitoring.

> **_NOTE:_** This will upload computed metrics to Google Cloud Monitoring
by default.

Following metrics are uploaded:

  - Productive Time (Goodput)
  - Unproductive Time (Badput Breakdown)
  - Total Elapsed Time
  - Maximum Productive Step Count
  - Disruptions Count
  - Step Time Deviation
  - Ideal Step Time
  - Stepping & Available Slice Efficiency (Elastic only, see
    [Elastic Goodput](elastic.md))

```python
goodput_monitor.start_goodput_uploader()
```

### Stop the Goodput Uploader

Call the `stop_goodput_uploader` API to perform a final upload of all metrics
and safely exit.

> _NOTE:_ This will stop all cumulative metrics upload processes.

You can optionally skip the final flush during shutdown to avoid blocking job
termination (as the final flush can take several minutes due to Cloud Logging
reads). To do this, pass `skip_final_flush=True` to `stop_goodput_uploader`:

```python
goodput_monitor.stop_goodput_uploader(skip_final_flush=True)
```

Alternatively, you can configure this during initialization of `GoodputMonitor`
by passing `skip_final_flush=True` to the constructor.

Similarly, you can also pass `skip_final_flush=True` to
`stop_rolling_window_goodput_uploader()`.

## Monitor Rolling Window Goodput Metrics

### Start asynchronous "query and upload" of Rolling Window Goodput

Call the `start_rolling_window_goodput_uploader` API to start a background
process that continuously queries and uploads **rolling window goodput metrics**
to Google Cloud Monitoring.

You must provide a list of window durations in seconds (e.g., `[60, 300, 900]`
for 1 min, 5 min, and 15 min windows).

Following metrics are uploaded:

  - Rolling Window Goodput
  - Rolling Window Badput Breakdown
  - Rolling Window Stepping & Available Slice Efficiency (Elastic only, see
    [Elastic Goodput](elastic.md))

```python
goodput_monitor.start_rolling_window_goodput_uploader(rolling_windows_seconds=[60, 300, 900])
```

### Stop the Rolling Window Goodput Uploader

Call the `stop_rolling_window_goodput_uploader` API to perform a final upload
of rolling window metrics and safely shut down the background uploader process.

> **_NOTE:_** This will stop all rolling window metrics upload processes.

```python
goodput_monitor.stop_rolling_window_goodput_uploader()
```

## Visualize on Tensorboard

1. Make sure you have `tensorboard-plugin-profile`, `tensorflow` and
   `tensorboard` packages installed
2. Follow instructions [here](https://cloud.google.com/tpu/docs/profile-tpu-vm#start_profiling_the_model_training) to start the Tensorboard server

## Access Metrics on Google Cloud Monitoring

By default, performance data is automatically sent to Google Cloud Monitoring,
enabling visualization and alerting on dashboards. This includes both cumulative
and rolling window metrics.

The metrics currently sent to Google Cloud Monitoring are:

- **Cumulative Goodput:**
  [workload/goodput_time](https://cloud.google.com/monitoring/api/metrics_gcp#:~:text=workload/goodput_time)
- **Cumulative Badput:**
  [workload/badput_time](https://cloud.google.com/monitoring/api/metrics_gcp#:~:text=workload/badput_time)
- **Rolling Window Goodput:**
  [workload/interval_goodput](https://cloud.google.com/monitoring/api/metrics_gcp#:~:text=workload/interval_goodput)
- **Rolling Window Badput:**
  [workload/interval_badput](https://cloud.google.com/monitoring/api/metrics_gcp#:~:text=workload/interval_badput)
- **Total Elapsed Time:**
  [workload/total_elapsed_time](https://cloud.google.com/monitoring/api/metrics_gcp#:~:text=workload/total_elapsed_time)
- **Maximum Productive Step:**
  [workload/max_productive_steps](https://cloud.google.com/monitoring/api/metrics_gcp#:~:text=workload/max_productive_steps)
- **Disruption Count:**
  [workload/disruptions](https://cloud.google.com/monitoring/api/metrics_gcp#:~:text=workload/disruptions)
- **Step Time Deviation:**
  [workload/step_time_deviation](https://cloud.google.com/monitoring/api/metrics_gcp#:~:text=workload/step_time_deviation)
- **Ideal Step Time:**
  [workload/performance](https://cloud.google.com/monitoring/api/metrics_gcp#:~:text=workload/performance)
- **Stepping Slice Efficiency** (Elastic only):
  [workload/stepping_slice_efficiency](https://cloud.google.com/monitoring/api/metrics_gcp#:~:text=workload/stepping_slice_efficiency)
- **Available Slice Efficiency** (Elastic only):
  [workload/available_slice_efficiency](https://cloud.google.com/monitoring/api/metrics_gcp#:~:text=workload/available_slice_efficiency)

The slice efficiency metrics are only uploaded when using
`ElasticGoodputMonitor` with `include_slice_efficiency=True` (see
[Elastic Goodput](elastic.md)). They are reported with
`window_type=CUMULATIVE` by the cumulative uploader and with
`window_type=INTERVAL` (one series per `rolling_window_size`) by the rolling
window uploader.

This feature leverages Google VM metadata (project ID, location, accelerator
type) and supports replica IDs for uniquely identifying workloads in
multi-replica deployments.

```python

gcp_options = goodput_utils.GCPOptions(
      project_id=None, # If None, the library will automatically identify from GCE internal metadata
      location=None, # If None, the library will automatically identify from GCE internal metadata
      replica_id='0', # Default is '0'
      acc_type=None, # If None, the library will automatically identify from GCE internal metadata
      cluster_name=None, # Optional GKE cluster name
      enable_gcp_goodput_metrics=True,
      enable_gcp_step_deviation_metrics=True,
    )

goodput_monitor = monitoring.GoodputMonitor(
      job_name=config.run_name,
      logger_name=logger_name,
      tensorboard_dir=config.tensorboard_dir,
      upload_interval=config.goodput_upload_interval_seconds,
      monitoring_enabled=True,
      include_badput_breakdown=True,
      include_step_deviation=True,
      configured_ideal_step_time=None, # Optional, the library will compute ideal step time if it is not provided
      gcp_options=gcp_options
    )
```

If you do not wish to send metrics to Google Cloud Monitoring then please set
the flag `enable_gcp_goodput_metrics` to `False` for disabling goodput metrics
and `enable_gcp_step_deviation_metrics` to `False` for disabling step deviation
metrics while creating the GCPOptions object.

Setting `monitoring_enabled` to `False` will disable both tensorboard and GCM
monitoring.

```python

gcp_options = goodput_utils.GCPOptions(
      project_id=None, # If None, the library will automatically identify from GCE internal metadata
      location=None, # If None, the library will automatically identify from GCE internal metadata
      replica_id='0', # Default is '0'
      acc_type=None, # If None, the library will automatically identify from GCE internal metadata
      cluster_name=None, # Optional GKE cluster name
      enable_gcp_goodput_metrics=False,
      enable_gcp_step_deviation_metrics=False,
    )


goodput_monitor = monitoring.GoodputMonitor(
      job_name=config.run_name,
      logger_name=logger_name,
      tensorboard_dir=config.tensorboard_dir,
      upload_interval=config.goodput_upload_interval_seconds,
      monitoring_enabled=True,
      include_badput_breakdown=True,
      include_step_deviation=True,
      configured_ideal_step_time=None,
      gcp_options=gcp_options,
    )
```

If you want to monitor Goodput and Badput metrics computed in a specific window
of time, you can use the `start_rolling_window_goodput_uploader` monitoring API.

### Create the `GoodputMonitor` with `enable_gcp_goodput_metrics` set to `True` in `GCPOptions`

```python

gcp_options = goodput_utils.GCPOptions(
      project_id=None, # If None, the library will automatically identify from GCE internal metadata
      location=None, # If None, the library will automatically identify from GCE internal metadata
      replica_id='0', # Default is '0'
      acc_type=None, # If None, the library will automatically identify from GCE internal metadata
      cluster_name=None, # Optional GKE cluster name
      enable_gcp_goodput_metrics=True,
    )

goodput_monitor = monitoring.GoodputMonitor(
      job_name=config.run_name,
      logger_name=logger_name,
      tensorboard_dir=config.tensorboard_dir,
      upload_interval=config.goodput_upload_interval_seconds,
      monitoring_enabled=True,
      include_badput_breakdown=True,
      gcp_options=gcp_options
    )
```

See also: [Badput types](badput.md), [Elastic Goodput](elastic.md),
[Troubleshooting](troubleshooting.md).
