# Troubleshooting

Common causes of missing or wrong Goodput numbers and what addresses each.

## Duplicate or inflated entries

Exactly one process per job records (`logging_enabled=True`) and exactly one
monitors (`monitoring_enabled=True`); with more, every entry is written or
uploaded once per process. In JAX the check could be
`jax.process_index() == 0`; in PyTorch `torch.distributed.get_rank() == 0`.

Use a unique `run_name`, and so a unique logger name, for every workload; a
reused name mixes the previous run's entries into the cumulative Goodput.

## Restarts and the local cache

The calculator keeps `<cache_dir>/<job_name>_timeline.jsonl` and
`<cache_dir>/<job_name>_metadata.json`; `cache_dir` defaults to `/tmp`. A
process restarted on the same host resumes from these files instead of reading
the whole job again.

## The last step is not credited

The final flush credits the last step only once the `job_end_time` entry is
readable from Cloud Logging, which lags the write by a few seconds. Set
`wait_for_job_end_seconds` on `GoodputMonitor` so the flush waits, polling once
per second, up to that long; it must fit inside `final_flush_timeout_seconds`.

## Entries that become visible late

Each incremental Cloud Logging read starts `read_overlap_seconds` (default
`300`) before the last entry read, de-duplicated by `insertId`, and stops
`read_lag_seconds` (default `0`) before the query time; the environment
variables `ML_GOODPUT_READ_OVERLAP_SECONDS` and `ML_GOODPUT_READ_LAG_SECONDS`
override them. Raise them if live Goodput is inflated by entries that become
visible late.

## "No ideal step time available"

`get_step_deviation()` raises `ValueError` with this message when no step time
reaches `min_step_time` (default `1.0` second). For workloads with sub-second
steps, pass `min_step_time=0.0` to `GoodputCalculator` or `GoodputMonitor`, or
set `configured_ideal_step_time`.

## Long runs

`max_logs_retention_period` bounds how far back logs are queried (default 7
days). Reduce it for large or long-running jobs if queries time out.

## Checkpoint Badput is zero

`UNPRODUCTIVE_CHECKPOINT_SAVE_TIME` and `UNPRODUCTIVE_CHECKPOINT_RESTORE_TIME`
are reported only when Orbax is used with its structured logger configured.
For other checkpointers, wrap the blocking part of the save in a
[custom badput event](recording.md#record-custom-badput-events-eg-evaluation-sdc-checks);
do not do both.

See also: [Recording intervals](recording.md), [Monitoring](monitoring.md),
[Badput types](badput.md).
