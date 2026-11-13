# PyTorch and asynchronous backends

A host that issues work asynchronously (PyTorch on TPU or GPU) runs ahead
of the accelerator, so a host timestamp marks when a step was issued, not
when it ran. `StepClock` takes step boundaries from accelerator events
instead.

## Record Step Time from Accelerator Events (PyTorch)

When the host issues work asynchronously (e.g. PyTorch on TPU or GPU), the
host clock sees when a step was *dispatched*, not when the accelerator
*finished* it: the host runs ahead and only blocks when it reads a result,
so host-timed steps look very short most of the time and very long at every
sync. `StepClock` instead derives step starts from the backend's timing events
(`torch.tpu.Event` / `torch.cuda.Event` with `enable_timing=True`), which
report the host-observed completion time of the work queued before them:

- Before issuing each step, `on_step_start(step)` notes the host start; after
  issuing it, `on_step_end(step)` records an event on all ranks.
- The lead rank calls `poll()` once per step. It never blocks: once step N's
  event has completed and step N+1 has been issued, it records
  `max(done(N), host start of N+1)` as step N+1's start. In steady run-ahead
  that is the accelerator finish of step N; after a synchronous save, or when
  the host is the bottleneck, it is the host start, so time the accelerator
  spent idle between the two steps stays in step N's window.
- Events only measure time *between* events, so `on_reference_point()` pairs
  an event with the wall clock while the device is idle (job start, and after
  a sync such as a logging step) to pin them to wall-clock time and bound
  clock drift.
- `corrected_start(host_start, last_issued_step)` moves the host start time of
  a checkpoint save or eval past the compute that was still queued on the
  device, so that time is not misattributed to the save or eval.
- If the backend raises, the clock disables itself after one warning and
  records the pending steps' host starts; check `disabled` and record host
  starts yourself from then on.

The trainer records the host start of step 1; the clock records steps 2..N.

```python
from ml_goodput_measurement import step_clock

clock = step_clock.StepClock(
    goodput_recorder if is_lead else None,
    lambda: torch.tpu.Event(enable_timing=True),
    is_lead=is_lead,
)
clock.on_reference_point()
goodput_recorder.record_step_start_time(first_step)
for step in range(first_step, final_step + 1):
  clock.on_step_start(step)
  train_step()
  clock.on_step_end(step, last_step=step == final_step)
  clock.poll()
  if step % log_freq == 0:
    log_metrics()  # blocks on the device
    clock.on_reference_point()
clock.flush()
```

## Lead process

Exactly one process per job records and one monitors. In PyTorch the
check is `torch.distributed.get_rank() == 0` once the process group is
initialized; before that, `int(os.environ.get("RANK", "0")) == 0` uses
the rank set by the launcher.

The rest of the recorder API (job start and end, initialization, data
loading, custom badput events) is the same as in
[Recording intervals](recording.md).

See also: [Recording intervals](recording.md),
[Troubleshooting](troubleshooting.md).
