# Synthetic goodput logs

The screenshots in `docs/images/` are rendered from these logs. They are made
up: the workload names, times and numbers are all synthetic.

*   `full-restart-1.json`, `elastic-gcs-1.json` and `elastic-snapshot-1.json`:
    goodput log payloads of three example workloads, a full restart, an elastic
    restore from a checkpoint and an elastic in-memory snapshot.
*   `compare_config.json`: a `compare` config for the three workloads.
*   `gen_synthetic.py`: writes the four files above. It only uses the Python
    standard library.

To render the pages again, for example after the timeline UI changes, run the
tool from a scratch directory outside the repository checkout (see the
[timeline README](../../../ml_goodput_measurement/timeline/README.md) for
installing the tool). The tool writes the HTML pages and an `outputs/`
directory to the current directory; inside the checkout, they would show up as
new files that are easy to commit by accident. The commands below create the
scratch directory with `mktemp -d`, under `/tmp` (or `$TMPDIR`). Start them in
this directory, so that `LOGS` points at the logs:

```bash
LOGS="$(pwd)"  # this directory, in the repository checkout
cd "$(mktemp -d)"  # a scratch directory outside the checkout
python "$LOGS/gen_synthetic.py"  # optional: rewrites the logs in $LOGS
python -m ml_goodput_measurement.timeline --command=export \
  --entries_path="$LOGS/full-restart-1.json" \
  --output_path=full-restart-1_timeline.html --entries_cache_dir=
python -m ml_goodput_measurement.timeline --command=compare \
  --compare_config="$LOGS/compare_config.json" --output_path=comparison.html \
  --nocompare_jobset_lookup --entries_cache_dir=
```

Only `gen_synthetic.py` writes to the checkout: it rewrites the logs and
`compare_config.json` next to it, which are meant to be committed. `compare`
reads the logs that `compare_config.json` lists from the config's directory.
`elastic-snapshot-1.json` renders the same way with `--command=export`. The
images are 1440 px wide screenshots at 2x, with plotly's mode bar hidden.
