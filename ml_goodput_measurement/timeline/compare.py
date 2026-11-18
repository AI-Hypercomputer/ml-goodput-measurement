"""Cross-run comparison of goodput and recovery metrics across workloads.

Compares several workloads side by side, for example the same model trained
with different recovery strategies. Each workload is summarized with the same
metrics as its single-workload timeline page (Goodput, MTBI and the 10-phase
recovery breakdown from metrics.extract_recovery_incidents). The model and the
scale (slices × chips per slice) of a workload are read from the GKE audit log
entry that created its JobSet; see load_jobset. The comparison is rendered as:

  * a cross-run comparison table with a model column and a row per group (by
    default the model plus the scale, e.g. 'Llama-3.1-405B · 2×1k'): Goodput,
    incidents and MTBI, mean MTTR, Time to Recover, Lost Progress, a mini bar
    with each phase group's share of MTTR and the primary bottleneck. Each
    group has a baseline that the "vs. baseline" changes, the "N× faster/slower
    MTTR" text and the MTTR colors compare with;
  * one Recovery anatomy chart per group: (A) mean recovery time per
    incident, stacked by phase group, and (B) each group's share of MTTR;
    hovering a bar segment lists its 10-phase breakdown, and a collapsed
    table under the chart lists the phase groups and all sub-phases;
  * an optional CSV file with one row per workload.
"""

import concurrent.futures
import csv
import dataclasses
import datetime
import html
import json
import os
import re
import threading
import time
from typing import Any, Callable, Optional, Sequence, TypeVar, Union

from cloud_goodput.ml_goodput_measurement.timeline import builder
from cloud_goodput.ml_goodput_measurement.timeline import event_parser
from cloud_goodput.ml_goodput_measurement.timeline import events
from cloud_goodput.ml_goodput_measurement.timeline import log_fetcher
from cloud_goodput.ml_goodput_measurement.timeline import metrics
import plotly.graph_objects as go


@dataclasses.dataclass(frozen=True)
class PhaseGroup:
  """Recovery sub-phases that are shown as one color in the comparison."""

  key: str
  label: str
  short: str  # Mini-bar text, e.g. '15% Wait'.
  brief: str  # Primary bottleneck text, e.g. 'Slice Wait 1.8m'.
  color: str
  phases: tuple[str, ...]


# Every recovery sub-phase belongs to exactly one group, so the group means of
# a workload add up to its mean MTTR. JIT compile overhead is counted in the
# XLA group, so the last group is Lost Progress (repeated steps, excl. JIT).
# The timeline page's "Wasted Progress" is Lost Progress plus that JIT compile,
# so phase 10 is not called Wasted Progress here.
PHASE_GROUPS: tuple[PhaseGroup, ...] = (
    PhaseGroup(
        key='infra',
        label='Slice Wait / Infra Recovery',
        short='Wait',
        brief='Slice Wait',
        color='#4285f4',
        phases=('fault_detection_teardown', 'slice_wait_reschedule'),
    ),
    PhaseGroup(
        key='init',
        label='Init (Pathways \u00b7 Mesh \u00b7 Data Init)',
        short='Init',
        brief='Init',
        color='#9aa0a6',
        phases=(
            'pathways_cluster_init',
            'tpu_mesh_model_init',
            'data_iterator_init',
            'first_batch_data_loading',
        ),
    ),
    PhaseGroup(
        key='restore',
        label='Ckpt / Snapshot Restore',
        short='Ckpt',
        brief='Ckpt Restore',
        color='#a142f4',
        phases=('checkpoint_restore',),
    ),
    PhaseGroup(
        key='xla',
        label='XLA (AOT + JIT)',
        short='XLA',
        brief='XLA Compile',
        color='#f9ab00',
        phases=('aot_xla_compilation', 'jit_compile_overhead'),
    ),
    PhaseGroup(
        key='lost',
        label='Lost Progress (Repeated Steps)',
        short='Lost',
        brief='Lost Progress',
        color='#ea4335',
        phases=('lost_progress',),
    ),
)

_SPEC_KEYS = (
    'workload_id',
    'label',
    'model',
    'group',
    'baseline',
    'timeline_url',
    'project_id',
    'start_time',
    'end_time',
    'entries_path',
    'shape',
    'note',
)
_CONFIG_KEYS = (
    'title',
    'subtitle',
    'description',
    'groups',
    'project_id',
    'start_time',
    'end_time',
    'model',
    'group',
    'workloads',
)
# Top-level config keys that are defaults for workloads that do not set them.
_DEFAULT_KEYS = ('project_id', 'start_time', 'end_time', 'model', 'group')
_GROUP_KEYS = ('name', 'description')
_SHAPE_RE = re.compile(r'(?<![0-9a-z])(\d+)x(\d+k?)(?![0-9a-z])', re.IGNORECASE)
_NOTE_MARKERS = ('*', '\u2020', '\u2021', '\u00a7', '\u00b6')
_FONT = (
    '"Google Sans", Roboto, -apple-system, BlinkMacSystemFont, "Segoe UI",'
    ' sans-serif'
)
_DEFAULT_TITLE = 'Goodput Cross-Run Comparison'
_DEFAULT_SUBTITLE = (
    'Cross-run comparison table and recovery anatomy, mean per incident.'
)
_TOOL_NAME = 'ml-goodput-measurement timeline'
# Panel A labels a segment with its time from this fraction of the largest
# MTTR in the chart on.
_MIN_VALUE_LABEL_FRACTION = 0.06
# Panel B labels a segment with its share of MTTR from this percentage on.
_MIN_SHARE_LABEL_PCT = 3.0


def parse_shape(workload_id: str) -> str:
  """Returns the slice shape in a workload id, e.g. '4×1k' for 'pw-4x1k-1'."""
  m = _SHAPE_RE.search(workload_id)
  return f'{m.group(1)}\u00d7{m.group(2)}' if m else ''


_SCALE_RE = re.compile(r'(\d+)\s*[\u00d7x]\s*(\d+)(k?)', re.IGNORECASE)


def _shape_size(shape: str) -> Optional[tuple[int, int]]:
  """(slices, chips per slice) of a shape like '4×1k' (4, 1024), or None."""
  m = _SCALE_RE.fullmatch(shape.strip()) if shape else None
  if not m:
    return None
  return int(m.group(1)), int(m.group(2)) * (1024 if m.group(3) else 1)


def shape_chips(shape: str) -> Optional[int]:
  """Returns the chip count of a shape like '4×1k' (4096), or None."""
  size = _shape_size(shape)
  return size[0] * size[1] if size else None


def format_scale_chips(shape: str) -> str:
  """Returns the shape and its chip count, e.g. '4×1k (4,096)' for '4×1k'."""
  chips = shape_chips(shape)
  return f'{shape} ({chips:,})' if chips else shape


def format_chips(chips: int) -> str:
  """A chip count as in shapes: '1k' for 1024, '4k' for 4096, '64' for 64."""
  if chips >= 1024 and chips % 1024 == 0:
    return f'{chips // 1024}k'
  return str(chips)


@dataclasses.dataclass(frozen=True)
class JobSetInfo:
  """Model and scale of a workload, from the request that created its JobSet."""

  name: str = ''  # JobSet name, e.g. 'restart-4x1k-1'.
  model_name: str = ''  # model_name=... in the container args.
  slices: int = 0  # Replicas of the TPU replicated jobs.
  # Pods per replica × TPU chips per pod; 0 if the TPU jobs differ.
  chips_per_slice: int = 0
  accelerator: str = ''  # TPU node selector, e.g. 'tpu7x'.

  @property
  def shape(self) -> str:
    """E.g. '2×1k' for 2 slices of 1,024 chips; '' if unknown."""
    if not self.slices or not self.chips_per_slice:
      return ''
    return f'{self.slices}\u00d7{format_chips(self.chips_per_slice)}'

  def describe(self) -> str:
    """E.g. 'llama3.1-405b, 2×1k (2 slices × 1,024 chips, tpu7x)'."""
    slices = f'{self.slices} slice{"s" if self.slices != 1 else ""}'
    if self.shape:
      detail = f'{slices} \u00d7 {self.chips_per_slice:,} chips'
      if self.accelerator:
        detail += f', {self.accelerator}'
      scale = f'{self.shape} ({detail})'
    elif self.slices:
      scale = f'{slices}, chips per slice unknown'
    else:
      scale = 'no TPU jobs'
    return f'{self.model_name or "no model_name"}, {scale}'


@dataclasses.dataclass
class WorkloadSpec:
  """A workload to compare and where to read its goodput logs from.

  Logs are read from entries_path (JSON list of log payloads) if set, else
  from Cloud Logging.
  """

  workload_id: str
  label: str = ''  # Recovery mode, e.g. 'Full JobSet restart'.
  model: str = ''  # Model column; defaults to the JobSet's model_name.
  group: str = ''  # Table group; defaults to 'model · scale'.
  baseline: bool = False  # Use as its group's baseline.
  timeline_url: str = ''  # Link to the workload's single-run timeline page.
  project_id: Optional[str] = None
  start_time: Optional[str] = None
  end_time: Optional[str] = None
  entries_path: Optional[str] = None
  # Scale, e.g. '2×1k'; the JobSet's is used when it is known.
  shape: str = ''
  note: str = ''

  @classmethod
  def from_dict(cls, d: dict[str, Any]) -> 'WorkloadSpec':
    unknown = sorted(set(d) - set(_SPEC_KEYS))
    if unknown:
      raise ValueError(
          f'Unknown workload keys {unknown}; allowed keys: {list(_SPEC_KEYS)}'
      )
    if not d.get('workload_id'):
      raise ValueError(f'Workload without a workload_id: {d}')
    if not isinstance(d.get('baseline', False), bool):
      raise ValueError(
          f'"baseline" must be true or false: {d["workload_id"]} has'
          f' {d["baseline"]!r}.'
      )
    return cls(**d)


@dataclasses.dataclass
class GroupInfo:
  """A table group (usually a model plus a scale range); order and header."""

  name: str
  description: str = ''

  @classmethod
  def from_item(cls, item: Any) -> 'GroupInfo':
    if isinstance(item, str):
      return cls(name=item)
    if not isinstance(item, dict):
      raise ValueError(f'A group must be a name or an object: {item!r}')
    unknown = sorted(set(item) - set(_GROUP_KEYS))
    if unknown:
      raise ValueError(
          f'Unknown group keys {unknown}; allowed keys: {list(_GROUP_KEYS)}'
      )
    if not item.get('name'):
      raise ValueError(f'Group without a name: {item}')
    return cls(name=item['name'], description=item.get('description') or '')


@dataclasses.dataclass
class ComparisonConfig:
  """Workloads to compare plus the page title, subtitle and groups."""

  workloads: list[WorkloadSpec]
  title: str = ''
  subtitle: str = ''
  description: str = ''  # Paragraph above the comparison table.
  groups: list[GroupInfo] = dataclasses.field(default_factory=list)


def _name_key(text: str) -> str:
  """Lower-cased text without spaces and with 'x' for '×'.

  Group names that only differ in these match, and shapes are found in group
  names and descriptions with it.

  Args:
    text: A group name, description or shape.

  Returns:
    The normalized text.
  """
  return re.sub(r'\s+', '', text).lower().replace('\u00d7', 'x')


def _check_baselines(members: Sequence[tuple[str, str, bool]]) -> None:
  """Raises ValueError if a group has more than one "baseline": true.

  Args:
    members: (group, workload id, baseline) of each workload.
  """
  baselines: dict[str, list[str]] = {}
  names: dict[str, str] = {}
  for group, workload_id, is_baseline in members:
    if is_baseline:
      key = _name_key(group)
      names.setdefault(key, group)
      baselines.setdefault(key, []).append(workload_id)
  for key, workload_ids in baselines.items():
    if len(workload_ids) > 1:
      group = names[key]
      where = f'group {group!r}' if group else 'the workloads without a group'
      raise ValueError(
          f'More than one "baseline": true in {where}:'
          f' {", ".join(workload_ids)}. Set it on at most one workload per'
          ' group.'
      )


def config_from_dict(
    raw: dict[str, Any], base_dir: Optional[str] = None
) -> ComparisonConfig:
  """Builds a ComparisonConfig from a parsed JSON config.

  Top-level project_id, start_time, end_time, model and group are defaults for
  workloads that do not set them. A workload can be a bare workload id string.
  Relative entries_path values are resolved against base_dir. A workload
  without a group is grouped by its model and scale once they are known (see
  summarize_workload); optional "groups" list the table groups (a name or
  {"name", "description"}) in display order. At most one workload per group
  can set "baseline": true; this is checked here for workloads with a group,
  and by group_summaries for the others.

  Args:
    raw: The parsed JSON object.
    base_dir: Directory of the config file.

  Returns:
    The comparison config.
  """
  if not isinstance(raw, dict):
    raise ValueError('The comparison config must be a JSON object.')
  unknown = sorted(set(raw) - set(_CONFIG_KEYS))
  if unknown:
    raise ValueError(
        f'Unknown config keys {unknown}; allowed keys: {list(_CONFIG_KEYS)}'
    )
  items = raw.get('workloads')
  if not items or not isinstance(items, list):
    raise ValueError(
        'The comparison config needs a non-empty "workloads" list.'
    )
  group_items = raw.get('groups') or []
  if not isinstance(group_items, list):
    raise ValueError('"groups" must be a list of names or objects.')
  defaults = {k: raw[k] for k in _DEFAULT_KEYS if raw.get(k)}
  specs = []
  for item in items:
    if isinstance(item, str):
      item = {'workload_id': item}
    spec = WorkloadSpec.from_dict({**defaults, **item})
    if base_dir and spec.entries_path and not os.path.isabs(spec.entries_path):
      spec.entries_path = os.path.join(base_dir, spec.entries_path)
    specs.append(spec)
  _check_baselines(
      [(s.group, s.workload_id, s.baseline) for s in specs if s.group]
  )
  return ComparisonConfig(
      workloads=specs,
      title=raw.get('title') or '',
      subtitle=raw.get('subtitle') or '',
      description=raw.get('description') or '',
      groups=[GroupInfo.from_item(g) for g in group_items],
  )


def load_config(path: str) -> ComparisonConfig:
  """Reads a JSON comparison config file (see config_from_dict)."""
  with open(path, encoding='utf-8') as f:
    raw = json.load(f)
  return config_from_dict(raw, base_dir=os.path.dirname(os.path.abspath(path)))


def read_entries_file(path: str) -> list[dict[str, Any]]:
  """Reads log payloads from a JSON list or an entries cache file."""
  with open(path, encoding='utf-8') as f:
    data = json.load(f)
  if isinstance(data, dict):
    data = data.get('entries', [])
  if not isinstance(data, list):
    raise ValueError(f'{path}: expected a JSON list of log entries.')
  return data


def display_name(spec: WorkloadSpec) -> str:
  """Returns 'label (workload_id)', or the workload id without a label."""
  if spec.label and spec.label != spec.workload_id:
    return f'{spec.label} ({spec.workload_id})'
  return spec.workload_id


def _is_deadline_error(e: Exception) -> bool:
  names = {cls.__name__ for cls in type(e).__mro__}
  return 'DeadlineExceeded' in names or 'deadline' in str(e).lower()


_T = TypeVar('_T')


def _with_quota_retries(workload_id: str, fetch: Callable[[], _T]) -> _T:
  """Calls fetch, retrying it on Cloud Logging quota errors."""
  waits = log_fetcher.QUOTA_RETRY_WAITS_S
  for attempt, wait_s in enumerate(waits, start=1):
    try:
      return fetch()
    except Exception as e:  # pylint: disable=broad-exception-caught
      if not log_fetcher.is_quota_error(e):
        raise
      print(
          f'  {workload_id}: Cloud Logging quota exceeded; retrying in'
          f' {wait_s}s ({attempt}/{len(waits)}).'
      )
      time.sleep(wait_s)
  return fetch()


def load_entries(
    spec: WorkloadSpec,
    project_id: str,
    cache_dir: Optional[str] = None,
    refresh: bool = False,
) -> list[dict[str, Any]]:
  """Returns the goodput log entries of a workload.

  Reads spec.entries_path if it is set; otherwise fetches the logs from Cloud
  Logging (see log_fetcher.fetch_log_entries). With cache_dir, the fetched
  entries and their timestamps are saved to <cache_dir>/<workload_id>.json,
  even if there are none. Later calls read that file instead of Cloud
  Logging while the project and workload are the same and the requested time
  range is within the cached one (see _cached_entries); entries outside the
  requested range are left out. The file is not refreshed automatically,
  e.g. for a job that was still running when it was read.

  Args:
    spec: The workload.
    project_id: GCP project used when spec.project_id is not set.
    cache_dir: Optional directory for cached entries.
    refresh: Fetch the logs even if they are cached, and update the cache.

  Returns:
    The log entry payloads.
  """
  if spec.entries_path:
    return read_entries_file(spec.entries_path)

  project = spec.project_id or project_id
  query = {
      'project_id': project,
      'workload_id': spec.workload_id,
      'start_time': spec.start_time,
      'end_time': spec.end_time,
  }
  cache_path = (
      os.path.join(cache_dir, f'{spec.workload_id}.json') if cache_dir else None
  )
  if cache_path and not refresh:
    cached = _read_json(cache_path)
    entries = _cached_entries(cached, query) if cached else None
    if cached and entries is not None:
      fetched_at = cached.get('fetched_at')
      print(
          f'  Using cached entries from {cache_path}'
          + (f', read at {fetched_at}.' if fetched_at else '')
      )
      if _read_before_end(cached, query):
        print(
            '  NOTE: the cache was read before the end of the time range;'
            ' logs written later are not in it (--refresh_cache reads them).'
        )
      return entries

  if not spec.start_time:
    print(
        f'  WARNING: no start_time for {spec.workload_id}. Without a time'
        ' range, Cloud Logging only returns the last 24 hours of logs.'
    )
  timed = log_fetcher.fetch_log_entries(
      project, spec.workload_id, spec.start_time, spec.end_time
  )
  entries = [payload for _, payload in timed]
  if cache_path:
    now = datetime.datetime.now(datetime.timezone.utc)
    _write_cache(
        cache_path,
        {
            'query': query,
            'fetched_at': now.isoformat(timespec='seconds'),
            'timestamps': [timestamp for timestamp, _ in timed],
            'entries': entries,
        },
    )
  return entries


def _parse_time(value: Any) -> datetime.datetime:
  """Parses an ISO 8601 time; a time without a UTC offset is in UTC.

  Args:
    value: The time, e.g. '2026-01-21T18:00:00-08:00'.

  Returns:
    The time with its time zone.

  Raises:
    ValueError: value is not such a time.
  """
  if not isinstance(value, str):
    raise ValueError(f'Not a time: {value!r}')
  t = datetime.datetime.fromisoformat(value.strip().replace('Z', '+00:00'))
  return t if t.tzinfo else t.replace(tzinfo=datetime.timezone.utc)


def _cached_entries(
    cached: dict[str, Any], query: dict[str, Any]
) -> Optional[list[dict[str, Any]]]:
  """The entries of a cache file in the time range of a query, or None.

  A cache file is used for the same query, and for a query with the same
  project and workload whose time range is within the cached one. A query
  without a start or end time is covered by any cached one; a cache without
  one only covers a query without it.

  Args:
    cached: The data of a cache file (see load_entries).
    query: The project, workload and time range to read.

  Returns:
    The cached entries from the query's start to its end time (inclusive), or
    None if the cache does not cover the query.
  """
  entries = cached.get('entries')
  old = cached.get('query')
  if not isinstance(entries, list) or not isinstance(old, dict):
    return None
  if old == query:
    return entries
  timestamps = cached.get('timestamps')
  if (
      old.get('project_id') != query['project_id']
      or old.get('workload_id') != query['workload_id']
      or not isinstance(timestamps, list)
      or len(timestamps) != len(entries)
  ):
    return None  # Another workload, or written by an earlier version.
  try:
    start = _parse_time(query['start_time']) if query['start_time'] else None
    end = _parse_time(query['end_time']) if query['end_time'] else None
    if start and not (
        old.get('start_time') and _parse_time(old['start_time']) <= start
    ):
      return None
    if end and not (
        old.get('end_time') and end <= _parse_time(old['end_time'])
    ):
      return None
  except ValueError:
    return None
  kept = []
  for timestamp, entry in zip(timestamps, entries):
    try:
      t = _parse_time(timestamp)
    except ValueError:
      t = None  # Kept: Cloud Logging returned it for the cached range.
    if t is None or ((not start or t >= start) and (not end or t <= end)):
      kept.append(entry)
  return kept


def _read_before_end(cached: dict[str, Any], query: dict[str, Any]) -> bool:
  """True if a cache file was read before the end of a query's time range.

  The range ends at the query's end time, else at the cached one; without
  either, logs written after the cache was read are missing from it.

  Args:
    cached: The data of a cache file that covers the query.
    query: The project, workload and time range to read.
  """
  fetched_at = cached.get('fetched_at')
  if not fetched_at:
    return False  # Written by an earlier version; the read time is unknown.
  end = query['end_time'] or (cached.get('query') or {}).get('end_time')
  try:
    return not end or _parse_time(end) > _parse_time(fetched_at)
  except ValueError:
    return False


def _read_json(path: str) -> Optional[dict[str, Any]]:
  """The data of a JSON cache file, or None if it is missing or unreadable."""
  if not os.path.exists(path):
    return None
  try:
    with open(path, encoding='utf-8') as f:
      data = json.load(f)
  except (OSError, ValueError) as e:
    print(f'  Ignoring unreadable cache file {path}: {e}')
    return None
  return data if isinstance(data, dict) else None


def _read_cache(
    path: Optional[str], query: dict[str, Any]
) -> Optional[dict[str, Any]]:
  """Returns the data of a cache file if it was written for the same query."""
  cached = _read_json(path) if path else None
  if cached is not None and cached.get('query') == query:
    return cached
  return None


def _write_cache(path: str, data: dict[str, Any]) -> None:
  """Saves data as JSON; prints a warning instead of raising on failure.

  The data is written to a temporary file that is then renamed, so an
  interrupted run, or two workloads with the same id, never leave a partial
  cache file.

  Args:
    path: The cache file.
    data: JSON-serializable data.
  """
  tmp_path = f'{path}.{os.getpid()}.{threading.get_ident()}.tmp'
  try:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp_path, 'w', encoding='utf-8') as f:
      json.dump(data, f)
    os.replace(tmp_path, path)
  except (OSError, TypeError, ValueError) as e:
    print(f'  WARNING: could not write cache file {path}: {e}')
    if os.path.exists(tmp_path):
      os.remove(tmp_path)


# The TPU replicated jobs of a JobSet are those whose containers use this
# resource; the node selector names the TPU type.
_TPU_RESOURCE = 'google.com/tpu'
_TPU_ACCELERATOR_SELECTOR = 'cloud.google.com/gke-tpu-accelerator'
# model_name=... or --model_name=..., but not e.g. base_model_name=...
_MODEL_NAME_RE = re.compile(r'(?<![\w.\-])(?:--)?model_name=[\'"]?([\w.\-]+)')
# The JobSet create entry of a workload is searched from _JOBSET_LOOKBACK
# before to _JOBSET_SLACK after its first goodput event, newest first.
_JOBSET_LOOKBACK = datetime.timedelta(days=2)
_JOBSET_SLACK = datetime.timedelta(minutes=10)


def _number(value: Any) -> Optional[float]:
  """A JSON number or numeric string as a float, else None."""
  if isinstance(value, bool):
    return None
  if isinstance(value, (int, float)):
    return float(value)
  if not isinstance(value, str):
    return None
  try:
    return float(value)
  except ValueError:
    return None


def _container_text(container: dict[str, Any]) -> str:
  """The command and args of a container as one string."""
  parts = []
  for key in ('command', 'args'):
    value = container.get(key) or []
    parts.extend(value if isinstance(value, list) else [value])
  return ' '.join(str(p) for p in parts)


def parse_jobset_request(proto: dict[str, Any]) -> Optional[JobSetInfo]:
  """Reads the model and scale from a jobsets.create audit log entry.

  The TPU replicated jobs are those whose containers use google.com/tpu
  chips. Their replicas add up to the slice count, and their chips per slice
  are the pods per replica (parallelism) times the chips per pod. model_name
  is the first model_name=... in the command or args of any container, e.g.
  of the Pathways head job.

  Args:
    proto: The protoPayload of the audit log entry.

  Returns:
    The JobSet info, or None if it has no TPU jobs and no model_name.
  """
  request = proto.get('request') or {}
  name = (request.get('metadata') or {}).get('name') or str(
      proto.get('resourceName', '')
  ).rsplit('/', 1)[-1]
  slices = 0
  chips_per_slice = set()
  accelerators = []
  model_name = ''
  for job in (request.get('spec') or {}).get('replicatedJobs') or []:
    job_spec = (job.get('template') or {}).get('spec') or {}
    pod_spec = (job_spec.get('template') or {}).get('spec') or {}
    chips_per_pod = 0.0
    for container in pod_spec.get('containers') or []:
      resources = container.get('resources') or {}
      chips = _number((resources.get('limits') or {}).get(_TPU_RESOURCE))
      if chips is None:
        chips = _number((resources.get('requests') or {}).get(_TPU_RESOURCE))
      chips_per_pod += chips or 0.0
      if not model_name:
        m = _MODEL_NAME_RE.search(_container_text(container))
        model_name = m.group(1) if m else ''
    if chips_per_pod <= 0:
      continue
    # Kubernetes defaults replicas and parallelism to 1.
    replicas = _number(job.get('replicas'))
    pods = _number(job_spec.get('parallelism'))
    slices += int(1 if replicas is None else replicas)
    chips_per_slice.add(int((1 if pods is None else pods) * chips_per_pod))
    accelerator = (pod_spec.get('nodeSelector') or {}).get(
        _TPU_ACCELERATOR_SELECTOR
    )
    if accelerator and accelerator not in accelerators:
      accelerators.append(accelerator)
  if not slices and not model_name:
    return None
  return JobSetInfo(
      name=name,
      model_name=model_name,
      slices=slices,
      chips_per_slice=chips_per_slice.pop() if len(chips_per_slice) == 1 else 0,
      accelerator='/'.join(accelerators),
  )


def _first_event_time(
    entries: Sequence[dict[str, Any]],
) -> Optional[datetime.datetime]:
  """The earliest *_time value (Unix seconds) of goodput log payloads."""
  times = []
  for entry in entries:
    if not isinstance(entry, dict):
      continue
    for key, value in entry.items():
      t = _number(value) if key.endswith('_time') else None
      if t is not None and t > 1e9:
        times.append(t)
  if not times:
    return None
  return datetime.datetime.fromtimestamp(min(times), tz=datetime.timezone.utc)


def load_jobset(
    spec: WorkloadSpec,
    project_id: str,
    first_event: datetime.datetime,
    cache_dir: Optional[str] = None,
    refresh: bool = False,
) -> Optional[JobSetInfo]:
  """Returns the model and scale from the JobSet create entry of a workload.

  Searches the Admin Activity audit log for the newest jobsets.create entry of
  <workload_id> or pw-<workload_id> from _JOBSET_LOOKBACK before to
  _JOBSET_SLACK after first_event (see log_fetcher.fetch_jobset_create),
  retrying on quota errors. With cache_dir, the result, also when no entry is
  found, is saved to <cache_dir>/<workload_id>.jobset.json and reused for the
  same project and first_event.

  Args:
    spec: The workload.
    project_id: GCP project used when spec.project_id is not set.
    first_event: The first goodput event of the workload.
    cache_dir: Optional directory for the cached result.
    refresh: Look the JobSet up even if it is cached, and update the cache.

  Returns:
    The JobSet info, or None if no create entry is found.
  """
  project = spec.project_id or project_id
  query = {
      'project_id': project,
      'workload_id': spec.workload_id,
      'first_event': first_event.isoformat(),
      'lookback_s': _JOBSET_LOOKBACK.total_seconds(),
  }
  cache_path = (
      os.path.join(cache_dir, f'{spec.workload_id}.jobset.json')
      if cache_dir
      else None
  )
  cached = None if refresh else _read_cache(cache_path, query)
  if cached is not None:
    info = cached.get('jobset')
    try:
      return JobSetInfo(**info) if info else None
    except TypeError:
      pass  # Written by another version of this tool; look it up again.
  proto = _with_quota_retries(
      spec.workload_id,
      lambda: log_fetcher.fetch_jobset_create(
          project,
          spec.workload_id,
          first_event - _JOBSET_LOOKBACK,
          first_event + _JOBSET_SLACK,
      ),
  )
  info = parse_jobset_request(proto) if proto else None
  if cache_path:
    _write_cache(
        cache_path,
        {'query': query, 'jobset': dataclasses.asdict(info) if info else None},
    )
  return info


def _same_shape(a: str, b: str) -> bool:
  size_a, size_b = _shape_size(a), _shape_size(b)
  if size_a and size_b:
    return size_a == size_b
  return _name_key(a) == _name_key(b)


def _model_key(model: str) -> str:
  """Letters and digits only, e.g. 'llama31405b' for 'Llama-3.1-405B'."""
  return re.sub(r'[^0-9a-z]', '', model.lower())


def jobset_warnings(spec: WorkloadSpec, jobset: JobSetInfo) -> list[str]:
  """Says where the config or the workload id disagrees with the JobSet.

  The JobSet's scale is used instead of the config's shape and the shape in
  the workload id, and the config's model is shown instead of the JobSet's
  model_name; see summarize_workload.

  Args:
    spec: The workload.
    jobset: Its JobSet.

  Returns:
    One warning per disagreement.
  """
  warnings = []
  if jobset.shape:
    for shape, where in (
        (spec.shape, 'the config'),
        (parse_shape(spec.workload_id), 'the workload id'),
    ):
      if shape and not _same_shape(shape, jobset.shape):
        warnings.append(
            f"shape {shape} in {where} differs from the JobSet's"
            f' {jobset.shape}; using {jobset.shape}.'
        )
    # A group named after one scale, e.g. 'Llama-3.1-405B · 4×1k'.
    group_shapes = [m.group(0) for m in _SCALE_RE.finditer(spec.group)]
    if len(group_shapes) == 1 and not _same_shape(
        group_shapes[0], jobset.shape
    ):
      warnings.append(
          f'shape {group_shapes[0]} in group {spec.group!r} differs from the'
          f" JobSet's {jobset.shape}."
      )
  if (
      spec.model
      and jobset.model_name
      and _model_key(spec.model) != _model_key(jobset.model_name)
  ):
    warnings.append(
        f"model {spec.model!r} in the config differs from the JobSet's"
        f' model_name {jobset.model_name!r}; showing {spec.model!r}.'
    )
  return warnings


@dataclasses.dataclass
class LoadedWorkload:
  """The logs of one workload, or why they could not be loaded."""

  spec: WorkloadSpec
  # Goodput log payloads.
  entries: list[dict[str, Any]] = dataclasses.field(default_factory=list)
  # Why the workload is skipped; empty if its logs were loaded.
  error: str = ''
  # Model and scale from its JobSet, if looked up and found.
  jobset: Optional[JobSetInfo] = None
  # Printed under the workload's status line, e.g. its JobSet and warnings.
  messages: list[str] = dataclasses.field(default_factory=list)


def _load_error(e: Exception) -> str:
  detail = f'{type(e).__name__}: {e}' if str(e) else type(e).__name__
  message = f'failed to load its logs: {detail}'.rstrip('. ')
  if _is_deadline_error(e):
    message += '. Try a narrower time range for this workload'
  return message + '.'


def _lookup_jobset(
    spec: WorkloadSpec,
    project_id: str,
    entries: Sequence[dict[str, Any]],
    cache_dir: Optional[str],
    refresh: bool = False,
) -> tuple[Optional[JobSetInfo], list[str]]:
  """load_jobset for a loaded workload; returns the JobSet and messages."""
  fallback = 'using the config and the workload id.'
  first_event = _first_event_time(entries)
  if first_event is None:
    return None, [f'JobSet not looked up (no event times); {fallback}']
  try:
    jobset = load_jobset(
        spec, project_id, first_event, cache_dir=cache_dir, refresh=refresh
    )
  except Exception as e:  # pylint: disable=broad-exception-caught
    detail = f'{type(e).__name__}: {e}'
    return None, [f'JobSet lookup failed ({detail}); {fallback}']
  if jobset is None:
    days = _JOBSET_LOOKBACK.days
    return None, [
        f'No JobSet create entry in the {days} days before the first goodput'
        f' event ({first_event:%Y-%m-%d %H:%M}Z); {fallback}'
    ]
  return jobset, [f'JobSet {jobset.name}: {jobset.describe()}'] + [
      f'WARNING: {w}' for w in jobset_warnings(spec, jobset)
  ]


def load_workloads(
    specs: Sequence[WorkloadSpec],
    project_id: str,
    cache_dir: Optional[str] = None,
    max_parallel: int = 4,
    jobset_lookup: bool = False,
    refresh: bool = False,
) -> list[LoadedWorkload]:
  """Loads the logs of several workloads, up to max_parallel at a time.

  Cloud Logging queries take seconds to minutes and are I/O bound, so they
  run in a thread pool. A workload whose logs cannot be loaded, or that has
  no log entries, gets an error message instead of stopping the others.

  Args:
    specs: The workloads.
    project_id: GCP project used when a spec has no project_id.
    cache_dir: Optional directory for cached entries and JobSets; see
      load_entries and load_jobset.
    max_parallel: Maximum number of workloads loaded at the same time.
    jobset_lookup: Also read the model and scale of the workloads whose logs
      come from Cloud Logging from their JobSet; see load_jobset. A failed
      lookup only adds a message.
    refresh: Read the logs and JobSets from Cloud Logging even if they are
      cached, and update the cache.

  Returns:
    One LoadedWorkload per spec, in the order of specs.
  """

  def load(spec: WorkloadSpec) -> LoadedWorkload:
    entries = load_entries(
        spec, project_id, cache_dir=cache_dir, refresh=refresh
    )
    if not entries:
      return LoadedWorkload(
          spec,
          error=(
              'no goodput log entries found; check project_id, workload_id'
              ' and the time range.'
          ),
      )
    result = LoadedWorkload(spec, entries=entries)
    if jobset_lookup and not spec.entries_path:
      result.jobset, result.messages = _lookup_jobset(
          spec, project_id, entries, cache_dir, refresh=refresh
      )
    return result

  loaded: dict[int, LoadedWorkload] = {}
  pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, max_parallel))
  try:
    futures = {pool.submit(load, spec): i for i, spec in enumerate(specs)}
    for future in concurrent.futures.as_completed(futures):
      i = futures[future]
      try:
        result = future.result()
      except Exception as e:  # pylint: disable=broad-exception-caught
        result = LoadedWorkload(specs[i], error=_load_error(e))
      loaded[i] = result
      if result.error:
        status = f'SKIPPED, {result.error}'
      else:
        status = f'{len(result.entries)} log entries.'
      print(f'  {display_name(result.spec)}: {status}')
      for message in result.messages:
        print(f'    {message}')
  finally:
    # On Ctrl+C, do not start the workloads that are still queued.
    pool.shutdown(wait=False, cancel_futures=True)
  return [loaded[i] for i in range(len(specs))]


@dataclasses.dataclass
class WorkloadSummary:
  """Comparison metrics of one workload."""

  workload_id: str
  label: str
  model: str = ''
  group: str = ''  # Table group: the spec's group, else 'model · scale'.
  baseline_override: bool = False  # "baseline": true in the config.
  timeline_url: str = ''
  shape: str = ''
  note: str = ''
  goodput_pct: Optional[float] = None  # None if Goodput is unavailable.
  total_job_time_s: float = 0.0
  last_step: int = 0
  mtbi_s: float = 0.0
  rec_stats: metrics.RecoverySummaryStats = dataclasses.field(
      default_factory=metrics.RecoverySummaryStats
  )

  @property
  def num_incidents(self) -> int:
    return self.rec_stats.num_incidents

  @property
  def mean_mttr_s(self) -> float:
    return self.rec_stats.mean_mttr_s

  @property
  def mean_time_to_recover_s(self) -> float:
    """Mean downtime + JIT compile overhead (phases 1-9)."""
    if not self.num_incidents:
      return 0.0
    return self.rec_stats.mean_downtime_s + self.phase_mean_s(
        'jit_compile_overhead'
    )

  @property
  def mean_lost_progress_s(self) -> float:
    return self.rec_stats.mean_lost_progress_s

  @property
  def baseline_step_time_s(self) -> float:
    return self.rec_stats.baseline_step_time_s

  def phase_mean_s(self, phase: str) -> float:
    return float(self.rec_stats.phase_stats.get(phase, {}).get('mean', 0.0))

  def group_mean_s(self, group: PhaseGroup) -> float:
    return sum(self.phase_mean_s(p) for p in group.phases)

  def group_share_pct(self, group: PhaseGroup) -> float:
    if self.mean_mttr_s <= 0:
      return 0.0
    return self.group_mean_s(group) / self.mean_mttr_s * 100.0


# Footnote of a workload whose goodput log has no training steps.
_NO_STEPS_NOTE = 'Goodput n/a: no training step events in its goodput log.'


def _has_steps(segments: Sequence[builder.Segment]) -> bool:
  """Whether the timeline has training steps, productive or repeated."""
  return any(
      s.kind in (events.EventType.STEP_BLOCK, events.EventType.WASTED_PROGRESS)
      for s in segments
  )


def summarize_workload(
    entries: list[dict[str, Any]],
    spec: WorkloadSpec,
    jobset: Optional[JobSetInfo] = None,
) -> WorkloadSummary:
  """Computes the comparison metrics of one workload.

  Without training steps (e.g. a goodput log with only checkpoint events),
  Goodput is n/a and the job time 0, and a footnote is added to the note.

  The model is the config's, else the JobSet's model_name. The scale is the
  JobSet's slices × chips per slice, else the config's shape, else the shape
  in the workload id. The table group is the config's, else 'model · scale',
  e.g. 'Llama-3.1-405B · 2×1k'.

  Args:
    entries: Goodput log payloads.
    spec: The workload.
    jobset: The workload's JobSet, if known; see load_jobset.

  Returns:
    The workload summary.
  """
  segments = builder.build_timeline(event_parser.parse_entries(entries))
  rec_stats = metrics.extract_recovery_incidents(entries, segments)
  mtbi_s = metrics.calculate_timeline_metrics(segments)['mtbi']

  goodput_pct = None
  total_job_time_s = 0.0
  last_step = 0
  note = spec.note
  if not _has_steps(segments):
    # cloud_goodput reports 0% for logs without steps (e.g. only checkpoint
    # events), over a job time that spans just those events. Show n/a and no
    # job time instead, and say why in a footnote.
    note = note.strip()
    if note and not note.endswith(('.', '!', '?')):
      note += '.'
    note = f'{note} {_NO_STEPS_NOTE}'.lstrip()
  else:
    try:
      # calculate_goodput_metrics appends job start/end entries to a list, so
      # give it a copy.
      goodput_val, _, last_step, total_job_time_s = (
          metrics.calculate_goodput_metrics(list(entries), spec.workload_id)
      )
      if total_job_time_s > 0:
        goodput_pct = float(goodput_val)
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f'  Goodput unavailable for {spec.workload_id}: {e}')

  model = spec.model or (jobset.model_name if jobset else '')
  shape = (
      (jobset.shape if jobset else '')
      or spec.shape
      or parse_shape(spec.workload_id)
  )
  return WorkloadSummary(
      workload_id=spec.workload_id,
      label=spec.label or spec.workload_id,
      model=model,
      group=spec.group or ' \u00b7 '.join(p for p in (model, shape) if p),
      baseline_override=spec.baseline,
      timeline_url=spec.timeline_url,
      shape=shape,
      note=note,
      goodput_pct=goodput_pct,
      total_job_time_s=float(total_job_time_s),
      last_step=int(last_step or 0),
      mtbi_s=float(mtbi_s),
      rec_stats=rec_stats,
  )


def _fmt_duration(seconds: float) -> str:
  if seconds < 60:
    return f'{seconds:.1f}s'
  if seconds < 3600:
    return f'{seconds / 60:.1f}m'
  return f'{seconds / 3600:.2f}h'


def _fmt_goodput(summary: WorkloadSummary) -> str:
  # Two decimals, like the Goodput card of the timeline page.
  if summary.goodput_pct is None:
    return 'n/a'
  return f'{summary.goodput_pct:.2f}%'


def _fmt_mtbi(summary: WorkloadSummary) -> str:
  # MTBI is 0 when the workload has no stepping window at all.
  return _fmt_duration(summary.mtbi_s) if summary.mtbi_s > 0 else '\u2014'


def _pick_unit(max_seconds: float) -> tuple[str, float, str]:
  """Returns (suffix, divisor, name) of the chart unit for the largest MTTR."""
  if max_seconds < 120:
    return 's', 1.0, 'seconds'
  if max_seconds < 3 * 3600:
    return 'm', 60.0, 'minutes'
  return 'h', 3600.0, 'hours'


def _note_markers(summaries: Sequence[WorkloadSummary]) -> list[str]:
  """Returns a footnote marker per workload ('' for workloads without note).

  Workloads with the same note text share a marker.

  Args:
    summaries: The workloads.

  Returns:
    One marker per workload.
  """
  by_note: dict[str, str] = {}
  for s in summaries:
    if s.note and s.note not in by_note:
      count = len(by_note)
      base = _NOTE_MARKERS[count % len(_NOTE_MARKERS)]
      by_note[s.note] = base * (count // len(_NOTE_MARKERS) + 1)
  return [by_note.get(s.note, '') for s in summaries]


def _esc(text: Any) -> str:
  return html.escape(str(text), quote=False)


_HOVER_MUTED = '<span style="color:#cbd5e1">'
_HOVER_RULE = (
    '<span'
    ' style="color:#475569">\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500</span>'
)


def _group_hover(summary: WorkloadSummary, group: PhaseGroup) -> str:
  """Hover text for one phase-group bar segment.

  Lists the group mean, every sub-phase of the group (mean, share of MTTR and
  min/max over the incidents) and the workload's Time to Recover and MTTR, so
  the chart replaces a per-sub-phase table.

  Args:
    summary: The workload.
    group: The phase group of the bar segment.

  Returns:
    HTML for plotly's hovertemplate.
  """
  mean_s = summary.group_mean_s(group)
  mttr_s = summary.mean_mttr_s
  lines = [
      f'<b>{_esc(group.label)}</b>',
      f'{_HOVER_MUTED}{_esc(summary.label)}</span>',
      (
          f'Mean per incident: <b>{_fmt_duration(mean_s)}</b> ({mean_s:.1f}s)'
          f' \u2022 <b>{summary.group_share_pct(group):.1f}%</b> of MTTR'
      ),
      _HOVER_RULE,
  ]
  for phase in group.phases:
    stats = summary.rec_stats.phase_stats.get(phase, {})
    phase_mean_s = float(stats.get('mean', 0.0))
    pct = phase_mean_s / mttr_s * 100.0 if mttr_s > 0 else 0.0
    line = (
        f'{_HOVER_MUTED}{_esc(metrics.RECOVERY_PHASE_LABELS[phase])}:</span>'
        f' {_fmt_duration(phase_mean_s)} ({pct:.1f}%)'
    )
    if summary.num_incidents > 1 and phase_mean_s > 0:
      line += (
          f' \u00b7 min {_fmt_duration(float(stats.get("min", 0.0)))}'
          f' \u00b7 max {_fmt_duration(float(stats.get("max", 0.0)))}'
      )
    lines.append(line)
  if group.key == 'lost':
    lines.append(
        f'{_HOVER_MUTED}Avg repeated steps:</span>'
        f' {summary.rec_stats.mean_rolled_back_steps:.1f}'
    )
  lines += [
      _HOVER_RULE,
      (
          f'{_HOVER_MUTED}Time to Recover (1\u20139):</span>'
          f' {_fmt_duration(summary.mean_time_to_recover_s)} \u00b7'
          f' {_HOVER_MUTED}MTTR (1\u201310):</span> {_fmt_duration(mttr_s)}'
          f' \u00b7 {_HOVER_MUTED}Incidents:</span> {summary.num_incidents}'
      ),
  ]
  return '<br>'.join(lines)


def _row_label(title: str, subline_parts: Sequence[str]) -> str:
  subline = ' \u00b7 '.join(_esc(p) for p in subline_parts if p)
  return (
      f'<b>{_esc(title)}</b><br>'
      f'<span style="font-size:11px;color:#64748b">{subline}</span>'
  )


def build_comparison_figure(
    summaries: Sequence[WorkloadSummary],
    title: str = '',
    subtitle: str = '',
    width: int = 1150,
) -> go.Figure:
  """Builds the two-panel cross-run recovery chart.

  Panel A stacks the mean time per incident of each phase group; panel B shows
  each group's share of the workload's MTTR. Workloads are listed top to
  bottom in the given order.

  Args:
    summaries: The workloads to compare.
    title: Chart title.
    subtitle: Chart subtitle.
    width: Figure width in pixels. The figure has a fixed size so that the label
      column and titles line up in the PNG download.

  Returns:
    The plotly figure.
  """
  if not summaries:
    raise ValueError('No workloads to compare.')
  n = len(summaries)
  markers = _note_markers(summaries)
  max_mttr = max(s.mean_mttr_s for s in summaries)
  unit, divisor, unit_name = _pick_unit(max_mttr)

  titles, sublines_a, sublines_b = [], [], []
  for s, marker in zip(summaries, markers):
    wid = s.workload_id if s.label != s.workload_id else ''
    incidents = f'{s.num_incidents} incident' + (
        's' if s.num_incidents != 1 else ''
    )
    titles.append(s.label + marker)
    sublines_a.append([
        wid,
        # Same unit as the bar-end totals.
        f'MTTR {s.mean_mttr_s / divisor:.1f}{unit}' if s.num_incidents else '',
        f'Goodput {_fmt_goodput(s)}',
    ])
    sublines_b.append([
        wid,
        s.shape,
        incidents,
        f'MTBI {_fmt_duration(s.mtbi_s)}' if s.mtbi_s > 0 else '',
    ])
  labels_a = [_row_label(t, sub) for t, sub in zip(titles, sublines_a)]
  labels_b = [_row_label(t, sub) for t, sub in zip(titles, sublines_b)]

  # Layout geometry in pixels. The label column fits the longest label
  # (13px bold) and subline (11px).
  longest_title = max(len(t) for t in titles)
  longest_sub = max(
      len(' \u00b7 '.join(p for p in sub if p))
      for sub in sublines_a + sublines_b
  )
  label_w = int(min(500, max(240, longest_title * 7.8, longest_sub * 6.0) + 24))
  right = 90
  row_h = 58
  panel_h = n * row_h
  gap = 80
  footnotes = list(
      dict.fromkeys(
          f'{marker} {s.note}'
          for s, marker in zip(summaries, markers)
          if marker
      )
  )
  top = 52 + (18 if subtitle else 0) + 30 + 30
  bottom = 26 + 18 * len(footnotes)
  height = top + 2 * panel_h + gap + bottom
  plot_w = width - label_w - right
  plot_h = 2 * panel_h + gap
  x_left = -(label_w - 12) / plot_w  # Paper x of the label column's left edge.
  dom_b_top = panel_h / plot_h
  dom_a_bottom = (panel_h + gap) / plot_h

  fig = go.Figure()
  ys = list(range(n))
  # Plotly shrinks the font of a label to fit its segment.
  inside_text = dict(
      textposition='inside',
      insidetextanchor='middle',
      textangle=0,
      constraintext='inside',
      textfont=dict(color='#ffffff', size=12),
  )
  min_label = _MIN_VALUE_LABEL_FRACTION * max_mttr / divisor  # Chart units.
  for group in PHASE_GROUPS:
    hovers = [_group_hover(s, group) for s in summaries]
    shares = [s.group_share_pct(group) for s in summaries]
    values = [s.group_mean_s(group) / divisor for s in summaries]
    common = dict(
        orientation='h',
        y=ys,
        width=0.62,
        name=group.label,
        legendgroup=group.key,
        marker=dict(color=group.color, line=dict(color='#ffffff', width=0.8)),
        customdata=hovers,
        hovertemplate='%{customdata}<extra></extra>',
    )
    fig.add_trace(
        go.Bar(
            x=values,
            xaxis='x',
            yaxis='y',
            # Segments below _MIN_VALUE_LABEL_FRACTION of the largest MTTR are
            # too narrow for a label; their value is in the hover.
            text=[
                f'<b>{v:.1f}{unit}</b>' if v > 0 and v >= min_label else ''
                for v in values
            ],
            **inside_text,
            **common,
        )
    )
    fig.add_trace(
        go.Bar(
            x=shares,
            xaxis='x2',
            yaxis='y2',
            showlegend=False,
            # Segments below _MIN_SHARE_LABEL_PCT are too narrow for a label;
            # their value is in the hover.
            text=[
                f'{v:.0f}%' if v >= _MIN_SHARE_LABEL_PCT else '' for v in shares
            ],
            **inside_text,
            **common,
        )
    )

  annotations = []
  for i, s in enumerate(summaries):
    for yref, label in (('y', labels_a[i]), ('y2', labels_b[i])):
      annotations.append(
          dict(
              xref='paper',
              x=x_left,
              yref=yref,
              y=i,
              text=label,
              xanchor='left',
              yanchor='middle',
              align='left',
              showarrow=False,
              font=dict(size=13, color='#0f172a'),
          )
      )
    if s.num_incidents:
      annotations.append(
          dict(
              xref='x',
              x=s.mean_mttr_s / divisor,
              yref='y',
              y=i,
              text=f'<b>{s.mean_mttr_s / divisor:.1f}{unit}</b>',
              xanchor='left',
              xshift=6,
              showarrow=False,
              font=dict(size=12.5, color='#0f172a'),
          )
      )
    else:
      for xref, yref in (('x', 'y'), ('x2', 'y2')):
        annotations.append(
            dict(
                xref=xref,
                x=0,
                yref=yref,
                y=i,
                text='<i>No recovery incidents</i>',
                xanchor='left',
                showarrow=False,
                font=dict(size=12, color='#94a3b8'),
            )
        )
  panel_titles = (
      (
          1.0,
          (
              '<b>A \u00b7 Mean recovery per incident</b> <span'
              f' style="color:#64748b">({unit_name}, linear scale)</span>'
          ),
      ),
      (
          dom_b_top,
          (
              '<b>B \u00b7 Where the recovery time goes</b> <span'
              ' style="color:#64748b">(share of MTTR)</span>'
          ),
      ),
  )
  for y, text in panel_titles:
    annotations.append(
        dict(
            xref='paper',
            x=x_left,
            yref='paper',
            y=y,
            yshift=8,
            text=text,
            xanchor='left',
            yanchor='bottom',
            showarrow=False,
            font=dict(size=13.5, color='#0f172a'),
        )
    )
  for k, note in enumerate(footnotes):
    annotations.append(
        dict(
            xref='paper',
            x=x_left,
            yref='paper',
            y=0,
            yshift=-14 - 18 * k,
            text=f'<span style="color:#64748b">{_esc(note)}</span>',
            xanchor='left',
            yanchor='top',
            showarrow=False,
            font=dict(size=11.5),
        )
    )

  title_text = f'<b>{_esc(title or f"Recovery anatomy across {n} runs")}</b>'
  if subtitle:
    title_text += (
        '<br><span style="font-size:13px;color:#64748b;font-weight:normal">'
        f'{_esc(subtitle)}</span>'
    )
  x_max = max_mttr / divisor * 1.14 if max_mttr > 0 else 1.0
  hidden_axis = dict(
      showticklabels=False, showgrid=False, zeroline=False, fixedrange=True
  )
  fig.update_layout(
      barmode='stack',
      width=width,
      height=height,
      autosize=False,
      margin=dict(l=label_w, r=right, t=top, b=bottom),
      font=dict(family=_FONT, color='#0f172a'),
      title=dict(
          text=title_text,
          xref='container',
          x=12 / width,
          xanchor='left',
          yref='container',
          y=1 - 14 / height,
          yanchor='top',
          font=dict(size=19, color='#0f172a'),
      ),
      legend=dict(
          orientation='h',
          # A horizontal legend above the plot that is anchored in the left
          # margin may use the margin width too, so the groups fit on one row.
          xref='paper',
          x=x_left,
          xanchor='left',
          yref='paper',
          y=1 + 38 / plot_h,
          yanchor='bottom',
          font=dict(size=12, color='#334155'),
          traceorder='normal',
      ),
      xaxis=dict(
          domain=[0, 1],
          anchor='y',
          range=[0, x_max],
          ticksuffix=unit,
          showgrid=True,
          gridcolor='#eef2f7',
          zeroline=False,
          fixedrange=True,
          tickfont=dict(size=11, color='#64748b'),
      ),
      yaxis=dict(
          domain=[dom_a_bottom, 1],
          anchor='x',
          range=[n - 0.5, -0.5],
          **hidden_axis,
      ),
      xaxis2=dict(domain=[0, 1], anchor='y2', range=[0, 100], **hidden_axis),
      yaxis2=dict(
          domain=[0, dom_b_top],
          anchor='x2',
          range=[n - 0.5, -0.5],
          **hidden_axis,
      ),
      annotations=annotations,
      hoverlabel=dict(
          bgcolor='#0f172a',
          bordercolor='#334155',
          font=dict(family=_FONT, size=12, color='#f8fafc'),
          align='left',
      ),
      paper_bgcolor='#ffffff',
      plot_bgcolor='#ffffff',
  )
  return fig


@dataclasses.dataclass
class SummaryGroup:
  """Workloads shown under one group row and in one chart."""

  name: str
  description: str
  summaries: list[WorkloadSummary]
  # The workload that the group's "vs. baseline" changes, "N× faster/slower
  # MTTR" text and MTTR colors compare with; see _pick_baseline.
  baseline: Optional[WorkloadSummary] = None


_OTHER_GROUP = 'Other'


def _pick_baseline(
    summaries: Sequence[WorkloadSummary],
) -> Optional[WorkloadSummary]:
  """Returns the baseline of a group of workloads.

  That is the workload with "baseline": true, else the first workload with
  recovery incidents, else the first workload (its Goodput is still compared).
  A "baseline": true workload whose logs could not be loaded has no summary,
  so the default applies then. One whose logs have no Goodput or no incidents
  stays the baseline; see _baseline_note.

  Args:
    summaries: The workloads of the group, in display order.

  Returns:
    The baseline, or None if there are no workloads.
  """
  for s in summaries:
    if s.baseline_override:
      return s
  for s in summaries:
    if s.num_incidents:
      return s
  return summaries[0] if summaries else None


def _baseline_note(group: SummaryGroup) -> str:
  """Says which values of a group are not compared with its baseline.

  Goodput is not compared when the baseline has no Goodput but another
  workload of the group has, and recovery times are not compared when the
  baseline has no recovery incidents but another workload has.

  Args:
    group: The group.

  Returns:
    The note, or '' if every value is compared.
  """
  base = group.baseline
  if base is None:
    return ''
  others = [s for s in group.summaries if s is not base]
  no_goodput = base.goodput_pct is None and any(
      s.goodput_pct is not None for s in others
  )
  no_incidents = not base.num_incidents and any(s.num_incidents for s in others)
  if no_goodput and no_incidents:
    return (
        f'No vs. baseline changes: the baseline, {base.label}, has no Goodput'
        ' and no recovery incidents.'
    )
  if no_goodput:
    return (
        f'Goodput is not compared: the baseline, {base.label}, has no'
        ' Goodput.'
    )
  if no_incidents:
    return (
        'Recovery times are not compared: the baseline,'
        f' {base.label}, has no recovery incidents.'
    )
  return ''


def check_baselines(summaries: Sequence[WorkloadSummary]) -> None:
  """Raises ValueError if a table group has more than one "baseline": true."""
  _check_baselines(
      [(s.group, s.workload_id, s.baseline_override) for s in summaries]
  )


def group_summaries(
    summaries: Sequence[WorkloadSummary], groups: Sequence[GroupInfo] = ()
) -> list[SummaryGroup]:
  """Groups workloads by WorkloadSummary.group and picks each baseline.

  Groups listed in `groups` come first, in that order; other groups follow in
  order of first appearance. Names match without case and spaces, and with
  'x' for '×'; a listed group is shown with its name from `groups`. When some
  workloads have a group and others do not, the ungrouped ones are listed
  last under 'Other'. Without any group, all workloads form one unnamed group.
  See _pick_baseline for the baseline of a group.

  Args:
    summaries: The workloads.
    groups: Group order and descriptions from the config.

  Returns:
    The non-empty groups in display order.

  Raises:
    ValueError: If a group has more than one "baseline": true workload.
  """
  check_baselines(summaries)
  by_key: dict[str, list[WorkloadSummary]] = {
      _name_key(g.name): [] for g in groups
  }
  names = {_name_key(g.name): g.name for g in groups}
  descriptions = {_name_key(g.name): g.description for g in groups}
  for s in summaries:
    key = _name_key(s.group)
    by_key.setdefault(key, []).append(s)
    names.setdefault(key, s.group)
  has_groups = any(s.group for s in summaries)
  result = []
  for key, items in by_key.items():
    if not items:
      continue
    result.append(
        SummaryGroup(
            name=names[key] or (_OTHER_GROUP if has_groups else ''),
            description=descriptions.get(key, ''),
            summaries=items,
            baseline=_pick_baseline(items),
        )
    )
  # Ungrouped workloads go last; the sort is stable.
  result.sort(
      key=lambda g: has_groups and not any(s.group for s in g.summaries)
  )
  return result


# Goodput % cell color: green from _GOODPUT_GREEN, yellow from
# _GOODPUT_YELLOW, red below.
_GOODPUT_GREEN = 85.0
_GOODPUT_YELLOW = 60.0
# A recovery time (MTTR, Time to Recover, Lost Progress) is faster (green) or
# slower (red) than its group's baseline when its "vs. baseline" change is at
# least this many percent. The MTTR cell color and the "N× faster/slower MTTR"
# text use the same rule, so they always agree with the change shown.
_CHANGE_PCT = 5
_GREEN = '#137333'
_RED = '#c5221f'
_GRAY = '#5f6368'


def _goodput_class(pct: Optional[float]) -> str:
  if pct is None:
    return ''
  if pct >= _GOODPUT_GREEN:
    return 'g'
  return 'y' if pct >= _GOODPUT_YELLOW else 'r'


def _pct_change(value: float, base: float) -> Optional[int]:
  """Percent change from base to value, truncated toward 0 as shown.

  Truncating keeps the shown change and its color consistent: a change shown
  as -5% is at least 5% lower.

  Args:
    value: The workload's value.
    base: The baseline's value.

  Returns:
    The change in whole percent, or None if base is not positive.
  """
  if base <= 0:
    return None
  return int((value - base) / base * 100.0)


def _change_class(value: float, base: float) -> str:
  """'g' if value is at least _CHANGE_PCT lower than base, 'r' if higher."""
  pct = _pct_change(value, base)
  if pct is None or abs(pct) < _CHANGE_PCT:
    return ''
  return 'g' if pct < 0 else 'r'


def _compares_recovery(s: WorkloadSummary, base: WorkloadSummary) -> bool:
  """Whether the recovery times of s are compared with the baseline's."""
  return base is not s and s.num_incidents > 0 and base.num_incidents > 0


def _mttr_class(s: WorkloadSummary, base: Optional[WorkloadSummary]) -> str:
  """MTTR cell color vs. the group's baseline; the baseline is not colored."""
  if base is None or not _compares_recovery(s, base):
    return ''
  return _change_class(s.mean_mttr_s, base.mean_mttr_s)


def _delta_html(value: float, base: float, points: bool = False) -> str:
  """Small 'vs. baseline' line under a table cell.

  Args:
    value: The workload's value.
    base: The baseline's value.
    points: Whether the values are Goodput percentages: the change is then in
      percentage points and higher is better. Otherwise they are recovery
      times: the change is in percent, lower is better and only changes of
      _CHANGE_PCT or more are colored.

  Returns:
    The HTML line, or '' if there is no change to show.
  """
  if points:
    diff = round(value - base, 2) + 0.0  # + 0.0 turns -0.0 into 0.0.
    text = f'{diff:+.2f}\u00a0pp'
    color = _GREEN if diff > 0 else _RED if diff < 0 else _GRAY
  else:
    pct = _pct_change(value, base)
    if pct is None:
      return ''
    text = f'{pct:+d}%' if pct else '0%'
    color = {'g': _GREEN, 'r': _RED}.get(_change_class(value, base), _GRAY)
  # The non-breaking spaces keep '+1.23 pp' and 'vs. baseline' together, so a
  # narrow cell wraps the line between the two.
  return (
      f'<div class="sub delta" style="color:{color}">'
      f'{text} vs.\u00a0baseline</div>'
  )


def _phase_share_html(s: WorkloadSummary) -> str:
  """Mini stacked bar of each group's share of MTTR plus a text summary."""
  if not s.num_incidents:
    return '<span class="mini-sub">No recovery incidents</span>'
  segs, texts = [], []
  for g in PHASE_GROUPS:
    pct = s.group_share_pct(g)
    if pct > 0:
      segs.append(
          f'<div class="mini-seg" style="width:{pct:.1f}%;'
          f'background:{g.color}"></div>'
      )
    texts.append(f'{pct:.0f}%\u00a0{_esc(g.short)}')
  text = ' \u00b7 '.join(texts)
  return (
      f'<div class="mini-bar">{"".join(segs)}</div>'
      f'<div class="mini-sub">{text}</div>'
  )


def _mttr_ratio_html(
    s: WorkloadSummary, base: Optional[WorkloadSummary]
) -> str:
  """'N× faster MTTR' or 'N× slower MTTR' vs. the group's baseline."""
  if base is None or not _compares_recovery(s, base):
    return ''
  if s.mean_mttr_s <= 0 or base.mean_mttr_s <= 0:
    return ''
  side = _change_class(s.mean_mttr_s, base.mean_mttr_s)
  if side == 'g':
    ratio = base.mean_mttr_s / s.mean_mttr_s
    return f'<b class="g">{ratio:.1f}\u00d7 faster MTTR</b>'
  if side == 'r':
    ratio = s.mean_mttr_s / base.mean_mttr_s
    return f'<b class="r">{ratio:.1f}\u00d7 slower MTTR</b>'
  return ''


def _bottleneck_html(
    s: WorkloadSummary, base: Optional[WorkloadSummary]
) -> str:
  """The two largest phase groups, plus the MTTR speed-up vs. the baseline."""
  if not s.num_incidents:
    return '\u2014'
  top = sorted(PHASE_GROUPS, key=s.group_mean_s, reverse=True)[:2]
  parts = [
      f'{_esc(g.brief)} {_fmt_duration(s.group_mean_s(g))}'
      f' ({s.group_share_pct(g):.1f}%)'
      for g in top
      if s.group_mean_s(g) > 0
  ]
  text = ' + '.join(parts) or '\u2014'
  ratio = _mttr_ratio_html(s, base)
  return f'{text} \u00b7 {ratio}' if ratio else text


def _scale_range(summaries: Sequence[WorkloadSummary]) -> str:
  """The smallest and largest shape by chip count, e.g. '2×1k–4×1k'."""
  shapes = sorted({s.shape for s in summaries if s.shape})
  if len(shapes) == 1:
    return shapes[0]
  sized: list[tuple[int, str]] = []
  for shape in shapes:
    chips = shape_chips(shape)
    if chips:
      sized.append((chips, shape))
  if not sized:
    return ''
  sized.sort()
  (lo_chips, lo), (hi_chips, hi) = sized[0], sized[-1]
  return lo if lo_chips == hi_chips else f'{lo}\u2013{hi}'


def _group_row_text(group: SummaryGroup) -> str:
  """E.g. 'Llama-405B (Scale 2×1k–4×1k · Step time 9.80–10.40s)'.

  The scale range is left out when the group name or description already has
  those shapes, e.g. 'Llama-405B · 2×1k–4×1k'. The step time is the
  steady-state step time of the workloads, as a range if they differ.

  Args:
    group: The group.

  Returns:
    The group row text.
  """
  parts = [group.description] if group.description else []
  scale = _scale_range(group.summaries)
  known = _name_key(f'{group.name} {group.description}')
  if scale and not all(_name_key(sh) in known for sh in scale.split('\u2013')):
    parts.append(f'Scale {scale}')
  step_times = sorted(
      s.baseline_step_time_s
      for s in group.summaries
      if s.baseline_step_time_s > 0
  )
  if step_times:
    lo, hi = f'{step_times[0]:.2f}', f'{step_times[-1]:.2f}'
    parts.append(
        f'Step time {lo}s' if lo == hi else f'Step time {lo}\u2013{hi}s'
    )
  detail = ' \u00b7 '.join(parts)
  return f'{group.name} ({detail})' if detail else group.name


_TABLE_COLUMNS = (
    'Recovery Mode',
    'Workload',
    'Scale',
    'Goodput %',
    'Incidents (MTBI)',
    'Mean MTTR (incl. Redo)',
    'Time to Recover (Phases 1\u20139)',
    'Lost Progress (Phase 10)',
    'Mean MTTR Phase Share',
    'Primary Bottleneck',
)


def _summary_row_html(
    s: WorkloadSummary,
    group: SummaryGroup,
    marker: str,
    show_model_column: bool,
) -> str:
  """One table row of the cross-run comparison table."""
  base = group.baseline
  dash = '\u2014'

  def sub(text: str) -> str:
    return f'<div class="sub">{text}</div>'

  mode = f'{_esc(s.label)}{_esc(marker)}'
  if base is s and len(group.summaries) > 1:
    mode += sub('baseline')
  # The cell does not wrap (td.wl) except at the <wbr> after each '_', so long
  # ids such as long_snake_case_run_id wrap but restart-4x1k-1 does not.
  workload = _esc(s.workload_id).replace('_', '_<wbr>')
  if s.timeline_url:
    workload = (
        f'<a href="{html.escape(s.timeline_url)}" target="_blank">'
        f'{workload}</a>'
    )
  scale = _esc(format_scale_chips(s.shape)) or dash
  if s.baseline_step_time_s > 0:
    scale += sub(f'Step time {s.baseline_step_time_s:.2f}s')
  goodput = _fmt_goodput(s)
  if (
      base is not None
      and base is not s
      and s.goodput_pct is not None
      and base.goodput_pct is not None
  ):
    goodput += _delta_html(s.goodput_pct, base.goodput_pct, points=True)
  incidents = str(s.num_incidents)
  if s.num_incidents and s.mtbi_s > 0:
    incidents += f' ({_fmt_duration(s.mtbi_s)})'
  if s.total_job_time_s > 0:
    incidents += sub(f'Job time {_fmt_duration(s.total_job_time_s)}')
  if s.num_incidents:
    stats = s.rec_stats
    mttr = f'{_fmt_duration(s.mean_mttr_s)} ({s.mean_mttr_s:.0f}s)'
    if s.num_incidents > 1:
      mttr += sub(
          f'{_fmt_duration(stats.min_mttr_s)} \u2013'
          f' {_fmt_duration(stats.max_mttr_s)}'
      )
    ttr_pct = (
        s.mean_time_to_recover_s / s.mean_mttr_s * 100.0
        if s.mean_mttr_s > 0
        else 0.0
    )
    ttr = f'{_fmt_duration(s.mean_time_to_recover_s)} ({ttr_pct:.1f}%)'
    lost = (
        f'{_fmt_duration(s.mean_lost_progress_s)} /'
        f' {stats.mean_rolled_back_steps:.1f} steps'
        f' ({s.group_share_pct(PHASE_GROUPS[-1]):.1f}%)'
    )
    if base is not None and _compares_recovery(s, base):
      mttr += _delta_html(s.mean_mttr_s, base.mean_mttr_s)
      ttr += _delta_html(s.mean_time_to_recover_s, base.mean_time_to_recover_s)
      lost += _delta_html(s.mean_lost_progress_s, base.mean_lost_progress_s)
  else:
    mttr, ttr, lost = dash, dash, dash

  def mono(text: str, color: str = '') -> str:
    cls = f'mono {color}' if color else 'mono'
    return f'<td class="{cls}">{text}</td>'

  cells = []
  if show_model_column:
    cells.append(f'<td class="model"><b>{_esc(s.model) or dash}</b></td>')
  cells += [
      f'<td>{mode}</td>',
      f'<td class="mono wl">{workload}</td>',
      mono(scale),
      mono(goodput, _goodput_class(s.goodput_pct)),
      mono(incidents),
      mono(mttr, _mttr_class(s, base)),
      mono(ttr),
      mono(lost),
      f'<td>{_phase_share_html(s)}</td>',
      f'<td>{_bottleneck_html(s, base)}</td>',
  ]
  return '<tr>' + ''.join(cells) + '</tr>'


def _summary_table_html(
    groups: Sequence[SummaryGroup], markers_by_id: dict[str, str]
) -> str:
  """The cross-run comparison table: one row per workload, grouped.

  A Model column is shown when any workload sets a model, and each named
  group starts with a group row. Within a group, the "vs. baseline" changes,
  the MTTR colors and the "N× faster/slower MTTR" text compare with the
  group's baseline. When the baseline lacks Goodput or recovery incidents, a
  note row at the start of the group says what is not compared.

  Args:
    groups: The workloads, grouped.
    markers_by_id: Footnote marker per workload id.

  Returns:
    The table HTML.
  """
  show_model_column = any(s.model for g in groups for s in g.summaries)
  columns = (('Model',) if show_model_column else ()) + _TABLE_COLUMNS
  rows = []
  for group in groups:
    if group.name:
      rows.append(
          f'<tr class="group-row"><td colspan="{len(columns)}">'
          f'{_esc(_group_row_text(group))}</td></tr>'
      )
    note = _baseline_note(group)
    if note:
      rows.append(
          f'<tr class="group-note"><td colspan="{len(columns)}">'
          f'{_esc(note)}</td></tr>'
      )
    for s in group.summaries:
      rows.append(
          _summary_row_html(
              s,
              group,
              markers_by_id.get(s.workload_id, ''),
              show_model_column,
          )
      )
  header = ''.join(f'<th>{_esc(c)}</th>' for c in columns)
  return (
      '<div class="table-wrap"><table class="summary">'
      f'<thead><tr>{header}</tr></thead><tbody>{"".join(rows)}</tbody>'
      '</table></div>'
  )


def _workload_header_cells(
    summaries: Sequence[WorkloadSummary], markers: Sequence[str]
) -> str:
  cells = []
  for s, marker in zip(summaries, markers):
    sub = (
        f'<div class="sub">{_esc(s.workload_id)}</div>'
        if s.label != s.workload_id
        else ''
    )
    cells.append(f'<th class="num">{_esc(s.label)}{_esc(marker)}{sub}</th>')
  return ''.join(cells)


def _value_cell(s: WorkloadSummary, mean_s: float) -> str:
  if not s.num_incidents:
    return '<td class="num">\u2014</td>'
  pct = mean_s / s.mean_mttr_s * 100.0 if s.mean_mttr_s > 0 else 0.0
  return (
      f'<td class="num"><b>{_fmt_duration(mean_s)}</b>'
      f'<div class="sub">{pct:.1f}%</div></td>'
  )


def _total_row(summaries: Sequence[WorkloadSummary], label: str) -> str:
  cells = ''.join(
      f'<td class="num">{_fmt_duration(s.mean_mttr_s)}</td>'
      if s.num_incidents
      else '<td class="num">\u2014</td>'
      for s in summaries
  )
  return f'<tr class="total"><td>{label}</td>{cells}</tr>'


def _group_table_html(
    summaries: Sequence[WorkloadSummary], markers: Sequence[str]
) -> str:
  """Mean time per incident of each phase group (rows) per workload."""
  rows = []
  for group in PHASE_GROUPS:
    cells = ''.join(_value_cell(s, s.group_mean_s(group)) for s in summaries)
    rows.append(
        f'<tr><td><span class="chip" style="background:{group.color}"></span>'
        f'{_esc(group.label)}</td>{cells}</tr>'
    )
  rows.append(_total_row(summaries, 'Total MTTR (mean per incident)'))
  return (
      '<div class="table-wrap"><table class="cmp"><thead><tr><th>Phase'
      f' group</th>{_workload_header_cells(summaries, markers)}</tr></thead>'
      f'<tbody>{"".join(rows)}</tbody></table></div>'
  )


def _phase_table_html(
    summaries: Sequence[WorkloadSummary], markers: Sequence[str]
) -> str:
  """Mean time per incident of each of the 10 recovery sub-phases."""
  group_of = {p: g for g in PHASE_GROUPS for p in g.phases}
  rows = []
  for phase in metrics.RECOVERY_PHASE_ORDER:
    color = group_of[phase].color if phase in group_of else '#cbd5e1'
    cells = ''.join(_value_cell(s, s.phase_mean_s(phase)) for s in summaries)
    rows.append(
        f'<tr><td><span class="chip" style="background:{color}"></span>'
        f'{_esc(metrics.RECOVERY_PHASE_LABELS[phase])}</td>{cells}</tr>'
    )
  ttr_cells = ''.join(
      _value_cell(s, s.mean_time_to_recover_s) for s in summaries
  )
  rows.append(
      f'<tr class="subtotal"><td>Time to Recover (phases 1\u20139)</td>'
      f'{ttr_cells}</tr>'
  )
  rows.append(_total_row(summaries, 'Total MTTR (phases 1\u201310)'))
  return (
      '<div class="table-wrap"><table class="cmp"><thead><tr><th>Recovery'
      f' sub-phase</th>{_workload_header_cells(summaries, markers)}</tr>'
      f'</thead><tbody>{"".join(rows)}</tbody></table></div>'
  )


def _legend_bar_html() -> str:
  items = ''.join(
      '<span class="leg-item"><span class="leg-dot"'
      f' style="background:{g.color}"></span>{_esc(g.label)}</span>'
      for g in PHASE_GROUPS
  )
  return f'<div class="legend-bar">{items}</div>'


def _footnotes_html(
    summaries: Sequence[WorkloadSummary], markers: Sequence[str]
) -> str:
  """Footnotes under the comparison table, one per distinct note."""
  notes = dict.fromkeys(
      f'{marker} {s.note}' for s, marker in zip(summaries, markers) if marker
  )
  if not notes:
    return ''
  return (
      '<div class="footnotes">'
      + ''.join(f'<div>{_esc(n)}</div>' for n in notes)
      + '</div>'
  )


def _notes_html(
    summaries: Sequence[WorkloadSummary], markers: Sequence[str]
) -> str:
  """Footnotes and metric definitions."""
  labels_by_marker: dict[str, list[str]] = {}
  note_by_marker: dict[str, str] = {}
  for s, marker in zip(summaries, markers):
    if marker:
      labels_by_marker.setdefault(marker, []).append(s.label)
      note_by_marker[marker] = s.note
  items = [
      f'<li><b>{_esc(marker)}</b> {_esc(", ".join(labels))}:'
      f' {_esc(note_by_marker[marker])}</li>'
      for marker, labels in labels_by_marker.items()
  ]
  groups = ''.join(
      f'<li><span class="chip" style="background:{g.color}"></span>'
      f'<b>{_esc(g.label)}</b>: '
      + ', '.join(_esc(metrics.RECOVERY_PHASE_LABELS[p]) for p in g.phases)
      + '</li>'
      for g in PHASE_GROUPS
  )
  definitions = (
      '<li><b>MTTR</b> (per incident, incl. redo) = Time to Recover + Lost'
      ' Progress (phases 1\u201310). Values are means over the incidents of'
      ' each workload; a workload needs at least two runs to have an'
      ' incident.</li>'
      '<li><b>Time to Recover</b> = downtime from the end of the last step'
      ' before the interruption to the first step after it, plus JIT compile'
      ' overhead (phases 1\u20139).</li>'
      '<li><b>Lost Progress</b> = time spent re-running steps that were'
      ' already done before the interruption (rollback), excluding JIT'
      ' compile (phase 10). The <b>Wasted Progress</b> of the timeline page'
      ' is Lost Progress plus the JIT compile overhead of the re-run.</li>'
      '<li><b>MTBI</b> = mean length of a run\u2019s stepping window.</li>'
      '<li><b>Goodput</b> = preserved productive time / total job time,'
      ' from the cloud_goodput library. Cells are green from'
      f' {_GOODPUT_GREEN:.0f}%, yellow from {_GOODPUT_YELLOW:.0f}% and red'
      ' below. Goodput is n/a when the goodput log has no training step'
      ' events.</li>'
      '<li><b>Groups</b>: workloads with the same model and scale form a'
      ' group, e.g. Llama-3.1-405B \u00b7 2\u00d71k. The model is the'
      ' config\u2019s "model", else the JobSet\u2019s; the scale is the'
      ' JobSet\u2019s slices \u00d7 chips per slice, else the config\u2019s'
      ' "shape", else the shape in the workload id. A "group" in the config'
      ' overrides model and scale. Each group has'
      ' a <b>baseline</b>: the workload with "baseline": true, else the first'
      ' workload with recovery incidents. A baseline without Goodput or'
      ' recovery incidents stays the baseline; a note at the top of its group'
      ' says what is not compared then.</li>'
      '<li><b>vs. baseline</b> lines compare with the group\u2019s baseline:'
      ' Goodput in percentage points, recovery times in percent. A recovery'
      f' time at least {_CHANGE_PCT}% lower than the baseline\u2019s is'
      f' green and one at least {_CHANGE_PCT}% higher is red. The <b>Mean'
      ' MTTR</b> cell colors and the <b>N\u00d7 faster/slower MTTR</b> text'
      ' follow the same rule; the baseline itself is not colored.</li>'
      '<li><b>Primary Bottleneck</b> lists the two largest phase'
      ' groups.</li>'
      '<li><b>Step time</b> is the steady-state step time (lower median of'
      ' consecutive step deltas; for elastic workloads, of the steps with'
      ' all slices active); a group row shows the range of its'
      ' workloads.</li>'
      '<li><b>Sub-phases</b>: hover a bar segment in the Recovery anatomy'
      ' chart for the mean, share of MTTR and min/max of each of the 10'
      ' recovery sub-phases in that group, or open the breakdown tables'
      ' under the chart.</li>'
  )
  footnotes = f'<h3>Notes</h3><ul>{"".join(items)}</ul>' if items else ''
  return (
      f'<div class="notes">{footnotes}<h3>Definitions</h3><ul>{definitions}'
      f'</ul><h3>Phase groups</h3><ul class="groups">{groups}</ul></div>'
  )


_CSS = """
    :root {
      --bg: #f8f9fa; --surface: #ffffff; --border: #dadce0;
      --text: #202124; --muted: #5f6368; --accent: #1a73e8;
    }
    * { box-sizing: border-box; }
    body {
      font-family: "Google Sans", Roboto, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
      background: var(--bg); color: var(--text); margin: 0;
      padding: 32px 24px 64px; line-height: 1.5;
    }
    .wrap { max-width: 1640px; margin: 0 auto; }
    header {
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 12px; padding: 24px 28px; margin-bottom: 20px;
      display: flex; justify-content: space-between; align-items: flex-start;
      gap: 20px;
    }
    header h1 { margin: 0 0 6px; font-size: 26px; font-weight: 600; }
    header p { margin: 0; color: var(--muted); font-size: 14.5px; }
    header .meta { text-align: right; color: var(--muted); white-space: nowrap; }
    nav {
      position: sticky; top: 0; z-index: 100;
      background: rgba(255, 255, 255, 0.92); backdrop-filter: blur(8px);
      border: 1px solid var(--border); border-radius: 10px;
      padding: 8px 12px; margin-bottom: 8px;
      display: flex; flex-wrap: wrap; gap: 6px; align-items: center;
      font-size: 13px;
    }
    nav .label {
      font-size: 11.5px; font-weight: 600; text-transform: uppercase;
      letter-spacing: 0.04em; color: var(--muted); margin-right: 4px;
    }
    nav a {
      color: #3c4043; background: #f1f3f4; border: 1px solid var(--border);
      text-decoration: none; padding: 3px 10px; border-radius: 6px;
      white-space: nowrap; font-weight: 500;
    }
    nav a:hover { background: #e8f0fe; color: var(--accent); }
    .section { margin: 36px 0 14px; scroll-margin-top: 56px; }
    .section h2 {
      margin: 0 0 4px; font-size: 20px; font-weight: 600; color: var(--accent);
    }
    .section p { margin: 0; color: var(--muted); font-size: 14px; }
    .legend-bar {
      display: flex; flex-wrap: wrap; gap: 22px; background: var(--surface);
      border: 1px solid var(--border); border-radius: 8px; padding: 10px 16px;
      margin-bottom: 14px; font-size: 13px; color: #3c4043;
    }
    .leg-item { display: flex; align-items: center; gap: 8px; font-weight: 500; }
    .leg-dot { width: 12px; height: 12px; border-radius: 2px; display: inline-block; }
    .table-wrap {
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 12px; overflow-x: auto;
      box-shadow: 0 1px 2px rgba(60,64,67,0.06);
    }
    table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
    th, td {
      padding: 11px 14px; text-align: left; border-bottom: 1px solid #e8eaed;
      vertical-align: middle;
    }
    th { background: #f1f3f4; font-weight: 600; color: #3c4043; white-space: nowrap; }
    tbody tr:last-child td { border-bottom: none; }
    table.summary tr.group-row td {
      background: #e8f0fe; color: #174ea6; font-weight: 700; font-size: 13px;
      padding: 7px 10px;
    }
    table.summary tr.group-note td {
      background: #fef7e0; color: #7a4f01; font-size: 12.5px; padding: 6px 10px;
    }
    .mono { font-family: "Roboto Mono", ui-monospace, monospace; font-size: 12.5px; }
    .g { color: #137333; font-weight: 700; }
    .y { color: #b06000; font-weight: 700; }
    .r { color: #c5221f; font-weight: 700; }
    a { color: var(--accent); text-decoration: none; font-weight: 500; }
    a:hover { text-decoration: underline; }
    .mini-bar {
      width: 170px; height: 12px; background: #f1f3f4; border-radius: 3px;
      overflow: hidden; display: flex; margin-bottom: 3px;
    }
    .mini-seg { height: 100%; }
    .mini-sub { font-size: 11px; color: var(--muted); }
    .footnotes { margin: 8px 4px 0; font-size: 12.5px; color: var(--muted); }
    .plots-grid { display: grid; grid-template-columns: 1fr; gap: 20px; margin-top: 14px; }
    .plot-card {
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 12px; padding: 16px;
      box-shadow: 0 1px 2px rgba(60,64,67,0.06);
    }
    .plot-card h3 {
      margin: 0 0 10px; font-size: 15px; display: flex;
      justify-content: space-between; align-items: center; gap: 12px;
    }
    .plot-card .hint { font-size: 12px; font-weight: 400; color: var(--muted); }
    .chart-scroll { overflow-x: auto; }
    .chip {
      display: inline-block; width: 9px; height: 9px; border-radius: 3px;
      margin-right: 8px; vertical-align: middle;
    }
    .notes {
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 12px; padding: 16px 20px; font-size: 13.5px; color: #3c4043;
    }
    .notes h3 { font-size: 14px; margin: 6px 0 4px; color: var(--text); }
    .notes ul { margin: 0 0 10px 20px; padding: 0; }
    .notes ul.groups { list-style: none; margin-left: 2px; }
    .notes li { margin: 3px 0; }
    .sub {
      font-size: 11.5px; color: var(--muted); font-weight: 400;
      white-space: nowrap; margin-top: 1px;
    }
    .sub.delta { white-space: normal; }
    table.summary th { white-space: normal; vertical-align: bottom; }
    table.summary th, table.summary td { padding: 11px 10px; }
    table.summary td.model, table.summary td.wl { white-space: nowrap; }
    details.breakdown { margin-top: 12px; }
    details.breakdown summary {
      cursor: pointer; font-size: 13px; font-weight: 600; color: var(--accent);
    }
    details.breakdown .table-wrap { margin-top: 10px; box-shadow: none; }
    table.cmp { font-size: 13px; }
    table.cmp th, table.cmp td { padding: 7px 12px; }
    table.cmp th { vertical-align: bottom; }
    table.cmp .num { text-align: right; font-variant-numeric: tabular-nums; }
    table.cmp tr.subtotal td { background: #f8f9fa; font-weight: 600; }
    table.cmp tr.total td {
      background: #e8f0fe; color: #174ea6; font-weight: 700;
      border-top: 2px solid var(--accent);
    }
"""

_PAGE_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>{css}</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1>{title}</h1>
      <p>{subtitle}</p>
    </div>
    <div class="meta mono">
      Tool: <b>{tool}</b><br>{badge}
    </div>
  </header>
{body}
</div>
</body>
</html>
"""


def _section_heading(section_id: str, title: str, text: str) -> str:
  para = f'<p>{text}</p>' if text else ''
  return (
      f'<div class="section" id="{section_id}"><h2>{_esc(title)}</h2>{para}'
      '</div>'
  )


def _chart_html(
    group: SummaryGroup,
    title: str,
    subtitle: str,
    markers_by_id: dict[str, str],
    include_plotlyjs: Union[bool, str],
) -> str:
  """One Recovery anatomy chart card for a group of workloads.

  Under the chart, a collapsed section has the group's phase-group and
  sub-phase tables.

  Args:
    group: The workloads of the chart.
    title: Chart title for an unnamed group.
    subtitle: Chart subtitle for a group without description.
    markers_by_id: Footnote marker per workload id.
    include_plotlyjs: Passed to plotly.

  Returns:
    The chart card HTML.
  """
  fig = build_comparison_figure(
      group.summaries,
      title=group.name or title,
      subtitle=group.description or subtitle,
  )
  chart = fig.to_html(
      full_html=False,
      include_plotlyjs=include_plotlyjs,
      config={
          'displaylogo': False,
          'toImageButtonOptions': {
              'format': 'png',
              'filename': (
                  'goodput_recovery_anatomy_'
                  + re.sub(r'[^0-9A-Za-z]+', '_', group.name or 'all').strip(
                      '_'
                  )
              ),
              'scale': 2,
          },
      },
  )
  markers = [markers_by_id.get(s.workload_id, '') for s in group.summaries]
  breakdown = (
      '<details class="breakdown"><summary>Breakdown tables: phase groups and'
      ' the 10 recovery sub-phases, mean per incident</summary>'
      f'{_group_table_html(group.summaries, markers)}'
      f'{_phase_table_html(group.summaries, markers)}</details>'
  )
  n = len(group.summaries)
  return (
      f'<div class="plot-card"><h3><span>{_esc(group.name or "All workloads")}'
      f' \u00b7 {n} workload{"s" if n != 1 else ""}</span><span class="hint">'
      'Mean per incident \u00b7 hover a bar for the sub-phase breakdown'
      ' \u00b7 camera icon downloads a PNG</span></h3>'
      f'<div class="chart-scroll">{chart}</div>{breakdown}</div>'
  )


def to_comparison_html(
    summaries: Sequence[WorkloadSummary],
    title: str = '',
    subtitle: str = '',
    description: str = '',
    groups: Sequence[GroupInfo] = (),
    include_plotlyjs: Union[bool, str] = True,
) -> str:
  """Returns a standalone HTML page comparing the workloads.

  The page has the cross-run comparison table (a Model column and a row per
  group; each group compares with its baseline), one Recovery anatomy chart
  per group (its hover lists the sub-phases, and collapsed tables under it
  list the phase groups and sub-phases) and the metric definitions.

  Args:
    summaries: The workloads.
    title: Page title.
    subtitle: Text under the title.
    description: Paragraph above the comparison table.
    groups: Group order and descriptions.
    include_plotlyjs: Passed to plotly for the first chart; later charts reuse
      it.

  Returns:
    The HTML page.
  """
  if not summaries:
    raise ValueError('No workloads to compare.')
  title = title or _DEFAULT_TITLE
  subtitle = subtitle or _DEFAULT_SUBTITLE
  markers = _note_markers(summaries)
  markers_by_id = {s.workload_id: m for s, m in zip(summaries, markers)}
  summary_groups = group_summaries(summaries, groups)
  n = len(summaries)

  plot_cards = [
      _chart_html(
          g,
          title,
          subtitle,
          markers_by_id,
          include_plotlyjs if i == 0 else False,
      )
      for i, g in enumerate(summary_groups)
  ]
  nav = (
      '<nav><span class="label">Jump to:</span>'
      '<a href="#table">Comparison table</a>'
      '<a href="#chart">Recovery anatomy</a>'
      '<a href="#notes">Definitions</a></nav>'
  )
  table_text = description or (
      f'{n} workload{"s" if n != 1 else ""}; MTTR, Time to Recover and'
      ' Lost Progress are means per recovery incident. Changes and MTTR'
      ' colors compare with the baseline of each group.'
  )
  body = '\n'.join([
      nav,
      _section_heading(
          'table', '1. Cross-Run Comparison Table', _esc(table_text)
      ),
      _legend_bar_html(),
      _summary_table_html(summary_groups, markers_by_id),
      _footnotes_html(summaries, markers),
      _section_heading(
          'chart',
          '2. Recovery Anatomy',
          'Mean recovery time per incident, stacked by phase group (A), and'
          ' each group\u2019s share of MTTR (B). Hover a bar segment for the'
          ' sub-phase breakdown, or open the breakdown tables under a'
          ' chart.',
      ),
      f'<div class="plots-grid">{"".join(plot_cards)}</div>',
      _section_heading('notes', '3. Notes & Definitions', ''),
      _notes_html(summaries, markers),
  ])
  generated = datetime.datetime.now(datetime.timezone.utc).strftime(
      '%Y-%m-%d %H:%M UTC'
  )
  return _PAGE_TEMPLATE.format(
      title=_esc(title),
      subtitle=_esc(subtitle),
      css=_CSS,
      tool=_TOOL_NAME,
      badge=f'{n} workloads \u00b7 generated {generated}',
      body=body,
  )


def output_file(path: Optional[str], default_name: str) -> str:
  """Returns the file to write and creates its directory if needed.

  Args:
    path: A file name, kept as is; a directory (one that exists, or a path
      ending with a slash), where `default_name` is written; or None, for
      `default_name` in the current directory.
    default_name: The file name used when `path` is not a file name.
  """
  if not path:
    path = default_name
  elif path.endswith(os.sep) or os.path.isdir(path):
    path = os.path.join(path, default_name)
  os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
  return path


def to_comparison_html_file(
    summaries: Sequence[WorkloadSummary],
    path: str,
    title: str = '',
    subtitle: str = '',
    description: str = '',
    groups: Sequence[GroupInfo] = (),
) -> None:
  """Writes the comparison page to a standalone HTML file."""
  page = to_comparison_html(
      summaries,
      title=title,
      subtitle=subtitle,
      description=description,
      groups=groups,
  )
  with open(path, 'w', encoding='utf-8') as f:
    f.write(page)


def csv_rows(
    summaries: Sequence[WorkloadSummary], groups: Sequence[GroupInfo] = ()
) -> list[dict[str, Any]]:
  """Returns one flat dict per workload; durations are in seconds.

  Args:
    summaries: The workloads.
    groups: Group order and descriptions; see group_summaries.

  Returns:
    The rows; is_group_baseline marks the baseline of each group.
  """
  baselines = [g.baseline for g in group_summaries(summaries, groups)]
  rows = []
  for s in summaries:
    stats = s.rec_stats
    row: dict[str, Any] = {
        'label': s.label,
        'workload_id': s.workload_id,
        'model': s.model,
        'group': s.group,
        'is_group_baseline': any(s is b for b in baselines),
        'timeline_url': s.timeline_url,
        'shape': s.shape,
        'note': s.note,
        'num_incidents': s.num_incidents,
        'goodput_pct': '' if s.goodput_pct is None else round(s.goodput_pct, 3),
        'total_job_time_s': round(s.total_job_time_s, 3),
        'last_step': s.last_step,
        'baseline_step_time_s': round(s.baseline_step_time_s, 3),
        'mtbi_s': round(s.mtbi_s, 3),
        'mean_mttr_s': round(s.mean_mttr_s, 3),
        'min_mttr_s': round(stats.min_mttr_s, 3),
        'max_mttr_s': round(stats.max_mttr_s, 3),
        'mean_time_to_recover_s': round(s.mean_time_to_recover_s, 3),
        'mean_downtime_s': round(stats.mean_downtime_s, 3),
        'mean_lost_progress_s': round(s.mean_lost_progress_s, 3),
        'mean_rolled_back_steps': round(stats.mean_rolled_back_steps, 3),
    }
    for g in PHASE_GROUPS:
      row[f'group_{g.key}_mean_s'] = round(s.group_mean_s(g), 3)
    for g in PHASE_GROUPS:
      row[f'group_{g.key}_pct_of_mttr'] = round(s.group_share_pct(g), 3)
    for phase in metrics.RECOVERY_PHASE_ORDER:
      row[f'phase_{phase}_mean_s'] = round(s.phase_mean_s(phase), 3)
    rows.append(row)
  return rows


def to_csv_file(
    summaries: Sequence[WorkloadSummary],
    path: str,
    groups: Sequence[GroupInfo] = (),
) -> None:
  """Writes csv_rows() to a CSV file."""
  rows = csv_rows(summaries, groups)
  with open(path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
    writer.writeheader()
    writer.writerows(rows)


def format_text_table(summaries: Sequence[WorkloadSummary]) -> str:
  """Returns a plain-text table of the headline metrics for the terminal."""
  header = [
      'Workload',
      'Incidents',
      'Goodput',
      'MTBI',
      'MTTR',
      'Time to Recover',
      'Lost Progress',
  ]
  table = [header]
  for s in summaries:
    has_inc = s.num_incidents > 0
    table.append([
        s.label,
        str(s.num_incidents),
        _fmt_goodput(s),
        _fmt_duration(s.mtbi_s) if s.mtbi_s > 0 else '-',
        _fmt_duration(s.mean_mttr_s) if has_inc else '-',
        _fmt_duration(s.mean_time_to_recover_s) if has_inc else '-',
        _fmt_duration(s.mean_lost_progress_s) if has_inc else '-',
    ])
  widths = [max(len(row[c]) for row in table) for c in range(len(header))]
  lines = []
  for row in table:
    cells = [row[0].ljust(widths[0])]
    cells += [v.rjust(w) for v, w in zip(row[1:], widths[1:])]
    lines.append('  '.join(cells))
  return '\n'.join(lines)
