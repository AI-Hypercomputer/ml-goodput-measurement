"""Cloud Logging query helpers for Goodput Timeline workloads."""

import collections
import datetime
import re
import time
from typing import Any, Optional
import google.cloud.logging
import google.cloud.logging_v2.services.logging_service_v2
import google.cloud.logging_v2.types

_WORKLOAD_ID_RE = re.compile(r'^[\w\-]+$')

# Waits (seconds) before retrying a Cloud Logging read that hit the read
# quota (60 entries.list requests per minute per project).
QUOTA_RETRY_WAITS_S = (5, 10, 20)

# The Cloud Logging client library adds an entry with only this payload key
# (its name and version) to the goodput logs; it has no goodput data.
_NO_DIAGNOSTIC_FILTER = 'NOT jsonPayload."logging.googleapis.com/diagnostic":*'


def is_quota_error(e: Exception) -> bool:
  """True for Cloud Logging rate limit and quota errors (HTTP 429)."""
  names = {cls.__name__ for cls in type(e).__mro__}
  if names & {'ResourceExhausted', 'TooManyRequests'}:
    return True
  text = str(e)
  return (
      text.startswith('429') or 'RateLimitExceeded' in text or 'Quota' in text
  )


def list_workloads(
    project_id: str, credentials: Optional[Any] = None
) -> list[str]:
  """Lists workload IDs with goodput logs in the given GCP project."""
  client = google.cloud.logging_v2.services.logging_service_v2.LoggingServiceV2Client(
      credentials=credentials
  )
  request = google.cloud.logging_v2.types.ListLogsRequest(
      parent=f'projects/{project_id}'
  )
  prefix = f'projects/{project_id}/logs/goodput_'
  workloads = set()
  for name in client.list_logs(request=request):
    if name.startswith(prefix):
      workload_id = name[len(prefix) :]
      if workload_id.startswith('pw-'):
        workload_id = workload_id[3:]
      workloads.add(workload_id)
  return sorted(list(workloads))


def fetch_log_entries(
    project_id: str,
    workload_id: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    credentials: Optional[Any] = None,
) -> list[tuple[str, dict[str, Any]]]:
  """Fetches the goodput log entries of a workload with their timestamps.

  Reads logs goodput_<workload_id> and goodput_pw-<workload_id>, oldest entry
  first, one page of 1,000 entries per request. A page that hits the Cloud
  Logging read quota is read again after a wait, so the pages already read
  are kept. Entries without a JSON payload are skipped.

  Args:
    project_id: GCP project.
    workload_id: Workload ID.
    start_time: Optional start time, e.g. '2026-08-21T18:00:00-07:00'.
    end_time: Optional end time.
    credentials: Optional credentials.

  Returns:
    (timestamp, payload) per entry; the timestamp is ISO 8601, '' if unset.
  """
  if not _WORKLOAD_ID_RE.match(workload_id):
    raise ValueError(
        f'Invalid workload_id {workload_id!r}'
    )

  log_name = f'projects/{project_id}/logs/goodput_{workload_id}'
  log_name_pw = f'projects/{project_id}/logs/goodput_pw-{workload_id}'
  filters = [f'(logName="{log_name}" OR logName="{log_name_pw}")']
  if start_time:
    filters.append(f'timestamp>="{start_time}"')
  if end_time:
    filters.append(f'timestamp<="{end_time}"')
  filters.append(_NO_DIAGNOSTIC_FILTER)

  print(f'Querying goodput logs with filter: {" AND ".join(filters)}')
  client = google.cloud.logging_v2.services.logging_service_v2.LoggingServiceV2Client(
      credentials=credentials
  )
  request = google.cloud.logging_v2.types.ListLogEntriesRequest(
      resource_names=[f'projects/{project_id}'],
      filter=' AND '.join(filters),
      order_by=google.cloud.logging.ASCENDING,
      page_size=1000,
  )
  log_entry = google.cloud.logging_v2.types.LogEntry
  entries = []
  page_number = 1
  while True:
    page = _list_page(client, request, workload_id, page_number)
    for entry in page.entries:
      if not log_entry.pb(entry).HasField('json_payload'):
        continue
      timestamp = entry.timestamp.isoformat() if entry.timestamp else ''
      entries.append((timestamp, log_entry.to_dict(entry)['json_payload']))
    if not page.next_page_token:
      return entries
    request.page_token = page.next_page_token
    page_number += 1


def _list_page(
    client: Any, request: Any, workload_id: str, page_number: int
) -> Any:
  """One entries.list request; waits and retries it on quota errors."""
  for attempt, wait_s in enumerate(QUOTA_RETRY_WAITS_S, start=1):
    try:
      # Only the first page of the returned pager is used: one request.
      return client.list_log_entries(request=request)
    except Exception as e:  # pylint: disable=broad-exception-caught
      if not is_quota_error(e):
        raise
      print(
          f'  {workload_id}: Cloud Logging quota exceeded on page'
          f' {page_number}; retrying it in {wait_s}s'
          f' ({attempt}/{len(QUOTA_RETRY_WAITS_S)}).'
      )
      time.sleep(wait_s)
  return client.list_log_entries(request=request)


def fetch_logs(
    project_id: str,
    workload_id: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    credentials: Optional[Any] = None,
) -> list[dict[str, Any]]:
  """Fetches goodput log entries for a single workload ID."""
  return [
      payload
      for _, payload in fetch_log_entries(
          project_id, workload_id, start_time, end_time, credentials
      )
  ]


def _utc(t: datetime.datetime) -> str:
  return t.astimezone(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def fetch_jobset_create(
    project_id: str,
    workload_id: str,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    credentials: Optional[Any] = None,
) -> Optional[dict[str, Any]]:
  """Returns the audit log entry that created the JobSet of a workload.

  Searches the Admin Activity audit log of the project, newest first, for a
  successful jobsets.create call of a JobSet named <workload_id> or
  pw-<workload_id> between start_time and end_time. The entry's request has
  the JobSet spec.

  Args:
    project_id: GCP project of the GKE cluster.
    workload_id: Workload ID, i.e. the JobSet name with or without 'pw-'.
    start_time: Start of the search window.
    end_time: End of the search window.
    credentials: Optional credentials.

  Returns:
    The protoPayload of the newest matching entry, or None.
  """
  if not _WORKLOAD_ID_RE.match(workload_id):
    raise ValueError(f'Invalid workload_id {workload_id!r}')
  names = (workload_id, f'pw-{workload_id}')
  activity = f'projects/{project_id}/logs/cloudaudit.googleapis.com%2Factivity'
  filters = [
      'resource.type="k8s_cluster"',
      f'logName="{activity}"',
      'protoPayload.methodName:"jobsets.create"',
      '('
      + ' OR '.join(f'protoPayload.resourceName:"/jobsets/{n}"' for n in names)
      + ')',
      f'timestamp>="{_utc(start_time)}"',
      f'timestamp<="{_utc(end_time)}"',
  ]
  client = google.cloud.logging.Client(
      project=project_id, credentials=credentials
  )
  # resourceName:"/jobsets/x" also matches JobSets named e.g. x-2.
  for entry in client.list_entries(
      filter_=' AND '.join(filters),
      order_by=google.cloud.logging.DESCENDING,
      page_size=50,
      max_results=200,
  ):
    proto = entry.to_api_repr().get('protoPayload') or {}
    name = str(proto.get('resourceName', '')).rsplit('/', 1)[-1]
    if name in names and not (proto.get('status') or {}).get('code'):
      return proto
  return None


def fetch_all_logs(
    project_id: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    credentials: Optional[Any] = None,
) -> dict[str, list[dict[str, Any]]]:
  """Fetches goodput log entries for all workloads in the given GCP project."""
  filters = [f'logName =~ "projects/{project_id}/logs/goodput_.*"']
  if start_time:
    filters.append(f'timestamp>="{start_time}"')
  if end_time:
    filters.append(f'timestamp<="{end_time}"')

  print(f'Querying all goodput logs with filter: {" AND ".join(filters)}')
  client = google.cloud.logging.Client(
      project=project_id, credentials=credentials
  )

  prefix = f'projects/{project_id}/logs/goodput_'
  grouped_entries = collections.defaultdict(list)

  retries = 0
  page_token = None
  iterator = None

  while True:
    try:
      iterator = client.list_entries(
          filter_=' AND '.join(filters),
          order_by=google.cloud.logging.ASCENDING,
          page_size=1000,
          page_token=page_token,
      )

      for entry in iterator:
        log_name = entry.log_name
        if log_name.startswith(prefix):
          workload_id = log_name[len(prefix) :]
          payload = entry.payload
          if isinstance(payload, dict):
            if payload.get('job_name'):
              workload_id = payload['job_name']
            elif payload.get('jobset_name'):
              workload_id = payload['jobset_name']

          if workload_id and workload_id.startswith('pw-'):
            workload_id = workload_id.replace('pw-', '', 1)

          if isinstance(payload, dict):
            grouped_entries[workload_id].append(payload)

      break  # Done iterating all pages
    except Exception as e:
      if 'RateLimitExceeded' in str(e) or '429' in str(e) or 'Quota' in str(e):
        if retries > 5:
          print('Giving up after 5 retries due to quota issues...')
          raise e
        sleep_time = (2**retries) * 5
        print(f'Hit Cloud Logging Quota limit. Retrying in {sleep_time}s...')
        time.sleep(sleep_time)
        retries += 1
        page_token = getattr(iterator, 'next_page_token', None)
      else:
        raise e

  return dict(grouped_entries)
