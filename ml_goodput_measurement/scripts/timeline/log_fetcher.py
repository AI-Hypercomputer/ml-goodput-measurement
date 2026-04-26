# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Optional

import google.cloud.logging
from google.cloud.logging_v2.services.logging_service_v2 import (
    LoggingServiceV2Client,
)
from google.cloud.logging_v2.types import ListLogsRequest

_WORKLOAD_ID_RE = re.compile(r'^[\w\-]+$')


def list_workloads(project_id: str) -> list[str]:
  client = LoggingServiceV2Client()
  request = ListLogsRequest(parent=f'projects/{project_id}')
  prefix = f'projects/{project_id}/logs/goodput_'
  return sorted(
      name[len(prefix):]
      for name in client.list_logs(request=request)
      if name.startswith(prefix)
  )


def fetch_logs(
    project_id: str,
    workload_id: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> list[dict]:
  if not _WORKLOAD_ID_RE.match(workload_id):
    raise ValueError(
        f'Invalid workload_id {workload_id!r}'
    )

  log_name = f'projects/{project_id}/logs/goodput_{workload_id}'
  filters = [f'logName="{log_name}"']
  if start_time:
    filters.append(f'timestamp>="{start_time}"')
  if end_time:
    filters.append(f'timestamp<="{end_time}"')

  client = google.cloud.logging.Client(project=project_id)
  entries = []
  for entry in client.list_entries(
      filter_=' AND '.join(filters),
      order_by=google.cloud.logging.ASCENDING,
  ):
    payload = entry.payload
    if isinstance(payload, dict):
      entries.append(payload)
  return entries
