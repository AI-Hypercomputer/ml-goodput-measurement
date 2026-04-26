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
import argparse
import sys

from . import log_fetcher, event_parser, builder, renderer


def _cmd_list(args: argparse.Namespace) -> None:
  workloads = log_fetcher.list_workloads(args.project_id)
  if not workloads:
    print('No workloads with goodput logs found.')
    return
  print(f'Found {len(workloads)} workload(s) with goodput logs in {args.project_id}:')
  for w in workloads:
    print(f' {w}')


def _cmd_export(args: argparse.Namespace) -> None:
  print(f'Fetching logs for {args.workload_id!r}')
  entries = log_fetcher.fetch_logs(
      args.project_id, args.workload_id, args.start_time, args.end_time
  )
  if not entries:
    print('No log entries found. Check project_id, workload_id, and time range.')
    sys.exit(1)
  print(f'  {len(entries)} entries retrieved.')

  parsed = event_parser.parse_entries(entries)

  segments = builder.build_timeline(parsed)
  renderer.to_html_file(segments, args.workload_id, args.output)
  print(f'Timeline written to {args.output}')


def main() -> None:
  parser = argparse.ArgumentParser(
      prog='goodput-timeline',
      description='Goodput Timeline Analyzer',
  )
  sub = parser.add_subparsers(dest='command', required=True)

  p_list = sub.add_parser('list', help='List workloads that have goodput logs')
  p_list.add_argument('--project_id', required=True, help='GCP project ID')

  p_export = sub.add_parser('export', help='Export timeline to a standalone HTML file')
  p_export.add_argument('--project_id', required=True)
  p_export.add_argument(
      '--workload_id', required=True, help='Workload ID or run name'
  )
  p_export.add_argument(
      '--start_time', default=None, help='ISO 8601 UTC.'
  )
  p_export.add_argument('--end_time', default=None, help='ISO 8601 UTC')
  p_export.add_argument('--output', default='goodput_timeline.html')

  args = parser.parse_args()
  {'list': _cmd_list, 'export': _cmd_export}[args.command](args)


if __name__ == '__main__':
  main()
