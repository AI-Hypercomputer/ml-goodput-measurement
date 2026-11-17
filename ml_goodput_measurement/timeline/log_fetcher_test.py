"""Tests for log_fetcher."""

import datetime
import types
from typing import Any, Optional
from unittest import mock

from absl.testing import absltest
from google.cloud.logging_v2 import types as logging_types
from google.cloud.logging_v2.services import logging_service_v2

from cloud_goodput.ml_goodput_measurement.timeline import log_fetcher


def _entry(
    minute: int,
    payload: Optional[dict[str, Any]] = None,
    text: Optional[str] = None,
) -> logging_types.LogEntry:
  """A log entry at 2026-08-01 00:<minute> UTC with a JSON or text payload."""
  timestamp = datetime.datetime(
      2026, 8, 1, 0, minute, tzinfo=datetime.timezone.utc
  )
  if text is not None:
    return logging_types.LogEntry(text_payload=text, timestamp=timestamp)
  return logging_types.LogEntry(json_payload=payload, timestamp=timestamp)


class _FakeClient:
  """Serves pages of log entries by page token, like LoggingServiceV2Client.

  A response only has the fields that log_fetcher reads. A page with errors
  raises them first, one per request.
  """

  def __init__(
      self,
      pages: dict[str, tuple[list[logging_types.LogEntry], str]],
      errors: Optional[dict[str, list[Exception]]] = None,
  ):
    self._pages = pages  # Page token -> (entries, next page token).
    self._errors = errors or {}  # Page token -> errors to raise first.
    self.requests = []

  def list_log_entries(self, request):
    self.requests.append({
        'resource_names': list(request.resource_names),
        'filter': request.filter,
        'order_by': request.order_by,
        'page_size': request.page_size,
        'page_token': request.page_token,
    })
    errors = self._errors.get(request.page_token)
    if errors:
      raise errors.pop(0)
    entries, next_page_token = self._pages[request.page_token]
    return types.SimpleNamespace(
        entries=entries, next_page_token=next_page_token
    )


class FetchLogEntriesTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.sleep = self.enter_context(
        mock.patch.object(log_fetcher.time, 'sleep')
    )

  def _fetch(self, client, *args):
    with mock.patch.object(
        logging_service_v2, 'LoggingServiceV2Client', return_value=client
    ):
      return log_fetcher.fetch_log_entries('proj', 'wl', *args)

  def test_reads_all_pages(self):
    client = _FakeClient({
        '': ([_entry(0, {'a': 1.0}), _entry(1, text='hello')], 'p2'),
        'p2': ([_entry(2, {'b': 'x', 'c': None})], ''),
    })
    entries = self._fetch(
        client, '2026-08-01T00:00:00Z', '2026-08-02T00:00:00Z'
    )
    # The text payload is skipped.
    self.assertEqual(
        entries,
        [
            ('2026-08-01T00:00:00+00:00', {'a': 1.0}),
            ('2026-08-01T00:02:00+00:00', {'b': 'x', 'c': None}),
        ],
    )
    self.assertEqual([r['page_token'] for r in client.requests], ['', 'p2'])
    self.assertEqual(
        client.requests[0],
        {
            'resource_names': ['projects/proj'],
            'filter': (
                '(logName="projects/proj/logs/goodput_wl" OR'
                ' logName="projects/proj/logs/goodput_pw-wl")'
                ' AND timestamp>="2026-08-01T00:00:00Z"'
                ' AND timestamp<="2026-08-02T00:00:00Z"'
                ' AND NOT jsonPayload."logging.googleapis.com/diagnostic":*'
            ),
            'order_by': 'timestamp asc',
            'page_size': 1000,
            'page_token': '',
        },
    )
    self.sleep.assert_not_called()

  def test_quota_error_retries_only_that_page(self):
    quota = RuntimeError('429 Quota exceeded')
    client = _FakeClient(
        {
            '': ([_entry(0, {'a': 1.0})], 'p2'),
            'p2': ([_entry(1, {'b': 2.0})], ''),
        },
        errors={'': [quota], 'p2': [quota]},
    )
    entries = self._fetch(client)
    self.assertEqual([p for _, p in entries], [{'a': 1.0}, {'b': 2.0}])
    # A failed page is read again after a wait; earlier pages are kept.
    self.assertEqual(
        [r['page_token'] for r in client.requests], ['', '', 'p2', 'p2']
    )
    self.assertEqual([c.args[0] for c in self.sleep.call_args_list], [5, 5])

  def test_quota_retries_give_up(self):
    client = _FakeClient(
        {'': ([], '')}, errors={'': [RuntimeError('429 Quota exceeded')] * 4}
    )
    with self.assertRaisesRegex(RuntimeError, 'Quota'):
      self._fetch(client)
    self.assertLen(client.requests, 4)
    self.assertEqual(
        [c.args[0] for c in self.sleep.call_args_list], [5, 10, 20]
    )

  def test_other_errors_are_not_retried(self):
    client = _FakeClient(
        {'': ([], '')}, errors={'': [RuntimeError('504 Deadline Exceeded')]}
    )
    with self.assertRaisesRegex(RuntimeError, 'Deadline'):
      self._fetch(client)
    self.assertLen(client.requests, 1)
    self.sleep.assert_not_called()

  def test_fetch_logs_returns_payloads(self):
    client = _FakeClient({'': ([_entry(0, {'a': 1.0})], '')})
    with mock.patch.object(
        logging_service_v2, 'LoggingServiceV2Client', return_value=client
    ):
      self.assertEqual(log_fetcher.fetch_logs('proj', 'wl'), [{'a': 1.0}])

  def test_invalid_workload_id(self):
    with self.assertRaisesRegex(ValueError, 'Invalid workload_id'):
      log_fetcher.fetch_log_entries('proj', 'wl" OR logName:"x')


if __name__ == '__main__':
  absltest.main()
