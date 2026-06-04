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

"""Pytest configuration for GitHub CI."""
import pathlib
import sys
import types

_cloud_goodput = types.ModuleType('cloud_goodput')
_cloud_goodput.__path__ = [str(pathlib.Path(__file__).parent.parent.parent)]
_cloud_goodput.__package__ = 'cloud_goodput'
sys.modules['cloud_goodput'] = _cloud_goodput

from absl.testing import absltest as _absltest

_g3 = types.ModuleType('google3')
_g3_testing = types.ModuleType('google3.testing')
_g3_pybase = types.ModuleType('google3.testing.pybase')
_g3_pybase.googletest = _absltest
sys.modules.update({
    'google3': _g3,
    'google3.testing': _g3_testing,
    'google3.testing.pybase': _g3_pybase,
})
