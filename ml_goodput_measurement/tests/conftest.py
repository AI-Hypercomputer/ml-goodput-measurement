"""Pytest configuration for GitHub CI."""
import pathlib
import sys
import types

_cloud_goodput = types.ModuleType('cloud_goodput')
_cloud_goodput.__path__ = [str(pathlib.Path(__file__).parent.parent.parent)]
_cloud_goodput.__package__ = 'cloud_goodput'
sys.modules['cloud_goodput'] = _cloud_goodput

from absl.testing import absltest as _absltest  # pylint: disable=g-import-not-at-top

_g3 = types.ModuleType('google3')
_g3_testing = types.ModuleType('google3.testing')
_g3_pybase = types.ModuleType('google3.testing.pybase')
_g3_pybase.googletest = _absltest
sys.modules.update({
    'google3': _g3,
    'google3.testing': _g3_testing,
    'google3.testing.pybase': _g3_pybase,
})
