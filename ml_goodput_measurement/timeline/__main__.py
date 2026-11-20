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

# `python -m` needs this module name.
# pylint: disable=invalid-name
"""Goodput timeline: renders and compares the goodput logs of workloads.

Run it with python -m ml_goodput_measurement.timeline; see README.md.
"""

import sys

# The imports are guarded to say which extra to install when one is missing.
# pylint: disable=g-import-not-at-top
try:
  from absl import app
  from absl import flags
  from cloud_goodput.ml_goodput_measurement.timeline import main
except ModuleNotFoundError as e:
  # pandas, plotly and absl-py come with the timeline extra, not the library.
  if e.name and e.name.split('.')[0] in ('absl', 'pandas', 'plotly'):
    sys.exit(
        f'{e}. The goodput timeline needs the timeline extra:'
        ' pip install "ml-goodput-measurement[timeline]"'
    )
  raise
# pylint: enable=g-import-not-at-top

# --help lists the flags of the tool, which main defines.
flags.adopt_module_key_flags(main)


if __name__ == '__main__':
  app.run(main.main)
