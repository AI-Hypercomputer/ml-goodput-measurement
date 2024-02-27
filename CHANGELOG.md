# Changelog

<!--

Changelog follow the https://keepachangelog.com/ standard (at least the headers)

This allows:

* auto-parsing release notes during the automated releases from github-action:
  https://github.com/marketplace/actions/pypi-github-auto-release
* Have clickable headers in the rendered markdown

To release a new version (e.g. from `1.0.0` -> `2.0.0`):

* Create a new `# [0.0.1] - YYYY-MM-DD` header and add the current
  `[Unreleased]` notes.
* At the end of the file:
  * Define the new link url:
  `[0.0.1]: https://github.com/google/cloud_tpu_goodput/compare/v1.0.0...v2.0.0`
  * Update the `[Unreleased]` url: `v1.0.0...HEAD` -> `v2.0.0...HEAD`

-->

## [Unreleased]

## [0.0.1] - 2024-02-26
* Initial release of ML Goodput Measurement PyPi package
* Feature: Contains the Goodput module which allows logging and retrieval of training job's overall productive Goodput

[1.0.0]: https://github.com/google/cloud_tpu_goodput/releases/tag/v1.0.0