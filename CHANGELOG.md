Change Log
==========

Versioned according to [Semantic Versioning](http://semver.org/).

## Unreleased

Changed:

  * `make docker`: Overrideable `$(DOCKER)` command and base on `latest` OCR-D/core, #124

## [2.0.1] - 2025-04-03

Fixed:

  - Remove duplicate `DOCKER_BASE_IMAGE` assignment, #123

## [2.0.0] - 2025-03-31

Changed:

  - :fire: Upgrade calamari API to v2, #118
  - :fire: Upgrade OCR-D API to v3, #118
  - :fire: Switch license from Apache 2.0 to GPLv3 for compatibility with Planet-AI-GmbH/tfaip#12, #118
  - Dockerfile: Pre-package ocrd-all-tool.json and update metadata, #118
  - CI: Use smaller calamari2-compatible model, #118
  - `ocrd-tool.json`: param `checkpoint_dir` obligatory now, #118
  - `ocrd-tool.json`: update and extend recognition model `resources`, including versioning metadata, #118
  - Processor API: Use background pipeline to efficiently batch processing tasks, #118
  - Tests: Test with all configurations of OCR-D/core's METS caching/page-parallel processing, #118

Removed:

  - obsolete `fix-calamari1-models` script, #118


<!-- link-labels -->
[2.0.1]: ../../compare/v2.0.1...v2.0.0
[2.0.0]: ../../compare/HEAD...v2.0.0
