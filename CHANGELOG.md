# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Most recent change on the bottom.

## [Unreleased]

## [0.5.2]
### Added
- `-e` option to `patch_lammps.sh`
- PyTorch 1.11 / fusion strategy support

## [0.5.1]
### Added
- Full suite of unit tests

### Fixed
- edge duplication issue

## [0.5.0] - 11/24/2021

### Changed
- The mapping from LAMMPS to NequIP types is now explicitly provided
- Don't print full config in LAMMPS log

### Fixed
- Edges outside the cutoff but within the skin are no longer processed by the model
- `allow_tf32` and `_jit_bailout_depth` are now respected
- Hack to allow freezing on demand for PyTorch < 1.10

## [0.1.0]