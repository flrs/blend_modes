# Changelog
All notable changes to this project will be documented in this file.

## [2.1.0] - 2019-07-16
### Added
- Added images in blending function documentation

### Changed
- Dropped support for Python versions < 3.5
- Refactored internal package structure, splitting type checks and image blending functions into separate modules
- Improved error messages for type checks to be more user-friendly
- Improved visual example gallery in documentation
- Ported tests from unittest framework to pytest framework
- Updated links in readme file

### Fixed
- Image blending functions no longer modify their inputs

## [2.0.2] - 2019-04-17
### Fixed
- Fixed bug where it would not be possible to import the package

## [2.0.1] - 2019-03-09
### Fixed
- Bug fix in setup.py for package creation

## [2.0.0] - 2019-03-09
### Added
- Added changelog

### Changed
- Changed overlay mode to deliver Adobe-style results. Use soft light mode for backwards compatibility.
- Changed docstrings to Google style.
- Improved API to make blending methods more accessible