# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [r1.1.0] - 2024-08-20

### Added

- DE TN Electronic recognizes social media handles `@Nvidia` (#177) 
- Japanese ITN Cardinal, Date, Decimal, Fraction, Ordinal, Time and Whitelist coverage (#141)

### Fixed

- Fixed Fr TN Electronic processing issue from (#166) (#181)
- Fixed En TN Electronic processing issue from (#166) (#185) (#206) (#207)
- Fixed It TN Electronic processing issue from (#166) (#183)
- Fixed Hu TN Electronic processing issue from (#166) (#184)


### Changed

- De TN Time coverage allows full stop delineation (e.g. `2.10h`) (#177)
- Es-En TN weights and data updated for unified (PnC) asr models (#143)
- En TN Expands coverage for technical terms for TTS processing (#167)
- En TN Money supports 'per unit' demarcation (e.g. `$20 per anum`) (#213)


## [r1.0.2] - 2024-05-03

### Added

- Sentence level ZH (Mandarin Chinese) TN (#112)
- Enabled post-processing support for Sparrowhawk TN test (#147) 

### Fixed

- `normalize_with_audio` text-field variable changed (#153)
- `run_evaluate` script for ITN updated for additional languages and casing (#164)

### Changed

- Docstring update (#157)


### Removed

- Removed unused function from AR (Arabic) TN decimals (#165)

