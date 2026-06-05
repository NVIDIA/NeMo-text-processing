# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [r1.2.0] - 2026-06-05

### Added

- Hi (Hindi) TN: Cardinal, Decimal, Fraction, Ordinal, Date, Time, Money, Measure, Telephone and Electronic semiotic classes (#241) (#271) (#346) (#380) (#414)
- Hi (Hindi) ITN: Cardinal, Decimal, Ordinal, Fraction, Date, Time, Money, Measure, Whitelist, Word, Telephone and Address semiotic classes (#223) (#248) (#266) (#377) (#380)
- Hi-En (Hindi-English) Code-Switched TN (#415)
- Vi (Vietnamese) TN: Cardinal, Ordinal, Decimal, Date, Time, Fraction, Measure, Money, Telephone and Electronic semiotic classes (#338)
- Vi (Vietnamese) ITN: Cardinal, Ordinal, Decimal, Date, Time, Fraction, Measure, Money, Telephone and Electronic semiotic classes (#338)
- He (Hebrew) ITN (#366) (#367)
- Ko (Korean) ITN: Cardinal, Decimal, Ordinal, Fraction, Date, Time, Money, Measure, Telephone and Whitelist semiotic classes (#400)
- Ko (Korean) TN: Post-processing rules for particle agreement and month handling (#413)
- Pt-Br (Portuguese Brazilian) TN: Cardinal, Ordinal, Decimal, Fraction, Date, Time, Money, Measure, Telephone and Electronic semiotic classes (#421)
- Rw (Kinyarwanda) TN: Cardinal coverage up to one hundred trillion, Time and Transliteration support (#209)
- Ja (Japanese) TN: Ordinal and Time semiotic classes (#240)
- Fr (French) TN: Date base coverage (#269)


### Fixed

- Fixed En TN addresses ending in terminal period abbreviations (nv-bug 4786225) (#218)
- Fixed En TN Electronic module (nv-bug 4786263) (#220)
- Fixed En TN invalid escape sequences (#219)
- Fixed En TN comma handling in Electronic semiotic class (#332)
- Fixed It TN Electronic processing issue from (#166) (#221)
- Fixed Es TN Electronic processing issue from (#166) (#224)
- Fixed Es TN whitelist entries causing issue (#211) (#232)
- Fixed De TN whitelist and Electronic pattern matching for issue (#228) (#234) (#237)
- Fixed Zh ITN space handling (#244)
- Jenkinsfile updates and fixes (#325) (#341) (#419)


### Changed

- En TN Money expanded per/unit mappings to include weight, volume, distance and area units (e.g. `$20/kg` → `twenty dollars per kilogram`, `$5/sq ft` → `five dollars per square foot`) (#227)
- Ja ITN updated post-processing and verbalization rules (#208)
- Updated contributing guidelines (#251)
- Updated Dockerfile (#254)
- Pinned Docker base image and added tty fix for tests (#385)


### Contributors

Alex Cui, Anand Joseph, dankeinan1, Jinwoo Bae, Kevin James, kurt0cougar, Mai Anh, Mariana Graterol Fuenmayor, Namrata Gachchi, P V RAJAN, Rajan Putty, Shreesh D, Shreyas Pawar, Simon Zuberek, Tarushi V, tbartley94


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

