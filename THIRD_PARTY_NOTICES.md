# Third-Party Notices

This file lists third-party code and datasets referenced, adapted, or used by OpenSTBench.

## SimulEval

- Project: SimulEval
- Repository: https://github.com/facebookresearch/SimulEval
- License: Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)
- License URL: https://creativecommons.org/licenses/by-sa/4.0/
- Usage in OpenSTBench: Portions of the latency evaluation implementation are adapted from SimulEval.
- Modified: Yes. The adapted code was modified for OpenSTBench's evaluation interfaces and metric organization.
- Affected files:
  - `src/openstbench/latency/...`

The adapted portions are distributed under CC BY-SA 4.0. Users of these portions must comply with the terms of CC BY-SA 4.0, including attribution and ShareAlike requirements.

## Datasets

OpenSTBench references third-party datasets for evaluation and reproduction of the paper results. These datasets are not distributed under the OpenSTBench code license.

Users must review and comply with the original license and terms of use for each dataset before downloading, redistributing, or using it. Some datasets are restricted to research or non-commercial use.

Datasets referenced by the project include:

| Dataset | Source | Notes |
| --- | --- | --- |
| MSLT | Microsoft Research | Subject to the original Microsoft dataset terms. |
| LibriTTS-based paired speaker set | Constructed from LibriTTS-derived data | The constructed data will be provided through GitHub Releases. Users must also comply with the source corpus terms. |
| RAVDESS | Zenodo | OpenSTBench uses `Audio_Speech_Actors_01-24.zip` from the RAVDESS Zenodo release. |
| MCAE-SPPS | OSF | Subject to the original dataset terms. |
| NonverbalTTS | Hugging Face | Subject to the original dataset terms. |
| SynParaSpeech | Hugging Face | Subject to the original dataset terms. |

The OpenSTBench repository does not grant additional rights to any third-party dataset beyond the rights granted by the original dataset providers.