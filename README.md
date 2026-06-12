# PyStamps

**PyStamps — Python Stanford Method for Persistent Scatterers**

![License: ESA-PL Permissive v2.4](https://img.shields.io/badge/License-ESA--PL%20Permissive%20v2.4-blue.svg)
![Python](https://img.shields.io/badge/python-3.x-blue.svg)
![Method: StaMPS](https://img.shields.io/badge/method-StaMPS-orange.svg)

PyStamps is a software package for extracting ground displacements from time series of
Synthetic Aperture Radar (SAR) acquisitions using the Persistent Scatterer (PS) technique.
It is a Python implementation of the Stanford Method for Persistent Scatterers
([StaMPS](https://github.com/dbekaert/StaMPS/tree/master)). APS Linear is integrated as part
of the tropospheric correction methods in the processing workflow.


## Table of Contents

- [PyStamps](#pystamps)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [About KorrAI](#about-korrai)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Configuration](#configuration)
    - [Running PyStamps](#running-pystamps)
  - [Sample Test Case](#sample-test-case)
  - [Supported Pre-processors](#supported-pre-processors)
  - [How to Cite](#how-to-cite)
    - [Citing PyStamps](#citing-pystamps)
    - [Citing the Original StaMPS Method](#citing-the-original-stamps-method)
    - [Citing the Pre-processor (SNAP)](#citing-the-pre-processor-snap)
  - [Acknowledgements](#acknowledgements)
  - [Contributors](#contributors)
  - [License](#license)
  - [Contact](#contact)


## Overview

PyStamps processes a stack of co-registered SAR acquisitions and identifies persistent
scatterers stable, point-like targets that maintain coherent radar reflectivity over time.
By tracking the interferometric phase of these scatterers, PyStamps derives line-of-sight
ground deformation time series suitable for monitoring subsidence, landslides, infrastructure
stability, and other land-based risks.

The processing workflow includes candidate selection, phase estimation and weeding, patch
merging, 3-D phase unwrapping, linear atmospheric phase screen (APS) correction, spatially
correlated look-angle (SCLA) error correction, and export of the final displacement product.


## About KorrAI

KorrAI builds Traceable AI for billion-dollar builds. Engineers, insurers, and asset owners
make billion-dollar decisions with fragmented ground data, KorrAI closes that gap. Our
platform, **TRAIL**, unifies satellite InSAR, geotechnical studies, multi-hazard data, and
engineering documents into traceable, citation-backed desktop risk studies. Today KorrAI
monitors 5+ million sq. km and over $100B in critical assets across mining, energy, insurance,
and infrastructure. Ground deformation is one of TRAIL's core evidence layers, and PyStamps is
the open-source engine we built to measure it at scale. Learn more at
[korrai.com](https://www.korrai.com).


## Requirements

- Python 3.x
- [`numpy`](https://numpy.org/)
- [`scipy`](https://scipy.org/)
- [`h5py`](https://www.h5py.org/)
- [`tqdm`](https://tqdm.github.io/)
- [`snaphu`](https://web.stanford.edu/group/radar/softwareandlinks/sw/snaphu/) (phase unwrapping)


## Installation

Clone the repository and install the Python dependencies:

```bash
git clone https://github.com/korraitech/PyStamps.git
cd PyStamps
pip install -r requirements.txt
```


## Usage

### Configuration

Edit `input.json` to set the processing parameters:

```json
{
    "Pstm": {
        "ADT": 0.4,
        "NRP": 2,
        "NAP": 2,
        "RPO": 50,
        "APO": 200
    },
    "Prdt": "20231107",
    "Pinp": "test/export",
    "Pout": "test/output"
}
```

| Parameter | Description |
|-----------|-------------|
| `Pstm.ADT` | Amplitude dispersion threshold for PS candidate selection |
| `Pstm.NRP` | Number of patches in the range direction |
| `Pstm.NAP` | Number of patches in the azimuth direction |
| `Pstm.RPO` | Patch overlap in the range direction (pixels) |
| `Pstm.APO` | Patch overlap in the azimuth direction (pixels) |
| `Prdt`     | Master / reference acquisition date (`YYYYMMDD`) |
| `Pinp`     | Input directory containing the exported RSLC stack |
| `Pout`     | Output directory for processing results |

### Running PyStamps

```bash
python3 main.py
```

The final ground displacement product is written as `ground_displacement.csv` in the
output directory (`Pout`).


## Sample Test Case

1. Extract the data stored in the `test` folder.
2. Update the paths in `input.json` accordingly.
3. Run `python3 main.py`.
4. The output CSV is generated in the `Pout` folder.


## Supported Pre-processors

- **SNAP** (ESA Sentinel Application Platform)


## How to Cite

### Citing PyStamps

> KorrAI Technologies Ltd., Sharan, R. V., Mann, R., Khosla, R., & Girohi, P. (2026).
> *PyStamps: Python Stanford Method for Persistent Scatterers* (v0.2.0) [Computer software].
> https://github.com/korraitech/PyStamps

```bibtex
@software{pystamps2026,
  author       = {{KorrAI Technologies Ltd.} and Sharan, Rahul V and Mann, Rajat and Khosla, Ritwek and Girohi, Priti},
  title        = {PyStamps: Python Stanford Method for Persistent Scatterers},
  version      = {0.2.0},
  year         = {2026},
  publisher    = {KorrAI Technologies Ltd.},
  url          = {https://github.com/korraitech/PyStamps}
}
```

### Citing the Original StaMPS Method

PyStamps implements the StaMPS methodology developed by Prof. Andrew Hooper and colleagues.

> Hooper, A., Zebker, H., Segall, P., & Kampes, B. (2004). A new method for measuring deformation
> on volcanoes and other natural terrains using InSAR persistent scatterers.
> *Geophysical Research Letters*, 31, L23611. https://doi.org/10.1029/2004GL021737

> Hooper, A., Bekaert, D., Spaans, K., & Arikan, M. (2012). Recent advances in SAR interferometry
> time series analysis for measuring crustal deformation.
> *Tectonophysics*, 514–517, pp. 1–13. https://doi.org/10.1016/j.tecto.2011.10.013

```bibtex
@article{hooper2004,
  author  = {Hooper, Andrew and Zebker, Howard and Segall, Paul and Kampes, Bert},
  title   = {A new method for measuring deformation on volcanoes and other natural terrains using InSAR persistent scatterers},
  journal = {Geophysical Research Letters},
  volume  = {31},
  number  = {23},
  pages   = {L23611},
  year    = {2004},
  doi     = {10.1029/2004GL021737}
}

@article{hooper2012,
  author  = {Hooper, Andrew and Bekaert, David and Spaans, Karsten and Arikan, Mahmut},
  title   = {Recent advances in SAR interferometry time series analysis for measuring crustal deformation},
  journal = {Tectonophysics},
  volume  = {514--517},
  pages   = {1--13},
  year    = {2012},
  doi     = {10.1016/j.tecto.2011.10.013}
}
```

Original StaMPS repository: https://github.com/dbekaert/StaMPS/tree/master

### Citing the Pre-processor (SNAP)

PyStamps uses **SNAP** for pre-processing.

> Foumelis, M., Blasco, J. M. D., Desnos, Y.-L., Engdahl, M., Fernández, D., Veci, L., Lu, J.,
> & Wong, C. (2018). ESA SNAP–StaMPS integrated processing for Sentinel-1 persistent scatterer
> interferometry. *IGARSS 2018 – IEEE International Geoscience and Remote Sensing Symposium*,
> pp. 1364–1367. https://doi.org/10.1109/IGARSS.2018.8519545

```bibtex
@inproceedings{foumelis2018,
  author    = {Foumelis, Michael and Blasco, Jose Manuel Delgado and Desnos, Yves-Louis and Engdahl, Marcus and Fern{\'a}ndez, Diego and Veci, Luis and Lu, Jun and Wong, Cecilia},
  title      = {ESA SNAP--StaMPS integrated processing for Sentinel-1 persistent scatterer interferometry},
  booktitle = {IGARSS 2018 -- IEEE International Geoscience and Remote Sensing Symposium},
  pages     = {1364--1367},
  year      = {2018},
  doi       = {10.1109/IGARSS.2018.8519545}
}
```


## Acknowledgements

PyStamps is built upon the Stanford Method for Persistent Scatterers (StaMPS), originally
developed by **Prof. Andrew Hooper** (`A.Hooper@leeds.ac.uk`, University of Leeds) and
collaborators. We gratefully acknowledge their foundational work and the open-source StaMPS
project: https://github.com/dbekaert/StaMPS


## Contributors

| Name | Role |
|------|------|
| Rahul V Sharan | Principal Engineer, KorrAI |
| Ritwek Khosla | Senior Data Scientist, KorrAI |
| Rajat Mann | CTO, KorrAI |
| Priti Girohi | Senior Data Scientist, KorrAI |


## License

PyStamps is released under the **European Space Agency Public License (ESA-PL) Permissive
(Type 3) – v2.4**. See the [LICENSE](LICENSE) file for the full text and
[https://essr.esa.int/license/european-space-agency-public-license-v2-4-permissive-type-3](https://essr.esa.int/license/european-space-agency-public-license-v2-4-permissive-type-3).


## Contact

Rahul V Sharan — rahul.sharan@korrai.com
