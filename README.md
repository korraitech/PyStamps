# pyStaMPS
pyStaMPS - Python Stanford Method for Persistent Scatterers

pyStaMPS is a software package that allows to extract ground displacements from time series of synthetic aperture radar (SAR) acquisitions. The package incorporates persistent scatterer only. APS Linear is integrated as part of tropospheric correction methods in the processing workflow.

### How to run pyStaMPS.
1. Update the params in input.json
    ```
   {
    "Pstm": {
        "ADT": 0.4,
        "NRP": 2,
        "NAP": 2,
        "RPO": 50,
        "APO": 200
    },
    "Prdt": "20231107",
    "Pinp": "test/export",   # change the path to rslc exported folder
    "Pout": "test/output"    # Ouput directory to process the path to rslc exported folder
    }
    ```
3. Run the code
    ```
    # Now run the  command
    python3 main.py
    ```
### Supported pyStaMPS pre-processors:
SNAP

### Installation packages.
shaphu
tqdm
h5py
scipy
numpy

### How to run sample testcase.
1. Extract the data stored in test folder.
2. Then update the path in input.json accordingly
3. Finally run the python main.py
4. Output csv will be generated in Pout folder.
