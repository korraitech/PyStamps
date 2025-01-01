# UrbanSAR signature
urbansar service using SNAP and STAMPS

### How to run InSAR service.
1. Update the params
    This aoi is used to subset the scene.
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
    "Pinp": "/Users/rvs/myspace/workspace/services/service-utils/manual-stamps/manual_stamps_v1/data/export",   # change the path to rslc exported folder
    "Pout": "/Users/rvs/myspace/workspace/services/service-utils/manual-stamps/manual_stamps_v1/data/output"    # Ouput directory to process the path to rslc exported folder
    }
    ```
3. Run the code
    ```
    # Now run the  command
    python3 main.py
    ```
