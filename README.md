# pyStaMPS signature
pystamps

### How to run pystamps.
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
### How to run unit testcase.
1. Extract the data stored in test folder.
2. Then update the path in input.json accordingly
3. Finally run the python main.py
4. Output csv will be generated in Pout folder.