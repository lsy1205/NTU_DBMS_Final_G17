# NTU_DBMS_Final_G17
## Basic Information
### Topic: DataGuardSQL: OOD Detection
* Group 17
* Members: 
  * 組長：機械所碩一 包杰修 (Email: canorivera0221@gmail.com)
  * 組員：機械所碩一 侯貝霖 (Email: oscar104cn@gmail.com)
  * 組員：電機系大四 劉瑄穎 (Email: stevenliu901205@gmail.com)
  * *If you encounter problems, feel free to send us an email*
## Project Abstract
In this project, we used **SWAG model** to detect **out of distribution(OOD)** data and integrated it with MySQL. The dataset we used is [**Transaction Fraud Detection**](https://www.kaggle.com/code/benroshan/transaction-fraud-detection/input). Using our model, users can find OOD data which is likely to be fraud transactions. The possible application scenarios are banks, risk control centers, and so on. With this repository, users only need to know how to write MySQL programs and they can **detect OOD data, update model, set threshold** in MySQL workbench. That is to say, users can finish the whole process in MySQL.

## Project Configuration
* **C++** 
  * mysql.h (provided by mysql)
  * cstring
  * iostream
  * fstream
  * string
  * cstdlib
  * stdlib
  * exception
* **Python** 
  * Version: > 3.9.18
  * Packages: 
    * torch
    * pandas
    * torchvision
    * sys
    * joblib
    * scikit-learn
    * enum
    * matplotlib
    * numpy
    * openpyxl
  
* **MySQL**
  * Community Version: >= 8.0.36
* **Data Set**
  * [Transaction Fraud Detection](https://www.kaggle.com/code/benroshan/transaction-fraud-detection/input)
  
### File Structure
* please lookup *file_structure.txt*
## How to Use this Repository
*We use MacOS with M1 chip for DEMO*
### Preparation
*You might need to enter your password when source .src or using sudo*
1. Change the path in ./MySQL/source_all.src
    ```bash
    g++ -shared -o ./my_function.so ./my_function.cpp -I[/path/to/mysql.h/folder]
    sudo cp ./my_function.so [/path/to/mysql/plugin]
    ```
2. Create .so(MacOS and Linux)/.dll(Windows) files(*in MySQL folder*)
    ```bash
    source source_all.src
    ```
3. Make Directory
    ```bash
    sudo mkdir [path/to/mysql/data]/final
    ```
4. Copy Files to MySQL data/final Folder(*in ML folder*)
    ``` bash
    sudo cp -R ./final/* [path/to/mysql/data/final/]
    ```
5. Change the Accessibility of the Files in MySQL data Folder
    ```bash
    sudo chmod -R 777 [path/to/mysql/data/final/]
    ```
6. Create Virtual Environment In MySQL data/final Folder
    ```bash
    sudo python3 -m venv [path/to/mysql/data/final/virtual]
    ```
7. Activate Virtual Environment
    ```bash
    sudo -s
    source [path/to/mysql/data/final/virtual/bin/activate]
    ```
8. Download Python Packages (*make sure you activate the virtual environment*)
    ```bash
    source install_python_pkg.src
    ```
9. Deactivate Virtual Environment
    ```bash
    deactivate
    ```

### MySQL 
*In MySQL Workbench*
1. Create functions
    ```bash
    source [path/to/create_functions.sql]
    ```
2. Drop functions (Optional)
    ```bash
    source [path/to/drop_functions.sql]
    ```
### Examples
1. The example file is *./MySQL/sample.sql*, you can see the schemas and how to run functions in MySQL. Remember to follow the schema in the file
2. The Basic Form to Execute the Supported Function
    * my_function:
      ```sql
      SELECT id, my_function(step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, orig_diff, dest_diff, surge, freq_dest, true_type) as is_ood 
      FROM transact;
      ```
    * create_model:
      ```sql
      UPDATE result SET message = create_model(
          (SELECT epochs FROM Model where id = 1),
          (SELECT learning_rate FROM Model where id = 1),
          (SELECT momentum FROM Model where id = 1)
        ) 
        WHERE id = 1;
      ```
    * change_threshold:
      ```sql
      UPDATE result SET message = change_threshold((SELECT(threshold) FROM config WHERE id = 1))
      WHERE id = 2;
      ```
3. More Transaction Data
    * you can find more transaction data in the *inputdata* folder
    * load the data into your database with load_data_to_MySQL.ipynb
    * four files are provided
      1. test1.csv: in-distribution data
      2. test2.csv: in-distribution data
      3. test3.csv: in-distribution data
      4. OOD_test.csv: all of the data in this file is OOD
### If You Encounter Any Problem When Executing MySQL Functions
* Try to solve the problems according to the error messages, you can see the error message by
    ```bash
    sudo tail -f /usr/local/mysql/data/mysqld.local.err
    ```

## Problems You Might Encounter
* For MacOS Intel Chip Users:
    * When using MacOS intel version, you might encounter problems when using C++ UDF to system call python, which we currently don't have good methods for the issue
    *  python3 ML/final/model.py
        * error message: 
        *Found Intel OpenMP ('libiomp') and LLVM OpenMP ('libomp') loaded at the same time. Both libraries are known to be incompatible and this can cause random crashes or deadlocks on Linux when loaded in the same Python program.*
        * [Refernce Link 1](https://github.com/joblib/threadpoolctl/blob/master/multiple_openmp.md)
        * [Refernce Link 2](https://github.com/ContinuumIO/anaconda-issues/issues/13221)
## Reference Links
* [MySQL Loadable Function](https://dev.mysql.com/doc/extending-mysql/8.4/en/adding-loadable-function.html)
