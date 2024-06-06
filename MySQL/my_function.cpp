#include <mysql.h>
#include <cstring>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <exception>

using namespace std;

extern "C"
{
    // 初始化函數
    bool my_function_init(UDF_INIT *initid, UDF_ARGS *args, char *message);
    // 實際的UDF函數
    long long my_function(UDF_INIT *initid, UDF_ARGS *args, char *is_null, char *message, char *error);
    // 清理函數
    void my_function_deinit(UDF_INIT *initid);
}

bool my_function_init(UDF_INIT *initid, UDF_ARGS *args, char *message)
{
    if (args->arg_count != 11)
    {
        strcpy(message, "wrong number of arguments, please check arguments");
        return 1;
    }
    // step
    if (args->arg_type[0] != INT_RESULT)
    {
        strcpy(message, "argument0 must be an integer");
        return 1;
    }
    // amount
    if (args->arg_type[1] != REAL_RESULT)
    {
        strcpy(message, "argument1 must be a double");
        return 1;
    }
    // oldbalanceOrg
    if (args->arg_type[2] != REAL_RESULT)
    {
        strcpy(message, "argument2 must be a double");
        return 1;
    }
    // newbalanceOrig
    if (args->arg_type[3] != REAL_RESULT)
    {
        strcpy(message, "argument3 must be a double");
        return 1;
    }
    // oldbalanceDest
    if (args->arg_type[4] != REAL_RESULT)
    {
        strcpy(message, "argument4 must be a double");
        return 1;
    }
    // newbalanceDest
    if (args->arg_type[5] != REAL_RESULT)
    {
        strcpy(message, "argument5 must be a double");
        return 1;
    }
    // orig_diff
    if (args->arg_type[6] != INT_RESULT)
    {
        strcpy(message, "argument6 must be an integer");
        return 1;
    }
    // dest_diff
    if (args->arg_type[7] != INT_RESULT)
    {
        strcpy(message, "argument7 must be an integer");
        return 1;
    }
    // surge
    if (args->arg_type[8] != INT_RESULT)
    {
        strcpy(message, "argument8 must be an integer");
        return 1;
    }
    // freq_dest
    if (args->arg_type[9] != INT_RESULT)
    {
        strcpy(message, "argument9 must be an integer");
        return 1;
    }
    // true_type
    if (args->arg_type[10] != INT_RESULT)
    {
        strcpy(message, "argument10 must be an integer");
        return 1;
    }

    return 0;
}

long long my_function(UDF_INIT *initid, UDF_ARGS *args, char *is_null, char *message, char *error)
{
    // long long a = *((long long *)args->args[0]);
    long long step = *((long long *)args->args[0]);
    double amount = *((double *)args->args[1]);
    double oldbalanceOrg = *((double *)args->args[2]);
    double newbalanceOrig = *((double *)args->args[3]);
    double oldbalanceDest = *((double *)args->args[4]);
    double newbalanceDest = *((double *)args->args[5]);
    long long orig_diff = *((long long *)args->args[6]);
    long long dest_diff = *((long long *)args->args[7]);
    long long surge = *((long long *)args->args[8]);
    long long freq_dest = *((long long *)args->args[9]);
    long long true_type = *((long long *)args->args[10]);

    long long result = 0;

    try
    {
        int i = 0;
        string command = "source final/virtual/bin/activate && python3 ./final/predict.py " + to_string(step) + ' ' + to_string(amount) + ' ' + to_string(oldbalanceOrg) + ' ' + to_string(newbalanceOrig) + ' ' + to_string(oldbalanceDest) + ' ' + to_string(newbalanceDest) + ' ' + to_string(orig_diff) + ' ' + to_string(dest_diff) + ' ' + to_string(surge) + ' ' + to_string(freq_dest) + ' ' + to_string(true_type);
        // 執行Python腳本
        i = system(command.c_str());
        if (i)
        {
            throw runtime_error("Python script failed to execute");
        }
    }
    catch (const exception &e)
    {
        cerr << "Error: " << e.what() << endl;
    }

    try
    {
        // 讀取Python腳本寫入的文件
        ifstream file("./final/result.txt");
        string data;
        long long temp = 0;
        if (file.is_open())
        {
            getline(file, data);
            temp = stoll(data);
            result = temp;
            file.close();
        }
        else
        {
            throw runtime_error("failed to open file");
        }
    }
    catch (const exception &e)
    {
        cerr << e.what() << '\n';
    }
    return result;
}

void my_function_deinit(UDF_INIT *initid)
{
    // 清理代碼（如果有的話）
}