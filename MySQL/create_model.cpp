#include <mysql.h>
#include <string>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <exception>

using namespace std;
extern "C"
{
    // 初始化函数
    bool create_model_init(UDF_INIT *initid, UDF_ARGS *args, char *message);
    // 实现函数
    char *create_model(UDF_INIT *initid, UDF_ARGS *args, char *result, unsigned long *length, char *is_null, char *error);
    // 清理函数
    void create_model_deinit(UDF_INIT *initid);
}

bool create_model_init(UDF_INIT *initid, UDF_ARGS *args, char *message)
{
    if (args->arg_count != 3)
    {
        strcpy(message, "create_model() requires three arguments");
        return 1;
    }
    // epochs
    if (args->arg_type[0] != INT_RESULT)
    {
        strcpy(message, "argument 0 must be a INT");
        return 1;
    }

    if (args->arg_type[1] != REAL_RESULT)
    {
        strcpy(message, "argument 1 must be a DOUBLE");
        return 1;
    }

    if (args->arg_type[2] != REAL_RESULT)
    {
        strcpy(message, "argument 2 must be a DOUBLE");
        return 1;
    }
    return 0;
}

char *create_model(UDF_INIT *initid, UDF_ARGS *args, char *result, unsigned long *length, char *is_null, char *error)
{
    string output = "Fail to Create Model";
    long long epochs = *((long long *)args->args[0]);
    double lr = *((double *)args->args[1]);
    double momentum = *((double *)args->args[2]);

    int i = 0;
    string command = "source final/virtual/bin/activate && python3 ./final/model.py " + to_string(epochs) + ' ' + to_string(lr) + ' ' + to_string(momentum);
    i = system(command.c_str());
    if (i)
    {
        throw runtime_error("Python script failed to execute");
    }
    output = string("Model Create Successfully\n") + "Epochs: " + to_string(epochs) + ", lr: " + to_string(lr) + ", momentum: " + to_string(momentum);

    memcpy(result, output.c_str(), output.size());
    *length = output.size();

    return result;
}

void create_model_deinit(UDF_INIT *initid)
{
}