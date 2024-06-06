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
    bool change_threshold_init(UDF_INIT *initid, UDF_ARGS *args, char *message);
    // 實際的UDF函數
    char *change_threshold(UDF_INIT *initid, UDF_ARGS *args, char *result, unsigned long *length, char *is_null, char *error);
    // 清理函數
    void change_threshold_deinit(UDF_INIT *initid);
}

bool change_threshold_init(UDF_INIT *initid, UDF_ARGS *args, char *message)
{
    if (args->arg_count != 1)
    {
        strcpy(message, "wrong number of arguments, please check arguments");
        return 1;
    }
    // threshold
    if (args->arg_type[0] != REAL_RESULT)
    {
        strcpy(message, "argument1 must be a double");
        return 1;
    }

    return 0;
}

char *change_threshold(UDF_INIT *initid, UDF_ARGS *args, char *result, unsigned long *length, char *is_null, char *error)
{

    string output = "Threshold Not Changed";
    cout << output.size() << endl;
    double threshold = *((double *)args->args[0]);

    ofstream outFile("./final/threshold.txt");
    if (!outFile.is_open())
    {
        cerr << "Failed to open file." << std::endl;
        *length = output.size();
        strcpy(result, output.c_str());

        return result;
    }

    outFile << to_string(threshold) << endl;

    outFile.close();

    output = "Threshold Changed to " + to_string(threshold);

    *length = output.size();
    memcpy(result, output.c_str(), output.size());

    return result;
}

void change_threshold_deinit(UDF_INIT *initid)
{
    // 清理代碼（如果有的話）
}