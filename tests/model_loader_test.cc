#include "libtensorflow/c/c_api.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include <numeric>
#include <string>
#include "ModelLoader.hpp"


int main()
{
    ModelLoader model;

    model.ModelLoader();

    std::vector<float> input = {0.68890405, 0.07822049, 0.5780419, 1.0, 0.5767162,
        0.14886844, 0.65039325, 0.52042484, 1.0, 0.7518059,
        0.3736292, 1.0, 0.86391735, 0.53165483, 0.35212266, 0.0,
    };

    std::vector<float> q_values = model.Evaluator(input);

    for (int i=0;i<=3;i++)
    {
        std::cout << q_values[i] << std::endl;
    }
    
    return 0;

}