#include "libtensorflow/include/tensorflow/c/c_api.h"
#include <vector>


void NoOpDeallocator() {};

class ModelLoader {
public:
    ModelLoader();
    std::vector<float> Evaluator(std::vector<float> neural_network_input, int actions_number);

private:
    TF_Status* Status;
    TF_Graph* Graph;
    const char* saved_model_dir; 
    const char* tags;
    TF_Session* Session;
    const int NumInputs = 1;
    TF_Output* Input;
    TF_Output t0;
    TF_Output t2;
    const int NumOutputs = 1;
    TF_Output* Output;
    TF_Tensor** InputValues;
    TF_Tensor** OutputValues;
    

};