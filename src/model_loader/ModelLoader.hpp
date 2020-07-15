#include "tensorflow/c/c_api.h"
#include <vector>


static TF_Status* Status = TF_NewStatus();
const char* saved_model_dir; 
const char* tags;
static TF_Session* Session;
const int NumInputs = 1;
static TF_Output* Input;
static TF_Output t0;
const int NumOutputs = 1;
static TF_Output* Output;
static TF_Tensor** InputValues;
static TF_Tensor** OutputValues;

void NoOpDeallocator() {};

class ModelLoader {
public:
    void ModelLoader();
    std::vector<float> Evaluator(std::vector<float> neural_network_input);

};