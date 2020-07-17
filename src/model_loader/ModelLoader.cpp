#include "libtensorflow/include/tensorflow/c/c_api.h"
#include "ModelLoader.hpp"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include <numeric>
#include <string.h>

//using namespace std;

void NoOpDeallocator(void* data, size_t a, void* b) {}

void ModelLoader::ModelLoader() 
    {
        //********* Read model
        TF_Graph* Graph = TF_NewGraph();
        TF_Status* Status = TF_NewStatus();
        TF_SessionOptions* SessionOpts = TF_NewSessionOptions();
        TF_Buffer* RunOpts = NULL;
        
        char* saved_model_dir = "/Users/wejdene/Desktop/Praktikum/model/"; 
        char* tags = "serve"; 
        
        int ntags = 1; 
        TF_Session* Session = TF_LoadSessionFromSavedModel(SessionOpts, RunOpts, saved_model_dir, &tags, ntags, Graph, NULL, Status);
        
        if (TF_GetCode(Status) == TF_OK) {
            std::cout << "TF_LoadSessionFromSavedModel OK\n" << std::endl;
        }
        else {
            std::cout << "%s" << TF_Message(Status) << std::endl;
        }
    

        //********* Get input tensor
        int NumInputs = 1;
        TF_Output* Input = (TF_Output*) malloc(sizeof(TF_Output) * NumInputs);
        TF_Output t0 = {TF_GraphOperationByName(Graph, "serving_default_input"), 0};

        if(t0.oper == NULL) {
            std::cout << "ERROR: Failed TF_GraphOperationByName serving_default_input\n" << std::endl;
        }
        else {
            std::cout << "TF_GraphOperationByName serving_default_input is OK\n" << std::endl;
        }
        Input[0] = t0;

        
        //********* Get Output tensor
        int NumOutputs = 1;
        TF_Output* Output = (TF_Output*) malloc(sizeof(TF_Output) * NumOutputs);
        TF_Output t2 = {TF_GraphOperationByName(Graph, "StatefulPartitionedCall"), 0};
        
        if(t2.oper == NULL) {
            std::cout << "ERROR: Failed TF_GraphOperationByName StatefulPartitionedCall\n" << std::endl;
        }
        else{
            std::cout << "TF_GraphOperationByName StatefulPartitionedCall is OK\n" << std::endl;
        }
        Output[0] = t2; 


        //********* Allocate data for inputs & outputs
        TF_Tensor** InputValues  = (TF_Tensor**)malloc(sizeof(TF_Tensor*)*NumInputs);
        TF_Tensor** OutputValues = (TF_Tensor**)malloc(sizeof(TF_Tensor*)*NumOutputs);

    }

ModelLoader::std::vector<float> Evaluator(std::vector<float> neural_network_input)
    {
        //std::vector<double> EvaluateModel(std::vector<double> neural_network_input) const = 0;

        
        int ndims = 2; 
        int len = neural_network_input.size();
        std::vector<std::int64_t> dims = {1, len};
        
        // ndata is total byte size of our data, not the length of the array
        int data_size = std::accumulate(dims.begin(), dims.end(), sizeof(float), std::multiplies<std::int64_t>{}); 
        auto data = static_cast<float*>(std::malloc(data_size));
        std::copy(neural_network_input.begin(), neural_network_input.end(), data);
        TF_Tensor* input_tensor = TF_NewTensor(TF_FLOAT, dims.data(), ndims, data, data_size, &NoOpDeallocator, 0);
        
        if (input_tensor != NULL) {
            std::cout << "TF_NewTensor is OK\n" << std::endl;
        }
        else{
            std::cout << "ERROR: Failed TF_NewTensor\n" << std::endl;
        }
        InputValues[0] = input_tensor;

        // Run the Session
        TF_SessionRun(Session, NULL, Input, InputValues, NumInputs, Output, OutputValues, NumOutputs, NULL, 0,NULL , Status);
        
        if (TF_GetCode(Status) == TF_OK) {
            std::cout << "Session is OK\n" << std::endl;
        }
        else {
            std::cout << "%s" << TF_Message(Status) << std::endl;
        }
        
        //auto buff = TF_TensorData(OutputValues[0]);
        //std::vector<float*> q_values = std::vector<float*>(TF_TensorData(OutputValues[0]));
        
        //auto values = static_cast<float*> (TF_TensorData(OutputValues[0]));
        //std::vector<float> q_values(sizeof(values));
        //memcpy(q_values.data(), &values, sizeof(values));

        
        auto values = (float*) (TF_TensorData(OutputValues[0]));
        std::vector<float> q_values(values,values+4);

        return q_values;
    }