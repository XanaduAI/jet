#include <chrono>
#include <complex>
#include<iostream>
#include<fstream>
#include<sstream>
#include <streambuf>
#include<string>

#include <Jet.hpp>
#include <cuda.h>
#include <cuComplex.h>

int main(int argc, char** argv)
{
    // Set simple using statements
    using namespace Jet;
    using namespace Jet::Utilities;
    using c_fp32 = cuComplex;
    using namespace std::chrono;

    std::string file_name = ""; 
    size_t num_threads = 1;
    if (argc > 2){
        file_name = argv[1];
        num_threads = std::atoi(argv[2]);
    }
    else{
        std::cerr << "Please specify a JSON file to contract" << std::endl;
        exit(1);
    }
    TensorNetworkFile<CudaTensor<c_fp32>> tensor_file;
    try {
        std::ifstream tn_data(file_name);
        std::string m10_str {std::istreambuf_iterator<char>(tn_data), std::istreambuf_iterator<char>()};
        // Load data into TensorNetwork and PathInfo objects
        Jet::TensorNetworkSerializer<CudaTensor<c_fp32>> serializer;
        tensor_file = serializer(m10_str, true);
    }
    catch(...){
        std::cerr << "Please specify a valid JSON file to contract" << std::endl;
        exit(1);
    }

    Jet::TensorNetwork<CudaTensor<c_fp32>> tn = tensor_file.tensors;
    Jet::PathInfo path = tensor_file.path.value(); // std::optional requires value()

    // Create contractor and add TN and path data
    TaskBasedContractor<CudaTensor<c_fp32>> tbc(num_threads);
    tbc.AddContractionTasks(tn, path);
    
    // GPUs have limited memory, so free unneeded tensors
    tbc.AddDeletionTasks();
    
    // Time the contraction operation
    auto t1 = high_resolution_clock::now();
    tbc.Contract().wait(); // Contract() non-blocking so requires wait()
    auto t2 = high_resolution_clock::now();

    // Output timings
    auto duration = duration_cast<duration<float>>(t2 - t1).count();

    std::cout << "t=" << duration << "s" << std::endl;
    auto res = tbc.GetResults()[0].GetHostDataVector();
    std::cout << "result=" << res << "" << std::endl;

    return 0;
}
