#include <complex>
#include <fstream>
#include <iostream>
#include <sstream>
#include <streambuf>

#include <chrono>
#include <string>

#include <Jet.hpp>

int main(int argc, char **argv)
{
    // Set simple using statements
    using namespace Jet;
    using namespace Jet::Utilities;
    using c_fp32 = std::complex<float>;
    using namespace std::chrono;

    std::string file_name = "";
    size_t num_threads = 1;
    if (argc > 2) {
        file_name = argv[1];
        num_threads = std::atoi(argv[2]);
    }
    else {
        std::cerr << "Please specify a JSON file to contract and number of taskflow threads" << std::endl;
        exit(1);
    }
    TensorNetworkFile<Tensor<c_fp32>> tensor_file;
    try {
        std::ifstream tn_data(file_name);
        std::string m10_str{std::istreambuf_iterator<char>(tn_data),
                            std::istreambuf_iterator<char>()};
        // Load data into TensorNetwork and PathInfo objects
        Jet::TensorNetworkSerializer<Tensor<c_fp32>> serializer;
        tensor_file = serializer(m10_str);
    }
    catch (...) {
        std::cerr << "Please specify a valid JSON file to contract"
                  << std::endl;
        exit(1);
    }

    Jet::TensorNetwork<Tensor<c_fp32>> tn = tensor_file.tensors;
    Jet::PathInfo path =
        tensor_file.path.value(); // std::optional requires value()

    // Create contractor and add TN and path data
    TaskBasedContractor<Tensor<c_fp32>> tbc(num_threads);
    tbc.AddContractionTasks(tn, path);

    // Time the contraction operation
    auto t1 = high_resolution_clock::now();
    tbc.Contract().wait(); // Contract() non-blocking so requires wait()
    auto t2 = high_resolution_clock::now();

    // Output timings
    auto duration = duration_cast<duration<float>>(t2 - t1).count();
    std::cout << "t=" << duration << "s" << std::endl;
    std::cout << "result=" << tbc.GetResults() << std::endl;

    return 0;
}
