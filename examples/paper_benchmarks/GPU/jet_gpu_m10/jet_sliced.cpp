#include <complex>
#include <fstream>
#include <iostream>
#include <sstream>
#include <streambuf>

#include <chrono>
#include <string>

#include <Jet.hpp>

int main(int argc, char *argv[])
{
    using namespace Jet;
    using namespace Jet::Utilities;
    using c_fp32 = cuComplex;
    using namespace std::chrono;

    size_t num_threads = 1;
    size_t indices_to_slice = 1;

    std::string file_name = "";
    if (argc > 3) {
        file_name = argv[1];
        num_threads = std::atoi(argv[2]);
        indices_to_slice = std::atoi(argv[3]);
    }
    else {
        std::cerr << "Please specify a JSON file to contract" << std::endl;
        exit(1);
    }

    using Tensor = Jet::CudaTensor<c_fp32>;

    TensorNetworkFile<Tensor> tensor_file;
    try {
        std::ifstream tn_data(file_name);
        std::string m10_str{std::istreambuf_iterator<char>(tn_data),
                            std::istreambuf_iterator<char>()};
        // Load data into TensorNetwork and PathInfo objects
        Jet::TensorNetworkSerializer<Tensor> serializer;
        tensor_file = serializer(m10_str, true);
    }
    catch (...) {
        std::cerr << "Please specify a valid JSON file to contract"
                  << std::endl;
        exit(1);
    }

    auto tn = tensor_file.tensors;
    auto path = tensor_file.path.value().GetPath();

    std::vector<std::string> possible_sliced_indices = {
        "p7", "s7", "h4", "m1", "m2", "I2", "V4", "z2", "t4", "C1"};

    // since these are qubits, each sliced index correspond so two slices
    size_t slice_size = 2;

    std::vector<std::string> sliced_indices;
    for (size_t i = 0; i < indices_to_slice; i++) {
        sliced_indices.push_back(possible_sliced_indices[i]);
    }

    size_t number_of_slices = std::pow(slice_size, sliced_indices.size());

    std::cout << "sliced_indices = " << sliced_indices << std::endl;
    std::cout << "number_of_slices = " << number_of_slices << std::endl;

    std::vector<TensorNetwork<Tensor>> slices(number_of_slices);

    for (size_t i = 0; i < number_of_slices; i++) {
        slices[i] = tn;
        slices[i].SliceIndices(sliced_indices, i);
    }

    TaskBasedContractor<Tensor> contractor(num_threads);
    
    size_t shared = 0;

    for (size_t i = 0; i < number_of_slices; i++) {
        PathInfo pinfo(slices[i], path);
        auto t_shared = contractor.AddContractionTasks(slices[i], pinfo);

        if (shared < t_shared) {
            shared = t_shared;
        }
    }
    std::cout << "shared = " << shared << std::endl;

    contractor.AddDeletionTasks();

    contractor.AddReductionTask();
    auto t1 = high_resolution_clock::now();
    contractor.Contract().wait();
    auto t2 = high_resolution_clock::now();

    auto duration = duration_cast<duration<float>>(t2 - t1).count();

    std::cout << "t=" << duration << "s" << std::endl;
    auto res = contractor.GetReductionResult().GetHostDataVector();
    std::cout << "result=" << res << "" << std::endl;
}
