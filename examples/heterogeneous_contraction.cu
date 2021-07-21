/**
 * @file   heterogeneous_contraction.cu
 *
 * @brief  Contracts three tensor network files on two gpus
 *         and one cpu simultaneously
 *
 */

#include <iostream>

#include "CudaTensor.hpp"
#include "PathInfo.hpp"
#include "TaskBasedContractor.hpp"
#include "Tensor.hpp"
#include "TensorNetwork.hpp"
#include "TensorNetworkIO.hpp"

#include <cuComplex.h>
#include <taskflow/cudaflow.hpp>

// using namespace tf;

template <typename T, int device = 0> struct CudaflowContractionTask {

    std::vector<std::unique_ptr<Jet::CudaTensor<T, device>>> tensors;
    std::vector<typename Jet::CudaTensor<T, device>::CudaContractionPlan> plans;
    std::vector<tf::cudaTask> kernel_tasks;
    std::vector<T> result;
};

template <typename T, int device = 0>
void AddCudaContractionToTaskflow(
    const Jet::TensorNetwork<Jet::CudaTensor<T, device>> &tn,
    const Jet::PathInfo &path_info, tf::Taskflow &taskflow,
    CudaflowContractionTask<T, device> &gpu_task)
{
    auto &tensors = gpu_task.tensors;
    auto &plans = gpu_task.plans;
    auto &result = gpu_task.result;
    auto &kernel_tasks = gpu_task.kernel_tasks;

    using namespace Jet;
    const std::vector<PathStepInfo> &path_node_info = path_info.GetSteps();
    const std::vector<std::pair<size_t, size_t>> &path = path_info.GetPath();
    const auto &nodes = tn.GetNodes();
    size_t num_leafs = nodes.size();
    tensors.resize(path_node_info.size());
    plans.resize(path.size());

    for (size_t i = 0; i < path.size(); i++) {

        const PathStepInfo &pnia = path_node_info[path[i].first];
        const PathStepInfo &pnib = path_node_info[path[i].second];
        const PathStepInfo &pnic = path_node_info[num_leafs + i];

        if (pnia.id >= num_leafs) {
            tensors[path[i].first] =
                std::make_unique<CudaTensor<cuComplex, device>>(
                    CudaTensor<cuComplex, device>(pnia.tensor_indices,
                                                  pnia.shape));
        }
        else {
            tensors[path[i].first] =
                std::make_unique<CudaTensor<cuComplex, device>>(
                    CudaTensor<cuComplex, device>(
                        tn.GetNodes()[pnia.id].tensor));
        }

        if (pnib.id >= num_leafs) {
            tensors[path[i].second] =
                std::make_unique<CudaTensor<cuComplex, device>>(
                    CudaTensor<cuComplex, device>(pnib.tensor_indices,
                                                  pnib.shape));
        }
        else {
            tensors[path[i].second] =
                std::make_unique<CudaTensor<cuComplex, device>>(
                    CudaTensor<cuComplex, device>(
                        tn.GetNodes()[pnib.id].tensor));
        }

        tensors[num_leafs + i] =
            std::make_unique<CudaTensor<cuComplex, device>>(
                CudaTensor<cuComplex, device>(pnic.tensor_indices, pnic.shape));

        CudaTensor<cuComplex, device>::GetCudaContractionPlan(
            plans[i], *tensors[path[i].first], *tensors[path[i].second],
            *tensors[num_leafs + i]);
    }

    tf::Task task = taskflow.emplace_on(
        [&tensors, &plans, &result, &kernel_tasks, path_node_info, path,
         num_leafs, &tn](tf::cudaFlowCapturer &capturer) {
            for (int i = 0; i < path.size(); i++) {

                const PathStepInfo &pnia = path_node_info[path[i].first];
                const PathStepInfo &pnib = path_node_info[path[i].second];
                const PathStepInfo &pnic = path_node_info[num_leafs + i];

                auto tensor_a = tensors[path[i].first]->GetData();
                auto tensor_b = tensors[path[i].second]->GetData();
                auto tensor_c = tensors[num_leafs + i]->GetData();

                auto &c_plan = plans[i];
                tf::cudaTask kernel =
                    capturer.on([&, c_plan, tensor_a, tensor_b,
                                 tensor_c](cudaStream_t stream) {
                        cuComplex alpha;
                        alpha.x = 1.;
                        alpha.y = 0.;

                        cuComplex beta;
                        beta.x = 0.;
                        beta.y = 0.;

                        cutensorContraction(&c_plan.handle, &c_plan.plan,
                                            &alpha, tensor_a, tensor_b, &beta,
                                            tensor_c, tensor_c, c_plan.work,
                                            c_plan.work_size, stream);
                    });

                kernel_tasks.push_back(kernel);

                if (pnia.id >= num_leafs) {
                    kernel_tasks[pnia.id - num_leafs].precede(kernel);
                }

                if (pnib.id >= num_leafs) {
                    kernel_tasks[pnib.id - num_leafs].precede(kernel);
                }

                // copy data from gpu_data to host_data
                if (i == path.size() - 1) {
                    result.resize(tensors[pnic.id]->GetSize());
                    tf::cudaTask d2h = capturer.memcpy(
                        result.data(), tensors[pnic.id]->GetData(),
                        tensors[pnic.id]->GetSize() * sizeof(cuComplex));

                    kernel.precede(d2h);
                }
            }
        },
        device);
}

int main(int argc, char *argv[])
{
    using namespace Jet;
    using c_fp32 = cuComplex;

    if (argc != 4) {
        std::cout << "heterogeneous_contraction.cu <tensor network file 1 on gpu 0> "
                     "<tensor network file 2 on gpu 1> <tensor network file 3 on gpu 2>"
                  << std::endl;
        std::cout << "Contracts three circuits on two gpus and one cpu"
                  << std::endl;
    }

    std::string file_name_0 = argv[1];
    std::string file_name_1 = argv[2];
    std::string file_name_2 = argv[3];

    /**
    /* Load first tensor network file onto GPU 0
     */

    TensorNetworkFile<CudaTensor<cuComplex, 0>> tensor_file_0;
    try {
        std::ifstream tn_data(file_name_0);
        std::string circuit_str{std::istreambuf_iterator<char>(tn_data),
                                std::istreambuf_iterator<char>()};
        // Load data into TensorNetwork and PathInfo objects
        Jet::TensorNetworkSerializer<CudaTensor<cuComplex, 0>> serializer;
        tensor_file_0 = serializer(circuit_str, true);
    }
    catch (...) {
        std::cerr << "Please specify a valid first JSON file to contract"
                  << std::endl;
        exit(1);
    }

    Jet::TensorNetwork<CudaTensor<cuComplex, 0>> tn_0 = tensor_file_0.tensors;
    Jet::PathInfo path_0 =
        tensor_file_0.path.value(); // std::optional requires value()

    /**
     * Load second tensor network file onto GPU 1
     */

    TensorNetworkFile<CudaTensor<cuComplex, 1>> tensor_file_1;
    try {
        std::ifstream tn_data(file_name_1);
        std::string circuit_str{std::istreambuf_iterator<char>(tn_data),
                                std::istreambuf_iterator<char>()};
        // Load data into TensorNetwork and PathInfo objects
        Jet::TensorNetworkSerializer<CudaTensor<cuComplex, 1>> serializer;
        tensor_file_1 = serializer(circuit_str, true);
    }
    catch (...) {
        std::cerr << "Please specify a valid JSON file to contract"
                  << std::endl;
        exit(1);
    }

    Jet::TensorNetwork<CudaTensor<cuComplex, 1>> tn_1 = tensor_file_1.tensors;
    Jet::PathInfo path_1 =
        tensor_file_1.path.value(); // std::optional requires value()

    /**
     * Load third tensor network file onto CPU
     */

    TensorNetworkFile<Tensor<std::complex<float>>> tensor_file_2;
    try {
        std::ifstream tn_data(file_name_2);
        std::string circuit_str{std::istreambuf_iterator<char>(tn_data),
                                std::istreambuf_iterator<char>()};
        // Load data into TensorNetwork and PathInfo objects
        Jet::TensorNetworkSerializer<Tensor<std::complex<float>>> serializer;
        tensor_file_2 = serializer(circuit_str, true);
    }
    catch (...) {
        std::cerr << "Please specify a valid JSON file to contract"
                  << std::endl;
        exit(1);
    }
    Jet::TensorNetwork<Jet::Tensor<std::complex<float>>> tn_2 =
        tensor_file_2.tensors;
    Jet::PathInfo path_2 =
        tensor_file_2.path.value(); // std::optional requires value()

    tf::Taskflow taskflow;

    /* set up gpu 0 contraction task */
    CudaflowContractionTask<cuComplex, 0> gpu_task_0;
    AddCudaContractionToTaskflow<cuComplex, 0>(tn_0, path_0, taskflow,
                                               gpu_task_0);

    /* set up gpu 1 contraction task */
    CudaflowContractionTask<cuComplex, 1> gpu_task_1;
    AddCudaContractionToTaskflow<cuComplex, 1>(tn_1, path_1, taskflow,
                                               gpu_task_1);

    /* set up cpu contraction task */
    Jet::TaskBasedContractor<Jet::Tensor<std::complex<float>>> contractor;
    contractor.AddContractionTasks(tn_2, path_2);

    // Add gpu task graph to cpu task graph
    contractor.AddTaskflow(taskflow);

    /* Contract on all devices */
    contractor.Contract().wait();

    /* Display results */
    auto result0 = gpu_task_0.result;
    std::cout << "GPU 0 result = " << result0[0].x << " " << result0[0].y
              << std::endl;

    auto result1 = gpu_task_1.result;
    std::cout << "GPU 1 result = " << result1[0].x << " " << result1[0].y
              << std::endl;

    auto result2 = contractor.GetResults()[0];
    std::cout << "CPU result = " << result2 << std::endl;

    return 0;
}
