#include "CudaTensor.hpp"
#include "PathInfo.hpp"
#include "TensorNetwork.hpp"
#include <cuComplex.h>
#include <vector>

#include <memory>

#include <taskflow/taskflow.hpp>

// module load cuda && module load gcc && nvcc -o Test_GpuTask
// Test_GpuContractionTaskCreator.cpp -I./taskflow -I./cutt/include -L./cutt/lib
// -lcutt -lcuda -std=c++14 --extended-lambda -lcublas
namespace Jet {

std::vector<int> ConvertSizeVecToIntVec(const std::vector<size_t> size_vec)
{
    std::vector<int> int_vec(size_vec.size());
    for (int i = 0; i < int_vec.size(); i++) {
        int_vec[i] = size_vec[i];
    }
    return int_vec;
}

template <typename CpuTensor>
void CopyCpuTensorToGpuTensor(const CpuTensor &cpu_tensor,
                              CudaTensor<cuComplex> &gpu_tensor)
{
    size_t gpu_size = gpu_tensor.GetSize();
    cuComplex *cpu_tensor_data = new cuComplex[gpu_size];

    for (size_t i = 0; i < gpu_size; i++) {
        cpu_tensor_data[i].x = cpu_tensor[i].real();
        cpu_tensor_data[i].y = cpu_tensor[i].imag();
    }

    gpu_tensor.CopyHostDataToGpu(cpu_tensor_data);
    delete[] cpu_tensor_data;
}

template <typename CpuTensor>
void CopyGpuTensorToCpuTensor(CudaTensor<cuComplex> &gpu_tensor,
                              CpuTensor &cpu_tensor)
{
    size_t gpu_size = gpu_tensor.GetSize();
    cuComplex *cpu_tensor_data = new cuComplex[gpu_size];

    gpu_tensor.CopyGpuDataToHost(cpu_tensor_data);

    for (int i = 0; i < gpu_size; i++) {
        cpu_tensor[i] =
            std::complex<float>(cpu_tensor_data[i].x, cpu_tensor_data[i].y);
    }

    delete[] cpu_tensor_data;
}

template <typename CpuTensor> class GpuContractionTaskCreator {

  private:
    std::vector<std::unique_ptr<CudaTensor<cuComplex>>> tensors_;
    std::vector<CudaContractionPlan> plans_;
    std::vector<std::pair<size_t, size_t>> path_;
    tf::Task task_;
    size_t num_leafs_;

  public:
    GpuContractionTaskCreator(TensorNetwork<CpuTensor> &tn, PathInfo &path_info)
    {
        const std::vector<PathStepInfo> &path_node_info = path_info.GetSteps();
        const std::vector<std::pair<size_t, size_t>> &path =
            path_info.GetPath();
        path_ = path;
        const auto &nodes = tn.GetNodes();
        size_t num_leafs = nodes.size();
        bool store_transpose = true;
        tensors_.resize(path_node_info.size());
        plans_.resize(path.size());
        num_leafs_ = num_leafs;

        for (int i = 0; i < path.size(); i++) {

            const PathStepInfo &pnia = path_node_info[path[i].first];
            const PathStepInfo &pnib = path_node_info[path[i].second];
            const PathStepInfo &pnic = path_node_info[num_leafs + i];

            std::cout << "i = " << i << std::endl;

            std::cout << "problem after 1" << std::endl;
            std::cout << "pnia.shape = " << pnia.shape << std::endl;
            std::cout << "pnia.tensor_indices = " << pnia.tensor_indices
                      << std::endl;
            std::cout << "tensors_.size() = " << tensors_.size() << std::endl;
            std::cout << "path[i].first = " << path[i].first << std::endl;
            tensors_[path[i].first] = std::make_unique<CudaTensor<cuComplex>>(
                CudaTensor<cuComplex>());
            tensors_[path[i].first].get()->SetIndicesShapeAndMemory(
                pnia.tensor_indices, pnia.shape, store_transpose, -1);
            std::cout << "problem after 2" << std::endl;

            tensors_[path[i].second] = std::make_unique<CudaTensor<cuComplex>>(
                CudaTensor<cuComplex>());
            tensors_[path[i].second].get()->SetIndicesShapeAndMemory(
                pnib.tensor_indices, pnib.shape, store_transpose, -1);

            std::cout << "problem after 3" << std::endl;
            tensors_[num_leafs + i] = std::make_unique<CudaTensor<cuComplex>>(
                CudaTensor<cuComplex>());
            tensors_[num_leafs + i].get()->SetIndicesShapeAndMemory(
                pnic.tensor_indices, (pnic.shape), store_transpose, -1);

            std::cout << "problem after 4" << std::endl;

            if (pnia.id < num_leafs) {
                CopyCpuTensorToGpuTensor(nodes[pnia.id].tensor,
                                         *tensors_[path[i].first]);
            }
            std::cout << "problem after 5" << std::endl;

            if (pnib.id < num_leafs) {
                CopyCpuTensorToGpuTensor(nodes[pnib.id].tensor,
                                         *tensors_[path[i].second]);
            }
            std::cout << "problem after 6" << std::endl;

            plans_[i] = GetContractionPlan(*tensors_[path[i].first],
                                           *tensors_[path[i].second]);
        }
    }

    void AddContractionTask(tf::Taskflow &task_flow)
    {
        task_ = task_flow.emplace([this]() {
            for (int i = 0; i < path_.size(); i++) {
                Contract(*tensors_[path_[i].first], *tensors_[path_[i].second],
                         *tensors_[num_leafs_ + i], plans_[i]);
            }
        });
    }

    CpuTensor GetResult()
    {
        CudaTensor<cuComplex> &gpu_tensor =
            *tensors_[num_leafs_ + path_.size() - 1];
        CpuTensor cpu_tensor(gpu_tensor.GetIndices, gpu_tensor.GetShape());
        CopyGpuTensorToCpuTensor(gpu_tensor, cpu_tensor);
    }
};

}; // namespace Jet
