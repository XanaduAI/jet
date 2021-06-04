#include "Tensor.hpp"
#include "TensorNetwork.hpp"
#include "TensorNetworkIO.hpp"
#include "PathInfo.hpp"

#include <chrono>
#include <string>
#include <vector>
#include <fstream>

using namespace Jet;

template<class T>
std::unordered_map<T, size_t> SortByFreq(std::vector<T>& v)
{
    std::unordered_map<T, size_t> count;

    for (T i : v) {
        count[i]++;
    }

    std::sort(
        v.begin(), 
        v.end(),
        [&count](T const& a, T const& b) {
        if (a == b) {
            return false;
        }
        if (count[a] > count[b]) {
            return true;
        }
        else if (count[a] < count[b]) {
            return false;
        }
        return a < b;
    });

    v.erase( unique( v.begin(), v.end() ), v.end() );
    
    return count;
}


struct ContractionCosts {

    double total_flops;
    double total_shared_work_per_amp_sliced_flops;
    double total_shared_work_per_amp_not_sliced_flops;
    double total_shared_work_per_slice_flops;

    double total_memory_write;
    double total_shared_work_per_amp_sliced_memory;
    double total_shared_work_per_amp_not_sliced_memory;
    double total_shared_work_per_slice_memory;

    double total_slice_unshared_work_dependent_memory;
    double largest_intermediate_rank;
    size_t total_shared_slice_tasks;
    size_t total_shared_amp_tasks;

    ContractionCosts() {}

    ContractionCosts(const PathInfo &pinfo)
    {

        total_flops = 0;
        total_shared_work_per_amp_sliced_flops = 0;
        total_shared_work_per_amp_not_sliced_flops = 0;
        total_shared_work_per_slice_flops = 0;

        total_shared_work_per_amp_sliced_memory = 0;
        total_shared_work_per_amp_not_sliced_memory = 0;
        total_shared_work_per_slice_memory = 0;

        total_slice_unshared_work_dependent_memory = 0;
        total_memory_write = 0;
        largest_intermediate_rank = 0;
        total_shared_slice_tasks = 0;
        total_shared_amp_tasks = 0;
        size_t num_leafs = pinfo.GetNumLeaves();

        auto &step_data = pinfo.GetSteps();

        for (size_t i = 0; i < num_leafs; i++) {
            double path_mem = pinfo.GetPathStepMemory(i);
            total_memory_write += path_mem;

            if (step_data[i].parent != -1 &&
                step_data[step_data[i].parent].name.find("(") !=
                    std::string::npos) {
                total_slice_unshared_work_dependent_memory += path_mem;
            }
        }

        for (size_t i = num_leafs; i < step_data.size(); i++) {
            double path_flops = pinfo.GetPathStepFlops(i);
            double path_mem = pinfo.GetPathStepMemory(i);
            total_memory_write += path_mem;
            total_flops += path_flops;

            if (step_data[i].tensor_indices.size() >
                largest_intermediate_rank) {
                largest_intermediate_rank = step_data[i].tensor_indices.size();
            }

            // does not contain a slice related indice
            if (step_data[i].name.find("(") == std::string::npos) {
                total_shared_slice_tasks++;
                total_shared_work_per_slice_flops += path_flops;
                total_shared_work_per_slice_memory += path_mem;

                if (step_data[i].parent != -1 &&
                    step_data[step_data[i].parent].name.find("(") !=
                        std::string::npos) {
                    total_slice_unshared_work_dependent_memory += path_mem;
                }
            }

            // does not contain an amplitude related indice
            if (step_data[i].name.find("[") == std::string::npos) {
                total_shared_amp_tasks++;
                // does not contain a slice related indice
                if (step_data[i].name.find("(") == std::string::npos) {
                    total_shared_work_per_amp_not_sliced_flops += path_flops;
                    total_shared_work_per_amp_not_sliced_memory += path_mem;
                }
                // does contain a slice related indice;
                else {
                    total_shared_work_per_amp_sliced_flops += path_flops;
                    total_shared_work_per_amp_sliced_memory += path_mem;
                }
            }
        }
    }

    double GetSharedWorkPerSliceFraction()
    {
        return total_shared_work_per_slice_flops / total_flops;
    }

    double GetSharedWorkPerAmplitudeFraction(double num_slices)
    {
        double fansl = total_shared_work_per_amp_not_sliced_flops / total_flops;
        double fasl = total_shared_work_per_amp_sliced_flops / total_flops;
        double fsl = GetSharedWorkPerSliceFraction();

        return (fansl * total_flops + num_slices * (fasl)*total_flops) /
               (fsl * total_flops + num_slices * (1 - fsl) * total_flops);
    }

    double GetTotalTaskFlopsForBatchAmplitude(double num_slices,
                                              double num_amps)
    {
        double f_slice = GetSharedWorkPerSliceFraction();
        double f_amp = GetSharedWorkPerAmplitudeFraction(num_slices);
        double slice_flops = total_flops;
        double amp_flops =
            f_slice * slice_flops + num_slices * (1. - f_slice) * slice_flops;
        double batch_flops =
            f_amp * amp_flops + num_amps * (1. - f_amp) * amp_flops;
        return batch_flops;
    }

    double GetTotalNaiveFlopsForOneAmplitude(double num_slices)
    {
        return total_flops * num_slices;
    }

    void PrintContractionCosts(int num_slices, int num_amps)
    {
        double f_slice = GetSharedWorkPerSliceFraction();
        double f_amp = GetSharedWorkPerAmplitudeFraction(num_slices);
        double slice_flops = total_flops;
        double amp_flops =
            f_slice * slice_flops + num_slices * (1. - f_slice) * slice_flops;
        double batch_flops =
            f_amp * amp_flops + num_amps * (1. - f_amp) * amp_flops;
        double naive_flops = num_amps * num_slices * total_flops;
        double total_mem_gb =
            sizeof(std::complex<double>) * total_memory_write / (1.e9);
        double total_shared_slice_mem_gb = sizeof(std::complex<double>) *
                                           total_shared_work_per_slice_memory /
                                           (1.e9);
        double total_dependent_slice_mem_gb =
            sizeof(std::complex<double>) *
            total_slice_unshared_work_dependent_memory / (1.e9);


	std::cout << "num_amps = " << num_amps << std::endl;
	std::cout << "f_slice = " << f_slice << std::endl;
	std::cout << "amp_flops = " << amp_flops << std::endl;
	std::cout << "naive_flops/num_amps = " << naive_flops/(double)num_amps << std::endl;
	
        // std::cout
        //     << "num_slices | num_amps | total_memory_write (GB) | "
        //        "total_shared_slice_memory | total_dependent_slice_mem_gb "
        //        "|f_slice | f_amp | slice_flops | amp_flops | "
        //        "num_slices*(1.-f_slice)*slice_flops | batch_flops | "
        //        "naive_flops | total_shared_slice_task | total_shared_amp_tasks "
        //     << std::endl;
        // printf("%d | %d | %e | %e | %e | %e | %e | %e | %e | %e | %e | %e | %d "
        //        "| %d \n",
        //        num_slices, num_amps, total_mem_gb, total_shared_slice_mem_gb,
        //        total_dependent_slice_mem_gb, f_slice, f_amp, slice_flops,
        //        amp_flops, num_slices * (1. - f_slice) * slice_flops,
        //        batch_flops, naive_flops, (int)total_shared_slice_tasks,
        //        (int)total_shared_amp_tasks);
    }
};


int main(int argc, char *argv[])
{
  using namespace Jet;
  using namespace Jet::Utilities;
  using c_fp32 = std::complex<float>;
  using namespace std::chrono;

  std::string file_name = ""; 
  if (argc > 1){
    file_name = argv[1];
  }
  else{
    std::cerr << "Please specify a JSON file to contract" << std::endl;
    exit(1);
  }

  using Tensor = Jet::Tensor<c_fp32>;
  
  TensorNetworkFile<Tensor> tensor_file;
  try {
    std::ifstream tn_data(file_name);
    std::string m10_str {std::istreambuf_iterator<char>(tn_data), std::istreambuf_iterator<char>()};
    // Load data into TensorNetwork and PathInfo objects
    Jet::TensorNetworkSerializer<Tensor> serializer;
    tensor_file = serializer(m10_str);
  }
  catch(...){
    std::cerr << "Please specify a valid JSON file to contract" << std::endl;
    exit(1);
  }

  auto tn = tensor_file.tensors;
  auto path = tensor_file.path.value().GetPath();
  int num_sliced_indices = std::stoi(argv[2]);

  std::vector<std::string> slices;
  std::vector<std::string> indices_already_sliced = {};


  std::cout << "tensor_file.path.value().GetTotalflops = " << tensor_file.path.value().GetTotalFlops() << std::endl;
  
  for (int i = 0; i < num_sliced_indices; i++){
    PathInfo pinfo(tn, path);
    std::vector<std::string> indices_from_biggest =
      pinfo.GetIndicesOfBiggestRankTensors();

    // std::cout << "indices_from_biggest = " << indices_from_biggest.size() << std::endl;
    size_t biggest_rank = pinfo.GetLargestIntermediaryTensorRank();
    auto indfreq = SortByFreq(indices_from_biggest);
    
    std::string best_index;
    double fraction = 0;
    double total_flops = 1e100;
    double task_flops = 1e100;
    double largest_rank = 0;

    std::vector<std::string> indices_that_reduce_rank;
    
    for (int j = 0; j < indices_from_biggest.size(); j++){
      auto tn2 = tn;
      tn2.SliceIndices({indices_from_biggest[j]}, 0);
      // tn2.LabelFinalStateIndices("PSI_f", 0);
      // tn2.PrintNetwork();
      PathInfo pinfo(tn2, path);
      ContractionCosts c(pinfo);
      if (c.largest_intermediate_rank < biggest_rank){
	indices_that_reduce_rank.push_back(indices_from_biggest[j]);
      }
    }

    if (indices_that_reduce_rank.size() == 0){
      size_t top_freq = indfreq[indices_from_biggest[0]];
      for (auto & j : indices_from_biggest){
	if (indfreq[j] == top_freq){
	  indices_that_reduce_rank.push_back(j);
	}
      }
    }
    
    std::cout << "indices_that_reduce_rank.size() = " << indices_that_reduce_rank.size() << std::endl;
    for (int j = 0; j < indices_that_reduce_rank.size(); j++){
      auto tn2 = tn;
      tn2.SliceIndices({indices_that_reduce_rank[j]}, 0);
      PathInfo pinfo(tn2, path);
      ContractionCosts c(pinfo);

      if (c.GetSharedWorkPerSliceFraction() > fraction){
	fraction = c.GetSharedWorkPerSliceFraction();
	best_index = indices_that_reduce_rank[j];
	total_flops = c.total_flops;
	largest_rank = c.largest_intermediate_rank;
      }
    }
    std::cout << best_index << " | " << fraction << " | " << total_flops << " | " << largest_rank << std::endl;
    tn.SliceIndices({best_index}, 0);
    slices.push_back(best_index);
  }

  // tn.LabelFinalStateIndices("PSI_f", 0);
  double num_slices = std::pow(2, indices_already_sliced.size() + num_sliced_indices);
  PathInfo pinfo(tn, path);
  ContractionCosts c(pinfo);
  c.PrintContractionCosts(num_slices,13000);
  
  // std::cout << "Total Naive Flops = " << c.GetTotalNaiveFlopsForOneAmplitude(num_slices) << std::endl;
  // std::cout << "Total Task Flops = " << c.GetTotalTaskFlopsForOneAmplitude(num_slices) << std::endl;
  // std::cout << "Fugaku = " << c.GetTotalTaskFlopsForOneAmplitude(num_slices)/(4.15e17) << std::endl;
  // std::cout << "Largest Int Rank = " << c.largest_intermediate_rank << std::endl;
  std::cout << "num_slices = " << num_slices << std::endl;
  std::cout << "slices = " << slices << std::endl;

  // pinfo.PrintPath();
  
}
