#include <iostream>
#include "Jet.hpp"

int main()
{
  using namespace Jet;
  CudaTensor tensor1_dev({"a", "b", "c"}, {2, 3, 5});
  CudaTensor tensor2_dev({"d", "b", "c"}, {5, 3, 5});
  CudaTensor tensor3_dev({"a","d"}, {2,5});
  
  tensor1_dev.FillRandom(7);
  tensor2_dev.FillRandom(7);
  tensor3_dev.FillRandom(7);
  
  Tensor<std::complex<float>> tensor1_host_conv(tensor1_dev);
  Tensor<std::complex<float>> tensor2_host_conv(tensor2_dev);
  Tensor<std::complex<float>> tensor3_host_conv(tensor3_dev);
    
  TensorNetwork<Tensor<std::complex<float>>> tn;
  tn.AddTensor(tensor1_host_conv,{""});
  tn.AddTensor(tensor2_host_conv,{""});
  tn.AddTensor(tensor3_host_conv,{""});
  auto res = tn.Contract();  

  TensorNetwork<CudaTensor<cuComplex>> tn_cuda;
  tn_cuda.AddTensor(tensor1_dev,{""});
  tn_cuda.AddTensor(tensor2_dev,{""});
  tn_cuda.AddTensor(tensor3_dev,{""});
  auto res_cuda = tn_cuda.Contract();

  std::vector<cuComplex> res_cuda_host(res_cuda.GetSize());
  res_cuda.CopyGpuDataToHost(res_cuda_host.data());
  
  std::cout << res << std::endl;
  std::cout << res_cuda_host[0].x << " " << res_cuda_host[0].y << std::endl;
    
  return 0;
}