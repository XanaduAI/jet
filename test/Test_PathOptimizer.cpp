#include "path_optimizer/GreedyPathOptimizer.hpp"
#include "Tensor.hpp"
#include "TensorNetwork.hpp"

int main(int argc, char *argv[])
{
  using Tensor= Jet::Tensor<std::complex<float>>;

  Tensor tens1({"a","b","c"}, {4,4,4});
  Tensor tens2({"a","f","g"}, {4,2,6});
  Tensor tens3({"f","b","k"}, {2,4,3});
  Tensor tens4({"k","g","c"}, {3,6,4});
  
  Jet::TensorNetwork<Tensor> tn;
  tn.AddTensor(tens1, {});
  tn.AddTensor(tens2, {});
  tn.AddTensor(tens3, {});
  tn.AddTensor(tens4, {});

  double alpha = 1.0;
  double temperature = 1.0;
  
  Jet::GreedyPathOptimizer gpo(alpha,temperature);
  auto path = gpo.Search(tn).GetPath();
  std::cout << "path.size() = " << path.size() << std::endl;
  for (auto i : path){
    std::cout << i.first << " " << i.second << std::endl;
  }

  
  return 0;
}
