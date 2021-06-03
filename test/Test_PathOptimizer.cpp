#include "path_optimizer/GreedyPathOptimizer.hpp"
#include "Tensor.hpp"
#include "TensorNetwork.hpp"
#include "TensorNetworkIO.hpp"


#include <ensmallen.hpp>

template <typename Tensor>
class GreedyFunction
{
private:
  Jet::TensorNetwork<Tensor> tn_;
 public:

  GreedyFunction(Jet::TensorNetwork<Tensor> & tn){
    tn_ = tn;
  }
  // This returns f(x) = 2 |x|^2.
  double Evaluate(const arma::mat& x)
  {
    double alpha = x[0];
    double temperature = x[1];
    Jet::GreedyPathOptimizer gpo(alpha,temperature);
    auto path = gpo.Search(tn_);
    std::cout << "path.GetTotalFlops() = " << path.GetTotalFlops() << std::endl;
    return path.GetTotalFlops();
  }
};


int main(int argc, char *argv[])
{
  using Tensor= Jet::Tensor<std::complex<float>>;

  // Tensor tens1({"a","b","c"}, {4,4,4});
  // Tensor tens2({"a","f","g"}, {4,2,6});
  // Tensor tens3({"f","b","k"}, {2,4,3});
  // Tensor tens4({"k","c"}, {3,4});
  // Tensor tens5({"g"}, {6});

  
  // Jet::TensorNetwork<Tensor> tn;
  // tn.AddTensor(tens1, {});
  // tn.AddTensor(tens2, {});
  // tn.AddTensor(tens3, {});
  // tn.AddTensor(tens4, {});
  // tn.AddTensor(tens5, {});

  // arma::mat x("0.1 0.1");
  // ens::CNE optimizer;
  // GreedyFunction<Tensor> f(tn);
  // optimizer.Optimize(f,x);
  // std::cout << "x = " << x << std::endl;


  // Jet::GreedyPathOptimizer gpo(x[0], x[1]);
  // auto flops = gpo.Search(tn).GetTotalFlops();
  // std::cout << "flops = " << flops << std::endl;


  // std::cout << "tn.NumTensors = " << tn.NumTensors() << std::endl;
  // tn.RankSimplify({});

  // std::cout << "tn.NumTensors = " << tn.NumTensors() << std::endl;
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

  
  // Jet::TensorNetworkSerializer<Tensor> serializer;
  // auto file = serializer("test.json");
  auto tnm10 = tensor_file.tensors;

  std::cout << "tnm10.NumTensors = " << tnm10.NumTensors() << std::endl;
  tnm10.RankSimplify({});
  std::cout << "tnm10.NumTensors = " << tnm10.NumTensors() << std::endl;


  arma::mat x("0.1 0.1");
  ens::CNE optimizer;
  GreedyFunction<Tensor> f(tnm10);
  optimizer.Optimize(f,x);
  std::cout << "x = " << x << std::endl;


  Jet::GreedyPathOptimizer gpo(x[0], x[1]);
  auto flops = gpo.Search(tnm10).GetTotalFlops();
  std::cout << "flops = " << flops << std::endl;
  
  
  // double alpha = 1.0;
  // double temperature = 1.0;

  // Jet::GreedyPathOptimizer gpo(alpha,temperature);
  // auto path = gpo.Search(tn).GetPath();
  // std::cout << "path.size() = " << path.size() << std::endl;
  // for (auto i : path){
  //   std::cout << i.first << " " << i.second << std::endl;
  // }
  
  return 0;
}
