#pragma once

#include "TensorNetwork.hpp"
#include "Tensor.hpp"
#include "PathInfo.hpp"


namespace Jet {

class GreedyPathOptimizer {

  private:
    double alpha_;
    double temperature_;
    double amplitude_modifier_;
    size_t seed_;
    size_t max_rejections_;

  public:
    GreedyPathOptimizer
    (
     double alpha,
     double temperature,
     double amplitude_modifier = 1.0,
     size_t max_rejections = 100000,
     size_t seed = 0
    )
    {
        alpha_ = alpha;
        amplitude_modifier_ = amplitude_modifier;
        temperature_ = temperature;
        seed_ = seed;
        max_rejections_ = max_rejections;
    }

    template <typename Tensor>
    void UpdateIndexMapAfterContraction_(
        std::unordered_map<std::string, typename TensorNetwork<Tensor>::Edge> &index_to_edge_map,
        const PathStepInfo &node_a,
	const PathStepInfo &node_b,
	const PathStepInfo &new_node)
    {
        using namespace Jet::Utilities;
        const std::vector<std::string> &new_indices = new_node.node_indices;

        // Update edges with new tensor info
        for (auto &ind : new_node.node_indices) {
            if (index_to_edge_map.count(ind)) {
                auto &edge = index_to_edge_map.at(ind);
                auto &nodes = edge.node_ids;
                for (auto &n : nodes) {
                    if (n == node_a.id or n == node_b.id) {
                        n = new_node.id;
                    }
                }
            }
        }

        const std::vector<std::string> &contracted_indices =
            VectorIntersection(node_a.tensor_indices, node_b.tensor_indices);

        // delete contracted indices and tensors
        for (auto &i : contracted_indices) {
            index_to_edge_map.erase(i);
        }
    }
  
  size_t GetSizeFromTensorIndices_(const std::vector<std::string> & tensor_indices,
				  const std::unordered_map<std::string, size_t> map){

    size_t multiplier = 1;
    for (auto i : tensor_indices){
      multiplier *= map.at(i);
    }
    return multiplier;
    
  }

    double AlphaFlopCost_(const PathInfo &pinfo,
                          const std::pair<size_t, size_t> &node_pair)
    {

        const auto &step_data = pinfo.GetSteps();
        using namespace Jet::Utilities;
        const PathStepInfo &a = step_data[node_pair.first];
        const PathStepInfo &b = step_data[node_pair.second];

        auto contracted_indices =
            VectorIntersection(a.tensor_indices, b.tensor_indices);
        auto tensor_indices = VectorSubtraction(
            VectorConcatenation(a.tensor_indices, b.tensor_indices),
            contracted_indices);

        double a_size = GetSizeFromTensorIndices_(a.tensor_indices,pinfo.GetIndexSizes());
        double b_size = GetSizeFromTensorIndices_(b.tensor_indices,pinfo.GetIndexSizes());
        double c_size = GetSizeFromTensorIndices_(tensor_indices,pinfo.GetIndexSizes());
        return c_size - alpha_ * (a_size + b_size);
    }

    template <typename Tensor>
    void GetEdgeProbs_(
        const PathInfo &pinfo,
        const std::unordered_map<
            std::string, typename TensorNetwork<Tensor>::Edge> &edge_map,
        std::vector<double> &probs, std::vector<std::string> &indices)
    {
        probs.resize(edge_map.size());
        indices.resize(edge_map.size());
        std::vector<bool> is_shared(edge_map.size());
        size_t k = 0;
        double arg_min = 0.;

        for (auto i : edge_map) {
            size_t n0 = i.second.node_ids[0];
            size_t n1 = i.second.node_ids[1];
            // is_shared[k] = pinfo.IsPathStepShared({n0,n1});
            double arg = AlphaFlopCost_(pinfo, {n0, n1}) / temperature_;
            if (k == 0)
                arg_min = arg;
            if (arg < arg_min)
                arg_min = arg;
            probs[k] = arg;
            indices[k] = i.first;
            k++;
        }

        double temperature =
            temperature_ * (fabs(arg_min) > 1 ? fabs(arg_min) : 1);
        double prob_total = 0.;

        k = 0;
        for (auto &i : probs) {
            i = exp(-(i - arg_min) / temperature);
            prob_total += i;
            k++;
        }
        for (auto &i : probs) {
            i /= prob_total;
        }
    }

    template <typename Tensor> PathInfo Search(const TensorNetwork<Tensor> &tn)
    {

        PathInfo pinfo(tn, {});
        std::vector<double> probs;
        std::vector<std::string> indices;
        auto index_to_edge_map = tn.GetIndexToEdgeMap();

        std::mt19937 gen(
            seed_); // Standard mersenne_twister_engine seeded with rd()
        std::uniform_real_distribution<double> unif(0., 1.0);
        size_t counter = tn.NumTensors();

        while (!index_to_edge_map.empty()) {
            GetEdgeProbs_<Tensor>(pinfo, index_to_edge_map, probs, indices);
            std::uniform_int_distribution<int> int_dist(0, indices.size() - 1);
            bool found = false;
            size_t rejections = 0;
            while (!found) {
                int i = int_dist(gen);
                auto &edge = index_to_edge_map.at(indices[i]);
                auto nodes = edge.node_ids;
                size_t node0 = nodes[0];
                size_t node1 = nodes[1];
                double deviate = unif(gen);		
                if (deviate < probs[i] || rejections > max_rejections_) {
                    found = true;
		    pinfo = PathInfo(pinfo, {{node0, node1}});
                    UpdateIndexMapAfterContraction_<Tensor>(
                        index_to_edge_map, pinfo.GetSteps()[node0],
                        pinfo.GetSteps()[node1],
                        pinfo.GetSteps()[counter]);
                    counter++;
                    break;
                }
                else {
                    rejections++;
                }
            }
        }
        return pinfo;
    }
};

}; // namespace Jet
