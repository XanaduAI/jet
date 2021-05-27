#pragma once

#include <hptt.h>
#include <iostream>
#include <memory>
#include <unordered_map>

namespace Jet {

class HpttPermuter {
  public:
    HpttPermute() {}

    template <class DataType>
    void Transpose(const std::vector<DataType> &data,
                                    const std::vector<size_t> &shape,
                                    std::vector<DataType> data_out,
                                    const std::vector<std::string> &old_indices,
                                    const std::vector<std::string> &new_indices)
    {
        using namespace Jet::Utilities;
        data_out = data;

        if (new_indices == old_indices)
            return;

        std::vector<int> perm(old_indices.size());

        for (size_t i = 0; i < old_indices.size(); i++) {
            for (size_t j = 0; j < new_indices.size(); j++) {
                if (old_indices[i] == new_indices[j]) {
                    perm[i] = j;
                    break;
                }
            }
        }

        std::vector<int> local_shape(shape.begin(), shape.end());

        auto plan = hptt::create_plan(
            perm.data(), local_shape.size(),
            1, data.data(),
            local_shape.data(), nullptr,
            static_cast<typename DataType::value_type>(0.0), data_out.data(),
            nullptr, hptt::PATIENT, 1, nullptr, true);

        plan->execute();

        return data_out;
    }

    template <class DataType>
    std::vector<DataType> Transpose(const std::vector<DataType> &data,
                                    const std::vector<size_t> &shape,
                                    const std::vector<std::string> &old_indices,
                                    const std::vector<std::string> &new_indices)
    {
        auto data_out(data);
        Transpose(data, shape, data_out, old_indices, new_indices);
        return data_out;
    }
};

} // namespace Jet
