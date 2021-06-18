#pragma once

#include <complex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "external/nlohmann/json.hpp"

#include "PathInfo.hpp"
#ifdef CUTENSOR
#include "CudaTensor.hpp"
#endif
#include "Tensor.hpp"
#include "TensorNetwork.hpp"

namespace Jet {

using json = nlohmann::json;

/**
 * @brief `%TensorNetworkFile` is a POD that contains the possible contents
 *         of a tensor network file.
 */
template <class Tensor> struct TensorNetworkFile {
    /// Optional contraction path in the tensor network file.
    std::optional<PathInfo> path;
    /// Tensor network in the tensor network file.
    TensorNetwork<Tensor> tensors;
};

/**
 * @brief `%TensorFileException` is thrown when the contents of a tensor network
 *        file are invalid.
 */
class TensorFileException : public Exception {
  public:
    /**
     * @brief Constructs a new `%TensorFileException` exception.
     *
     * @param what_arg Error message explaining what went wrong while loading a
     *                 tensor network file.
     */
    explicit TensorFileException(const std::string &what_arg)
        : Exception("Error parsing tensor network file: " + what_arg){};

    /**
     * @see TensorFileException(const std::string&).
     */
    explicit TensorFileException(const char *what_arg)
        : TensorFileException(std::string(what_arg)){};
};

/**
 * @brief `%TensorNetworkSerializer` is a functor class for serializing
 * and deserializing `TensorNetwork` and `PathInfo` to/from a JSON document.
 *
 * If called with an instance of `TensorNetwork` and, optionally, `PathInfo`,
 * will return a JSON string representing the tensor network and the path.
 *
 * If called with a string, it will parse it as a JSON object and return a `
 * TensorNetworkFile`. The object must contain a `"tensors"` key containing
 * an array of tensors. It may optionally contain a `"path"` key describing
 * a contraction path.
 *
 * The elements of `"tensors"` are arrays which contain 4 arrays specifying
 * the tensors. The arrays contain the tags, ordered indices, shape and
 * elements of the tensor, respectively. Each element is an array of size two
 * representing the real and imaginary components of a complex number.
 * For example, the array
 *
 * ```
 * [
 *  ["A", "hermitian"],
 *  ["a", "b"],
 *  [2, 2],
 *  [[1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [1.0, 0.0]]
 * ]
 * ```
 *
 * corresponds to the 2x2 matrix:
 *
 *  \f$\begin{bmatrix}1 & i \\ -i & 1 \end{bmatrix}\f$
 *
 * where "a" is the row index and "b" is the column index.
 *
 * The "path" value is a list of integer pairs e.g `[i, j]` where `i` and `j`
 * are the indexes of two tensors in the tensor array (or the index of an
 * intermediate tensor).
 *
 * @tparam Tensor Type of the tensor in the tensor network.
 */
template <class TensorType> class TensorNetworkSerializer {
  public:
    /**
     * @brief Constructs a new `%TensorNetworkSerializer`.
     *
     * @param indent Indent level to use when serializing to json. Default value
     *               will use the most compact representation.
     */
    TensorNetworkSerializer<TensorType>(int indent = -1)
        : indent(indent), js(json::object())
    {
    }

    /**
     * @brief Dump tensor network and path to string.
     *
     * @return JSON string representing tensor network and path
     */
    std::string operator()(const TensorNetwork<TensorType> &tn,
                           const PathInfo &path)
    {
        js["path"] = path.GetPath();

        return operator()(tn);
    }

    /**
     * @brief Dump tensor network to string.
     *
     * @return JSON string representing tensor network.
     */
    std::string operator()(const TensorNetwork<TensorType> &tn)
    {
        js["tensors"] = json::array();

        for (const auto &node : tn.GetNodes()) {
            js["tensors"].push_back(TensorToJSON_(node.tensor, node.tags));
        }

        std::string ret = js.dump(indent);
        js = json::object();

        return ret;
    }

    /**
     * @brief Load tensor network file from JSON string.
     *
     * Raises exception if string is invalid JSON, or if it does not
     * describe a valid tensor network.
     *
     * @return TensorNetworkFile containing contents of js_str
     */
    TensorNetworkFile<TensorType> operator()(std::string js_str,
                                             bool col_major = false)
    {
        TensorNetworkFile<TensorType> tf;
        LoadAndValidateJSON_(js_str);

        size_t i = 0;
        for (auto &js_tensor : js["tensors"]) {
            auto data = TensorDataFromJSON_<typename TensorType::scalar_type_t>(
                js_tensor[3], i);

            if (col_major) {
                std::vector<std::string> rev_idx(js_tensor[1].rbegin(),
                                                 js_tensor[1].rend());
                std::vector<size_t> rev_shape(js_tensor[2].rbegin(),
                                              js_tensor[2].rend());
                tf.tensors.AddTensor(TensorType(rev_idx, rev_shape, data),
                                     js_tensor[0]);
            }
            else
                tf.tensors.AddTensor(
                    TensorType(js_tensor[1], js_tensor[2], data), js_tensor[0]);
            i++;
        }

        if (js.find("path") != js.end()) {
            tf.path = PathInfo(tf.tensors, js["path"]);
        }

        js = json::object();

        return tf;
    }

  private:
    /// Indent level for string.
    int indent;

    /// JSON data for (de)serialization.
    json js;

    /**
     * @brief Parse json string and check root object keys
     * are correct.
     *
     * Throw json::exception if string is invalid json,
     * TensorFileException if it does not have the correct
     * keys.
     */
    void LoadAndValidateJSON_(const std::string &js_str)
    {
        js = json::parse(js_str); // throws json::exception if invalid json

        if (!js.is_object()) {
            throw TensorFileException("root element must be an object.");
        }

        if (js.find("tensors") == js.end()) {
            throw TensorFileException("root object must contain 'tensors' key");
        }
    }

    /**
     * @brief Convert Tensor to json array format.
     */
    static json TensorToJSON_(const TensorType &tensor,
                              const std::vector<std::string> &tags)
    {
        auto js_tensor = json::array();

        js_tensor.push_back(tags);
        if constexpr (std::is_same_v<TensorType,
                                     Jet::Tensor<std::complex<float>>> ||
                      std::is_same_v<TensorType,
                                     Jet::Tensor<std::complex<double>>>) {
            js_tensor.push_back(tensor.GetIndices());
            js_tensor.push_back(tensor.GetShape());
            js_tensor.push_back(TensorDataToJSON_(tensor.GetData()));
        }
        else { // CudaTensor column-major branch
            std::vector<std::string> rev_idx{tensor.GetIndices().rbegin(),
                                             tensor.GetIndices().rend()};
            std::vector<size_t> rev_shape{tensor.GetShape().rbegin(),
                                          tensor.GetShape().rend()};
            js_tensor.push_back(rev_idx);
            js_tensor.push_back(rev_shape);
            js_tensor.push_back(TensorDataToJSON_(tensor.GetHostDataVector()));
        }

        return js_tensor;
    }

    /**
     * @brief Convert complex elements of tensor to json
     * array representation.
     *
     * @tparam S element type: one of complex<double> or complex<float>
     */
    template <typename S>
    static json TensorDataToJSON_(const std::vector<S> &data)
    {
        auto js_data = json::array();
        for (const auto &x : data) {
            js_data.push_back({std::real(x), std::imag(x)});
        }

        return js_data;
    }

    /**
     * @brief Convert json array of complex values into native
     * format.
     *
     * Throws TensorFileException exception if any of elements
     * of js_data to not encode a complex value.
     */
    template <typename S>
    static std::vector<S> TensorDataFromJSON_(const json &js_data,
                                              size_t tensor_index)
    {
        std::vector<S> data(js_data.size());

        size_t i = 0;
        try {
            while (i < js_data.size()) {
                data[i] = S{js_data[i].at(0), js_data[i].at(1)};
                i++;
            }
        }
        catch (const json::exception &) {
            throw TensorFileException(
                "Invalid element at index " + std::to_string(i) +
                " of tensor " + std::to_string(tensor_index) +
                ": Could not parse " + js_data[i].dump() + " as complex.");
        }

        return data;
    }
};

}; // namespace Jet
