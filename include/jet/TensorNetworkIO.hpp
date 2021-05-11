#pragma once

#include <complex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "external/nlohmann/json.hpp"

#include "PathInfo.hpp"
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
 * @brief `%invalid_tensor_file` is thrown when the contents of a tensor network
 *        file are invalid.
 */
class invalid_tensor_file : public std::invalid_argument {
  public:
    /**
     * @brief Constructs a new `%invalid_tensor_file` exception.
     *
     * @param what_arg Error message explaining what went wrong while loading a
     *                 tensor network file.
     */
    explicit invalid_tensor_file(const std::string &what_arg)
        : std::invalid_argument("Error parsing tensor network file: " +
                                what_arg){};

    /**
     * @see invalid_tensor_file(const std::string&).
     */
    explicit invalid_tensor_file(const char *what_arg)
        : invalid_tensor_file(std::string(what_arg)){};
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
template <class Tensor> class TensorNetworkSerializer {
  public:
    /**
     * @brief Constructs a new `%TensorNetworkSerializer`.
     *
     * @param indent Indent level to use when serializing to json. Default value
     *               will use the most compact representation.
     */
    TensorNetworkSerializer<Tensor>(int indent = -1)
        : indent(indent), js(json::object())
    {
    }

    /**
     * @brief Dump tensor network and path to string.
     *
     * @return JSON string representing tensor network and path
     */
    std::string operator()(const TensorNetwork<Tensor> &tn,
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
    std::string operator()(const TensorNetwork<Tensor> &tn)
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
    TensorNetworkFile<Tensor> operator()(std::string js_str)
    {
        TensorNetworkFile<Tensor> tf;
        LoadAndValidateJSON_(js_str);

        size_t i = 0;
        for (auto &js_tensor : js["tensors"]) {
            auto data = TensorDataFromJSON_<typename Tensor::scalar_type_t>(
                js_tensor[3], i);

            tf.tensors.AddTensor(Tensor(js_tensor[1], js_tensor[2], data),
                                 js_tensor[0]);
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
     * invalid_tensor_file if it does not have the correct
     * keys.
     */
    void LoadAndValidateJSON_(const std::string &js_str)
    {
        js = json::parse(js_str); // throws json::exception if invalid json

        if (!js.is_object()) {
            throw invalid_tensor_file("root element must be an object.");
        }

        if (js.find("tensors") == js.end()) {
            throw invalid_tensor_file("root object must contain 'tensors' key");
        }
    }

    /**
     * @brief Convert Tensor to json array format.
     */
    static json TensorToJSON_(const Tensor &tensor,
                              const std::vector<std::string> &tags)
    {
        auto js_tensor = json::array();

        js_tensor.push_back(tags);
        js_tensor.push_back(tensor.GetIndices());
        js_tensor.push_back(tensor.GetShape());
        js_tensor.push_back(TensorDataToJSON_(tensor.GetData()));

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
     * Throws invalid_tensor_file exception if any of elements
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
                data[i] = S(js_data[i].at(0), js_data[i].at(1));
                i++;
            }
        }
        catch (const json::exception &) {
            throw invalid_tensor_file(
                "Invalid element at index " + std::to_string(i) +
                " of tensor " + std::to_string(tensor_index) +
                ": Could not parse " + js_data[i].dump() + " as complex.");
        }

        return data;
    }
};

}; // namespace Jet
