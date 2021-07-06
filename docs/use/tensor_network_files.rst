Tensor network files
====================

Jet defines and provides tools for saving (and loading) tensor networks to (and
from) JSON strings.

Tensor networks are represented as JSON objects with a ``"tensors"`` key, which
contains a list of tensors with labeled indices, and an optional ``"path"`` key
describing a contraction path through those tensors.

Tensors are represented as a tuple of 4 elements:

* **Tags**: A list of string tags.
* **Indices**: A list of string labels for each index.
* **Shape**: A list of integers containing the dimension of each index.
* **Data**: An array containing the unshaped complex data of the tensor, in
  row-major order. Complex numbers are represented using 2-element arrays
  ``[real, imaginary]``.

In the C++ API, saving and loading are both handled by the
:doc:`TensorNetworkSerializer </api/classJet_1_1TensorNetworkSerializer>` class.
Like the :doc:`TensorNetwork </api/classJet_1_1TensorNetwork>` class, the 
``TensorNetworkSerializer`` class is templated by a ``Tensor`` type and can
serialize or deserialize any valid ``TensorNetwork<Tensor>`` instance.

A :doc:`TensorFileException </api/classJet_1_1TensorFileException>` exception
is thrown when a string cannot be parsed as JSON or the string does not encode a
valid tensor network.

Example
-------

The following C++ and Python programs demonstrate creating a tensor network,
dumping it to a JSON string, and then reading the JSON string to populate a
``TensorNetworkFile`` data structure.

.. tabs::

    .. code-tab:: c++

        #include <complex>
        #include <iostream>
        #include <string>

        #include <Jet.hpp>

        int main()
        {
            using Tensor = Jet::Tensor<std::complex<float>>;

            Tensor A({"i", "j"}, {2, 2}, {{1, 0}, {0, 1}, {0, -1}, {1, 0}});
            Tensor B({"j", "k"}, {2, 2}, {{1, 0}, {0, 0}, {0, 0}, {1, 0}});
            Tensor C({"k"}, {2}, {{1, 0}, {0, 0}});

            Jet::TensorNetwork<Tensor> tn;
            tn.AddTensor(A, {"A", "hermitian"});
            tn.AddTensor(B, {"B", "identity", "real"});
            tn.AddTensor(C, {"C", "vec", "real"});

            Jet::PathInfo path(tn, {{0, 2}, {2, 1}});

            Jet::TensorNetworkSerializer<Tensor> serializer;

            // Serialization
            std::string tnf_str = serializer(tn, path);

            // Deserialization
            Jet::TensorNetworkFile<Tensor> tnf_obj = serializer(tnf_str);

            return 0;
        }

    .. code-tab:: py

        import jet

        A = jet.Tensor(["i", "j"], [2, 2], [1, 1j, -1j, 1])
        B = jet.Tensor(["j", "k"], [2, 2], [1, 0, 0, 1])
        C = jet.Tensor(["k"], [2], [1, 0])

        tn = jet.TensorNetwork()
        tn.add_tensor(A, ["A", "hermitian"])
        tn.add_tensor(B, ["B", "identity", "real"])
        tn.add_tensor(C, ["C", "vec", "real"])

        path = jet.PathInfo(tn, [(0, 2), (2, 1)]);

        serializer = jet.TensorNetworkSerializer();

        # Serialization
        tnf_str = serializer(tn, path);

        # Deserialization
        tnf_obj = serializer(tnf_str)


Serialization
-------------
To serialize a tensor network (and, optionally, a contraction path), call the
serializer with a tensor network (and the contraction path):

.. tabs::

    .. code-tab:: c++

        // Serialization
        std::string tnf_str = serializer(tn, path);
        std::cout << tnf_str << std::endl;

    .. code-tab:: py

        # Serialization
        tnf_str = serializer(tn, path);
        print(tnf_str)

The (formatted) output of this program is

.. code-block:: json

  {
    "path": [[0, 2], [2, 1]],
    "tensors": [
      [["A", "hermitian"], ["i", "j"], [2, 2], [[1, 0], [0, 1], [0, -1], [1, 0]]],
      [["B", "identity", "real"], ["j", "k"], [2, 2], [[1, 0], [0, 0], [0, 0], [1, 0]]],
      [["C", "vec", "real"], ["k"], [2], [[1, 0], [0, 0]]]
    ]
  }


Deserialization
---------------
To deserialize a tensor network (and, optionally, a contraction path), call the
serializer with a string:

.. tabs::

    .. code-tab:: c++

        // Deserialization
        Jet::TensorNetworkFile<Tensor> tnf_obj = serializer(tn_json);
        Jet::TensorNetwork<Tensor> tn = tnf_obj.tensors;
        Jet::PathInfo path = tnf_obj.path.value(); // Uses std::optional.

    .. code-tab:: py

        # Deserialization
        tnf_obj = serializer(tn_json)
        tn = tnf_obj.tensors
        path = tnf_obj.path


JSON Schema
-----------

:download:`Download </_static/schema/tensor_network.json>`

.. literalinclude:: /_static/schema/tensor_network.json
