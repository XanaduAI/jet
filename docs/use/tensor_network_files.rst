Tensor network files
====================

Jet defines and provides tools for saving and loading tensor
networks to/from JSON strings. 

Tensor networks are represented as JSON objects with a 'tensors' key, 
which contains a list of tensors with labeled indices, and an optional 
'path' key describing a contraction path through those tensors.

Tensors are represented as a tuple of 4 elements:  

* **tags**: a list of string tags  
* **indices**: a list of string labels for each index  
* **shape**: a list of integers containing the dimension of each index  
* **data**: an array containing the unshaped complex data of the tensor, in row-major order. Complex numbers are represented using 2-element arrays ``[real, imaginary]``

Saving and loading are both handled by the :doc:`TensorNetworkSerializer </api/classJet_1_1TensorNetworkSerializer>` class. 
Like the :doc:`TensorNetwork </api/classJet_1_1TensorNetwork>` class, the 
``TensorNetworkSerializer`` class is templated by a ``Tensor`` type and 
can serialize or deserialize any valid ``TensorNetwork<Tensor>`` instance.

An ``invalid_tensor_file`` exception will be thrown when a string cannot 
be parsed as JSON, or does not encode a valid tensor network.


Example
-------

The following C++ program demonstrates creating a tensor network and dumping it
to a JSON string. First, create a simple network  of three tensors with a 
contraction path and initialize the serializer:

.. code-block:: cpp

    #include <complex>
    #include <iostream>
    #include <string>

    #include <Jet.hpp>

    int main(){
        using Tensor = Jet::Tensor<std::complex<float>>;

        Jet::TensorNetwork<Tensor> tn;

        Tensor A({"i", "j"}, {2, 2}, {{1, 0}, {0, 1}, {0, -1}, {1, 0}});
        tn.AddTensor(A, {"A", "hermitian"});

        Tensor B({"j", "k"}, {2, 2}, {{1, 0}, {0, 0}, {0, 0}, {1, 0}});
        tn.AddTensor(B, {"B", "identity", "real"});

        Tensor C({"k"}, {2}, {{1,0}, {0, 0}});
        tn.AddTensor(C, {"C", "vec", "real"});
        
        Jet::PathInfo path(tn, {{0, 2}, {2, 1}});

        Jet::TensorNetworkSerializer<Tensor> serializer();
        ...


Serialization
-------------
To serialize, call the serializer with a tensor network (and an optional path):

.. code-block:: cpp
    
    ...
    
    std::string tn_json = serializer(tn, path);
    std::cout << tn_json;


The output of this program will be:

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
To deserialize a tensor network, call the serializer with a string:

.. code-block:: cpp

    ...
    
    std::string tn_json = serializer(tn_path);

    TensorNetworkFile<Tensor> tensor_file = serializer(tn_json);

    Jet::TensorNetwork<Tensor> tn_copy = tensor_file.tensors;
    Jet::PathInfo path_copy = tensor_file.path.value(); // uses std::optional



JSON Schema
-----------

:download:`Download </_static/schema/tensor_network.json>`

.. literalinclude:: /_static/schema/tensor_network.json
