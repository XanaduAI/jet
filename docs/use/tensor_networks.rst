Tensor networks
===============
.. |br| raw:: html

   <br />

A **tensor network** is a graph where each node represents a tensor and each
edge represents a shared index between tensors.  In the `Tensor <tensors.html>`_
section, it was shown that a tensor can be graphically modelled as a circle with
a leg for each index of the tensor.  It follows that a tensor network can be
represented as a collection of circles and lines where stray legs denote free
indices:

|br|

.. image:: ../_static/tensor_network_example.svg
  :width: 400
  :alt: Example of a tensor network.
  :align: center

|br|

One of the key operations that can be performed over a tensor network is a
*contraction*.  Local contractions are discussed in the `Tensor <tensors.html>`_
section and combine two tensors which share at least one index in a tensor
network.  Global contractions reduce a tensor network to a single node by
iteratively performing local contractions until all of the edges in the tensor
network are consumed.  For example:

|br|

.. image:: ../_static/tensor_network_contraction_example.svg
  :width: 600
  :alt: Example of a tensor network contraction.
  :align: center

|br|

Above, the result of the tensor network contraction is the node :math:`E`.
Observe that :math:`E` was produced by first contracting nodes :math:`C` and
:math:`D` to create node :math:`CD`, then contracting node :math:`CD` with
node :math:`B` to generate node :math:`BCD`, and then finally contracting node
:math:`BCD` with node :math:`A`.  Here, nodes :math:`CD` and :math:`BCD` are
*intermediary tensors* and the order of contractions is summarized by the
contraction path

.. math::

   P = \{(C, D), (CD, B), (BCD, A)\}\,.

In general, the contraction path of a tensor network is not unique and has a
significant impact on the memory requirements and running time of a tensor
network contraction.

Modelling quantum circuits
--------------------------

Although tensor networks are pure mathematical objects, they are often used to
model and simulate quantum circuits.  For example, consider the following
circuit which generates an EPR pair from two unentangled :math:`\vert 0 \rangle`
qubits:

|br|

.. image:: ../_static/tensor_network_bell_state_circuit.svg
  :width: 600
  :alt: Diagram of a circuit that generates an EPR pair.
  :align: center

|br|

This circuit can be directly modelled with the following tensor network:

|br|

.. image:: ../_static/tensor_network_bell_state_network.svg
  :width: 300
  :alt: Tensor network modelling the EPR pair quantum circuit.
  :align: center

|br|

To construct this tensor network in Jet, it is necessary to first define each
of the consituent tensors using the ``Tensor`` class.  Recall that:

.. math::

   \vert 0 \rangle = \begin{bmatrix} 1 \\ 0 \end{bmatrix} \qquad H = \begin{bmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \end{bmatrix} \qquad CNOT = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{bmatrix} \,.

The :math:`\vert 0 \rangle` qubits are relatively simple to create:

.. tabs::

    .. code-tab:: c++

        using Tensor = Jet::Tensor<std::complex<float>>;

        // The control qubit is defined as a 1-D vector with 2 elements.
        Tensor q0({"i"}, {2}, {1, 0});

        // Note that the index of ``q1`` differs from ``q0``.
        Tensor q1({"j"}, {2}, {1, 0});

    .. code-tab:: py

        import jet

        # The control qubit is defined as a 1-D vector with 2 elements.
        q0 = jet.Tensor(["i"], [2], [1, 0])

        # Note that the index of ``q1`` differs from ``q0``.
        q1 = jet.Tensor(["j"], [2], [1, 0])

The Hadamard gate :math:`H` can also be constructed in the usual way:

.. tabs::

    .. code-tab:: c++

        const float inv_sqrt_2 = 1 / std::sqrt(2);
        Tensor H({"i", "k"}, {2, 2}, {inv_sqrt_2, inv_sqrt_2, inv_sqrt_2, -inv_sqrt_2});

    .. code-tab:: py

        inv_sqrt_2 = 2 ** -0.5
        H = jet.Tensor(["i", "k"], [2, 2], [inv_sqrt_2, inv_sqrt_2, inv_sqrt_2, -inv_sqrt_2])


The controlled NOT gate :math:`CNOT` is slightly trickier.  From the diagram,
:math:`CNOT \in \mathbb{C}^{2 \times 2 \times 2 \times 2}`.  To derive this
:math:`CNOT` tensor, note that a two-qubit state
:math:`\vert \psi \rangle \in \mathbb{C}^{4}` can be encoded as a
:math:`\mathbb{C}^{2 \times 2}` matrix:

.. math::

    \vert \psi \rangle = \alpha_{00} \vert 00 \rangle + \alpha_{01} \vert 01 \rangle + \alpha_{10} \vert 10 \rangle + \alpha_{11} \vert 11 \rangle = \begin{bmatrix} \alpha_{00} & \alpha_{01} \\ \alpha_{10} & \alpha_{11} \end{bmatrix}\,.

It follows that

.. math::

    CNOT_{0, 0} = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix} \quad
    CNOT_{0, 1} = \begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix} \quad
    CNOT_{1, 0} = \begin{bmatrix} 0 & 0 \\ 0 & 1 \end{bmatrix} \quad
    CNOT_{1, 1} = \begin{bmatrix} 0 & 0 \\ 1 & 0 \end{bmatrix}\,.

The :math:`CNOT` gate is then given by

.. tabs::

    .. code-tab:: c++

        Tensor CNOT({"k", "j", "m", "n"}, {2, 2, 2, 2});
        CNOT.SetValue({0, 0, 0, 0}, 1); // |00> -> |00>
        CNOT.SetValue({0, 1, 0, 1}, 1); // |01> -> |01>
        CNOT.SetValue({1, 0, 1, 1}, 1); // |10> -> |11>
        CNOT.SetValue({1, 1, 1, 0}, 1); // |11> -> |10>

    .. code-tab:: py

        CNOT = jet.Tensor(["k", "j", "m", "n"], [2, 2, 2, 2])
        CNOT.set_value((0, 0, 0, 0), 1) # |00> -> |00>
        CNOT.set_value((0, 1, 0, 1), 1) # |01> -> |01>
        CNOT.set_value((1, 0, 1, 1), 1) # |10> -> |11>
        CNOT.set_value((1, 1, 1, 0), 1) # |11> -> |10>

Now, creating the tensor network is easy with the ``TensorNetwork`` class:

.. tabs::

    .. code-tab:: c++

        using TensorNetwork = Jet::TensorNetwork<Tensor>;
        TensorNetwork tn;

        // The second argument can be used to associate a tensor with a set of tags.
        tn.AddTensor(q0);
        tn.AddTensor(q1);
        tn.AddTensor(H);
        tn.AddTensor(CNOT);

    .. code-tab:: py

        tn = jet.TensorNetwork()

        # A second argument can be provided to associate a tensor with a set of tags.
        tn.add_tensor(q0)
        tn.add_tensor(q1)
        tn.add_tensor(H)
        tn.add_tensor(CNOT)

By default, the ``TensorNetwork`` class performs contractions in random order:

.. tabs::

    .. code-tab:: c++

        tn.Contract();

    .. code-tab:: py

        tn.contract()

An explicit contraction path can also be specified by providing a list of pair
of node IDs (0-indexed) to the ``Contract()`` function.  The ID of a node is the
order in which it was added to the tensor network.  Intermediate tensors are
assigned node IDs according to SSA convention (i.e., they are assigned the node
ID immediately following the largest node ID in the tensor network in use at the
time the intermediate tensor was created).

.. tabs::

    .. code-tab:: c++

        tn.Contract({{0, 2}, {1, 3}, {4, 5}});

    .. code-tab:: py

        tn.contract([(0, 2), (1, 3), (4, 5)])


Putting it all together,

.. tabs::

    .. code-tab:: c++

        #include <cmath>
        #include <complex>
        #include <iostream>

        #include <Jet.hpp>

        int main()
        {
            using Tensor = Jet::Tensor<std::complex<float>>;
            using TensorNetwork = Jet::TensorNetwork<Tensor>;

            Tensor q0({"i"}, {2}, {1, 0});
            Tensor q1({"j"}, {2}, {1, 0});

            const float inv_sqrt_2 = 1 / std::sqrt(2);
            Tensor H({"i", "k"}, {2, 2}, {inv_sqrt_2, inv_sqrt_2, inv_sqrt_2, -inv_sqrt_2});

            Tensor CNOT({"k", "j", "m", "n"}, {2, 2, 2, 2});
            CNOT.SetValue({0, 0, 0, 0}, 1);
            CNOT.SetValue({0, 1, 0, 1}, 1);
            CNOT.SetValue({1, 0, 1, 1}, 1);
            CNOT.SetValue({1, 1, 1, 0}, 1);

            TensorNetwork tn;
            tn.AddTensor(q0);
            tn.AddTensor(q1);
            tn.AddTensor(H);
            tn.AddTensor(CNOT);

            Tensor result = tn.Contract();
            std::cout << "Amplitude |00> = " << result.GetValue({0, 0}) << std::endl;
            std::cout << "Amplitude |01> = " << result.GetValue({0, 1}) << std::endl;
            std::cout << "Amplitude |10> = " << result.GetValue({1, 0}) << std::endl;
            std::cout << "Amplitude |11> = " << result.GetValue({1, 1}) << std::endl;

            return 0;
        }

    .. code-tab:: py

        import jet

        q0 = jet.Tensor(["i"], [2], [1, 0])
        q1 = jet.Tensor(["j"], [2], [1, 0])

        inv_sqrt_2 = 2 ** -0.5
        H = jet.Tensor(["i", "k"], [2, 2], [inv_sqrt_2, inv_sqrt_2, inv_sqrt_2, -inv_sqrt_2])

        CNOT = jet.Tensor(["k", "j", "m", "n"], [2, 2, 2, 2])
        CNOT.set_value([0, 0, 0, 0], 1)
        CNOT.set_value([0, 1, 0, 1], 1)
        CNOT.set_value([1, 0, 1, 1], 1)
        CNOT.set_value([1, 1, 1, 0], 1)

        tn = jet.TensorNetwork()
        tn.add_tensor(q0)
        tn.add_tensor(q1)
        tn.add_tensor(H)
        tn.add_tensor(CNOT)

        result = tn.contract()
        print("Amplitude |00> =", result.get_value([0, 0]))
        print("Amplitude |01> =", result.get_value([0, 1]))
        print("Amplitude |10> =", result.get_value([1, 0]))
        print("Amplitude |11> =", result.get_value([1, 1]))

    .. code-tab:: py Python (using the XIR)

        import jet
        import xir

        # Write and parse an XIR program that prepares a Bell state.
        xir_program = xir.parse_script(
            "use xstd;\n"
            "H | [0];\n"
            "CNOT | [0, 1];\n"
            "amplitude(state: [0, 0]) | [0, 1];\n"
            "amplitude(state: [0, 1]) | [0, 1];\n"
            "amplitude(state: [1, 0]) | [0, 1];\n"
            "amplitude(state: [1, 1]) | [0, 1];"
        )

        # Run the program using Jet and wait for the results.
        result = jet.run_xir_program(xir_program)

        # Display the returned amplitudes.
        print("Amplitude |00> =", result.get_value([0, 0]))
        print("Amplitude |01> =", result.get_value([0, 1]))
        print("Amplitude |10> =", result.get_value([1, 0]))
        print("Amplitude |11> =", result.get_value([1, 1]))

The output of the program is

.. tabs::

    .. code-tab:: text C++

        Amplitude |00> = (0.707107,0)
        Amplitude |01> = (0,0)
        Amplitude |10> = (0,0)
        Amplitude |11> = (0.707107,0)

    .. code-tab:: text Python

        Amplitude |00> = (0.7071067811865476+0j)
        Amplitude |01> = 0j
        Amplitude |10> = 0j
        Amplitude |11> = (0.7071067811865476+0j)

Task-based contraction
----------------------

While ``TensorNetwork::Contract()`` is simple to use, it is unlikely to exhibit
optimal performance for large tensor networks.  One alternative to the vanilla
tensor network contractor is the ``TaskBasedContractor`` class which models a
tensor network contraction as a parallel task scheduling problem where each task
encapsulates a local tensor contraction.  Such a formulation enables
intermediate tensors which do not depend on each another to be contracted
concurrently.  As an example, consider the task graph for the quantum circuit
described in the previous section:

|br|

.. image:: ../_static/tensor_network_task_graph.svg
  :width: 500
  :alt: Task graph for the EPR pair quantum circuit.
  :align: center

|br|

Clearly, the leftmost nodes in the top row (:math:`\vert 0 \rangle` and
:math:`CNOT`) may be contracted in parallel with the rightmost nodes in the
top row (the other :math:`\vert 0 \rangle` and :math:`H`); however, the
contraction representing the final output of the circuit may only be performed
once nodes :math:`A_k` and :math:`B_{m,n,k}` have been computed.

Despite its underlying complexity, the interface to ``TaskBasedContractor``
is relatively straightforward.  After constructing the ``TensorNetwork`` in the
previous section, the contraction path is specified using a ``PathInfo`` object:

.. tabs::

    .. code-tab:: cpp

        PathInfo path_info(tn, {{0, 2}, {1, 3}, {4, 5}});

    .. code-tab:: py

        path_info = jet.PathInfo(tn, [(0, 2), (1, 3), (4, 5)])

The contraction tasks can then be added to a new ``TaskBasedContractor``
instance:

.. tabs::

    .. code-tab:: cpp

        TaskBasedContractor<Tensor<std::complex<float>>> tbc;
        tbc.AddContractionTasks(tn, path_info);

    .. code-tab:: py

        tbc = jet.TaskBasedContractor()
        tbc.add_contraction_tasks(tn, path_info)

Finally, ``TaskBasedContractor::Contract()`` launches the contraction and
returns a future that becomes available when the contraction is complete:

.. tabs::

    .. code-tab:: cpp

        // Start the tensor network contraction and wait for it to finish.
        auto future = tbc.Contract();
        future.wait();

        // Each call to AddContractionTasks() generates a new result.
        const auto results = tbc.GetResults();
        const auto result = results[0];

    .. code-tab:: py

        # Start the tensor network contraction and wait for it to finish.
        tbc.contract();

        # Each call to add_contraction_tasks() generates a new result.
        results = tbc.results
        result = results[0]