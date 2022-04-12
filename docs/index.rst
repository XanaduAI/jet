Jet Documentation
#################

.. rst-class:: lead grey-text ml-2

:Release: |release|

.. raw:: html

    <style>
        .breadcrumb {
            display: none;
        }
        h1 {
            text-align: center;
            margin-bottom: 15px;
        }
        p.lead.grey-text {
            margin-bottom: 30px;
        }
        .footer-relations {
            border-top: 0px;
        }
    </style>

    <div class="container mt-2 mb-2">
        <p align="center" class="lead grey-text">
            Jet is a task-based tensor network contraction engine for quantum circuit simulation.
        </p>
        <div class="row mt-3">

.. index-card::
    :name: Key Concepts
    :link: use/introduction.html
    :description: Learn about tensors and tensor networks

.. index-card::
    :name: Getting Started
    :link: dev/guide.html
    :description: Learn how to quickly get started using Jet

.. index-card::
    :name: API
    :link: api/library_root.html
    :description: Explore the Jet API

.. raw:: html

        </div>
    </div>

Features
========

* *Heterogeneous.*  Runs on a **variety of systems**, from single-board machines to massively parallel supercomputers.

..

* *Speed.* Accelerates tensor contractions using a novel **task-based parallelism** approach.

..

* *Qudits.* Models quantum systems with an **arbitrary number of basis states**.

How to cite
===========

If you are doing research using Jet, please cite our paper:

    Trevor Vincent, Lee J. O'Riordan, Mikhail Andrenkov, Jack Brown, Nathan Killoran, Haoyu Qi, and Ish Dhand. *Jet: Fast quantum circuit simulations with parallel task-based tensor-network contraction.* 2021. `arxiv:2107.09793 <https://arxiv.org/abs/2107.09793>`_

Support
=======

- **Source Code:** https://github.com/XanaduAI/Jet
- **Issue Tracker:** https://github.com/XanaduAI/Jet/issues

If you are having issues, please let us know, either by email or by posting the
issue on our GitHub issue tracker.

License
=======

The Jet library is **free** and **open source**, released under the Apache
License, Version 2.0.

.. toctree::
   :maxdepth: 2
   :caption: Using Jet
   :hidden:

   use/introduction
   use/tensors
   use/tensor_networks
   use/tensor_network_files

.. toctree::
   :maxdepth: 2
   :caption: Development
   :hidden:

   dev/guide
   dev/research
   dev/releases

.. toctree::
   :maxdepth: 2
   :caption: C++ API
   :hidden:

   Overview <api/library_root>
   Jet <api/namespace_Jet>
   Jet::PathInfo <api/classJet_1_1PathInfo>
   Jet::TaskBasedContractor <api/classJet_1_1TaskBasedContractor>
   Jet::Tensor <api/classJet_1_1Tensor>
   Jet::TensorNetwork <api/classJet_1_1TensorNetwork>
   Jet::TensorNetworkSerializer <api/classJet_1_1TensorNetworkSerializer>
   Jet::Utilities <api/namespace_Jet__Utilities>

.. toctree::
   :maxdepth: 2
   :caption: Python API
   :hidden:

   code/jet
   code/jet_bindings
   code/jet_circuit
   code/jet_factory
   code/jet_gate
   code/jet_interpreter
   code/jet_state