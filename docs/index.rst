Jet Documentation
#################

.. rst-class:: lead grey-text ml-2

:Release: |release|

.. raw:: html

    <style>
        #right-column.card {
            box-shadow: none!important;
        }
        #right-column.card:hover {
            box-shadow: none!important;
        }
        .breadcrumb {
            display: none;
        }
        h1 {
            text-align: center;
            margin-bottom: 15px;
        }
        .footer-relations {
            border-top: 0px;
        }
    </style>
    <div style='clear:both'></div>
    <div class="container mt-2 mb-2">
        <p class="lead grey-text">
            Jet is a task-based tensor network contraction engine for quantum circuit simulation.
        </p>
        <div class="row mt-3">
            <div class="col-lg-4 mb-2 adlign-items-stretch">
                <a href="use/introduction.html">
                    <div class="card rounded-lg" style="height:100%;">
                        <div class="d-flex">
                            <div>
                                <h3 class="card-title pl-3 mt-4">
                                Key Concepts
                                </h3>
                                <p class="mb-3 grey-text px-3">
                                    Learn about tensors and tensor networks <i class="fas fa-angle-double-right"></i>
                                </p>
                            </div>
                        </div>
                    </div>
                </a>
            </div>
            <div class="col-lg-4 mb-2 align-items-stretch">
                <a href="dev/guide.html">
                <div class="card rounded-lg" style="height:100%;">
                    <div class="d-flex">
                        <div>
                            <h3 class="card-title pl-3 mt-4">
                            Getting Started
                            </h3>
                                <p class="mb-3 grey-text px-3">
                                    Learn how to quickly get started using Jet <i class="fas fa-angle-double-right"></i>
                                </p>
                        </div>
                    </div>
                </div>
            </a>
            </div>
            <div class="col-lg-4 mb-2 align-items-stretch">
                <a href="api/library_root.html">
                <div class="card rounded-lg" style="height:100%;">
                    <div class="d-flex">
                        <div>
                            <h3 class="card-title pl-3 mt-4">
                            API
                            </h3>
                            <p class="mb-3 grey-text px-3">
                                Explore the Jet API <i class="fas fa-angle-double-right"></i>
                            </p>
                        </div>
                    </div>
                </div>
            </a>
            </div>
	    
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