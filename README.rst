.. raw:: html

    <p align="center">
        <img width="250" alt="Jet" src="docs/_static/jet_title.svg">
    </p>

##################################################

.. image:: https://github.com/XanaduAI/jet/actions/workflows/tests.yml/badge.svg
    :alt: GitHub Actions
    :target: https://github.com/XanaduAI/jet/actions/workflows/tests.yml

.. image:: https://img.shields.io/badge/Docs-English-yellow.svg
    :alt: Documentation
    :target: https://quantum-jet.readthedocs.io

.. image:: https://img.shields.io/badge/C%2B%2B-17-blue.svg
    :alt: Standard
    :target: https://en.wikipedia.org/wiki/C%2B%2B17

.. image:: https://img.shields.io/badge/License-Apache%202.0-orange.svg
    :alt: License
    :target: https://www.apache.org/licenses/LICENSE-2.0

`Jet <https://quantum-jet.readthedocs.io>`_ is a cross-platform C++ library for
simulating quantum circuits using tensor network contractions.

Features
========

* Runs on a variety of systems, from single-board machines to massively parallel
  supercomputers.

* Accelerates tensor contractions using a novel task-based parallelism approach.

* Models quantum systems with an arbitrary number of basis states.

To get started with Jet, read one of our `tutorial walkthroughs
<https://quantum-jet.readthedocs.io/en/latest/use/introduction.html>`__ or
browse the full `API documentation
<https://quantum-jet.readthedocs.io/en/latest/api/library_root.html>`__.

Installation
============

C++
^^^

The Jet C++ library requires `Taskflow <https://github.com/taskflow/taskflow>`_,
a BLAS library with a CBLAS interface, and a C++ compiler with C++17 support.
To use Jet, add ``#include <Jet.hpp>`` to the top of your header file and link
your program with the CBLAS library.

For example, assuming that the Taskflow headers can be found in your ``g++``
include path and OpenBLAS is installed on your system, you can compile the
``hellojet.cpp`` program below

.. code-block:: cpp

    #include <array>
    #include <complex>
    #include <iostream>

    #include <Jet.hpp>

    int main(){
        using Tensor = Jet::Tensor<std::complex<float>>;

        Tensor lhs({"i", "j", "k"}, {2, 2, 2});
        Tensor rhs({"j", "k", "l"}, {2, 2, 2});

        lhs.FillRandom();
        rhs.FillRandom();

        Tensor res = Tensor::ContractTensors(lhs, rhs);

        for (const auto &datum : res.GetData()) {
            std::cout << datum << std::endl;
        }

        std::cout << "You have successfully used Jet version " << Jet::Version() << std::endl;

        return 0;
    }

by running

.. code-block:: bash

    git clone https://github.com/XanaduAI/jet
    g++ --std=c++17 -O3 -Ijet/include hellojet.cpp -lopenblas

The output of this program should resemble

.. code-block:: text

    $ ./hellojet
    (-0.936549,0.0678852)
    (-0.0786964,-0.771624)
    (2.98721,-0.657124)
    (-1.90032,1.58051)
    You have successfully used Jet version 0.2.0

For more detailed instructions, see the `development guide
<https://quantum-jet.readthedocs.io/en/stable/dev/guide.html>`_.

Python
^^^^^^

The Jet Python package requires Python version 3.7 and above. Installation of Jet,
as well as all dependencies, can be done using pip:

.. code-block:: bash

    pip install quantum-jet

To build the Jet Python distribution locally, a BLAS library with a CBLAS
interface and a C++ compiler with C++17 support is required.  Simply run

.. code-block:: bash

    make dist
    pip install dist/*.whl

To verify that Jet is installed, you can run the ``hellojet.py`` program below

.. code-block:: python

    import jet

    lhs = jet.Tensor(["i", "j", "k"], [2, 2, 2])
    rhs = jet.Tensor(["j", "k", "l"], [2, 2, 2])

    lhs.fill_random()
    rhs.fill_random()
    res = jet.contract_tensors(lhs, rhs)

    for datum in res.data:
        print(f"{datum:.5f}")

    print("You have successfully used Jet version", jet.version())

The output of this program should resemble

.. code-block:: text

    $ python hellojet.py
    1.96289+0.25257j
    -0.16588-1.44652j
    -1.43005+0.49516j
    1.66881-1.67099j
    You have successfully used Jet version 0.2.0

Contributing to Jet
===================

We welcome new contributions - simply fork the Jet repository and make a
`pull request <https://help.github.com/articles/about-pull-requests/>`_
containing your contribution.  All contributers to Jet will be listed as authors
on the releases.  See our `changelog <.github/CHANGELOG.md>`_ for more details.

We also encourage bug reports, suggestions for new features and enhancements,
and even links to cool projects or applications built using Jet.  Visit the
`contributions page <.github/CONTRIBUTIONS.md>`_ to learn more about sharing
your ideas with the Jet team.

Support
=======

- **Source Code:** https://github.com/XanaduAI/jet
- **Issue Tracker:** https://github.com/XanaduAI/jet/issues

If you are having issues, please let us know by posting the issue on our GitHub
issue tracker.

License
=======

Jet is **free** and **open source**, released under the
`Apache License, Version 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_.

Jet contains a copy of
`JSON for Modern C++ <https://github.com/nlohmann/json>`_ 
from Niels Lohmann which is licenced under the
`MIT License <https://opensource.org/licenses/MIT>`_.
