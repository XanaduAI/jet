Development guide
=================

Dependencies
------------

Jet requires the following libraries be installed:

* `Taskflow <https://github.com/taskflow/taskflow>`_ >= 3.1.0
* BLAS library with support for the C API (CBLAS interface)
* C++ compiler with C++17 support (GCC >= 7, clang >= 5, Intel icpc >= 19.0.1)
* [Optional] MPI library (OpenMPI >= 3.0, MPICH >= 3.0)
* [Optional] CMake >=3.14

Installation
------------

For development purposes, a few additional dependencies are required. 
For the examples below, we will use OpenBLAS as the BLAS library. 
Please choose whichever option below best suits your system:

.. code-block:: bash

    # Ubuntu installation
    sudo apt-get install libopenblas-dev

    # CentOS/RedHat/Fedora
    sudo yum install openblas openblas-devel

    # Custom OpenBLAS installation
    git clone https://github.com/xianyi/OpenBLAS
    cd OpenBLAS && make && sudo make install

We also require the header-only library Taskflow:

.. code-block:: bash

    git clone https://github.com/taskflow/taskflow
    export TASKFLOW=$PWD/taskflow

Finally, to install Jet:

.. code-block:: bash

    git clone https://github.com/XanaduAI/jet
    export JET=$PWD/jet/include

You are now ready to build your first Jet program! 

.. _ex1-section:

Example (Hello World)
---------------------

Let us create a file called `ex1.cpp` as:

.. code-block:: cpp

    #include <array>
    #include <complex>
    #include <iostream>

    #include <Jet.hpp>

    int main(){
        using Tensor = Jet::Tensor<std::complex<float>>;

        std::array<Tensor, 3> tensors;
        tensors[0] = Tensor({"i", "j", "k"}, {2, 2, 2});
        tensors[1] = Tensor({"j", "k", "l"}, {2, 2, 2});

        tensors[0].FillRandom();
        tensors[1].FillRandom();
        tensors[2] = Tensor::ContractTensors(tensors[0], tensors[1]);
        
        for (const auto &datum : tensors[2].GetData()) {
            std::cout << datum << std::endl;
        }

        std::cout << "You have successfully used Jet version " << Jet::Version() << std::endl;

        return 0;
    }

To compile this example and verify that Jet works on your machine, you can build with the following command (assuming GCC):

.. code-block:: bash

    g++ --std=c++17 -O3 -I$JET -I$TASKFLOW ./ex1.cpp -lopenblas

Running the example should produce output similar to:

.. code-block:: text

    $ ./ex1
    (0.804981,0)
    (1.53207,0)
    (0.414398,0)
    (0.721263,0)
    You have successfully used Jet version 0.2.0

Congratulations, you have successfully run your first Jet program!


Example (CMake Project)
-----------------------

.. note:: CMake is required to build this project example.

Now that we can run a simple single-file example, we can build upon this and run a larger-scale
project example. We now build a CMake-enabled project that explictly depends on Jet. 

Begin by creating a directory for our project and adding our example code from :ref:`Example 1<ex1-section>` as well as a `CMakeLists.txt` file:

.. code-block:: bash

    mkdir my_project
    cd my_project
    touch ./ex1.cpp
    touch CMakeLists.txt

The purpose of our `CMakeLists.txt` file is to label our project, define its dependencies, acquire them, and ensure all paths are set to compile our program. Copy the following block into the `CMakeLists.txt` file:

.. code-block:: cmake

    #############################
    ## I. Set project details
    #############################
    cmake_minimum_required(VERSION 3.14)

    project("MyProject"
            VERSION 0.1.0
            DESCRIPTION "A sample Jet project"
            LANGUAGES CXX C
    )

    #############################
    ## II. Fetch Jet project
    #############################

    Include(FetchContent)

    FetchContent_Declare(
        Jet
        GIT_REPOSITORY  https://github.com/XanaduAI/jet.git
        GIT_TAG         v0.1.0
    )
    FetchContent_MakeAvailable(Jet)

    #############################
    ## III. Create project target
    #############################

    add_executable(my_jet_project ex1.cpp)
    target_link_libraries(my_jet_project Jet)

Section `I.` sets up your project with a given name, source-code type, and version information.

Section `II.` labels Jet as an external project to fetch, and will automatically pull the repository, as well as set up all of the Jet dependencies.

Section `III.` defines an executable for your project, and sets Jet as a dependency of it. This will ensure all headers and libraries are available at compile-time. You can now build your project with the following code-block:

.. code-block:: bash

    cmake .
    make
    ./my_jet_project

The output will be the same as :ref:`Example 1<ex1-section>`. Congratulations, you have now built a project with Jet as a dependency!

Performance optimization
------------------------

Jet has several options for improving the performance of your application. They can be enabled from the CMake builder using the following flags:

* :code:`-DENABLE_OPENMP=on` : Jet uses shared-memory parallelism via OpenMP where applicable.
* :code:`-DENABLE_NATIVE=on` : Jet compiles all code targetted specifically for your CPU architecture.
* :code:`-DENABLE_IPO=on` : Jet will compile with inter-procedural (link-time) optimisation.

For example, to enable the OpenMP and native architecture options with CMake, you may use the following:

.. code-block:: bash

    cmake . -DENABLE_OPENMP=on -DENABLE_NATIVE=on 

Any project that depends on Jet will now also be built using these options. Try combining these in various ways to determine the options best suited for your system. 

Similarly, Jet features support to find the best available BLAS library on your system. If you wish to use a different BLAS library than what is found, please ensure your required BLAS library is available on your path.

.. _test-section:

Tests
-----

.. note:: CMake is required to build the test suite, which uses the `Catch2 <https://github.com/catchorg/Catch2>`_ testing framework.
    

To ensure that Jet is working correctly after installation, the test suite can
be run by creating a top-level ``build/`` directory in the Jet repository and running

.. code-block:: bash

    cd build
    cmake .. -DBUILD_TESTS=ON
    make
    ./test/runner

All available tests for Jet will be run, with output similar to 

.. code-block:: text

    ===============================================================================
    All tests passed (414 assertions in 64 test cases)


To see all test options, run

.. code-block:: bash

    ./test/runner --help

Format
------

Contributions are checked for format alignment in the pipeline. Changes can be
formatted locally using:

.. code-block:: bash

    ./bin/format include test

All files within the listed directories will be modified to fit the expected format, if required.

Documentation
-------------

A few Python packages are required to build the documentation, as specified in
``docs/requirements.txt``. These packages can be installed using:

.. code-block:: bash

    pip install -r docs/requirements.txt

To build the HTML documentation, change into the ``docs/`` folder and run

.. code-block:: bash

    make html

The documentation can then be found in the :file:`docs/_build/html/` directory.

Submitting a pull request
-------------------------

Before submitting a pull request, please make sure the following is done:

* **All new features must include a unit test.** If you've fixed a bug or added
  code that should be tested, add a test to the ``test/`` directory.

* **All new functions and code must be clearly commented and documented.**

  Have a look through the source code at some of the existing functions ---
  the easiest approach is to simply copy an existing Doxygen comment and modify
  it as appropriate.

  If you do make documentation changes, make sure that the docs build and render
  correctly by running ``cd docs && make html``.

* **Ensure that the test suite passes**, by following the :ref:`test suite guide<test-section>`.

When ready, submit your fork as a `pull request <https://help.github.com/articles/about-pull-requests>`_
to the Jet repository, filling out the pull request template. This template is
added automatically to the comment box when you create a new issue.

* When describing the pull request, please include as much detail as possible
  regarding the changes made/new features added/performance improvements. If
  including any bug fixes, mention the issue numbers associated with the bugs.

* Once you have submitted the pull request, the **test suite** will
  automatically run on CircleCI to ensure that all tests continue to pass.
