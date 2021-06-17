import inspect
import strawberryfields as sf

# Write a Blackbird program to prepare an approximate Gottesman-Kitaev-Preskill (GKP) state.
bbx_script = inspect.cleandoc(
    """
    name ApproximateGKP
    version 1.0
    target simulon_gaussian (shots=10)

    Sgate(-1.38155106) | 0
    Sgate(-1.21699567) | 1
    Sgate(0.779881700) | 2

    BSgate(1.04182349, 1.483536390) | [0, 1]
    BSgate(0.87702211, 1.696290600) | [1, 2]
    BSgate(0.90243916, -0.24251599) | [0, 1]

    Sgate(0.1958) | 2

    MeasureFock() | [0, 1, 2]
    """
)

# Load the Blackbird script into a Strawberry Fields program.
program = sf.io.loads(bbx_script)

# Create an engine which communicates with the "simulon_gaussian" backend.
engine = sf.RemoteEngine("simulon_gaussian")

# Submit the program to the Xanadu Quantum Cloud (XQC) and wait for the results.
result = engine.run(program)

# Display the returned samples.
print(f"Samples = {result.samples}")
