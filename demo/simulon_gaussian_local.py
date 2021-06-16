import inspect
import strawberryfields as sf

# Write a Blackbird program to be executed with the Gaussian simulator.
bbx_script = inspect.cleandoc(
    """
    name PartialTeleportation
    version 1.0
    target gaussian

    Coherent(sqrt(5/4), arctan(1/2)) | 0
    Sgate(2, 0) | 2
    Sgate(-2, 0) | 1
    BSgate(pi/4, 0) | [1, 2]
    BSgate(pi/4, 0) | [0, 1]
    MeasureHomodyne(pi/2) | 1
    MeasureHomodyne(0) | 0
    # Xgate("1.41421356237*q0") | 2
    # Zgate("1.41421356237*q1") | 2
    """
)

# Load the Blackbird script into a Strawberry Fields program.
program = sf.io.loads(bbx_script)

# Create an engine which communicates with the "gaussian" backend.
engine = sf.LocalEngine("gaussian")

# Run the program locally and wait for the results.
result = engine.run(program, shots=1)

# Display the returned samples.
print(f"Samples = {result.samples}")
