import inspect
import jet
import xir

# Write an XIR program to be executed using Jet.
xir_script = inspect.cleandoc(
    """
    use xstd;

    // Prepare the GHZ state.
    H | [0];
    CNOT | [0, 1];
    CNOT | [0, 2];

    // Measure the resulting amplitudes.
    amplitude(state: 0) | [0, 1, 2];
    amplitude(state: 1) | [0, 1, 2];
    amplitude(state: 2) | [0, 1, 2];
    amplitude(state: 3) | [0, 1, 2];
    amplitude(state: 4) | [0, 1, 2];
    amplitude(state: 5) | [0, 1, 2];
    amplitude(state: 6) | [0, 1, 2];
    amplitude(state: 7) | [0, 1, 2];
    """
)

# Parse the XIR script into an XIR program.
xir_program = xir.XIRTransformer().transform(xir.xir_parser.parse(xir_script))

# Run the program locally using Jet.
result = jet.run_xir_program(xir_program)

# Display the returned amplitudes.
for i in range(len(result)):
    print(f"Amplitude |{i:03b}> = {result[i]}")
