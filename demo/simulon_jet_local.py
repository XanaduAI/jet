import inspect
import jet
import xir

# Write an XIR program to prepare a Greenberger–Horne–Zeilinger (GHZ) state.
xir_script = inspect.cleandoc(
    """
    use xstd;

    H | [0];
    CNOT | [0, 1];
    CNOT | [0, 2];

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

# Run the program locally using Jet and wait for the results.
result = jet.run_xir_program(xir_program)

# Display the returned amplitudes.
for i in range(len(result)):
    print(f"Amplitude |{i:03b}> = {result[i]}")
