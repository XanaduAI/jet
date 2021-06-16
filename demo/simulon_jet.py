import inspect
import numpy as np
import strawberryfields as sf
import time

class Connection(sf.api.Connection):
    @property
    def api_version(self) -> str:
        """str: The platform API version to request."""
        return "0.4.0"

    def run_job(self, target: str, script: str) -> np.generic:
        """Runs the given XIR script on a remote device.

        Args:
            target (str): The target device.
            script (str): The quantum circuit.

        Returns:
            np.generic: The execution result.
        """
        response = self._request(
            "POST", self._url("/jobs"), headers=self._headers, json={
                "name": "reveal_single_xor",
                "target": target,
                "language": "xir:0.1",
                "circuit": script,
                # The fields below are required until the 0.4.0 API is updated.
                "result_type": "amplitude",
                "options": {
                    "observable": "fock",
                    "cutoff": 2,
                    "state": "01",
                }
            }
        )

        if response.status_code != 201:
            raise sf.api.RequestFailedError(
                "Failed to create job: {}".format(self._format_error_message(response))
            )

        job = sf.Job(
            id_=response.json()["id"],
            status=sf.JobStatus(response.json()["status"]),
            connection=self,
        )

        try:
            while time.sleep(1) is None:
                job.refresh()
                if job.status == "complete":
                    self.log.info(f"The remote job {job.id} has been completed.")
                    return job.result.samples

                if job.status == "failed":
                    raise sf.api.FailedJobError(
                        f"The remote job {job.id} failed due to an internal server error. "
                        f"Please try again. {job.meta}."
                    )

        except KeyboardInterrupt as e:
            self.cancel_job(job.id)
            raise KeyboardInterrupt("The job has been cancelled.") from e


# Write an XIR program to be executed on the XQC.
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

# Establish a connection to the cloud platform.
conn = Connection()

# Submit the XIR program to run on Jet.
result = conn.run_job(target="simulon_jet", script=xir_script)

# Display the returned amplitudes.
for i in range(len(result)):
    print(f"Amplitude |{i:03b}> = {result[i]}")
