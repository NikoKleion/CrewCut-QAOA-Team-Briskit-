import runpy
import sys
from pathlib import Path

shots = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
depth = int(sys.argv[2]) if len(sys.argv) > 2 else 2
device_name = sys.argv[3].lower() if len(sys.argv) > 3 else "basic"

if device_name == "basic":
    from qiskit.providers.basic_provider import BasicSimulator
    device = BasicSimulator()
else:
    from qiskit_ibm_runtime import fake_provider
    device = getattr(fake_provider, "Fake" + device_name.capitalize())()

project = runpy.run_path(str(Path(__file__).with_name("main.py")), init_globals={"backend": device})
project["main"](shots=shots, depth=depth)
