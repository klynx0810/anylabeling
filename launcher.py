import os, site, subprocess, sys

# Activate venv is not needed if already inside it

# Build LD_LIBRARY_PATH
p = site.getsitepackages()[0]
libs = [
    "cublas","cudnn","curand","cufft",
    "cuda_nvrtc","cuda_runtime","nvjitlink"
]
paths = [os.path.join(p,"nvidia",x,"lib") for x in libs]

os.environ["LD_LIBRARY_PATH"] = ":".join(paths) + ":" + os.environ.get("LD_LIBRARY_PATH","")

# Run app.py
subprocess.run([sys.executable, "app.py"])
