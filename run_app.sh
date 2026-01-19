#!/usr/bin/env bash
set -e

# =========================
# Auto setup CUDA libs for ONNXRuntime GPU
# =========================
VENV_PYTHON="$(pwd)/../env/bin/python"

CUDA_LIBS="$($VENV_PYTHON - <<'PY'
import site, os
p = site.getsitepackages()[0]
libs = [
  ('cublas','lib'),
  ('cudnn','lib'),
  ('curand','lib'),
  ('cufft','lib'),
  ('cuda_nvrtc','lib'),
  ('cuda_runtime','lib'),
  ('nvjitlink','lib'),
]
paths = [os.path.join(p,'nvidia',name,sub) for name,sub in libs]
print(":".join(paths))
PY
)"

export LD_LIBRARY_PATH="$CUDA_LIBS:$LD_LIBRARY_PATH"

echo "[INFO] LD_LIBRARY_PATH set for ONNXRuntime GPU"
echo "[INFO] Launching X-AnyLabeling..."

# =========================
# Run app
# =========================
exec "$VENV_PYTHON" app.py
