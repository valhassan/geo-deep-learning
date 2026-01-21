# Remove old environment and lockfile
rm -rf .venv
rm -f uv.lock

# Clear uv cache
uv cache clean

# Create + sync environment with CUDA 12.8 wheels
uv sync --native-tls --extra cu128

# Activate environment
source .venv/bin/activate
