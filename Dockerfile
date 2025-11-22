FROM python:3.10-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 1. RDKit
RUN pip install --no-cache-dir rdkit-pypi

# 2. Torch CPU (PINNED)
RUN pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cpu

# 3. PyG deps (matching Torch 2.1.0)
RUN pip install --no-cache-dir torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.1.0+cpu.html

# 4. PyG + misc
RUN pip install --no-cache-dir torch-geometric
RUN pip install --no-cache-dir streamlit pandas scikit-learn joblib

# Copy code
COPY . /app
WORKDIR /app

EXPOSE 7860

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0"]
