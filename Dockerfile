# Runtime image for Radio Map Prediction-based Shadow Region Detection.
# CUDA 11.8 and PyTorch 2.2.0 are the versions used for the results in results/.
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    HOME=/tmp

# torch, torchvision, numpy and pillow come with the base image
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace
COPY . /workspace

CMD ["python", "scripts/check_setup.py"]
