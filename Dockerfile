# Reference only (local builds). CI uses crane append, see .github/workflows/build-image.yml.
# Mandated base image (shared League rules): checked by layer identity. Do not change, do not squash.
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

WORKDIR /app
COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Small text model shipped inside the image, so startup never depends on a download.
ARG MODEL_ID=Qwen/Qwen3-4B-Instruct-2507
RUN python3 -c "from huggingface_hub import snapshot_download; \
snapshot_download('${MODEL_ID}', local_dir='/models/llm', allow_patterns=['*.json','*.safetensors','*.txt','*.jinja'])" \
 && rm -rf /root/.cache

ENV MC4_MODEL_DIR=/models/llm HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
COPY app/ /app/
RUN mkdir -p /app/input /app/output && chmod -R a+rwX /app/input /app/output
# No service, no port: the harness execs into the running container.
# Model server starts with the container (loads during the startup budget); one per container.
CMD ["/bin/bash", "-c", "python3 /app/mc4/server.py >>/tmp/mc4-server.out 2>&1 & exec sleep infinity"]
