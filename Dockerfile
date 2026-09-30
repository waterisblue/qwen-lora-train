FROM quay.io/ascend/vllm-ascend:v0.9.1

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 不允许 pip 自动升级 torch / torch-npu
COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir \
    -r /tmp/requirements.txt

RUN mkdir -p \
    /workspace/models \
    /workspace/datasets \
    /workspace/scripts \
    /workspace/output

ENV PYTHONPATH=/workspace:$PYTHONPATH

CMD ["/bin/bash"]