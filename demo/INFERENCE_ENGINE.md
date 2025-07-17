# Demo Architecture

```mermaid
graph TB
    VLM[VLM Server] --> GS[Gradio Server]

    GS -.-> VPN[Intel VPN]
    VPN -.-> Internet[Internet]

    Internet --> LG[Leaderboard Gradio]
    LG --> CSV[Responses.csv]
    CSV --> SG[Submit Gradio]

    U1[User 1] --> SG
    U2[...] --> SG
    UN[User N] --> SG

    style VLM fill:#e1f5fe
    style GS fill:#e8f5e8
    style LG fill:#fff3e0
    style SG fill:#fce4ec
    style CSV fill:#f3e5f5
```

# Inference Setup with vLLM

On a GPU node, install the dependencies:

```shell
uv venv --python 3.12
source .venv/bin/activate
uv pip install vllm
```
To serve the model:

```shell
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --task generate \
  --model-impl transformers \
  --host 0.0.0.0
```
