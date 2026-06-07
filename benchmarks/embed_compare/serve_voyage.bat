@echo off
REM Serve voyageai/voyage-4-nano as an embedding endpoint on port 8003.
REM Custom architecture (VoyageQwen3BidirectionalEmbedModel) requires --trust-remote-code.
REM Single-line vllm invocation: caret (^) continuation is unreliable in LF-ended .bat files.
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin;%PATH%"
set PYTHONUTF8=1
REM Distinct internal ZMQ base port so this runs alongside the BGE server.
set VLLM_PORT=34600
REM Run from OUTSIDE the repo (avoid source-shadowing of the installed vllm._C).
cd /d %TEMP%
C:\dev\vllm-windows\.venv\Scripts\vllm.exe serve voyageai/voyage-4-nano --served-model-name voyage-4-nano --runner pooling --convert embed --trust-remote-code --hf-overrides "{\"architectures\":[\"VoyageQwen3BidirectionalEmbedModel\"]}" --pooler-config "{\"pooling_type\":\"MEAN\"}" --dtype bfloat16 --max-model-len 32768 --host 127.0.0.1 --port 8003 --gpu-memory-utilization 0.25 --enforce-eager
