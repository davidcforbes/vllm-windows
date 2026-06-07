@echo off
REM Serve BAAI/bge-m3 as an embedding endpoint on port 8001 (max input 8192 tokens).
REM Single-line vllm invocation: caret (^) continuation is unreliable in LF-ended .bat files.
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin;%PATH%"
set PYTHONUTF8=1
REM Distinct internal ZMQ base port so this runs alongside the voyage server.
set VLLM_PORT=34550
REM Run from OUTSIDE the repo (avoid source-shadowing of the installed vllm._C).
cd /d %TEMP%
C:\dev\vllm-windows\.venv\Scripts\vllm.exe serve BAAI/bge-m3 --served-model-name bge-m3 --runner pooling --convert embed --host 127.0.0.1 --port 8001 --gpu-memory-utilization 0.2 --enforce-eager
