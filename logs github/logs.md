Skip to content
rosachilov-design
whisperX-medical-transcribe
Repository navigation
Code
Issues
Pull requests
Actions
Projects
Wiki
Security
Insights
Settings
Build and Push Docker Image
fixes #24
All jobs
Run details
Annotations
1 error
build-and-push
failed 4 minutes ago in 2m 42s
Search logs
4s
0s
5s
1s
2m 24s
#16 1.076 Requirement already satisfied: rich>=12.3.0 in /usr/local/lib/python3.11/dist-packages (from typer>=0.24.0->typer-slim->huggingface-hub>=0.21->faster-whisper>=1.1.1) (14.3.3)
#16 1.076 Requirement already satisfied: annotated-doc>=0.0.2 in /usr/local/lib/python3.11/dist-packages (from typer>=0.24.0->typer-slim->huggingface-hub>=0.21->faster-whisper>=1.1.1) (0.0.4)
#16 1.085 Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.11/dist-packages (from rich>=12.3.0->typer>=0.24.0->typer-slim->huggingface-hub>=0.21->faster-whisper>=1.1.1) (4.0.0)
#16 1.085 Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.11/dist-packages (from rich>=12.3.0->typer>=0.24.0->typer-slim->huggingface-hub>=0.21->faster-whisper>=1.1.1) (2.18.0)
#16 1.088 Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.11/dist-packages (from markdown-it-py>=2.2.0->rich>=12.3.0->typer>=0.24.0->typer-slim->huggingface-hub>=0.21->faster-whisper>=1.1.1) (0.1.2)
#16 1.105 Downloading faster_whisper-1.2.1-py3-none-any.whl (1.1 MB)
#16 1.145    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.1/1.1 MB 50.9 MB/s eta 0:00:00
#16 1.156 Downloading av-16.1.0-cp311-cp311-manylinux_2_28_x86_64.whl (40.8 MB)
#16 1.266    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 40.8/40.8 MB 379.0 MB/s eta 0:00:00
#16 1.276 Downloading huggingface_hub-1.4.1-py3-none-any.whl (553 kB)
#16 1.279    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 553.3/553.3 kB 663.5 MB/s eta 0:00:00
#16 1.289 Downloading onnxruntime-1.24.2-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (17.1 MB)
#16 1.351    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 17.1/17.1 MB 283.0 MB/s eta 0:00:00
#16 1.362 Downloading tokenizers-0.22.2-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.3 MB)
#16 1.373    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.3/3.3 MB 327.7 MB/s eta 0:00:00
#16 1.383 Downloading hf_xet-1.2.0-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.3 MB)
#16 1.390    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.3/3.3 MB 656.6 MB/s eta 0:00:00
#16 1.400 Downloading typer_slim-0.24.0-py3-none-any.whl (3.4 kB)
#16 1.636 Installing collected packages: hf-xet, av, onnxruntime, typer-slim, huggingface-hub, tokenizers, faster-whisper
#16 3.417 Successfully installed av-16.1.0 faster-whisper-1.2.1 hf-xet-1.2.0 huggingface-hub-1.4.1 onnxruntime-1.24.2 tokenizers-0.22.2 typer-slim-0.24.0
#16 3.417 WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable.It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
#16 3.525 
#16 3.525 [notice] A new release of pip is available: 24.2 -> 26.0.1
#16 3.525 [notice] To update, run: python -m pip install --upgrade pip
#16 DONE 3.9s

#17 [10/17] RUN pip install --no-cache-dir "pyannote.audio>=4.0.0" -c /tmp/constraints.txt
#17 0.469 Collecting pyannote.audio>=4.0.0
#17 0.501   Downloading pyannote_audio-4.0.4-py3-none-any.whl.metadata (13 kB)
#17 0.520 Collecting asteroid-filterbanks>=0.4.0 (from pyannote.audio>=4.0.0)
#17 0.531   Downloading asteroid_filterbanks-0.4.0-py3-none-any.whl.metadata (3.3 kB)
#17 0.546 Collecting einops>=0.8.1 (from pyannote.audio>=4.0.0)
#17 0.557   Downloading einops-0.8.2-py3-none-any.whl.metadata (13 kB)
#17 0.560 Requirement already satisfied: huggingface-hub>=0.28.1 in /usr/local/lib/python3.11/dist-packages (from pyannote.audio>=4.0.0) (1.4.1)
#17 0.676 Collecting lightning>=2.4 (from pyannote.audio>=4.0.0)
#17 0.687   Downloading lightning-2.6.1-py3-none-any.whl.metadata (44 kB)
#17 0.892 Collecting matplotlib>=3.10.0 (from pyannote.audio>=4.0.0)
#17 0.902   Downloading matplotlib-3.10.8-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (52 kB)
#17 0.939 Collecting opentelemetry-api>=1.34.0 (from pyannote.audio>=4.0.0)
#17 0.949   Downloading opentelemetry_api-1.39.1-py3-none-any.whl.metadata (1.5 kB)
#17 0.975 Collecting opentelemetry-exporter-otlp>=1.34.0 (from pyannote.audio>=4.0.0)
#17 0.985   Downloading opentelemetry_exporter_otlp-1.39.1-py3-none-any.whl.metadata (2.4 kB)
#17 1.012 Collecting opentelemetry-sdk>=1.34.0 (from pyannote.audio>=4.0.0)
#17 1.022   Downloading opentelemetry_sdk-1.39.1-py3-none-any.whl.metadata (1.5 kB)
#17 1.045 Collecting pyannote-core>=6.0.1 (from pyannote.audio>=4.0.0)
#17 1.058   Downloading pyannote_core-6.0.1-py3-none-any.whl.metadata (1.9 kB)
#17 1.079 Collecting pyannote-database>=6.1.1 (from pyannote.audio>=4.0.0)
#17 1.089   Downloading pyannote_database-6.1.1-py3-none-any.whl.metadata (30 kB)
#17 1.110 Collecting pyannote-metrics>=4.0.0 (from pyannote.audio>=4.0.0)
#17 1.120   Downloading pyannote_metrics-4.0.0-py3-none-any.whl.metadata (2.2 kB)
#17 1.136 Collecting pyannote-pipeline>=4.0.0 (from pyannote.audio>=4.0.0)
#17 1.146   Downloading pyannote_pipeline-4.0.0-py3-none-any.whl.metadata (5.4 kB)
#17 1.160 Collecting pyannoteai-sdk>=0.3.0 (from pyannote.audio>=4.0.0)
#17 1.173   Downloading pyannoteai_sdk-0.4.0-py3-none-any.whl.metadata (2.4 kB)
#17 1.230 Collecting pytorch-metric-learning>=2.8.1 (from pyannote.audio>=4.0.0)
#17 1.243   Downloading pytorch_metric_learning-2.9.0-py3-none-any.whl.metadata (18 kB)
#17 1.246 Requirement already satisfied: rich>=13.9.4 in /usr/local/lib/python3.11/dist-packages (from pyannote.audio>=4.0.0) (14.3.3)
#17 1.360 Collecting safetensors>=0.5.2 (from pyannote.audio>=4.0.0)
#17 1.370   Downloading safetensors-0.7.0-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.1 kB)
#17 1.390 Collecting torch-audiomentations>=0.12.0 (from pyannote.audio>=4.0.0)
#17 1.400   Downloading torch_audiomentations-0.12.0-py3-none-any.whl.metadata (15 kB)
#17 1.451 INFO: pip is looking at multiple versions of pyannote-audio to determine which version is compatible with other requirements. This could take a while.
#17 1.452 Collecting pyannote.audio>=4.0.0
#17 1.464   Downloading pyannote_audio-4.0.3-py3-none-any.whl.metadata (13 kB)
#17 1.482 Collecting soundfile>=0.13.1 (from pyannote.audio>=4.0.0)
#17 1.492   Downloading soundfile-0.13.1-py2.py3-none-manylinux_2_28_x86_64.whl.metadata (16 kB)
#17 1.495 Collecting pyannote.audio>=4.0.0
#17 1.506   Downloading pyannote_audio-4.0.2-py3-none-any.whl.metadata (13 kB)
#17 1.523   Downloading pyannote_audio-4.0.1-py3-none-any.whl.metadata (14 kB)
#17 1.537   Downloading pyannote_audio-4.0.0-py3-none-any.whl.metadata (14 kB)
#17 1.542 Collecting opentelemetry-api==1.34.0 (from pyannote.audio>=4.0.0)
#17 1.552   Downloading opentelemetry_api-1.34.0-py3-none-any.whl.metadata (1.5 kB)
#17 1.556 Collecting opentelemetry-exporter-otlp==1.34.0 (from pyannote.audio>=4.0.0)
#17 1.566   Downloading opentelemetry_exporter_otlp-1.34.0-py3-none-any.whl.metadata (2.4 kB)
#17 1.570 Collecting opentelemetry-sdk==1.34.0 (from pyannote.audio>=4.0.0)
#17 1.580   Downloading opentelemetry_sdk-1.34.0-py3-none-any.whl.metadata (1.6 kB)
#17 1.583 ERROR: Cannot install pyannote-audio==4.0.0, pyannote-audio==4.0.1, pyannote-audio==4.0.2, pyannote-audio==4.0.3 and pyannote-audio==4.0.4 because these package versions have conflicting dependencies.
#17 1.584 
#17 1.584 The conflict is caused by:
#17 1.584     pyannote-audio 4.0.4 depends on torch>=2.8.0
#17 1.584     pyannote-audio 4.0.3 depends on torch==2.8.0
#17 1.584     pyannote-audio 4.0.2 depends on torch==2.8.0
#17 1.584     pyannote-audio 4.0.1 depends on torch>=2.8.0
#17 1.584     pyannote-audio 4.0.0 depends on torch>=2.8.0
#17 1.584     The user requested (constraint) torch==2.4.0
#17 1.584 
#17 1.584 To fix this you could try to:
#17 1.584 1. loosen the range of package versions you've specified
#17 1.584 2. remove package versions to allow pip to attempt to solve the dependency conflict
#17 1.584 
#17 1.715 
#17 1.715 [notice] A new release of pip is available: 24.2 -> 26.0.1
#17 1.715 [notice] To update, run: python -m pip install --upgrade pip
#17 1.715 ERROR: ResolutionImpossible: for help visit https://pip.pypa.io/en/latest/topics/dependency-resolution/#dealing-with-dependency-conflicts
#17 ERROR: process "/bin/bash -o pipefail -c pip install --no-cache-dir \"pyannote.audio>=4.0.0\" -c /tmp/constraints.txt" did not complete successfully: exit code: 1
------
 > importing cache manifest from ***/whisperx-medical:buildcache:
------
------
 > [10/17] RUN pip install --no-cache-dir "pyannote.audio>=4.0.0" -c /tmp/constraints.txt:
1.584     The user requested (constraint) torch==2.4.0
1.584 
1.584 To fix this you could try to:
1.584 1. loosen the range of package versions you've specified
1.584 2. remove package versions to allow pip to attempt to solve the dependency conflict
1.584 
1.715 
Notice: 1.715 [notice] A new release of pip is available: 24.2 -> 26.0.1
Notice: 1.715 [notice] To update, run: python -m pip install --upgrade pip
1.715 ERROR: ResolutionImpossible: for help visit https://pip.pypa.io/en/latest/topics/dependency-resolution/#dealing-with-dependency-conflicts
------

 2 warnings found (use docker --debug to expand):
 - SecretsUsedInArgOrEnv: Do not use ARG or ENV instructions for sensitive data (ARG "HF_TOKEN") (line 64)
 - SecretsUsedInArgOrEnv: Do not use ARG or ENV instructions for sensitive data (ENV "HF_TOKEN") (line 65)
Dockerfile:51
--------------------
  49 |     
  50 |     # pyannote.audio v4 (breaking change: use_auth_token → token)
  51 | >>> RUN pip install --no-cache-dir "pyannote.audio>=4.0.0" -c /tmp/constraints.txt
  52 |     
  53 |     # transformers — pinned via constraints, install explicitly
--------------------
ERROR: failed to build: failed to solve: process "/bin/bash -o pipefail -c pip install --no-cache-dir \"pyannote.audio>=4.0.0\" -c /tmp/constraints.txt" did not complete successfully: exit code: 1
Error: buildx failed with: ERROR: failed to build: failed to solve: process "/bin/bash -o pipefail -c pip install --no-cache-dir \"pyannote.audio>=4.0.0\" -c /tmp/constraints.txt" did not complete successfully: exit code: 1
0s
1s
4s
1s
0s
