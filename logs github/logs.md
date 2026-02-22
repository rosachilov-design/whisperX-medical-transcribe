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
fixes #26
All jobs
Run details
Annotations
1 error
build-and-push
failed 2 minutes ago in 6m 32s
Search logs
2s
1s
6s
1s
6m 10s
#21 40.59     Found existing installation: filelock 3.20.0
#21 40.59     Uninstalling filelock-3.20.0:
#21 40.60       Successfully uninstalled filelock-3.20.0
#21 40.61   Attempting uninstall: triton
#21 40.61     Found existing installation: triton 3.0.0
#21 40.63     Uninstalling triton-3.0.0:
#21 41.84       Successfully uninstalled triton-3.0.0
#21 44.53   Attempting uninstall: nvidia-cusparse-cu12
#21 44.54     Found existing installation: nvidia-cusparse-cu12 12.3.0.142
#21 44.54     Uninstalling nvidia-cusparse-cu12-12.3.0.142:
#21 44.98       Successfully uninstalled nvidia-cusparse-cu12-12.3.0.142
#21 46.92   Attempting uninstall: nvidia-cudnn-cu12
#21 46.93     Found existing installation: nvidia-cudnn-cu12 9.1.0.70
#21 46.93     Uninstalling nvidia-cudnn-cu12-9.1.0.70:
#21 51.19       Successfully uninstalled nvidia-cudnn-cu12-9.1.0.70
#21 59.40   Attempting uninstall: jinja2
#21 59.40     Found existing installation: Jinja2 3.1.6
#21 59.41     Uninstalling Jinja2-3.1.6:
#21 59.43       Successfully uninstalled Jinja2-3.1.6
#21 59.50   Attempting uninstall: nvidia-cusolver-cu12
#21 59.51     Found existing installation: nvidia-cusolver-cu12 11.6.0.99
#21 59.51     Uninstalling nvidia-cusolver-cu12-11.6.0.99:
#21 59.82       Successfully uninstalled nvidia-cusolver-cu12-11.6.0.99
#21 61.21   Attempting uninstall: torch
#21 61.21     Found existing installation: torch 2.4.0+cu124
#21 61.54     Uninstalling torch-2.4.0+cu124:
#21 73.46       Successfully uninstalled torch-2.4.0+cu124
#21 89.66   Attempting uninstall: torchvision
#21 89.67     Found existing installation: torchvision 0.19.0+cu124
#21 89.69     Uninstalling torchvision-0.19.0+cu124:
#21 89.87       Successfully uninstalled torchvision-0.19.0+cu124
#21 90.28   Attempting uninstall: torchaudio
#21 90.28     Found existing installation: torchaudio 2.4.0+cu124
#21 90.30     Uninstalling torchaudio-2.4.0+cu124:
#21 90.46       Successfully uninstalled torchaudio-2.4.0+cu124
#21 90.70 ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
#21 90.70 whisperx 3.8.1 requires pyannote-audio>=4.0.0, but you have pyannote-audio 3.4.0 which is incompatible.
#21 90.70 whisperx 3.8.1 requires torch~=2.8.0, but you have torch 2.4.0+cu124 which is incompatible.
#21 90.70 whisperx 3.8.1 requires torchaudio~=2.8.0, but you have torchaudio 2.4.0+cu124 which is incompatible.
#21 90.70 whisperx 3.8.1 requires transformers>=4.48.0, but you have transformers 4.47.1 which is incompatible.
#21 90.70 whisperx 3.8.1 requires triton>=3.3.0; sys_platform == "linux" and platform_machine == "x86_64", but you have triton 3.0.0 which is incompatible.
#21 90.70 Successfully installed MarkupSafe-3.0.2 filelock-3.20.0 fsspec-2025.12.0 jinja2-3.1.6 mpmath-1.3.0 networkx-3.6.1 numpy-2.3.5 nvidia-cublas-cu12-12.4.2.65 nvidia-cuda-cupti-cu12-12.4.99 nvidia-cuda-nvrtc-cu12-12.4.99 nvidia-cuda-runtime-cu12-12.4.99 nvidia-cudnn-cu12-9.1.0.70 nvidia-cufft-cu12-11.2.0.44 nvidia-curand-cu12-10.3.5.119 nvidia-cusolver-cu12-11.6.0.99 nvidia-cusparse-cu12-12.3.0.142 nvidia-nccl-cu12-2.20.5 nvidia-nvjitlink-cu12-12.4.99 nvidia-nvtx-cu12-12.4.99 pillow-12.0.0 sympy-1.14.0 torch-2.4.0+cu124 torchaudio-2.4.0+cu124 torchvision-0.19.0+cu124 triton-3.0.0 typing-extensions-4.15.0
#21 90.70 WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable.It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
#21 DONE 94.5s

#22 [15/18] RUN python -c "import whisperx; whisperx.load_model('large-v3', 'cpu', compute_type='int8', download_root='/app/models')"
#22 6.185 2026-02-22 09:59:54.193904149 [W:onnxruntime:Default, device_discovery.cc:131 GetPciBusId] Skipping pci_bus_id for PCI path at "/sys/devices/LNXSYSTM:00/LNXSYBUS:00/ACPI0004:00/MSFT1000:00/5620e0c7-8062-4dce-aeb7-520c7ef76171" because filename ""5620e0c7-8062-4dce-aeb7-520c7ef76171"" dit not match expected pattern of [0-9a-f]+:[0-9a-f]+:[0-9a-f]+[.][0-9a-f]+
#22 17.40 2026-02-22 10:00:05 - whisperx.asr - INFO - No language specified, language will be detected for each audio file (increases inference time)
#22 17.40 2026-02-22 10:00:05 - whisperx.vads.pyannote - INFO - Performing voice activity detection using Pyannote...
#22 17.41 /usr/local/lib/python3.11/dist-packages/lightning_fabric/utilities/cloud_io.py:73: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
#22 17.50 Lightning automatically upgraded your loaded checkpoint from v1.5.4 to v2.6.1. To apply the upgrade to your files permanently, run `python -m pytorch_lightning.utilities.upgrade_checkpoint ../usr/local/lib/python3.11/dist-packages/whisperx/assets/pytorch_model.bin`
#22 17.52 Traceback (most recent call last):
#22 17.52   File "<string>", line 1, in <module>
#22 17.52   File "/usr/local/lib/python3.11/dist-packages/whisperx/__init__.py", line 21, in load_model
#22 17.52     return asr.load_model(*args, **kwargs)
#22 17.52            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#22 17.52   File "/usr/local/lib/python3.11/dist-packages/whisperx/asr.py", line 412, in load_model
#22 17.52     vad_model = Pyannote(torch.device(device_vad), token=None, **default_vad_options)
#22 17.52                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#22 17.52   File "/usr/local/lib/python3.11/dist-packages/whisperx/vads/pyannote.py", line 240, in __init__
#22 17.52     self.vad_pipeline = load_vad_model(device, token=token, model_fp=model_fp)
#22 17.52                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#22 17.52   File "/usr/local/lib/python3.11/dist-packages/whisperx/vads/pyannote.py", line 48, in load_vad_model
#22 17.52     vad_pipeline = VoiceActivitySegmentation(segmentation=vad_model, device=torch.device(device))
#22 17.52                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#22 17.52   File "/usr/local/lib/python3.11/dist-packages/whisperx/vads/pyannote.py", line 199, in __init__
#22 17.52     super().__init__(segmentation=segmentation, fscore=fscore, token=token, **inference_kwargs)
#22 17.52   File "/usr/local/lib/python3.11/dist-packages/pyannote/audio/pipelines/voice_activity_detection.py", line 128, in __init__
#22 17.52     self._segmentation = Inference(model, **inference_kwargs)
#22 17.52                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#22 17.52 TypeError: Inference.__init__() got an unexpected keyword argument 'token'
#22 17.52 Model was trained with pyannote.audio 0.0.1, yours is 3.4.0. Bad things might happen unless you revert pyannote.audio to 0.x.
#22 17.52 Model was trained with torch 1.10.0+cu102, yours is 2.4.0+cu124. Bad things might happen unless you revert torch to 1.x.
#22 ERROR: process "/bin/bash -o pipefail -c python -c \"import whisperx; whisperx.load_model('large-v3', 'cpu', compute_type='int8', download_root='/app/models')\"" did not complete successfully: exit code: 1
------
 > importing cache manifest from ***/whisperx-medical:buildcache:
------
------
 > [15/18] RUN python -c "import whisperx; whisperx.load_model('large-v3', 'cpu', compute_type='int8', download_root='/app/models')":
17.52     vad_pipeline = VoiceActivitySegmentation(segmentation=vad_model, device=torch.device(device))
17.52                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
17.52   File "/usr/local/lib/python3.11/dist-packages/whisperx/vads/pyannote.py", line 199, in __init__
17.52     super().__init__(segmentation=segmentation, fscore=fscore, token=token, **inference_kwargs)
17.52   File "/usr/local/lib/python3.11/dist-packages/pyannote/audio/pipelines/voice_activity_detection.py", line 128, in __init__
17.52     self._segmentation = Inference(model, **inference_kwargs)
17.52                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
17.52 TypeError: Inference.__init__() got an unexpected keyword argument 'token'
17.52 Model was trained with pyannote.audio 0.0.1, yours is 3.4.0. Bad things might happen unless you revert pyannote.audio to 0.x.
17.52 Model was trained with torch 1.10.0+cu102, yours is 2.4.0+cu124. Bad things might happen unless you revert torch to 1.x.
------

 2 warnings found (use docker --debug to expand):
 - SecretsUsedInArgOrEnv: Do not use ARG or ENV instructions for sensitive data (ARG "HF_TOKEN") (line 74)
 - SecretsUsedInArgOrEnv: Do not use ARG or ENV instructions for sensitive data (ENV "HF_TOKEN") (line 75)
Dockerfile:77
--------------------
  75 |     ENV HF_TOKEN=$HF_TOKEN
  76 |     
  77 | >>> RUN python -c "import whisperx; whisperx.load_model('large-v3', 'cpu', compute_type='int8', download_root='/app/models')"
  78 |     
  79 |     # Pyannote diarization (v3 uses 'use_auth_token')
--------------------
ERROR: failed to build: failed to solve: process "/bin/bash -o pipefail -c python -c \"import whisperx; whisperx.load_model('large-v3', 'cpu', compute_type='int8', download_root='/app/models')\"" did not complete successfully: exit code: 1
Error: buildx failed with: ERROR: failed to build: failed to solve: process "/bin/bash -o pipefail -c python -c \"import whisperx; whisperx.load_model('large-v3', 'cpu', compute_type='int8', download_root='/app/models')\"" did not complete successfully: exit code: 1
0s
1s
7s
0s
0s
