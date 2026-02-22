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
adapted to new libs #23
All jobs
Run details
Annotations
1 error
build-and-push
failed 1 minute ago in 5m 42s
Search logs
2s
0s
4s
1s
5m 26s
#19 2.713   File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
#19 2.713   File "/usr/local/lib/python3.11/dist-packages/transformers/pipelines/__init__.py", line 26, in <module>
#19 2.713     from ..image_processing_utils import BaseImageProcessor
#19 2.713   File "/usr/local/lib/python3.11/dist-packages/transformers/image_processing_utils.py", line 21, in <module>
#19 2.713     from .image_processing_base import BatchFeature, ImageProcessingMixin
#19 2.713   File "/usr/local/lib/python3.11/dist-packages/transformers/image_processing_base.py", line 26, in <module>
#19 2.713     from .image_utils import is_valid_image, load_image
#19 2.713   File "/usr/local/lib/python3.11/dist-packages/transformers/image_utils.py", line 55, in <module>
#19 2.713     from torchvision.transforms import InterpolationMode
#19 2.713   File "/usr/local/lib/python3.11/dist-packages/torchvision/__init__.py", line 10, in <module>
#19 2.714     from torchvision import _meta_registrations, datasets, io, models, ops, transforms, utils  # usort:skip
#19 2.714     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#19 2.714   File "/usr/local/lib/python3.11/dist-packages/torchvision/_meta_registrations.py", line 163, in <module>
#19 2.714     @torch.library.register_fake("torchvision::nms")
#19 2.714      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#19 2.714   File "/usr/local/lib/python3.11/dist-packages/torch/library.py", line 1069, in register
#19 2.714     use_lib._register_fake(
#19 2.714   File "/usr/local/lib/python3.11/dist-packages/torch/library.py", line 219, in _register_fake
#19 2.714     handle = entry.fake_impl.register(
#19 2.714              ^^^^^^^^^^^^^^^^^^^^^^^^^
#19 2.714   File "/usr/local/lib/python3.11/dist-packages/torch/_library/fake_impl.py", line 50, in register
#19 2.714     if torch._C._dispatch_has_kernel_for_dispatch_key(self.qualname, "Meta"):
#19 2.714        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#19 2.714 RuntimeError: operator torchvision::nms does not exist
#19 2.714 
#19 2.714 The above exception was the direct cause of the following exception:
#19 2.714 
#19 2.714 Traceback (most recent call last):
#19 2.714   File "<string>", line 1, in <module>
#19 2.714   File "/usr/local/lib/python3.11/dist-packages/whisperx/__init__.py", line 20, in load_model
#19 2.715     asr = _lazy_import("asr")
#19 2.715           ^^^^^^^^^^^^^^^^^^^
#19 2.715   File "/usr/local/lib/python3.11/dist-packages/whisperx/__init__.py", line 5, in _lazy_import
#19 2.715     module = importlib.import_module(f"whisperx.{name}")
#19 2.715              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#19 2.715   File "/usr/lib/python3.11/importlib/__init__.py", line 126, in import_module
#19 2.715     return _bootstrap._gcd_import(name[level:], package, level)
#19 2.715            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#19 2.715   File "<frozen importlib._bootstrap>", line 1204, in _gcd_import
#19 2.715   File "<frozen importlib._bootstrap>", line 1176, in _find_and_load
#19 2.715   File "<frozen importlib._bootstrap>", line 1147, in _find_and_load_unlocked
#19 2.715   File "<frozen importlib._bootstrap>", line 690, in _load_unlocked
#19 2.715   File "<frozen importlib._bootstrap_external>", line 940, in exec_module
#19 2.715   File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
#19 2.715   File "/usr/local/lib/python3.11/dist-packages/whisperx/asr.py", line 11, in <module>
#19 2.715     from transformers import Pipeline
#19 2.715   File "/usr/local/lib/python3.11/dist-packages/transformers/utils/import_utils.py", line 2320, in __getattr__
#19 2.715     raise ModuleNotFoundError(
#19 2.715 ModuleNotFoundError: Could not import module 'Pipeline'. Are this object's requirements defined correctly?
#19 ERROR: process "/bin/bash -o pipefail -c python -c \"import whisperx; whisperx.load_model('large-v3', 'cpu', compute_type='int8', download_root='/app/models')\"" did not complete successfully: exit code: 1
------
 > importing cache manifest from ***/whisperx-medical:buildcache:
------
------
 > [12/15] RUN python -c "import whisperx; whisperx.load_model('large-v3', 'cpu', compute_type='int8', download_root='/app/models')":
2.715   File "<frozen importlib._bootstrap>", line 1176, in _find_and_load
2.715   File "<frozen importlib._bootstrap>", line 1147, in _find_and_load_unlocked
2.715   File "<frozen importlib._bootstrap>", line 690, in _load_unlocked
2.715   File "<frozen importlib._bootstrap_external>", line 940, in exec_module
2.715   File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
2.715   File "/usr/local/lib/python3.11/dist-packages/whisperx/asr.py", line 11, in <module>
2.715     from transformers import Pipeline
2.715   File "/usr/local/lib/python3.11/dist-packages/transformers/utils/import_utils.py", line 2320, in __getattr__
2.715     raise ModuleNotFoundError(
2.715 ModuleNotFoundError: Could not import module 'Pipeline'. Are this object's requirements defined correctly?
------

 2 warnings found (use docker --debug to expand):
 - SecretsUsedInArgOrEnv: Do not use ARG or ENV instructions for sensitive data (ARG "HF_TOKEN") (line 55)
 - SecretsUsedInArgOrEnv: Do not use ARG or ENV instructions for sensitive data (ENV "HF_TOKEN") (line 56)
Dockerfile:58
--------------------
  56 |     ENV HF_TOKEN=$HF_TOKEN
  57 |     
  58 | >>> RUN python -c "import whisperx; whisperx.load_model('large-v3', 'cpu', compute_type='int8', download_root='/app/models')"
  59 |     
  60 |     # Pyannote diarization (v4 uses 'token' instead of 'use_auth_token')
--------------------
ERROR: failed to build: failed to solve: process "/bin/bash -o pipefail -c python -c \"import whisperx; whisperx.load_model('large-v3', 'cpu', compute_type='int8', download_root='/app/models')\"" did not complete successfully: exit code: 1
Error: buildx failed with: ERROR: failed to build: failed to solve: process "/bin/bash -o pipefail -c python -c \"import whisperx; whisperx.load_model('large-v3', 'cpu', compute_type='int8', download_root='/app/models')\"" did not complete successfully: exit code: 1
0s
0s
7s
0s
0s
