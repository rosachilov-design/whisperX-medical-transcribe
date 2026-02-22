stalling collected packages: whisperx
#19 3.319 Successfully installed whisperx-3.1.1
#19 3.319 WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv
#19 3.428 
#19 3.428 [notice] A new release of pip is available: 23.3.1 -> 26.0.1
#19 3.428 [notice] To update, run: python -m pip install --upgrade pip
#19 DONE 3.5s

#20 [13/16] RUN python -c "import whisperx; whisperx.load_model('large-v3', 'cpu', compute_type='int8', download_root='/app/models')"
#20 3.851 /usr/local/lib/python3.10/dist-packages/pyannote/audio/core/io.py:43: UserWarning: torchaudio._backend.set_audio_backend has been deprecated. With dispatcher enabled, this function is no-op. You can remove the function call.
#20 3.851   torchaudio.set_audio_backend("soundfile")
#20 3.974 /usr/local/lib/python3.10/dist-packages/pyannote/audio/pipelines/speaker_verification.py:43: UserWarning: torchaudio._backend.get_audio_backend has been deprecated. With dispatcher enabled, this function is no-op. You can remove the function call.
#20 3.974   backend = torchaudio.get_audio_backend()
#20 4.212 /usr/local/lib/python3.10/dist-packages/pyannote/audio/pipelines/speaker_verification.py:45: UserWarning: Module 'speechbrain.pretrained' was deprecated, redirecting to 'speechbrain.inference'. Please update your script. This is a change from SpeechBrain 1.0. See: https://github.com/speechbrain/speechbrain/releases/tag/v1.0.0
#20 4.212   from speechbrain.pretrained import (
#20 4.227 /usr/local/lib/python3.10/dist-packages/pyannote/audio/pipelines/speaker_verification.py:53: UserWarning: torchaudio._backend.set_audio_backend has been deprecated. With dispatcher enabled, this function is no-op. You can remove the function call.
#20 4.227   torchaudio.set_audio_backend(backend)
#20 4.248 /usr/local/lib/python3.10/dist-packages/pyannote/audio/tasks/segmentation/mixins.py:37: UserWarning: `torchaudio.backend.common.AudioMetaData` has been moved to `torchaudio.AudioMetaData`. Please update the import path.
#20 4.248   from torchaudio.backend.common import AudioMetaData
#20 15.44 Traceback (most recent call last):
#20 15.44   File "<string>", line 1, in <module>
#20 15.44   File "/usr/local/lib/python3.10/dist-packages/whisperx/asr.py", line 332, in load_model
#20 15.44     default_asr_options = faster_whisper.transcribe.TranscriptionOptions(**default_asr_options)
#20 15.44 TypeError: TranscriptionOptions.__init__() missing 5 required positional arguments: 'multilingual', 'max_new_tokens', 'clip_timestamps', 'hallucination_silence_threshold', and 'hotwords'
#20 15.44 No language specified, language will be first be detected for each audio file (increases inference time).
#20 ERROR: process "/bin/bash -o pipefail -c python -c \"import whisperx; whisperx.load_model('large-v3', 'cpu', compute_type='int8', download_root='/app/models')\"" did not complete successfully: exit code: 1
------
 > importing cache manifest from ***/whisperx-medical:buildcache:
------
------
 > [13/16] RUN python -c "import whisperx; whisperx.load_model('large-v3', 'cpu', compute_type='int8', download_root='/app/models')":
4.227 /usr/local/lib/python3.10/dist-packages/pyannote/audio/pipelines/speaker_verification.py:53: UserWarning: torchaudio._backend.set_audio_backend has been deprecated. With dispatcher enabled, this function is no-op. You can remove the function call.
4.227   torchaudio.set_audio_backend(backend)
4.248 /usr/local/lib/python3.10/dist-packages/pyannote/audio/tasks/segmentation/mixins.py:37: UserWarning: `torchaudio.backend.common.AudioMetaData` has been moved to `torchaudio.AudioMetaData`. Please update the import path.
4.248   from torchaudio.backend.common import AudioMetaData
15.44 Traceback (most recent call last):
15.44   File "<string>", line 1, in <module>
15.44   File "/usr/local/lib/python3.10/dist-packages/whisperx/asr.py", line 332, in load_model
15.44     default_asr_options = faster_whisper.transcribe.TranscriptionOptions(**default_asr_options)
15.44 TypeError: TranscriptionOptions.__init__() missing 5 required positional arguments: 'multilingual', 'max_new_tokens', 'clip_timestamps', 'hallucination_silence_threshold', and 'hotwords'
15.44 No language specified, language will be first be detected for each audio file (increases inference time).
------

 2 warnings found (use docker --debug to expand):
 - SecretsUsedInArgOrEnv: Do not use ARG or ENV instructions for sensitive data (ARG "HF_TOKEN") (line 33)
 - SecretsUsedInArgOrEnv: Do not use ARG or ENV instructions for sensitive data (ENV "HF_TOKEN") (line 34)
Dockerfile:36
--------------------
  34 |     ENV HF_TOKEN=$HF_TOKEN
  35 |     
  36 | >>> RUN python -c "import whisperx; whisperx.load_model('large-v3', 'cpu', compute_type='int8', download_root='/app/models')"
  37 |     
  38 |     # Pyannote diarization (try to pre-download, but don't fail build if token is missing/invalid)
--------------------
ERROR: failed to build: failed to solve: process "/bin/bash -o pipefail -c python -c \"import whisperx; whisperx.load_model('large-v3', 'cpu', compute_type='int8', download_root='/app/models')\"" did not complete successfully: exit code: 1
Error: buildx failed with: ERROR: failed to build: failed to solve: process "/bin/bash -o pipefail -c python -c \"import whisperx; whisperx.load_model('large-v3', 'cpu', compute_type='int8', download_root='/app/models')\"" did not complete successfully: exit code: 1