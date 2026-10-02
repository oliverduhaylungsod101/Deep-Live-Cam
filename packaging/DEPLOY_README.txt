Deep-Live-Cam / Swapface Live — Windows deploy
==============================================

Fully offline package: no pip install and no model download required.

Contents
--------
- run.exe              Main launcher (defaults to CUDA)
- _internal\           Python runtime + libraries (incl. CUDA ORT)
- ffmpeg\bin\          Bundled ffmpeg tools
- models\              Face-swap / enhancer / Whisper small.en / buffalo_l
- locales\             UI translations

How to run
----------
1. Keep this whole folder together (do not move run.exe alone).
2. Double-click run.exe
   or:
     run.exe --execution-provider cuda

System requirements
-------------------
- Windows 10/11 64-bit
- NVIDIA GPU + recent Game Ready / Studio driver
  (driver only — CUDA/cuDNN libs are already bundled)
- No Python / pip install needed

Notes
-----
- First launch can take longer while models and WebEngine initialize.
- ChatGPT login data is stored under browser_profile\ next to run.exe.
- Torch is optional and not bundled; ONNX Runtime CUDA is used instead.

Rebuild from source
-------------------
  packaging\build_exe.bat
