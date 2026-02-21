# Cloud Migration Guide: Scaling Transcription Speed

This guide explains how to move the transcription tool from a local machine to a high-performance cloud GPU server (like RunPod) to achieve **10x-20x faster processing**.

## 1. Choose a Cloud Provider
We recommend **[RunPod.io](https://www.runpod.io/)** for best value/speed.

### Recommended GPU: NVIDIA RTX 6000 Ada
*   **Why**: Best balance of speed (Ada Lovelace architecture) and memory (48GB VRAM). 
*   **Cost**: ~$0.77 per hour.
*   **Alternative (Ultra-Speed)**: **RTX 5090** (~$0.89/hr) for the absolute latest architecture.
*   **Alternative (Budget)**: **RTX A6000** (~$0.49/hr) if speed is less critical but you still want 48GB VRAM.

## 2. Server Setup (One-Time)
Once your Pod is running, click **Connect** and open the terminal/SSH.

### Install System Dependencies:
```bash
apt-get update && apt-get install -y ffmpeg
```

### Clone and Install:
```bash
git clone [YOUR_GITHUB_REPO_URL]
cd transcriber
pip install -r requirements.txt
```

## 3. Configuration & Running
### Network
1.  In RunPod, ensure you have an **exposed HTTP port** (usually 8000).
2.  Start the server:
    ```bash
    python server.py
    ```
3.  Use the **Public Proxy URL** provided by RunPod to access the web UI from your home browser.

## 4. Performance Optimization (Cloud-Mode)
To maximize speed on enterprise GPUs, we recommend the following code upgrades:

### WhisperX Integration
Unlike standard Whisper, WhisperX uses batched inference.
*   **Result**: 90 minutes of audio transcribes in **3-5 minutes**.
*   **Installation**: `pip install git+https://github.com/m-bain/whisperX.git`

### Persistent VRAM
The server is already configured to keep the model loaded in VRAM, which is essential for cloud billing efficiency. Always ensure `server.py` is running as a persistent process.

## 5. Speed Comparison (90-Minute File)
| Environment | Processing Time | Cost per Run | Verdict |
| :--- | :--- | :--- | :--- |
| **Local CPU/GPU** | ~45-60 Minutes | Free | Slow, ties up your PC. |
| **RTX A6000** ($0.49/hr) | ~8-10 Minutes | ~$0.08 | Reliable budget pro-card. |
| **RTX 6000 Ada** ($0.77/hr) | **~3-4 Minutes** | **~$0.05** | **Optimal. Best value/speed.** |
| **RTX 5090** ($0.89/hr) | ~2-3 Minutes | ~$0.04 | Current speed king. |
