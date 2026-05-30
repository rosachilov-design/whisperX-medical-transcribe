let wavesurfer;
let currentTaskId = null;
let currentFile = null;
let statusInterval = null;
let segments = [];
let lastSegmentCount = 0;

// DOM Elements
const launchScreen = document.getElementById('launch-screen');
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const mainInterface = document.getElementById('main-interface');
const filenameDisplay = document.getElementById('filename-display');
const progressBar = document.getElementById('progress-bar');
const statusText = document.getElementById('status-text');
const percentText = document.getElementById('percent-text');
const playPauseBtn = document.getElementById('play-pause');
const playIcon = document.getElementById('play-icon');
const transcriptionContent = document.getElementById('transcription-content');
const footerActions = document.getElementById('footer-actions');
const jsonFilenameSpan = document.getElementById('json-filename-span');
const openJsonBtn = document.getElementById('open-json-btn');
const resetTranscriptionBtn = document.getElementById('reset-transcription-btn');
const removeFileBtn = document.getElementById('remove-file');
const currentTimeDisplay = document.getElementById('current-time');
const durationDisplay = document.getElementById('duration');
const startDiarizationBtn = document.getElementById('start-diarization-btn');
const startTranscriptionBtn = document.getElementById('start-transcription-btn');
const newSessionBtn = document.getElementById('new-session-btn');
const progressSection = document.getElementById('progress-section');
const improveAudioDropzone = document.getElementById('improve-audio-dropzone');
const improveAudioInput = document.getElementById('improve-audio-input');
const improveAudioBrowseBtn = document.getElementById('improve-audio-browse');
const improveAudioStatus = document.getElementById('improve-audio-status');
const s3FileList = document.getElementById('s3-file-list');
const s3BrowserStatus = document.getElementById('s3-browser-status');
const refreshS3Btn = document.getElementById('refresh-s3-btn');
const deleteAllS3Btn = document.getElementById('delete-all-s3-btn');
const waveformContainer = document.getElementById('waveform-container');
const videoPlayer = document.getElementById('video-player');
const videoParticipantsPanel = document.getElementById('video-participants-panel');
const videoParticipantsInput = document.getElementById('video-participants-input');
const ocrSummary = document.getElementById('ocr-summary');
const localTestBtn = document.getElementById('local-test-btn');
const localHunyuanTestBtn = document.getElementById('local-hunyuan-test-btn');

// Pod Control Elements
const setupPodBtn = document.getElementById('setup-pod-btn');
const resumePodBtn = document.getElementById('resume-pod-btn');
const startWorkerBtn = document.getElementById('start-worker-btn');
const stopPodBtn = document.getElementById('stop-pod-btn');
const podStatusBadge = document.getElementById('pod-status-badge');
const logConsole = document.getElementById('log-console');
const saveConfigBtn = document.getElementById('save-config-btn');
const saveWorkersBtn = document.getElementById('save-workers-btn');
const podIpInput = document.getElementById('pod-ip-input');
const podPortInput = document.getElementById('pod-port-input');
const podIdInput = document.getElementById('pod-id-input');
const endpointIdInput = document.getElementById('endpoint-id-input');
const workersMaxInput = document.getElementById('workers-max-input');
const podKeyInput = document.getElementById('pod-key-input');

let podPollingInterval = null;
let logPollingInterval = null;
let currentIsVideo = false;

// ─── Dropzone & File Input ───

// Drag events on the entire launch screen
document.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragging');
});

document.addEventListener('dragleave', (e) => {
    if (e.relatedTarget === null) {
        dropzone.classList.remove('dragging');
        improveAudioDropzone.classList.remove('dragging');
    }
});

document.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragging');
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFile(files[0]);
});

// Click on card opens file browser
dropzone.addEventListener('click', (e) => {
    // Don't trigger if clicking a button
    if (e.target.closest('button')) return;
    fileInput.click();
});

fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
});

improveAudioDropzone.addEventListener('click', () => improveAudioInput.click());
improveAudioBrowseBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    improveAudioInput.click();
});
improveAudioInput.addEventListener('change', () => {
    if (improveAudioInput.files.length > 0) handleImproveAudio(improveAudioInput.files[0]);
});
improveAudioDropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.remove('dragging');
    improveAudioDropzone.classList.add('dragging');
});
improveAudioDropzone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!improveAudioDropzone.contains(e.relatedTarget)) {
        improveAudioDropzone.classList.remove('dragging');
    }
});
improveAudioDropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    e.stopPropagation();
    improveAudioDropzone.classList.remove('dragging');
    const files = e.dataTransfer.files;
    if (files.length > 0) handleImproveAudio(files[0]);
});

function setImproveAudioStatus(message, tone = 'muted') {
    improveAudioStatus.textContent = message;
    improveAudioStatus.className = `improve-audio-status ${tone}`;
}

async function handleImproveAudio(file) {
    if (!file.name.toLowerCase().endsWith('.m4a')) {
        setImproveAudioStatus('Please drop an .m4a file here.', 'error');
        return;
    }

    improveAudioDropzone.classList.remove('dragging');
    improveAudioDropzone.classList.add('processing');
    setImproveAudioStatus('Improving audio quality...', 'working');

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/improve-audio', {
            method: 'POST',
            body: formData
        });
        const data = await response.json().catch(() => ({}));

        if (!response.ok || data.error) {
            throw new Error(data.error || 'Audio improvement failed.');
        }

        const a = document.createElement('a');
        a.href = data.download_url;
        a.download = data.filename;
        document.body.appendChild(a);
        a.click();
        a.remove();

        setImproveAudioStatus(`Done. Downloading ${data.filename}`, 'success');
    } catch (err) {
        setImproveAudioStatus(`Error: ${err.message}`, 'error');
    } finally {
        improveAudioDropzone.classList.remove('processing');
        improveAudioInput.value = '';
    }
}

function formatBytes(bytes) {
    if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB'];
    let value = bytes;
    let unitIndex = 0;
    while (value >= 1024 && unitIndex < units.length - 1) {
        value /= 1024;
        unitIndex += 1;
    }
    return `${value.toFixed(value >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function formatS3Date(value) {
    if (!value) return 'Unknown time';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return date.toLocaleString();
}

function renderS3Files(files) {
    if (!files.length) {
        s3FileList.innerHTML = '<div class="s3-empty-state">No files found in the bucket.</div>';
        return;
    }

    s3FileList.innerHTML = '';
    files.forEach((file) => {
        const row = document.createElement('div');
        row.className = 's3-file-row';
        row.innerHTML = `
            <div class="s3-file-meta">
                <div class="s3-file-name">${file.name}</div>
                <div class="s3-file-key">${file.key}</div>
                <div class="s3-file-info">${formatBytes(file.size)} · ${formatS3Date(file.last_modified)}</div>
            </div>
            <button class="s3-delete-btn" type="button">Delete</button>
        `;

        row.querySelector('.s3-delete-btn').addEventListener('click', () => {
            deleteS3File(file.key, file.name);
        });
        s3FileList.appendChild(row);
    });
}

async function loadS3Files() {
    s3BrowserStatus.textContent = 'Loading files from S3...';
    refreshS3Btn.disabled = true;
    deleteAllS3Btn.disabled = true;

    try {
        const response = await fetch('/s3-files');
        const data = await response.json().catch(() => ({}));

        if (!response.ok || data.error) {
            throw new Error(data.error || 'Failed to load S3 files.');
        }

        renderS3Files(data.files || []);
        s3BrowserStatus.textContent = `${(data.files || []).length} file(s) currently on the server.`;
        deleteAllS3Btn.disabled = !(data.files || []).length;
    } catch (err) {
        s3FileList.innerHTML = '<div class="s3-empty-state">Could not load bucket contents.</div>';
        s3BrowserStatus.textContent = `Error: ${err.message}`;
    } finally {
        refreshS3Btn.disabled = false;
    }
}

async function deleteAllS3Files() {
    const warning = [
        'Delete EVERY file from the RunPod S3 bucket?',
        '',
        'This removes audio, video, JSON, and any other objects listed here from RunPod storage.',
        'It also clears the local server uploads/results cache.',
        'This cannot be undone.'
    ].join('\n');
    if (!confirm(warning)) return;

    s3BrowserStatus.textContent = 'Deleting all files from RunPod storage...';
    refreshS3Btn.disabled = true;
    deleteAllS3Btn.disabled = true;

    try {
        const response = await fetch('/delete-all-s3-files', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ confirm: 'DELETE ALL' })
        });
        const data = await response.json().catch(() => ({}));

        if (!response.ok || data.error) {
            throw new Error(data.error || 'Delete all failed.');
        }

        s3BrowserStatus.textContent = `Deleted ${data.deleted_remote || 0} remote and ${data.deleted_local || 0} local server file(s).`;
        await loadS3Files();
    } catch (err) {
        s3BrowserStatus.textContent = `Error: ${err.message}`;
        deleteAllS3Btn.disabled = false;
    } finally {
        refreshS3Btn.disabled = false;
    }
}

async function deleteS3File(key, name) {
    if (!confirm(`Delete ${name} from the S3 bucket?`)) return;

    s3BrowserStatus.textContent = `Deleting ${name}...`;

    try {
        const response = await fetch('/delete-s3-file', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ key })
        });
        const data = await response.json().catch(() => ({}));

        if (!response.ok || data.error) {
            throw new Error(data.error || 'Delete failed.');
        }

        await loadS3Files();
    } catch (err) {
        s3BrowserStatus.textContent = `Error: ${err.message}`;
    }
}

function initWaveSurfer(url) {
    if (wavesurfer) wavesurfer.destroy();
    videoPlayer.pause();
    videoPlayer.removeAttribute('src');
    videoPlayer.load();

    wavesurfer = WaveSurfer.create({
        container: '#waveform',
        waveColor: '#475569',
        progressColor: '#f97316',
        cursorColor: '#f97316',
        barWidth: 3,
        barGap: 3,
        barRadius: 4,
        responsive: true,
        height: 180,
        normalize: true
    });

    wavesurfer.load(url);

    wavesurfer.on('ready', () => {
        durationDisplay.textContent = formatTime(wavesurfer.getDuration());
    });

    wavesurfer.on('audioprocess', (time) => {
        currentTimeDisplay.textContent = formatTime(time);
        highlightTranscription(time);
    });

    wavesurfer.on('seek', (prog) => {
        const time = prog * wavesurfer.getDuration();
        currentTimeDisplay.textContent = formatTime(time);
        highlightTranscription(time);
    });

    wavesurfer.on('play', () => { playIcon.className = 'pause-icon'; });
    wavesurfer.on('pause', () => { playIcon.className = 'play-icon'; });
}

function initVideoPlayer(url) {
    if (wavesurfer) {
        wavesurfer.destroy();
        wavesurfer = null;
    }

    videoPlayer.src = url;
    videoPlayer.load();

    videoPlayer.onloadedmetadata = () => {
        durationDisplay.textContent = formatTime(videoPlayer.duration || 0);
    };
    videoPlayer.ontimeupdate = () => {
        currentTimeDisplay.textContent = formatTime(videoPlayer.currentTime || 0);
        highlightTranscription(videoPlayer.currentTime || 0);
    };
    videoPlayer.onplay = () => { playIcon.className = 'pause-icon'; };
    videoPlayer.onpause = () => { playIcon.className = 'play-icon'; };
}

function setPlaybackMode(isVideo) {
    currentIsVideo = isVideo;
    document.getElementById('waveform').classList.toggle('hidden', isVideo);
    videoPlayer.classList.toggle('hidden', !isVideo);
    waveformContainer.classList.toggle('video-mode', isVideo);
    document.querySelector('.audio-controls').classList.toggle('hidden', isVideo);
    document.querySelector('.speaker-config').classList.remove('hidden');
    videoParticipantsPanel.classList.toggle('hidden', !isVideo);
    localTestBtn.classList.toggle('hidden', !isVideo);
    localHunyuanTestBtn.classList.toggle('hidden', !isVideo);
    document.querySelector('.speaker-management').classList.toggle('hidden', isVideo);
    ocrSummary.classList.add('hidden');
    ocrSummary.innerHTML = '';
}

function formatTime(seconds) {
    const minutes = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
}

function formatWait(totalSeconds) {
    const mins = Math.floor(totalSeconds / 60);
    const secs = totalSeconds % 60;
    if (mins > 0) return `${mins}m ${secs}s`;
    return `${secs}s`;
}

function getJsonFilename(data) {
    if (data.json_path) return data.json_path;
    const filename = data.filename || currentTaskId || 'transcript';
    return filename.includes('.') ? filename.replace(/\.[^.]+$/, '.json') : `${filename}.json`;
}

let knownSpeakers = [];

function parseKnownParticipants() {
    return videoParticipantsInput.value
        .split(';')
        .map(name => name.trim().replace(/\s+/g, ' '))
        .filter((name, index, all) => name && all.indexOf(name) === index);
}

function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, char => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;'
    }[char]));
}

function renderOcrSummary(data) {
    const summary = data?.ocr_diarization;
    if (!summary || !currentIsVideo) {
        ocrSummary.classList.add('hidden');
        ocrSummary.innerHTML = '';
        return;
    }

    const unknownRate = Math.round((summary.unknown_rate || 0) * 100);
    const namedRate = Math.max(0, 100 - unknownRate);
    const known = summary.known_speakers || [];
    const discovered = (summary.discovered_speakers || []).map(item => item.speaker);
    const speakers = known.length ? known : discovered;
    const reviewItems = (summary.low_confidence_intervals || []).slice(0, 4)
        .map(item => formatTime(item.start))
        .join(', ');

    ocrSummary.innerHTML = `
        <div class="ocr-summary-title">OCR speaker timeline</div>
        <div class="ocr-summary-grid">
            <span>Named ${namedRate}% / Unknown ${unknownRate}%</span>
            <span>Speakers: ${speakers.length ? escapeHtml(speakers.join(', ')) : 'auto-discovery found none'}</span>
            <span>Needs review: ${escapeHtml(reviewItems || 'none')}</span>
        </div>
    `;
    ocrSummary.classList.remove('hidden');
}

async function uploadCurrentFileToCloud() {
    if (!currentFile) {
        statusText.textContent = 'Choose the source file again to upload it for a fresh transcription.';
        return;
    }

    startDiarizationBtn.classList.remove('hidden');
    startDiarizationBtn.disabled = true;
    startDiarizationBtn.textContent = 'Uploading...';
    startTranscriptionBtn.classList.add('hidden');
    startTranscriptionBtn.disabled = true;
    footerActions.classList.add('hidden');
    progressSection.classList.remove('hidden');
    statusText.textContent = 'Uploading to cloud storage...';
    percentText.textContent = '';
    progressBar.style.width = '0%';
    transcriptionContent.innerHTML = '<div class="placeholder-text">Your transcription will appear here...</div>';
    segments = [];
    lastSegmentCount = 0;

    const formData = new FormData();
    formData.append('file', currentFile);

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        currentTaskId = data.task_id;

        if (data.task_id) {
            const statusResp = await fetch(`/status/${encodeURIComponent(data.task_id)}`);
            const statusData = await statusResp.json();
            if (statusData.status === 'completed') {
                loadCompletedTranscription(statusData);
                return;
            }
        }

        startPolling();
    } catch (err) {
        statusText.textContent = `Upload failed: ${err.message}`;
    }
}

async function handleFile(file) {
    currentFile = file;
    filenameDisplay.textContent = file.name;
    launchScreen.classList.add('hidden');
    mainInterface.classList.remove('hidden');
    knownSpeakers = [];
    renderSpeakerList();
    setPlaybackMode(file.name.toLowerCase().endsWith('.mp4'));

    // Load local preview immediately
    const url = URL.createObjectURL(file);
    if (currentIsVideo) {
        initVideoPlayer(url);
    } else {
        initWaveSurfer(url);
    }

    // Check if a transcription already exists for this file
    try {
        const checkResponse = await fetch(`/check/${encodeURIComponent(file.name)}`);
        const checkData = await checkResponse.json();

        if (checkData.status === 'completed' && checkData.result) {
            // Transcription exists! Load it directly
            currentTaskId = file.name;
            loadCompletedTranscription(checkData);
            return;
        }
    } catch (err) {
        console.log('No existing transcription found, proceeding with upload.');
    }

    await uploadCurrentFileToCloud();
    return;
}


// ─── Load Completed Transcription ───

function loadCompletedTranscription(data) {
    progressSection.classList.add('hidden');
    startDiarizationBtn.classList.add('hidden');
    startTranscriptionBtn.classList.add('hidden');
    statusText.textContent = '✅ Transcription loaded';

    transcriptionContent.innerHTML = '';
    segments = data.result;
    lastSegmentCount = 0;
    renderOcrSummary(data);

    data.result.forEach((seg, idx) => {
        const div = createSegmentEl(seg, idx);
        transcriptionContent.appendChild(div);
    });
    lastSegmentCount = data.result.length;

    const jsonPath = getJsonFilename(data);
    jsonFilenameSpan.textContent = jsonPath;
    openJsonBtn.onclick = () => {
        const a = document.createElement('a');
        a.href = `/download/${jsonPath}`;
        a.download = jsonPath;
        a.click();
    };
    footerActions.classList.remove('hidden');

    // Collect unique speakers from the transcription
    const speakerSet = new Set(data.result.map(s => s.speaker));
    knownSpeakers = [...speakerSet];
    renderSpeakerList();
    refreshAllSegments();
}

async function resetCurrentTranscription() {
    if (!currentTaskId) return;

    resetTranscriptionBtn.disabled = true;
    const filename = currentTaskId;

    try {
        const storageResp = await fetch(`/transcription-storage/${encodeURIComponent(filename)}`);
        const storage = await storageResp.json().catch(() => ({}));
        if (!storageResp.ok || storage.error) {
            throw new Error(storage.error || 'Could not check transcription storage.');
        }

        const remoteKeys = storage.remote_json_keys || [];
        const message = remoteKeys.length
            ? [
                `Reset transcription for ${filename}?`,
                '',
                'This will delete the local JSON and the RunPod/S3 JSON result:',
                ...remoteKeys.map(key => `- ${key}`),
                '',
                'The source media will stay available so you can transcribe again.'
            ].join('\n')
            : [
                `Reset transcription for ${filename}?`,
                '',
                'This will delete the local JSON result. The source media will stay available.'
            ].join('\n');

        if (!confirm(message)) return;

        statusText.textContent = 'Resetting transcription...';
        const resetResp = await fetch('/reset-transcription', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename })
        });
        const resetData = await resetResp.json().catch(() => ({}));
        if (!resetResp.ok || resetData.error) {
            throw new Error(resetData.error || 'Reset failed.');
        }

        footerActions.classList.add('hidden');
        transcriptionContent.innerHTML = '<div class="placeholder-text">Your transcription will appear here...</div>';
        segments = [];
        lastSegmentCount = 0;
        renderOcrSummary({});

        if (resetData.task?.status === 'uploaded') {
            updateUI(resetData.task);
            return;
        }

        await uploadCurrentFileToCloud();
    } catch (err) {
        statusText.textContent = `Reset failed: ${err.message}`;
    } finally {
        resetTranscriptionBtn.disabled = false;
    }
}

resetTranscriptionBtn.addEventListener('click', resetCurrentTranscription);


// ─── Start Diarization & Transcription ───

startDiarizationBtn.onclick = async () => {
    if (!currentTaskId) return;

    const minSpeakers = document.getElementById('min-speakers-input').value;
    const maxSpeakers = document.getElementById('max-speakers-input').value;
    const numSpeakers = document.getElementById('num-speakers-input').value;

    startDiarizationBtn.disabled = true;
    startDiarizationBtn.textContent = 'Processing...';
    progressSection.classList.remove('hidden');
    statusText.textContent = '🗣️ Identifying speakers (GPU Processing)...';

    try {
        let url = `/process-cloud/${encodeURIComponent(currentTaskId)}?min_speakers=${minSpeakers}&max_speakers=${maxSpeakers}`;
        if (numSpeakers) url += `&num_speakers=${numSpeakers}`;
        const body = currentIsVideo ? { known_speakers: parseKnownParticipants() } : {};
        const r = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        const cloudResult = await r.json();
        if (cloudResult.status === 'started' || cloudResult.status === 'completed') {
            startPolling();
        } else {
            statusText.textContent = `❌ Cloud Error: ${cloudResult.error || JSON.stringify(cloudResult)}`;
        }
    } catch (err) {
        statusText.textContent = `❌ Network Error: ${err.message}`;
    }
};

async function runLocalOcr(engine) {
    if (!currentTaskId || !currentIsVideo) return;

    const activeBtn = engine === 'hybrid' ? localHunyuanTestBtn : localTestBtn;
    activeBtn.disabled = true;
    activeBtn.textContent = engine === 'hybrid' ? 'Running fallback...' : 'Running Paddle...';
    progressSection.classList.remove('hidden');
    statusText.textContent = `Running local ${engine} OCR diarization on this computer...`;

    try {
        const response = await fetch(`/process-local/${encodeURIComponent(currentTaskId)}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                known_speakers: parseKnownParticipants(),
                ocr_engine: engine
            })
        });
        const result = await response.json();

        if (!response.ok || result.error) {
            throw new Error(result.error || 'Local OCR failed to start.');
        }

        startPolling(500);
    } catch (err) {
        statusText.textContent = `Local ${engine} OCR error: ${err.message}`;
        activeBtn.disabled = false;
        activeBtn.textContent = engine === 'hybrid' ? 'Run Paddle + Hunyuan fallback' : 'Run local Paddle OCR';
    }
}

deleteAllS3Btn.addEventListener('click', deleteAllS3Files);

localTestBtn.onclick = () => runLocalOcr('paddle');
localHunyuanTestBtn.onclick = () => runLocalOcr('hybrid');

startTranscriptionBtn.onclick = async () => {
    if (!currentTaskId) return;

    startTranscriptionBtn.disabled = true;
    startTranscriptionBtn.textContent = 'Transcribing...';
    progressSection.classList.remove('hidden');
    statusText.textContent = '🧠 Transcribing and aligning (GPU Processing)...';

    try {
        const r = await fetch(`/transcribe-cloud/${encodeURIComponent(currentTaskId)}`, { method: 'POST' });
        const cloudResult = await r.json();
        if (cloudResult.status === 'started' || cloudResult.status === 'completed') {
            startPolling();
        } else {
            statusText.textContent = `❌ Cloud Error: ${cloudResult.error || JSON.stringify(cloudResult)}`;
        }
    } catch (err) {
        statusText.textContent = `❌ Network Error: ${err.message}`;
    }
};

newSessionBtn.onclick = () => {
    location.reload();
};


// ─── Polling ───

function startPolling(intervalMs = 2000) {
    if (statusInterval) clearInterval(statusInterval);
    lastSegmentCount = 0;

    statusInterval = setInterval(async () => {
        if (!currentTaskId) return;

        try {
            const response = await fetch(`/status/${encodeURIComponent(currentTaskId)}`);
            const data = await response.json();

            updateUI(data);

            if (data.status === 'completed' || data.status === 'not_found' || data.status === 'error') {
                clearInterval(statusInterval);
            }
        } catch (err) {
            console.error('Status check failed:', err);
        }
    }, intervalMs);
}


// ─── UI Updates ───

function updateUI(data) {
    if (data.status === 'uploading') {
        progressSection.classList.remove('hidden');
        statusText.textContent = '☁️ Uploading to cloud...';
        percentText.textContent = `${data.progress || 0}%`;
        progressBar.style.width = `${data.progress || 0}%`;
    }
    else if (data.status === 'uploaded') {
        // S3 upload done — Ready for Diarization
        progressSection.classList.add('hidden');
        startDiarizationBtn.classList.remove('hidden');
        startDiarizationBtn.disabled = false;
        startDiarizationBtn.textContent = 'Start Full Transcription';
        statusText.textContent = '☁️ File uploaded to cloud. Ready to Diarize.';
        startTranscriptionBtn.classList.add('hidden');
        clearInterval(statusInterval);
    }
    else if (data.status === 'processing') {
        progressSection.classList.remove('hidden');
        statusText.textContent = 'Diarizing, transcribing, and aligning...';
        if (data.runpod_progress_message) statusText.textContent = data.runpod_progress_message;
        percentText.textContent = `${data.progress || 0}%`;
        progressBar.style.width = `${data.progress || 0}%`;
    }
    else if (data.status === 'diarizing') {
        progressSection.classList.remove('hidden');
        statusText.textContent = '🗣️ Identifying speakers...';
        percentText.textContent = `${data.progress || 0}%`;
        progressBar.style.width = `${data.progress || 0}%`;
    }
    else if (data.status === 'local_ocr_processing') {
        progressSection.classList.remove('hidden');
        const frame = data.ocr_frame;
        const total = data.ocr_frames_total;
        const device = data.ocr_device ? ` [${data.ocr_device}]` : '';
        if (frame != null && total != null && total > 0) {
            statusText.textContent = `Local OCR: frame ${frame}/${total}${device}`;
        } else {
            statusText.textContent = `Running local OCR diarization...${device}`;
        }
        percentText.textContent = `${data.progress || 0}%`;
        progressBar.style.width = `${data.progress || 0}%`;
    }
    else if (data.status === 'diarization_complete') {
        progressSection.classList.add('hidden');
        startDiarizationBtn.classList.add('hidden');
        startTranscriptionBtn.classList.remove('hidden');
        startTranscriptionBtn.disabled = false;
        localTestBtn.disabled = false;
        localTestBtn.textContent = 'Run local Paddle OCR';
        localHunyuanTestBtn.disabled = false;
        localHunyuanTestBtn.textContent = 'Run Paddle + Hunyuan fallback';
        renderOcrSummary(data);
        statusText.textContent = '✅ Diarization complete. Ready to Transcribe.';
        clearInterval(statusInterval);
    }
    else if (data.status === 'transcribing') {
        progressSection.classList.remove('hidden');
        statusText.textContent = '🧠 Transcribing...';
        percentText.textContent = `${data.progress || 0}%`;
        progressBar.style.width = `${data.progress || 0}%`;
    }
    else if (data.status === 'completed') {
        loadCompletedTranscription(data);
        clearInterval(statusInterval);
    }
    else if (data.status === 'error') {
        progressSection.classList.remove('hidden');
        statusText.textContent = `❌ Error: ${data.error || 'Unknown'}`;
        percentText.textContent = '';
        progressBar.style.width = '0%';
    }

    if (data.runpod_progress_message && ['processing', 'diarizing', 'transcribing'].includes(data.status)) {
        statusText.textContent = data.runpod_progress_message;
    }

    // Handle live result segments (if transcribing locally)
    if (data.result && data.result.length > 0 && data.result.length > lastSegmentCount) {
        if (lastSegmentCount === 0) transcriptionContent.innerHTML = '';

        const newSegments = data.result.slice(lastSegmentCount);
        newSegments.forEach((seg, idx) => {
            const globalIdx = lastSegmentCount + idx;
            const div = createSegmentEl(seg, globalIdx);
            transcriptionContent.appendChild(div);
        });

        segments = data.result;
        lastSegmentCount = data.result.length;
        transcriptionContent.scrollTop = transcriptionContent.scrollHeight;
    }
}


// ─── Speaker Management ───

const speakerInput = document.getElementById('speaker-input');
const addSpeakerBtn = document.getElementById('add-speaker-btn');
const speakerList = document.getElementById('speaker-list');

addSpeakerBtn.onclick = () => {
    const name = speakerInput.value.trim();
    if (name && !knownSpeakers.includes(name) && knownSpeakers.length < 10) {
        knownSpeakers.push(name);
        speakerInput.value = '';
        renderSpeakerList();
        refreshAllSegments();
    }
};

function renderSpeakerList() {
    speakerList.innerHTML = '';
    knownSpeakers.forEach(name => {
        const chip = document.createElement('div');
        chip.className = 'speaker-name-chip';
        chip.textContent = name;
        speakerList.appendChild(chip);
    });
}

function refreshAllSegments() {
    segments.forEach((seg, index) => {
        const el = document.getElementById(`segment-${index}`);
        if (el) {
            const selector = el.querySelector('.segment-speaker-selector');
            if (selector) selector.innerHTML = getSpeakerPillsHTML(index, seg.speaker);
        }
    });
}

function getSpeakerPillsHTML(index, activeSpeaker) {
    return knownSpeakers.map(name => `
        <span class="speaker-pill ${activeSpeaker === name ? 'active' : ''}" 
              onclick="setSegmentSpeaker(${index}, '${name}')">${name}</span>
    `).join('');
}

async function setSegmentSpeaker(index, name) {
    if (!currentTaskId) return;
    try {
        const response = await fetch('/update_speaker', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                task_id: currentTaskId,
                segment_index: index,
                speaker_name: name
            })
        });

        if (response.ok) {
            // Update all segments with the old speaker name
            const oldName = segments[index].speaker;
            segments.forEach((seg, i) => {
                if (seg.speaker === oldName) {
                    seg.speaker = name;
                    const el = document.getElementById(`segment-${i}`);
                    if (el) {
                        el.querySelector('.speaker-name').textContent = name;
                        el.querySelectorAll('.speaker-pill').forEach(pill => {
                            pill.classList.toggle('active', pill.textContent === name);
                        });
                    }
                }
            });
        }
    } catch (err) {
        console.error('Update failed:', err);
    }
}


// ─── Segment Rendering ───

function createSegmentEl(seg, index) {
    const div = document.createElement('div');
    div.className = 'transcription-segment';
    div.id = `segment-${index}`;
    div.innerHTML = `
        <div class="segment-header">
            <span class="speaker-name">${seg.speaker}</span>
            <span class="timestamp" onclick="seekTo(${seg.start})">${seg.timestamp}</span>
        </div>
        <div class="segment-text">${seg.text}</div>
        <div class="segment-speaker-selector">
            ${getSpeakerPillsHTML(index, seg.speaker)}
        </div>
    `;
    return div;
}

function seekTo(time) {
    if (currentIsVideo) {
        videoPlayer.currentTime = time;
        videoPlayer.play();
        return;
    }
    wavesurfer.setTime(time);
    wavesurfer.play();
}

function highlightTranscription(currentTime) {
    let activeIndex = -1;
    segments.forEach((seg, index) => {
        const el = document.getElementById(`segment-${index}`);
        if (!el) return;

        const nextStart = segments[index + 1] ? segments[index + 1].start : Infinity;
        if (currentTime >= seg.start && currentTime < nextStart) {
            el.classList.add('active');
            activeIndex = index;
        } else {
            el.classList.remove('active');
        }
    });

    if (activeIndex !== -1) {
        const activeEl = document.getElementById(`segment-${activeIndex}`);
        const container = transcriptionContent;
        const target = activeEl.offsetTop - container.offsetTop - (container.clientHeight / 2) + (activeEl.clientHeight / 2);
        container.scrollTo({ top: target, behavior: 'smooth' });
    }
}


// ─── Controls ───

function skipPlayback(seconds) {
    if (currentIsVideo) {
        videoPlayer.currentTime = Math.max(0, Math.min((videoPlayer.duration || Infinity), videoPlayer.currentTime + seconds));
        return;
    }
    if (wavesurfer) wavesurfer.skip(seconds);
}

playPauseBtn.onclick = () => {
    if (currentIsVideo) {
        if (videoPlayer.paused) videoPlayer.play();
        else videoPlayer.pause();
        return;
    }
    if (wavesurfer) wavesurfer.playPause();
};
document.getElementById('skip-back-15').onclick = () => skipPlayback(-15);
document.getElementById('skip-back-5').onclick = () => skipPlayback(-5);
document.getElementById('skip-forward-5').onclick = () => skipPlayback(5);
document.getElementById('skip-forward-15').onclick = () => skipPlayback(15);

document.querySelectorAll('.speed-btn').forEach(btn => {
    btn.onclick = () => {
        const speed = parseFloat(btn.dataset.speed);
        if (currentIsVideo) {
            videoPlayer.playbackRate = speed;
            document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        } else if (wavesurfer) {
            wavesurfer.setPlaybackRate(speed);
            document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        }
    };
});

removeFileBtn.onclick = () => {
    launchScreen.classList.remove('hidden');
    mainInterface.classList.add('hidden');
    if (wavesurfer) wavesurfer.destroy();
    videoPlayer.pause();
    videoPlayer.removeAttribute('src');
    videoPlayer.load();
    currentTaskId = null;
    currentIsVideo = false;
    segments = [];
    footerActions.classList.add('hidden');
    ocrSummary.classList.add('hidden');
    ocrSummary.innerHTML = '';
    transcriptionContent.innerHTML = '<div class="placeholder-text">Your transcription will appear here...</div>';
};


// ═══════════════════════════════════════════
//  POD AUTOMATION LOGIC
// ═══════════════════════════════════════════

function addLog(text) {
    const line = document.createElement('div');
    line.className = 'log-line';
    line.textContent = `[${new Date().toLocaleTimeString()}] ${text}`;
    logConsole.appendChild(line);
    logConsole.scrollTop = logConsole.scrollHeight;
}

function getErrorMessage(data) {
    return data?.error || data?.detail || data?.message || 'Unknown error';
}

saveConfigBtn.onclick = async () => {
    const config = {
        ip: podIpInput.value.trim(),
        ssh_port: parseInt(podPortInput.value),
        pod_id: podIdInput.value.trim(),
        endpoint_id: endpointIdInput.value.trim(),
        key_path: podKeyInput.value.trim() || null
    };

    const resp = await fetch('/update-pod-config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
    });
    if (resp.ok) {
        addLog(`✅ Config updated. Endpoint: ${config.endpoint_id || 'Not set'}`);
        startPodPolling();
        if (config.endpoint_id) await loadEndpointWorkers(config.endpoint_id);
    }
};

saveWorkersBtn.onclick = async () => {
    const endpointId = endpointIdInput.value.trim();
    if (!endpointId) {
        addLog('Serverless Endpoint ID is required before saving max workers.');
        return;
    }

    const workersMax = parseInt(workersMaxInput.value, 10);
    if (Number.isNaN(workersMax) || workersMax < 0) {
        addLog('Max workers must be a whole number 0 or greater.');
        return;
    }

    addLog(`Saving RunPod max workers=${workersMax}...`);
    saveWorkersBtn.disabled = true;

    try {
        const resp = await fetch('/endpoint-workers', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                endpoint_id: endpointId,
                workers_max: workersMax
            })
        });
        const data = await resp.json();

        if (!resp.ok) {
            addLog(`RunPod update failed: ${getErrorMessage(data)}`);
            return;
        }

        workersMaxInput.value = data.workers_max ?? workersMax;
        addLog(`RunPod saved. Live workersMax=${data.workers_max}.`);
    } catch (e) {
        console.error('Failed to save endpoint workers', e);
        addLog('RunPod update failed: network error');
    } finally {
        saveWorkersBtn.disabled = false;
    }
};

async function loadConfig() {
    try {
        const resp = await fetch('/get-pod-config');
        const data = await resp.json();
        if (data.ip) podIpInput.value = data.ip;
        if (data.ssh_port) podPortInput.value = data.ssh_port;
        if (data.pod_id) podIdInput.value = data.pod_id;
        if (data.endpoint_id) endpointIdInput.value = data.endpoint_id;
        if (data.key_path) podKeyInput.value = data.key_path;

        if (data.endpoint_id || data.ip) {
            addLog("📁 Loaded existing configuration from server.");
        }
        if (data.endpoint_id) await loadEndpointWorkers(data.endpoint_id);
    } catch (e) {
        console.error("Failed to load config", e);
    }
}

async function loadEndpointWorkers(endpointId = endpointIdInput.value.trim()) {
    const resolvedEndpointId = endpointId.trim();
    if (!resolvedEndpointId) return;

    try {
        const resp = await fetch(`/endpoint-workers?endpoint_id=${encodeURIComponent(resolvedEndpointId)}`);
        const data = await resp.json();
        if (!resp.ok) {
            addLog(`RunPod workers load failed: ${getErrorMessage(data)}`);
            return;
        }

        workersMaxInput.value = data.workers_max ?? '';
        addLog(`Loaded RunPod max workers: ${data.workers_max}`);
    } catch (e) {
        console.error('Failed to load endpoint workers', e);
        addLog('RunPod workers load failed: network error');
    }
}

async function checkPodStatus() {
    try {
        const resp = await fetch('/pod-status');
        const data = await resp.json();

        podStatusBadge.textContent = data.status;
        podStatusBadge.className = `status-badge ${data.status.toLowerCase()}`;

        if (data.status === 'RUNNING') {
            startLogPolling();
        } else {
            stopLogPolling();
        }
    } catch (e) {
        podStatusBadge.textContent = 'ERROR';
        podStatusBadge.className = 'status-badge offline';
    }
}

function startPodPolling() {
    if (podPollingInterval) clearInterval(podPollingInterval);
    checkPodStatus();
    podPollingInterval = setInterval(checkPodStatus, 5000);
}

function startLogPolling() {
    if (logPollingInterval) return;
    logPollingInterval = setInterval(async () => {
        try {
            const resp = await fetch('/pod-logs');
            const data = await resp.json();
            if (data.logs) {
                const html = data.logs.split('\n').map(l => `<div class="log-line">${l}</div>`).join('');

                const consoleMain = document.getElementById('log-console');
                const consoleMini = document.getElementById('log-console-mini');

                if (consoleMain) {
                    consoleMain.innerHTML = html;
                    consoleMain.scrollTop = consoleMain.scrollHeight;
                }
                if (consoleMini) {
                    consoleMini.innerHTML = html;
                    consoleMini.scrollTop = consoleMini.scrollHeight;
                }
            }
        } catch (e) { console.error('Log fetch failed', e); }
    }, 3000);
}

function stopLogPolling() {
    if (logPollingInterval) {
        clearInterval(logPollingInterval);
        logPollingInterval = null;
    }
}

setupPodBtn.onclick = async () => {
    addLog('🚀 Starting Automated Setup...');
    setupPodBtn.disabled = true;
    const resp = await fetch('/setup-pod', { method: 'POST' });
    const data = await resp.json();
    addLog(data.message || data.status);
    setupPodBtn.disabled = false;
};

resumePodBtn.onclick = async () => {
    addLog('🟢 Sending Wake Command...');
    await fetch('/start-pod', { method: 'POST' });
};

startWorkerBtn.onclick = async () => {
    addLog('🎬 Starting Worker...');
    await fetch('/start-transcription', { method: 'POST' });
};

stopPodBtn.onclick = async () => {
    if (!confirm('Stop Pod? This will stop billing and turn off the GPU.')) return;
    addLog('⏹️ Stopping Pod...');
    await fetch('/stop-pod', { method: 'POST' });
};

refreshS3Btn.onclick = () => {
    loadS3Files();
};

// Start polling and load config on load
startPodPolling();
loadConfig();
loadS3Files();

window.seekTo = seekTo;
window.setSegmentSpeaker = setSegmentSpeaker;
