let wavesurfer;
let currentTaskId = null;
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

// Pod Control Elements
const setupPodBtn = document.getElementById('setup-pod-btn');
const resumePodBtn = document.getElementById('resume-pod-btn');
const startWorkerBtn = document.getElementById('start-worker-btn');
const stopPodBtn = document.getElementById('stop-pod-btn');
const podStatusBadge = document.getElementById('pod-status-badge');
const logConsole = document.getElementById('log-console');
const saveConfigBtn = document.getElementById('save-config-btn');
const podIpInput = document.getElementById('pod-ip-input');
const podPortInput = document.getElementById('pod-port-input');
const podIdInput = document.getElementById('pod-id-input');
const endpointIdInput = document.getElementById('endpoint-id-input');
const podKeyInput = document.getElementById('pod-key-input');

let podPollingInterval = null;
let logPollingInterval = null;

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

    try {
        const response = await fetch('/s3-files');
        const data = await response.json().catch(() => ({}));

        if (!response.ok || data.error) {
            throw new Error(data.error || 'Failed to load S3 files.');
        }

        renderS3Files(data.files || []);
        s3BrowserStatus.textContent = `${(data.files || []).length} file(s) currently on the server.`;
    } catch (err) {
        s3FileList.innerHTML = '<div class="s3-empty-state">Could not load bucket contents.</div>';
        s3BrowserStatus.textContent = `Error: ${err.message}`;
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

async function handleFile(file) {
    filenameDisplay.textContent = file.name;
    launchScreen.classList.add('hidden');
    mainInterface.classList.remove('hidden');
    knownSpeakers = [];
    renderSpeakerList();

    // Load audio player immediately
    const url = URL.createObjectURL(file);
    initWaveSurfer(url);

    // Check if a transcription already exists for this file
    try {
        const checkResponse = await fetch(`/check/${file.name}`);
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

    // No existing transcription — upload to server + S3
    startDiarizationBtn.classList.remove('hidden');
    startDiarizationBtn.disabled = true;
    startDiarizationBtn.textContent = 'Uploading...';
    startTranscriptionBtn.classList.add('hidden');
    startTranscriptionBtn.disabled = true;
    progressSection.classList.remove('hidden');
    statusText.textContent = '☁️ Uploading to cloud storage...';
    percentText.textContent = '';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        currentTaskId = data.task_id;

        // Check if upload returned an existing transcription
        if (data.task_id) {
            const statusResp = await fetch(`/status/${data.task_id}`);
            const statusData = await statusResp.json();
            if (statusData.status === 'completed') {
                loadCompletedTranscription(statusData);
                return;
            }
        }

        // Start polling for S3 upload progress
        startPolling();
    } catch (err) {
        statusText.textContent = 'Upload failed';
    }
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
        let url = `/process-cloud/${currentTaskId}?min_speakers=${minSpeakers}&max_speakers=${maxSpeakers}`;
        if (numSpeakers) url += `&num_speakers=${numSpeakers}`;
        const r = await fetch(url, { method: 'POST' });
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

startTranscriptionBtn.onclick = async () => {
    if (!currentTaskId) return;

    startTranscriptionBtn.disabled = true;
    startTranscriptionBtn.textContent = 'Transcribing...';
    progressSection.classList.remove('hidden');
    statusText.textContent = '🧠 Transcribing and aligning (GPU Processing)...';

    try {
        const r = await fetch(`/transcribe-cloud/${currentTaskId}`, { method: 'POST' });
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

function startPolling() {
    if (statusInterval) clearInterval(statusInterval);
    lastSegmentCount = 0;

    statusInterval = setInterval(async () => {
        if (!currentTaskId) return;

        try {
            const response = await fetch(`/status/${currentTaskId}`);
            const data = await response.json();

            updateUI(data);

            if (data.status === 'completed' || data.status === 'not_found' || data.status === 'error') {
                clearInterval(statusInterval);
            }
        } catch (err) {
            console.error('Status check failed:', err);
        }
    }, 2000);
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
        percentText.textContent = `${data.progress || 0}%`;
        progressBar.style.width = `${data.progress || 0}%`;
    }
    else if (data.status === 'diarizing') {
        progressSection.classList.remove('hidden');
        statusText.textContent = '🗣️ Identifying speakers...';
        percentText.textContent = `${data.progress || 0}%`;
        progressBar.style.width = `${data.progress || 0}%`;
    }
    else if (data.status === 'diarization_complete') {
        progressSection.classList.add('hidden');
        startDiarizationBtn.classList.add('hidden');
        startTranscriptionBtn.classList.remove('hidden');
        startTranscriptionBtn.disabled = false;
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

playPauseBtn.onclick = () => wavesurfer.playPause();
document.getElementById('skip-back-15').onclick = () => wavesurfer.skip(-15);
document.getElementById('skip-back-5').onclick = () => wavesurfer.skip(-5);
document.getElementById('skip-forward-5').onclick = () => wavesurfer.skip(5);
document.getElementById('skip-forward-15').onclick = () => wavesurfer.skip(15);

document.querySelectorAll('.speed-btn').forEach(btn => {
    btn.onclick = () => {
        const speed = parseFloat(btn.dataset.speed);
        if (wavesurfer) {
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
    currentTaskId = null;
    segments = [];
    footerActions.classList.add('hidden');
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
    } catch (e) {
        console.error("Failed to load config", e);
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
