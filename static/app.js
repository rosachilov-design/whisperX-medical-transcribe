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
const mdFilenameSpan = document.getElementById('md-filename-span');
const docxFilenameSpan = document.getElementById('docx-filename-span');
const removeFileBtn = document.getElementById('remove-file');
const currentTimeDisplay = document.getElementById('current-time');
const durationDisplay = document.getElementById('duration');
const startDiarizationBtn = document.getElementById('start-diarization-btn');
const startTranscriptionBtn = document.getElementById('start-transcription-btn');
const newSessionBtn = document.getElementById('new-session-btn');
const progressSection = document.getElementById('progress-section');

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
    if (e.relatedTarget === null) dropzone.classList.remove('dragging');
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
    startTranscriptionBtn.classList.remove('hidden');
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

    // Show download buttons
    if (data.md_path) mdFilenameSpan.textContent = data.md_path;
    if (data.docx_path) docxFilenameSpan.textContent = data.docx_path;
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

    startDiarizationBtn.disabled = true;
    startDiarizationBtn.textContent = 'Diarizing...';
    progressSection.classList.remove('hidden');
    statusText.textContent = '🗣️ Identifying speakers (GPU Processing)...';

    try {
        const r = await fetch(`/diarize-cloud/${currentTaskId}`, { method: 'POST' });
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
        startDiarizationBtn.disabled = false;
        startDiarizationBtn.textContent = '1. Start Diarization';
        statusText.textContent = '☁️ File uploaded to cloud. Ready to Diarize.';
        clearInterval(statusInterval);
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

document.getElementById('open-md-btn').onclick = () => {
    const md = mdFilenameSpan.textContent;
    if (md) window.open(`/download/${md}`, '_blank');
};

document.getElementById('open-docx-btn').onclick = () => {
    const docx = docxFilenameSpan.textContent;
    if (docx) window.open(`/download/${docx}`, '_blank');
};

removeFileBtn.onclick = () => {
    launchScreen.classList.remove('hidden');
    mainInterface.classList.add('hidden');
    if (wavesurfer) wavesurfer.destroy();
    currentTaskId = null;
    segments = [];
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

// Start polling and load config on load
startPodPolling();
loadConfig();

window.seekTo = seekTo;
window.setSegmentSpeaker = setSegmentSpeaker;
