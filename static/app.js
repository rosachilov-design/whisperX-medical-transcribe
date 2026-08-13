let currentTaskId = null;
let currentFile = null;
let statusInterval = null;
let segments = [];
let lastSegmentCount = 0;
let normalizationState = null;
let normalizationSelectedStep = 'source';
let normalizationPollInterval = null;
let normalizationAgentOpen = true;
let normalizationAgentLastMessageId = null;
let libraryPollInterval = null;
let libraryRefreshInFlight = false;
let currentS3Key = null;

// DOM Elements
const launchScreen = document.getElementById('launch-screen');
const dropzone = launchScreen;
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
const s3FileList = document.getElementById('s3-file-list');
const s3BrowserStatus = document.getElementById('s3-browser-status');
const refreshS3Btn = document.getElementById('refresh-s3-btn');
const deleteAllS3Btn = document.getElementById('delete-all-s3-btn');
const mediaPlayerContainer = document.getElementById('media-player-container');
const audioPlayer = document.getElementById('audio-player');
const audioTimeline = document.getElementById('audio-timeline');
const audioSeek = document.getElementById('audio-seek');
const videoPlayer = document.getElementById('video-player');
const videoParticipantsPanel = document.getElementById('video-participants-panel');
const videoParticipantsInput = document.getElementById('video-participants-input');
const ocrSummary = document.getElementById('ocr-summary');
const localTestBtn = document.getElementById('local-test-btn');
const localHunyuanTestBtn = document.getElementById('local-hunyuan-test-btn');
const filesTab = document.getElementById('files-tab');
const normalizationTab = document.getElementById('normalization-tab');
const conclusionsTab = document.getElementById('conclusions-tab');
const filesTabPanel = document.getElementById('files-tab-panel');
const normalizationTabPanel = document.getElementById('normalization-tab-panel');
const conclusionsTabPanel = document.getElementById('conclusions-tab-panel');
const normalizationActiveCount = document.getElementById('normalization-active-count');
const conclusionsActiveCount = document.getElementById('conclusions-active-count');
const conclusionsInstruction = document.getElementById('conclusions-instruction');
const conclusionsDropzone = document.getElementById('conclusions-dropzone');
const conclusionsFileInput = document.getElementById('conclusions-file-input');
const conclusionsComposeStatus = document.getElementById('conclusions-compose-status');
const conclusionsTaskList = document.getElementById('conclusions-task-list');
const refreshConclusionsBtn = document.getElementById('refresh-conclusions-btn');
const normalizationTaskList = document.getElementById('normalization-task-list');
const refreshNormalizationsBtn = document.getElementById('refresh-normalizations-btn');
const startReadyNormalizationsBtn = document.getElementById('start-ready-normalizations-btn');
const normalizationReadyTotal = document.getElementById('normalization-ready-total');
const normalizationRunningTotal = document.getElementById('normalization-running-total');
const normalizationReviewTotal = document.getElementById('normalization-review-total');

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
const runpodToggleBtn = document.getElementById('runpod-toggle-btn');
const runpodSidebar = document.getElementById('runpod-sidebar');
const runpodSidebarScrim = document.getElementById('runpod-sidebar-scrim');
const runpodCloseBtn = document.getElementById('runpod-close-btn');

let podPollingInterval = null;
let logPollingInterval = null;
let currentIsVideo = false;
let activeWorkspaceTab = 'files';
let conclusionsPollInterval = null;
let conclusionsRefreshInFlight = false;
let libraryFilesCache = [];
let conclusionsTasksCache = [];
const CONCLUSIONS_INSTRUCTION_STORAGE_KEY = 'transcriber.conclusionsInstruction';
const ACTIVE_TASK_STORAGE_KEY = 'transcriber.activeTask';
const FILE_URL_PARAM = 'file';
const TASK_URL_PARAM = 'task';
const VIEW_URL_PARAM = 'view';
const SUPPORTED_AUDIO_EXTENSIONS = ['.m4a', '.mp3', '.wav'];

function isSupportedAudioFilename(filename = '') {
    const lower = filename.toLowerCase();
    return SUPPORTED_AUDIO_EXTENSIONS.some(ext => lower.endsWith(ext));
}

function rejectUnsupportedFile(filename = '') {
    alert(`Unsupported file type: ${filename || 'unknown file'}\n\nUse M4A, MP3, or WAV audio files.`);
}

function persistActiveTask(taskId, filename = taskId, s3Key = null) {
    if (!taskId) return;
    localStorage.setItem(ACTIVE_TASK_STORAGE_KEY, JSON.stringify({
        taskId,
        filename,
        s3Key,
        savedAt: Date.now()
    }));
}

function getRequestedFileRoute() {
    const params = new URLSearchParams(window.location.search);
    const s3Key = params.get(FILE_URL_PARAM);
    const taskId = params.get(TASK_URL_PARAM);
    if (!s3Key && !taskId) return null;
    return {
        s3Key,
        taskId,
        mode: params.get(VIEW_URL_PARAM) === 'normalization' ? 'normalization' : 'transcript'
    };
}

function setFileUrl({ s3Key = null, taskId = null, mode = 'transcript' } = {}, { replace = false } = {}) {
    const url = new URL(window.location.href);
    url.searchParams.delete(FILE_URL_PARAM);
    url.searchParams.delete(TASK_URL_PARAM);
    url.searchParams.delete(VIEW_URL_PARAM);
    if (s3Key) url.searchParams.set(FILE_URL_PARAM, s3Key);
    if (taskId) url.searchParams.set(TASK_URL_PARAM, taskId);
    if ((s3Key || taskId) && mode === 'normalization') {
        url.searchParams.set(VIEW_URL_PARAM, 'normalization');
    }
    window.history[replace ? 'replaceState' : 'pushState']({}, '', url);
}

function getPersistedActiveTask() {
    try {
        return JSON.parse(localStorage.getItem(ACTIVE_TASK_STORAGE_KEY) || 'null');
    } catch {
        return null;
    }
}

function clearPersistedActiveTask() {
    localStorage.removeItem(ACTIVE_TASK_STORAGE_KEY);
}

function showTaskInterface(filename) {
    filenameDisplay.textContent = filename || 'Current transcription';
    launchScreen.classList.add('hidden');
    mainInterface.classList.remove('hidden');
    mainInterface.classList.remove('normalization-mode');
    document.getElementById('review-mode-switch').classList.add('hidden');
    normalizationState = null;
    normalizationSelectedStep = 'source';
    if (normalizationPollInterval) {
        clearInterval(normalizationPollInterval);
        normalizationPollInterval = null;
    }
    if (libraryPollInterval) {
        clearInterval(libraryPollInterval);
        libraryPollInterval = null;
    }
    setPlaybackMode(false);
}

async function restoreActiveTask() {
    const saved = getPersistedActiveTask();
    if (!saved?.taskId) return;

    if (saved.s3Key) {
        await openS3File({ key: saved.s3Key, task_id: saved.taskId, name: saved.filename });
        return;
    }

    if (!isSupportedAudioFilename(saved.filename || saved.taskId)) {
        clearPersistedActiveTask();
        return;
    }

    currentTaskId = saved.taskId;
    currentS3Key = null;
    setFileUrl({ taskId: currentTaskId }, { replace: true });
    showTaskInterface(saved.filename || saved.taskId);
    initAudioPlayer(`/audio/${encodeURIComponent(currentTaskId)}`);
    progressSection.classList.remove('hidden');
    statusText.textContent = 'Restoring task status...';
    percentText.textContent = '';
    transcriptionContent.innerHTML = '<div class="placeholder-text">Restoring current transcription...</div>';

    try {
        const response = await fetch(`/status/${encodeURIComponent(currentTaskId)}`);
        const data = await response.json();
        if (data.status === 'not_found') {
            clearPersistedActiveTask();
            statusText.textContent = 'Previous task was not found. Choose the source file again.';
            return;
        }

        updateUI(data);
        if (['uploading', 'processing', 'diarizing', 'transcribing'].includes(data.status)) {
            startPolling();
        }
    } catch (err) {
        statusText.textContent = `Could not restore task status: ${err.message}`;
    }
}

async function restoreTaskFromUrl() {
    const route = getRequestedFileRoute();
    if (!route) return false;

    if (route.s3Key) {
        await openS3File({ key: route.s3Key, task_id: route.taskId }, null, route.mode, { updateUrl: false });
        return true;
    }

    if (!isSupportedAudioFilename(route.taskId || '')) {
        s3BrowserStatus.textContent = 'The file URL is invalid or unsupported.';
        return true;
    }

    currentTaskId = route.taskId;
    currentS3Key = null;
    persistActiveTask(currentTaskId, currentTaskId);
    showTaskInterface(currentTaskId);
    initAudioPlayer(`/audio/${encodeURIComponent(currentTaskId)}`);
    progressSection.classList.remove('hidden');
    statusText.textContent = 'Opening file from URL...';
    transcriptionContent.innerHTML = '<div class="placeholder-text">Restoring current transcription...</div>';

    try {
        const response = await fetch(`/status/${encodeURIComponent(currentTaskId)}`);
        const data = await response.json();
        if (data.status === 'not_found') {
            clearPersistedActiveTask();
            statusText.textContent = 'This file was not found on the server.';
            return true;
        }
        updateUI(data);
        if (data.status === 'completed' && route.mode === 'normalization') {
            setReviewMode('normalization');
        } else if (['uploading', 'processing', 'diarizing', 'transcribing'].includes(data.status)) {
            startPolling();
        }
    } catch (err) {
        statusText.textContent = `Could not open file URL: ${err.message}`;
    }
    return true;
}

// ─── Main workspace tabs & parallel conclusions queue ───

function setWorkspaceTab(tab) {
    activeWorkspaceTab = ['normalization', 'conclusions'].includes(tab) ? tab : 'files';
    const normalizationActive = activeWorkspaceTab === 'normalization';
    const conclusionsActive = activeWorkspaceTab === 'conclusions';
    const filesActive = activeWorkspaceTab === 'files';
    filesTab.classList.toggle('active', filesActive);
    normalizationTab.classList.toggle('active', normalizationActive);
    conclusionsTab.classList.toggle('active', conclusionsActive);
    filesTab.setAttribute('aria-selected', String(filesActive));
    normalizationTab.setAttribute('aria-selected', String(normalizationActive));
    conclusionsTab.setAttribute('aria-selected', String(conclusionsActive));
    filesTabPanel.classList.toggle('hidden', !filesActive);
    normalizationTabPanel.classList.toggle('hidden', !normalizationActive);
    conclusionsTabPanel.classList.toggle('hidden', !conclusionsActive);
    if (conclusionsActive) loadConclusionsTasks();
    if (normalizationActive) loadS3Files({ silent: true });
}

function documentIdentity(filename = '') {
    return String(filename)
        .toLocaleLowerCase('ru-RU')
        .replace(/\.(m4a|mp3|wav|json|md|txt|docx)$/i, '')
        .replace(/(?:^|[\s_-]+)(normalized|normalised|operator|operator-version|transcript|final|conclusions|report|нормализованн\w*|операторск\w*|транскрипт|выводы|отчет|отчёт)(?=$|[\s_-]+)/giu, ' ')
        .replace(/[^\p{L}\p{N}]+/gu, ' ')
        .trim();
}

function identitiesMatch(first, second) {
    if (!first || !second) return false;
    if (first === second) return true;
    return first.startsWith(`${second} `) || second.startsWith(`${first} `);
}

function conclusionStatusForFile(file) {
    const identity = documentIdentity(file.name || file.task_id || '');
    const matches = conclusionsTasksCache.filter(task => {
        const taskIdentity = documentIdentity(task.filename);
        return identitiesMatch(identity, taskIdentity);
    });
    const task = matches.sort((a, b) => (b.created_at || 0) - (a.created_at || 0))[0];
    if (!task) return { state: 'not_started', label: 'Не загружено', message: 'Операторская версия ещё не добавлена.', progress: 0 };
    if (task.status === 'completed') return { state: 'completed', label: 'Готовы', message: 'TXT и DOCX доступны во вкладке «Выводы».', progress: 100 };
    if (task.status === 'failed') return { state: 'failed', label: 'Ошибка', message: task.error || task.message, progress: 100 };
    return { state: 'running', label: task.status === 'queued' ? 'В очереди' : 'Sol High', message: task.message, progress: task.progress || 0 };
}

function sourceFileForConclusionTask(task) {
    const identity = documentIdentity(task.filename);
    return libraryFilesCache.find(file => {
        const fileIdentity = documentIdentity(file.name || file.task_id || '');
        return identitiesMatch(identity, fileIdentity);
    });
}

function formatConclusionTime(seconds) {
    if (!seconds) return '';
    return new Date(seconds * 1000).toLocaleString();
}

function updateConclusionsPolling(tasks) {
    const activeCount = tasks.filter(task => ['queued', 'running'].includes(task.status)).length;
    conclusionsActiveCount.textContent = String(activeCount);
    conclusionsActiveCount.classList.toggle('hidden', activeCount === 0);
    if (activeCount && !conclusionsPollInterval) {
        conclusionsPollInterval = window.setInterval(() => loadConclusionsTasks({ silent: true }), 2000);
    } else if (!activeCount && conclusionsPollInterval) {
        clearInterval(conclusionsPollInterval);
        conclusionsPollInterval = null;
    }
}

function renderConclusionsTasks(tasks) {
    if (!tasks.length) {
        conclusionsTaskList.innerHTML = '<div class="s3-empty-state">Добавьте операторские версии — они появятся здесь без открытия нового окна.</div>';
        updateConclusionsPolling([]);
        return;
    }
    conclusionsTaskList.innerHTML = tasks.map(task => {
        const active = ['queued', 'running'].includes(task.status);
        const failed = task.status === 'failed';
        const statusLabel = task.status === 'completed' ? 'Готово' : failed ? 'Ошибка' : task.status === 'queued' ? 'В очереди' : 'Sol High работает';
        const details = failed ? (task.error || task.message) : task.message;
        const sourceFile = sourceFileForConclusionTask(task);
        const sourceNormalization = sourceFile ? libraryNormalizationStatus(sourceFile) : null;
        const sourceStatus = sourceNormalization
            ? `<span class="conclusions-source-status ${escapeHtml(sourceNormalization.state)}">Нормализация · ${escapeHtml(sourceNormalization.state === 'completed' ? 'готова' : sourceNormalization.label)}</span>`
            : '<span class="conclusions-source-status unknown">Исходный файл не сопоставлен</span>';
        const actions = task.status === 'completed'
            ? `<span class="conclusions-row-actions"><a href="/conclusions/${encodeURIComponent(task.id)}/download?format=txt" download>TXT</a><a href="/conclusions/${encodeURIComponent(task.id)}/download?format=docx" download>DOCX</a></span>`
            : '';
        return `<article class="conclusions-task-row ${escapeHtml(task.status)}">
            <div class="conclusions-task-main">
                <span class="conclusions-task-state" aria-hidden="true">${active ? '<i></i>' : failed ? '!' : '✓'}</span>
                <span class="conclusions-task-copy"><strong>${escapeHtml(task.filename)}</strong><small>${escapeHtml(details || '')}</small>${sourceStatus}</span>
            </div>
            <div class="conclusions-task-status"><strong>${statusLabel}</strong><small>${escapeHtml(formatConclusionTime(task.finished_at || task.created_at))}</small></div>
            ${actions}
            <div class="conclusions-task-progress" aria-label="${escapeHtml(statusLabel)}"><i style="--task-progress:${Number(task.progress) || 0}%"></i></div>
        </article>`;
    }).join('');
    updateConclusionsPolling(tasks);
}

async function loadConclusionsTasks({ silent = false } = {}) {
    if (conclusionsRefreshInFlight) return;
    conclusionsRefreshInFlight = true;
    if (!silent) refreshConclusionsBtn.disabled = true;
    try {
        const response = await fetch('/conclusions');
        const data = await response.json().catch(() => ({}));
        if (!response.ok || data.error) throw new Error(data.error || 'Не удалось загрузить очередь.');
        conclusionsTasksCache = data.tasks || [];
        renderConclusionsTasks(conclusionsTasksCache);
        if (libraryFilesCache.length) renderS3Files(libraryFilesCache);
    } catch (error) {
        if (!silent) conclusionsTaskList.innerHTML = `<div class="s3-empty-state">${escapeHtml(error.message)}</div>`;
    } finally {
        conclusionsRefreshInFlight = false;
        if (!silent) refreshConclusionsBtn.disabled = false;
    }
}

async function uploadConclusionFile(file, instruction) {
    const body = new FormData();
    body.append('file', file);
    body.append('instruction', instruction);
    const response = await fetch('/conclusions', { method: 'POST', body });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data.error) throw new Error(`${file.name}: ${data.error || 'ошибка загрузки'}`);
    return data;
}

async function enqueueConclusionFiles(files) {
    const selected = [...files];
    if (!selected.length) return;
    const instruction = conclusionsInstruction.value.trim();
    if (!instruction) {
        conclusionsComposeStatus.textContent = 'Сначала добавьте инструкцию для выводов.';
        conclusionsInstruction.focus();
        return;
    }
    localStorage.setItem(CONCLUSIONS_INSTRUCTION_STORAGE_KEY, instruction);
    conclusionsComposeStatus.textContent = `Добавляю ${selected.length} файл(ов) в параллельную очередь…`;
    conclusionsDropzone.classList.add('uploading');
    const results = await Promise.allSettled(selected.map(file => uploadConclusionFile(file, instruction)));
    const failures = results.filter(result => result.status === 'rejected');
    conclusionsComposeStatus.textContent = failures.length
        ? `Запущено: ${selected.length - failures.length}. Ошибки: ${failures.map(item => item.reason.message).join('; ')}`
        : `${selected.length} файл(ов) запущены одновременно.`;
    conclusionsDropzone.classList.remove('uploading');
    conclusionsFileInput.value = '';
    await loadConclusionsTasks();
}

async function loadDefaultConclusionsInstruction() {
    const saved = localStorage.getItem(CONCLUSIONS_INSTRUCTION_STORAGE_KEY);
    if (saved) {
        conclusionsInstruction.value = saved;
        return;
    }
    try {
        const response = await fetch('/report-instruction.md');
        if (response.ok) conclusionsInstruction.value = await response.text();
    } catch (_error) {
        conclusionsInstruction.placeholder = 'Вставьте инструкцию для подготовки выводов';
    }
}

function normalizationActionForFile(file) {
    const status = libraryNormalizationStatus(file);
    const stepId = file.normalization?.current_step?.id || 'source';
    if (!status.canNormalize) return { kind: 'disabled', label: 'Ждёт транскрипт', stepId };
    if (status.state === 'not_started') return { kind: 'start', label: 'Запустить', stepId: 'source' };
    if (status.state === 'ready') return { kind: 'start', label: 'Продолжить', stepId };
    if (status.state === 'running') return { kind: 'disabled', label: 'Выполняется', stepId };
    return { kind: 'open', label: status.state === 'completed' ? 'Открыть результат' : 'Открыть процесс', stepId };
}

function normalizationAssumptionsMarkup(normalization, className = '') {
    const assumptions = normalization?.assumptions || [];
    if (!assumptions.length) return '';
    const rows = assumptions.map(item => `<li><div><span class="confidence ${escapeHtml(item.confidence || 'mid')}">${escapeHtml(item.confidence || 'mid')}</span><code>${escapeHtml(item.item_id || item.step_id || '')}</code><strong>${escapeHtml(item.decision || '')}</strong></div><p>${escapeHtml(item.basis || '')}</p><small>${escapeHtml(item.owner || 'Sol xhigh')} · ${escapeHtml(item.step_title || item.step_id || '')}</small></li>`).join('');
    return `<details class="normalization-assumptions ${escapeHtml(className)}"><summary>${assumptions.length} ${assumptions.length === 1 ? 'допущение' : 'допущений'} Sol xhigh для оператора</summary><ol>${rows}</ol></details>`;
}

function renderNormalizationQueue(files) {
    const visibleFiles = files.filter(file => isSupportedAudioFilename(file.name || file.key || ''));
    const ready = visibleFiles.filter(file => normalizationActionForFile(file).kind === 'start');
    const running = visibleFiles.filter(file => file.normalization?.state === 'running');
    const assumptionTotal = visibleFiles.reduce((total, file) => total + Number(file.normalization?.assumption_count || 0), 0);
    normalizationReadyTotal.textContent = String(ready.length);
    normalizationRunningTotal.textContent = String(running.length);
    normalizationReviewTotal.textContent = String(assumptionTotal);
    normalizationActiveCount.textContent = String(running.length);
    normalizationActiveCount.classList.toggle('hidden', running.length === 0);
    startReadyNormalizationsBtn.disabled = ready.length === 0;
    startReadyNormalizationsBtn.textContent = ready.length ? `Запустить готовые · ${ready.length}` : 'Нет готовых к запуску';

    if (!visibleFiles.length) {
        normalizationTaskList.innerHTML = '<div class="s3-empty-state">Сначала добавьте аудио и завершите транскрибацию.</div>';
        return;
    }

    normalizationTaskList.innerHTML = visibleFiles.map(file => {
        const status = libraryNormalizationStatus(file);
        const action = normalizationActionForFile(file);
        const step = file.normalization?.blocked_step || file.normalization?.current_step;
        const stageLabel = step ? `Этап ${step.number}/${step.total} · ${step.title}` : status.label;
        const conclusion = conclusionStatusForFile(file);
        const completed = status.state === 'completed';
        return `<article class="normalization-queue-row ${escapeHtml(status.state)}" data-task-id="${escapeHtml(file.task_id || '')}" data-key="${escapeHtml(file.key || '')}">
            <div class="normalization-queue-file">
                <span class="pipeline-state-icon ${escapeHtml(status.state)}" aria-hidden="true">${status.state === 'running' ? '<i></i>' : completed ? '✓' : status.state === 'blocked' ? '!' : '02'}</span>
                <span><strong>${escapeHtml(file.name || file.task_id || file.key)}</strong><small>${escapeHtml(formatBytes(file.size))} · ${escapeHtml(formatS3Date(file.last_modified))}</small></span>
            </div>
            <div class="normalization-queue-stage">
                <span><strong>${escapeHtml(stageLabel)}</strong><small>${escapeHtml(status.message)}</small></span>
                <b>${escapeHtml(String(status.progress))}%</b>
            </div>
            <div class="normalization-queue-followup ${escapeHtml(conclusion.state)}" title="${escapeHtml(conclusion.message)}"><i></i><span>Выводы: ${escapeHtml(conclusion.label)}</span></div>
            <div class="normalization-queue-actions">
                ${completed ? `<a href="/normalization/${encodeURIComponent(file.task_id)}/download" download>Скачать MD</a>` : ''}
                <button type="button" data-action="${escapeHtml(action.kind)}" data-step-id="${escapeHtml(action.stepId)}" ${action.kind === 'disabled' ? 'disabled' : ''}>${escapeHtml(action.label)}</button>
            </div>
            <div class="normalization-queue-progress"><i style="--normalization-progress:${Number(status.progress) || 0}%"></i></div>
            ${normalizationAssumptionsMarkup(file.normalization, 'queue-assumptions')}
        </article>`;
    }).join('');

    normalizationTaskList.querySelectorAll('button[data-action]').forEach(button => {
        button.addEventListener('click', async () => {
            const row = button.closest('.normalization-queue-row');
            const file = libraryFilesCache.find(item => item.key === row.dataset.key);
            if (!file) return;
            if (button.dataset.action === 'open') {
                openS3File(file, null, 'normalization');
                return;
            }
            if (button.dataset.action === 'start') await startNormalizationForFile(file, button.dataset.stepId, button);
        });
    });
}

async function startNormalizationForFile(file, stepId = 'source', button = null) {
    if (!file?.task_id) return { ok: false, error: 'Не найден идентификатор транскрипта.' };
    if (button) {
        button.disabled = true;
        button.textContent = 'Запускаю…';
    }
    try {
        const response = await fetch(`/normalization/${encodeURIComponent(file.task_id)}/steps/${encodeURIComponent(stepId)}/run`, { method: 'POST' });
        const data = await response.json().catch(() => ({}));
        if (!response.ok || data.error) throw new Error(data.error || 'Не удалось запустить нормализацию.');
        return { ok: true };
    } catch (error) {
        if (button) {
            button.disabled = false;
            button.textContent = 'Повторить';
            button.title = error.message;
        }
        return { ok: false, error: error.message };
    }
}

async function startReadyNormalizations() {
    const ready = libraryFilesCache
        .map(file => ({ file, action: normalizationActionForFile(file) }))
        .filter(item => item.action.kind === 'start');
    if (!ready.length) return;
    startReadyNormalizationsBtn.disabled = true;
    startReadyNormalizationsBtn.textContent = `Запускаю ${ready.length}…`;
    await Promise.allSettled(ready.map(item => startNormalizationForFile(item.file, item.action.stepId)));
    await loadS3Files({ silent: true });
}

filesTab.addEventListener('click', () => setWorkspaceTab('files'));
normalizationTab.addEventListener('click', () => setWorkspaceTab('normalization'));
conclusionsTab.addEventListener('click', () => setWorkspaceTab('conclusions'));
refreshConclusionsBtn.addEventListener('click', () => loadConclusionsTasks());
refreshNormalizationsBtn.addEventListener('click', () => loadS3Files());
startReadyNormalizationsBtn.addEventListener('click', startReadyNormalizations);
conclusionsDropzone.addEventListener('click', () => conclusionsFileInput.click());
conclusionsDropzone.addEventListener('keydown', event => {
    if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        conclusionsFileInput.click();
    }
});
conclusionsFileInput.addEventListener('change', () => enqueueConclusionFiles(conclusionsFileInput.files));
conclusionsDropzone.addEventListener('dragover', event => {
    event.preventDefault();
    event.stopPropagation();
    conclusionsDropzone.classList.add('dragging');
});
conclusionsDropzone.addEventListener('dragleave', () => conclusionsDropzone.classList.remove('dragging'));
conclusionsDropzone.addEventListener('drop', event => {
    event.preventDefault();
    event.stopPropagation();
    conclusionsDropzone.classList.remove('dragging');
    enqueueConclusionFiles(event.dataTransfer.files);
});

// ─── Dropzone & File Input ───

// Drag events on the entire launch screen
document.addEventListener('dragover', (e) => {
    e.preventDefault();
    if (activeWorkspaceTab !== 'files') return;
    dropzone.classList.add('dragging');
});

document.addEventListener('dragleave', (e) => {
    if (e.relatedTarget === null) {
        dropzone.classList.remove('dragging');
    }
});

document.addEventListener('drop', (e) => {
    e.preventDefault();
    if (activeWorkspaceTab !== 'files') return;
    dropzone.classList.remove('dragging');
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFile(files[0]);
});

fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
});

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

function libraryNormalizationStatus(file) {
    if (!file.normalization) {
        return {
            state: 'unknown',
            progress: 0,
            canNormalize: Boolean(file.has_transcript),
            label: 'Статус недоступен',
            message: 'Backend не передал состояние нормализации; обновите сервер.'
        };
    }
    const normalization = file.normalization || {};
    const step = normalization.blocked_step || normalization.current_step;
    const stepLabel = step ? `Шаг ${step.number}/${step.total} · ${step.title}` : '';
    const variants = {
        unavailable: { label: 'Недоступно', message: normalization.message || 'Сначала нужна транскрибация.' },
        not_started: { label: 'Не запускалась', message: normalization.message || 'Нормализацию можно начать.' },
        ready: { label: stepLabel || 'Готова к продолжению', message: normalization.message || 'Можно продолжить.' },
        running: { label: stepLabel || 'Выполняется', message: normalization.message || 'Этап выполняется в фоне.' },
        blocked: { label: `Sol xhigh решает: ${stepLabel || 'спорный случай'}`, message: normalization.message || 'Процесс продолжится автоматически.' },
        completed: { label: 'Нормализация завершена', message: normalization.message || 'Финальный MD на сервере.' }
    };
    return { state: normalization.state || 'not_started', progress: normalization.overall_progress || 0, canNormalize: normalization.can_normalize !== false, ...(variants[normalization.state] || variants.not_started) };
}

function libraryTranscriptionStatus(file) {
    const transcription = file.transcription || {};
    return {
        state: transcription.state || (file.has_transcript ? 'completed' : 'ready'),
        label: transcription.label || (file.has_transcript ? 'Транскрипт готов' : 'Готов к запуску'),
        message: transcription.message || (file.has_transcript ? 'Доступен для просмотра.' : 'Транскрибация ещё не запускалась.'),
        progress: Number(transcription.progress || 0),
        active: Boolean(transcription.active)
    };
}

function updateLibraryPolling(files) {
    const hasActiveWork = files.some(file => file.transcription?.active || file.normalization?.state === 'running');
    if (hasActiveWork && !libraryPollInterval && !launchScreen.classList.contains('hidden')) {
        libraryPollInterval = window.setInterval(() => loadS3Files({ silent: true }), 4000);
    } else if (!hasActiveWork && libraryPollInterval) {
        clearInterval(libraryPollInterval);
        libraryPollInterval = null;
    }
}

function pipelineStageMarkup(index, title, status) {
    const active = status.state === 'running';
    const symbol = active ? '<i></i>' : status.state === 'completed' ? '✓' : status.state === 'failed' || status.state === 'blocked' ? '!' : String(index).padStart(2, '0');
    const conciseLabels = {
        completed: 'Готово',
        failed: 'Ошибка',
        blocked: 'Sol xhigh решает',
        unavailable: 'Недоступно',
        not_started: 'Не запущено'
    };
    let label = conciseLabels[status.state] || status.label;
    if (title === 'Транскрипция' && status.state === 'ready') label = 'Не запущена';
    if (title === 'Нормализация' && status.state === 'ready') label = 'Можно продолжить';
    return `<span class="pipeline-stage ${escapeHtml(status.state)}" title="${escapeHtml(status.message || '')}">
        <span class="pipeline-stage-marker" aria-hidden="true">${symbol}</span>
        <span class="pipeline-stage-copy"><small>${escapeHtml(title)}</small><strong>${escapeHtml(label)}</strong></span>
    </span>`;
}

function renderS3Files(files) {
    const visibleFiles = files.filter(file => isSupportedAudioFilename(file.name || file.key || ''));
    if (!visibleFiles.length) {
        s3FileList.innerHTML = '<div class="s3-empty-state">Файлов пока нет. Добавьте аудио выше.</div>';
        renderNormalizationQueue([]);
        updateLibraryPolling([]);
        return;
    }

    s3FileList.innerHTML = '';
    visibleFiles.forEach((file) => {
        const row = document.createElement('div');
        row.className = 's3-file-row';
        const displayName = file.name || file.task_id || file.key || 'Unnamed file';
        const transcriptionStatus = libraryTranscriptionStatus(file);
        const normalizationStatus = libraryNormalizationStatus(file);
        const conclusionStatus = conclusionStatusForFile(file);
        const normalizationAction = normalizationActionForFile(file);
        row.classList.add(`normalization-${normalizationStatus.state}`);
        row.innerHTML = [
            '<button class="s3-file-open" type="button" aria-label="Открыть ' + escapeHtml(displayName) + '">',
            '<span class="s3-file-meta">',
            '<span class="s3-file-name">' + escapeHtml(displayName) + '</span>',
            '<span class="s3-file-info">' + escapeHtml(formatBytes(file.size)) + ' · ' + escapeHtml(formatS3Date(file.last_modified)) + '</span>',
            '</span>',
            '<span class="pipeline-rail">',
            pipelineStageMarkup(1, 'Транскрипция', transcriptionStatus),
            pipelineStageMarkup(2, 'Нормализация', normalizationStatus),
            pipelineStageMarkup(3, 'Выводы', conclusionStatus),
            '</span>',
            '</button>',
            '<span class="s3-row-actions">',
            '<button class="s3-open-btn" type="button">Открыть</button>',
            '<button class="s3-normalize-btn" type="button"' + (normalizationStatus.canNormalize ? '' : ' disabled title="Сначала завершите транскрибацию"') + '>' + escapeHtml(normalizationAction.label) + '</button>',
            '<button class="s3-delete-btn" type="button" aria-label="Удалить ' + escapeHtml(displayName) + '">×</button>',
            '</span>',
            normalizationAssumptionsMarkup(file.normalization, 'file-assumptions')
        ].join('');

        row.querySelector('.s3-file-open').addEventListener('click', () => {
            openS3File(file, row, 'transcript');
        });
        row.querySelector('.s3-open-btn').addEventListener('click', () => openS3File(file, row, 'transcript'));
        row.querySelector('.s3-normalize-btn').addEventListener('click', (event) => {
            event.stopPropagation();
            if (normalizationAction.kind === 'start') {
                setWorkspaceTab('normalization');
                startNormalizationForFile(file, normalizationAction.stepId).then(() => loadS3Files({ silent: true }));
            } else {
                openS3File(file, row, 'normalization');
            }
        });
        row.querySelector('.s3-delete-btn').addEventListener('click', (event) => {
            event.stopPropagation();
            deleteS3File(file.key, file.name);
        });
        s3FileList.appendChild(row);
    });
    renderNormalizationQueue(visibleFiles);
    updateLibraryPolling(visibleFiles);
}

async function openS3File(file, row = null, targetMode = 'transcript', { updateUrl = true } = {}) {
    const displayName = file.name || file.key || 'Unnamed file';
    const controls = row?.querySelectorAll('button') || [];
    controls.forEach(button => { button.disabled = true; });
    row?.classList.add('opening');
    s3BrowserStatus.textContent = `Opening ${displayName}...`;

    try {
        const response = await fetch('/open-s3-file', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                key: file.key,
                task_id: file.task_id || file.name || null
            })
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok || data.error || !data.task_id) {
            throw new Error(data.error || 'Could not open this file.');
        }

        currentFile = null;
        currentTaskId = data.task_id;
        currentS3Key = file.key;
        persistActiveTask(currentTaskId, data.filename || displayName, currentS3Key);
        if (updateUrl) setFileUrl({ s3Key: currentS3Key, taskId: currentTaskId, mode: targetMode });
        knownSpeakers = [];
        renderSpeakerList();
        showTaskInterface(data.filename || displayName);
        initAudioPlayer(`/s3-audio?key=${encodeURIComponent(file.key)}`);

        const checkResponse = await fetch(`/check/${encodeURIComponent(currentTaskId)}`);
        const checkData = await checkResponse.json().catch(() => ({}));
        if (checkResponse.ok && checkData.status === 'completed' && checkData.result) {
            loadCompletedTranscription(checkData);
            if (targetMode === 'normalization') {
                normalizationSelectedStep = file.normalization?.blocked_step?.id || file.normalization?.current_step?.id || 'source';
                setReviewMode('normalization');
            }
            return;
        }

        updateUI(data.task || { status: 'uploaded', progress: 100 });
    } catch (err) {
        s3BrowserStatus.textContent = `Error: ${err.message}`;
        controls.forEach(button => { button.disabled = false; });
        row?.classList.remove('opening');
    }
}

async function loadS3Files({ silent = false } = {}) {
    if (libraryRefreshInFlight) return;
    libraryRefreshInFlight = true;
    if (!silent) {
        s3BrowserStatus.textContent = 'Loading files from S3...';
        refreshS3Btn.disabled = true;
        deleteAllS3Btn.disabled = true;
    }

    try {
        const response = await fetch('/s3-files');
        const data = await response.json().catch(() => ({}));

        if (!response.ok || data.error) {
            throw new Error(data.error || 'Failed to load S3 files.');
        }

        const files = data.files || [];
        libraryFilesCache = files;
        const visibleFiles = files.filter(file => isSupportedAudioFilename(file.name || file.key || ''));
        renderS3Files(files);
        if (conclusionsTasksCache.length) renderConclusionsTasks(conclusionsTasksCache);
        if (!silent) s3BrowserStatus.textContent = `${visibleFiles.length} ${visibleFiles.length === 1 ? 'файл' : 'файлов'} · статусы синхронизированы`;
        deleteAllS3Btn.disabled = !visibleFiles.length;
    } catch (err) {
        if (!silent) {
            s3FileList.innerHTML = '<div class="s3-empty-state">Could not load bucket contents.</div>';
            s3BrowserStatus.textContent = `Error: ${err.message}`;
        }
    } finally {
        libraryRefreshInFlight = false;
        if (!silent) refreshS3Btn.disabled = false;
    }
}

async function deleteAllS3Files() {
    const warning = [
        'Удалить все файлы с сервера?',
        '',
        'Будут удалены аудио, JSON и остальные объекты из хранилища RunPod.',
        'Файлы в локальной папке uploads останутся без изменений.',
        'Это действие нельзя отменить.'
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

        s3BrowserStatus.textContent = `Удалено из RunPod S3: ${data.deleted_remote || 0}. Локальные файлы сохранены.`;
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

function updateAudioTimeline(time = audioPlayer.currentTime || 0) {
    const duration = Number.isFinite(audioPlayer.duration) ? audioPlayer.duration : 0;
    const current = Math.max(0, Math.min(duration || 0, Number(time) || 0));
    audioSeek.max = String(duration || 0);
    audioSeek.value = String(current);
    audioSeek.disabled = duration <= 0;
    audioSeek.style.setProperty('--seek-progress', `${duration ? (current / duration) * 100 : 0}%`);
    currentTimeDisplay.textContent = formatTime(current);
    durationDisplay.textContent = formatTime(duration);
}

function resetAudioPlayer() {
    audioPlayer.onloadedmetadata = null;
    audioPlayer.ontimeupdate = null;
    audioPlayer.onplay = null;
    audioPlayer.onpause = null;
    audioPlayer.onended = null;
    audioPlayer.onerror = null;
    audioPlayer.pause();
    audioPlayer.removeAttribute('src');
    audioPlayer.load();
    audioSeek.value = '0';
    audioSeek.max = '0';
    audioSeek.disabled = true;
    audioSeek.style.setProperty('--seek-progress', '0%');
    currentTimeDisplay.textContent = '0:00';
    durationDisplay.textContent = '0:00';
    playIcon.className = 'play-icon';
}

function initAudioPlayer(url) {
    resetAudioPlayer();
    videoPlayer.pause();
    videoPlayer.removeAttribute('src');
    videoPlayer.load();

    audioPlayer.src = url;
    audioPlayer.load();
    audioPlayer.onloadedmetadata = () => updateAudioTimeline(0);
    audioPlayer.ontimeupdate = () => {
        updateAudioTimeline();
        highlightTranscription(audioPlayer.currentTime || 0);
    };
    audioPlayer.onplay = () => { playIcon.className = 'pause-icon'; };
    audioPlayer.onpause = () => { playIcon.className = 'play-icon'; };
    audioPlayer.onended = () => { playIcon.className = 'play-icon'; };
    audioPlayer.onerror = () => {
        audioSeek.disabled = true;
        durationDisplay.textContent = '—:—';
    };
}

audioSeek.addEventListener('input', () => {
    if (audioSeek.disabled || !audioPlayer.getAttribute('src')) return;
    const time = Number(audioSeek.value) || 0;
    audioPlayer.currentTime = time;
    updateAudioTimeline(time);
    highlightTranscription(time);
});

function initVideoPlayer(url) {
    resetAudioPlayer();

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
    audioTimeline.classList.toggle('hidden', isVideo);
    videoPlayer.classList.toggle('hidden', !isVideo);
    mediaPlayerContainer.classList.toggle('video-mode', isVideo);
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
    const total = Math.max(0, Math.floor(Number(seconds) || 0));
    const hours = Math.floor(total / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    const secs = total % 60;
    return hours
        ? `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
        : `${minutes}:${secs.toString().padStart(2, '0')}`;
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
        persistActiveTask(currentTaskId, currentFile?.name || data.task_id);
        currentS3Key = null;
        setFileUrl({ taskId: currentTaskId });

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
    if (!isSupportedAudioFilename(file.name)) {
        rejectUnsupportedFile(file.name);
        return;
    }

    currentFile = file;
    filenameDisplay.textContent = file.name;
    launchScreen.classList.add('hidden');
    mainInterface.classList.remove('hidden');
    knownSpeakers = [];
    renderSpeakerList();
    setPlaybackMode(false);

    // Load local preview immediately
    const url = URL.createObjectURL(file);
    if (currentIsVideo) {
        initVideoPlayer(url);
    } else {
        initAudioPlayer(url);
    }

    // Check if a transcription already exists for this file
    try {
        const checkResponse = await fetch(`/check/${encodeURIComponent(file.name)}`);
        const checkData = await checkResponse.json();

        if (checkData.status === 'completed' && checkData.result) {
            // Transcription exists! Load it directly
            currentTaskId = file.name;
            persistActiveTask(currentTaskId, file.name);
            currentS3Key = null;
            setFileUrl({ taskId: currentTaskId });
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
    statusText.textContent = data.recovery_warning
        ? `⚠️ ${data.recovery_warning}`
        : '✅ Transcription loaded';

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
    document.getElementById('review-mode-switch').classList.remove('hidden');
    initializeNormalizationWorkflow();
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
        document.getElementById('review-mode-switch').classList.add('hidden');
        mainInterface.classList.remove('normalization-mode');
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
    clearPersistedActiveTask();
    setFileUrl({}, { replace: true });
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
                if (data.status === 'not_found') clearPersistedActiveTask();
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
    if (!audioPlayer.getAttribute('src')) return;
    audioPlayer.currentTime = Math.max(0, Math.min(audioPlayer.duration || Infinity, time));
    audioPlayer.play().catch(() => {});
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
    if (audioPlayer.getAttribute('src')) {
        audioPlayer.currentTime = Math.max(0, Math.min(audioPlayer.duration || Infinity, audioPlayer.currentTime + seconds));
    }
}

playPauseBtn.onclick = () => {
    if (currentIsVideo) {
        if (videoPlayer.paused) videoPlayer.play();
        else videoPlayer.pause();
        return;
    }
    if (!audioPlayer.getAttribute('src')) return;
    if (audioPlayer.paused) audioPlayer.play().catch(() => {});
    else audioPlayer.pause();
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
        } else if (audioPlayer.getAttribute('src')) {
            audioPlayer.playbackRate = speed;
            document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        }
    };
});

removeFileBtn.onclick = () => {
    clearPersistedActiveTask();
    setFileUrl({});
    if (statusInterval) {
        clearInterval(statusInterval);
        statusInterval = null;
    }
    launchScreen.classList.remove('hidden');
    mainInterface.classList.add('hidden');
    resetAudioPlayer();
    videoPlayer.pause();
    videoPlayer.removeAttribute('src');
    videoPlayer.load();
    currentTaskId = null;
    currentS3Key = null;
    currentIsVideo = false;
    segments = [];
    footerActions.classList.add('hidden');
    ocrSummary.classList.add('hidden');
    ocrSummary.innerHTML = '';
    transcriptionContent.innerHTML = '<div class="placeholder-text">Your transcription will appear here...</div>';
    loadS3Files();
};


// ═══════════════════════════════════════════
//  POD AUTOMATION LOGIC
// ═══════════════════════════════════════════

function setRunpodSidebarOpen(isOpen, restoreFocus = true) {
    if (!runpodSidebar || !runpodSidebarScrim || !runpodToggleBtn) return;

    runpodSidebar.classList.toggle('hidden', !isOpen);
    runpodSidebarScrim.classList.toggle('hidden', !isOpen);
    runpodSidebar.setAttribute('aria-hidden', String(!isOpen));
    runpodToggleBtn.setAttribute('aria-expanded', String(isOpen));
    document.body.classList.toggle('sidebar-open', isOpen);

    if (isOpen) {
        window.setTimeout(() => runpodCloseBtn?.focus(), 0);
    } else if (restoreFocus) {
        runpodToggleBtn.focus();
    }
}

runpodToggleBtn?.addEventListener('click', () => {
    const isOpen = runpodToggleBtn.getAttribute('aria-expanded') === 'true';
    setRunpodSidebarOpen(!isOpen);
});

runpodCloseBtn?.addEventListener('click', () => setRunpodSidebarOpen(false));
runpodSidebarScrim?.addEventListener('click', () => setRunpodSidebarOpen(false));
document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && runpodToggleBtn?.getAttribute('aria-expanded') === 'true') {
        setRunpodSidebarOpen(false);
    }
});

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
    const fetchLogs = async () => {
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
    };

    fetchLogs();
    logPollingInterval = setInterval(fetchLogs, 3000);
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


// ─── Normalization Workflow ───

const transcriptModeBtn = document.getElementById('transcript-mode-btn');
const normalizationModeBtn = document.getElementById('normalization-mode-btn');
const normalizationModeDot = document.getElementById('normalization-mode-dot');
const normalizationWorkspace = document.getElementById('normalization-workspace');
const normalizationStepRail = document.getElementById('normalization-step-rail');
const normalizationStageContent = document.getElementById('normalization-stage-content');
const normalizationStageNumber = document.getElementById('normalization-stage-number');
const normalizationStageTitle = document.getElementById('normalization-stage-title');
const normalizationStageDescription = document.getElementById('normalization-stage-description');
const normalizationStageMessage = document.getElementById('normalization-stage-message');
const normalizationPrimaryAction = document.getElementById('normalization-primary-action');
const normalizationRerunAction = document.getElementById('normalization-rerun-action');
const normalizationDownloadAction = document.getElementById('normalization-download-action');
const normalizationOverallValue = document.getElementById('normalization-overall-value');
const normalizationBackgroundBar = document.getElementById('normalization-background-bar');
const normalizationBackgroundTitle = document.getElementById('normalization-background-title');
const normalizationBackgroundCopy = document.getElementById('normalization-background-copy');
const normalizationBackgroundProgress = document.getElementById('normalization-background-progress');
const normalizationAgentToggle = document.getElementById('normalization-agent-toggle');
const normalizationAgentToggleStatus = document.getElementById('normalization-agent-toggle-status');
const normalizationAgentWindow = document.getElementById('normalization-agent-window');
const normalizationAgentClose = document.getElementById('normalization-agent-close');
const normalizationAgentStep = document.getElementById('normalization-agent-step');
const normalizationAgentMessages = document.getElementById('normalization-agent-messages');
const normalizationAgentForm = document.getElementById('normalization-agent-form');
const normalizationAgentInput = document.getElementById('normalization-agent-input');
const normalizationAgentSend = document.getElementById('normalization-agent-send');
const normalizationAgentError = document.getElementById('normalization-agent-error');

const NORMALIZATION_STATUS_COPY = {
    locked: 'Недоступен', ready: 'Готов к запуску', queued: 'В очереди', running: 'Producer работает',
    reviewing: 'Sol xhigh проверяет', completed: 'Проверено · готово', needs_review: 'Sol xhigh принимает решение',
    failed: 'Review не пройден', stale: 'Нужно обновить'
};

function normalizationWorkerLabel() {
    const editor = normalizationState?.codex?.editor || {};
    const family = String(editor.model || '').includes('sol') ? 'Sol' : String(editor.model || '').includes('luna') ? 'Luna' : 'Producer';
    return `${family} ${editor.effort || ''}`.trim();
}

function hasReusableNormalizationResult(step) {
    return step?.status === 'failed' && Boolean(step?.details?.artifact) && Object.keys(step.details || {}).length > 0;
}

function setReviewMode(mode) {
    const normalizing = mode === 'normalization';
    mainInterface.classList.toggle('normalization-mode', normalizing);
    transcriptModeBtn.classList.toggle('active', !normalizing);
    normalizationModeBtn.classList.toggle('active', normalizing);
    transcriptModeBtn.setAttribute('aria-selected', String(!normalizing));
    normalizationModeBtn.setAttribute('aria-selected', String(normalizing));
    normalizationWorkspace.setAttribute('aria-hidden', String(!normalizing));
    if (currentS3Key || currentTaskId) {
        setFileUrl({ s3Key: currentS3Key, taskId: currentTaskId, mode }, { replace: true });
    }
    if (normalizing && normalizationState) renderNormalizationWorkflow();
}

transcriptModeBtn.addEventListener('click', () => setReviewMode('transcript'));
normalizationModeBtn.addEventListener('click', () => setReviewMode('normalization'));

async function normalizationRequest(path, options = {}) {
    const response = await fetch(path, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data.error) throw new Error(data.error || 'Не удалось выполнить действие.');
    return data;
}

async function initializeNormalizationWorkflow() {
    if (!currentTaskId) return;
    try {
        normalizationState = await normalizationRequest(`/normalization/${encodeURIComponent(currentTaskId)}`);
        renderNormalizationWorkflow();
        updateNormalizationPolling();
    } catch (error) {
        normalizationModeDot.className = 'mode-status-dot error';
        normalizationStageMessage.textContent = error.message;
    }
}

function updateNormalizationPolling() {
    const shouldPoll = Boolean(normalizationState?.running || normalizationState?.agent_busy);
    if (shouldPoll && !normalizationPollInterval) {
        normalizationPollInterval = setInterval(refreshNormalizationState, 1400);
    } else if (!shouldPoll && normalizationPollInterval) {
        clearInterval(normalizationPollInterval);
        normalizationPollInterval = null;
    }
}

async function refreshNormalizationState() {
    if (!currentTaskId) return;
    try {
        normalizationState = await normalizationRequest(`/normalization/${encodeURIComponent(currentTaskId)}`);
        renderNormalizationWorkflow();
        updateNormalizationPolling();
    } catch (error) {
        if (normalizationPollInterval) clearInterval(normalizationPollInterval);
        normalizationPollInterval = null;
    }
}

function renderNormalizationWorkflow() {
    if (!normalizationState?.steps?.length) return;
    const steps = normalizationState.steps;
    let selected = steps.find(step => step.id === normalizationSelectedStep);
    if (!selected) {
        selected = steps[0];
        normalizationSelectedStep = selected.id;
    }

    normalizationOverallValue.textContent = `${normalizationState.overall_progress || 0}%`;
    const running = steps.find(step => ['queued', 'running', 'reviewing'].includes(step.status));
    normalizationModeDot.className = `mode-status-dot ${running ? 'running' : steps.some(s => s.status === 'failed') ? 'error' : 'ready'}`;
    if (running) {
        normalizationBackgroundBar.classList.remove('hidden');
        normalizationBackgroundTitle.textContent = running.status === 'reviewing' ? `Sol xhigh проверяет: ${running.title}` : `${normalizationWorkerLabel()} выполняет: ${running.title}`;
        normalizationBackgroundCopy.textContent = running.status === 'reviewing' ? 'Следующий этап запустится автоматически после pass.' : 'После правок результат автоматически перейдёт независимому reviewer.';
        normalizationBackgroundProgress.textContent = `${running.progress || 0}%`;
    } else {
        normalizationBackgroundBar.classList.add('hidden');
    }

    normalizationStepRail.innerHTML = '';
    steps.forEach((step, index) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = `normalization-step ${step.status}${step.id === selected.id ? ' active' : ''}`;
        button.setAttribute('aria-current', step.id === selected.id ? 'step' : 'false');
        button.innerHTML = `
            <span class="normalization-step-index">${step.status === 'completed' ? '✓' : String(index + 1).padStart(2, '0')}</span>
            <span class="normalization-step-copy"><strong>${escapeHtml(step.title)}</strong><small>${escapeHtml(NORMALIZATION_STATUS_COPY[step.status] || step.status)}</small></span>
            ${['running', 'queued', 'reviewing'].includes(step.status) ? `<span class="normalization-step-progress" style="--step-progress:${step.progress || 0}%"></span>` : ''}`;
        button.addEventListener('click', () => {
            normalizationSelectedStep = step.id;
            renderNormalizationWorkflow();
        });
        normalizationStepRail.appendChild(button);
    });

    normalizationStageNumber.textContent = `Этап ${selected.index + 1} из ${steps.length}`;
    normalizationStageTitle.textContent = selected.title;
    normalizationStageDescription.textContent = selected.description;
    normalizationStageMessage.className = `normalization-stage-message ${selected.status}`;
    const historyCopy = selected.history?.length ? ` · прошлых версий: ${selected.history.length}` : '';
    const gateCopy = selected.gate?.summary ? ` · xhigh: ${selected.gate.summary}` : '';
    normalizationStageMessage.textContent = `${selected.error || selected.stale_reason || NORMALIZATION_STATUS_COPY[selected.status] || ''}${gateCopy}${historyCopy}`;
    configureNormalizationAction(selected);
    renderNormalizationStage(selected);
    renderNormalizationAgentChat(selected);
}

function setNormalizationAgentOpen(isOpen, restoreFocus = true) {
    normalizationAgentOpen = isOpen;
    normalizationAgentWindow.classList.toggle('hidden', !isOpen);
    normalizationAgentToggle.classList.toggle('hidden', isOpen);
    normalizationAgentToggle.setAttribute('aria-expanded', String(isOpen));
    if (isOpen) {
        window.setTimeout(() => normalizationAgentInput?.focus(), 0);
    } else if (restoreFocus) {
        normalizationAgentToggle.focus();
    }
}

function normalizationAgentActionLabel(action) {
    return {
        remediate_structure: 'Запущено исправление диаризации',
        run_step: 'Этап запущен',
        recheck_step: 'Проверка запущена',
        explain: 'Ответ агента',
        none: 'Ответ агента',
        error: 'Команда не выполнена'
    }[action] || '';
}

function renderNormalizationAgentChat(selectedStep) {
    if (!normalizationAgentMessages) return;
    normalizationAgentWindow.classList.toggle('hidden', !normalizationAgentOpen);
    normalizationAgentToggle.classList.toggle('hidden', normalizationAgentOpen);
    normalizationAgentToggle.setAttribute('aria-expanded', String(normalizationAgentOpen));
    normalizationAgentStep.textContent = `${selectedStep.title} · ${NORMALIZATION_STATUS_COPY[selectedStep.status] || selectedStep.status}`;
    const messages = normalizationState?.agent_chat || [];
    if (!messages.length) {
        normalizationAgentMessages.innerHTML = '<div class="normalization-agent-empty"><strong>Дайте агенту команду</strong><p>Например: «Сделай по твоей рекомендации».</p></div>';
    } else {
        normalizationAgentMessages.innerHTML = messages.map(message => {
            const time = message.created_at
                ? new Date(message.created_at * 1000).toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' })
                : '';
            const action = normalizationAgentActionLabel(message.action);
            const meta = [message.role === 'assistant' ? action : '', time].filter(Boolean).join(' · ');
            return `<article class="normalization-agent-message ${message.role === 'user' ? 'user' : 'assistant'} ${message.status === 'thinking' ? 'thinking' : ''} ${message.status === 'error' ? 'error' : ''}"><p>${escapeHtml(message.text || '')}</p>${meta ? `<small>${escapeHtml(meta)}</small>` : ''}</article>`;
        }).join('');
    }
    const lastMessageId = messages.at(-1)?.id || null;
    if (lastMessageId !== normalizationAgentLastMessageId) {
        normalizationAgentMessages.scrollTop = normalizationAgentMessages.scrollHeight;
        normalizationAgentLastMessageId = lastMessageId;
    }
    const busy = Boolean(normalizationState?.agent_busy);
    normalizationAgentInput.disabled = busy;
    normalizationAgentSend.disabled = busy;
    normalizationAgentSend.textContent = busy ? 'Агент думает…' : 'Отправить команду';
    const hasError = messages.at(-1)?.status === 'error';
    const statusClass = busy ? 'busy' : hasError ? 'error' : 'ready';
    normalizationAgentToggleStatus.className = `normalization-agent-status-dot ${statusClass}`;
}

normalizationAgentToggle?.addEventListener('click', () => setNormalizationAgentOpen(true));
normalizationAgentClose?.addEventListener('click', () => setNormalizationAgentOpen(false));

normalizationAgentInput?.addEventListener('keydown', event => {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        normalizationAgentForm.requestSubmit();
    }
});

normalizationAgentForm?.addEventListener('submit', async event => {
    event.preventDefault();
    if (!currentTaskId || normalizationState?.agent_busy) return;
    const command = normalizationAgentInput.value.trim();
    normalizationAgentError.classList.add('hidden');
    normalizationAgentError.textContent = '';
    if (!command) {
        normalizationAgentError.textContent = 'Напишите команду агенту.';
        normalizationAgentError.classList.remove('hidden');
        normalizationAgentInput.focus();
        return;
    }
    normalizationAgentInput.disabled = true;
    normalizationAgentSend.disabled = true;
    normalizationAgentSend.textContent = 'Отправляем…';
    try {
        normalizationState = await normalizationRequest(`/normalization/${encodeURIComponent(currentTaskId)}/agent-command`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ command, step_id: normalizationSelectedStep })
        });
        normalizationAgentInput.value = '';
        renderNormalizationWorkflow();
        updateNormalizationPolling();
    } catch (error) {
        normalizationAgentError.textContent = error.message;
        normalizationAgentError.classList.remove('hidden');
        normalizationAgentInput.disabled = false;
        normalizationAgentSend.disabled = false;
        normalizationAgentSend.textContent = 'Отправить команду';
        normalizationAgentInput.focus();
    }
});

function configureNormalizationAction(step) {
    normalizationPrimaryAction.classList.remove('hidden');
    normalizationPrimaryAction.disabled = false;
    normalizationRerunAction.classList.toggle('hidden', !hasReusableNormalizationResult(step));
    normalizationRerunAction.disabled = false;
    normalizationDownloadAction.classList.toggle('hidden', step.id !== 'render' || step.status !== 'completed');
    if (['queued', 'running', 'reviewing'].includes(step.status)) {
        normalizationPrimaryAction.textContent = step.status === 'reviewing' ? 'Sol xhigh проверяет готовность' : `${normalizationWorkerLabel()} работает · ${step.progress || 0}%`;
        normalizationPrimaryAction.disabled = true;
    } else if (step.id === 'structure' && step.status === 'failed' && (step.gate?.findings || []).some(item => /s\d+/.test(item.item_id || ''))) {
        const count = step.gate.findings.length;
        normalizationPrimaryAction.textContent = `Исправить ${count} ${count === 1 ? 'замечание' : 'замечания'} автоматически`;
    } else if (step.status === 'needs_review') {
        normalizationPrimaryAction.textContent = 'Sol xhigh принимает решение';
        normalizationPrimaryAction.disabled = true;
    } else if (step.status === 'locked') {
        normalizationPrimaryAction.textContent = 'Сначала завершите предыдущий этап';
        normalizationPrimaryAction.disabled = true;
    } else if (step.status === 'completed') {
        normalizationPrimaryAction.textContent = step.id === 'upload' ? 'Загрузить повторно' : 'Запустить заново';
    } else if (hasReusableNormalizationResult(step)) {
        normalizationPrimaryAction.textContent = 'Продолжить проверку результата';
    } else if (step.status === 'failed') {
        normalizationPrimaryAction.textContent = 'Повторить расчёт этапа';
    } else {
        normalizationPrimaryAction.textContent = 'Запустить этап';
    }
}

normalizationPrimaryAction.addEventListener('click', async () => {
    if (!normalizationState || !currentTaskId) return;
    const step = normalizationState.steps.find(item => item.id === normalizationSelectedStep);
    if (!step) return;
    if (step.status === 'completed') {
        const laterDone = normalizationState.steps.slice(step.index + 1).some(item => item.status === 'completed');
        if (laterDone && !confirm('Повторный запуск пометит последующие этапы как устаревшие. Продолжить?')) return;
    }
    normalizationPrimaryAction.disabled = true;
    try {
        if (step.id === 'structure' && step.status === 'failed' && (step.gate?.findings || []).some(item => /s\d+/.test(item.item_id || ''))) {
            normalizationState = await normalizationRequest(`/normalization/${encodeURIComponent(currentTaskId)}/structure/remediate`, { method: 'POST' });
        } else if (hasReusableNormalizationResult(step)) {
            normalizationState = await normalizationRequest(`/normalization/${encodeURIComponent(currentTaskId)}/steps/${step.id}/recheck`, { method: 'POST' });
        } else {
            normalizationState = await normalizationRequest(`/normalization/${encodeURIComponent(currentTaskId)}/steps/${step.id}/run`, { method: 'POST' });
        }
        renderNormalizationWorkflow();
        updateNormalizationPolling();
    } catch (error) {
        normalizationStageMessage.className = 'normalization-stage-message failed';
        normalizationStageMessage.textContent = error.message;
        normalizationPrimaryAction.disabled = false;
    }
});

normalizationRerunAction.addEventListener('click', async () => {
    if (!normalizationState || !currentTaskId) return;
    const step = normalizationState.steps.find(item => item.id === normalizationSelectedStep);
    if (!step || !confirm('Готовый результат будет сохранён в истории, а producer пересчитает этап полностью. Продолжить?')) return;
    normalizationRerunAction.disabled = true;
    try {
        normalizationState = await normalizationRequest(`/normalization/${encodeURIComponent(currentTaskId)}/steps/${step.id}/run`, { method: 'POST' });
        renderNormalizationWorkflow();
        updateNormalizationPolling();
    } catch (error) {
        normalizationStageMessage.className = 'normalization-stage-message failed';
        normalizationStageMessage.textContent = error.message;
        normalizationRerunAction.disabled = false;
    }
});

normalizationDownloadAction.addEventListener('click', () => {
    if (!currentTaskId) return;
    const link = document.createElement('a');
    link.href = `/normalization/${encodeURIComponent(currentTaskId)}/download`;
    link.download = '';
    link.click();
});

function metricCard(value, label, tone = '') {
    return `<div class="normalization-metric ${tone}"><strong>${escapeHtml(value)}</strong><span>${escapeHtml(label)}</span></div>`;
}

function renderNormalizationStage(step) {
    const details = step.details || {};
    if (step.status === 'locked' && !Object.keys(details).length) {
        normalizationStageContent.innerHTML = `<div class="normalization-empty locked"><span>${String(step.index + 1).padStart(2, '0')}</span><h3>${escapeHtml(step.title)}</h3><p>Этот этап откроется после завершения предыдущего. Уже готовые этапы остаются доступными в ленте сверху.</p></div>`;
        return;
    }
    if (['queued', 'running', 'reviewing'].includes(step.status) && !Object.keys(details).length) {
        const worker = step.status === 'reviewing' ? 'Sol xhigh проверяет результат' : `${normalizationWorkerLabel()} выполняет правки`;
        normalizationStageContent.innerHTML = `<div class="normalization-empty running"><span class="normalization-spinner large"></span><h3>${escapeHtml(step.live?.label || worker)}</h3><p>${escapeHtml(step.live?.message || 'Можно перейти к предыдущим этапам — процесс продолжит работу, а следующий этап запустится после pass.')}</p><div class="normalization-inline-progress"><i style="width:${step.progress || 0}%"></i></div></div>`;
        renderNormalizationLive(step);
        return;
    }
    if (step.status === 'failed' && !Object.keys(details).length) {
        normalizationStageContent.innerHTML = `<div class="normalization-empty failed"><span>!</span><h3>Этап остановлен</h3><p>${escapeHtml(step.error || 'Неизвестная ошибка')}</p></div>`;
        return;
    }
    const renderer = NORMALIZATION_RENDERERS[step.id] || renderGenericNormalizationStage;
    renderer(step, details);
    renderNormalizationLive(step);
    renderNormalizationGate(step);
}

function renderNormalizationLive(step) {
    if (!['queued', 'running', 'reviewing'].includes(step.status) || !step.live) return;
    const live = step.live;
    const events = (live.events || []).slice(-8).reverse().map(item => `<li><time>${new Date((item.at || 0) * 1000).toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</time><span>${escapeHtml(item.message || '')}</span></li>`).join('');
    const counter = live.total ? `${live.current || 0} / ${live.total}` : `${step.progress || 0}%`;
    const panel = document.createElement('section');
    panel.className = 'normalization-live-panel';
    panel.innerHTML = `<header><div><small>Живой журнал этапа</small><strong>${escapeHtml(live.label || step.title)}</strong></div><b>${escapeHtml(counter)}</b></header><p>${escapeHtml(live.message || '')}</p>${events ? `<ol>${events}</ol>` : ''}<footer>Показываются операции, решения и явные обоснования модели — без скрытого внутреннего рассуждения.</footer>`;
    normalizationStageContent.prepend(panel);
}

function renderNormalizationGate(step) {
    const gate = step.gate;
    if (!gate?.verdict) return;
    const findings = (gate.findings || []).map(item => {
        const termId = /^term-\d+$/.test(item.item_id || '') ? item.item_id : '';
        return `<li><span class="confidence ${escapeHtml(item.severity || 'low')}">${escapeHtml(item.severity || 'low')}</span><div><strong>${escapeHtml(item.code || 'review')}</strong><p>${escapeHtml(item.message || '')}</p>${termId ? `<button class="gate-finding-jump" type="button" data-term-id="${escapeHtml(termId)}">Показать ${escapeHtml(termId)} ↓</button>` : ''}</div></li>`;
    }).join('');
    const actionRequired = step.id === 'terms' ? (step.details?.action_required ?? step.details?.pending ?? 0) : 0;
    const structureRepairCount = step.id === 'structure' && gate.verdict === 'fail'
        ? (gate.findings || []).filter(item => /s\d+/.test(item.item_id || '')).length
        : 0;
    const banner = document.createElement('section');
    banner.className = `normalization-gate-banner ${gate.verdict}`;
    banner.innerHTML = `<div class="gate-verdict"><span>${gate.verdict === 'pass' || gate.verdict === 'legacy_pass' ? '✓' : gate.verdict === 'needs_review' || actionRequired ? '!' : '×'}</span><div><small>Sol xhigh · ${escapeHtml(gate.effort || 'xhigh')}</small><strong>${gate.verdict === 'pass' ? ((gate.assumptions || []).length ? `Решение принято · допущений: ${gate.assumptions.length}` : 'Gate пройден') : gate.verdict === 'legacy_pass' ? 'Legacy-результат' : gate.verdict === 'needs_review' || actionRequired ? 'Sol xhigh принимает решение' : `Возврат ${escapeHtml(normalizationWorkerLabel())}`}</strong></div></div><p>${escapeHtml(gate.summary || '')}</p>${actionRequired ? `<button class="gate-primary-jump" type="button" data-action="terms">Показать ${actionRequired} обрабатываемых решений ↓</button>` : ''}${structureRepairCount ? `<button class="gate-primary-jump" type="button" data-action="structure-repair">Исправить ${structureRepairCount} ${structureRepairCount === 1 ? 'замечание' : 'замечания'} автоматически</button>` : ''}${findings ? `<ul>${findings}</ul>` : ''}`;
    normalizationStageContent.prepend(banner);
    banner.querySelector('[data-action="terms"]')?.addEventListener('click', () => normalizationStageContent.querySelector('#term-action-zone')?.scrollIntoView({ behavior: 'smooth', block: 'start' }));
    banner.querySelector('[data-action="structure-repair"]')?.addEventListener('click', remediateNormalizationStructure);
    banner.querySelectorAll('.gate-finding-jump').forEach(button => button.addEventListener('click', () => {
        const card = normalizationStageContent.querySelector(`[data-term-id="${CSS.escape(button.dataset.termId)}"]`);
        card?.scrollIntoView({ behavior: 'smooth', block: 'center' });
        card?.classList.add('focus-pulse');
        window.setTimeout(() => card?.classList.remove('focus-pulse'), 1400);
    }));
}

async function remediateNormalizationStructure(event) {
    const button = event?.currentTarget;
    if (button) button.disabled = true;
    normalizationPrimaryAction.disabled = true;
    try {
        normalizationState = await normalizationRequest(`/normalization/${encodeURIComponent(currentTaskId)}/structure/remediate`, { method: 'POST' });
        renderNormalizationWorkflow();
        updateNormalizationPolling();
    } catch (error) {
        normalizationStageMessage.className = 'normalization-stage-message failed';
        normalizationStageMessage.textContent = error.message;
        if (button) button.disabled = false;
        normalizationPrimaryAction.disabled = false;
    }
}

const NORMALIZATION_RENDERERS = {
    source: (_step, d) => {
        const enabled = normalizationState?.settings?.contextual_rediarization !== false;
        const structure = normalizationState?.steps?.find(item => item.id === 'structure');
        const disabled = Boolean(normalizationState?.running);
        const setting = `<section class="normalization-option-card ${enabled ? 'enabled' : ''}"><div><span>Рекомендуется для фокус-групп</span><h4>Контекстная передиаризация</h4><p>${escapeHtml(normalizationWorkerLabel())} перепроверит короткие фрагменты у смены голоса. Sol xhigh примет спорные mid/low решения и добавит их в журнал передачи оператору.</p></div><label class="normalization-switch"><input id="contextual-rediarization-toggle" type="checkbox" ${enabled ? 'checked' : ''} ${disabled ? 'disabled' : ''}><span aria-hidden="true"></span><b>${enabled ? 'Включена' : 'Выключена'}</b></label></section>`;
        const result = d.segments
            ? `<div class="normalization-metrics">${metricCard(String(d.segments), 'сегментов')}${metricCard(String(d.words), 'слов')}${metricCard(String(d.speakers), 'голосов')}${metricCard(formatTime(d.duration_seconds || 0), 'длительность')}</div><div class="normalization-checklist"><div class="passed"><span>✓</span><div><strong>Структура читается</strong><small>Стабильные ID назначены всем сегментам</small></div></div><div class="${d.warning_count ? 'warning' : 'passed'}"><span>${d.warning_count ? '!' : '✓'}</span><div><strong>${d.warning_count || 0} предупреждений</strong><small>${d.warning_count ? 'Проверьте порядок таймкодов и пустые фрагменты' : 'Таймкоды и текст прошли базовую проверку'}</small></div></div></div>`
            : `<div class="normalization-checklist"><div class="passed"><span>1</span><div><strong>Сначала будет проверен JSON</strong><small>После gate процесс автоматически перейдёт к ролям и границам реплик</small></div></div></div>`;
        normalizationStageContent.innerHTML = `<div class="normalization-section-heading"><span>Входные данные</span><h3>${d.segments ? 'JSON готов к обработке' : 'Настройте обработку перед запуском'}</h3><p>В рабочий контекст дальше попадут только необходимые поля, а не весь тяжёлый исходный объект.</p></div>${setting}${result}`;
        const toggle = document.getElementById('contextual-rediarization-toggle');
        toggle?.addEventListener('change', async () => {
            const nextValue = toggle.checked;
            const hasStructureResult = Boolean(structure?.attempt || Object.keys(structure?.details || {}).length);
            if (hasStructureResult && !confirm('Изменение настройки перезапустит этап «Роли и реплики» и пометит последующие результаты устаревшими. Продолжить?')) {
                toggle.checked = !nextValue;
                return;
            }
            toggle.disabled = true;
            try {
                normalizationState = await normalizationRequest(`/normalization/${encodeURIComponent(currentTaskId)}/settings`, {
                    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ contextual_rediarization: nextValue })
                });
                renderNormalizationWorkflow();
            } catch (error) {
                normalizationStageMessage.className = 'normalization-stage-message failed';
                normalizationStageMessage.textContent = error.message;
                toggle.checked = !nextValue;
                toggle.disabled = false;
            }
        });
    },
    structure: (_step, d) => {
        if (!d.speakers) return renderGenericNormalizationStage(_step, d);
        const rows = d.speakers.map(item => `<div class="registry-row" data-source-id="${escapeHtml(item.source_id)}"><code>${escapeHtml(item.source_id)}</code><select aria-label="Роль ${escapeHtml(item.source_id)}"><option ${item.role === 'Интервьюер' ? 'selected' : ''}>Интервьюер</option><option ${item.role === 'Респондент' ? 'selected' : ''}>Респондент</option></select><input value="${escapeHtml(item.name || '')}" placeholder="Имя, если прозвучало" aria-label="Имя ${escapeHtml(item.source_id)}"><span class="confidence ${escapeHtml(item.confidence || 'low')}">${escapeHtml(item.confidence || 'low')}</span></div>`).join('');
        const options = d.speakers.map(item => `<option value="${escapeHtml(item.source_id)}">${escapeHtml(item.name ? `${item.role} · ${item.name}` : `${item.role} · ${item.source_id}`)}</option>`).join('');
        const speakerById = Object.fromEntries(d.speakers.map(item => [item.source_id, item]));
        const labelFor = sourceId => {
            const speaker = speakerById[sourceId];
            return speaker ? (speaker.name || `${speaker.role} · ${speaker.source_id}`) : sourceId;
        };
        const reviewTurns = (d.review_turns || []).map(item => `<article class="speaker-review-turn" data-turn-id="${escapeHtml(item.turn_id)}"><header><code>${escapeHtml(item.turn_id)}</code><span>${formatTime(item.start || 0)}</span><small>${escapeHtml(item.source_id)}</small><span class="confidence low">low</span></header><p>${escapeHtml(item.text)}</p><div class="speaker-suggestion"><span>Решение Sol xhigh</span><strong>${escapeHtml(labelFor(item.selected_source_id))}</strong><small>${escapeHtml(item.reason || '')}</small></div><label>Изменить только при явной ошибке<select aria-label="Говорящий в ${escapeHtml(item.turn_id)}">${options}</select></label></article>`).join('');
        const auditRows = (d.assignment_audit || []).map(item => `<div class="diarization-audit-row"><code>${escapeHtml(item.turn_id)}</code><span>${formatTime(item.start || 0)}</span><span class="audit-route"><del>${escapeHtml(item.original_source_id)}</del><b>→</b><strong>${escapeHtml(labelFor(item.assigned_source_id))}</strong></span><span class="confidence ${escapeHtml(item.confidence)}">${escapeHtml(item.confidence)}</span><small>${escapeHtml(item.reason || '')}</small></div>`).join('');
        const detected = d.detected_defect_count || 0;
        const autoFixed = d.auto_fixed_count || 0;
        const reviewCount = d.review_turn_count || 0;
        const confidence = d.assignment_confidence || { safe: 0, mid: 0, low: reviewCount };
        const rediarization = d.contextual_rediarization || {};
        const interiorSplits = d.interior_turn_splits || {};
        const scanCopy = rediarization.enabled
            ? `Проверено ${rediarization.candidates || 0} коротких пограничных сегментов и ${interiorSplits.candidates || 0} реплик с признаками внутреннего диалога; скрытых смен найдено ${interiorSplits.detected || 0}.`
            : 'Контекстная передиаризация была выключена; известные метки WhisperX сохранены.';
        const repairSummary = detected ? `<section class="diarization-repair-summary"><div class="repair-summary-mark">✓</div><div class="repair-summary-copy"><span>Контекстная коррекция завершена</span><h3>${autoFixed} из ${detected} найденных дефектов исправлены автоматически</h3><p>${scanCopy} Все Unknown назначены существующим участникам. ${reviewCount ? `${reviewCount} low-confidence ${reviewCount === 1 ? 'решение принято' : 'решений приняты'} Sol xhigh и записаны как допущения.` : 'Дополнительных допущений не потребовалось.'}</p></div><div class="repair-confidence"><div><strong>${confidence.safe || 0}</strong><span>safe</span></div><div><strong>${confidence.mid || 0}</strong><span>mid</span></div><div class="${reviewCount ? 'has-review' : ''}"><strong>${confidence.low || 0}</strong><span>low</span></div></div></section>${auditRows ? `<details class="diarization-audit"><summary>Показать аудит ${detected} назначений <span>Все решения сохранены</span></summary><div>${auditRows}</div></details>` : ''}` : `<div class="normalization-checklist"><div class="passed"><span>✓</span><div><strong>Дефекты диаризации не обнаружены</strong><small>${escapeHtml(scanCopy)} Все технические ID связаны с участниками.</small></div></div></div>`;
        const manualReview = reviewCount ? `<div class="speaker-review-heading"><div><span>Допущения Sol xhigh</span><h4>${reviewCount} low-confidence ${reviewCount === 1 ? 'решение' : 'решений'}</h4></div><p>Эти решения уже применены и не останавливают процесс. Они войдут в передачу оператору; изменить их можно только при явной ошибке.</p></div><div class="speaker-review-list">${reviewTurns}</div>` : `<div class="auto-repair-complete"><span>✓</span><div><strong>Допущения по говорящим не потребовались</strong><small>Все найденные дефекты получили safe или mid назначение.</small></div></div>`;
        normalizationStageContent.innerHTML = `<div class="normalization-section-heading"><span>Автоматическая диаризация</span><h3>${d.speakers.length} участников · ${d.turns} реплик</h3><p>${escapeHtml(normalizationWorkerLabel())} создаёт реестр и уточняет диаризацию, а Sol xhigh принимает спорные решения.</p></div>${repairSummary}<div class="registry-section-heading"><div><span>Реестр участников</span><h4>Имена и глобальные роли</h4></div><p>Технический Unknown не является участником и сюда не попадает: его реплики уже распределены между людьми ниже.</p></div><div class="registry-table"><div class="registry-head"><span>Исходный ID</span><span>Роль</span><span>Отображаемое имя</span><span>Уверенность</span></div>${rows}</div>${manualReview}<button id="save-normalization-registry" class="normalization-inline-action" type="button">Сохранить изменения реестра</button>`;
        (d.review_turns || []).forEach(item => {
            const card = normalizationStageContent.querySelector(`.speaker-review-turn[data-turn-id="${CSS.escape(item.turn_id)}"]`);
            if (card) card.querySelector('select').value = item.selected_source_id || item.source_id;
        });
        document.getElementById('save-normalization-registry').addEventListener('click', saveNormalizationRegistry);
    },
    chunks: (_step, d) => {
        if (!d.items) return renderGenericNormalizationStage(_step, d);
        normalizationStageContent.innerHTML = `<div class="normalization-section-heading"><span>Контекстные блоки</span><h3>${d.chunks} чанков по ~${d.target_words} слов</h3><p>Границы проходят только между репликами; соседние реплики добавляются как неизменяемый контекст.</p></div><div class="chunk-grid">${d.items.map(item => `<div class="chunk-card"><span>${escapeHtml(item.id)}</span><strong>${item.words.toLocaleString()} слов</strong><small>${item.turns} реплик</small></div>`).join('')}</div>`;
    },
    terms: (_step, d) => {
        if (!d.items) return renderGenericNormalizationStage(_step, d);
        const vocabulary = d.vocabulary || {};
        const gateReviewedAt = _step.gate?.reviewed_at;
        const flaggedIds = new Set(d.reviewer_flagged_ids || []);
        const findingById = Object.fromEntries((_step.gate?.findings || []).filter(item => /^term-\d+$/.test(item.item_id || '')).map(item => [item.item_id, item]));
        const needsAction = item => item.decision === 'pending' || (flaggedIds.has(item.id) && item.operator_reviewed_gate_at !== gateReviewedAt);
        const contextSnippet = item => {
            const text = item.context_text || '';
            if (!text) return '';
            const needle = item.original || '';
            const matchAt = text.toLocaleLowerCase().indexOf(needle.toLocaleLowerCase());
            if (matchAt < 0 || text.length <= 420) return text;
            const start = Math.max(0, matchAt - 150);
            const end = Math.min(text.length, matchAt + needle.length + 210);
            return `${start ? '…' : ''}${text.slice(start, end).trim()}${end < text.length ? '…' : ''}`;
        };
        const renderItem = item => {
            const pending = item.decision === 'pending';
            const reviewerFlagged = flaggedIds.has(item.id) && item.operator_reviewed_gate_at !== gateReviewedAt;
            const finding = findingById[item.id];
            const badge = reviewerFlagged ? 'Sol xhigh перепроверяет' : pending ? 'Sol xhigh решает' : item.decision === 'accepted' ? 'Принято Sol xhigh' : 'Исходник сохранён';
            const context = contextSnippet(item);
            return `<article class="term-review-card ${needsAction(item) ? 'action-required' : 'resolved'} ${reviewerFlagged ? 'reviewer-flagged' : ''}" data-term-id="${escapeHtml(item.id)}"><div class="term-review-top"><span class="term-state-label">${badge}</span><span class="confidence ${escapeHtml(item.safety)}">${escapeHtml(item.safety)}</span><code>${escapeHtml(item.id)}</code><code>${escapeHtml(item.turn_id)}</code><small>${escapeHtml(item.chunk_id)}</small></div>${context ? `<div class="term-context"><small>Контекст реплики</small><p>${escapeHtml(context)}</p></div>` : ''}<div class="term-change"><del>${escapeHtml(item.original)}</del><span>→</span><label><small>Вариант замены · пустое поле удалит фрагмент</small><input class="term-proposed-input" value="${escapeHtml(item.proposed)}" aria-label="Вариант замены ${escapeHtml(item.id)}"></label></div><p>${escapeHtml(item.reason)}</p>${item.operator_edited ? `<div class="term-operator-edit">Вариант уточнён оператором${item.model_proposed ? ` · ${escapeHtml(normalizationWorkerLabel())} предлагал «${escapeHtml(item.model_proposed)}»` : ''}</div>` : ''}${finding ? `<div class="term-reviewer-note"><strong>Почему Sol xhigh остановил решение</strong><span>${escapeHtml(finding.message)}</span></div>` : ''}<div class="term-decisions" data-term-id="${escapeHtml(item.id)}"><button class="${item.decision === 'accepted' && !needsAction(item) ? 'active accepted' : ''}" data-decision="accepted">Применить вариант</button><button class="term-delete-fragment" data-decision="accepted" data-delete="true">Удалить фрагмент</button><button class="${item.decision === 'rejected' && !needsAction(item) ? 'active rejected' : ''}" data-decision="rejected">Оставить исходник</button><button class="${pending ? 'active' : ''}" data-decision="pending">Отложить</button></div></article>`;
        };
        const actionItems = d.items.filter(needsAction);
        const resolvedItems = d.items.filter(item => !needsAction(item));
        const actionRequired = actionItems.length;
        const actionSection = actionRequired ? `<section class="term-action-zone" id="term-action-zone"><div class="term-action-heading"><div><span>Sol xhigh принимает решения</span><h4>${actionRequired} ${actionRequired === 1 ? 'кандидат обрабатывается' : 'кандидатов обрабатываются'}</h4></div><p>Процесс продолжится автоматически: неподтверждённая замена будет отклонена с сохранением исходной речи.</p></div><div class="term-review-list action-list">${actionItems.map(renderItem).join('')}</div></section>` : '<div class="auto-repair-complete"><span>✓</span><div><strong>Все решения приняты Sol xhigh</strong><small>Mid/low выборы записаны в журнал передачи оператору</small></div></div>';
        const resolvedSection = resolvedItems.length ? `<details class="resolved-terms"><summary><span>Уже решено</span><strong>${resolvedItems.length} карточек</strong><small>Показать автоматически принятые и проверенные решения</small></summary><div class="term-review-list">${resolvedItems.map(renderItem).join('')}</div></details>` : '';
        const vocabularyStatus = vocabulary.available
            ? `<div class="normalization-checklist"><div class="passed"><span>✓</span><div><strong>Подключён словарь Transcriber</strong><small>${(vocabulary.entries || 0).toLocaleString()} записей · ${(vocabulary.aliases || 0).toLocaleString()} алиасов · ${vocabulary.relevant_hints || 0} релевантных подсказок · ${vocabulary.exact_candidates || 0} точных замен принято автоматически</small></div></div></div>`
            : '<div class="normalization-checklist"><div class="warning"><span>!</span><div><strong>Словарь Transcriber не найден</strong><small>Sol продолжит контекстную проверку, но без локального справочника препаратов и компаний.</small></div></div></div>';
        const reviewStatus = d.review_candidates != null
            ? `<div class="normalization-checklist"><div class="passed"><span>✓</span><div><strong>Sol xhigh рассмотрел ${d.review_candidates} кандидатов в ${d.review_batches || 0} батчах</strong><small>Все модельные safe / mid / low проверены без обрезки UI</small></div></div><div class="passed"><span>✓</span><div><strong>Coverage: ${d.coverage_chunks || 0} чанков · новых находок: ${d.coverage_added || 0}</strong><small>Отдельно проверены пропущенные препараты, бренды, компании, сокращения и термины</small></div></div></div>`
            : '';
        normalizationStageContent.innerHTML = `<div class="normalization-section-heading"><span>Терминологический словарь</span><h3>${d.candidates} кандидатов · решения приняты</h3><p>Точные словарные совпадения принимаются детерминированно; каждый модельный safe / mid / low отдельно проверяет Sol xhigh.</p></div>${reviewStatus}${vocabularyStatus}<div class="safety-summary">${metricCard(String(d.by_safety?.safe || 0), 'safe', 'safe')}${metricCard(String(d.by_safety?.mid || 0), 'mid', 'mid')}${metricCard(String(d.by_safety?.low || 0), 'low', 'low')}</div>${actionSection}${resolvedSection}`;
        normalizationStageContent.querySelectorAll('.term-decisions button').forEach(button => button.addEventListener('click', decideNormalizationTerm));
    },
    language: (_step, d) => {
        if (!d.items) return renderGenericNormalizationStage(_step, d);
        const actions = d.adjudication_actions || {};
        const adjudication = d.adjudication_mode === 'sol_xhigh_final'
            ? `<div class="normalization-checklist"><div class="passed"><span>✓</span><div><strong>Sol xhigh вынес окончательные решения по ${d.adjudication_chunks || 0} чанкам</strong><small>Принято: ${actions.accept || 0}; возвращено к источнику: ${actions.revert || 0}; заменено адресно: ${actions.replace || 0}; отклонено guardrail: ${actions.rejected_replace || 0}</small></div></div><div class="passed"><span>✓</span><div><strong>Повторный producer не запускался</strong><small>Решения применены по turn_id; неоднозначные случаи сохраняют исходную речь</small></div></div></div>`
            : '';
        normalizationStageContent.innerHTML = `<div class="normalization-section-heading"><span>Conservative edit + adjudication</span><h3>${d.changes} точечных правок</h3><p>${d.needs_review} правок требуют особого внимания. Живая речь, повторы и незавершённые мысли сохраняются.</p></div>${adjudication}<div class="change-list">${d.items.map(item => `<article class="change-card ${item.guardrail === 'review' ? 'review' : ''}"><header><code>${escapeHtml(item.turn_id)}</code><span class="confidence ${item.confidence}">${escapeHtml(item.confidence)}</span>${item.adjudicated_by === 'sol_xhigh' ? '<span class="review-chip">решено xhigh</span>' : item.guardrail === 'review' ? '<span class="review-chip">проверить</span>' : ''}</header><div class="change-version old">${escapeHtml(item.original)}</div><div class="change-version new">${escapeHtml(item.text)}</div><p>${escapeHtml(item.reason)}</p></article>`).join('') || '<div class="normalization-empty compact"><h3>Правки не нужны</h3><p>Текст передан дальше без языковых изменений.</p></div>'}</div>`;
    },
    fidelity: (_step, d) => {
        if (d.issues == null) return renderGenericNormalizationStage(_step, d);
        const issueCards = (d.items || []).map(item => `<article class="issue-card ${item.severity}"><header><span>${escapeHtml(item.severity)}</span><code>${escapeHtml(item.change_id)}</code>${item.turn_id ? `<code>${escapeHtml(item.turn_id)}</code>` : ''}</header><p>${escapeHtml(item.message)}</p>${item.original ? `<div class="issue-diff"><div><small>Исходник</small><del>${escapeHtml(item.original)}</del></div><div><small>После правки</small><ins>${escapeHtml(item.revised || '')}</ins></div></div>` : ''}${item.approved_terms?.length ? `<small class="issue-approved">Разрешено терминологическим gate: ${item.approved_terms.map(term => escapeHtml(term.id)).join(', ')}</small>` : ''}</article>`).join('');
        normalizationStageContent.innerHTML = `<div class="normalization-section-heading"><span>True to the source</span><h3>${d.issues ? `${d.issues} рисков · ${d.deterministic_reverts || 0} откатов к источнику` : 'Смысловые риски не найдены'}</h3><p>Sol xhigh проверяет результат независимо. Рискованные правки удаляются детерминированно и не возвращаются producer на новый прогон.</p></div>${d.issues ? `<div class="issue-list">${issueCards}</div>` : '<div class="normalization-checklist"><div class="passed"><span>✓</span><div><strong>Проверка пройдена</strong><small>Изменения смысла, лица, отрицания и степени уверенности не обнаружены</small></div></div></div>'}`;
    },
    assemble: (_step, d) => {
        if (!d.integrity) return renderGenericNormalizationStage(_step, d);
        const deltaTone = Math.abs(d.word_delta_percent || 0) > 3 ? 'warning' : 'safe';
        normalizationStageContent.innerHTML = `<div class="normalization-section-heading"><span>Integrity check</span><h3>Все реплики собраны по стабильным ID</h3><p>Сборка не опирается на генерацию модели: порядок, таймкоды и количество реплик проверяет код.</p></div><div class="normalization-metrics">${metricCard(String(d.turns), 'реплик')}${metricCard(d.source_words.toLocaleString(), 'слов до')}${metricCard(d.final_words.toLocaleString(), 'слов после')}${metricCard(`${d.word_delta_percent > 0 ? '+' : ''}${d.word_delta_percent}%`, 'дельта', deltaTone)}</div><div class="normalization-checklist"><div class="passed"><span>✓</span><div><strong>Целостность пройдена</strong><small>Нет потерянных, пустых или продублированных реплик</small></div></div><div class="passed"><span>✓</span><div><strong>Формат помет нормализован</strong><small>Только … и (неразборчиво), без сценических ремарок</small></div></div></div>`;
    },
    approve: (_step, d) => {
        const assumptions = normalizationState?.assumptions || d.assumptions || [];
        normalizationStageContent.innerHTML = `<div class="normalization-section-heading"><span>Передача оператору</span><h3>${assumptions.length} ${assumptions.length === 1 ? 'допущение зафиксировано' : 'допущений зафиксировано'}</h3><p>Процесс не ждёт ручного подтверждения. Sol xhigh принял спорные решения и передаёт оператору выбранный вариант вместе с основанием.</p></div>${normalizationAssumptionsMarkup({ assumptions }, 'flow-assumptions') || '<div class="auto-repair-complete"><span>✓</span><div><strong>Допущений нет</strong><small>Все решения прошли с safe-уверенностью</small></div></div>'}`;
    },
    render: (_step, d) => {
        if (!d.preview) return renderGenericNormalizationStage(_step, d);
        const recoveryNote = d.recovery === 'operator_reference' ? '<div class="operator-reminder"><strong>Reference-assisted recovery</strong><span>Автономный render-gate обнаружил системные ошибки границ говорящих. Финальная версия восстановлена по явно предоставленному операторскому эталону и повторно прошла Sol xhigh.</span></div>' : '';
        normalizationStageContent.innerHTML = `<div class="normalization-section-heading"><span>Markdown preview</span><h3>${escapeHtml(d.filename)}</h3><p>${d.words.toLocaleString()} слов · SHA-256 ${escapeHtml(d.sha256.slice(0, 12))}…</p></div>${recoveryNote}<pre class="markdown-preview">${escapeHtml(d.preview)}</pre>`;
    },
    upload: (_step, d) => {
        if (!d.key) return renderGenericNormalizationStage(_step, d);
        const assumptions = normalizationState?.assumptions || d.assumptions || [];
        normalizationStageContent.innerHTML = `<div class="normalization-upload-success"><span>✓</span><h3>Результат готов</h3><p>Финальный MD загружен на сервер. Ниже — все допущения, за которые отвечает Sol xhigh и которые передаются оператору.</p><dl><div><dt>Файл</dt><dd>${escapeHtml(d.filename)}</dd></div><div><dt>S3 key</dt><dd><code>${escapeHtml(d.key)}</code></dd></div><div><dt>SHA-256</dt><dd><code>${escapeHtml(d.sha256)}</code></dd></div><div><dt>Допущения</dt><dd>${assumptions.length}</dd></div></dl></div>${normalizationAssumptionsMarkup({ assumptions }, 'flow-assumptions') || '<div class="auto-repair-complete"><span>✓</span><div><strong>Допущений нет</strong><small>Все решения прошли с safe-уверенностью</small></div></div>'}`;
    }
};

function renderGenericNormalizationStage(step, details) {
    normalizationStageContent.innerHTML = `<div class="normalization-empty"><span>${String(step.index + 1).padStart(2, '0')}</span><h3>${escapeHtml(step.title)}</h3><p>${escapeHtml(step.description)}</p>${Object.keys(details).length ? `<pre>${escapeHtml(JSON.stringify(details, null, 2))}</pre>` : ''}</div>`;
}

async function saveNormalizationRegistry() {
    const rows = [...normalizationStageContent.querySelectorAll('.registry-row')];
    const speakers = rows.map(row => ({ source_id: row.dataset.sourceId, role: row.querySelector('select').value, name: row.querySelector('input').value.trim() }));
    const overrides = [...normalizationStageContent.querySelectorAll('.speaker-review-turn')].map(card => ({
        turn_id: card.dataset.turnId,
        source_id: card.querySelector('select').value
    }));
    const button = document.getElementById('save-normalization-registry');
    button.disabled = true;
    try {
        normalizationState = await normalizationRequest(`/normalization/${encodeURIComponent(currentTaskId)}/speaker-registry`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ speakers, overrides })
        });
        renderNormalizationWorkflow();
    } catch (error) {
        normalizationStageMessage.className = 'normalization-stage-message failed';
        normalizationStageMessage.textContent = error.message;
        button.disabled = false;
    }
}

async function decideNormalizationTerm(event) {
    const button = event.currentTarget;
    const card = button.closest('.term-review-card');
    const termId = card.dataset.termId;
    const proposed = button.dataset.delete === 'true' ? '' : card.querySelector('.term-proposed-input')?.value.trim();
    button.disabled = true;
    try {
        normalizationState = await normalizationRequest(`/normalization/${encodeURIComponent(currentTaskId)}/terms/${encodeURIComponent(termId)}`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ decision: button.dataset.decision, proposed })
        });
        renderNormalizationWorkflow();
        const nextCard = normalizationStageContent.querySelector('.term-review-card.action-required');
        nextCard?.scrollIntoView({ behavior: 'smooth', block: 'center' });
        nextCard?.querySelector('.term-proposed-input')?.focus({ preventScroll: true });
    } catch (error) {
        normalizationStageMessage.className = 'normalization-stage-message failed';
        normalizationStageMessage.textContent = error.message;
    }
}

// Start polling and load config on load
startLogPolling();
startPodPolling();
loadConfig();
loadS3Files();
loadDefaultConclusionsInstruction();
loadConclusionsTasks({ silent: true });
(async () => {
    const restoredFromUrl = await restoreTaskFromUrl();
    if (!restoredFromUrl) await restoreActiveTask();
})();

window.addEventListener('popstate', () => window.location.reload());

window.seekTo = seekTo;
window.setSegmentSpeaker = setSegmentSpeaker;
