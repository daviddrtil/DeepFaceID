const startScreen = document.getElementById('start-screen');
const mainScreen = document.getElementById('main-screen');
const completionScreen = document.getElementById('completion-screen');
const startBtn = document.getElementById('start-btn');
const resetBtn = document.getElementById('reset-btn');
const tryAgainBtn = document.getElementById('try-again-btn');
const permissionError = document.getElementById('permission-error');
const webcam = document.getElementById('webcam');
const overlay = document.getElementById('overlay');
const actionPrompt = document.getElementById('action-prompt');
const progressFill = document.getElementById('progress-fill');
const statusText = document.getElementById('status-text');
const finalStatus = document.getElementById('final-status');
const finalMessage = document.getElementById('final-message');
const successSound = document.getElementById('success-sound');
const failSound = document.getElementById('fail-sound');
const sessionNameInput = document.getElementById('session-name');
const sessionError = document.getElementById('session-error');

const LAST_SESSION_NAME_KEY = 'deepfaceid:last-session-name';

let ws = null;
let stream = null;
let ctx = null;
let frameInterval = null;
let drawInterval = null;
let lastCompletedAction = null;
let lastServerData = null;
let canvasInitialized = false;

let target_fps = 30;
let frameCaptureInterval = 1000 / target_fps;

function sanitizeSessionName(name) {
    return name.trim().replace(/\s+/g, '_').toLowerCase();
}

function showScreen(screen) {
    [startScreen, mainScreen, completionScreen].forEach(s => s.classList.remove('active'));
    screen.classList.add('active');
}

async function requestCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'user' }
        });
        webcam.srcObject = stream;
        await webcam.play();
        return true;
    } catch (err) {
        permissionError.textContent = 'Camera access denied. Please allow camera permissions.';
        permissionError.classList.remove('hidden');
        return false;
    }
}

function connectWebSocket(sessionName) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const encodedName = encodeURIComponent(sessionName);
    ws = new WebSocket(`${protocol}//${window.location.host}/ws?session_name=${encodedName}`);

    ws.onopen = () => {
        console.log('WebSocket connected');
        startFrameCapture();
        startDrawLoop();
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleServerResponse(data);
    };

    ws.onclose = () => {
        console.log('WebSocket closed');
        stopFrameCapture();
        stopDrawLoop();
    };

    ws.onerror = (err) => {
        console.error('WebSocket error:', err);
    };
}

function startFrameCapture() {
    frameInterval = setInterval(captureAndSendFrame, frameCaptureInterval);
}

function stopFrameCapture() {
    if (frameInterval) {
        clearInterval(frameInterval);
        frameInterval = null;
    }
}

function startDrawLoop() {
    ctx = overlay.getContext('2d');
    drawInterval = setInterval(drawOverlay, frameCaptureInterval);
}

function stopDrawLoop() {
    if (drawInterval) {
        clearInterval(drawInterval);
        drawInterval = null;
    }
}

function initCanvasSize() {
    if (webcam.videoWidth === 0) return false;
    
    const container = webcam.parentElement;
    const containerW = container.clientWidth;
    const containerH = container.clientHeight;
    const videoW = webcam.videoWidth;
    const videoH = webcam.videoHeight;
    
    const containerAspect = containerW / containerH;
    const videoAspect = videoW / videoH;
    
    let displayW, displayH;
    if (videoAspect > containerAspect) {
        displayW = containerW;
        displayH = containerW / videoAspect;
    } else {
        displayH = containerH;
        displayW = containerH * videoAspect;
    }
    
    overlay.width = videoW;
    overlay.height = videoH;
    overlay.style.width = displayW + 'px';
    overlay.style.height = displayH + 'px';
    
    canvasInitialized = true;
    return true;
}

function captureAndSendFrame() {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (webcam.videoWidth === 0) return;

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = webcam.videoWidth;
    tempCanvas.height = webcam.videoHeight;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.translate(tempCanvas.width, 0);
    tempCtx.scale(-1, 1);
    tempCtx.drawImage(webcam, 0, 0);

    const dataUrl = tempCanvas.toDataURL('image/jpeg', 0.7);
    const base64 = dataUrl.split(',')[1];

    ws.send(JSON.stringify({
        type: 'frame',
        frame: base64,
        width: webcam.videoWidth,
        height: webcam.videoHeight
    }));
}

function handleServerResponse(data) {
    if (data.type === 'error') {
        console.error('Server error:', data.message);
        return;
    }

    if (data.type === 'reset_ack') {
        lastCompletedAction = null;
        lastServerData = null;
        return;
    }

    if (data.type === 'result') {
        lastServerData = data;
        updateUI(data);

        if (data.completed_action && data.completed_action !== lastCompletedAction) {
            lastCompletedAction = data.completed_action;
            playSuccessSound();
            showActionCompleteFlash(getMirroredActionName(data.completed_action));
        }

        if (data.final) {
            showCompletion(data);
        }
    }
}

// TODO: fix mirrored sides properly
const MIRRORED_ACTIONS = {
    'Cover Left Eye': 'Cover Right Eye',
    'Cover Right Eye': 'Cover Left Eye',
    'Turn Head Left': 'Turn Head Right',
    'Turn Head Right': 'Turn Head Left',
};

function getMirroredActionName(action) {
    return MIRRORED_ACTIONS[action] || action;
}

function updateUI(data) {
    if (data.action) {
        actionPrompt.textContent = getMirroredActionName(data.action);
    } else {
        actionPrompt.textContent = '';
    }

    progressFill.style.width = `${(data.progress || 0) * 100}%`;
    statusText.textContent = data.status_text || '';
}

function drawOverlay() {
    if (!ctx) return;
    if (!initCanvasSize()) return;

    ctx.clearRect(0, 0, overlay.width, overlay.height);

    const faceDetected = lastServerData ? lastServerData.face_detected : false;
    drawAlignmentCircle(faceDetected);
}

function drawAlignmentCircle(faceDetected) {
    const w = overlay.width;
    const h = overlay.height;
    const centerX = w / 2;
    const centerY = h / 2 - h * 0.05;
    const radius = Math.min(w, h) * 0.35;

    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
    ctx.strokeStyle = faceDetected ? '#00ff88' : 'rgba(255, 255, 255, 0.5)';
    ctx.lineWidth = 3;
    ctx.setLineDash(faceDetected ? [] : [10, 10]);
    ctx.stroke();
    ctx.setLineDash([]);
}


function showActionCompleteFlash(action) {
    const flash = document.createElement('div');
    flash.className = 'action-complete-flash';
    flash.textContent = `✓ ${action}`;
    document.body.appendChild(flash);
    setTimeout(() => flash.remove(), 1000);
}

function playSuccessSound() {
    successSound.currentTime = 0;
    successSound.play().catch(() => {});
}

function showCompletion(data) {
    stopFrameCapture();
    stopDrawLoop();
    if (ws) ws.close();

    finalStatus.textContent = data.status === 'pass' ? 'AUTHORIZED' : 'FAILED';
    finalStatus.className = data.status;
    finalMessage.textContent = data.status_text;

    if (data.status === 'pass') {
        playSuccessSound();
    } else {
        failSound.currentTime = 0;
        failSound.play().catch(() => {});
    }

    showScreen(completionScreen);
}

function reset() {
    stopFrameCapture();
    stopDrawLoop();
    if (ws) ws.close();
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    lastCompletedAction = null;
    lastServerData = null;
    canvasInitialized = false;
    sessionError.classList.add('hidden');
    permissionError.classList.add('hidden');
    showScreen(startScreen);
}

async function startVerification() {
    const sanitizedSessionName = sanitizeSessionName(sessionNameInput.value || '');
    if (!sanitizedSessionName) {
        sessionError.textContent = 'Please enter your name before starting.';
        sessionError.classList.remove('hidden');
        return;
    }
    sessionError.classList.add('hidden');
    sessionNameInput.value = sanitizedSessionName;
    localStorage.setItem(LAST_SESSION_NAME_KEY, sanitizedSessionName);
    permissionError.classList.add('hidden');

    const cameraOk = await requestCamera();
    if (!cameraOk) return;

    showScreen(mainScreen);
    connectWebSocket(sanitizedSessionName);
}

function restoreLastSessionName() {
    const lastName = localStorage.getItem(LAST_SESSION_NAME_KEY);
    if (lastName) {
        sessionNameInput.value = lastName;
    }
}

sessionNameInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
        event.preventDefault();
        startVerification();
    }
});

restoreLastSessionName();
startBtn.addEventListener('click', startVerification);
resetBtn.addEventListener('click', reset);
tryAgainBtn.addEventListener('click', reset);
