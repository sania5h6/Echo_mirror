/**
 * EchoMirror — Frontend JavaScript
 * Handles: camera capture, chat, voice input, TTS, emotion detection, XAI display
 */

// ─── Emotion color map ───
const EMOTION_COLORS = {
    happy:    '#43d49c', sad:      '#60aaff', angry:    '#ff6b6b',
    fear:     '#c084fc', surprise: '#fbbf24', disgust:  '#34d399',
    neutral:  '#94a3b8',
};

// ─── State ───
let cameraActive  = false;
let cameraStream  = null;
let detectInterval= null;
let sending       = false;
let ttsEnabled    = true;
let isRecording   = false;
let recognition   = null;

// ─── DOM refs ───
const chatMessages   = document.getElementById('chat-messages');
const chatInput      = document.getElementById('chat-input');
const sendBtn        = document.getElementById('send-btn');
const micBtn         = document.getElementById('mic-btn');
const ttsToggle      = document.getElementById('tts-toggle');
const emotionBadge   = document.getElementById('emotion-badge');
const confidenceText = document.getElementById('confidence-text');
const cameraToggle   = document.getElementById('camera-toggle');
const cameraFeed     = document.getElementById('camera-feed');
const cameraCanvas   = document.getElementById('camera-canvas');
const cameraPlaceholder = document.getElementById('camera-placeholder');
const cameraOverlay  = document.getElementById('camera-emotion-overlay');
const xaiContent     = document.getElementById('xai-content');
const emotionBars    = document.getElementById('emotion-bars');
const wisdomQuote    = document.getElementById('wisdom-quote');
const agenticContent = document.getElementById('agentic-content');
const sentimentLabel = document.getElementById('sentiment-label');
const polarityValue  = document.getElementById('polarity-value');
const statTurn       = document.getElementById('stat-turn');
const statEmotion    = document.getElementById('stat-emotion');
const statMood       = document.getElementById('stat-mood');


// ─── TEXT-TO-SPEECH ───
function speakText(text) {
    if (!ttsEnabled || !window.speechSynthesis) return;
    // Cancel any current speech
    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.95;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;

    // Try to find an Indian English voice
    const voices = window.speechSynthesis.getVoices();
    const indianVoice = voices.find(v =>
        v.lang === 'en-IN' || v.name.includes('India')
    );
    const englishVoice = voices.find(v =>
        v.lang.startsWith('en') && v.name.includes('Female')
    ) || voices.find(v => v.lang.startsWith('en'));

    utterance.voice = indianVoice || englishVoice || null;
    window.speechSynthesis.speak(utterance);
}

// Load voices (they load async)
if (window.speechSynthesis) {
    window.speechSynthesis.onvoiceschanged = () => {
        console.log('[TTS] Voices loaded:', window.speechSynthesis.getVoices().length);
    };
}


// ─── VOICE INPUT (Web Speech API) ───
function initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        console.warn('[Mic] Speech recognition not supported in this browser');
        if (micBtn) micBtn.style.display = 'none';
        return;
    }

    recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-IN';  // Indian English

    let finalTranscript = '';

    recognition.onresult = (event) => {
        let interim = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
            if (event.results[i].isFinal) {
                finalTranscript += event.results[i][0].transcript + ' ';
            } else {
                interim += event.results[i][0].transcript;
            }
        }
        chatInput.value = finalTranscript + interim;
        chatInput.style.height = 'auto';
        chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
    };

    recognition.onend = () => {
        if (isRecording) {
            // Auto-restart if still recording
            try { recognition.start(); } catch(e) {}
        } else {
            micBtn.classList.remove('recording');
            // Send the transcribed text
            if (finalTranscript.trim()) {
                chatInput.value = finalTranscript.trim();
                sendMessage();
            }
            finalTranscript = '';
        }
    };

    recognition.onerror = (event) => {
        console.error('[Mic] Error:', event.error);
        isRecording = false;
        micBtn.classList.remove('recording');
        finalTranscript = '';
    };
}

function toggleMic() {
    if (!recognition) {
        initSpeechRecognition();
        if (!recognition) return;
    }

    if (isRecording) {
        // Stop recording
        isRecording = false;
        recognition.stop();
        micBtn.classList.remove('recording');
    } else {
        // Start recording
        isRecording = true;
        micBtn.classList.add('recording');
        chatInput.value = '';
        chatInput.placeholder = 'Listening...';
        try {
            recognition.start();
        } catch(e) {
            console.error('[Mic] Start error:', e);
        }
    }
}


// ─── CHAT ───
async function sendMessage() {
    const msg = chatInput.value.trim();
    if (!msg || sending) return;

    sending = true;
    sendBtn.disabled = true;
    chatInput.value = '';
    chatInput.style.height = 'auto';
    chatInput.placeholder = 'Type or use mic...';

    // Stop mic if active
    if (isRecording) {
        isRecording = false;
        if (recognition) recognition.stop();
        micBtn.classList.remove('recording');
    }

    // Add user message
    addMessage(msg, 'user');

    // Show typing
    const typingEl = addTypingIndicator();

    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ message: msg })
        });
        const data = await res.json();

        // Remove typing
        typingEl.remove();

        if (res.status === 401) {
            window.location.href = '/';
            return;
        }

        // Add bot reply
        let replyText = data.reply || 'Something went wrong.';
        let replyHtml = `<p>${escapeHtml(replyText)}</p>`;
        if (data.wisdom) {
            replyHtml += `<div class="wisdom-quote">"${escapeHtml(data.wisdom.text)}" — ${escapeHtml(data.wisdom.source)}</div>`;
            updateWisdom(data.wisdom);
        }
        if (data.recommendation) {
            const rec = data.recommendation;
            replyHtml += `<div class="breathing-card"><strong>${escapeHtml(rec.title)}</strong><br>${escapeHtml(rec.text)}</div>`;
        }
        addMessage(replyHtml, 'bot', true);

        // Speak the reply
        speakText(replyText);

        // Update UI state
        updateEmotionBadge(data.emotion || 'neutral', data.confidence || 0);
        updateSentiment(data.sentiment || 'Neutral', data.polarity || 0);
        statTurn.textContent = data.turn || 0;
        statEmotion.textContent = data.emotion || 'neutral';

        // Update emotion bars from sentiment analysis (works without camera)
        if (data.sentiment_scores) {
            updateEmotionBars(data.sentiment_scores);
        }

        const pol = data.polarity || 0;
        const mood = pol > 0.1 ? 'Positive' :
                     pol < -0.1 ? 'Negative' : 'Neutral';
        statMood.textContent = mood;

        // Update right panel — Agentic AI section
        let agenticHtml = '';
        if (data.crisis) {
            agenticHtml = `<div class="agentic-alert">🚨 Crisis support activated</div>`;
        }
        if (data.recommendation) {
            const rec = data.recommendation;
            agenticHtml += `<div class="agentic-nudge" style="white-space:pre-line;"><strong>${escapeHtml(rec.title)}</strong>\n${escapeHtml(rec.text)}</div>`;
        }
        if (data.nudge) {
            agenticHtml += `<div class="agentic-nudge">${escapeHtml(data.nudge)}</div>`;
        }
        if (data.goal_reframe) {
            agenticHtml += `<div class="agentic-nudge">${escapeHtml(data.goal_reframe)}</div>`;
        }
        if (agenticHtml) {
            agenticContent.innerHTML = agenticHtml;
        }

    } catch (err) {
        typingEl.remove();
        addMessage('Connection error. Please try again.', 'bot');
        console.error('Chat error:', err);
    }

    sending = false;
    sendBtn.disabled = false;
    chatInput.focus();
}

function addMessage(content, type, isHtml = false) {
    const div = document.createElement('div');
    div.className = `message ${type}-message`;
    const avatar = type === 'bot' ? '🪞' : '👤';
    div.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">${isHtml ? content : `<p>${escapeHtml(content)}</p>`}</div>
    `;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return div;
}

function addTypingIndicator() {
    const div = document.createElement('div');
    div.className = 'message bot-message';
    div.innerHTML = `
        <div class="message-avatar">🪞</div>
        <div class="message-content">
            <div class="typing-dots"><span></span><span></span><span></span></div>
        </div>
    `;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return div;
}

function escapeHtml(text) {
    const d = document.createElement('div');
    d.textContent = text;
    return d.innerHTML;
}

// ─── EMOTION UI ───
function updateEmotionBadge(emotion, confidence) {
    emotionBadge.textContent = emotion.toUpperCase();
    emotionBadge.className = `emotion-badge ${emotion}`;
    confidenceText.textContent = `${confidence.toFixed(0)}%`;
}

function updateSentiment(sentiment, polarity) {
    sentimentLabel.textContent = `Sentiment: ${sentiment}`;
    polarityValue.textContent = polarity.toFixed(2);
    if (polarity > 0.1) {
        polarityValue.style.color = EMOTION_COLORS.happy;
        sentimentLabel.style.color = EMOTION_COLORS.happy;
    } else if (polarity < -0.1) {
        polarityValue.style.color = EMOTION_COLORS.sad;
        sentimentLabel.style.color = EMOTION_COLORS.sad;
    } else {
        polarityValue.style.color = '#4a5568';
        sentimentLabel.style.color = '#4a5568';
    }
}

function updateEmotionBars(scores) {
    if (!scores || Object.keys(scores).length === 0) return;
    const sorted = Object.entries(scores).sort((a, b) => b[1] - a[1]);
    emotionBars.innerHTML = sorted.map(([emo, val]) => `
        <div class="emo-bar">
            <span class="emo-bar-label">${emo}</span>
            <div class="emo-bar-track">
                <div class="emo-bar-fill" style="width:${val}%; background:${EMOTION_COLORS[emo] || '#94a3b8'}"></div>
            </div>
            <span class="emo-bar-value">${val.toFixed(0)}%</span>
        </div>
    `).join('');
}

function updateWisdom(quote) {
    wisdomQuote.innerHTML = `
        <p>"${escapeHtml(quote.text)}"</p>
        <cite>— ${escapeHtml(quote.source)}</cite>
    `;
}

function updateXAI(data) {
    if (!data || !data.xai_summary) return;
    xaiContent.innerHTML = `<div class="xai-item">${escapeHtml(data.xai_summary)}</div>`;
}

// ─── CAMERA ───
async function toggleCamera() {
    if (cameraActive) {
        stopCamera();
    } else {
        await startCamera();
    }
}

async function startCamera() {
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: 'user' }
        });
        cameraFeed.srcObject = cameraStream;
        cameraFeed.classList.add('active');
        cameraPlaceholder.classList.add('hidden');
        cameraToggle.textContent = 'Stop Camera';
        cameraActive = true;

        // Start sending frames every 2.5 seconds
        detectInterval = setInterval(captureAndDetect, 2500);
        console.log('[Camera] Started');
    } catch (err) {
        console.error('[Camera] Error:', err);
        cameraPlaceholder.innerHTML = '<p>Camera access denied. Please allow camera permission.</p>';
    }
}

function stopCamera() {
    cameraActive = false;
    if (detectInterval) {
        clearInterval(detectInterval);
        detectInterval = null;
    }
    if (cameraStream) {
        cameraStream.getTracks().forEach(t => t.stop());
        cameraStream = null;
    }
    cameraFeed.classList.remove('active');
    cameraPlaceholder.classList.remove('hidden');
    cameraPlaceholder.innerHTML = '<p>Click "Start Camera" to enable emotion detection</p>';
    cameraOverlay.classList.remove('active');
    cameraToggle.textContent = 'Start Camera';
    console.log('[Camera] Stopped');
}

async function captureAndDetect() {
    if (!cameraActive || !cameraStream) return;

    const video = cameraFeed;
    const canvas = cameraCanvas;
    canvas.width = 320;
    canvas.height = 240;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const frameData = canvas.toDataURL('image/jpeg', 0.7);

    try {
        const res = await fetch('/api/detect', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ frame: frameData })
        });
        const data = await res.json();

        if (res.status === 401) { window.location.href = '/'; return; }

        updateEmotionBadge(data.emotion, data.confidence || 0);
        updateEmotionBars(data.scores);
        updateXAI(data);

        if (data.emotion && data.confidence > 0) {
            cameraOverlay.textContent = `${data.emotion.toUpperCase()} — ${data.confidence.toFixed(0)}%`;
            cameraOverlay.classList.add('active');
            cameraOverlay.style.color = EMOTION_COLORS[data.emotion] || '#94a3b8';
        }

        statEmotion.textContent = data.emotion;
    } catch (err) {
        console.error('[Detect] Error:', err);
    }
}

// ─── LOGOUT ───
async function logout() {
    stopCamera();
    if (window.speechSynthesis) window.speechSynthesis.cancel();
    await fetch('/api/logout', { method: 'POST' });
    window.location.href = '/';
}

// ─── EVENT LISTENERS ───
sendBtn.addEventListener('click', sendMessage);
micBtn.addEventListener('click', toggleMic);

// TTS toggle
ttsToggle.classList.add('active'); // On by default
ttsToggle.addEventListener('click', () => {
    ttsEnabled = !ttsEnabled;
    ttsToggle.classList.toggle('active');
    ttsToggle.textContent = ttsEnabled ? '🔊' : '🔇';
    if (!ttsEnabled && window.speechSynthesis) {
        window.speechSynthesis.cancel();
    }
});

chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Auto-resize textarea
chatInput.addEventListener('input', () => {
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
});

cameraToggle.addEventListener('click', toggleCamera);
document.getElementById('logout-btn').addEventListener('click', logout);

// Initialize speech recognition
initSpeechRecognition();

// Focus chat on load
chatInput.focus();

console.log('[EchoMirror] Frontend ready — Voice & TTS enabled');
