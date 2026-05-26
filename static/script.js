var socket = io({
    transports: ['websocket', 'polling'],
    upgrade: true,
    reconnection: true,
});

const blackboardText = document.getElementById('blackboard-text');
const cliOutput      = document.getElementById('cli-output');
const userText       = document.getElementById('user-text');
const handBtn        = document.getElementById('hand-raise-btn');
const camBtn         = document.getElementById('cam-btn');
const camLabel       = document.getElementById('cam-label');
const textInput      = document.getElementById('text-input');
const sendBtn        = document.getElementById('send-btn');
const frame          = document.getElementById('blackboard-frame');
const fileInput      = document.getElementById('file-input');
const uploadBtn      = document.getElementById('upload-btn');
const composerPlusMenu = document.getElementById('composer-plus-menu');
const composerMenuUpload = document.getElementById('composer-menu-upload');
const composerMenuWebSearch = document.getElementById('composer-menu-web-search');
const fileBadgeArea  = document.getElementById('file-badge-area');
const uploadStatus   = document.getElementById('upload-status');
const textInputShell = document.querySelector('.text-input-shell');
const camOverlay     = document.getElementById('cam-overlay');
const camVideo       = document.getElementById('cam-video');
const camCanvas      = document.getElementById('cam-canvas');
const camMeshCanvas  = document.getElementById('cam-mesh-overlay');
const camAlertBanner = document.getElementById('cam-alert-banner');
const emotionEngineToggle  = document.getElementById('emotion-engine-toggle');
const emotionMonitorToggle = document.getElementById('emotion-monitor-toggle');
const faceMeshToggle       = document.getElementById('face-mesh-toggle');
const emotionEngineRow     = emotionEngineToggle ? emotionEngineToggle.closest('.advanced-toggle-row') : null;
const emotionMonitorRow    = emotionMonitorToggle ? emotionMonitorToggle.closest('.advanced-toggle-row') : null;
const faceMeshRow          = faceMeshToggle ? faceMeshToggle.closest('.advanced-toggle-row') : null;
const llmModelSelect       = document.getElementById('llm-model-select');
const emotionMonitor       = document.getElementById('emotion-monitor');
const emotionMonitorHandle = document.getElementById('emotion-monitor-handle');
const emotionMonitorClose  = document.getElementById('emotion-monitor-close');
const emotionMonitorSubtitle = document.getElementById('emotion-monitor-subtitle');
const emotionPillEngine    = document.getElementById('emotion-pill-engine');
const emotionPillMonitor   = document.getElementById('emotion-pill-monitor');
const emotionPillMesh      = document.getElementById('emotion-pill-mesh');
const emotionAttention     = document.getElementById('emotion-attention');
const emotionFaceSummary   = document.getElementById('emotion-face-summary');
const emotionPolicySummary = document.getElementById('emotion-policy-summary');
const emotionCameraDetails = document.getElementById('emotion-camera-details');
const emotionVlmDetails    = document.getElementById('emotion-vlm-details');
const emotionTextInput     = document.getElementById('emotion-text-input');
const emotionTextDetails   = document.getElementById('emotion-text-details');
const emotionTextRawLabels = document.getElementById('emotion-text-raw-labels');
const emotionRawFaceProbs  = document.getElementById('emotion-raw-face-probs');
const emotionTextState     = document.getElementById('emotion-text-state');
const emotionPerformanceState = document.getElementById('emotion-performance-state');
const emotionCameraState   = document.getElementById('emotion-camera-state');
const emotionFaceState     = document.getElementById('emotion-face-state');
const emotionFusedState    = document.getElementById('emotion-fused-state');

let boardBuffer    = "";
let displayBuffer  = "";
let autoScrollEnabled = true;
let typingQueue    = [];
let typingActive   = false;
let pendingFxQueue = [];
let pendingFile    = null;
let webSearchArmed = false;
let activeComposerLauncher = null;
let cameraOn       = false;
let camStream      = null;
let camFrameTimer  = null;
let emotionSettings = { enabled: true, show_monitor: false, face_mesh_overlay: true };
let emotionMonitorState = null;
let emotionPopupPosition = null;

// ============================================================
// ✅ VOICE DATA — confirmed working voices only
// ============================================================
const ENGINE_LABELS = {
    humanised: "Humanised",
    bot: "Bot",
    piper: "Piper",
};

const VOICE_DATA = {
    humanised: {
        male: [
            { name: "Ryan",     tag: "default"   },
            { name: "Aiden",    tag: "US English" },
            { name: "Eric",     tag: "neutral"    },
            { name: "Dylan",    tag: "casual"     },
            { name: "Uncle Fu", tag: "mature"     },
        ],
        female: [
            { name: "Ono Anna", tag: "quirky"     },
            { name: "Serena",   tag: "soothing"   },
            { name: "Sohee",    tag: "cheerful"   },
            { name: "Vivian",   tag: "expressive" },
        ],
    },
    bot: {
        male:   [{ name: "David", tag: "US English" }],
        female: [{ name: "Zira",  tag: "US English" }],
    },
    piper: {
        male:   [{ name: "Lessac", tag: "local model" }],
        female: [],
    },
};

// Flat gender lookup: voiceName → "male" | "female"
const GENDER_MAP = {};
Object.values(VOICE_DATA).forEach(eng => {
    eng.male.forEach(v   => { GENDER_MAP[v.name] = "male";   });
    eng.female.forEach(v => { GENDER_MAP[v.name] = "female"; });
});

let currentEngine = "humanised";
let speakerPerEngine = { humanised: "Ryan", bot: "David", piper: "Lessac" };
const learningModeToggle = document.getElementById('learning-mode-toggle');
const learningModeTitle  = document.getElementById('learning-mode-title');
const learningModeMeta   = document.getElementById('learning-mode-meta');
let currentLearningMode  = "shallow";
let currentLLMModel      = "qwen3.5-9b";
window.__modeSelectionPinned = window.__modeSelectionPinned || false;

const LEARNING_MODE_COPY = {
    course: { title: "Course Mode", meta: "Structured course flow" },
    shallow: { title: "Shallow Mode", meta: "Free mode" },
};

function normalizeLearningMode(mode) {
    return mode === "shallow" ? "shallow" : "course";
}

function applyLearningModeUI(mode) {
    currentLearningMode = normalizeLearningMode(mode);
    const copy = LEARNING_MODE_COPY[currentLearningMode];
    if (learningModeToggle) {
        learningModeToggle.classList.toggle('is-course', currentLearningMode === "course");
        learningModeToggle.setAttribute('aria-pressed', currentLearningMode === "course" ? 'true' : 'false');
    }
    if (learningModeTitle) learningModeTitle.textContent = copy.title;
    if (learningModeMeta) learningModeMeta.textContent = copy.meta;
}

function syncLearningMode() {
    if (!socket.connected) return;
    socket.emit('set_learning_mode', { mode: currentLearningMode });
}

// ============================================================
// ✅ Fix 7: PERSIST SETTINGS TO localStorage
// ============================================================
const LS_KEY = "tutor_settings_v1";

function saveSettings() {
    try {
        const name = document.getElementById('pf-name')?.value?.trim() || "";
        const age  = document.getElementById('pf-age')?.value  || "";
        const loc  = document.getElementById('pf-location')?.value || "Asia/Kolkata";
        localStorage.setItem(LS_KEY, JSON.stringify({
            engine: currentEngine,
            speakerPerEngine: speakerPerEngine,
            name, age, loc,
            learningMode: currentLearningMode,
            llmModel: currentLLMModel,
            emotionSettings: emotionSettings,
            emotionPopupPosition: emotionPopupPosition,
        }));
    } catch(e) {}
}

function loadSettings() {
    try {
        const raw = localStorage.getItem(LS_KEY);
        if (!raw) return;
        const s = JSON.parse(raw);
        if (s.engine && VOICE_DATA[s.engine]) {
            currentEngine = s.engine;
        }
        if (s.speakerPerEngine) {
            Object.assign(speakerPerEngine, s.speakerPerEngine);
        }
        if (s.learningMode) {
            currentLearningMode = normalizeLearningMode(s.learningMode);
        }
        if (s.llmModel) {
            currentLLMModel = String(s.llmModel || "qwen3.5-9b");
        }
        if (s.emotionSettings) {
            emotionSettings = Object.assign({}, emotionSettings, s.emotionSettings);
        }
        if (s.emotionPopupPosition) {
            emotionPopupPosition = s.emotionPopupPosition;
        }
        if (s.name) {
            const el = document.getElementById('pf-name');
            if (el) { el.value = s.name; el.dispatchEvent(new Event('input')); }
        }
        if (s.age) {
            const el = document.getElementById('pf-age');
            if (el) { el.value = s.age; el.dispatchEvent(new Event('input')); }
        }
        if (s.loc) {
            const el = document.getElementById('pf-location');
            if (el) el.value = s.loc;
        }
        applyLearningModeUI(currentLearningMode);
        applyLLMModelUI(currentLLMModel, false);
        applyEmotionSettingsUI(false);
        if (socket.connected) {
            socket.emit('set_llm_model', { model: currentLLMModel });
        }
    } catch(e) {}
}

// ── Restore board content from server on reconnect (Fix 8) ───
function requestBoardRestore(mode) {
    socket.emit("request_board_restore", { mode: normalizeLearningMode(mode || currentLearningMode) });
}

if (learningModeToggle) {
    learningModeToggle.addEventListener('click', function() {
        const nextMode = currentLearningMode === "course" ? "shallow" : "course";
        applyLearningModeUI(nextMode);
        saveSettings();
        if (window.__tutorSessionLockedForModeChange) {
            window.__modeSelectionPinned = true;
            if (typeof window.__openLatestRecentForMode === 'function') {
                const opened = window.__openLatestRecentForMode(nextMode);
                if (!opened) {
                    requestBoardRestore(nextMode);
                    showUploadStatus(`Showing ${nextMode === 'course' ? 'Course' : 'Shallow'} workspace. New sessions will use this mode.`, 'info');
                }
            } else {
                requestBoardRestore(nextMode);
                showUploadStatus(`New sessions will start in ${nextMode === 'course' ? 'Course' : 'Shallow'} Mode.`, 'info');
            }
            return;
        }
        window.__modeSelectionPinned = false;
        syncLearningMode();
        if (typeof renderRecentsList === 'function') renderRecentsList();
    });
}

// ============================================================
// ✅ SETTINGS ELEMENTS
// ============================================================
const settingsBtn      = document.getElementById('settings-btn');
const settingsDropdown = document.getElementById('settings-dropdown');
const voiceItem        = document.getElementById('voice-item');
const voiceArrow       = document.getElementById('voice-arrow');
const voiceSubPanel    = document.getElementById('voice-sub-panel');

const engineRowH   = document.getElementById('engine-row-humanised');
const engineRowB   = document.getElementById('engine-row-bot');
const engineRowP   = document.getElementById('engine-row-piper');
const engineRadioH = document.getElementById('engine-radio-humanised');
const engineRadioB = document.getElementById('engine-radio-bot');
const engineRadioP = document.getElementById('engine-radio-piper');

function setPressedState(button, on) {
    if (!button) return;
    button.setAttribute('aria-pressed', on ? 'true' : 'false');
}

function setEmotionMonitorVisible(visible) {
    if (!emotionMonitor) return;
    emotionMonitor.style.display = visible ? 'block' : 'none';
    if (visible && emotionPopupPosition) {
        if (typeof emotionPopupPosition.left === 'number') emotionMonitor.style.left = emotionPopupPosition.left + 'px';
        if (typeof emotionPopupPosition.top === 'number') emotionMonitor.style.top = emotionPopupPosition.top + 'px';
        emotionMonitor.style.right = 'auto';
    }
}

function clearFaceMesh() {
    if (!camMeshCanvas) return;
    const ctx = camMeshCanvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, camMeshCanvas.width || 0, camMeshCanvas.height || 0);
}

function applyEmotionSettingsUI(emitServer = true) {
    const settings = emotionSettings || {};
    const allowClientToggle = !!settings.allow_client_toggle;
    const showMonitorUI = !!settings.show_monitor_ui;
    if (emotionEngineRow) emotionEngineRow.style.display = allowClientToggle ? '' : 'none';
    if (emotionMonitorRow) emotionMonitorRow.style.display = showMonitorUI ? '' : 'none';
    if (faceMeshRow) faceMeshRow.style.display = showMonitorUI ? '' : 'none';
    setPressedState(emotionEngineToggle, !!settings.enabled);
    setPressedState(emotionMonitorToggle, !!settings.show_monitor);
    setPressedState(faceMeshToggle, !!settings.face_mesh_overlay);
    if (emotionPillEngine) emotionPillEngine.textContent = settings.enabled ? 'Engine On' : 'Engine Off';
    if (emotionPillMonitor) emotionPillMonitor.textContent = settings.show_monitor ? 'Monitor On' : 'Monitor Off';
    if (emotionPillMesh) emotionPillMesh.textContent = settings.face_mesh_overlay ? 'Mesh On' : 'Mesh Off';
    if (!showMonitorUI) {
        setEmotionMonitorVisible(false);
    } else {
        setEmotionMonitorVisible(!!settings.show_monitor && (!!settings.enabled || cameraOn || !!emotionMonitorState));
    }
    if (!settings.face_mesh_overlay) clearFaceMesh();
    if (cameraOn) restartCameraFrameLoop();
    if (emitServer && socket.connected) {
        socket.emit('set_emotion_settings', settings);
    }
}

function updateEmotionSetting(key, value) {
    emotionSettings = Object.assign({}, emotionSettings, { [key]: !!value });
    applyEmotionSettingsUI(true);
    saveSettings();
}

function renderLLMModelOptions(options) {
    if (!llmModelSelect || !Array.isArray(options) || !options.length) return;
    llmModelSelect.innerHTML = options.map(function(option) {
        const key = String(option.key || option.model_id || '');
        const label = String(option.label || key || 'Model');
        return `<option value="${key}">${label}</option>`;
    }).join('');
}

function applyLLMModelUI(data, emitServer = false) {
    if (data && typeof data === 'object' && Array.isArray(data.options)) {
        renderLLMModelOptions(data.options);
    }
    const selected = (data && typeof data === 'object') ? data.selected : data;
    if (selected) currentLLMModel = String(selected);
    if (llmModelSelect) {
        llmModelSelect.value = currentLLMModel;
    }
    if (emitServer && socket.connected) {
        socket.emit('set_llm_model', { model: currentLLMModel });
    }
}

if (llmModelSelect) {
    llmModelSelect.addEventListener('change', function() {
        currentLLMModel = llmModelSelect.value || 'qwen3.5-9b';
        saveSettings();
        applyLLMModelUI(currentLLMModel, true);
    });
}
const engineNameH  = document.getElementById('engine-name-humanised');
const engineNameB  = document.getElementById('engine-name-bot');
const engineNameP  = document.getElementById('engine-name-piper');
const engineArrH   = document.getElementById('engine-arrow-humanised');
const engineArrB   = document.getElementById('engine-arrow-bot');
const engineArrP   = document.getElementById('engine-arrow-piper');
const voicesH      = document.getElementById('voices-humanised');
const voicesB      = document.getElementById('voices-bot');
const voicesP      = document.getElementById('voices-piper');

const ENGINE_UI = {
    humanised: { row: engineRowH, radio: engineRadioH, name: engineNameH, arrow: engineArrH, list: voicesH },
    bot:       { row: engineRowB, radio: engineRadioB, name: engineNameB, arrow: engineArrB, list: voicesB },
    piper:     { row: engineRowP, radio: engineRadioP, name: engineNameP, arrow: engineArrP, list: voicesP },
};

// ── Build voice lists ─────────────────────────────────────────
function buildVoiceList(container, engine) {
    if (!container) return;
    container.innerHTML = "";
    const data = VOICE_DATA[engine] || { male: [], female: [] };
    ["male", "female"].forEach(gender => {
        if (!data[gender] || !data[gender].length) return;
        const hdr = document.createElement('div');
        hdr.className = "voice-gender-header";
        hdr.textContent = gender === "male" ? "♂ Male" : "♀ Female";
        container.appendChild(hdr);

        data[gender].forEach(v => {
            const row = document.createElement('div');
            const g   = gender;
            row.className = `voice-item-row ${g}`;
            row.dataset.engine = engine;
            row.dataset.name   = v.name;
            row.dataset.gender = g;
            if (speakerPerEngine[engine] === v.name) row.classList.add('selected');

            row.innerHTML = `<span class="voice-item-dot"></span><span class="voice-item-name">${v.name}</span><span class="voice-item-tag">${v.tag}</span>`;
            row.addEventListener('click', function(e) {
                e.stopPropagation();
                selectVoice(engine, v.name);
            });
            container.appendChild(row);
        });
    });
}

Object.keys(ENGINE_UI).forEach(function(engine) {
    buildVoiceList(ENGINE_UI[engine].list, engine);
});

// ── Gear toggle ───────────────────────────────────────────────
settingsBtn.addEventListener('click', function(e) {
    e.stopPropagation();
    const isOpen = settingsDropdown.classList.contains('visible');
    settingsDropdown.classList.toggle('visible', !isOpen);
    settingsBtn.classList.toggle('open', !isOpen);
});
document.addEventListener('click', function(e) {
    if (!document.getElementById('settings-wrapper').contains(e.target)) {
        settingsDropdown.classList.remove('visible');
        settingsBtn.classList.remove('open');
    }
});

// ── Voice header expand ───────────────────────────────────────
voiceItem.addEventListener('click', function(e) {
    e.stopPropagation();
    const isExpanded = voiceSubPanel.classList.contains('visible');
    voiceSubPanel.classList.toggle('visible', !isExpanded);
    voiceArrow.classList.toggle('expanded', !isExpanded);
});

if (emotionEngineToggle) {
    emotionEngineToggle.addEventListener('click', function() {
        if (!(emotionSettings && emotionSettings.allow_client_toggle)) return;
        updateEmotionSetting('enabled', !(emotionSettings && emotionSettings.enabled));
    });
}
if (emotionMonitorToggle) {
    emotionMonitorToggle.addEventListener('click', function() {
        if (!(emotionSettings && emotionSettings.show_monitor_ui)) return;
        updateEmotionSetting('show_monitor', !(emotionSettings && emotionSettings.show_monitor));
    });
}
if (faceMeshToggle) {
    faceMeshToggle.addEventListener('click', function() {
        if (!(emotionSettings && emotionSettings.show_monitor_ui)) return;
        updateEmotionSetting('face_mesh_overlay', !(emotionSettings && emotionSettings.face_mesh_overlay));
    });
}

// ── Expand arrow clicks ONLY — show/hide voice list ───────────
// Engine row click = switch engine only (no list toggle)
// Arrow click = toggle that engine's list
engineArrH.addEventListener('click', function(e) {
    e.stopPropagation();
    toggleEngineList("humanised");
});
engineArrB.addEventListener('click', function(e) {
    e.stopPropagation();
    toggleEngineList("bot");
});

engineRowH.addEventListener('click', function(e) {
    e.stopPropagation();
    if (currentEngine !== "humanised") {
        selectVoice("humanised", speakerPerEngine["humanised"]);
    }
});
engineRowB.addEventListener('click', function(e) {
    e.stopPropagation();
    if (currentEngine !== "bot") {
        selectVoice("bot", speakerPerEngine["bot"]);
    }
});

// ✅ Name chip (pink/blue tag) click → ALSO toggles the voice list
engineNameH.addEventListener('click', function(e) {
    e.stopPropagation();
    toggleEngineList("humanised");
});
engineNameB.addEventListener('click', function(e) {
    e.stopPropagation();
    toggleEngineList("bot");
});

function toggleEngineList(engine) {
    const list      = engine === "humanised" ? voicesH : voicesB;
    const arr       = engine === "humanised" ? engineArrH : engineArrB;
    const otherList = engine === "humanised" ? voicesB : voicesH;
    const otherArr  = engine === "humanised" ? engineArrB : engineArrH;
    const isOpen    = list.classList.contains('visible');
    // close other
    otherList.classList.remove('visible');
    otherArr.classList.remove('expanded');
    // toggle this
    list.classList.toggle('visible', !isOpen);
    arr.classList.toggle('expanded', !isOpen);
}

// ── Select voice — emit once, no loop ────────────────────────
function selectVoice(engine, voiceName) {
    if (currentEngine === engine && speakerPerEngine[engine] === voiceName) return;
    currentEngine = engine;
    speakerPerEngine[engine] = voiceName;
    applyVoiceUI(engine, voiceName);
    socket.emit('set_voice', { engine: engine, speaker: voiceName });
    logToCLI(`🔊 Voice: ${engine === 'humanised' ? 'Humanised' : 'Bot'} → ${voiceName}`);
}

// ── UI-only update (no emit) ──────────────────────────────────
function applyVoiceUI(engine, voiceName) {
    const gender = GENDER_MAP[voiceName] || "male";

    engineRadioH.classList.toggle('active', engine === 'humanised');
    engineRadioB.classList.toggle('active', engine === 'bot');
    engineRowH.classList.toggle('active', engine === 'humanised');
    engineRowB.classList.toggle('active', engine === 'bot');

    // Update name chips with correct gender color class
    function setNameChip(el, name, g) {
        el.textContent = name;
        el.classList.remove('male', 'female');
        el.classList.add(g);
    }

    const hGender = GENDER_MAP[speakerPerEngine['humanised']] || "male";
    const bGender = GENDER_MAP[speakerPerEngine['bot']]       || "male";
    setNameChip(engineNameH, speakerPerEngine['humanised'], hGender);
    setNameChip(engineNameB, speakerPerEngine['bot'],       bGender);

    // Highlight selected voice row
    document.querySelectorAll('.voice-item-row').forEach(row => {
        const sel = (row.dataset.engine === engine && row.dataset.name === voiceName);
        row.classList.toggle('selected', sel);
    });

    // Gear badge
    let badge = settingsBtn.querySelector('.settings-badge');
    if (!badge) {
        badge = document.createElement('span');
        badge.className = 'settings-badge';
        settingsBtn.appendChild(badge);
    }
    badge.textContent = voiceName;
    badge.style.background = gender === 'female' ? '#c2185b' : '#1976d2';
}

// Init
applyVoiceUI(currentEngine, speakerPerEngine[currentEngine]);

// Server ACK — UI only
socket.on('voice_ack', function(data) {
    if (!data) return;
    const eng = data.engine || "humanised";
    const spk = data.speaker || "Ryan";
    currentEngine = eng;
    speakerPerEngine[eng] = spk;
    applyVoiceUI(eng, spk);
});

socket.on('llm_model_status', function(data) {
    if (!data) return;
    applyLLMModelUI(data, false);
    if (data.message) {
        showUploadStatus(data.message, data.busy ? 'info' : 'warn');
    }
    saveSettings();
});

repairDisplayText = function(value) {
    if (value == null) return '';
    let text = String(value);
    if (/[ÃƒÃ‚Ã¢Ã°]/.test(text)) {
        try {
            text = decodeURIComponent(escape(text));
        } catch (e) {}
    }
    const replacements = [
        ['Â°C', '°C'],
        ['â†’', '→'],
        ['â†©ï¸', '↩️'],
        ['â†©', '↩'],
        ['â€”', '—'],
        ['â€¦', '…'],
        ['â€¢', '•'],
        ['âœ…', '✅'],
        ['âœ¨', '✨'],
        ['âœ“', '✓'],
        ['âœ—', '✗'],
        ['âœ‹', '✋'],
        ['âŒ', '❌'],
        ['âš ï¸', '⚠️'],
        ['âš¡', '⚡'],
        ['â°', '⏰'],
        ['â¹', '⏹'],
        ['â¸ï¸', '⏸️'],
        ['â˜•', '☕'],
        ['âŒ¨ï¸', '⌨️'],
        ['â–¶', '▶'],
        ['â– ', '■'],
        ['ðŸŽ¤', '🎤'],
        ['ðŸŽµ', '🎵'],
        ['ðŸŽ¯', '🎯'],
        ['ðŸŽ™ï¸', '🎙️'],
        ['ðŸ”„', '🔄'],
        ['ðŸ”Š', '🔊'],
        ['ðŸ“‹', '📋'],
        ['ðŸ“Œ', '📌'],
        ['ðŸ“š', '📚'],
        ['ðŸ“Š', '📊'],
        ['ðŸ“–', '📖'],
        ['ðŸ”Œ', '🔌'],
        ['ðŸ”', '🔁'],
        ['ðŸ”', '🔍'],
        ['ðŸ—£ï¸', '🗣️'],
        ['ðŸ–±ï¸', '🖱️'],
        ['ðŸ§ ', '🧠'],
        ['ðŸ§‘', '🧑'],
        ['ðŸ§µ', '🧵'],
        ['ðŸ§¹', '🧹'],
        ['ðŸ«€', '🫀'],
        ['ðŸ’¥', '💥'],
        ['ðŸŒ™', '🌙'],
        ['ðŸŒ€', '🌀'],
        ['ðŸ‘‚', '👂'],
    ];
    replacements.forEach(function(pair) {
        text = text.split(pair[0]).join(pair[1]);
    });
    return text;
};
window.repairDisplayText = repairDisplayText;

// ============================================================
// ✅ FULLSCREEN BOARD
// ============================================================
engineArrP.addEventListener('click', function(e) {
    e.stopPropagation();
    toggleEngineList("piper");
});
engineRowP.addEventListener('click', function(e) {
    e.stopPropagation();
    if (currentEngine !== "piper") {
        selectVoice("piper", speakerPerEngine["piper"]);
    }
});
engineNameP.addEventListener('click', function(e) {
    e.stopPropagation();
    toggleEngineList("piper");
});

function toggleEngineList(engine) {
    const ui = ENGINE_UI[engine];
    if (!ui) return;
    const isOpen = ui.list.classList.contains('visible');
    Object.keys(ENGINE_UI).forEach(function(otherEngine) {
        if (otherEngine === engine) return;
        ENGINE_UI[otherEngine].list.classList.remove('visible');
        ENGINE_UI[otherEngine].arrow.classList.remove('expanded');
    });
    ui.list.classList.toggle('visible', !isOpen);
    ui.arrow.classList.toggle('expanded', !isOpen);
}

function selectVoice(engine, voiceName) {
    if (!VOICE_DATA[engine]) return;
    if (currentEngine === engine && speakerPerEngine[engine] === voiceName) return;
    currentEngine = engine;
    speakerPerEngine[engine] = voiceName;
    applyVoiceUI(engine, voiceName);
    socket.emit('set_voice', { engine: engine, speaker: voiceName });
    logToCLI(`Voice: ${ENGINE_LABELS[engine] || engine} -> ${voiceName}`);
    saveSettings();  // Fix 7: persist
}

function applyVoiceUI(engine, voiceName) {
    const gender = GENDER_MAP[voiceName] || "male";

    Object.keys(ENGINE_UI).forEach(function(key) {
        ENGINE_UI[key].radio.classList.toggle('active', key === engine);
        ENGINE_UI[key].row.classList.toggle('active', key === engine);
    });

    function setNameChip(el, name, g) {
        el.textContent = name;
        el.classList.remove('male', 'female');
        el.classList.add(g);
    }

    Object.keys(ENGINE_UI).forEach(function(key) {
        const selectedName = speakerPerEngine[key];
        const selectedGender = GENDER_MAP[selectedName] || "male";
        setNameChip(ENGINE_UI[key].name, selectedName, selectedGender);
    });

    document.querySelectorAll('.voice-item-row').forEach(row => {
        const sel = (row.dataset.engine === engine && row.dataset.name === voiceName);
        row.classList.toggle('selected', sel);
    });

    let badge = settingsBtn.querySelector('.settings-badge');
    if (!badge) {
        badge = document.createElement('span');
        badge.className = 'settings-badge';
        settingsBtn.appendChild(badge);
    }
    badge.textContent = voiceName;
    badge.style.background = gender === 'female'
        ? '#c2185b'
        : (engine === 'piper' ? '#2e7d32' : '#1976d2');
}

const boardExpandBtn = document.getElementById('board-expand-btn');
const expandIcon     = document.getElementById('expand-icon');
const compressIcon   = document.getElementById('compress-icon');
const fsInputBar     = document.getElementById('fs-input-bar');
const fsUserText     = document.getElementById('fs-user-text');
const fsTextInput    = document.getElementById('fs-text-input');
const fsSendBtn      = document.getElementById('fs-send-btn');
const fsHandBtn      = document.getElementById('fs-hand-btn');
const fsUploadBtn    = document.getElementById('fs-upload-btn');
const fsCamBtn       = document.getElementById('fs-cam-btn');
const fsMicBtn       = document.getElementById('fs-mic-btn');
const fsSpkBtn       = document.getElementById('fs-spk-btn');

syncComposerLauncherState();

let isFullscreen = false;
let restoreSidebarOpen = true;
let setSidebarOpenState = null;

function applyFullscreenState(fullscreen) {
    isFullscreen = fullscreen;
    document.body.classList.toggle('board-fullscreen', isFullscreen);
    expandIcon.style.display   = isFullscreen ? 'none'  : 'block';
    compressIcon.style.display = isFullscreen ? 'block' : 'none';

    if (typeof setSidebarOpenState === 'function') {
        if (isFullscreen) {
            restoreSidebarOpen = document.body.classList.contains('sb-open');
            setSidebarOpenState(false, { remember: false });
        } else {
            setSidebarOpenState(restoreSidebarOpen, { remember: false });
        }
    }

    // Scroll after layout settles so the board stays anchored to the latest content.
    setTimeout(() => { frame.scrollTop = frame.scrollHeight; }, 100);
}

boardExpandBtn.addEventListener('click', function() {
    applyFullscreenState(!isFullscreen);
});

fsSendBtn.addEventListener('click', function() { submitTypedTextFrom(fsTextInput); });
fsTextInput.addEventListener('keydown', function(e) {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        submitTypedTextFrom(fsTextInput);
    }
});
// ✅ Auto-grow fs textarea (same cap as main)
fsTextInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 90) + 'px';
});

fsHandBtn.addEventListener('click', function() {
    if (_handDebouncing) return;
    _handDebouncing = true;
    setTimeout(() => { _handDebouncing = false; }, 2500);
    socket.emit('hand_raise');
    fsHandBtn.classList.add('raised');
    handBtn.classList.add('raised');
    handBtn.textContent = "✋ Raised!";
    logToCLI("⚠️ Interrupt signal sent...");
});

fsMicBtn.addEventListener('click', function() { toggleMic(); });
fsSpkBtn.addEventListener('click', function() { toggleSpk(); });

// Sync fs-user-text when user speaks
function updateUserText(txt) {
    if (userText) userText.textContent = txt;
    if (fsUserText) fsUserText.textContent = txt;
}

function showUploadStatus(msg, kind = 'info') {
    if (!uploadStatus) return;
    uploadStatus.textContent = msg || "";
    uploadStatus.dataset.kind = kind;
    uploadStatus.classList.add('visible');
    clearTimeout(showUploadStatus._timer);
    showUploadStatus._timer = setTimeout(function() {
        uploadStatus.classList.remove('visible');
    }, 5000);
}

function hideUploadStatus() {
    if (!uploadStatus) return;
    clearTimeout(showUploadStatus._timer);
    uploadStatus.classList.remove('visible');
    uploadStatus.textContent = "";
    delete uploadStatus.dataset.kind;
}

function syncComposerLauncherState() {
    const hasFile = !!pendingFile;
    if (uploadBtn) {
        uploadBtn.classList.toggle('has-file', hasFile);
        uploadBtn.classList.toggle('has-web-search', webSearchArmed);
    }
    if (fsUploadBtn) {
        fsUploadBtn.classList.toggle('has-file', hasFile);
        fsUploadBtn.classList.toggle('has-web-search', webSearchArmed);
    }
    if (composerMenuWebSearch) {
        composerMenuWebSearch.setAttribute('aria-pressed', webSearchArmed ? 'true' : 'false');
    }
}

function markUploadButtons(hasFile) {
    if (fileBadgeArea) fileBadgeArea.classList.toggle('visible', hasFile);
    if (sendBtn) sendBtn.classList.toggle('has-pending-file', hasFile);
    if (fsSendBtn) fsSendBtn.classList.toggle('has-pending-file', hasFile);
    syncComposerLauncherState();
}

function closeComposerMenu() {
    if (composerPlusMenu) composerPlusMenu.hidden = true;
    if (uploadBtn) uploadBtn.setAttribute('aria-expanded', 'false');
    if (fsUploadBtn) fsUploadBtn.setAttribute('aria-expanded', 'false');
    activeComposerLauncher = null;
}

function openComposerMenu(anchorEl) {
    if (!composerPlusMenu || !anchorEl) return;
    activeComposerLauncher = anchorEl;
    composerPlusMenu.hidden = false;
    const rect = anchorEl.getBoundingClientRect();
    const menuWidth = composerPlusMenu.offsetWidth || 300;
    const menuHeight = composerPlusMenu.offsetHeight || 140;
    const margin = 12;
    const left = Math.min(
        Math.max(margin, rect.left),
        Math.max(margin, window.innerWidth - menuWidth - margin)
    );
    let top = rect.top - menuHeight - 10;
    if (top < margin) top = Math.min(window.innerHeight - menuHeight - margin, rect.bottom + 10);
    composerPlusMenu.style.left = left + 'px';
    composerPlusMenu.style.top = Math.max(margin, top) + 'px';
    if (uploadBtn) uploadBtn.setAttribute('aria-expanded', anchorEl === uploadBtn ? 'true' : 'false');
    if (fsUploadBtn) fsUploadBtn.setAttribute('aria-expanded', anchorEl === fsUploadBtn ? 'true' : 'false');
}

function toggleComposerMenu(anchorEl) {
    if (!composerPlusMenu) return;
    const shouldOpen = composerPlusMenu.hidden || activeComposerLauncher !== anchorEl;
    if (!shouldOpen) {
        closeComposerMenu();
        return;
    }
    openComposerMenu(anchorEl);
}

function setWebSearchArmed(enabled, options = {}) {
    webSearchArmed = !!enabled;
    syncComposerLauncherState();
    if (options.announce !== false) {
        showUploadStatus(
            webSearchArmed
                ? 'Web search is on for your next typed question.'
                : 'Web search is off.',
            webSearchArmed ? 'info' : 'success'
        );
    }
}

function renderFileBadge(name, state) {
    if (!fileBadgeArea) return;
    fileBadgeArea.innerHTML = '';
    if (!name) {
        markUploadButtons(false);
        return;
    }

    const badge = document.createElement('div');
    badge.className = 'file-badge ' + (state === 'uploading' ? 'is-uploading' : 'is-ready');

    const icon = document.createElement('span');
    icon.className = 'file-badge-icon';
    icon.textContent = getFileIcon(name);

    const text = document.createElement('span');
    text.className = 'file-badge-name';
    text.textContent = name;

    badge.appendChild(icon);
    badge.appendChild(text);

    if (state === 'uploading') {
        const spinner = document.createElement('span');
        spinner.className = 'file-badge-spinner';
        spinner.textContent = '...';
        badge.appendChild(spinner);
    } else {
        const removeBtn = document.createElement('button');
        removeBtn.type = 'button';
        removeBtn.className = 'file-badge-remove';
        removeBtn.textContent = 'x';
        removeBtn.addEventListener('click', clearFileBadge);
        badge.appendChild(removeBtn);
    }

    fileBadgeArea.appendChild(badge);
    markUploadButtons(true);
}

function clearFileBadge() {
    pendingFile = null;
    if (fileBadgeArea) fileBadgeArea.innerHTML = '';
    markUploadButtons(false);
}

function getFileIcon(name) {
    const ext = ((name || '').split('.').pop() || '').toLowerCase();
    const icons = {
        pdf: 'PDF',
        doc: 'DOC',
        docx: 'DOC',
        txt: 'TXT',
        md: 'TXT',
        png: 'IMG',
        jpg: 'IMG',
        jpeg: 'IMG',
        webp: 'IMG',
        csv: 'CSV',
        py: 'PY',
        js: 'JS',
        html: 'HTML',
        css: 'CSS',
        json: 'JSON',
    };
    return icons[ext] || 'FILE';
}

function ensureNamedFile(file, fallbackPrefix = 'Upload') {
    if (!file) return null;
    if (file.name) return file;
    const safeExt = (file.type || '').split('/').pop() || 'bin';
    try {
        return new File([file], `${fallbackPrefix}-${Date.now()}.${safeExt}`, { type: file.type || 'application/octet-stream' });
    } catch (error) {
        file.name = `${fallbackPrefix}-${Date.now()}.${safeExt}`;
        return file;
    }
}

function getFirstUploadableFile(source) {
    const files = source && source.files ? Array.from(source.files) : [];
    for (const rawFile of files) {
        const file = ensureNamedFile(rawFile);
        if (file && file.size >= 0) return file;
    }
    return null;
}

function handleFileUpload(file) {
    if (!file) return;

    pendingFile = { name: file.name, chunks: 0, warning: '', ready: false };
    renderFileBadge(file.name, 'uploading');
    showUploadStatus(`Processing "${file.name}"...`);

    const form = new FormData();
    form.append('file', file);

    const controller = typeof AbortController !== 'undefined' ? new AbortController() : null;
    const timeoutMs = 90000;
    const timeoutId = controller ? window.setTimeout(function() {
        controller.abort();
    }, timeoutMs) : null;

    fetch('/upload', {
        method: 'POST',
        body: form,
        signal: controller ? controller.signal : undefined,
    })
        .then(async function(response) {
            const raw = await response.text();
            let data = {};
            try {
                data = raw ? JSON.parse(raw) : {};
            } catch (parseError) {
                throw new Error(raw || 'Upload failed.');
            }
            if (!response.ok || !data.ok) {
                throw new Error(data.error || 'Upload failed.');
            }
            pendingFile = {
                name: data.filename || file.name,
                chunks: data.chunks || 0,
                warning: data.warning || '',
                ready: true,
            };
            renderFileBadge(pendingFile.name, 'ready');
            if (typeof markSessionDirty === 'function') markSessionDirty();
            if (pendingFile.warning) {
                showUploadStatus(pendingFile.warning, 'warn');
            } else {
                showUploadStatus(`"${pendingFile.name}" is ready.`, 'success');
            }
        })
        .catch(function(error) {
            pendingFile = null;
            clearFileBadge();
            const isTimeout = error && (error.name === 'AbortError' || /aborted|timeout/i.test(String(error.message || error)));
            const message = isTimeout
                ? 'Upload timed out while the file was being analysed. Please try again or check the vision server.'
                : (error.message || String(error));
            showUploadStatus(message, 'error');
        })
        .finally(function() {
            if (timeoutId) window.clearTimeout(timeoutId);
            if (fileInput) fileInput.value = '';
        });
}

function buildPendingFilePrompt() {
    if (!pendingFile) return '';
    return `I uploaded "${pendingFile.name}". Use that file as context for your next response.`;
}

function applyCameraUI(opts = {}) {
    const enabled = !!opts.enabled;
    const attention = opts.attention || '';
    const alerting = !!opts.alerting;
    const watching = enabled && !alerting;

    if (camBtn) {
        camBtn.classList.toggle('active', watching);
        camBtn.classList.toggle('alerting', alerting);
        camBtn.title = attention ? `Camera: ${attention}` : 'Camera Monitoring On/Off';
    }
    if (camLabel) camLabel.textContent = alerting ? 'ALRT' : (enabled ? 'LIVE' : 'CAM');
    if (fsCamBtn) {
        fsCamBtn.classList.toggle('active', watching);
        fsCamBtn.classList.toggle('alerting', alerting);
        fsCamBtn.title = attention ? `Camera: ${attention}` : 'Camera';
    }
}

function pushCameraFrame() {
    if (!cameraOn || !camStream || !camVideo || !camCanvas) return;
    const width = camVideo.videoWidth || 0;
    const height = camVideo.videoHeight || 0;
    if (!width || !height) return;

    const targetWidth = Math.min(width, 640);
    camCanvas.width = targetWidth;
    camCanvas.height = Math.round(height * (targetWidth / width));
    const ctx = camCanvas.getContext('2d');
    if (!ctx) return;
    ctx.drawImage(camVideo, 0, 0, camCanvas.width, camCanvas.height);
    socket.emit('camera_frame', {
        frame: camCanvas.toDataURL('image/jpeg', 0.6).split(',')[1],
    });
}

function getCameraFrameIntervalMs() {
    if (emotionSettings && (emotionSettings.enabled || emotionSettings.show_monitor || emotionSettings.face_mesh_overlay)) {
        return 450;
    }
    return 1200;
}

function restartCameraFrameLoop() {
    clearInterval(camFrameTimer);
    camFrameTimer = null;
    if (!cameraOn) return;
    camFrameTimer = setInterval(pushCameraFrame, getCameraFrameIntervalMs());
}

async function startCamera() {
    if (cameraOn) return;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        showUploadStatus('Camera is not available in this browser.', 'error');
        return;
    }

    try {
        camStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        camVideo.srcObject = camStream;
        try {
            await camVideo.play();
        } catch (err) {}
        cameraOn = true;
        if (camOverlay) {
            camOverlay.style.left = 'auto';
            camOverlay.style.bottom = 'auto';
            camOverlay.style.right = '20px';
            camOverlay.style.top = document.body.classList.contains('board-fullscreen') ? '56px' : '60px';
            camOverlay.style.display = 'block';
        }
        hideUploadStatus();
        applyCameraUI({ enabled: true });
        restartCameraFrameLoop();
        pushCameraFrame();
        socket.emit('set_camera', { enabled: true });
        socket.emit('request_emotion_settings');
    } catch (error) {
        camStream = null;
        cameraOn = false;
        applyCameraUI({ enabled: false });
        showUploadStatus(`Camera error: ${error.message}`, 'error');
        socket.emit('set_camera', { enabled: false });
    }
}

function stopCamera(options = {}) {
    const emitServer = options.emitServer !== false;
    clearInterval(camFrameTimer);
    camFrameTimer = null;
    if (camStream) {
        camStream.getTracks().forEach(function(track) { track.stop(); });
    }
    if (camVideo) camVideo.srcObject = null;
    if (camOverlay) camOverlay.style.display = 'none';
    if (camAlertBanner) camAlertBanner.style.display = 'none';
    clearFaceMesh();
    camStream = null;
    cameraOn = false;
    applyCameraUI({ enabled: false });
    if (emitServer) socket.emit('set_camera', { enabled: false });
}

function toggleCamera(forceValue) {
    const nextState = (typeof forceValue === 'boolean') ? forceValue : !cameraOn;
    if (nextState) startCamera();
    else stopCamera();
}

if (camBtn) camBtn.addEventListener('click', function() { toggleCamera(); });
if (fsCamBtn) fsCamBtn.addEventListener('click', function() { toggleCamera(); });
if (fileInput) {
    fileInput.addEventListener('change', function() {
        const file = getFirstUploadableFile(this);
        if (file) handleFileUpload(file);
    });
}
if (uploadBtn) {
    uploadBtn.addEventListener('click', function(event) {
        event.preventDefault();
        toggleComposerMenu(uploadBtn);
    });
}
if (fsUploadBtn) {
    fsUploadBtn.addEventListener('click', function(event) {
        event.preventDefault();
        toggleComposerMenu(fsUploadBtn);
    });
}
if (composerMenuUpload) {
    composerMenuUpload.addEventListener('click', function() {
        closeComposerMenu();
        if (fileInput) fileInput.click();
    });
}
if (composerMenuWebSearch) {
    composerMenuWebSearch.addEventListener('click', function() {
        setWebSearchArmed(!webSearchArmed);
        closeComposerMenu();
    });
}

document.addEventListener('click', function(event) {
    const target = event.target;
    if (!target) return;
    if (composerPlusMenu && !composerPlusMenu.hidden && composerPlusMenu.contains(target)) return;
    if (target === uploadBtn || target === fsUploadBtn) return;
    closeComposerMenu();
});

document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') closeComposerMenu();
});

window.addEventListener('resize', closeComposerMenu);
window.addEventListener('scroll', closeComposerMenu, true);

function attachDropZone(el) {
    if (!el) return;
    el.addEventListener('dragenter', function(event) {
        const file = getFirstUploadableFile(event.dataTransfer);
        if (!file) return;
        event.preventDefault();
        if (textInputShell) textInputShell.classList.add('is-drop-target');
    });
    el.addEventListener('dragover', function(event) {
        const file = getFirstUploadableFile(event.dataTransfer);
        if (!file) return;
        event.preventDefault();
        if (event.dataTransfer) event.dataTransfer.dropEffect = 'copy';
        if (textInputShell) textInputShell.classList.add('is-drop-target');
    });
    el.addEventListener('dragleave', function() {
        if (textInputShell) textInputShell.classList.remove('is-drop-target');
    });
    el.addEventListener('drop', function(event) {
        const file = getFirstUploadableFile(event.dataTransfer);
        if (!file) return;
        event.preventDefault();
        if (textInputShell) textInputShell.classList.remove('is-drop-target');
        closeComposerMenu();
        handleFileUpload(file);
    });
}

[frame, textInputShell, textInput, fsTextInput].forEach(attachDropZone);

document.addEventListener('drop', function() {
    if (textInputShell) textInputShell.classList.remove('is-drop-target');
});
document.addEventListener('dragend', function() {
    if (textInputShell) textInputShell.classList.remove('is-drop-target');
});

document.addEventListener('paste', function(event) {
    const clipboard = event.clipboardData;
    const file = getFirstUploadableFile(clipboard);
    if (!file) return;
    event.preventDefault();
    closeComposerMenu();
    handleFileUpload(ensureNamedFile(file, 'Pasted'));
    showUploadStatus(`Pasted "${file.name || 'file'}". Processing...`);
});

if (frame) {
    frame.addEventListener('dragover', function(event) {
        const file = getFirstUploadableFile(event.dataTransfer);
        if (file) event.preventDefault();
    });
}

socket.on('file_uploaded', function(data) {
    if (pendingFile && data && data.filename) {
        pendingFile.chunks = data.chunks || pendingFile.chunks || 0;
        pendingFile.warning = data.warning || pendingFile.warning || '';
    }
});

socket.on('camera_status', function(data) {
    const attention = (data && data.attention) || '';
    const isProblem = !(data && data.excused) && ['phone', 'distracted_side', 'sleepy', 'away'].includes(attention);
    applyCameraUI({ enabled: cameraOn, attention: attention, alerting: cameraOn && isProblem });
    if (attention !== 'sleepy') clearCameraAlertTone();
    if (attention === 'focused' || attention === 'looking_down' || attention === 'text_active') {
        window.__cameraLessonPaused = false;
    }
});

socket.on('camera_alert', function(data) {
    const msg = (data && data.message) || 'Camera alert.';
    const pauseLesson = !!(data && data.pause_lesson);
    applyCameraUI({ enabled: cameraOn, attention: data && data.attention, alerting: cameraOn });
    if (camAlertBanner) {
        camAlertBanner.textContent = msg;
        camAlertBanner.className = 'cam-alert-banner cam-alert-' + ((data && data.attention) || 'default');
        camAlertBanner.style.display = 'block';
        clearTimeout(camAlertBanner._timer);
        camAlertBanner._timer = setTimeout(function() {
            camAlertBanner.style.display = 'none';
        }, pauseLesson ? 10000 : 8000);
    }
    showUploadStatus(msg, data && data.soft ? 'info' : 'warn');
    logToCLI(`Camera alert: ${msg}`);
    if (data && !data.soft) {
        const level = data.severity === 'high' ? 'critical' : 'medium';
        playEmotionBeep(level);
        if (data.continuous) startCameraAlertTone(level);
        else clearCameraAlertTone();
    }
    if (pauseLesson) {
        window.__cameraLessonPaused = true;
        if (data && data.resume_hint) {
            showUploadStatus(data.resume_hint, 'info');
        }
    }
});

socket.on('camera_returned', function(data) {
    applyCameraUI({ enabled: cameraOn, attention: 'focused', alerting: false });
    if (camAlertBanner) camAlertBanner.style.display = 'none';
    clearCameraAlertTone();
    window.__cameraLessonPaused = false;
    if (data && (data.resume_message || data.message)) showUploadStatus((data.resume_message || data.message), 'success');
});

socket.on('camera_excuse', function(data) {
    if (!data) return;
    if (data.active) showUploadStatus(`Camera paused: ${data.reason || 'excuse noted'}.`, 'info');
    else if (cameraOn) showUploadStatus('Camera monitoring resumed.', 'success');
    if (data.active) clearCameraAlertTone();
});

socket.on('camera_ack', function(data) {
    if (data && data.enabled === false && !cameraOn) {
        applyCameraUI({ enabled: false });
    }
});

socket.on('connect', function() {
    if (cameraOn) socket.emit('set_camera', { enabled: true });
    socket.emit('request_emotion_settings');
});

window.addEventListener('beforeunload', function() {
    if (cameraOn) stopCamera({ emitServer: false });
});

(function initCamDrag() {
    const overlay = document.getElementById('cam-overlay');
    const handle = document.getElementById('cam-drag-handle');
    if (!overlay || !handle) return;

    let dragging = false;
    let startX = 0;
    let startY = 0;
    let startLeft = 0;
    let startTop = 0;

    handle.addEventListener('mousedown', startDrag);
    handle.addEventListener('touchstart', startDrag, { passive: false });

    function startDrag(event) {
        event.preventDefault();
        const point = getPoint(event);
        const rect = overlay.getBoundingClientRect();
        dragging = true;
        overlay.classList.add('dragging');
        overlay.style.left = rect.left + 'px';
        overlay.style.top = rect.top + 'px';
        overlay.style.right = 'auto';
        overlay.style.bottom = 'auto';
        startX = point.x;
        startY = point.y;
        startLeft = rect.left;
        startTop = rect.top;
        document.addEventListener('mousemove', moveDrag);
        document.addEventListener('mouseup', stopDrag);
        document.addEventListener('touchmove', moveDrag, { passive: false });
        document.addEventListener('touchend', stopDrag);
    }

    function moveDrag(event) {
        if (!dragging) return;
        event.preventDefault();
        const point = getPoint(event);
        overlay.style.left = Math.max(12, Math.min(startLeft + point.x - startX, window.innerWidth - overlay.offsetWidth - 12)) + 'px';
        overlay.style.top = Math.max(12, Math.min(startTop + point.y - startY, window.innerHeight - overlay.offsetHeight - 12)) + 'px';
    }

    function stopDrag() {
        dragging = false;
        overlay.classList.remove('dragging');
        document.removeEventListener('mousemove', moveDrag);
        document.removeEventListener('mouseup', stopDrag);
        document.removeEventListener('touchmove', moveDrag);
        document.removeEventListener('touchend', stopDrag);
    }

    function getPoint(event) {
        if (event.touches && event.touches.length) {
            return { x: event.touches[0].clientX, y: event.touches[0].clientY };
        }
        return { x: event.clientX, y: event.clientY };
    }
})();

function playEmotionBeep(level) {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return;
    if (!playEmotionBeep._ctx) playEmotionBeep._ctx = new AudioCtx();
    const ctx = playEmotionBeep._ctx;
    const now = ctx.currentTime;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = level === 'critical' ? 'square' : 'sine';
    osc.frequency.value = level === 'critical' ? 940 : level === 'high' ? 880 : 660;
    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.exponentialRampToValueAtTime(level === 'critical' ? 0.13 : 0.08, now + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + (level === 'critical' ? 0.34 : 0.18));
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start(now);
    osc.stop(now + (level === 'critical' ? 0.36 : 0.2));
}

function clearCameraAlertTone() {
    if (window.__cameraAlertToneTimer) {
        clearInterval(window.__cameraAlertToneTimer);
        window.__cameraAlertToneTimer = null;
    }
}

function startCameraAlertTone(level) {
    clearCameraAlertTone();
    window.__cameraAlertToneTimer = setInterval(function() {
        playEmotionBeep(level || 'critical');
    }, 850);
}

function formatPct(value) {
    if (value === null || value === undefined || isNaN(value)) return '0%';
    return `${Math.round(Number(value) * 100)}%`;
}

function formatSigned(value) {
    if (value === null || value === undefined || isNaN(value)) return '—';
    const num = Number(value);
    return `${num >= 0 ? '+' : ''}${num.toFixed(2)}`;
}

function escapeHtml(value) {
    return String(value === null || value === undefined ? '' : value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function renderBadgeList(container, dataMap, limit) {
    if (!container) return;
    const entries = Object.entries(dataMap || {}).sort(function(a, b) { return (b[1] || 0) - (a[1] || 0); });
    const filtered = entries.filter(function(entry) { return (entry[1] || 0) > 0.02; }).slice(0, limit);
    if (!filtered.length) {
        container.innerHTML = '<span class="emotion-badge muted">Not available</span>';
        return;
    }
    container.innerHTML = filtered.map(function(entry) {
        return `<span class="emotion-badge"><b>${entry[0]}</b> ${formatPct(entry[1])}</span>`;
    }).join('');
}

function renderKvList(container, rows) {
    if (!container) return;
    const filtered = (rows || []).filter(function(row) {
        return row && row.value !== null && row.value !== undefined && String(row.value).trim() !== '';
    });
    if (!filtered.length) {
        container.innerHTML = '<div class="emotion-monitor-kv"><div class="emotion-monitor-kv-label">Status</div><div class="emotion-monitor-kv-value">Not available</div></div>';
        return;
    }
    container.innerHTML = filtered.map(function(row) {
        return (
            `<div class="emotion-monitor-kv">` +
            `<div class="emotion-monitor-kv-label">${escapeHtml(row.label || '')}</div>` +
            `<div class="emotion-monitor-kv-value">${escapeHtml(row.value || '')}</div>` +
            `</div>`
        );
    }).join('');
}

function resizeMeshCanvas() {
    if (!camMeshCanvas || !camVideo) return null;
    const rect = camVideo.getBoundingClientRect();
    const width = Math.max(1, Math.round(rect.width));
    const height = Math.max(1, Math.round(rect.height));
    if (camMeshCanvas.width !== width) camMeshCanvas.width = width;
    if (camMeshCanvas.height !== height) camMeshCanvas.height = height;
    return { width: width, height: height };
}

function drawMeshConnections(ctx, landmarks, connections, width, height, strokeStyle, lineWidth) {
    if (!connections || !connections.length) return;
    ctx.strokeStyle = strokeStyle;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    connections.forEach(function(pair) {
        const a = landmarks[pair[0]];
        const b = landmarks[pair[1]];
        if (!a || !b) return;
        ctx.moveTo(a[0] * width, a[1] * height);
        ctx.lineTo(b[0] * width, b[1] * height);
    });
    ctx.stroke();
}

function drawFaceMeshOverlay(payload) {
    if (!camMeshCanvas) return;
    const dims = resizeMeshCanvas();
    const ctx = camMeshCanvas.getContext('2d');
    if (!ctx || !dims) return;
    ctx.clearRect(0, 0, dims.width, dims.height);
    if (!cameraOn || !emotionSettings.face_mesh_overlay || !payload || !payload.landmarks || !payload.landmarks.length) {
        return;
    }
    const mesh = (((payload.extras || {}).mesh_connections) || {});
    ctx.save();
    ctx.translate(dims.width, 0);
    ctx.scale(-1, 1);
    drawMeshConnections(ctx, payload.landmarks, mesh.tesselation || [], dims.width, dims.height, 'rgba(120, 220, 255, 0.22)', 0.7);
    drawMeshConnections(ctx, payload.landmarks, mesh.contours || [], dims.width, dims.height, 'rgba(255, 210, 120, 0.58)', 1.2);
    drawMeshConnections(ctx, payload.landmarks, mesh.irises || [], dims.width, dims.height, 'rgba(255, 255, 255, 0.82)', 1.1);
    ctx.fillStyle = 'rgba(255,255,255,0.55)';
    payload.landmarks.forEach(function(point) {
        ctx.beginPath();
        ctx.arc(point[0] * dims.width, point[1] * dims.height, 1.1, 0, Math.PI * 2);
        ctx.fill();
    });
    ctx.restore();
}

function renderEmotionMonitor(data) {
    emotionMonitorState = data || null;
    if (!data) return;
    const extras = (data.extras || {});
    const textScores = (((data.text || {}).scores) || {});
    const textDetails = (((data.text || {}).details) || {});
    const vlm = extras.vlm || {};
    const topTextState = Object.entries(textScores).sort(function(a, b) { return (b[1] || 0) - (a[1] || 0); })[0];
    const rawTextLabels = {};
    ((textDetails.raw_labels) || []).forEach(function(entry) {
        if (Array.isArray(entry) && entry.length >= 2) {
            rawTextLabels[String(entry[0])] = Number(entry[1]) || 0;
        }
    });
    if (emotionMonitorSubtitle) {
        const action = (((data.policy || {}).pedagogical_action) || 'normal_explain').replace(/_/g, ' ');
        const empathy = ((data.policy || {}).empathy_type) || 'none';
        emotionMonitorSubtitle.textContent = `${action} | empathy ${empathy}`;
    }
    if (emotionAttention) {
        const focus = (extras.attention_line || data.attention_status || 'Camera off');
        const vlmState = data.vlm_state && data.vlm_state !== 'Not available' ? ` / VLM ${data.vlm_state}` : '';
        emotionAttention.textContent = `${focus}${vlmState}`;
    }
    if (emotionFaceSummary) {
        const raw = data.raw_face_emotion || 'Not available';
        const tutor = data.tutoring_face_label || 'Not available';
        const engagement = data.engagement_label || 'Not available';
        emotionFaceSummary.textContent = `${raw} ${formatPct(data.raw_face_confidence)} | ${tutor} | ${engagement} | V ${formatSigned(data.valence)} A ${formatSigned(data.arousal)}`;
    }
    if (emotionPolicySummary) {
        const rules = (((data.policy || {}).response_rules) || []).slice(0, 2).join(' | ');
        const base = ((data.policy || {}).justification) || ((data.policy || {}).suppressed_reason) || 'Emotion Engine disabled';
        emotionPolicySummary.textContent = rules ? `${base} | ${rules}` : base;
    }
    if (emotionTextInput) {
        emotionTextInput.textContent = textDetails.input_text || 'No text analyzed yet.';
    }
    renderKvList(emotionCameraDetails, [
        { label: 'Eyes', value: extras.eyes_line || 'Not available' },
        { label: 'Attention', value: extras.attention_line || data.attention_status || 'Camera off' },
        { label: 'Mouth', value: extras.mouth_line || 'Not available' },
        {
            label: 'Runtime',
            value: `${extras.detector_info || 'detector n/a'} | ${extras.emotion_runtime || 'emotion n/a'} | landmarks ${extras.landmarks_enabled ? 'on' : 'off'}`,
        },
    ]);
    renderKvList(emotionVlmDetails, [
        {
            label: 'Scene',
            value: `people ${vlm.person_count || 0} | phone ${vlm.phone_present ? 'yes' : 'no'} | object ${vlm.object_label || 'none'}`,
        },
        { label: 'Phone', value: vlm.phone_present ? (vlm.phone_activity || 'unknown') : 'none' },
        { label: 'Action', value: vlm.observed_action || 'none' },
        { label: 'Text', value: vlm.visible_text || '' },
        { label: 'Note', value: vlm.attention_comment || '' },
        { label: 'Runtime', value: vlm.vlm_runtime || '' },
    ]);
    renderKvList(emotionTextDetails, [
        { label: 'Backend', value: textDetails.backend || ((data.text || {}).source) || 'Not available' },
        { label: 'Evidence', value: ((textDetails.evidence || []).join(', ')) || 'None' },
        { label: 'Mapped', value: topTextState ? `${topTextState[0]} ${formatPct(topTextState[1])}` : 'Not available' },
    ]);
    renderBadgeList(emotionRawFaceProbs, data.raw_face_probabilities || {}, 8);
    renderBadgeList(emotionTextRawLabels, rawTextLabels, 6);
    renderBadgeList(emotionTextState, textScores, 4);
    renderBadgeList(emotionPerformanceState, ((data.performance || {}).scores) || {}, 4);
    renderBadgeList(emotionCameraState, ((data.camera || {}).scores) || {}, 4);
    renderBadgeList(emotionFaceState, ((data.face || {}).scores) || {}, 4);
    renderBadgeList(emotionFusedState, (((data.fused || {}).scores) || {}), 6);
    drawFaceMeshOverlay(data);
}

socket.on('emotion_engine_status', function(data) {
    if (data && data.settings) {
        emotionSettings = Object.assign({}, emotionSettings, data.settings);
        applyEmotionSettingsUI(false);
        saveSettings();
    }
});

socket.on('emotion_monitor_update', function(data) {
    renderEmotionMonitor(data);
});

socket.on('emotion_alert', function(data) {
    const msg = (data && data.message) || 'Focus check.';
    if (camAlertBanner) {
        camAlertBanner.textContent = msg;
        camAlertBanner.className = 'cam-alert-banner cam-alert-' + ((data && data.key) || 'default');
        camAlertBanner.style.display = 'block';
        clearTimeout(camAlertBanner._timer);
        camAlertBanner._timer = setTimeout(function() {
            camAlertBanner.style.display = 'none';
        }, 8000);
    }
    if (data && data.severity === 'high') {
        playEmotionBeep('high');
    }
});

(function initEmotionMonitorDrag() {
    if (!emotionMonitor || !emotionMonitorHandle) return;

    let dragging = false;
    let startX = 0;
    let startY = 0;
    let startLeft = 0;
    let startTop = 0;

    function getPoint(event) {
        if (event.touches && event.touches.length) {
            return { x: event.touches[0].clientX, y: event.touches[0].clientY };
        }
        return { x: event.clientX, y: event.clientY };
    }

    function startDrag(event) {
        event.preventDefault();
        const point = getPoint(event);
        const rect = emotionMonitor.getBoundingClientRect();
        dragging = true;
        emotionMonitor.style.left = rect.left + 'px';
        emotionMonitor.style.top = rect.top + 'px';
        emotionMonitor.style.right = 'auto';
        startX = point.x;
        startY = point.y;
        startLeft = rect.left;
        startTop = rect.top;
        document.addEventListener('mousemove', moveDrag);
        document.addEventListener('mouseup', stopDrag);
        document.addEventListener('touchmove', moveDrag, { passive: false });
        document.addEventListener('touchend', stopDrag);
    }

    function moveDrag(event) {
        if (!dragging) return;
        event.preventDefault();
        const point = getPoint(event);
        const left = Math.max(12, Math.min(startLeft + point.x - startX, window.innerWidth - emotionMonitor.offsetWidth - 12));
        const top = Math.max(12, Math.min(startTop + point.y - startY, window.innerHeight - emotionMonitor.offsetHeight - 12));
        emotionMonitor.style.left = left + 'px';
        emotionMonitor.style.top = top + 'px';
    }

    function stopDrag() {
        if (!dragging) return;
        dragging = false;
        document.removeEventListener('mousemove', moveDrag);
        document.removeEventListener('mouseup', stopDrag);
        document.removeEventListener('touchmove', moveDrag);
        document.removeEventListener('touchend', stopDrag);
        const rect = emotionMonitor.getBoundingClientRect();
        emotionPopupPosition = { left: Math.round(rect.left), top: Math.round(rect.top) };
        saveSettings();
    }

    emotionMonitorHandle.addEventListener('mousedown', startDrag);
    emotionMonitorHandle.addEventListener('touchstart', startDrag, { passive: false });
    if (emotionMonitorClose) {
        emotionMonitorClose.addEventListener('click', function() {
            updateEmotionSetting('show_monitor', false);
        });
    }
})();

window.addEventListener('resize', function() {
    drawFaceMeshOverlay(emotionMonitorState);
});

// ============================================================
// ✅ MIC / SPEAKER TOGGLE
// ============================================================
let micMuted = false;
let spkMuted = false;

const micBtn     = document.getElementById('mic-btn');
const micOnIcon  = document.getElementById('mic-on-icon');
const micOffIcon = document.getElementById('mic-off-icon');
const micLabel   = document.getElementById('mic-label');

const spkBtn     = document.getElementById('spk-btn');
const spkOnIcon  = document.getElementById('spk-on-icon');
const spkOffIcon = document.getElementById('spk-off-icon');
const spkLabel   = document.getElementById('spk-label');

micBtn.addEventListener('click', function() { toggleMic(); });
spkBtn.addEventListener('click', function() { toggleSpk(); });

function toggleMic() {
    micMuted = !micMuted;
    // UI — main buttons
    micBtn.classList.toggle('muted', micMuted);
    micOnIcon.style.display  = micMuted ? 'none'  : 'block';
    micOffIcon.style.display = micMuted ? 'block' : 'none';
    micLabel.textContent     = micMuted ? 'MUTED' : 'MIC';
    // UI — fs button
    fsMicBtn.textContent = micMuted ? '🚫🎙️' : '🎙️';
    fsMicBtn.classList.toggle('muted', micMuted);
    // Emit to server
    socket.emit('set_mic', { muted: micMuted });
    logToCLI(`🎙️ Mic: ${micMuted ? 'MUTED' : 'ON'}`);
}

function toggleSpk() {
    spkMuted = !spkMuted;
    // UI — main buttons
    spkBtn.classList.toggle('muted', spkMuted);
    spkOnIcon.style.display  = spkMuted ? 'none'  : 'block';
    spkOffIcon.style.display = spkMuted ? 'block' : 'none';
    spkLabel.textContent     = spkMuted ? 'MUTED' : 'SPK';
    // UI — fs button
    fsSpkBtn.textContent = spkMuted ? '🔇' : '🔊';
    fsSpkBtn.classList.toggle('muted', spkMuted);
    // Emit to server
    socket.emit('set_speaker', { muted: spkMuted });
    logToCLI(`🔊 Speaker: ${spkMuted ? 'MUTED' : 'ON'}`);
}

// Sync ack from server
socket.on('mic_ack',     function(d) { if (d && d.muted !== micMuted) toggleMic(); });
socket.on('speaker_ack', function(d) { if (d && d.muted !== spkMuted) toggleSpk(); });


// ============================================================
// AUTO SCROLL
// ============================================================
function isAtBottom(el, tolerance = 120) {
    return (el.scrollHeight - el.scrollTop - el.clientHeight) <= tolerance;
}
frame.addEventListener('scroll', () => { autoScrollEnabled = isAtBottom(frame); });
function smartScrollToBottom(force = false) {
    if (!force && !autoScrollEnabled) return;
    requestAnimationFrame(() => { frame.scrollTop = frame.scrollHeight; });
}
function shouldStickToBottom() { return isAtBottom(frame, 120); }


// ============================================================
// SOCKET EVENTS (unchanged)
// ============================================================
socket.on('connect', function() {
    logToCLI("✅ Connected to Server");
    socket.emit('set_voice', { engine: currentEngine, speaker: speakerPerEngine[currentEngine] });
    applyVoiceUI(currentEngine, speakerPerEngine[currentEngine]);
    socket.emit('set_llm_model', { model: currentLLMModel });
    socket.emit('request_emotion_settings');
});

socket.on('learning_mode_state', function(data) {
    const mode = normalizeLearningMode((data || {}).mode);
    if (!window.__tutorSessionLockedForModeChange || !window.__modeSelectionPinned) {
        applyLearningModeUI(mode);
        saveSettings();
        if (typeof renderRecentsList === 'function') setTimeout(renderRecentsList, 0);
    }
    requestBoardRestore(mode);
});

socket.on('learning_mode_busy', function(data) {
    const mode = normalizeLearningMode((data || {}).mode || currentLearningMode);
    if (!window.__modeSelectionPinned) {
        applyLearningModeUI(mode);
        saveSettings();
    }
    showUploadStatus('Wait for the current response to finish before switching modes.', 'info');
    requestBoardRestore(mode);
});

// Re-orientation after reconnect
socket.on('tutor_reorient', function(data) {
    if (!data || !data.message) return;
    logToCLI("↩️ " + data.message);
    const el = document.getElementById('ai-status-text');
    if (el) el.innerHTML = '<span style="color:#ffcc02;font-weight:600">↩️ ' + data.message + '</span>';
    setTimeout(function() { if (typeof setAIStatus === 'function') setAIStatus('LISTENING'); }, 5000);
});

// Break auto-expired by server (reconnect deadlock guard)
socket.on('break_auto_ended', function() {
    if (typeof stopBreakTimer === 'function' && typeof breakState !== 'undefined' && breakState.active) {
        stopBreakTimer();
    }
    const ex = document.getElementById('break-overlay');
    if (ex) ex.remove();
    logToCLI("⏰ Break auto-expired by server");
});

// qaMode: true while MCQ is displayed on board — blocks board_text from overwriting
let qaMode = false;

socket.on('board_text', function(data, ackCb) {
    // Block board overwrites while QA MCQ is on screen
    // Exception: append=false is a deliberate full-replace (e.g. QA header) — allow it
    if (qaMode && data.append !== false) {
        if (typeof ackCb === "function") { try { ackCb({ ok: true }); } catch(e) {} }
        return;
    }
    const text    = repairDisplayText(data.text || "");
    const append  = (data.append !== false);
    const mode    = data.mode || "instant";
    const cps     = Number(data.cps || 35);
    const scrollF = (data.scroll !== false);
    const stick   = scrollF && shouldStickToBottom();
    if (!append) boardBuffer = text;
    else boardBuffer += text;
    enqueueRender({
        append, text, mode, cps, stick,
        done: () => { if (typeof ackCb === "function") { try { ackCb({ ok: true }); } catch(e) {} } }
    });
});

// ✅ v3: Scroll to concept on board (QA review re-teach)
socket.on('board_highlight', function(data) {
    const searchText = data.text || "";
    if (!searchText) return;
    const boardEl = document.getElementById('blackboard-text');
    if (!boardEl) return;
    // Find text node containing searchText and scroll to it
    const walker = document.createTreeWalker(boardEl, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) {
        const node = walker.currentNode;
        if (node.textContent.toLowerCase().includes(searchText.toLowerCase())) {
            const range = document.createRange();
            range.selectNodeContents(node);
            range.collapse(true);
            const rect = node.parentElement.getBoundingClientRect();
            const boardRect = boardEl.getBoundingClientRect();
            const scrollTop = node.parentElement.offsetTop - 20;
            boardEl.scrollTop = scrollTop;
            // flash highlight
            node.parentElement.style.transition = 'background 0.3s';
            node.parentElement.style.background = 'rgba(255,214,0,0.35)';
            setTimeout(() => { node.parentElement.style.background = ''; }, 2000);
            break;
        }
    }
});

socket.on('clear_board', function() {
    qaMode = false;   // always reset QA lock on explicit clear
    boardBuffer = ""; displayBuffer = "";
    blackboardText.innerHTML = "";
    typingQueue = []; typingActive = false; pendingFxQueue = [];
    logToCLI("🧹 Blackboard cleared");
});

socket.on('cli_log', function(data) { logToCLI(data.msg); });

socket.on('user_speech', function(data) {
    updateUserText(data.text);
    logToCLI(`🧑 User: ${data.text}`);
    // ✅ Reset BOTH hand buttons when user speaks
    handBtn.classList.remove('raised', 'active');
    setTimeout(function() {
        if (handBtn) handBtn.textContent = repairDisplayText(handBtn.textContent || '');
    }, 0);
    handBtn.textContent = "✋ Raise Hand";
    fsHandBtn.classList.remove('raised');
    // Show thinking immediately
    setAIStatus('ANALYZING');
});

socket.on('board_fx', function(data) {
    try {
        const actions = data.actions || [];
        if (!actions.length) return;
        pendingFxQueue.push(...actions);
        if (!typingActive && typingQueue.length === 0) flushPendingFx();
    } catch(e) {}
});

// ✅ AI status elements (main + fullscreen)
function setAIStatus(state) {
    const el    = document.getElementById('ai-status-text');
    const fsEl  = document.getElementById('fs-ai-text');
    let html = '', text = '';
    switch(state) {
        case 'ANALYZING':
            html = '<span class="ai-thinking">🧠 Thinking</span>';
            text = '🧠 Thinking...';
            break;
        case 'SPEAKING':
            html = '<span class="ai-speaking">🔊 Speaking</span>';
            text = '🔊 Speaking';
            break;
        case 'LISTENING':
            html = '<span class="ai-listening">👂 Listening</span>';
            text = '👂 Listening';
            break;
        default:
            html = '<span class="ai-idle">Ready</span>';
            text = 'Ready';
    }
    if (el)   el.innerHTML = repairDisplayText(html);
    if (fsEl) fsEl.textContent = repairDisplayText(text);
}

// ── In-Board QA Rendering (Issue 1: MCQ inside chalkboard) ──────────────
let currentQACard = null;        // legacy popup card (kept for compat)
let qaSelectedLetter = '';       // track what student clicked

// Render MCQ *inside* the chalkboard
socket.on('qa_board', function(data) {
    const idx     = data.idx;
    const total   = data.total;
    const type    = data.type || 'simple';
    const question = data.question || '';
    const options  = data.options || {};

    const boardEl = document.getElementById('blackboard-text');
    if (!boardEl) return;

    qaMode = true;           // ← lock board from teaching content
    qaSelectedLetter = '';
    if (typeof window.__setCourseProgressHidden === 'function') window.__setCourseProgressHidden(true);

    // Show/update progress circle
    showQAProgress(idx + 1, total);

    let html = `<div class="qb-header">Q${idx+1} of ${total}</div>`;
    html += `<div class="qb-question">${repairDisplayText(question)}</div>`;

    if (type === 'mcq' && Object.keys(options).length > 0) {
        html += '<div class="qb-options">';
        for (const [letter, text] of Object.entries(options)) {
            html += `<div class="qb-option" data-letter="${letter}" onclick="selectBoardOption(this)">
                <span class="qb-circle">${letter}</span>
                <span class="qb-text">${repairDisplayText(text)}</span>
            </div>`;
        }
        html += '</div>';
    } else {
        html += '<div class="qb-hint">🎤 Speak or type your answer below</div>';
    }

    boardEl.innerHTML = repairDisplayText(html);
});

function selectBoardOption(el) {
    document.querySelectorAll('.qb-option').forEach(o => o.classList.remove('selected'));
    el.classList.add('selected');
    qaSelectedLetter = el.getAttribute('data-letter');

    // Emit selected letter to server so it can match voice vs click
    socket.emit('qa_option_click', { letter: qaSelectedLetter });
}

// Show correct/wrong highlights inside board after answer
socket.on('qa_board_result', function(data) {
    const correct       = data.correct;
    const correctAnswer = repairDisplayText(data.correct_answer || '');
    const feedback      = repairDisplayText(data.feedback || '');

    const boardEl = document.getElementById('blackboard-text');
    if (!boardEl) return;

    // qaMode stays ON so feedback isn't overwritten by speech board_text
    qaMode = true;

    // If no MCQ options (voice answer), show a standalone result banner
    const opts = boardEl.querySelectorAll('.qb-option');
    if (opts.length === 0) {
        // Voice/typed answer — inject result banner into existing board content
        const banner = document.createElement('div');
        banner.className = correct ? 'qb-feedback qb-feedback-correct' : 'qb-feedback qb-feedback-wrong';
        banner.innerHTML = correct
            ? `<strong>✅ Correct!</strong> ${feedback}`
            : `<strong>❌ Wrong.</strong> Correct: <em>${correctAnswer}</em> — ${feedback}`;
        banner.innerHTML = repairDisplayText(banner.innerHTML);
        boardEl.appendChild(banner);
        return;
    }

    // MCQ options — highlight correct/wrong
    opts.forEach(opt => {
        const letter = opt.getAttribute('data-letter');
        opt.style.pointerEvents = 'none';
        if (letter === correctAnswer) {
            opt.classList.add('qb-correct');
            opt.querySelector('.qb-circle').innerHTML = '✓';
        } else if (opt.classList.contains('selected') && !correct) {
            opt.classList.add('qb-wrong');
            opt.querySelector('.qb-circle').innerHTML = '✗';
        }
    });

    // Add feedback strip inside board
    const fb = document.createElement('div');
    fb.className = correct ? 'qb-feedback qb-feedback-correct' : 'qb-feedback qb-feedback-wrong';
    fb.innerHTML = correct
        ? `<strong>✅ Correct!</strong> ${feedback}`
        : `<strong>❌ Wrong.</strong> Correct answer: <em>${correctAnswer}</em> — ${feedback}`;
    fb.innerHTML = repairDisplayText(fb.innerHTML);
    boardEl.appendChild(fb);
});

// Server signals QA session is done — resume normal board rendering
socket.on('qa_end', function(data) {
    hideQAProgress();
    if (typeof window.__setCourseProgressHidden === 'function') window.__setCourseProgressHidden(false);

    // Show summary card on board, then release qaMode after 10s
    const items    = (data && data.items)   || [];
    const topic    = repairDisplayText((data && data.topic) || '');
    const correct  = (data && data.correct) || 0;
    const total    = (data && data.total)   || items.length;

    if (items.length > 0) {
        const boardEl = document.getElementById('blackboard-text');
        if (boardEl) {
            let html = `<div class="qb-summary-title">📋 Q&A Summary${topic ? ' — ' + topic : ''}</div>`;
            html += '<div class="qb-summary-divider"></div>';
            items.forEach((it, i) => {
                const icon = it.correct ? '✅' : '❌';
                html += `<div class="qb-summary-row ${it.correct ? 'qb-sr-correct' : 'qb-sr-wrong'}">
                    <span class="qb-sr-icon">${icon}</span>
                    <span class="qb-sr-q">Q${i+1}. ${it.question}</span>
                    ${!it.correct ? `<span class="qb-sr-ans">→ ${it.answer}</span>` : ''}
                </div>`;
            });
            html += '<div class="qb-summary-divider"></div>';
            html += `<div class="qb-summary-score">Score: ${correct}/${total}</div>`;
            boardEl.innerHTML = repairDisplayText(html);
        }
        // Release board after 12 seconds
        setTimeout(() => {
            qaMode = false;
            boardBuffer = "";
        }, 12000);
    } else {
        qaMode = false;
        boardBuffer = "";
    }
});

// Legacy popup card (kept for fallback, hidden when board version active)
socket.on('qa_question', function(data) {
    if (currentQACard) { currentQACard.remove(); currentQACard = null; }
    // Board version handles display; popup card is now secondary
});

function selectQAOption(el) {
    document.querySelectorAll('.qa-option').forEach(o => o.classList.remove('selected'));
    el.classList.add('selected');
}

socket.on('qa_result', function(data) {
    // Legacy popup result — handled by qa_board_result now; keep for compat
    if (!currentQACard) return;
    const correct = data.correct;
    const correctAnswer = data.correct_answer || '';
    const options = currentQACard.querySelectorAll('.qa-option');
    options.forEach(opt => {
        const letter = opt.getAttribute('data-letter');
        opt.style.pointerEvents = 'none';
        if (letter === correctAnswer) {
            opt.classList.add('qa-correct');
            opt.querySelector('.qa-option-circle').innerHTML = '✓';
        } else if (opt.classList.contains('selected') && !correct) {
            opt.classList.add('qa-wrong');
            opt.querySelector('.qa-option-circle').innerHTML = '✗';
        }
    });
    setTimeout(() => {
        if (currentQACard) { currentQACard.style.opacity = '0'; currentQACard.style.transition = 'opacity 0.5s';
            setTimeout(() => { if (currentQACard) { currentQACard.remove(); currentQACard = null; } }, 500); }
    }, 3500);
});
// ── Bug 2: Pause mic immediately on MCQ click to prevent ambient speech race ──
socket.on('pause_mic_for_click', function() {
    // Temporarily disable mic input (same as when tutor is speaking)
    setAIStatus('ANALYZING');
});

// ── QA Progress Circle ────────────────────────────────────────────────────
let qaProgressEl = null;

function showQAProgress(current, total) {
    // Remove existing if any
    if (qaProgressEl) qaProgressEl.remove();

    const container = document.getElementById('blackboard-container') || document.getElementById('blackboard-frame');
    if (!container) return;

    qaProgressEl = document.createElement('div');
    qaProgressEl.id = 'qa-progress-circle';
    qaProgressEl.className = 'qa-progress-circle';

    // SVG donut
    const r = 22, cx = 28, cy = 28;
    const circ = 2 * Math.PI * r;
    const filled = (current / total) * circ;
    const gap    = circ - filled;

    qaProgressEl.innerHTML = `
        <svg width="56" height="56" viewBox="0 0 56 56">
            <circle cx="${cx}" cy="${cy}" r="${r}"
                fill="none" stroke="rgba(255,255,255,0.15)" stroke-width="5"/>
            <circle cx="${cx}" cy="${cy}" r="${r}"
                fill="none" stroke="#ffffff" stroke-width="5"
                stroke-dasharray="${filled} ${gap}"
                stroke-dashoffset="${circ / 4}"
                stroke-linecap="round"
                class="qa-progress-arc"/>
        </svg>
        <span class="qa-progress-label">${current}/${total}</span>`;

    // Position top-right of blackboard frame
    qaProgressEl.style.cssText = `
        position:absolute; top:12px; right:12px;
        width:56px; height:56px;
        display:flex; align-items:center; justify-content:center;
        z-index:50; pointer-events:none;`;

    // Make container relative if not already
    const frame = document.getElementById('blackboard-frame') || container;
    const existingPos = getComputedStyle(frame).position;
    if (existingPos === 'static') frame.style.position = 'relative';
    frame.appendChild(qaProgressEl);
}

function hideQAProgress() {
    if (qaProgressEl) { qaProgressEl.remove(); qaProgressEl = null; }
}

socket.on('tutor_status', function(data) {
    setAIStatus(data.status);
    if (data.status === 'LISTENING') {
        handBtn.classList.remove('active');
        handBtn.textContent = "✋ Raise Hand";
        flushBoardQueueInstant();
    }
});


// ============================================================
// UI INTERACTIONS (unchanged)
// ============================================================
let _handDebouncing = false;
handBtn.addEventListener('click', function() {
    if (_handDebouncing) return;
    _handDebouncing = true;
    setTimeout(() => { _handDebouncing = false; }, 2500);
    socket.emit('hand_raise');
    handBtn.classList.remove('active');
    handBtn.classList.add('raised');
    handBtn.textContent = "✋ Raised!";
    logToCLI("⚠️ Interrupt signal sent...");
});

sendBtn.addEventListener('click', function() { submitTypedTextFrom(textInput); });
textInput.addEventListener('keydown', function(e) {
    if (e.key === "Enter" && !e.shiftKey) {
        // ✅ Enter = submit; Shift+Enter = new line
        e.preventDefault();
        submitTypedTextFrom(textInput);
    }
    // Shift+Enter falls through naturally (textarea newline)
});

// ✅ Auto-grow textarea height (capped at max-height in CSS)
textInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 100) + 'px';
});

function submitTypedTextFrom(inputEl) {
    const txt = (inputEl.value || "").trim();
    const filePrompt = buildPendingFilePrompt();
    const finalText = txt || filePrompt;
    if (!finalText) return;
    const useWebSearch = webSearchArmed && !!txt;

    socket.emit('text_submit', { text: finalText, web_search: useWebSearch });

    textInput.value = "";
    fsTextInput.value = "";
    textInput.style.height = 'auto';
    fsTextInput.style.height = 'auto';

    updateUserText(txt || `[Uploaded: ${pendingFile ? pendingFile.name : 'file'}]`);
    setAIStatus('ANALYZING');   // ✅ immediate feedback on typed submit
    logToCLI(`⌨️ User (typed): ${txt || finalText}`);

    if (pendingFile) clearFileBadge();
    if (useWebSearch) {
        setWebSearchArmed(false, { announce: false });
        logToCLI("Web search enabled for this turn.");
    }
    closeComposerMenu();
}


// ============================================================
// RENDER PIPELINE (unchanged)
// ============================================================
function enqueueRender(op) {
    typingQueue.push(op);
    if (!typingActive) processRenderQueue();
}

// Board sync: when tutor finishes speaking, speed-flush remaining queued animations
// so board text never lags 30-60s behind the audio
function flushBoardQueueInstant() {
    typingQueue.forEach(function(op) {
        if (op.mode === 'type') op.cps = 600;
    });
}

function processRenderQueue() {
    if (typingQueue.length === 0) { typingActive = false; flushPendingFx(); return; }
    typingActive = true;
    const op = typingQueue.shift();
    const doneCb = (typeof op.done === "function") ? op.done : null;
    if (op.mode === "type") {
        typewriteApply(op.text, op.append, op.cps, () => {
            if (op.stick) smartScrollToBottom(true);
            if (doneCb) doneCb();
            processRenderQueue();
        });
    } else {
        if (!op.append) displayBuffer = op.text;
        else displayBuffer += op.text;
        renderBoard(displayBuffer);
        if (op.stick) smartScrollToBottom(true);
        if (doneCb) doneCb();
        processRenderQueue();
    }
}

function typewriteApply(text, append, cps, done) {
    if (!append) displayBuffer = "";
    const start = displayBuffer;
    let i = 0;
    const delay = Math.max(5, Math.floor(1000 / Math.max(5, cps)));
    function tick() {
        displayBuffer = start + text.slice(0, i);
        renderBoard(displayBuffer);
        smartScrollToBottom(false);
        i++;
        if (i <= text.length) setTimeout(tick, delay);
        else { displayBuffer = start + text; renderBoard(displayBuffer); smartScrollToBottom(false); done(); }
    }
    tick();
}

function renderBoard(raw) {
    let html = raw.replace(/\*\*(.*?)\*\*/g, '<b>$1</b>');
    blackboardText.innerHTML = html;
    if (window.renderMathInElement) {
        renderMathInElement(blackboardText, {
            delimiters: [
                { left: '$$', right: '$$', display: true },
                { left: '\\[', right: '\\]', display: true },
                { left: '$',  right: '$',  display: false },
                { left: '\\(', right: '\\)', display: false }
            ],
            throwOnError: false
        });
    }
}

function flushPendingFx() {
    if (!pendingFxQueue.length) return;
    const remaining = [];
    pendingFxQueue.forEach(a => { if (!tryApplyFxWhenVisible(a)) remaining.push(a); });
    pendingFxQueue = remaining;
}

function tryApplyFxWhenVisible(action) {
    const target = (action.target || "").trim();
    if (!target) return true;
    if (!isTargetVisibleNow(target)) return false;
    applyFx(action);
    return true;
}

function isTargetVisibleNow(target) {
    if (target === "[]") return displayBuffer.includes("[") && displayBuffer.includes("]");
    return displayBuffer.includes(target);
}

function applyFx(action) {
    const type     = (action.type || "glow").toLowerCase();
    const target   = (action.target || "").trim();
    const duration = action.duration_ms || 900;
    if (!target) return;
    if (target === "[]") {
        highlightLastMatch("[", type === "pop" ? "fx-pop" : "fx-glow", duration);
        highlightLastMatch("]", type === "pop" ? "fx-pop" : "fx-glow", duration);
        return;
    }
    highlightLastMatch(target, type === "pop" ? "fx-pop" : "fx-glow", duration);
}

function highlightLastMatch(textToFind, cssClass, durationMs) {
    if (!textToFind) return;
    const raw = displayBuffer;
    const idx = raw.lastIndexOf(textToFind);
    if (idx < 0) return;
    const before = raw.slice(0, idx);
    const match  = raw.slice(idx, idx + textToFind.length);
    const after  = raw.slice(idx + textToFind.length);
    blackboardText.innerHTML = (before + `<span class="${cssClass}" data-fx="1">` + match + `</span>` + after).replace(/\*\*(.*?)\*\*/g, '<b>$1</b>');
    smartScrollToBottom(false);
    setTimeout(() => { renderBoard(displayBuffer); smartScrollToBottom(false); }, durationMs);
}

// ✅ CLI toggle
const cliToggleBar   = document.getElementById('cli-toggle-bar');
const cliToggleArrow = document.getElementById('cli-toggle-arrow');
const cliTerminalWrap = document.getElementById('cli-terminal-wrap');
let cliOpen = false;

if (cliToggleBar) {
    cliToggleBar.addEventListener('click', function(e) {
        e.stopPropagation();
        cliOpen = !cliOpen;
        cliTerminalWrap.classList.toggle('open', cliOpen);
        cliToggleArrow.classList.toggle('open', cliOpen);
    });
}

function logToCLI(msg) {
    const time = new Date().toLocaleTimeString();
    const line = document.createElement('div');
    line.textContent = `[${time}] ${repairDisplayText(msg)}`;
    cliOutput.appendChild(line);
    cliOutput.scrollTop = cliOutput.scrollHeight;
}

function repairDisplayText(value) {
    if (value == null) return '';
    let text = String(value);
    if (!/[ÃÂâð]/.test(text)) return text;
    try {
        text = decodeURIComponent(escape(text));
    } catch (e) {}
    return text
        .replace(/Â°C/g, '°C')
        .replace(/â†’/g, '→')
        .replace(/â†©ï¸/g, '↩️')
        .replace(/â†©/g, '↩')
        .replace(/â€”/g, '—')
        .replace(/â€¢/g, '•')
        .replace(/âœ…/g, '✅')
        .replace(/âœ¨/g, '✨')
        .replace(/âœ“/g, '✓')
        .replace(/âœ—/g, '✗')
        .replace(/âŒ/g, '❌')
        .replace(/âš ï¸/g, '⚠️')
        .replace(/â°/g, '⏰')
        .replace(/ðŸŽ¤/g, '🎤')
        .replace(/ðŸŽµ/g, '🎵')
        .replace(/ðŸ”„/g, '🔄')
        .replace(/ðŸ”Š/g, '🔊')
        .replace(/ðŸ“‹/g, '📋')
        .replace(/ðŸ“Œ/g, '📌')
        .replace(/ðŸ“š/g, '📚')
        .replace(/ðŸ”Œ/g, '🔌')
        .replace(/ðŸ§ /g, '🧠')
        .replace(/ðŸ—£ï¸/g, '🗣️')
        .replace(/ðŸ’¥/g, '💥')
        .replace(/ðŸ«€/g, '🧠');
}


// ============================================================
// ✅ SKY + WEATHER SYSTEM
// API Key: 0f320af0b243e54e3b2c0986259ea820
// ============================================================
const OWM_KEY = "0f320af0b243e54e3b2c0986259ea820";

// ── Sky phases definition ─────────────────────────────────────
// gradients are [top-color, bottom-color], all hand-crafted for beauty
const SKY_PHASES = {
    dawn: {         // 5:00–6:30
        label: "🌅 Dawn",
        emoji: "🌅",
        gradient: ["#1a1a3e", "#4a2060", "#e8604c", "#f4a261"],
        textColor: "#fff3e0",
        stars: true,
        starsFade: true,
    },
    morning: {      // 6:30–11:00
        label: "🌤️ Morning",
        emoji: "☀️",
        gradient: ["#4fc3f7", "#81d4fa", "#e1f5fe", "#fff9f0"],
        textColor: "#01579b",
        clouds: true,
        sunRise: true,
    },
    afternoon: {    // 11:00–17:00
        label: "☀️ Afternoon",
        emoji: "☀️",
        gradient: ["#0288d1", "#29b6f6", "#81d4fa", "#e3f2fd"],
        textColor: "#01579b",
        clouds: true,
        sunHigh: true,
    },
    evening: {      // 17:00–19:30
        label: "🌇 Evening",
        emoji: "🌇",
        gradient: ["#1a237e", "#4a148c", "#e91e63", "#ff9800", "#ffd54f"],
        textColor: "#fff8e1",
        clouds: true,
        sunSet: true,
    },
    night: {        // 19:30–23:00
        label: "🌙 Night",
        emoji: "🌙",
        gradient: ["#0a0a1a", "#0d1b3e", "#1a237e", "#283593"],
        textColor: "#e8eaf6",
        stars: true,
        moon: true,
    },
    nightOwl: {     // 23:00–5:00
        label: "🌌 Night Owl",
        emoji: "🌌",
        gradient: ["#020408", "#060d1a", "#0a1628", "#0d1f3c"],
        textColor: "#b0bec5",
        stars: true,
        shootingStars: true,
        moon: true,
    },
};

// Weather overlay modifiers
const WEATHER_OVERLAYS = {
    Rain: { type: "rain", label: "🌧️ Raining" },
    Drizzle: { type: "rain", light: true, label: "🌦️ Drizzle" },
    Thunderstorm: { type: "storm", label: "⛈️ Storm" },
    Snow: { type: "snow", label: "🌨️ Snow" },
    Clouds: { type: "clouds", label: "☁️ Cloudy" },
    Mist: { type: "mist", label: "🌫️ Mist" },
    Fog: { type: "mist", label: "🌫️ Foggy" },
    Haze: { type: "mist", label: "🌫️ Hazy" },
    Clear: { type: null, label: "☀️ Clear" },
};

// ── Canvas setup ──────────────────────────────────────────────
const skyCanvas = document.getElementById('sky-canvas');
const skyCtx = skyCanvas ? skyCanvas.getContext('2d') : null;

let weatherCanvas = null;
let weatherCtx = null;
let weatherOverlay = document.getElementById('weather-overlay');
if (!weatherOverlay) {
    weatherOverlay = document.createElement('canvas');
    weatherOverlay.id = 'weather-overlay';
    weatherOverlay.style.cssText = 'position:fixed;inset:0;pointer-events:none;z-index:0;width:100%;height:100%;';
    document.body.appendChild(weatherOverlay);
}
weatherCanvas = weatherOverlay;
weatherCtx = weatherCanvas.getContext('2d');

function resizeCanvases() {
    [skyCanvas, weatherCanvas].forEach(c => {
        if (!c) return;
        c.width  = window.innerWidth;
        c.height = window.innerHeight;
    });
}
resizeCanvases();
window.addEventListener('resize', resizeCanvases);

// ── State ─────────────────────────────────────────────────────
let skyState = {
    phase: "morning",
    weatherType: null,
    weatherLabel: "Clear",
    userTZ: "Asia/Kolkata",
    lat: null,
    lon: null,
    cityName: "",
    lastWeatherFetch: 0,
    animFrame: null,
    weatherAnimFrame: null,
    // Particle pools
    stars: [],
    shootingStars: [],
    rainDrops: [],
    snowFlakes: [],
    clouds: [],
    lightning: 0,
};

// ── Time phase calculator ─────────────────────────────────────
function getCurrentPhase(tz) {
    const now = new Date();
    const localStr = now.toLocaleString("en-US", { timeZone: tz, hour: "numeric", hour12: false });
    const hour = parseInt(localStr, 10);
    if (hour >= 5   && hour < 7)  return "dawn";
    if (hour >= 7   && hour < 11) return "morning";
    if (hour >= 11  && hour < 17) return "afternoon";
    if (hour >= 17  && hour < 20) return "evening";
    if (hour >= 20  && hour < 23) return "night";
    return "nightOwl";
}

// ── Draw sky gradient ─────────────────────────────────────────
function drawSkyGradient(phase) {
    if (!skyCtx) return;
    const cfg = SKY_PHASES[phase];
    const stops = cfg.gradient;
    const W = skyCanvas.width, H = skyCanvas.height;
    const grad = skyCtx.createLinearGradient(0, 0, 0, H);
    stops.forEach((c, i) => grad.addColorStop(i / (stops.length - 1), c));
    skyCtx.fillStyle = grad;
    skyCtx.fillRect(0, 0, W, H);
}

// ── Stars ─────────────────────────────────────────────────────
function initStars() {
    skyState.stars = [];
    for (let i = 0; i < 220; i++) {
        skyState.stars.push({
            x: Math.random(),
            y: Math.random() * 0.7,
            r: Math.random() * 1.6 + 0.3,
            a: Math.random(),
            twinkleSpeed: 0.008 + Math.random() * 0.015,
            twinkleDir: Math.random() > 0.5 ? 1 : -1,
        });
    }
}

function drawStars(fade) {
    if (!skyCtx) return;
    const W = skyCanvas.width, H = skyCanvas.height;
    skyState.stars.forEach(s => {
        s.a += s.twinkleSpeed * s.twinkleDir;
        if (s.a > 1 || s.a < 0.2) s.twinkleDir *= -1;
        const alpha = fade ? s.a * 0.4 : s.a;
        skyCtx.beginPath();
        skyCtx.arc(s.x * W, s.y * H, s.r, 0, Math.PI * 2);
        skyCtx.fillStyle = `rgba(255,255,245,${alpha})`;
        skyCtx.fill();
    });
}

// ── Shooting stars ────────────────────────────────────────────
function initShootingStars() {
    skyState.shootingStars = [];
}

function spawnShootingStar() {
    if (skyState.shootingStars.length > 4) return;
    skyState.shootingStars.push({
        x: Math.random() * skyCanvas.width,
        y: Math.random() * skyCanvas.height * 0.5,
        len: 80 + Math.random() * 120,
        speed: 8 + Math.random() * 12,
        angle: Math.PI / 4 + (Math.random() - 0.5) * 0.3,
        alpha: 1,
        life: 1,
    });
}

function drawShootingStars() {
    if (!skyCtx) return;
    if (Math.random() < 0.008) spawnShootingStar();
    skyState.shootingStars = skyState.shootingStars.filter(s => s.life > 0);
    skyState.shootingStars.forEach(s => {
        const dx = Math.cos(s.angle) * s.len;
        const dy = Math.sin(s.angle) * s.len;
        const grad = skyCtx.createLinearGradient(s.x, s.y, s.x + dx, s.y + dy);
        grad.addColorStop(0, `rgba(255,255,255,${s.life})`);
        grad.addColorStop(1, "rgba(255,255,255,0)");
        skyCtx.beginPath();
        skyCtx.moveTo(s.x, s.y);
        skyCtx.lineTo(s.x + dx, s.y + dy);
        skyCtx.strokeStyle = grad;
        skyCtx.lineWidth = 1.5;
        skyCtx.stroke();
        s.x += Math.cos(s.angle) * s.speed;
        s.y += Math.sin(s.angle) * s.speed;
        s.life -= 0.025;
    });
}

// ── Moon ─────────────────────────────────────────────────────
function drawMoon() {
    if (!skyCtx) return;
    const W = skyCanvas.width, H = skyCanvas.height;
    const mx = W * 0.82, my = H * 0.12, mr = 28;
    // glow
    const glow = skyCtx.createRadialGradient(mx, my, mr * 0.5, mx, my, mr * 3.5);
    glow.addColorStop(0, "rgba(220,220,180,0.18)");
    glow.addColorStop(1, "rgba(220,220,180,0)");
    skyCtx.fillStyle = glow;
    skyCtx.fillRect(mx - mr * 4, my - mr * 4, mr * 8, mr * 8);
    // moon body
    skyCtx.beginPath();
    skyCtx.arc(mx, my, mr, 0, Math.PI * 2);
    skyCtx.fillStyle = "#f5f0d8";
    skyCtx.fill();
    // crescent shadow
    skyCtx.beginPath();
    skyCtx.arc(mx + mr * 0.3, my, mr * 0.88, 0, Math.PI * 2);
    const phase = SKY_PHASES[skyState.phase];
    const shadowCol = phase ? phase.gradient[0] : "#0a0a1a";
    skyCtx.fillStyle = shadowCol;
    skyCtx.fill();
}

// ── Sun ───────────────────────────────────────────────────────
function drawSun(yFrac, size) {
    if (!skyCtx) return;
    const W = skyCanvas.width, H = skyCanvas.height;
    const sx = W * 0.15, sy = H * yFrac;
    const glow = skyCtx.createRadialGradient(sx, sy, size * 0.3, sx, sy, size * 4);
    glow.addColorStop(0, "rgba(255,255,200,0.45)");
    glow.addColorStop(0.4, "rgba(255,220,100,0.15)");
    glow.addColorStop(1, "rgba(255,200,50,0)");
    skyCtx.fillStyle = glow;
    skyCtx.fillRect(sx - size * 5, sy - size * 5, size * 10, size * 10);
    skyCtx.beginPath();
    skyCtx.arc(sx, sy, size, 0, Math.PI * 2);
    skyCtx.fillStyle = "#fff5cc";
    skyCtx.fill();
}

// ── Clouds ───────────────────────────────────────────────────
function initClouds() {
    if (skyState.clouds.length > 0) return;
    for (let i = 0; i < 6; i++) {
        skyState.clouds.push({
            x: Math.random(),
            y: 0.05 + Math.random() * 0.3,
            w: 0.15 + Math.random() * 0.18,
            speed: 0.00003 + Math.random() * 0.00005,
            alpha: 0.5 + Math.random() * 0.4,
            bumps: Math.floor(3 + Math.random() * 4),
        });
    }
}

function drawClouds(dark) {
    if (!skyCtx) return;
    const W = skyCanvas.width, H = skyCanvas.height;
    skyState.clouds.forEach(c => {
        c.x += c.speed;
        if (c.x > 1.2) c.x = -0.2;
        const cx = c.x * W, cy = c.y * H, cw = c.w * W;
        const ch = cw * 0.38;
        const base = dark ? `rgba(80,80,90,${c.alpha * 0.75})` : `rgba(255,255,255,${c.alpha})`;
        skyCtx.fillStyle = base;
        // draw puffball cloud
        for (let b = 0; b < c.bumps; b++) {
            const bx = cx + (b / c.bumps) * cw - cw * 0.5;
            const by = cy - Math.sin((b / (c.bumps - 1)) * Math.PI) * ch * 0.55;
            const br = ch * (0.45 + Math.sin((b / (c.bumps - 1)) * Math.PI) * 0.35);
            skyCtx.beginPath();
            skyCtx.arc(bx, by, br, 0, Math.PI * 2);
            skyCtx.fill();
        }
        // base rectangle
        skyCtx.fillRect(cx - cw * 0.5, cy - ch * 0.22, cw, ch * 0.55);
    });
}

// ── Rain ─────────────────────────────────────────────────────
let rainTick = 0;
function initRain(light) {
    skyState.rainDrops = [];
    const count = light ? 80 : 220;
    for (let i = 0; i < count; i++) {
        skyState.rainDrops.push({
            x: Math.random(),
            y: Math.random(),
            speed: 0.008 + Math.random() * 0.012,
            len: 8 + Math.random() * 14,
            alpha: 0.3 + Math.random() * 0.5,
        });
    }
}

function drawRain() {
    if (!weatherCtx) return;
    const W = weatherCanvas.width, H = weatherCanvas.height;
    weatherCtx.clearRect(0, 0, W, H);
    weatherCtx.strokeStyle = "rgba(180,210,255,0.55)";
    weatherCtx.lineWidth = 1;
    skyState.rainDrops.forEach(d => {
        d.y += d.speed;
        d.x += d.speed * 0.18;
        if (d.y > 1) { d.y = -0.05; d.x = Math.random(); }
        weatherCtx.globalAlpha = d.alpha;
        weatherCtx.beginPath();
        weatherCtx.moveTo(d.x * W, d.y * H);
        weatherCtx.lineTo(d.x * W + 2, d.y * H + d.len);
        weatherCtx.stroke();
    });
    weatherCtx.globalAlpha = 1;
    // dark rain overlay
    const rainGrad = weatherCtx.createLinearGradient(0, 0, 0, H);
    rainGrad.addColorStop(0, "rgba(20,30,60,0.15)");
    rainGrad.addColorStop(1, "rgba(20,30,60,0.08)");
    weatherCtx.fillStyle = rainGrad;
    weatherCtx.fillRect(0, 0, W, H);
}

// ── Snow ─────────────────────────────────────────────────────
function initSnow() {
    skyState.snowFlakes = [];
    for (let i = 0; i < 120; i++) {
        skyState.snowFlakes.push({
            x: Math.random(), y: Math.random(),
            r: 1.5 + Math.random() * 3.5,
            speed: 0.002 + Math.random() * 0.004,
            drift: (Math.random() - 0.5) * 0.001,
            alpha: 0.5 + Math.random() * 0.5,
        });
    }
}

function drawSnow() {
    if (!weatherCtx) return;
    const W = weatherCanvas.width, H = weatherCanvas.height;
    weatherCtx.clearRect(0, 0, W, H);
    skyState.snowFlakes.forEach(s => {
        s.y += s.speed;
        s.x += s.drift;
        if (s.y > 1) { s.y = -0.02; s.x = Math.random(); }
        weatherCtx.beginPath();
        weatherCtx.arc(s.x * W, s.y * H, s.r, 0, Math.PI * 2);
        weatherCtx.fillStyle = `rgba(255,255,255,${s.alpha})`;
        weatherCtx.fill();
    });
}

// ── Storm lightning ───────────────────────────────────────────
function drawStorm() {
    if (!weatherCtx) return;
    drawRain();
    skyState.lightning -= 1;
    if (Math.random() < 0.004) skyState.lightning = 6;
    if (skyState.lightning > 0) {
        weatherCtx.fillStyle = `rgba(220,220,255,${skyState.lightning * 0.06})`;
        weatherCtx.fillRect(0, 0, weatherCanvas.width, weatherCanvas.height);
    }
}

// ── Mist ────────────────────────────────────────────────────
function drawMist() {
    if (!weatherCtx) return;
    const W = weatherCanvas.width, H = weatherCanvas.height;
    weatherCtx.clearRect(0, 0, W, H);
    const grad = weatherCtx.createLinearGradient(0, H * 0.4, 0, H);
    grad.addColorStop(0, "rgba(200,210,220,0)");
    grad.addColorStop(0.5, "rgba(200,210,220,0.32)");
    grad.addColorStop(1, "rgba(200,210,220,0.55)");
    weatherCtx.fillStyle = grad;
    weatherCtx.fillRect(0, 0, W, H);
}

// ── Cloudy overlay ───────────────────────────────────────────
function drawCloudyOverlay() {
    if (!weatherCtx) return;
    weatherCtx.clearRect(0, 0, weatherCanvas.width, weatherCanvas.height);
    // extra dark clouds already handled in sky drawClouds(dark=true)
}

// ── Main animation loop ───────────────────────────────────────
function skyTick() {
    if (!skyCtx) return;
    const W = skyCanvas.width, H = skyCanvas.height;
    const phase = skyState.phase;
    const cfg   = SKY_PHASES[phase];

    skyCtx.clearRect(0, 0, W, H);
    drawSkyGradient(phase);

    if (cfg.stars)        drawStars(cfg.starsFade || false);
    if (cfg.moon)         drawMoon();
    if (cfg.sunRise)      drawSun(0.72, 28);
    if (cfg.sunHigh)      drawSun(0.12, 32);
    if (cfg.sunSet)       drawSun(0.82, 26);
    if (cfg.shootingStars) drawShootingStars();

    const cloudDark = (skyState.weatherType === "clouds" || skyState.weatherType === "storm" || skyState.weatherType === "rain");
    if (cfg.clouds || cloudDark) {
        initClouds();
        drawClouds(cloudDark);
    }

    // Weather overlays
    switch (skyState.weatherType) {
        case "rain":  drawRain(); break;
        case "storm": drawStorm(); break;
        case "snow":  drawSnow(); break;
        case "mist":  drawMist(); break;
        case "clouds": drawCloudyOverlay(); break;
        default:
            if (weatherCtx) weatherCtx.clearRect(0, 0, weatherCanvas.width, weatherCanvas.height);
    }

    skyState.animFrame = requestAnimationFrame(skyTick);
}

// ── Weather fetch ─────────────────────────────────────────────
async function fetchWeather(lat, lon) {
    try {
        const url = `https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lon}&appid=${OWM_KEY}&units=metric`;
        const res  = await fetch(url);
        const data = await res.json();
        const main = data.weather?.[0]?.main || "Clear";
        const desc = data.weather?.[0]?.description || "";
        const temp = Math.round(data.main?.temp ?? 0);
        const city = data.name || "";
        skyState.cityName = city;

        const overlay = WEATHER_OVERLAYS[main] || WEATHER_OVERLAYS.Clear;
        skyState.weatherType = overlay.type;
        skyState.weatherLabel = overlay.label;
        skyState.lastWeatherFetch = Date.now();

        const statusEl = document.getElementById('profile-weather-status');
        if (statusEl) statusEl.textContent = `${overlay.label} · ${temp}°C · ${city}`;

        // Init particle pools for weather type
        if (overlay.type === "rain")  initRain(main === "Drizzle");
        if (overlay.type === "storm") initRain(false);
        if (overlay.type === "snow")  initSnow();

        updateWeatherStatus();
        logToCLI(`🌤 Weather: ${overlay.label} ${temp}°C in ${city}`);
    } catch(e) {
        const statusEl = document.getElementById('profile-weather-status');
        if (statusEl) statusEl.textContent = "⚠️ Weather fetch failed";
    }
}

// ── Update sky time-pill ──────────────────────────────────────
function updateWeatherStatus() {
    const pill = document.getElementById('sky-time-pill');
    const cfg  = SKY_PHASES[skyState.phase];
    if (pill) pill.textContent = cfg.emoji;
}

// ── Geolocation ───────────────────────────────────────────────
function geoLocate() {
    if (!navigator.geolocation) {
        setManualLocation("Asia/Kolkata");
        return;
    }
    navigator.geolocation.getCurrentPosition(
        pos => {
            skyState.lat = pos.coords.latitude;
            skyState.lon = pos.coords.longitude;
            const statusEl = document.getElementById('profile-weather-status');
            if (statusEl) statusEl.textContent = "📍 Location acquired, fetching weather...";
            fetchWeather(skyState.lat, skyState.lon);
        },
        () => {
            // fallback to timezone-based rough coords
            setManualLocation(skyState.userTZ);
        },
        { timeout: 8000 }
    );
}

// Rough lat/lon from known TZs for weather API fallback
const TZ_COORDS = {
    "Asia/Kolkata":         [20.5937, 78.9629],
    "America/New_York":     [40.7128, -74.0060],
    "America/Los_Angeles":  [34.0522, -118.2437],
    "Europe/London":        [51.5074, -0.1278],
    "Europe/Paris":         [48.8566, 2.3522],
    "Asia/Dubai":           [25.2048, 55.2708],
    "Asia/Singapore":       [1.3521, 103.8198],
    "Asia/Tokyo":           [35.6762, 139.6503],
    "Australia/Sydney":     [-33.8688, 151.2093],
};

function setManualLocation(tz) {
    skyState.userTZ = tz;
    skyState.phase  = getCurrentPhase(tz);
    const coords = TZ_COORDS[tz] || [20.59, 78.96];
    skyState.lat = coords[0];
    skyState.lon = coords[1];

    const statusEl = document.getElementById('profile-weather-status');
    if (statusEl) statusEl.textContent = "🌐 Using timezone location, fetching weather...";
    fetchWeather(skyState.lat, skyState.lon);

    const metaEl = document.getElementById('profile-meta');
    if (metaEl) {
        const age = document.getElementById('pf-age')?.value || "21";
        const tzName = tz.split("/")[1]?.replace("_", " ") || tz;
        metaEl.textContent = `Age ${age} · 📍 ${tzName}`;
    }
}

// ── Profile panel interactions ────────────────────────────────
const profileCard     = document.getElementById('profile-card');
const profileSubPanel = document.getElementById('profile-sub-panel');
const pfLocation      = document.getElementById('pf-location');
const pfName          = document.getElementById('pf-name');
const pfAge           = document.getElementById('pf-age');

if (profileCard) {
    profileCard.addEventListener('click', function(e) {
        e.stopPropagation();
        profileSubPanel.classList.toggle('visible');
    });
}

if (pfLocation) {
    pfLocation.addEventListener('change', function() {
        const tz = pfLocation.value;
        if (tz === "auto") {
            geoLocate();
        } else {
            setManualLocation(tz);
        }
    });
}

if (pfName) {
    pfName.addEventListener('input', function() {
        const nameEl = document.getElementById('profile-name');
        const avatarEl = document.getElementById('profile-avatar');
        const n = pfName.value.trim() || "S";
        if (nameEl) nameEl.textContent = n;
        if (avatarEl) avatarEl.textContent = n[0].toUpperCase();
    });
}

if (pfAge) {
    pfAge.addEventListener('input', function() {
        const metaEl = document.getElementById('profile-meta');
        if (metaEl) {
            const tz = pfLocation?.value || "Asia/Kolkata";
            const tzName = tz === "auto" ? "Auto" : tz.split("/")[1]?.replace("_", " ") || tz;
            metaEl.textContent = `Age ${pfAge.value} · 📍 ${tzName}`;
        }
    });
}

// ── Phase refresh every 5 min ─────────────────────────────────
setInterval(() => {
    const newPhase = getCurrentPhase(skyState.userTZ);
    if (newPhase !== skyState.phase) {
        skyState.phase = newPhase;
        skyState.clouds = [];
        updateWeatherStatus();
    }
    // Refresh weather every 10 min
    if (Date.now() - skyState.lastWeatherFetch > 600000 && skyState.lat) {
        fetchWeather(skyState.lat, skyState.lon);
    }
}, 300000);

// ══════════════════════════════════════════════════════════════
// ✅ EMPATHY SYSTEM — Real GPS tracking + profile sync
// ══════════════════════════════════════════════════════════════

let empathy = {
    realLat:  10.7905,
    realLon:  78.7047,
    realCity: "Tiruchirappalli",
    realTZ:   Intl.DateTimeFormat().resolvedOptions().timeZone || "Asia/Kolkata",
    temp:     30,
    weather:  "Clear",
};

// Get real GPS location silently — completely separate from display location
function syncRealGPS() {
    if (!navigator.geolocation) return;
    navigator.geolocation.getCurrentPosition(
        async pos => {
            empathy.realLat = pos.coords.latitude;
            empathy.realLon = pos.coords.longitude;
            empathy.realTZ  = Intl.DateTimeFormat().resolvedOptions().timeZone;
            // Fetch weather for real coords
            try {
                const url = `https://api.openweathermap.org/data/2.5/weather?lat=${empathy.realLat}&lon=${empathy.realLon}&appid=${OWM_KEY}&units=metric`;
                const d = await (await fetch(url)).json();
                empathy.temp    = Math.round(d.main?.temp ?? 30);
                empathy.weather = d.weather?.[0]?.main || "Clear";
                empathy.realCity= d.name || empathy.realCity;
            } catch(e) {}
            pushEmpathyProfile();
        },
        () => { pushEmpathyProfile(); },  // fallback: send what we have
        { timeout: 8000, maximumAge: 300000 }
    );
}

function pushEmpathyProfile() {
    const name = document.getElementById('pf-name')?.value?.trim() || "Student";
    const age  = document.getElementById('pf-age')?.value  || "20";
    const email = document.getElementById('pf-email')?.value?.trim() || "";
    socket.emit("update_empathy_profile", {
        name:     name,
        age:      parseInt(age),
        email:    email,
        student_id: window.__activeStudentId || "",
        real_lat: empathy.realLat,
        real_lon: empathy.realLon,
        city:     empathy.realCity,
        tz:       empathy.realTZ,
        temp:     empathy.temp,
        weather:  empathy.weather,
    });
}

// Push profile when name/age changes
document.getElementById('pf-name')?.addEventListener('input', () => { pushEmpathyProfile(); saveSettings(); });
document.getElementById('pf-age')?.addEventListener('input',  () => { pushEmpathyProfile(); saveSettings(); });

// Sync GPS on boot + every 5 min
syncRealGPS();
setInterval(syncRealGPS, 5 * 60 * 1000);


// ══════════════════════════════════════════════════════════════
// ✅ BREAK TIMER SYSTEM
// ══════════════════════════════════════════════════════════════

let breakState = {
    active:       false,
    startTs:      0,
    countdownSecs: 300,   // 5 min countdown
    timerEl:      null,
    intervalId:   null,
    overflowing:  false,   // true when counting UP after 5 min
};

const BREAK_ALARM_START_URL  = null;  // we generate tones via AudioContext
const BREAK_ALARM_RETURN_URL = null;

function playBreakTone(type) {
    // type: "start" = soft chime | "end" = sharp ping
    try {
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const osc = ctx.createOscillator();
        const gain= ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);
        if (type === "start") {
            osc.frequency.value = 528;   // calm, soothing C5
            gain.gain.setValueAtTime(0.6, ctx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 1.2);
            osc.start();
            osc.stop(ctx.currentTime + 1.2);
        } else {
            // "end" — 3 short sharp pings
            [0, 0.25, 0.5].forEach(delay => {
                const o2 = ctx.createOscillator();
                const g2 = ctx.createGain();
                o2.connect(g2); g2.connect(ctx.destination);
                o2.frequency.value = 880;  // A5 — sharp ping
                g2.gain.setValueAtTime(0.8, ctx.currentTime + delay);
                g2.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + delay + 0.3);
                o2.start(ctx.currentTime + delay);
                o2.stop(ctx.currentTime + delay + 0.3);
            });
        }
    } catch(e) {}
}

function startBreakTimer() {
    if (breakState.active) return;
    breakState.active      = true;
    breakState.startTs     = Date.now();
    breakState.overflowing = false;
    breakState.countdownSecs = 300;

    playBreakTone("start");
    socket.emit("break_started");

    // Build timer overlay
    const overlay = document.createElement('div');
    overlay.id    = 'break-overlay';
    overlay.innerHTML = `
        <div class="break-card" id="break-card">
            <div class="break-label">☕ Break Time</div>
            <div class="break-timer" id="break-timer-display">5:00</div>
            <div class="break-sub" id="break-sub">Relax — come back in 5 minutes</div>
            <button class="break-stop-btn" id="break-stop-btn">▶ I'm Back</button>
        </div>`;
    document.body.appendChild(overlay);
    breakState.timerEl = overlay;

    // Animate card in
    requestAnimationFrame(() => { overlay.classList.add('visible'); });

    document.getElementById('break-stop-btn').addEventListener('click', stopBreakTimer);

    // Tick every second
    breakState.intervalId = setInterval(() => {
        const elapsed = Math.floor((Date.now() - breakState.startTs) / 1000);
        const disp = document.getElementById('break-timer-display');
        const sub  = document.getElementById('break-sub');

        if (elapsed < 300) {
            // Counting DOWN
            const rem = 300 - elapsed;
            const m = Math.floor(rem / 60);
            const s = rem % 60;
            if (disp) disp.textContent = `${m}:${String(s).padStart(2,'0')}`;
            if (sub)  sub.textContent  = "Relax — come back in " + m + (m===1?" minute":" minutes");
        } else {
            // 5 min done — play alarm once, then count UP
            if (!breakState.overflowing) {
                breakState.overflowing = true;
                playBreakTone("end");
                if (disp) { disp.style.color = "#ff6b6b"; }
                if (sub)  sub.textContent = "Break over! Tap when ready.";
            }
            const over = elapsed - 300;
            const m = Math.floor(over / 60);
            const s = over % 60;
            if (disp) disp.textContent = `+${m}:${String(s).padStart(2,'0')}`;
        }
    }, 1000);
}

function stopBreakTimer() {
    if (!breakState.active) return;
    const actualSecs = Math.floor((Date.now() - breakState.startTs) / 1000);
    breakState.active = false;
    clearInterval(breakState.intervalId);

    // Animate out
    const overlay = document.getElementById('break-overlay');
    if (overlay) {
        overlay.classList.add('hiding');
        overlay.addEventListener('transitionend', () => overlay.remove(), { once: true });
        setTimeout(() => { if (overlay.parentNode) overlay.remove(); }, 600);
    }
    breakState.timerEl = null;

    socket.emit("break_ended", { actual_secs: actualSecs });
}

// ── Listen for break offer from server ──────────────────────
socket.on('empathy_nudge', function(data) {
    if (data.type === 'break_offer') {
        // Show a subtle "Take Break" button for 30 seconds
        showBreakOfferButton();
    }
});

// Fix Issue 3: Restore break overlay when page reconnects mid-break
socket.on('break_restore', function(data) {
    if (!data || !data.active) return;
    if (breakState.active) return;  // already running
    // Rebuild break state from server
    breakState.active = true;
    breakState.startTs = Date.now() - (data.elapsed_secs || 0) * 1000;
    breakState.overflowing = (data.elapsed_secs || 0) >= 300;
    // Rebuild overlay
    if (document.getElementById('break-overlay')) return;
    const overlay = document.createElement('div');
    overlay.id = 'break-overlay';
    overlay.innerHTML = `
        <div class="break-card" id="break-card">
            <div class="break-label">☕ Break Time</div>
            <div class="break-timer" id="break-timer-display">5:00</div>
            <div class="break-sub" id="break-sub">Relax — come back in 5 minutes</div>
            <button class="break-stop-btn" id="break-stop-btn">▶ I'm Back</button>
        </div>`;
    document.body.appendChild(overlay);
    requestAnimationFrame(() => { overlay.classList.add('visible'); });
    document.getElementById('break-stop-btn').addEventListener('click', stopBreakTimer);
    breakState.timerEl = overlay;
    // Restart tick
    breakState.intervalId = setInterval(() => {
        const elapsed = Math.floor((Date.now() - breakState.startTs) / 1000);
        const disp = document.getElementById('break-timer-display');
        const sub  = document.getElementById('break-sub');
        if (elapsed < 300) {
            const rem = 300 - elapsed;
            const m = Math.floor(rem / 60), s = rem % 60;
            if (disp) disp.textContent = `${m}:${String(s).padStart(2,'0')}`;
            if (sub)  sub.textContent  = "Relax — come back in " + m + (m===1?" minute":" minutes");
        } else {
            if (!breakState.overflowing) {
                breakState.overflowing = true;
                playBreakTone("end");
                if (disp) disp.style.color = "#ff6b6b";
                if (sub)  sub.textContent = "Break over! Tap when ready.";
            }
            const over = elapsed - 300;
            const m = Math.floor(over / 60), s = over % 60;
            if (disp) disp.textContent = `+${m}:${String(s).padStart(2,'0')}`;
        }
    }, 1000);
    logToCLI("☕ Break overlay restored after reconnect");
});

// Fix Issue 3: Server auto-expired break — remove overlay if visible
socket.on('break_auto_ended', function() {
    if (breakState.active) stopBreakTimer();
    const existing = document.getElementById('break-overlay');
    if (existing) existing.remove();
    logToCLI("⏰ Break auto-expired by server");
});

function showBreakOfferButton() {
    const existing = document.getElementById('break-offer-btn');
    if (existing) return;
    const btn = document.createElement('button');
    btn.id    = 'break-offer-btn';
    btn.className = 'break-offer-float';
    btn.textContent = "☕ Take Break";
    btn.onclick = () => { btn.remove(); startBreakTimer(); };
    document.body.appendChild(btn);
    // Auto-hide after 30s if ignored
    setTimeout(() => { if (btn.parentNode) btn.remove(); }, 30000);
}

// ── Boot ──────────────────────────────────────────────────────
(function bootSky() {
    // Fix 7: Load persisted settings FIRST
    loadSettings();
    applyLearningModeUI(currentLearningMode);

    initStars();
    initShootingStars();
    skyState.userTZ = "Asia/Kolkata";
    skyState.phase  = getCurrentPhase("Asia/Kolkata");
    updateWeatherStatus();

    // Auto-geolocate on load
    geoLocate();

    // Start animation loop
    skyTick();

    logToCLI(`🌤 Sky: ${SKY_PHASES[skyState.phase].label} — fetching weather...`);
})();

// ============================================================
// ✅ LEFT SIDEBAR — User panel + Log panel
// ============================================================
(function initSidebar() {
    const sidebar      = document.getElementById('left-sidebar');
    const toggleBtn    = document.getElementById('sb-toggle-btn');
    const tabs         = document.querySelectorAll('.sb-tab');
    const panelUser    = document.getElementById('sb-panel-user');
    const panelLog     = document.getElementById('sb-panel-log');
    const logBody      = document.getElementById('sb-log-body');
    const clearLogBtn  = document.getElementById('sb-log-clear-btn');

    // ── Sidebar open/close ────────────────────────────────────
    let sbOpen = true;

    function setSidebarOpen(open, options = {}) {
        const remember = options.remember !== false;
        sbOpen = !!open;
        sidebar.classList.toggle('collapsed', !sbOpen);
        document.body.classList.toggle('sb-open', sbOpen);

        if (remember && !document.body.classList.contains('board-fullscreen')) {
            restoreSidebarOpen = sbOpen;
        }
    }

    setSidebarOpenState = setSidebarOpen;
    setSidebarOpen(true, { remember: true });

    toggleBtn.addEventListener('click', function(e) {
        e.stopPropagation();
        if (document.body.classList.contains('board-fullscreen')) return;
        setSidebarOpen(!sbOpen);
    });

    // ── Tab switching ─────────────────────────────────────────
    tabs.forEach(tab => {
        tab.addEventListener('click', function() {
            tabs.forEach(t => t.classList.remove('active'));
            this.classList.add('active');
            const which = this.dataset.tab;
            panelUser.style.display = which === 'user' ? 'flex' : 'none';
            panelLog.style.display  = which === 'log'  ? 'flex'  : 'none';
            if (which === 'log') panelLog.style.flexDirection = 'column';
        });
    });

    // ── Clear log ─────────────────────────────────────────────
    clearLogBtn.addEventListener('click', function() {
        logBody.innerHTML = '';
    });

    // ── Sync profile from settings panel ─────────────────────
    function syncProfile() {
        const name = (document.getElementById('pf-name') || {}).value || 'Student';
        const age  = (document.getElementById('pf-age')  || {}).value  || '';
        const nameEl = document.getElementById('sb-profile-name') || document.getElementById('sb-bottom-name');
        const metaEl = document.getElementById('sb-profile-meta') || document.getElementById('sb-bottom-meta');
        const avatarEl = document.getElementById('sb-avatar') || document.getElementById('sb-bottom-avatar');
        if (nameEl) nameEl.textContent = name;
        if (metaEl) metaEl.textContent = age ? 'Age ' + age : '';
        if (avatarEl) avatarEl.textContent = name.charAt(0).toUpperCase();
    }
    const pfName = document.getElementById('pf-name');
    const pfAge  = document.getElementById('pf-age');
    if (pfName) pfName.addEventListener('input', syncProfile);
    if (pfAge)  pfAge.addEventListener('input',  syncProfile);
    syncProfile();

    // ── Indicator helpers ─────────────────────────────────────
    function setIndicator(id, state, statusText) {
        const dots = [
            document.getElementById('ind-' + id + '-dot'),
            document.getElementById('log-ind-' + id + '-dot'),
        ].filter(Boolean);
        const statuses = [
            document.getElementById('ind-' + id + '-status'),
            document.getElementById('log-ind-' + id + '-status'),
        ].filter(Boolean);
        if (!dots.length && !statuses.length) return;
        dots.forEach(function(dot) {
            dot.className = 'sb-ind-dot ' + (state || '');
        });
        if (statusText !== undefined) {
            statuses.forEach(function(status) {
                status.textContent = statusText;
            });
        }
    }

    // ── Log entry builder ─────────────────────────────────────
    const MAX_LOG_ENTRIES = 200;

    function fmt_ms(ms) {
        if (ms >= 1000) return (ms / 1000).toFixed(2) + 's';
        return ms + 'ms';
    }

    function addLogEntry(icon, text, variant, badge, badgeColor) {
        const entry = document.createElement('div');
        entry.className = 'sb-log-entry sbv-' + (variant || 'info');
        icon = repairDisplayText(icon || '');
        text = repairDisplayText(text || '');
        badge = badge ? repairDisplayText(badge) : '';

        const now = new Date();
        const ts  = now.toTimeString().slice(0, 8) + '.' + String(now.getMilliseconds()).padStart(3, '0');

        let badgeHtml = '';
        if (badge) {
            badgeHtml = `<span class="sb-log-badge" style="background:${badgeColor||'rgba(255,255,255,0.1)'};color:#fff">${badge}</span>`;
        }

        entry.innerHTML = `
            <div class="sb-log-time">${ts}</div>
            <div class="sb-log-text"><span class="sb-log-icon">${icon}</span>${text}${badgeHtml}</div>
        `;
        logBody.prepend(entry);  // newest on top

        // Trim
        while (logBody.children.length > MAX_LOG_ENTRIES) {
            logBody.removeChild(logBody.lastChild);
        }
    }

    // ── Timing state (for summary) ─────────────────────────────
    const timingAcc = { llm_ms: 0, tts_ms: 0, speak_ms: 0 };

    function updateSummary() {
        [document.getElementById('st-llm'), document.getElementById('log-st-llm')].filter(Boolean).forEach(function(el) {
            el.textContent = timingAcc.llm_ms ? fmt_ms(timingAcc.llm_ms) : '???';
        });
        [document.getElementById('st-tts'), document.getElementById('log-st-tts')].filter(Boolean).forEach(function(el) {
            el.textContent = timingAcc.tts_ms ? fmt_ms(timingAcc.tts_ms) : '???';
        });
        [document.getElementById('st-spk'), document.getElementById('log-st-spk')].filter(Boolean).forEach(function(el) {
            el.textContent = timingAcc.speak_ms ? fmt_ms(timingAcc.speak_ms) : '???';
        });
    }

    // ── Socket: service_status ────────────────────────────────
    socket.on('service_status', function(data) {
        if (data.stt) setIndicator('stt', 'idle', 'Ready');
        if (data.llm) setIndicator('llm', 'idle', 'Idle');
        if (data.tts) setIndicator('tts', 'idle', 'Idle');
        addLogEntry('✅', 'Services online', 'info', 'BOOT', '#388e3c');
    });

    // ── Socket: timing_event ──────────────────────────────────
    socket.on('timing_event', function(ev) {
        const type = ev.type || '';

        if (type === 'stt_received') {
            const preview = (ev.text || '').slice(0, 50);
            setIndicator('stt', 'busy', 'Received');
            addLogEntry('🎤', `STT → "${preview}"`, 'stt', 'STT', '#1565c0');
            // Reset timing acc for new turn
            timingAcc.llm_ms = 0; timingAcc.tts_ms = 0; timingAcc.speak_ms = 0;
            updateSummary();
            setIndicator('stt', 'ok', 'Got text');
        }

        else if (type === 'llm_start') {
            const lbl = ev.label || 'LLM';
            setIndicator('llm', 'busy', 'Running…');
            addLogEntry('🧠', `LLM: ${lbl}`, 'llm', 'START', '#f57f17');
        }

        else if (type === 'llm_done') {
            const ms  = ev.duration_ms || 0;
            const lbl = ev.label || 'LLM';
            const tok = ev.tokens || '';
            timingAcc.llm_ms += ms;
            setIndicator('llm', 'idle', 'Idle');
            addLogEntry('✅', `LLM done: ${lbl} — ${fmt_ms(ms)}${tok ? ' (~' + tok + ' tok)' : ''}`,
                        'llm', fmt_ms(ms), '#f9a825');
            updateSummary();
        }

        else if (type === 'tts_start') {
            const chunk = ev.chunk !== undefined ? '#' + ev.chunk : '';
            const prev  = (ev.preview || '').slice(0, 40);
            setIndicator('tts', 'busy', 'Generating…');
            addLogEntry('🔄', `TTS gen ${chunk}: "${prev}"`, 'tts', 'GEN', '#6a1b9a');
        }

        else if (type === 'tts_gen_done') {
            const ms   = ev.duration_ms || 0;
            const ok   = ev.ok !== false;
            const chunk = ev.chunk !== undefined ? '#' + ev.chunk : '';
            timingAcc.tts_ms += ms;
            setIndicator('tts', ok ? 'idle' : 'error', ok ? 'Ready' : 'Error');
            addLogEntry(ok ? '🎵' : '❌',
                        `TTS ${chunk} ready — ${fmt_ms(ms)}`,
                        'tts', fmt_ms(ms), '#7b1fa2');
            updateSummary();
        }

        else if (type === 'speak_start') {
            const chunk = ev.chunk !== undefined ? '#' + ev.chunk : '';
            setIndicator('tts', 'busy', 'Speaking…');
            addLogEntry('🔊', `▶ Speaking chunk ${chunk}`, 'speak', 'PLAY', '#1b5e20');
        }

        else if (type === 'speak_end') {
            const ms   = ev.duration_ms || 0;
            const chunk = ev.chunk !== undefined ? '#' + ev.chunk : '';
            timingAcc.speak_ms += ms;
            setIndicator('tts', 'idle', 'Done');
            addLogEntry('⏹', `■ Chunk ${chunk} done — ${fmt_ms(ms)}`, 'speak', fmt_ms(ms), '#2e7d32');
            updateSummary();
        }

        else if (type === 'board_text') {
            const ts      = new Date().toTimeString().slice(0, 8);
            const preview = (ev.preview || '').slice(0, 55);
            const chars   = ev.chars || 0;
            addLogEntry('📋', `[${ts}] BOARD: ${preview}`, 'info', `${chars}c`, '#37474f');
        }
    });

    // FX animation timing in sidebar log
    socket.on('fx_timing_log', function(data) {
        if (!data) return;
        const icon = data.type === 'glow' ? '✨' : '💥';
        const badge = data.ts ? data.ts.slice(6) : '';
        const color = data.type === 'glow' ? '#e65100' : '#6a1b9a';
        addLogEntry(icon, `FX ${(data.type || '').toUpperCase()}: "${(data.target || '').slice(0, 45)}"`, 'info', badge, color);
    });

    // ── Sync indicators with AI status updates ─────────────────
    // Patch setAIStatus to also update indicators
    const _origSetAIStatus = window.setAIStatus || function(){};
    window.setAIStatus = function(state) {
        _origSetAIStatus(state);
        if (state === 'ANALYZING') {
            setIndicator('llm', 'busy', 'Running…');
        } else if (state === 'SPEAKING') {
            setIndicator('tts', 'busy', 'Speaking…');
            setIndicator('llm', 'idle', 'Idle');
        } else if (state === 'LISTENING') {
            setIndicator('stt', 'idle', 'Listening');
            setIndicator('tts', 'idle', 'Idle');
            setIndicator('llm', 'idle', 'Idle');
        } else {
            setIndicator('stt', 'idle', 'Ready');
            setIndicator('tts', 'idle', 'Idle');
            setIndicator('llm', 'idle', 'Idle');
        }
    };

    // ── Log generic cli_log messages into sidebar log too ──────
    socket.on('cli_log', function(data) {
        const msg = repairDisplayText((data && data.msg) || '');
        if (!msg) return;
        let normalizedIcon = '📋', normalizedVariant = 'info';
        if (msg.startsWith('❌')) normalizedIcon = '❌';
        else if (msg.startsWith('✅')) normalizedIcon = '✅';
        else if (msg.startsWith('🎤')) { normalizedIcon = '🎤'; normalizedVariant = 'stt'; }
        else if (msg.startsWith('🗣️')) { normalizedIcon = '🗣️'; normalizedVariant = 'tts'; }
        else if (msg.startsWith('🧠')) { normalizedIcon = '🧠'; normalizedVariant = 'llm'; }
        const normalizedText = msg.replace(/^\S+\s*/, '').trim() || msg;
        addLogEntry(normalizedIcon, normalizedText, normalizedVariant);
        return;
        // Decide icon
        let icon = '📋', variant = 'info';
        if (msg.startsWith('❌')) { icon = '❌'; variant = 'info'; }
        else if (msg.startsWith('✅')) { icon = '✅'; }
        else if (msg.startsWith('🎤')) { icon = '🎤'; variant = 'stt'; }
        else if (msg.startsWith('🗣️')) { icon = '🗣️'; variant = 'tts'; }
        else if (msg.startsWith('🧠')) { icon = '🧠'; variant = 'llm'; }
        addLogEntry(icon, msg.slice(2).trim() || msg, variant);
    });

    // ── Socket connect/disconnect status ──────────────────────
    socket.on('connect', function() {
        addLogEntry('🔌', 'Connected to server', 'info', 'WS', '#388e3c');
        setIndicator('stt', 'idle', 'Ready');
    });
    socket.on('disconnect', function() {
        addLogEntry('⚠️', 'Disconnected from server', 'info', 'WS', '#c62828');
        setIndicator('stt', 'error', 'Disconnected');
        setIndicator('llm', 'error', 'Disconnected');
        setIndicator('tts', 'error', 'Disconnected');
    });

})();

function renderReportEnhancements(report) {
    if (!report) return;
    ensureReportDownloadControls();
    const body = document.querySelector(".session-report-modal .sr-body");
    if (!body) return;
    let panel = document.getElementById("sr-empathy-panel");
    if (!panel) {
        panel = document.createElement("div");
        panel.id = "sr-empathy-panel";
        panel.className = "sr-empathy-panel";
        const grid = body.querySelector(".sr-charts-grid");
        if (grid) body.insertBefore(panel, grid);
        else body.appendChild(panel);
    }
    const summary = (report.empathy_summary || []).map(function(line) {
        return `<li>${escapeHtml(line)}</li>`;
    }).join("");
    const toneTimeline = (report.tone_timeline || []).slice(0, 6).map(function(item) {
        const minute = item.t != null ? `${item.t}m` : "Turn";
        const cue = item.cue ? ` - ${escapeHtml(item.cue)}` : "";
        return `<div class="sr-tone-item"><span>${escapeHtml(minute)}</span><strong>${escapeHtml(item.tone || 'supportive')}</strong><em>${cue}</em></div>`;
    }).join("");
    const emailStatus = report.email_status && report.email_status.sent
        ? 'Email delivery: sent'
        : (report.student_email ? `Email delivery: ${escapeHtml((report.email_status || {}).reason || 'pending')}` : 'Email delivery: no email saved');
    panel.innerHTML = '' +
        '<div class="sr-empathy-card">' +
            '<div class="sr-chart-title">Empathy Impact</div>' +
            `<ul class="sr-empathy-list">${summary || '<li>No empathy summary available.</li>'}</ul>` +
            `<div class="sr-email-status">${emailStatus}</div>` +
        '</div>' +
        '<div class="sr-empathy-card">' +
            '<div class="sr-chart-title">Tutor Tone Timeline</div>' +
            `<div class="sr-tone-list">${toneTimeline || '<div class="sr-tone-item"><span>Turn</span><strong>supportive</strong><em></em></div>'}</div>` +
        '</div>';
}

(function initStudentAccountsAndCourseProgress() {
    let studentStateCache = null;
    let courseProgressState = { mode: 'shallow', topics: [], completed_topics: [], current_topic: '', percent: 0 };
    window.__qaCourseProgressHidden = false;

    function ensureProfileEmailField() {
        const panel = document.getElementById('profile-sub-panel');
        if (!panel || document.getElementById('pf-email')) return;
        const field = document.createElement('div');
        field.className = 'profile-field';
        field.innerHTML = '<label>Email</label><input type="email" id="pf-email" class="pf-input" placeholder="Optional"/>';
        const locationField = document.getElementById('pf-location')?.closest('.profile-field');
        if (locationField) locationField.insertAdjacentElement('beforebegin', field);
        else panel.appendChild(field);
        field.querySelector('#pf-email')?.addEventListener('input', function() {
            const email = (this.value || '').trim();
            socket.emit('update_empathy_profile', {
                name: document.getElementById('pf-name')?.value?.trim() || 'Student',
                age: parseInt(document.getElementById('pf-age')?.value || '20', 10),
                email: email,
            });
        });
    }

    function ensureStudentModal() {
        let modal = document.getElementById('student-login-modal');
        if (modal) return modal;
        modal = document.createElement('div');
        modal.id = 'student-login-modal';
        modal.className = 'student-login-modal';
        modal.innerHTML = '' +
            '<div class="student-login-card">' +
                '<div class="student-login-title">Student Login</div>' +
                '<div class="student-login-sub">Create or switch a student account. Session memory stays separate per student.</div>' +
                '<input id="student-login-name" class="student-login-input" type="text" placeholder="Name" />' +
                '<input id="student-login-age" class="student-login-input" type="number" placeholder="Age" min="3" max="120" />' +
                '<input id="student-login-email" class="student-login-input" type="email" placeholder="Email (optional)" />' +
                '<button id="student-login-submit" class="student-login-submit" type="button">Continue</button>' +
                '<div id="student-login-existing" class="student-login-existing"></div>' +
            '</div>';
        document.body.appendChild(modal);
        modal.querySelector('#student-login-submit').addEventListener('click', function() {
            const name = modal.querySelector('#student-login-name').value.trim();
            const age = parseInt(modal.querySelector('#student-login-age').value || '20', 10);
            const email = modal.querySelector('#student-login-email').value.trim();
            if (!name) return;
            const payload = { name, age, email };
            if (modal.dataset.mode === 'edit' && modal.dataset.studentId) {
                payload.student_id = modal.dataset.studentId;
            }
            socket.emit('student_login', payload);
        });
        return modal;
    }

    function toggleStudentModal(show) {
        const modal = ensureStudentModal();
        modal.classList.toggle('visible', !!show);
    }

    function openStudentModal(mode, active) {
        const modal = ensureStudentModal();
        const title = modal.querySelector('.student-login-title');
        const subtitle = modal.querySelector('.student-login-sub');
        const submit = modal.querySelector('#student-login-submit');
        const nameInput = modal.querySelector('#student-login-name');
        const ageInput = modal.querySelector('#student-login-age');
        const emailInput = modal.querySelector('#student-login-email');
        const existing = modal.querySelector('#student-login-existing');
        const profile = active || (studentStateCache && studentStateCache.active) || null;
        const isEdit = mode === 'edit' && profile && profile.student_id && profile.student_id !== 'guest_default';

        modal.dataset.mode = isEdit ? 'edit' : 'login';
        modal.dataset.studentId = isEdit ? String(profile.student_id || '') : '';
        if (title) title.textContent = isEdit ? 'Edit Profile' : 'Student Login';
        if (subtitle) {
            subtitle.textContent = isEdit
                ? 'Update the saved student details for this device.'
                : 'Create or switch a student account. Session memory stays separate per student.';
        }
        if (submit) submit.textContent = isEdit ? 'Save Profile' : 'Continue';
        if (nameInput) nameInput.value = isEdit ? String(profile.name || '') : '';
        if (ageInput) ageInput.value = isEdit ? String(profile.age != null ? profile.age : '') : '';
        if (emailInput) emailInput.value = isEdit ? String(profile.email || '') : '';
        if (existing) {
            existing.innerHTML = '';
            existing.style.display = 'none';
        }
        toggleStudentModal(true);
    }

    function cleanupUserSidebarDiagnostics() {
        const sidebar = document.getElementById('left-sidebar');
        const tabs = sidebar ? sidebar.querySelector('.sb-tabs') : null;
        const userPanel = document.getElementById('sb-panel-user');
        const logPanel = document.getElementById('sb-panel-log');
        const footer = document.querySelector('.sb-profile-bottom');
        const advanced = document.querySelector('.sb-advanced-settings');
        if (!userPanel) return;

        if (sidebar && tabs) {
            let panels = sidebar.querySelector('.sb-panels');
            if (!panels) {
                panels = document.createElement('div');
                panels.className = 'sb-panels';
                tabs.insertAdjacentElement('afterend', panels);
            }
            if (userPanel.parentElement !== panels) panels.appendChild(userPanel);
            if (advanced && advanced.parentElement !== userPanel) userPanel.appendChild(advanced);
            if (logPanel && logPanel.parentElement !== panels) panels.appendChild(logPanel);
            if (footer && footer.parentElement !== sidebar) sidebar.appendChild(footer);
        }

        Array.from(userPanel.querySelectorAll('.sb-section-label')).forEach(function(label) {
            const text = String(label.textContent || '').trim().toUpperCase();
            if (text !== 'SERVICES' && text !== 'LAST TURN') return;
            const next = label.nextElementSibling;
            if (text === 'SERVICES' && next && next.classList.contains('sb-indicators')) next.remove();
            if (text === 'LAST TURN' && next && next.classList.contains('sb-timing-summary')) next.remove();
            label.remove();
        });
    }

    function folderLabel(folderPath) {
        const raw = String(folderPath || '').trim().replace(/[\\/]+$/, '');
        if (!raw) return '';
        const parts = raw.split(/[\\/]/).filter(Boolean);
        return parts.length ? parts[parts.length - 1] : raw;
    }

    function folderMemoryPath(folderPath) {
        const raw = String(folderPath || '').trim().replace(/\//g, '\\');
        if (!raw) return '';
        const marker = 'student_login\\';
        const idx = raw.toLowerCase().indexOf(marker);
        return idx >= 0 ? raw.slice(idx) : raw;
    }

    function syncProfileChrome(active) {
        if (!active) return;
        const name = String(active.name || 'Student').trim() || 'Student';
        const age = active.age != null ? String(active.age) : '';
        const initial = name.charAt(0).toUpperCase();
        const profileName = document.getElementById('profile-name');
        const profileMeta = document.getElementById('profile-meta');
        const profileAvatar = document.getElementById('profile-avatar');
        const bottomName = document.getElementById('sb-bottom-name');
        const bottomAvatar = document.getElementById('sb-bottom-avatar');
        if (profileName) profileName.textContent = name;
        if (profileMeta) profileMeta.textContent = age ? `Age ${age}` : 'Student';
        if (profileAvatar) profileAvatar.textContent = initial;
        if (bottomName) bottomName.textContent = name;
        if (bottomAvatar) bottomAvatar.textContent = initial;
    }

    function syncStudentInputs(active) {
        if (!active) return;
        window.__activeStudentId = active.student_id || 'guest_default';
        const nameInput = document.getElementById('pf-name');
        const ageInput = document.getElementById('pf-age');
        const emailInput = document.getElementById('pf-email');
        if (nameInput && active.name) nameInput.value = active.name;
        if (ageInput && active.age != null) ageInput.value = active.age;
        if (emailInput) emailInput.value = active.email || '';
        syncProfileChrome(active);
        if (typeof saveSettings === 'function') saveSettings();
        const bottomMeta = document.getElementById('sb-bottom-meta');
        if (bottomMeta) {
            bottomMeta.textContent = active.age ? `Age ${active.age}` : 'Student';
        }
        renderRecentsList();
    }

    function renderStudentPopup(state) {
        const popup = document.getElementById('sb-profile-popup');
        if (!popup) return;
        let list = document.getElementById('sb-student-list');
        let switchBtn = document.getElementById('sb-pop-switch-user');
        const profileBtn = document.getElementById('sb-pop-profile');
        const divider = popup.querySelector('.sb-popup-divider');
        const logoutBtn = Array.from(popup.querySelectorAll('.sb-popup-item.danger')).find(function(node) {
            return /log out/i.test(node.textContent || '');
        });
        if (!switchBtn) {
            switchBtn = document.createElement('div');
            switchBtn.id = 'sb-pop-switch-user';
            switchBtn.className = 'sb-popup-item';
            switchBtn.innerHTML = '<span class="sb-popup-item-icon">⇄</span> Switch User';
            popup.insertBefore(switchBtn, profileBtn || divider || logoutBtn || null);
        }
        switchBtn.innerHTML = '<span class="sb-popup-item-icon">&#8646;</span> Switch User';
        popup.insertBefore(switchBtn, profileBtn || divider || logoutBtn || null);
        if (!list) {
            list = document.createElement('div');
            list.id = 'sb-student-list';
            list.className = 'sb-student-list';
            popup.insertBefore(list, divider || logoutBtn || null);
        }
        popup.insertBefore(list, divider || logoutBtn || null);
        const active = state && state.active ? state.active : null;
        const users = ((state && state.students) || []).filter(function(user) {
            return String(user.student_id || '') !== 'guest_default';
        });
        list.innerHTML = users.map(function(user) {
            const activeClass = active && active.student_id === user.student_id ? ' active' : '';
            return `<button class="sb-student-entry${activeClass}" data-student-id="${escapeHtml(user.student_id || '')}">${escapeHtml(user.name || 'Student')}</button>`;
        }).join('') || '<div class="sb-student-empty">No logged-in profiles yet</div>';
        list.innerHTML += '<button class="sb-student-entry sb-student-add" data-add-student="1">+ Add User / Member</button>';
        list.classList.remove('visible');
        switchBtn.classList.remove('is-open');
        if (!switchBtn.dataset.boundToggle) {
            switchBtn.dataset.boundToggle = '1';
            switchBtn.addEventListener('click', function(ev) {
                ev.stopPropagation();
                const open = !list.classList.contains('visible');
                list.classList.toggle('visible', open);
                switchBtn.classList.toggle('is-open', open);
            });
        }
        list.querySelectorAll('[data-student-id]').forEach(function(btn) {
            btn.addEventListener('click', function(ev) {
                ev.stopPropagation();
                list.classList.remove('visible');
                switchBtn.classList.remove('is-open');
                socket.emit('switch_student', { student_id: btn.dataset.studentId });
            });
        });
        list.querySelectorAll('[data-add-student]').forEach(function(btn) {
            btn.addEventListener('click', function(ev) {
                ev.stopPropagation();
                list.classList.remove('visible');
                switchBtn.classList.remove('is-open');
                popup.classList.remove('visible');
                openStudentModal('login');
            });
        });
        if (profileBtn && !profileBtn.dataset.boundProfile) {
            profileBtn.dataset.boundProfile = '1';
            profileBtn.addEventListener('click', function(ev) {
                ev.stopPropagation();
                popup.classList.remove('visible');
                openStudentModal('edit', studentStateCache && studentStateCache.active);
            });
        }
        if (logoutBtn && !logoutBtn.dataset.boundLogout) {
            logoutBtn.dataset.boundLogout = '1';
            logoutBtn.addEventListener('click', function(ev) {
                ev.stopPropagation();
                if (!window.confirm('Log out from this student profile?')) return;
                popup.classList.remove('visible');
                socket.emit('student_logout', {});
                toggleStudentModal(true);
            });
        }
        const modal = ensureStudentModal();
        const existing = modal.querySelector('#student-login-existing');
        existing.innerHTML = '';
        existing.style.display = 'none';
    }

    function handleStudentState(state) {
        studentStateCache = state || null;
        if (state && state.active) {
            syncStudentInputs(state.active);
            renderStudentPopup(state);
            const shouldPromptLogin = !state.active.student_id || state.active.student_id === 'guest_default';
            toggleStudentModal(shouldPromptLogin);
        }
    }

    function ensureCourseProgressUI() {
        if (document.getElementById('course-progress-orb')) return;
        const orb = document.createElement('div');
        orb.id = 'course-progress-orb';
        orb.className = 'course-progress-orb';
        orb.innerHTML = '' +
            '<svg viewBox="0 0 120 120" class="course-progress-svg">' +
                '<circle cx="60" cy="60" r="48" class="course-progress-track"></circle>' +
                '<circle cx="60" cy="60" r="48" class="course-progress-value" id="course-progress-value"></circle>' +
            '</svg>' +
            '<div class="course-progress-label" id="course-progress-label">0%</div>';
        const scroll = document.createElement('div');
        scroll.id = 'course-progress-scroll';
        scroll.className = 'course-progress-scroll';
        const toast = document.createElement('div');
        toast.id = 'course-progress-toast';
        toast.className = 'course-progress-toast';
        document.body.appendChild(orb);
        document.body.appendChild(scroll);
        document.body.appendChild(toast);
        orb.addEventListener('mouseenter', function() {
            if ((courseProgressState.topics || []).length) {
                renderCourseProgressScroll();
                scroll.classList.add('visible');
            }
        });
        orb.addEventListener('mouseleave', function() {
            scroll.classList.remove('visible');
        });
    }

    function renderCourseProgressScroll() {
        const scroll = document.getElementById('course-progress-scroll');
        if (!scroll) return;
        const completed = new Set(courseProgressState.completed_topics || []);
        scroll.innerHTML = '<div class="course-scroll-title">Course Topics</div>' +
            (courseProgressState.topics || []).map(function(topic) {
                const done = completed.has(topic);
                return `<div class="course-scroll-topic${done ? ' done' : ''}">${escapeHtml(topic)}</div>`;
            }).join('');
    }

    function renderCourseProgress(payload) {
        courseProgressState = Object.assign({ mode: 'shallow', topics: [], completed_topics: [], percent: 0 }, payload || {});
        ensureCourseProgressUI();
        const orb = document.getElementById('course-progress-orb');
        const label = document.getElementById('course-progress-label');
        const value = document.getElementById('course-progress-value');
        if (!orb || !label || !value) return;
        const topics = courseProgressState.topics || [];
        const visible = courseProgressState.mode === 'course' && topics.length > 0 && !window.__qaCourseProgressHidden;
        orb.classList.toggle('visible', visible);
        const percent = Math.max(0, Math.min(100, Number(courseProgressState.percent || 0)));
        orb.classList.toggle('is-complete', percent >= 100);
        label.textContent = `${percent}%`;
        const circumference = 2 * Math.PI * 48;
        const filled = circumference * (percent / 100);
        value.style.strokeDasharray = `${filled} ${circumference}`;
        renderCourseProgressScroll();
    }

    window.__setCourseProgressHidden = function(hidden) {
        window.__qaCourseProgressHidden = !!hidden;
        renderCourseProgress(courseProgressState);
    };

    function showCourseCompletionToast(topic) {
        ensureCourseProgressUI();
        const toast = document.getElementById('course-progress-toast');
        if (!toast) return;
        toast.innerHTML = `<div class="course-progress-toast-title">Topic Complete</div><div class="course-progress-toast-topic">${escapeHtml(topic || 'Topic')}</div>`;
        toast.classList.remove('visible', 'struck');
        void toast.offsetWidth;
        toast.classList.add('visible');
        setTimeout(function() { toast.classList.add('struck'); }, 900);
        setTimeout(function() { toast.classList.remove('visible', 'struck'); }, 2600);
    }

    cleanupUserSidebarDiagnostics();

    document.addEventListener('DOMContentLoaded', function() {
        ensureProfileEmailField();
        cleanupUserSidebarDiagnostics();
        ensureStudentModal();
        ensureCourseProgressUI();
        ensureReportDownloadControls();
        if (typeof applyLearningModeUI === 'function' && !window.__modeFilteredRecentsPatchApplied) {
            const baseApplyLearningModeUI = applyLearningModeUI;
            applyLearningModeUI = function(mode) {
                const result = baseApplyLearningModeUI(mode);
                setTimeout(renderRecentsList, 0);
                return result;
            };
            window.__modeFilteredRecentsPatchApplied = true;
        }
        socket.emit('request_student_state');
    });

    socket.on('student_state', handleStudentState);
    socket.on('student_login_result', function(data) {
        if (data && data.ok && data.student) syncStudentInputs(data.student);
        if (data && data.ok) {
            toggleStudentModal(false);
            socket.emit('request_student_state');
        }
    });
    socket.on('student_switch_result', function(data) {
        if (data && data.ok && data.student) syncStudentInputs(data.student);
        if (data && data.ok) {
            toggleStudentModal(false);
            socket.emit('request_student_state');
        }
    });
    socket.on('course_progress', renderCourseProgress);
    socket.on('course_topic_completed', function(payload) {
        renderCourseProgress(payload);
        showCourseCompletionToast(payload && payload.completed_topic);
    });
    socket.on('session_reset', function(data) {
        renderCourseProgress({ mode: ((data && data.mode) || currentLearningMode || 'shallow'), topics: [], completed_topics: [], percent: 0 });
    });
})();
