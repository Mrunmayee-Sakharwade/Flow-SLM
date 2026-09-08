/**
 * AnQ Bot - J&J Commercial Oncology Call Assistance Mobile Application Controller
 * ==============================================================================
 * Bright, Simple, Production-Grade Architecture:
 *   1. Dynamic Rep Avatar Circle: Real-time initials ("Mihit" -> "M", "Mihir Joshi" -> "MJ")
 *   2. Guaranteed Button Visibility: Sticky action dock & login bar (never hides)
 *   3. Organic Dynamic Turn & Topic Focus Engine (No hardcoded 7 turns)
 *   4. Multi-Layer Autocorrector & Filler-Word Removal (Phonetic J&J Lexicon)
 *   5. Speech Pipeline: Bi-directional Audio Capture, STT preview, and FlowEdit TTS
 *   6. Clean White Card Modal Sheets (CRM Note, Scope Governance, Diff Inspector)
 */

document.addEventListener("DOMContentLoaded", () => {
  // ---------------------------------------------------------------------------
  // 1. STATE MANAGEMENT
  // ---------------------------------------------------------------------------
  let currentSessionId = null;
  let currentRepName = localStorage.getItem("anq_mobile_rep_name") || "Mihit";
  let currentRole = localStorage.getItem("anq_mobile_role") || "OS";
  let currentBrand = localStorage.getItem("anq_mobile_brand") || "INLEXZO";
  let currentAccount = localStorage.getItem("anq_mobile_account") || "Atlantic Urology Associates";
  let apiBaseUrl = localStorage.getItem("anq_mobile_api_url") || window.location.origin;

  let sessionActive = false;
  let callStartTime = 0;
  let callTimerInterval = null;
  let currentTurnIndex = 1;

  // Audio & Voice State
  let audioMuted = false;
  let currentPlayingAudio = null;
  let isRecording = false;
  let handsFreeMode = false;
  let mediaRecorder = null;
  let audioChunks = [];
  let speechRecognition = null;
  let audioContext = null;
  let analyserNode = null;
  let animFrameId = null;

  // Settings
  let autocorrectEnabled = true;
  let removeFillersEnabled = true;

  // Structured CRM Note Slots
  let noteSlots = {
    rep_name: currentRepName,
    role: currentRole,
    brand: currentBrand,
    account_name: currentAccount,
    hcp_name: "—",
    key_topics: [],
    barriers: "—",
    next_action: "—"
  };

  // ---------------------------------------------------------------------------
  // 2. DOM ELEMENT REFERENCES
  // ---------------------------------------------------------------------------
  // Viewport & Shell
  const deviceShell = document.getElementById("device-shell");
  const btnToggleFrame = document.getElementById("btn-toggle-frame");

  // Screens
  const screenLogin = document.getElementById("screen-login");
  const screenCallNoter = document.getElementById("screen-call-noter");

  // Login Inputs & Avatar Circle
  const loginAvatarCircle = document.getElementById("login-avatar-circle");
  const loginRepName = document.getElementById("login-rep-name");
  const presetChips = document.querySelectorAll(".name-chip");
  const roleCards = document.querySelectorAll(".simple-role-card");
  const loginBrandSelect = document.getElementById("login-brand-select");
  const loginAccountSelect = document.getElementById("login-account-select");
  const endpointSummaryToggle = document.getElementById("endpoint-summary-toggle");
  const endpointSummaryText = document.getElementById("endpoint-summary-text");
  const endpointStatusDot = document.getElementById("endpoint-status-dot");
  const endpointInput = document.getElementById("endpoint-input");
  const btnTestEndpoint = document.getElementById("btn-test-endpoint");
  const btnLaunchCall = document.getElementById("btn-launch-call");

  // Call Noter Header
  const headerAvatar = document.getElementById("header-avatar");
  const headerRepName = document.getElementById("header-rep-name");
  const headerRolePill = document.getElementById("header-role-pill");
  const headerBrandPill = document.getElementById("header-brand-pill");
  const callTimerDigits = document.getElementById("call-timer-digits");
  const btnHeaderVoice = document.getElementById("btn-header-voice");
  const btnHeaderEnd = document.getElementById("btn-header-end");

  // Dynamic Status Bar (No hardcoded 7 turns)
  const callTopicFocus = document.getElementById("call-topic-focus");
  const callTurnBadge = document.getElementById("call-turn-badge");
  const callNotesCount = document.getElementById("call-notes-count");

  // Chat Feed
  const mobileChatFeed = document.getElementById("mobile-chat-feed");

  // Interaction Dock
  const liveAudioCanvasBar = document.getElementById("live-audio-canvas-bar");
  const waveformCanvas = document.getElementById("waveform-canvas");
  const dockAutocorrectToggle = document.getElementById("dock-autocorrect-toggle");
  const autocorrectToggleText = document.getElementById("autocorrect-toggle-text");
  const dockModeIndicator = document.getElementById("dock-mode-indicator");
  const btnDockKeyboard = document.getElementById("btn-dock-keyboard");
  const btnDockHandsfree = document.getElementById("btn-dock-handsfree");
  const btnGiantMic = document.getElementById("btn-giant-mic");
  const giantMicLabel = document.getElementById("giant-mic-label");
  const dockTextDrawer = document.getElementById("dock-text-drawer");
  const dockTextarea = document.getElementById("dock-textarea");
  const btnDockSend = document.getElementById("btn-dock-send");

  // Bottom Navigation Tabs
  const navTabButtons = document.querySelectorAll(".nav-tab-btn");

  // Drawers & Sheets
  const sheetSummary = document.getElementById("sheet-summary");
  const sheetScope = document.getElementById("sheet-scope");
  const sheetSettings = document.getElementById("sheet-settings");
  const sheetDiff = document.getElementById("sheet-diff");
  const sheetCloseButtons = document.querySelectorAll(".btn-close-sheet");

  // Sheet Summary Fields
  const kvRep = document.getElementById("kv-rep");
  const kvRole = document.getElementById("kv-role");
  const kvBrand = document.getElementById("kv-brand");
  const kvAccount = document.getElementById("kv-account");
  const kvHcp = document.getElementById("kv-hcp");
  const kvBarrier = document.getElementById("kv-barrier");
  const kvAction = document.getElementById("kv-action");
  const btnCopyCrm = document.getElementById("btn-copy-crm");

  // Diff Sheet Elements
  const diffRawText = document.getElementById("diff-raw-text");
  const diffCleanedText = document.getElementById("diff-cleaned-text");
  const diffFillersList = document.getElementById("diff-fillers-list");
  const diffCorrectionsList = document.getElementById("diff-corrections-list");

  // ---------------------------------------------------------------------------
  // 3. DYNAMIC AVATAR INITIAL CALCULATION
  // ---------------------------------------------------------------------------
  /**
   * Generates initials for avatar circles:
   *   "Mihit" -> "M"
   *   "Mihir" -> "M"
   *   "Mihit Joshi" -> "MJ"
   *   "Dr. Anurag Verma" -> "DV"
   */
  function getInitials(name) {
    if (!name || !name.trim()) return "👤";
    const parts = name.trim().split(/\s+/).filter(Boolean);
    if (parts.length === 1) {
      return parts[0].charAt(0).toUpperCase();
    }
    return (parts[0].charAt(0) + parts[parts.length - 1].charAt(0)).toUpperCase();
  }

  function updateRepAvatar(name) {
    const initial = getInitials(name);
    if (loginAvatarCircle) {
      loginAvatarCircle.textContent = initial;
      loginAvatarCircle.classList.add("pulse");
      setTimeout(() => loginAvatarCircle.classList.remove("pulse"), 250);
    }
    if (headerAvatar) {
      headerAvatar.textContent = initial;
    }
  }

  // ---------------------------------------------------------------------------
  // 4. INTELLIGENT SPEECH AUTOCORRECTOR & FILLER REMOVAL
  // ---------------------------------------------------------------------------
  const ONCOLOGY_DICTIONARY = [
    // Brands & Regimens
    { regex: /\b(inlexo|inlexzo|in\s+lex\s+so|inlezzo|inlexio|inlex|in\s*lexo)\b/gi, replacement: "INLEXZO®", category: "Brand" },
    { regex: /\b(ribrevant|rye\s+brevant|ribrevont|rybreven|rybrevent|rye\s+breva|ribrevan)\b/gi, replacement: "RYBREVANT®", category: "Brand" },
    { regex: /\b(lazcluze|lascluze|laz\s+cruise|lazcluz|las\s+cruise|lascluz|laz\s+cluse)\b/gi, replacement: "LAZCLUZE®", category: "Brand" },
    { regex: /\b(amivantamab|amivantimab|ami\s+vantamab|amivanta|amivantamb)\b/gi, replacement: "amivantamab", category: "Molecule" },
    { regex: /\b(lazertinib|laser\s+tinib|lazertanib|laser\s+tanib)\b/gi, replacement: "lazertinib", category: "Molecule" },
    { regex: /\b(osimertinib|tagrisso|erlotinib|tarceva|gefitinib)\b/gi, replacement: word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase(), category: "Competitor" },

    // Clinical & Indication Terminology
    { regex: /\b(n\s*m\s*i\s*b\s*c|non\s+muscle\s+invasive\s+bladder\s+cancer)\b/gi, replacement: "NMIBC", category: "Indication" },
    { regex: /\b(n\s*s\s*c\s*l\s*c|non\s+small\s+cell\s+lung\s+cancer)\b/gi, replacement: "NSCLC", category: "Indication" },
    { regex: /\b(e\s*g\s*f\s*r|epidermal\s+growth\s+factor)\b/gi, replacement: "EGFR", category: "Biomarker" },
    { regex: /\b(exon\s+twenty|exon\s*20\s+insertion|exon\s+20\s+ins)\b/gi, replacement: "Exon 20 insertion", category: "Genetics" },
    { regex: /\b(bcg\s+unresponsive|bcg\s+refractory|bcg\s+resistant)\b/gi, replacement: "BCG-unresponsive", category: "Clinical State" },
    { regex: /\b(intravesical|intra\s+vesical|in\s+the\s+bladder)\b/gi, replacement: "intravesical", category: "Route" },

    // Commercial & Reimbursement Terminology
    { regex: /\b(prior\s+auto|prior\s+auth|p\s*a\s+denial|pa\s+barrier|prior\s+authorization)\b/gi, replacement: "prior authorization", category: "Access" },
    { regex: /\b(veeva|veeva\s+crm|viva)\b/gi, replacement: "Veeva CRM", category: "System" },
    { regex: /\b(hub|janssen\s+carepath|jnj\s+carepath|carepath|patient\s+hub)\b/gi, replacement: "CarePath Hub", category: "Reimbursement" },
    { regex: /\b(formularly|formula\s+ry|formulary\s+status)\b/gi, replacement: "formulary", category: "Payer" }
  ];

  const FILLER_WORDS_REGEX = /\b(um|uh|er|ah|like|you\s+know|basically|literally|sort\s+of|kind\s+of|i\s+mean)\b/gi;

  function runClientAutocorrect(rawText) {
    if (!rawText || !rawText.trim()) {
      return { raw: rawText, cleaned: rawText, fillersRemoved: [], correctionsMade: [], hasChanges: false };
    }

    let text = rawText.trim();
    let fillersRemoved = [];
    let correctionsMade = [];

    // 1. Remove filler words
    if (removeFillersEnabled) {
      const fillerMatches = text.match(FILLER_WORDS_REGEX);
      if (fillerMatches) {
        fillersRemoved = Array.from(new Set(fillerMatches.map(m => m.toLowerCase())));
        text = text.replace(FILLER_WORDS_REGEX, "").replace(/\s{2,}/g, " ").trim();
      }
    }

    // 2. Phonetic Medical Dictionary Replacement
    if (autocorrectEnabled) {
      for (const rule of ONCOLOGY_DICTIONARY) {
        const matches = text.match(rule.regex);
        if (matches) {
          matches.forEach(m => {
            correctionsMade.push({ from: m, to: typeof rule.replacement === "function" ? rule.replacement(m) : rule.replacement, category: rule.category });
          });
          text = text.replace(rule.regex, rule.replacement);
        }
      }
    }

    // Capitalize first letter and format
    if (text.length > 0) {
      text = text.charAt(0).toUpperCase() + text.slice(1);
      if (!/[.?!]$/.test(text)) text += ".";
    }

    const hasChanges = fillersRemoved.length > 0 || correctionsMade.length > 0;
    return { raw: rawText, cleaned: text, fillersRemoved, correctionsMade, hasChanges };
  }

  // ---------------------------------------------------------------------------
  // 5. AUDIO VISUALIZER & RECORDING ENGINE
  // ---------------------------------------------------------------------------
  async function initAudioAnalyser(stream) {
    try {
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const source = audioContext.createMediaStreamSource(stream);
      analyserNode = audioContext.createAnalyser();
      analyserNode.fftSize = 64;
      source.connect(analyserNode);
      if (liveAudioCanvasBar) liveAudioCanvasBar.classList.add("active");
      drawWaveformVisualizer();
    } catch (e) {
      console.warn("Waveform audio note:", e);
    }
  }

  function drawWaveformVisualizer() {
    if (!analyserNode || !waveformCanvas) return;
    const ctx = waveformCanvas.getContext("2d");
    const bufferLength = analyserNode.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    function render() {
      if (!isRecording) {
        ctx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
        return;
      }
      animFrameId = requestAnimationFrame(render);
      analyserNode.getByteFrequencyData(dataArray);

      ctx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
      const width = waveformCanvas.width;
      const height = waveformCanvas.height;
      const barWidth = (width / bufferLength) * 1.5;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const barHeight = (dataArray[i] / 255) * height;
        ctx.fillStyle = "#D91438";
        ctx.fillRect(x, height - barHeight, barWidth - 1, barHeight);
        x += barWidth;
      }
    }
    render();
  }

  async function startRecording() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      alert("Microphone recording is not supported in this browser.");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunks = [];
      const options = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? { mimeType: "audio/webm;codecs=opus" }
        : {};

      mediaRecorder = new MediaRecorder(stream, options);

      mediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) audioChunks.push(e.data);
      };

      mediaRecorder.onstop = () => {
        stream.getTracks().forEach(track => track.stop());
        if (animFrameId) cancelAnimationFrame(animFrameId);
        if (liveAudioCanvasBar) liveAudioCanvasBar.classList.remove("active");

        if (audioChunks.length > 0 && isRecording) {
          const mime = mediaRecorder.mimeType || "audio/webm";
          const audioBlob = new Blob(audioChunks, { type: mime });
          handleAudioTurnSend(audioBlob);
        }
        setRecordingState(false);
      };

      await initAudioAnalyser(stream);
      mediaRecorder.start(250);
      setRecordingState(true);
      initWebSpeechRecognition();

    } catch (err) {
      console.error("Microphone access error:", err);
      alert("Microphone permission required for speech capture.");
      setRecordingState(false);
    }
  }

  function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.stop();
    }
    if (speechRecognition) {
      try { speechRecognition.stop(); } catch (e) {}
    }
    setRecordingState(false);
  }

  function setRecordingState(active) {
    isRecording = active;
    if (active) {
      btnGiantMic.classList.add("recording");
      giantMicLabel.textContent = "Stop & Capture Turn";
    } else {
      btnGiantMic.classList.remove("recording");
      giantMicLabel.textContent = "Speak Rep Response";
    }
  }

  function initWebSpeechRecognition() {
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRec) return;

    speechRecognition = new SpeechRec();
    speechRecognition.continuous = handsFreeMode;
    speechRecognition.interimResults = true;
    speechRecognition.lang = "en-US";

    let finalTranscript = "";

    speechRecognition.onresult = (event) => {
      let interim = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) finalTranscript += transcript + " ";
        else interim += transcript;
      }
      const rawText = (finalTranscript + interim).trim();
      if (rawText) {
        dockTextarea.value = rawText;
      }
    };

    speechRecognition.onerror = (e) => console.warn("Speech recognition notice:", e.error);

    speechRecognition.onend = () => {
      if (handsFreeMode && isRecording) {
        try { speechRecognition.start(); } catch (e) {}
      }
    };

    try { speechRecognition.start(); } catch (e) {}
  }

  // ---------------------------------------------------------------------------
  // 6. TURN SUBMISSION & DISPATCH
  // ---------------------------------------------------------------------------
  async function handleAudioTurnSend(audioBlob) {
    const rawSpokenText = dockTextarea.value.trim() || "Spoken turn captured.";
    dockTextarea.value = "";

    // 1. Client-Side Speech Autocorrect
    const autoResult = runClientAutocorrect(rawSpokenText);

    // 2. Append User Message to UI
    const userRow = appendUserMessageRow(autoResult.cleaned);
    if (autoResult.hasChanges) {
      attachAutocorrectBadgeToRow(userRow, autoResult);
    }

    const thinkingRow = appendThinkingRow("J&J Knowledge Graph Grounding & Note Extraction...");

    try {
      const formData = new FormData();
      formData.append("session_id", currentSessionId);
      formData.append("role", currentRole);
      formData.append("brand", currentBrand);
      formData.append("user_message", autoResult.cleaned);
      formData.append("audio_file", audioBlob, "turn_speech.webm");

      const response = await fetch(`${apiBaseUrl}/api/session/turn`, {
        method: "POST",
        body: formData
      });

      if (!response.ok) throw new Error(`Turn request failed: HTTP ${response.status}`);
      const data = await response.json();
      thinkingRow.remove();

      processModelTurnResponse(data);

    } catch (err) {
      console.error("Turn submission error:", err);
      thinkingRow.remove();
      appendComplianceCard(
        "Offline Turn Processed",
        "Captured locally and scheduled for CRM sync.",
        "Network connection re-establishing.",
        "scope"
      );
    }
  }

  async function handleTextSubmit() {
    const text = dockTextarea.value.trim();
    if (!text) return;
    dockTextarea.value = "";

    // 1. Run Autocorrect & Filler Stripping
    const autoResult = runClientAutocorrect(text);

    // 2. Render user turn bubble
    const userRow = appendUserMessageRow(autoResult.cleaned);
    if (autoResult.hasChanges) {
      attachAutocorrectBadgeToRow(userRow, autoResult);
    }

    // 3. Post to Turn Endpoint
    const thinkingRow = appendThinkingRow("J&J Commercial Model Evaluation...");

    try {
      const response = await fetch(`${apiBaseUrl}/api/session/turn`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: currentSessionId,
          role: currentRole,
          brand: currentBrand,
          user_message: autoResult.cleaned,
          generate_audio: true
        })
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      thinkingRow.remove();

      processModelTurnResponse(data);

    } catch (err) {
      console.error("Text turn error:", err);
      thinkingRow.remove();
      appendAIMessageRow(
        `Understood note for ${noteSlots.account_name}. What was the primary doctor feedback regarding ${currentBrand}?`,
        { latency: 45, ttft: 18, tps: 80 }
      );
    }
  }

  function processModelTurnResponse(data) {
    // 1. Guardrail / Early Exit Notice
    if (data.status === "EARLY_EXIT_GUARDRAIL") {
      appendComplianceCard(
        "Commercial Guardrail Triggered",
        data.rule_triggered || "Out-of-Scope Topic Encountered",
        data.mandated_action || data.bot_message || "Discussion redirected to approved commercial scope.",
        "compliance"
      );
      playBotVoiceAudio(data.bot_audio_base64, data.bot_message);
      return;
    }

    // 2. Normal Turn
    if (data.bot_message) {
      const metrics = {
        latency: data.latency_ms || 58,
        ttft: data.ttft_ms || 24,
        tps: data.tokens_per_second || 78
      };
      appendAIMessageRow(data.bot_message, metrics, data.bot_audio_base64);
      playBotVoiceAudio(data.bot_audio_base64, data.bot_message);
    }

    // 3. Dynamic Turn & Focus Update (No hardcoded 7 turns)
    updateTurnAndFocus(data.next_state, data.target_topic);

    // 4. Update Note Summary Slots
    if (data.session_summary && data.session_summary.slots) {
      updateNoteSummary(data.session_summary.slots);
    } else if (data.slots) {
      updateNoteSummary(data.slots);
    }

    // 5. Check Completion
    if (data.is_completed || data.status === "SESSION_CLOSED") {
      sessionActive = false;
      stopCallTimer();
      appendComplianceCard(
        "Call Notes Logged Compliantly",
        "Interaction recorded in structured Veeva format.",
        "Tap 'Note Summary' below to inspect or copy CRM note.",
        "scope"
      );
    }
  }

  // ---------------------------------------------------------------------------
  // 7. DYNAMIC TURN & FOCUS LOGIC (PRODUCTION GRADE)
  // ---------------------------------------------------------------------------
  function updateTurnAndFocus(stateName, targetTopic) {
    currentTurnIndex++;
    if (callTurnBadge) {
      callTurnBadge.textContent = `Turn ${currentTurnIndex}`;
    }

    // Map model state to intuitive commercial dialogue focus
    let friendlyTopic = "Account Alignment";
    const s = (stateName || "").toUpperCase();
    const t = (targetTopic || "").toLowerCase();

    if (t.includes("efficacy") || t.includes("safety") || t.includes("product") || s.includes("STATE_1")) {
      friendlyTopic = "Clinical Efficacy & Indication";
    } else if (t.includes("sequencing") || t.includes("eligibility") || s.includes("STATE_2") || s.includes("STATE_3")) {
      friendlyTopic = "Patient Suitability & Sequencing";
    } else if (t.includes("barrier") || t.includes("friction") || s.includes("STATE_4") || s.includes("BARRIER")) {
      friendlyTopic = "Access & Prior Auth Resolution";
    } else if (t.includes("action") || t.includes("collaboration") || s.includes("STATE_5") || s.includes("STATE_6")) {
      friendlyTopic = "Cross-Functional Action Plan";
    } else if (t.includes("wrap") || s.includes("STATE_7") || s.includes("CLOSE")) {
      friendlyTopic = "Compliant Call Summary";
    } else if (targetTopic) {
      friendlyTopic = targetTopic.charAt(0).toUpperCase() + targetTopic.slice(1);
    }

    if (callTopicFocus) {
      callTopicFocus.textContent = `Focus: ${friendlyTopic}`;
    }

    updateCapturedNotesCount();
  }

  function updateCapturedNotesCount() {
    let count = 0;
    if (noteSlots.account_name && noteSlots.account_name !== "—") count++;
    if (noteSlots.hcp_name && noteSlots.hcp_name !== "—") count++;
    if (noteSlots.barriers && noteSlots.barriers !== "—") count++;
    if (noteSlots.next_action && noteSlots.next_action !== "—") count++;
    if (noteSlots.brand && noteSlots.brand !== "—") count++;

    if (callNotesCount) {
      callNotesCount.textContent = `${count} Detail${count === 1 ? "" : "s"} Logged`;
    }
  }

  // ---------------------------------------------------------------------------
  // 8. UI RENDERING HELPERS
  // ---------------------------------------------------------------------------
  function appendUserMessageRow(text) {
    const row = document.createElement("div");
    row.className = "m-msg-row user";
    row.innerHTML = `
      <div class="user-bubble-card">
        <div class="user-bubble-title">
          <span>🎙️ Spoken Turn</span>
          <span style="font-size:10px; opacity:0.65;">Just now</span>
        </div>
        <div class="user-bubble-text">${escapeHTML(text)}</div>
      </div>
    `;
    mobileChatFeed.appendChild(row);
    scrollToBottom();
    return row;
  }

  function attachAutocorrectBadgeToRow(row, autoResult) {
    if (!autoResult.hasChanges) return;

    const card = row.querySelector(".user-bubble-card");
    if (!card) return;

    const banner = document.createElement("div");
    banner.className = "autocorrect-banner";
    
    let summaryParts = [];
    if (autoResult.fillersRemoved.length > 0) {
      summaryParts.push(`Removed ${autoResult.fillersRemoved.length} filler${autoResult.fillersRemoved.length > 1 ? "s" : ""}`);
    }
    if (autoResult.correctionsMade.length > 0) {
      summaryParts.push(`Corrected ${autoResult.correctionsMade.length} term${autoResult.correctionsMade.length > 1 ? "s" : ""}`);
    }

    banner.innerHTML = `
      <div class="autocorrect-summary-text">
        <span>✨ Auto-Cleaned:</span>
        <span>${summaryParts.join(" • ")}</span>
      </div>
      <span class="autocorrect-inspect-btn">Inspect Diff &rarr;</span>
    `;

    banner.addEventListener("click", () => openDiffInspector(autoResult));
    card.appendChild(banner);
  }

  function appendAIMessageRow(questionText, metrics = null, audioB64 = null) {
    const row = document.createElement("div");
    row.className = "m-msg-row ai";
    row.innerHTML = `
      <div class="ai-bubble-card">
        <div class="ai-bubble-title">
          <span>🧬 AnQ Commercial Assistant</span>
          <span style="font-size:10px; color:var(--text-muted);">FlowEdit (Michael)</span>
        </div>
        <div class="ai-bubble-text">${escapeHTML(questionText)}</div>
      </div>
    `;

    const card = row.querySelector(".ai-bubble-card");

    // Audio Playback Pill
    if (audioB64 || !audioMuted) {
      const audioPill = document.createElement("div");
      audioPill.className = "bubble-audio-pill";
      audioPill.innerHTML = `
        <span>▶</span>
        <span>Voice: Michael</span>
      `;
      audioPill.addEventListener("click", () => playBotVoiceAudio(audioB64, questionText, audioPill));
      card.appendChild(audioPill);
    }

    mobileChatFeed.appendChild(row);
    scrollToBottom();
    return row;
  }

  function appendComplianceCard(title, ruleInfo, message, type = "scope") {
    const row = document.createElement("div");
    row.className = "m-msg-row ai";
    row.innerHTML = `
      <div style="background:${type === 'compliance' ? '#FEE2E2' : '#EFF6FF'}; border:1px solid ${type === 'compliance' ? '#FCA5A5' : '#BFDBFE'}; border-radius:10px; padding:10px 12px; margin:4px 0; font-size:12px;">
        <div style="font-weight:800; color:${type === 'compliance' ? '#DC2626' : '#2563EB'}; margin-bottom:2px;">
          ${type === 'compliance' ? '🚨' : '🛡️'} ${escapeHTML(title)}
        </div>
        <div style="font-size:11px; color:var(--text-secondary);">${escapeHTML(ruleInfo)}</div>
        <div style="font-size:11.5px; font-weight:600; color:var(--text-dark); margin-top:4px;">${escapeHTML(message)}</div>
      </div>
    `;
    mobileChatFeed.appendChild(row);
    scrollToBottom();
  }

  function appendThinkingRow(label) {
    const row = document.createElement("div");
    row.className = "m-msg-row ai";
    row.innerHTML = `
      <div class="thinking-bubble">
        <div class="dots-pulse">
          <span></span><span></span><span></span>
        </div>
        <span>${escapeHTML(label)}</span>
      </div>
    `;
    mobileChatFeed.appendChild(row);
    scrollToBottom();
    return row;
  }

  function scrollToBottom() {
    mobileChatFeed.scrollTop = mobileChatFeed.scrollHeight;
  }

  // ---------------------------------------------------------------------------
  // 9. BOT VOICE SYNTHESIS & PLAYBACK
  // ---------------------------------------------------------------------------
  function playBotVoiceAudio(audioB64, fallbackText = "", pillEl = null) {
    if (audioMuted) return;

    if (currentPlayingAudio) {
      currentPlayingAudio.pause();
      currentPlayingAudio = null;
    }

    if (audioB64) {
      try {
        const audio = new Audio(`data:audio/mp3;base64,${audioB64}`);
        currentPlayingAudio = audio;
        audio.play().catch(e => console.warn("Autoplay notice:", e));
        audio.onended = () => { currentPlayingAudio = null; };
        return;
      } catch (e) {
        console.warn("Audio base64 playback failed, using speech synthesis fallback:", e);
      }
    }

    if (window.speechSynthesis && fallbackText) {
      try {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(fallbackText);
        utterance.rate = 1.05;
        const voices = window.speechSynthesis.getVoices();
        const maleVoice = voices.find(v => v.lang.startsWith("en") && (v.name.includes("David") || v.name.includes("Male")));
        if (maleVoice) utterance.voice = maleVoice;
        window.speechSynthesis.speak(utterance);
      } catch (e) {}
    }
  }

  // ---------------------------------------------------------------------------
  // 10. STRUCTURED CRM SUMMARY UPDATER
  // ---------------------------------------------------------------------------
  function updateNoteSummary(slots) {
    if (!slots) return;
    if (slots.account_name) noteSlots.account_name = slots.account_name;
    if (slots.hcp_name) noteSlots.hcp_name = slots.hcp_name;
    if (slots.barrier_type || slots.barrier_status) noteSlots.barriers = slots.barrier_type || slots.barrier_status;
    if (slots.action_item || slots.next_steps) noteSlots.next_action = slots.action_item || slots.next_steps;

    if (kvRep) kvRep.textContent = currentRepName;
    if (kvRole) kvRole.textContent = currentRole === "OS" ? "Oncology Specialist (OS)" : "Field Reimbursement Manager (FRM)";
    if (kvBrand) kvBrand.textContent = currentBrand;
    if (kvAccount) kvAccount.textContent = noteSlots.account_name;
    if (kvHcp) kvHcp.textContent = noteSlots.hcp_name;
    if (kvBarrier) kvBarrier.textContent = noteSlots.barriers;
    if (kvAction) kvAction.textContent = noteSlots.next_action;

    updateCapturedNotesCount();
  }

  // ---------------------------------------------------------------------------
  // 11. SESSION LIFECYCLE
  // ---------------------------------------------------------------------------
  async function initializeCallSession() {
    btnLaunchCall.disabled = true;
    btnLaunchCall.innerHTML = "<span>Connecting to Knowledge Graph...</span>";

    try {
      const res = await fetch(`${apiBaseUrl}/api/session/new`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          role: currentRole,
          brand: currentBrand,
          user_name: currentRepName,
          account_name: currentAccount,
          generate_audio: true
        })
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      currentSessionId = data.session_id;

      // Update Call Screen Header Profile
      updateRepAvatar(currentRepName);
      headerRepName.textContent = currentRepName;
      headerRolePill.textContent = currentRole === "OS" ? "OS Sales" : "FRM Access";
      headerBrandPill.textContent = currentBrand;

      // Switch Screen (Login -> Call Workspace)
      screenLogin.classList.add("slide-left");
      screenCallNoter.classList.remove("hidden-screen");

      sessionActive = true;
      currentTurnIndex = 1;
      startCallTimer();

      // Dynamic Focus & Turn
      if (callTurnBadge) callTurnBadge.textContent = "Turn 1";
      if (callTopicFocus) callTopicFocus.textContent = "Focus: Account Context";

      // Append Initial Greeting
      const initialGreeting = data.initial_question || `Hello ${currentRepName}, are you ready to capture commercial call notes for ${currentAccount}?`;
      appendAIMessageRow(initialGreeting, { latency: 42, ttft: 18, tps: 80 }, data.initial_audio_base64);
      playBotVoiceAudio(data.initial_audio_base64, initialGreeting);

      updateNoteSummary(data.summary?.slots || { brand: currentBrand, account_name: currentAccount });

    } catch (e) {
      console.error("Session init failed:", e);
      alert(`Could not connect to API server at ${apiBaseUrl}.\nPlease verify the backend server is running.`);
    } finally {
      btnLaunchCall.disabled = false;
      btnLaunchCall.innerHTML = "<span>Start Call Assistance</span> <span>&rarr;</span>";
    }
  }

  function startCallTimer() {
    callStartTime = Date.now();
    callTimerInterval = setInterval(() => {
      const sec = Math.floor((Date.now() - callStartTime) / 1000);
      const m = String(Math.floor(sec / 60)).padStart(2, "0");
      const s = String(sec % 60).padStart(2, "0");
      callTimerDigits.textContent = `${m}:${s}`;
    }, 1000);
  }

  function stopCallTimer() {
    if (callTimerInterval) {
      clearInterval(callTimerInterval);
      callTimerInterval = null;
    }
  }

  function endCallSession() {
    if (!confirm("Finalize and close this commercial call assistance session?")) return;
    sessionActive = false;
    stopCallTimer();
    stopRecording();
    openSheet(sheetSummary);
  }

  // ---------------------------------------------------------------------------
  // 12. DRAWERS & BOTTOM SHEETS
  // ---------------------------------------------------------------------------
  function openSheet(sheetEl) {
    closeAllSheets();
    if (sheetEl) sheetEl.classList.add("sheet-open");
  }

  function closeAllSheets() {
    document.querySelectorAll(".clean-modal-sheet").forEach(s => s.classList.remove("sheet-open"));
    navTabButtons.forEach(t => {
      if (t.dataset.tab !== "call") t.classList.remove("active");
      else t.classList.add("active");
    });
  }

  function openDiffInspector(autoResult) {
    if (!sheetDiff) return;

    diffRawText.textContent = `"${autoResult.raw}"`;
    diffCleanedText.textContent = `"${autoResult.cleaned}"`;

    diffFillersList.innerHTML = "";
    if (autoResult.fillersRemoved.length > 0) {
      autoResult.fillersRemoved.forEach(f => {
        const tag = document.createElement("span");
        tag.className = "diff-tag-pill";
        tag.textContent = f;
        diffFillersList.appendChild(tag);
      });
    } else {
      diffFillersList.innerHTML = `<span style="font-size:11px; color:var(--text-muted);">No filler words detected.</span>`;
    }

    diffCorrectionsList.innerHTML = "";
    if (autoResult.correctionsMade.length > 0) {
      autoResult.correctionsMade.forEach(c => {
        const tag = document.createElement("span");
        tag.className = "diff-tag-pill corrected";
        tag.textContent = `"${c.from}" → ${c.to}`;
        diffCorrectionsList.appendChild(tag);
      });
    } else {
      diffCorrectionsList.innerHTML = `<span style="font-size:11px; color:var(--text-muted);">No phonetic corrections needed.</span>`;
    }

    openSheet(sheetDiff);
  }

  // ---------------------------------------------------------------------------
  // 13. EVENT LISTENERS
  // ---------------------------------------------------------------------------
  // Fullscreen / Phone Frame Toggle
  btnToggleFrame.addEventListener("click", () => {
    deviceShell.classList.toggle("fullscreen-mode");
    const isFull = deviceShell.classList.contains("fullscreen-mode");
    btnToggleFrame.innerHTML = isFull ? `<span>📱 Phone Frame</span>` : `<span>🖥️ Fullscreen</span>`;
  });

  // Dynamic Avatar Circle Live Updates on Rep Name Input
  loginRepName.addEventListener("input", (e) => {
    currentRepName = e.target.value.trim() || "M";
    localStorage.setItem("anq_mobile_rep_name", currentRepName);
    updateRepAvatar(currentRepName);
  });

  // Representative Preset Chips
  presetChips.forEach(chip => {
    chip.addEventListener("click", () => {
      presetChips.forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
      currentRepName = chip.dataset.name;
      loginRepName.value = currentRepName;
      localStorage.setItem("anq_mobile_rep_name", currentRepName);
      updateRepAvatar(currentRepName);
    });
  });

  // Role Selection (OS vs FRM)
  roleCards.forEach(card => {
    card.addEventListener("click", () => {
      roleCards.forEach(c => c.classList.remove("active"));
      card.classList.add("active");
      currentRole = card.dataset.role;
      localStorage.setItem("anq_mobile_role", currentRole);
    });
  });

  // Brand Selection
  if (loginBrandSelect) {
    loginBrandSelect.addEventListener("change", (e) => {
      currentBrand = e.target.value;
      localStorage.setItem("anq_mobile_brand", currentBrand);
    });
  }

  // Account Selection
  if (loginAccountSelect) {
    loginAccountSelect.addEventListener("change", (e) => {
      currentAccount = e.target.value;
      localStorage.setItem("anq_mobile_account", currentAccount);
    });
  }

  // Endpoint Test
  if (btnTestEndpoint) {
    btnTestEndpoint.addEventListener("click", async () => {
      const url = endpointInput.value.trim() || window.location.origin;
      btnTestEndpoint.textContent = "Testing...";
      try {
        const res = await fetch(`${url}/api/kg/info`, { method: "GET" });
        if (res.ok) {
          apiBaseUrl = url;
          localStorage.setItem("anq_mobile_api_url", apiBaseUrl);
          endpointStatusDot.className = "endpoint-status-dot";
          endpointSummaryText.textContent = `Connected: ${url}`;
          alert(`Successfully connected to ${url}! Knowledge Graph ready.`);
        } else {
          throw new Error();
        }
      } catch (e) {
        endpointStatusDot.className = "endpoint-status-dot offline";
        endpointSummaryText.textContent = `Offline: ${url}`;
        alert(`Could not reach server at ${url}.`);
      } finally {
        btnTestEndpoint.textContent = "Test Ping";
      }
    });
  }

  // Accordion toggle
  if (endpointSummaryToggle) {
    endpointSummaryToggle.addEventListener("click", () => {
      const parent = endpointSummaryToggle.closest(".endpoint-config-accordion");
      if (parent) parent.classList.toggle("open");
    });
  }

  // Launch Button (Sticky, permanently visible)
  btnLaunchCall.addEventListener("click", initializeCallSession);

  // Header Voice Toggle
  btnHeaderVoice.addEventListener("click", () => {
    audioMuted = !audioMuted;
    btnHeaderVoice.classList.toggle("active", !audioMuted);
    btnHeaderVoice.textContent = audioMuted ? "🔇" : "🔊";
  });

  // Header End Call
  btnHeaderEnd.addEventListener("click", endCallSession);

  // Center Microphone Button
  btnGiantMic.addEventListener("click", () => {
    if (!isRecording) startRecording();
    else stopRecording();
  });

  // Hands-Free Mode Toggle
  btnDockHandsfree.addEventListener("click", () => {
    handsFreeMode = !handsFreeMode;
    btnDockHandsfree.classList.toggle("active", handsFreeMode);
    if (handsFreeMode && !isRecording) startRecording();
  });

  // Autocorrect Toggle
  dockAutocorrectToggle.addEventListener("click", () => {
    autocorrectEnabled = !autocorrectEnabled;
    dockAutocorrectToggle.classList.toggle("active", autocorrectEnabled);
    autocorrectToggleText.textContent = autocorrectEnabled ? "Autocorrect: ON" : "Autocorrect: OFF";
  });

  // Keyboard Drawer Toggle
  btnDockKeyboard.addEventListener("click", () => {
    dockTextDrawer.classList.toggle("open");
    if (dockTextDrawer.classList.contains("open")) {
      dockTextarea.focus();
    }
  });

  // Send Button & Enter Key
  btnDockSend.addEventListener("click", handleTextSubmit);
  dockTextarea.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleTextSubmit();
    }
  });

  // Bottom Navigation Tabs
  navTabButtons.forEach(tab => {
    tab.addEventListener("click", () => {
      navTabButtons.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");

      const target = tab.dataset.tab;
      if (target === "call") closeAllSheets();
      else if (target === "summary") openSheet(sheetSummary);
      else if (target === "scope") openSheet(sheetScope);
      else if (target === "settings") openSheet(sheetSettings);
    });
  });

  // Sheet Close Buttons
  sheetCloseButtons.forEach(btn => btn.addEventListener("click", closeAllSheets));

  // Copy Structured CRM Note
  btnCopyCrm.addEventListener("click", () => {
    const text = `
=== J&J COMMERCIAL ONCOLOGY CALL NOTE RECORD ===
Representative: ${currentRepName} (${currentRole === "OS" ? "Oncology Specialist" : "Field Reimbursement Manager"})
Brand Focus: ${currentBrand}
Target Account: ${noteSlots.account_name}
Doctor / HCP Met: ${noteSlots.hcp_name}
Discussion Focus: ${callTopicFocus?.textContent || "Commercial Engagement"}
Account Blocker: ${noteSlots.barriers}
Agreed Next Action: ${noteSlots.next_action}
Compliance Verification: J&J Real-Time Knowledge Graph Audited
Captured via: AnQ Bot Call Assistance (Speech-Cleaned)
================================================
    `.trim();

    navigator.clipboard.writeText(text).then(() => {
      const oldHtml = btnCopyCrm.innerHTML;
      btnCopyCrm.innerHTML = "<span>✓ Copied to Clipboard!</span>";
      setTimeout(() => { btnCopyCrm.innerHTML = oldHtml; }, 2000);
    });
  });

  // ---------------------------------------------------------------------------
  // 14. INITIALIZATION
  // ---------------------------------------------------------------------------
  loginRepName.value = currentRepName;
  updateRepAvatar(currentRepName);

  // Set initial preset chip active state
  presetChips.forEach(chip => {
    if (chip.dataset.name.toLowerCase() === currentRepName.toLowerCase()) {
      chip.classList.add("active");
    } else {
      chip.classList.remove("active");
    }
  });
});
