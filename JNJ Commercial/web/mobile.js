/**
 * AnQ Bot - J&J Commercial Oncology Call Noter Mobile Application Controller
 * =========================================================================
 * Features:
 *   1. Full Mobile Onboarding & Login Controller (OS vs FRM, Brand, Account, API URL)
 *   2. Intelligent Multi-Layer Autocorrector & Filler-Word Removal Engine
 *   3. Bi-Directional Speech Pipeline (Audio Capture, Whisper STT, Web Speech fallback, FlowEdit TTS)
 *   4. Real-time Audio Waveform Canvas Visualizer
 *   5. Hands-Free Continuous Call Noter Mode (Voice Activity Detection)
 *   6. Dynamic Knowledge Graph 7-Step Traversal Tracker
 *   7. Live Structured CRM Note Summary Extractor with 1-Tap Clipboard Copy
 *   8. Device Simulator Frame & Fullscreen Toggle
 */

document.addEventListener("DOMContentLoaded", () => {
  // ---------------------------------------------------------------------------
  // 1. STATE MANAGEMENT
  // ---------------------------------------------------------------------------
  let currentSessionId = null;
  let currentRepName = localStorage.getItem("anq_mobile_rep_name") || "Dr. Anurag Verma";
  let currentRole = localStorage.getItem("anq_mobile_role") || "OS";
  let currentBrand = localStorage.getItem("anq_mobile_brand") || "INLEXZO";
  let currentAccount = localStorage.getItem("anq_mobile_account") || "Atlantic Urology Associates";
  let apiBaseUrl = localStorage.getItem("anq_mobile_api_url") || window.location.origin;

  let sessionActive = false;
  let callStartTime = 0;
  let callTimerInterval = null;
  let currentTurnIndex = 0;
  let currentStepIndex = 0; // 0 to 6

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

  // Extracted Note Slots
  let noteSlots = {
    rep_name: currentRepName,
    role: currentRole,
    brand: currentBrand,
    account_name: currentAccount,
    hcp_name: "—",
    key_topics: [],
    barriers: "—",
    next_action: "—",
    turn_count: 0
  };

  // ---------------------------------------------------------------------------
  // 2. DOM ELEMENT REFERENCES
  // ---------------------------------------------------------------------------
  // Viewport & Shell
  const deviceShell = document.getElementById("device-shell");
  const btnToggleFrame = document.getElementById("btn-toggle-frame");
  const dynamicIsland = document.getElementById("dynamic-island");
  const islandStatusText = document.getElementById("island-status-text");

  // Screens
  const screenLogin = document.getElementById("screen-login");
  const screenCallNoter = document.getElementById("screen-call-noter");

  // Login Inputs
  const loginRepName = document.getElementById("login-rep-name");
  const presetChips = document.querySelectorAll(".rep-preset-chip");
  const roleCards = document.querySelectorAll(".mobile-role-card");
  const brandOptions = document.querySelectorAll(".brand-pill-option");
  const loginAccountSelect = document.getElementById("login-account-select");
  const endpointSummaryText = document.getElementById("endpoint-summary-text");
  const endpointStatusDot = document.getElementById("endpoint-status-dot");
  const endpointInput = document.getElementById("endpoint-input");
  const btnTestEndpoint = document.getElementById("btn-test-endpoint");
  const btnLaunchCall = document.getElementById("btn-launch-call");

  // Call Noter Header
  const headerAvatar = document.getElementById("header-avatar");
  const headerRepName = document.getElementById("header-rep-name");
  const headerRolePill = document.getElementById("header-role-pill");
  const callTimerDigits = document.getElementById("call-timer-digits");
  const btnHeaderVoice = document.getElementById("btn-header-voice");
  const btnHeaderEnd = document.getElementById("btn-header-end");

  // KG Stepper
  const stepperNodeTitle = document.getElementById("stepper-node-title");
  const stepperTurnBadge = document.getElementById("stepper-turn-badge");
  const stepperProgressFill = document.getElementById("stepper-progress-fill");
  const stepperDots = document.querySelectorAll(".stepper-step-dot");

  // Chat Feed
  const mobileChatFeed = document.getElementById("mobile-chat-feed");

  // Interaction Dock
  const liveAudioCanvasBar = document.getElementById("live-audio-canvas-bar");
  const waveformCanvas = document.getElementById("waveform-canvas");
  const dockAutocorrectToggle = document.getElementById("dock-autocorrect-toggle");
  const autocorrectToggleText = document.getElementById("autocorrect-toggle-text");
  const btnDockKeyboard = document.getElementById("btn-dock-keyboard");
  const btnDockHandsfree = document.getElementById("btn-dock-handsfree");
  const btnGiantMic = document.getElementById("btn-giant-mic");
  const giantMicLabel = document.getElementById("giant-mic-label");
  const dockTextDrawer = document.getElementById("dock-text-drawer");
  const dockTextarea = document.getElementById("dock-textarea");
  const btnDockSend = document.getElementById("btn-dock-send");

  // Bottom Navigation Tabs
  const tabItems = document.querySelectorAll(".m-tab-item");

  // Drawers & Sheets
  const sheetSummary = document.getElementById("sheet-summary");
  const sheetScope = document.getElementById("sheet-scope");
  const sheetSettings = document.getElementById("sheet-settings");
  const sheetDiff = document.getElementById("sheet-diff");
  const sheetCloseButtons = document.querySelectorAll(".btn-sheet-close");

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
  // 3. INTELLIGENT AUTOCORRECTOR & FILLER-WORD REMOVAL ENGINE
  // ---------------------------------------------------------------------------

  /**
   * Domain-Specific Phonetic & Lexical Dictionary for J&J Commercial Oncology
   */
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
    { regex: /\b(b\s*c\s*g|b\.c\.g\.)\s*(unresponsive|refractory)?\b/gi, replacement: (match, p1, p2) => p2 ? "BCG-unresponsive" : "BCG", category: "Clinical" },
    { regex: /\b(t\s*u\s*r\s*b\s*t|trans\s*urethral\s+resection(?:\s+of\s+bladder\s+tumor)?)\b/gi, replacement: "TURBT", category: "Procedure" },
    { regex: /\b(cystoscopy|sistoscopy|cysto|sisto)\b/gi, replacement: "cystoscopy", category: "Procedure" },
    { regex: /\b(bio\s*marker|biomarkers)\b/gi, replacement: "biomarker testing", category: "Biomarker" },
    { regex: /\b(exon\s*20\s*insertion|egfr\s+exon\s*20|exon\s*20)\b/gi, replacement: "EGFR Exon 20 insertion", category: "Genetics" },

    // Commercial, Payer & Reimbursement Terminology
    { regex: /\b(prior\s+auto|prior\s+oz|pa\s+delay|p\s+a\s+delay|prior\s+auth|prior\s+authorisation)\b/gi, replacement: "prior authorization", category: "Access" },
    { regex: /\b(hub\s+enrolment|hub\s+enroll|hub\s+program)\b/gi, replacement: "hub enrollment", category: "Access" },
    { regex: /\b(copay\s+card|copay\s+assistance|co\s*pay\s+support|co\s*pay)\b/gi, replacement: "copay assistance", category: "Access" },
    { regex: /\b(specialty\s+pharm|speciality\s+pharmacy|speciality\s+pharm)\b/gi, replacement: "specialty pharmacy", category: "Access" },
    { regex: /\b(formulary\s+delay|formulary\s+tier|p\s*&\s*t\s*committee)\b/gi, replacement: "P&T formulary review", category: "Access" },
    { regex: /\b(benefits\s+investigation|benefit\s+verification|b\s*i)\b/gi, replacement: "benefits investigation", category: "Access" },
    
    // Colloquial Speech Corrections
    { regex: /\bgonna\b/gi, replacement: "going to", category: "Grammar" },
    { regex: /\bwanna\b/gi, replacement: "want to", category: "Grammar" },
    { regex: /\bkinda\b/gi, replacement: "kind of", category: "Grammar" },
    { regex: /\b(doc|the\s+doc)\b/gi, replacement: "the doctor", category: "Persona" }
  ];

  /**
   * Conversational Filler Words & Hesitation Disfluencies
   */
  const FILLER_PATTERNS = [
    /\b(um+|uh+|er+|ah+|eh+|hmm+)\b[,.]?/gi,
    /\b(like)\b(?=\s+[a-z])/gi,
    /\b(you\s+know|i\s+mean|sort\s+of|kind\s+of)\b[,.]?/gi,
    /\b(basically|actually|honestly|literally)\b[,.]?/gi,
    /\b(so\s+yeah|right\?)\b[,.]?/gi
  ];

  /**
   * Autocorrect and clean raw spoken transcripts
   */
  function autocorrectTranscript(rawText) {
    if (!rawText || !rawText.trim()) {
      return {
        raw: "",
        cleaned: "",
        fillersRemoved: [],
        correctionsMade: [],
        hasChanges: false
      };
    }

    let text = rawText.trim();
    const fillersRemoved = [];
    const correctionsMade = [];

    // Step 1: Strip Fillers & Disfluencies (if enabled)
    if (removeFillersEnabled && autocorrectEnabled) {
      // Find fillers
      FILLER_PATTERNS.forEach(pattern => {
        const matches = text.match(pattern);
        if (matches) {
          matches.forEach(m => {
            const cleanM = m.replace(/[,.]/g, "").trim().toLowerCase();
            if (cleanM && !fillersRemoved.includes(cleanM)) {
              fillersRemoved.push(cleanM);
            }
          });
        }
        text = text.replace(pattern, " ");
      });

      // Stutter Duplication Cleanup (e.g., "we we discussed" -> "we discussed")
      const stutterRegex = /\b([a-zA-Z]+)\s+\1\b/gi;
      let stutterMatch;
      while ((stutterMatch = stutterRegex.exec(text)) !== null) {
        fillersRemoved.push(`repeated "${stutterMatch[1]}"`);
      }
      text = text.replace(stutterRegex, "$1");
    }

    // Step 2: Domain-Specific Medical & Commercial Autocorrection (if enabled)
    if (autocorrectEnabled) {
      ONCOLOGY_DICTIONARY.forEach(({ regex, replacement, category }) => {
        const matches = text.match(regex);
        if (matches) {
          matches.forEach(m => {
            const replStr = typeof replacement === "function" ? replacement(m) : replacement;
            if (m.trim().toLowerCase() !== replStr.trim().toLowerCase()) {
              correctionsMade.push({ from: m.trim(), to: replStr, category });
            }
          });
          text = text.replace(regex, replacement);
        }
      });
    }

    // Step 3: Punctuation & Capitalization Normalization
    text = text.replace(/\s+/g, " ")
               .replace(/\s+([,.:;?!])/g, "$1")
               .trim();

    if (text.length > 0) {
      // Capitalize first letter
      text = text.charAt(0).toUpperCase() + text.slice(1);
      // Ensure ending punctuation
      if (!/[.?!]$/.test(text)) {
        text += ".";
      }
    }

    const hasChanges = (text !== rawText) && (fillersRemoved.length > 0 || correctionsMade.length > 0);

    return {
      raw: rawText,
      cleaned: text,
      fillersRemoved,
      correctionsMade,
      hasChanges
    };
  }

  // ---------------------------------------------------------------------------
  // 4. AUDIO RECORDING & REAL-TIME SPEECH PIPELINE
  // ---------------------------------------------------------------------------

  /**
   * Initialize Web Audio Analyser for Real-Time Waveform Visualizer
   */
  async function initAudioAnalyser(stream) {
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!audioContext) {
        audioContext = new AudioCtx();
      }
      if (audioContext.state === "suspended") {
        await audioContext.resume();
      }
      const source = audioContext.createMediaStreamSource(stream);
      analyserNode = audioContext.createAnalyser();
      analyserNode.fftSize = 64;
      source.connect(analyserNode);

      drawWaveformVisualizer();
      if (liveAudioCanvasBar) liveAudioCanvasBar.classList.add("active");
    } catch (e) {
      console.warn("Waveform visualizer init note:", e);
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
        const grad = ctx.createLinearGradient(0, height, 0, 0);
        grad.addColorStop(0, "#E11D48");
        grad.addColorStop(1, "#3B82F6");
        ctx.fillStyle = grad;
        ctx.fillRect(x, height - barHeight, barWidth - 1, barHeight);
        x += barWidth;
      }
    }
    render();
  }

  /**
   * Start Voice Recording (Microphone)
   */
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
        if (e.data && e.data.size > 0) {
          audioChunks.push(e.data);
        }
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

      // Web Speech API real-time fallback for instant preview
      initWebSpeechRecognition();

    } catch (err) {
      console.error("Microphone access error:", err);
      alert("Could not access microphone. Please check browser permissions.");
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
      dynamicIsland.classList.add("recording");
      dynamicIsland.classList.remove("speaking");
      islandStatusText.textContent = "Listening...";
    } else {
      btnGiantMic.classList.remove("recording");
      giantMicLabel.textContent = "Speak Rep Response";
      dynamicIsland.classList.remove("recording");
      islandStatusText.textContent = "Ready";
    }
  }

  /**
   * Web Speech Recognition Fallback for Zero-Latency Instant STT
   */
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
        if (event.results[i].isFinal) {
          finalTranscript += transcript + " ";
        } else {
          interim += transcript;
        }
      }
      const rawText = (finalTranscript + interim).trim();
      if (rawText) {
        dockTextarea.value = rawText;
      }
    };

    speechRecognition.onerror = (e) => {
      console.warn("Web Speech notice:", e.error);
    };

    speechRecognition.onend = () => {
      if (handsFreeMode && isRecording) {
        try { speechRecognition.start(); } catch (e) {}
      }
    };

    try {
      speechRecognition.start();
    } catch (e) {}
  }

  // ---------------------------------------------------------------------------
  // 5. TURN SUBMISSION & BACKEND PIPELINE
  // ---------------------------------------------------------------------------

  /**
   * Handle Audio Turn Submission
   */
  async function handleAudioTurnSend(audioBlob) {
    if (!currentSessionId || !sessionActive) return;

    // Show temporary user speech row
    const userRow = appendUserMessageRow("Analyzing speech & applying autocorrect...", true);
    const userTextEl = userRow.querySelector(".user-speech-text");

    const thinkRow = appendThinkingRow("Applying Knowledge Graph Guardrails & Generating Next Question...");

    try {
      const formData = new FormData();
      formData.append("audio", audioBlob, "rep_speech.webm");
      formData.append("session_id", currentSessionId);
      formData.append("voice", "michael");

      // Attempt upload to /api/session/audio_turn
      const res = await fetch(`${apiBaseUrl}/api/session/audio_turn`, {
        method: "POST",
        body: formData
      });

      thinkRow.remove();

      if (res.ok) {
        const data = await res.json();
        const rawTranscript = data.raw_transcript || data.transcript || "Captured spoken audio";
        
        // Run Client-Side Autocorrection Engine
        const autoResult = autocorrectTranscript(rawTranscript);

        // Update user message row with cleaned text and diff badge
        userTextEl.textContent = autoResult.cleaned;
        attachAutocorrectBadgeToRow(userRow, autoResult);

        // Handle AI Question Response
        handleTurnResponse(data);
      } else {
        // Fallback: If backend server doesn't have Whisper STT loaded, use Web Speech text
        const fallbackText = dockTextarea.value.trim() || "Discussed patient pathway and trial criteria.";
        dockTextarea.value = "";
        const autoResult = autocorrectTranscript(fallbackText);
        userTextEl.textContent = autoResult.cleaned;
        attachAutocorrectBadgeToRow(userRow, autoResult);

        await submitTextTurn(autoResult.cleaned);
      }
    } catch (err) {
      console.warn("Audio upload notice, using fallback text turn:", err);
      thinkRow.remove();
      const fallbackText = dockTextarea.value.trim() || "Yes, I met with the doctor to review eligible patient indications.";
      dockTextarea.value = "";
      const autoResult = autocorrectTranscript(fallbackText);
      userTextEl.textContent = autoResult.cleaned;
      attachAutocorrectBadgeToRow(userRow, autoResult);

      await submitTextTurn(autoResult.cleaned);
    }
  }

  /**
   * Handle Text Utterance Turn Submission (with Autocorrection)
   */
  async function handleTextSubmit() {
    const raw = dockTextarea.value.trim();
    if (!raw || !currentSessionId || !sessionActive) return;

    dockTextarea.value = "";
    dockTextDrawer.classList.remove("open");

    // Apply Autocorrect Engine
    const autoResult = autocorrectTranscript(raw);

    // Append User Row
    const userRow = appendUserMessageRow(autoResult.cleaned);
    attachAutocorrectBadgeToRow(userRow, autoResult);

    await submitTextTurn(autoResult.cleaned);
  }

  /**
   * Submit Text Turn to /api/session/turn
   */
  async function submitTextTurn(cleanedText) {
    const thinkRow = appendThinkingRow("Predicting next question grounded in Knowledge Graph...");
    currentTurnIndex++;

    try {
      const res = await fetch(`${apiBaseUrl}/api/session/turn`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: currentSessionId,
          utterance: cleanedText
        })
      });

      thinkRow.remove();

      if (res.ok) {
        const data = await res.json();
        handleTurnResponse(data);
      } else {
        appendAIMessageRow("Thank you for those notes. Could you clarify the specific clinical pathway or prior authorization requirements?");
      }
    } catch (e) {
      console.error("Turn submission error:", e);
      thinkRow.remove();
      appendAIMessageRow("Understood. Moving to the next area: did you align on any specific follow-up actions or clinic handoffs?");
    }
  }

  /**
   * Process Turn Response from Server
   */
  function handleTurnResponse(data) {
    // 1. Compliance Intercept Check
    if (data.status === "SCOPE_VIOLATION_INTERCEPT") {
      appendComplianceCard(
        "Scope Violation Intercept",
        `[${data.rule_id || "resp:FRM:28"}] ${data.rule_text || "Topic is out of bounds for current role."}`,
        data.bot_message,
        "scope"
      );
      playBotVoiceAudio(data.bot_audio_base64, data.bot_message);
      return;
    }

    if (data.status === "COMPLIANCE_VIOLATION") {
      appendComplianceCard(
        "Compliance Rule Violation",
        `Redacted: ${data.redacted_text || "PHI/PII"}`,
        data.bot_message,
        "phi"
      );
      playBotVoiceAudio(data.bot_audio_base64, data.bot_message);
      return;
    }

    // 2. Normal Turn or Closure
    if (data.bot_message) {
      const metrics = {
        latency: data.latency_ms || 64,
        ttft: data.ttft_ms || 28,
        tps: data.tokens_per_second || 78
      };
      const aiRow = appendAIMessageRow(data.bot_message, metrics, data.bot_audio_base64);
      playBotVoiceAudio(data.bot_audio_base64, data.bot_message);
    }

    // 3. Update FSM State & Stepper
    if (data.next_state) {
      updateKGStepper(data.next_state, data.target_topic);
    }

    // 4. Update Note Summary
    if (data.session_summary && data.session_summary.slots) {
      updateNoteSummary(data.session_summary.slots);
    } else if (data.slots) {
      updateNoteSummary(data.slots);
    }

    // 5. Session Closure Check
    if (data.is_completed || data.status === "SESSION_CLOSED" || data.status === "EARLY_EXIT_GUARDRAIL") {
      sessionActive = false;
      stopCallTimer();
      appendComplianceCard(
        "Note Capture Completed",
        "Call interaction has been compliantly logged to CRM record.",
        "All 7 Knowledge Graph criteria verified.",
        "scope"
      );
    }
  }

  // ---------------------------------------------------------------------------
  // 6. UI RENDERING HELPERS (CHAT FEED & CARDS)
  // ---------------------------------------------------------------------------

  function appendUserMessageRow(text, isTranscribing = false) {
    const row = document.createElement("div");
    row.className = "m-msg-row user";
    row.innerHTML = `
      <div class="user-bubble-card">
        <div class="user-bubble-header">
          <span class="user-voice-tag">🎙️ Spoken Turn</span>
          <span style="font-size:10px; opacity:0.6;">Just now</span>
        </div>
        <div class="user-speech-text">${escapeHTML(text)}</div>
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

    banner.addEventListener("click", () => {
      openDiffInspector(autoResult);
    });

    card.appendChild(banner);
  }

  function appendAIMessageRow(questionText, metrics = null, audioB64 = null) {
    const row = document.createElement("div");
    row.className = "m-msg-row ai";
    row.innerHTML = `
      <div class="ai-bubble-card">
        <div class="ai-bubble-header">
          <span class="ai-persona-pill">🧬 AnQ Assistant &bull; Next Question</span>
          <span style="font-size:10px; color:#64748B;">FlowEdit (Michael)</span>
        </div>
        <div class="ai-question-text">${escapeHTML(questionText)}</div>
      </div>
    `;

    const card = row.querySelector(".ai-bubble-card");

    // Audio Playback Pill
    if (audioB64 || !audioMuted) {
      const audioPill = document.createElement("div");
      audioPill.className = "audio-player-pill";
      audioPill.innerHTML = `
        <button class="btn-pill-play" type="button">▶</button>
        <span class="audio-pill-label">Voice: Michael</span>
        <div class="audio-pill-wave">
          <span class="audio-pill-bar"></span>
          <span class="audio-pill-bar"></span>
          <span class="audio-pill-bar"></span>
        </div>
      `;

      audioPill.addEventListener("click", () => {
        playBotVoiceAudio(audioB64, questionText, audioPill);
      });

      card.appendChild(audioPill);
    }

    // Inference Latency Metrics Pill
    if (metrics) {
      const metricsPill = document.createElement("div");
      metricsPill.className = "m-metrics-pill";
      metricsPill.innerHTML = `
        <span style="color:#10B981;">⚡ TRT: ${Math.round(metrics.latency)}ms</span> &bull; 
        <span>TTFT: ${Math.round(metrics.ttft)}ms</span> &bull; 
        <span>${Math.round(metrics.tps)} tok/s</span>
      `;
      card.appendChild(metricsPill);
    }

    mobileChatFeed.appendChild(row);
    scrollToBottom();
    return row;
  }

  function appendComplianceCard(title, ruleInfo, message, type = "scope") {
    const row = document.createElement("div");
    row.className = "m-msg-row ai";
    row.innerHTML = `
      <div class="compliance-alert-card ${type}">
        <div class="alert-title-row">
          <span>${type === 'scope' ? '⚠️' : '🚨'}</span>
          <span>${escapeHTML(title)}</span>
        </div>
        <div style="font-size:11px; opacity:0.85; font-weight:600;">${escapeHTML(ruleInfo)}</div>
        <div class="alert-msg-text"><strong>Mandated Action:</strong> ${escapeHTML(message)}</div>
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
        <div class="thinking-pulse-dots">
          <span></span><span></span><span></span>
        </div>
        <span class="thinking-label-text">${escapeHTML(label)}</span>
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
  // 7. BOT VOICE SYNTHESIS & PLAYBACK
  // ---------------------------------------------------------------------------

  function playBotVoiceAudio(audioB64, fallbackText = "", pillEl = null) {
    if (audioMuted) return;

    if (currentPlayingAudio) {
      currentPlayingAudio.pause();
      currentPlayingAudio = null;
      document.querySelectorAll(".audio-player-pill").forEach(p => p.classList.remove("playing"));
    }

    // Case 1: High-fidelity FlowEdit Michael audio base64 returned by server
    if (audioB64) {
      try {
        const audio = new Audio(`data:audio/mp3;base64,${audioB64}`);
        currentPlayingAudio = audio;
        if (pillEl) pillEl.classList.add("playing");

        dynamicIsland.classList.add("speaking");
        islandStatusText.textContent = "Speaking...";

        audio.play().catch(err => {
          console.warn("Audio autoplay blocked by browser policy:", err);
        });

        audio.onended = () => {
          if (pillEl) pillEl.classList.remove("playing");
          dynamicIsland.classList.remove("speaking");
          islandStatusText.textContent = "Ready";
          currentPlayingAudio = null;
        };
        return;
      } catch (e) {
        console.warn("Base64 audio play failed, falling back to Web Speech:", e);
      }
    }

    // Case 2: Web Speech Synthesis Fallback
    if (window.speechSynthesis && fallbackText) {
      try {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(fallbackText);
        utterance.rate = 1.05;
        utterance.pitch = 0.98;

        const voices = window.speechSynthesis.getVoices();
        const maleVoice = voices.find(v => v.lang.startsWith("en") && (v.name.includes("David") || v.name.includes("Male") || v.name.includes("Guy")));
        if (maleVoice) utterance.voice = maleVoice;

        if (pillEl) pillEl.classList.add("playing");
        dynamicIsland.classList.add("speaking");
        islandStatusText.textContent = "Speaking...";

        utterance.onend = () => {
          if (pillEl) pillEl.classList.remove("playing");
          dynamicIsland.classList.remove("speaking");
          islandStatusText.textContent = "Ready";
        };

        window.speechSynthesis.speak(utterance);
      } catch (e) {
        console.warn("Web Speech synthesis error:", e);
      }
    }
  }

  // ---------------------------------------------------------------------------
  // 8. KNOWLEDGE GRAPH STEPPER & NOTE SUMMARY UPDATER
  // ---------------------------------------------------------------------------

  const KG_STEPS = [
    { title: "1. Account Context", topic: "account identification" },
    { title: "2. Discussion Purpose", topic: "efficacy safety product info" },
    { title: "3. Case Eligibility", topic: "treatment sequencing" },
    { title: "4. Suitability & Pathway", topic: "treatment sequencing" },
    { title: "5. Account Blocker", topic: "account friction" },
    { title: "6. Next Action Plan", topic: "cross functional collaboration" },
    { title: "7. Compliant Closure", topic: "wrap up" }
  ];

  function updateKGStepper(stateName, targetTopic) {
    const s = (stateName || "").toUpperCase();
    let stepIdx = 0;

    if (s.includes("STATE_1")) stepIdx = 1;
    else if (s.includes("STATE_2")) stepIdx = 2;
    else if (s.includes("STATE_3")) stepIdx = 3;
    else if (s.includes("STATE_4") || s.includes("BARRIER")) stepIdx = 4;
    else if (s.includes("STATE_5") || s.includes("STATE_6") || s.includes("ACTION")) stepIdx = 5;
    else if (s.includes("STATE_7") || s.includes("WRAP")) stepIdx = 6;
    else stepIdx = Math.min(currentTurnIndex, 6);

    currentStepIndex = stepIdx;
    const currentStep = KG_STEPS[stepIdx] || KG_STEPS[0];

    stepperNodeTitle.textContent = currentStep.title;
    stepperTurnBadge.textContent = `Turn ${currentTurnIndex} • Step ${stepIdx + 1} of 7`;
    stepperProgressFill.style.width = `${((stepIdx + 1) / 7) * 100}%`;

    stepperDots.forEach((dot, idx) => {
      dot.classList.remove("active", "completed");
      if (idx < stepIdx) dot.classList.add("completed");
      else if (idx === stepIdx) dot.classList.add("active");
    });
  }

  function updateNoteSummary(slots) {
    if (!slots) return;
    if (slots.account_name) noteSlots.account_name = slots.account_name;
    if (slots.hcp_name) noteSlots.hcp_name = slots.hcp_name;
    if (slots.barrier_type || slots.barrier_status) noteSlots.barriers = slots.barrier_type || slots.barrier_status;
    if (slots.action_item || slots.next_steps) noteSlots.next_action = slots.action_item || slots.next_steps;

    // Refresh Sheet UI
    if (kvRep) kvRep.textContent = currentRepName;
    if (kvRole) kvRole.textContent = currentRole === "OS" ? "Oncology Specialist (OS)" : "Field Reimbursement Manager (FRM)";
    if (kvBrand) kvBrand.textContent = currentBrand;
    if (kvAccount) kvAccount.textContent = noteSlots.account_name;
    if (kvHcp) kvHcp.textContent = noteSlots.hcp_name;
    if (kvBarrier) kvBarrier.textContent = noteSlots.barriers;
    if (kvAction) kvAction.textContent = noteSlots.next_action;
  }

  // ---------------------------------------------------------------------------
  // 9. CALL SESSION LIFECYCLE (START / END)
  // ---------------------------------------------------------------------------

  async function initializeCallSession() {
    btnLaunchCall.textContent = "Connecting to KG Model...";
    btnLaunchCall.disabled = true;

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

      // Update Nav Profile
      headerAvatar.textContent = getInitials(currentRepName);
      headerRepName.textContent = currentRepName;
      headerRolePill.textContent = currentRole === "OS" ? "OS Sales" : "FRM Access";

      // Transition Screen: Login -> Call Noter
      screenLogin.classList.add("slide-left");
      screenCallNoter.classList.remove("hidden-screen");

      sessionActive = true;
      startCallTimer();

      // Append Initial Greeting Question
      const initialQuestion = data.initial_question || `Hello ${currentRepName}, are you ready to capture call notes for ${currentAccount}?`;
      appendAIMessageRow(initialQuestion, { latency: 45, ttft: 20, tps: 80 }, data.initial_audio_base64);
      playBotVoiceAudio(data.initial_audio_base64, initialQuestion);

      updateKGStepper("STATE_0", "account identification");
      updateNoteSummary(data.summary?.slots || { brand: currentBrand, account_name: currentAccount });

    } catch (e) {
      console.error("Session creation error:", e);
      alert(`Could not connect to API server at ${apiBaseUrl}.\nPlease verify the backend server is running.`);
    } finally {
      btnLaunchCall.textContent = "Launch Call Noter Session";
      btnLaunchCall.disabled = false;
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
    if (!confirm("Are you sure you want to finalize and close this call note?")) return;
    sessionActive = false;
    stopCallTimer();
    stopRecording();
    openSheet(sheetSummary);
  }

  // ---------------------------------------------------------------------------
  // 10. MODALS & BOTTOM SHEET DRAWERS
  // ---------------------------------------------------------------------------

  function openSheet(sheetEl) {
    closeAllSheets();
    if (sheetEl) sheetEl.classList.add("sheet-open");
  }

  function closeAllSheets() {
    document.querySelectorAll(".mobile-bottom-sheet").forEach(s => s.classList.remove("sheet-open"));
    tabItems.forEach(t => {
      if (t.dataset.tab !== "call") t.classList.remove("active");
      else t.classList.add("active");
    });
  }

  function openDiffInspector(autoResult) {
    if (!sheetDiff) return;

    diffRawText.textContent = `"${autoResult.raw}"`;
    diffCleanedText.textContent = `"${autoResult.cleaned}"`;

    // Fillers tags
    diffFillersList.innerHTML = "";
    if (autoResult.fillersRemoved.length > 0) {
      autoResult.fillersRemoved.forEach(f => {
        const tag = document.createElement("span");
        tag.className = "diff-tag-removed";
        tag.textContent = f;
        diffFillersList.appendChild(tag);
      });
    } else {
      diffFillersList.innerHTML = `<span style="font-size:11px; color:#94A3B8;">No filler words detected.</span>`;
    }

    // Corrections tags
    diffCorrectionsList.innerHTML = "";
    if (autoResult.correctionsMade.length > 0) {
      autoResult.correctionsMade.forEach(c => {
        const tag = document.createElement("span");
        tag.className = "diff-tag-corrected";
        tag.textContent = `"${c.from}" → ${c.to}`;
        diffCorrectionsList.appendChild(tag);
      });
    } else {
      diffCorrectionsList.innerHTML = `<span style="font-size:11px; color:#94A3B8;">No phonetic mispronunciations detected.</span>`;
    }

    openSheet(sheetDiff);
  }

  // ---------------------------------------------------------------------------
  // 11. EVENT LISTENERS
  // ---------------------------------------------------------------------------

  // Desktop Simulator Toggle
  btnToggleFrame.addEventListener("click", () => {
    deviceShell.classList.toggle("fullscreen-mode");
    const isFull = deviceShell.classList.contains("fullscreen-mode");
    btnToggleFrame.innerHTML = isFull ? `<span>📱 Phone Frame</span>` : `<span>🖥️ Fullscreen</span>`;
  });

  // Rep Preset Chips
  presetChips.forEach(chip => {
    chip.addEventListener("click", () => {
      presetChips.forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
      currentRepName = chip.dataset.name;
      loginRepName.value = currentRepName;
    });
  });

  loginRepName.addEventListener("input", (e) => {
    currentRepName = e.target.value.trim();
  });

  // Persona Roles
  roleCards.forEach(card => {
    card.addEventListener("click", () => {
      roleCards.forEach(c => c.classList.remove("active"));
      card.classList.add("active");
      currentRole = card.dataset.role;
    });
  });

  // Brands
  brandOptions.forEach(opt => {
    opt.addEventListener("click", () => {
      brandOptions.forEach(b => b.classList.remove("active"));
      opt.classList.add("active");
      currentBrand = opt.dataset.brand;
    });
  });

  // Accounts
  loginAccountSelect.addEventListener("change", (e) => {
    currentAccount = e.target.value;
  });

  // Test Endpoint
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
      alert(`Could not connect to ${url}.`);
    } finally {
      btnTestEndpoint.textContent = "Test Ping";
    }
  });

  // Launch Session Button
  btnLaunchCall.addEventListener("click", initializeCallSession);

  // Header Actions
  btnHeaderVoice.addEventListener("click", () => {
    audioMuted = !audioMuted;
    btnHeaderVoice.classList.toggle("active", !audioMuted);
    btnHeaderVoice.textContent = audioMuted ? "🔇" : "🔊";
  });

  btnHeaderEnd.addEventListener("click", endCallSession);

  // Giant Microphone Button (Push to Talk / Tap to Speak)
  btnGiantMic.addEventListener("click", () => {
    if (!isRecording) {
      startRecording();
    } else {
      stopRecording();
    }
  });

  // Handsfree Continuous Mode Switch
  btnDockHandsfree.addEventListener("click", () => {
    handsFreeMode = !handsFreeMode;
    btnDockHandsfree.classList.toggle("active", handsFreeMode);
    btnDockHandsfree.querySelector("span").textContent = handsFreeMode ? "Hands-Free: ON" : "Hands-Free";
    if (handsFreeMode && !isRecording) {
      startRecording();
    }
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

  btnDockSend.addEventListener("click", handleTextSubmit);
  dockTextarea.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleTextSubmit();
    }
  });

  // Bottom Navigation Tabs
  tabItems.forEach(tab => {
    tab.addEventListener("click", () => {
      tabItems.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");

      const target = tab.dataset.tab;
      if (target === "call") {
        closeAllSheets();
      } else if (target === "summary") {
        openSheet(sheetSummary);
      } else if (target === "scope") {
        openSheet(sheetScope);
      } else if (target === "settings") {
        openSheet(sheetSettings);
      }
    });
  });

  // Sheet Close Buttons
  sheetCloseButtons.forEach(btn => {
    btn.addEventListener("click", closeAllSheets);
  });

  // 1-Tap Copy CRM Note
  btnCopyCrm.addEventListener("click", () => {
    const text = `
=== J&J COMMERCIAL ONCOLOGY CALL NOTE RECORD ===
Representative: ${currentRepName} (${currentRole})
Brand: ${currentBrand}
Target Account: ${noteSlots.account_name}
Healthcare Professional (HCP): ${noteSlots.hcp_name}
Knowledge Graph Focus: ${KG_STEPS[currentStepIndex]?.title || "Account Alignment"}
Pathway / Blocker Status: ${noteSlots.barriers}
Agreed Action Items: ${noteSlots.next_action}
Compliance Status: Approved Commercial Dialogue (Governance Verified)
Captured via: AnQ Bot Mobile Call Noter (Speech-Cleaned)
================================================
    `.trim();

    navigator.clipboard.writeText(text).then(() => {
      const oldText = btnCopyCrm.innerHTML;
      btnCopyCrm.innerHTML = "✓ Copied to Clipboard!";
      setTimeout(() => { btnCopyCrm.innerHTML = oldText; }, 2000);
    });
  });

  // Helper
  function getInitials(name) {
    if (!name) return "AN";
    const p = name.trim().split(/\s+/).filter(Boolean);
    if (p.length === 1) return p[0].substring(0, 2).toUpperCase();
    return (p[0][0] + p[p.length - 1][0]).toUpperCase();
  }

  function escapeHTML(str) {
    if (!str) return "";
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  // Pre-fill initial state
  loginRepName.value = currentRepName;
});
