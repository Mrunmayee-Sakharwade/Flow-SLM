"""
Johnson & Johnson Innovative Medicine - Commercial Oncology API & Swagger UI
=============================================================================
FastAPI Application providing OpenAPI / Swagger UI interactive documentation
for the Knowledge-Graph-Conditioned SLM Next-Question Prediction pipeline.

Endpoints:
  - /docs                     : Interactive Swagger UI
  - /redoc                    : Interactive ReDoc
  - /                         : J&J Oncology Call Note Assistant Web Application
  - /api/kg/info              : Scoped Knowledge Graph Roles & Rules
  - /api/slm/explain_context  : Inspect the exact KG Subgraph & Prompt injected into SLM
  - /api/session/new          : Initialize a New Interview Session
  - /api/session/turn         : Multi-Turn NLU, KG Guardrails & KG-Conditioned SLM Prediction
  - /api/audio/transcribe     : Audio-to-Text Transcription (Whisper STT)
  - /api/session/audio_turn   : Audio Input → SLM Next-Question Prediction (Full Pipeline)
  - /api/session/audio_turn_stream : Audio Input → SLM Streaming Prediction (SSE)
"""

import os
import sys
import json
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

# Ensure Flowedit is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
FLOWEDIT_DIR = os.path.join(PROJECT_ROOT, "Flowedit")
if FLOWEDIT_DIR not in sys.path:
    sys.path.insert(0, FLOWEDIT_DIR)

try:
    from flowedit.memory.s3_storage import s3_spelling_store
except ImportError:
    s3_spelling_store = None

from engine.chatbot_pipeline import RuleGovernedCallBot
from engine.audio_transcriber import AudioTranscriber
from engine.audio_synthesizer import AudioSynthesizer

# Initialize FastAPI App with AnQ Bot Metadata
app = FastAPI(
    title="AnQ Bot - Commercial Call Intelligence API",
    description="""
## 🧬 Commercial Oncology Field Call Note Assistant API (KG-Conditioned SLM Stack)

This API provides **deterministic Knowledge Graph rule enforcement** and **generative SLM next-question prediction** for field teams:
* **Oncology Sales Representatives (OS)**
* **Field Reimbursement Managers (FRM)**

### Audio-Enabled Endpoints
Supports **voice input** via Whisper STT transcription integrated with the SLM prediction pipeline.
Upload audio files (WAV, MP3, WebM, etc.) and receive the next predicted interview question.
""",
    version="2.2.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Bot Engine
bot = RuleGovernedCallBot()

# Initialize Audio Transcription Engine (Whisper STT)
MAX_AUDIO_UPLOAD_MB = int(os.getenv("MAX_AUDIO_UPLOAD_MB", "25"))
print("Initializing Audio Transcriber (Whisper STT)...")
audio_transcriber = AudioTranscriber()

# Initialize Audio Synthesis Engine (FlowEdit TTS, Default: Michael)
DEFAULT_TTS_VOICE = os.getenv("DEFAULT_TTS_VOICE", "michael")
print(f"Initializing Audio Synthesizer (FlowEdit TTS, default voice: {DEFAULT_TTS_VOICE})...")
audio_synthesizer = AudioSynthesizer(default_voice=DEFAULT_TTS_VOICE)

WEB_DIR = os.path.join(os.path.dirname(__file__), "web")

# ---------------------------------------------------------------------------
# Pydantic Schemas for Swagger / OpenAPI Documentation
# ---------------------------------------------------------------------------
class NewSessionRequest(BaseModel):
    role: str = Field(
        default="FRM", 
        description="Active Field Persona Role", 
        examples=["FRM", "OS"]
    )
    brand: str = Field(
        default="RYBREVANT", 
        description="Oncology Product Focus", 
        examples=["RYBREVANT", "INLEXZO", "RYBREVANT + LAZCLUZE"]
    )
    user_name: Optional[str] = Field(
        default=None,
        description="Field Representative Name",
        examples=["Aniruddha Joshi"]
    )
    account_name: Optional[str] = Field(
        default=None,
        description="Pre-selected target healthcare account",
        examples=["Apollo Hospitals"]
    )
    generate_audio: Optional[bool] = Field(
        default=True,
        description="Synthesize initial greeting question as audio (Default voice: Michael)"
    )

class SessionResponse(BaseModel):
    session_id: str
    role: str
    brand: Optional[str]
    user_name: Optional[str] = None
    account_name: Optional[str] = None
    initial_question: str
    current_state: str
    summary: Dict[str, Any]
    initial_audio_base64: Optional[str] = Field(
        default=None,
        description="Synthesized audio speech of the initial greeting question (Michael voice)"
    )
    initial_audio_format: Optional[str] = Field(
        default="audio/mp3",
        description="Audio format / MIME type of the initial question"
    )

class TurnRequest(BaseModel):
    session_id: str = Field(
        ..., 
        description="Active Session ID obtained from /api/session/new",
        examples=["FRM_SESS_001"]
    )
    utterance: str = Field(
        ..., 
        description="Raw rep utterance / voice transcription",
        examples=[
            "I spoke with Dr. Robert Avery and Lisa regarding prior authorization denials.",
            "The oncologist asked about clinical trial efficacy and biomarker testing."
        ]
    )

class TurnResponse(BaseModel):
    status: str = Field(
        description="Interaction status: 'SUCCESS', 'SCOPE_VIOLATION_INTERCEPT', or 'COMPLIANCE_VIOLATION'"
    )
    role: Optional[str] = None
    bot_message: str = Field(
        description="Next predicted interview question or exact regulatory scope violation intercept message"
    )
    engine: Optional[str] = Field(
        default="SLM (KG-Conditioned)",
        description="Active Next-Question prediction engine"
    )
    model_identifier: Optional[str] = Field(
        default=None,
        description="Underlying Small Language Model identifier"
    )
    target_topic: Optional[str] = Field(
        default=None,
        description="The specific Knowledge Graph topic targeted for this interview turn"
    )
    applicable_responsibilities: Optional[List[str]] = Field(
        default=None,
        description="Official J&J CoreResponsibility statements retrieved from KG"
    )
    pending_unaddressed_topics: Optional[List[str]] = Field(
        default=None,
        description="Checklist of remaining unaddressed mandatory topics on role roadmap"
    )
    cross_functional_handoff: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Detected cross-functional collaboration referral (e.g. to MSL or FRM)"
    )
    kg_context_injected: Optional[str] = Field(
        default=None,
        description="Full formatted KG context injected into the SLM prompt"
    )
    rule_id: Optional[str] = Field(
        default=None, 
        description="Citing KG Rule ID if a violation occurred (e.g. resp:FRM:28, rule:privacy_phi_pii)"
    )
    rule_text: Optional[str] = Field(
        default=None,
        description="Verbatim policy or exclusion rule text from Knowledge Graph"
    )
    violated_topic: Optional[str] = Field(
        default=None,
        description="Topic that triggered the scope violation"
    )
    allowed_in_scope_topics: Optional[List[str]] = Field(
        default=None,
        description="List of permitted in-scope topics for the active role"
    )
    redacted_text: Optional[str] = Field(
        default=None,
        description="Scrubbed text when PHI or compliance language control is applied"
    )
    current_state: Optional[str] = None
    next_state: Optional[str] = None
    confidence: Optional[float] = Field(
        default=None, 
        description="Prediction confidence score"
    )
    detected_topics: Optional[List[str]] = None
    session_summary: Optional[Dict[str, Any]] = None
    ttft_ms: Optional[float] = Field(
        default=None,
        description="Time To First Token in milliseconds"
    )
    trt_ms: Optional[float] = Field(
        default=None,
        description="Total Response Time / Total Generation Time in milliseconds"
    )
    latency_ms: Optional[float] = Field(
        default=None,
        description="End-to-end turn processing latency in milliseconds"
    )
    latency_formatted: Optional[str] = Field(
        default=None,
        description="Formatted latency display string (e.g. TRT: 45ms (TTFT: 28ms))"
    )
    generated_tokens: Optional[int] = Field(
        default=None,
        description="Number of tokens generated in response"
    )
    num_tokens: Optional[int] = Field(
        default=None,
        description="Token count of generated response"
    )
    tokens_per_second: Optional[float] = Field(
        default=None,
        description="Generation throughput in tokens per second"
    )
    metrics: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Full performance and telemetry metrics payload"
    )
    barrier_case: Optional[str] = Field(
        default=None,
        description="Account barrier trigger case: CASE_1_USER_INITIATED or CASE_2_PROACTIVE_FOLLOWUP"
    )
    account_name: Optional[str] = Field(
        default=None,
        description="Normalized healthcare account name (e.g. Apollo Hospitals, Max Healthcare)"
    )
    live_graph: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Live materialized Entity-Relationship Knowledge Graph nodes and edges"
    )
    s3_corrections: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="S3 phonetic spelling corrections applied to the utterance"
    )

class AudioTranscribeResponse(BaseModel):
    """Response schema for standalone audio transcription."""
    transcript: str = Field(
        description="Full transcribed text from the audio input"
    )
    raw_transcript: Optional[str] = Field(
        default=None,
        description="Raw transcribed text before phonetic post-corrections"
    )
    language: str = Field(
        description="Detected language code (e.g. 'en')"
    )
    language_probability: float = Field(
        description="Language detection confidence (0.0-1.0)"
    )
    confidence: float = Field(
        description="Average transcription confidence score (0.0-1.0)"
    )
    duration_seconds: float = Field(
        description="Duration of the audio file in seconds"
    )
    transcription_latency_ms: float = Field(
        description="Whisper transcription processing time in milliseconds"
    )
    s3_corrections: Optional[List[Dict[str, Any]]] = Field(
        default=[],
        description="Phonetic corrections applied via FlowEdit S3 Spelling Store"
    )
    hopfield_corrections: Optional[List[Dict[str, Any]]] = Field(
        default=[],
        description="Acoustic corrections applied via FlowEdit Hopfield Associative Memory"
    )
    flowedit_memory_applied: Optional[bool] = Field(
        default=False,
        description="Whether FlowEdit memory corrections (S3 or Hopfield) were applied"
    )
    hopfield_entries_count: Optional[int] = Field(
        default=0,
        description="Number of active patterns in Hopfield Associative Memory"
    )

class AudioTurnResponse(TurnResponse):
    """Response schema for the combined full audio → SLM → audio pipeline."""
    transcript: Optional[str] = Field(
        default=None,
        description="Whisper-transcribed text from the uploaded audio"
    )
    raw_transcript: Optional[str] = Field(
        default=None,
        description="Raw transcribed text before phonetic post-corrections"
    )
    transcription_latency_ms: Optional[float] = Field(
        default=None,
        description="Whisper STT processing latency in milliseconds"
    )
    audio_duration_seconds: Optional[float] = Field(
        default=None,
        description="Duration of the uploaded audio file in seconds"
    )
    transcription_language: Optional[str] = Field(
        default=None,
        description="Detected language of the audio input"
    )
    transcription_confidence: Optional[float] = Field(
        default=None,
        description="Whisper transcription confidence score"
    )
    s3_corrections: Optional[List[Dict[str, Any]]] = Field(
        default=[],
        description="S3 phonetic corrections applied to incoming speech transcript"
    )
    hopfield_corrections: Optional[List[Dict[str, Any]]] = Field(
        default=[],
        description="Hopfield memory corrections applied to incoming speech transcript"
    )
    flowedit_memory_applied: Optional[bool] = Field(
        default=False,
        description="Whether FlowEdit memory was applied to the transcript"
    )
    bot_audio_base64: Optional[str] = Field(
        default=None,
        description="Base64-encoded audio waveform of the predicted next question (Default voice: Michael)"
    )
    bot_audio_format: Optional[str] = Field(
        default="audio/mp3",
        description="Audio format/MIME type of the synthesized speech response (e.g. audio/mp3 or audio/wav)"
    )
    bot_audio_duration_seconds: Optional[float] = Field(
        default=None,
        description="Synthesized audio speech duration in seconds"
    )
    tts_latency_ms: Optional[float] = Field(
        default=None,
        description="FlowEdit / Neural TTS speech synthesis latency in milliseconds"
    )
    total_turn_latency_ms: Optional[float] = Field(
        default=None,
        description="Total end-to-end turn latency (Whisper STT + SLM Prediction + Audio Synthesis) in ms"
    )
    voice: Optional[str] = Field(
        default="michael",
        description="Voice persona used for synthesizing the bot response (Default: michael)"
    )
    tts_s3_applied: Optional[List[Dict[str, Any]]] = Field(
        default=[],
        description="S3 phonetic respellings applied to outbound text before TTS synthesis"
    )
    tts_normalized_text: Optional[str] = Field(
        default=None,
        description="Phonetically respelled text sent to TTS engine"
    )

class SynthesizeRequest(BaseModel):
    """Request schema for standalone text-to-speech synthesis."""
    text: str = Field(
        ...,
        description="Text to synthesize into spoken audio (supports oncology terms like Rybrevant, Inlexzo, Lazcluze)"
    )
    voice: Optional[str] = Field(
        default="michael",
        description="Target voice persona (default: 'michael', conditioned on michael.wav; supports 'blessing')"
    )
    speed: Optional[float] = Field(
        default=1.0,
        description="Speech rate multiplier (default 1.0)"
    )

class ContextInspectionRequest(BaseModel):
    role: str = Field(default="FRM", examples=["FRM", "OS"])
    current_state: str = Field(default="STATE_3A_PRIOR_AUTH_PAYER")
    current_question: str = Field(default="What was the main access topic discussed?")
    candidate_answer: str = Field(default="We discussed prior authorization denials with Dr. Robert Avery.")
    brand: Optional[str] = Field(default="RYBREVANT")
    hcp: Optional[str] = Field(default="Dr. Robert Avery")

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/kg/info", tags=["Knowledge Graph"])
def get_kg_info():
    """
    Returns the Knowledge Graph schema, in-scope topics, excluded out-of-scope topics,
    governed exclusion rules, and active SLM Next-Question engine metadata.
    """
    return {
        "engine": "Knowledge-Graph-Conditioned SLM Next-Question Predictor",
        "model_identifier": bot.slm_engine.model_name,
        "total_slm_training_samples": 21811,
        "roles": {
            "OS": {
                "name": "Oncology Sales Representative",
                "primary_domain": "Commercial / Promotional Engagement",
                "in_scope_topics": list(bot.kg.in_scope_topics.get("OS", [])),
                "out_of_scope_topics": list(bot.kg.out_of_scope_topics.get("OS", [])),
                "exclusion_rules": [
                    {"id": r["id"], "text": r["properties"].get("text")} 
                    for r in bot.kg.exclusion_rules.get("OS", [])
                ]
            },
            "FRM": {
                "name": "Field Reimbursement Manager",
                "primary_domain": "Patient Access & Reimbursement",
                "in_scope_topics": list(bot.kg.in_scope_topics.get("FRM", [])),
                "out_of_scope_topics": list(bot.kg.out_of_scope_topics.get("FRM", [])),
                "exclusion_rules": [
                    {"id": r["id"], "text": r["properties"].get("text")} 
                    for r in bot.kg.exclusion_rules.get("FRM", [])
                ]
            }
        }
    }

@app.get("/api/accounts/barriers", tags=["Account Intelligence"])
def get_account_barriers():
    """
    Returns the manager-provided Account Barrier profiles across major healthcare accounts
    (Apollo Hospitals, Fortis Healthcare, Manipal Hospitals, Max Healthcare, Narayana Health)
    for OS (Patient Identification Barriers) and FRM (Market Access Barriers).
    """
    return {
        "status": "SUCCESS",
        "accounts": [
            "Apollo Hospitals",
            "Fortis Healthcare",
            "Manipal Hospitals",
            "Max Healthcare",
            "Narayana Health"
        ],
        "barriers": bot.kg.account_barriers if hasattr(bot.kg, "account_barriers") else []
    }

@app.post("/api/slm/explain_context", tags=["SLM Explainability"])
def explain_slm_context(req: ContextInspectionRequest):
    """
    Directly inspects the exact Knowledge Graph Subgraph and Prompt Context
    that is constructed and passed to the SLM for next-question prediction.
    """
    entities = {
        "brands": [req.brand] if req.brand else ["RYBREVANT"],
        "hcps": [req.hcp] if req.hcp else ["Doctor"]
    }
    
    nlu_res = bot.nlu.analyze_utterance(req.candidate_answer)
    
    context = bot.slm_engine.kg_retriever.build_slm_prompt_context(
        role=req.role,
        current_state=req.current_state,
        current_question=req.current_question,
        candidate_answer=req.candidate_answer,
        extracted_entities=entities,
        detected_topics=nlu_res["topics"]
    )
    
    prediction = bot.slm_engine.generate_next_question(
        role=req.role,
        current_state=req.current_state,
        current_question=req.current_question,
        candidate_answer=req.candidate_answer,
        extracted_entities=entities,
        detected_topics=nlu_res["topics"]
    )
    
    return {
        "role": req.role,
        "state": req.current_state,
        "target_topic": context["target_topic"],
        "applicable_duties": context["relevant_duties"],
        "unaddressed_topics_checklist": context["unaddressed_topics"],
        "collaboration_trigger": context["collaboration_trigger"],
        "full_kg_context_injected": context["kg_context_str"],
        "full_dialogue_context_injected": context["dialogue_context_str"],
        "slm_predicted_question": prediction["predicted_question"]
    }

@app.post("/api/session/new", response_model=SessionResponse, tags=["Dialogue Management"])
def create_session(req: NewSessionRequest):
    """
    Initializes a new multi-turn call note capture session for a specified field role, brand, rep, and account.
    """
    session = bot.create_session(
        role=req.role,
        brand=req.brand,
        user_name=req.user_name,
        account_name=req.account_name
    )
        
    initial_audio_b64 = None
    initial_audio_fmt = "audio/mp3"
    if req.generate_audio and session.current_question:
        try:
            synth_res = audio_synthesizer.synthesize(session.current_question, voice="michael")
            initial_audio_b64 = synth_res["audio_base64"]
            initial_audio_fmt = synth_res["audio_format"]
        except Exception as e:
            print(f"[create_session] Notice: Initial audio generation skipped: {e}")

    return {
        "session_id": session.session_id,
        "role": session.role,
        "brand": session.slots.get("brand"),
        "user_name": session.slots.get("user_name"),
        "account_name": session.slots.get("account_name"),
        "initial_question": session.current_question,
        "current_state": session.current_state,
        "summary": session.get_summary(),
        "initial_audio_base64": initial_audio_b64,
        "initial_audio_format": initial_audio_fmt
    }

@app.post("/api/session/turn", response_model=TurnResponse, tags=["Dialogue Management"])
def process_turn(req: TurnRequest):
    """
    Processes a single conversational turn:
    1. **NLU Extraction**: Extracts topics, HCPs, clinics, payers, timings.
    2. **Global Compliance**: Checks PHI/PII redaction and MIR routing.
    3. **KG Rule Scope Check**: Evaluates role permissions (e.g. FRM vs OS).
       - *If Violation*: Intercepts with exact KG Rule Citation (e.g. `resp:FRM:28`).
       - *If Compliant*: Advances FSM state and generates next question using KG-Conditioned SLM.
    """
    result = bot.process_turn(session_id=req.session_id, candidate_answer=req.utterance)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

@app.post("/api/session/turn_stream", tags=["Dialogue Management"])
def process_turn_stream(req: TurnRequest):
    """
    Streams conversational turn tokens with real-time SSE chunks, first-token latency,
    generation speed metrics, and final state result.
    """
    def event_generator():
        for chunk in bot.process_turn_stream(session_id=req.session_id, candidate_answer=req.utterance):
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "close",
            "Access-Control-Allow-Origin": "*"
        }
    )

# ---------------------------------------------------------------------------
# Audio-Enabled Endpoints (Whisper STT → SLM Pipeline)
# ---------------------------------------------------------------------------

@app.post("/api/audio/transcribe", response_model=AudioTranscribeResponse, tags=["Audio Pipeline"])
async def transcribe_audio(
    audio: UploadFile = File(..., description="Audio file to transcribe (WAV, MP3, WebM, OGG, FLAC, M4A, etc.)"),
    language: Optional[str] = Form(None, description="Language code (default: 'en'). Pass 'auto' for auto-detection.")
):
    """
    **Standalone Audio Transcription (Whisper STT)**

    Transcribes an uploaded audio file to text using the local Whisper model.
    Supports multiple formats: WAV, MP3, WebM, OGG, FLAC, M4A, AAC, OPUS, etc.

    Max upload size: configured via MAX_AUDIO_UPLOAD_MB (default 25MB).
    """
    # Validate file size
    contents = await audio.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_AUDIO_UPLOAD_MB:
        raise HTTPException(
            status_code=413,
            detail=f"Audio file too large ({size_mb:.1f}MB). Maximum allowed: {MAX_AUDIO_UPLOAD_MB}MB"
        )

    try:
        result = audio_transcriber.transcribe_bytes(
            audio_bytes=contents,
            filename=audio.filename or "audio.wav",
            language=language
        )
        return {
            "transcript": result["transcript"],
            "raw_transcript": result.get("raw_transcript", result["transcript"]),
            "language": result["language"],
            "language_probability": result["language_probability"],
            "confidence": result["confidence"],
            "duration_seconds": result["duration_seconds"],
            "transcription_latency_ms": result["transcription_latency_ms"],
            "s3_corrections": result.get("s3_corrections", []),
            "hopfield_corrections": result.get("hopfield_corrections", []),
            "flowedit_memory_applied": result.get("flowedit_memory_applied", False),
            "hopfield_entries_count": result.get("hopfield_entries_count", 0),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.post("/api/session/audio_turn", response_model=AudioTurnResponse, tags=["Audio Pipeline"])
async def process_audio_turn(
    audio: UploadFile = File(..., description="Audio file containing the rep's spoken response"),
    session_id: str = Form(..., description="Active session ID from /api/session/new"),
    voice: Optional[str] = Form(None, description="Voice persona for bot speech response (default: 'michael')"),
    language: Optional[str] = Form(None, description="Transcription language code (default: 'en')")
):
    """
    **Full Audio → SLM Pipeline (Audio Input → Audio + Text Predicted Next Question)**

    Bi-directional voice pipeline integrating Whisper STT, KG-Conditioned SLM, and FlowEdit TTS:

    1. **Audio Upload** → Receives the field rep's spoken response as an audio file
    2. **Whisper STT** → Transcribes audio to text using local Whisper model
    3. **NLU Extraction** → Extracts topics, entities, and compliance signals
    4. **KG Rule Check** → Evaluates role scope permissions and compliance
    5. **SLM Prediction** → Generates the next interview question via KG-Conditioned SLM
    6. **FlowEdit TTS** → Synthesizes the predicted question into natural speech (Default voice: Michael)

    Returns the complete turn result containing both verbatim text and playable base64 audio.
    """
    # Validate file size
    contents = await audio.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_AUDIO_UPLOAD_MB:
        raise HTTPException(
            status_code=413,
            detail=f"Audio file too large ({size_mb:.1f}MB). Maximum allowed: {MAX_AUDIO_UPLOAD_MB}MB"
        )

    # Step 1: Transcribe audio (Whisper STT)
    try:
        stt_result = audio_transcriber.transcribe_bytes(
            audio_bytes=contents,
            filename=audio.filename or "audio.wav",
            language=language
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio transcription failed: {str(e)}")

    transcript = stt_result["transcript"]
    if not transcript or not transcript.strip():
        raise HTTPException(
            status_code=422,
            detail="No speech detected in the uploaded audio. Please try again with clearer audio."
        )

    # Step 2: Feed transcript into existing SLM pipeline
    result = bot.process_turn(session_id=session_id, candidate_answer=transcript)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    # Step 3: Synthesize predicted next question to speech (Default voice: Michael)
    bot_message = result.get("bot_message", "")
    target_voice = voice or DEFAULT_TTS_VOICE
    tts_latency = 0.0
    bot_audio_b64 = None
    bot_audio_fmt = "audio/mp3"
    audio_duration = None
    synth_res = None

    if bot_message:
        try:
            synth_res = await audio_synthesizer.synthesize_async(bot_message, voice=target_voice)
            bot_audio_b64 = synth_res.get("audio_base64")
            bot_audio_fmt = synth_res.get("audio_format", "audio/mp3")
            audio_duration = synth_res.get("duration_seconds")
            tts_latency = synth_res.get("tts_latency_ms", 0.0)
        except Exception as e:
            print(f"[process_audio_turn] Speech synthesis error: {e}")

    stt_latency = stt_result.get("transcription_latency_ms", 0.0)
    slm_latency = result.get("latency_ms", 0.0)

    # Enrich response with transcription and speech synthesis metadata
    result["transcript"] = transcript
    result["raw_transcript"] = stt_result.get("raw_transcript", transcript)
    result["transcription_latency_ms"] = stt_latency
    result["audio_duration_seconds"] = stt_result.get("duration_seconds", 0.0)
    result["transcription_language"] = stt_result.get("language")
    result["transcription_confidence"] = stt_result.get("confidence")
    result["s3_corrections"] = stt_result.get("s3_corrections", [])
    result["hopfield_corrections"] = stt_result.get("hopfield_corrections", [])
    result["flowedit_memory_applied"] = stt_result.get("flowedit_memory_applied", False)

    result["bot_audio_base64"] = bot_audio_b64
    result["bot_audio_format"] = bot_audio_fmt
    result["bot_audio_duration_seconds"] = audio_duration
    result["tts_latency_ms"] = tts_latency
    result["total_turn_latency_ms"] = round(stt_latency + slm_latency + tts_latency, 1)
    result["voice"] = target_voice
    result["tts_s3_applied"] = synth_res.get("s3_applied", []) if synth_res else []
    result["tts_normalized_text"] = synth_res.get("normalized_text", bot_message) if synth_res else bot_message

    return result


@app.post("/api/session/audio_turn_stream", tags=["Audio Pipeline"])
async def process_audio_turn_stream(
    audio: UploadFile = File(..., description="Audio file containing the rep's spoken response"),
    session_id: str = Form(..., description="Active session ID from /api/session/new"),
    voice: Optional[str] = Form(None, description="Voice persona for bot speech response (default: 'michael')"),
    language: Optional[str] = Form(None, description="Transcription language code (default: 'en')")
):
    """
    **Streaming Audio → SLM Pipeline (Audio Input → SSE Token Stream → Audio Speech)**

    Streams the SLM prediction token-by-token via Server-Sent Events (SSE).
    Events emitted:
      1. `transcription`: Transcribed rep utterance metadata
      2. `token`: Individual tokens as generated by SLM
      3. `metrics`: Generation telemetry (TTFT, TPS)
      4. `result`: Final turn state and question text
      5. `audio`: Synthesized speech audio of the predicted question (Michael voice)
    """
    # Validate file size
    contents = await audio.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_AUDIO_UPLOAD_MB:
        raise HTTPException(
            status_code=413,
            detail=f"Audio file too large ({size_mb:.1f}MB). Maximum allowed: {MAX_AUDIO_UPLOAD_MB}MB"
        )

    # Step 1: Transcribe audio
    try:
        stt_result = audio_transcriber.transcribe_bytes(
            audio_bytes=contents,
            filename=audio.filename or "audio.wav",
            language=language
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio transcription failed: {str(e)}")

    transcript = stt_result["transcript"]
    if not transcript or not transcript.strip():
        raise HTTPException(
            status_code=422,
            detail="No speech detected in the uploaded audio. Please try again with clearer audio."
        )

    target_voice = voice or DEFAULT_TTS_VOICE

    # Step 2: Stream SLM response followed by synthesized speech event
    def audio_event_generator():
        # First event: transcription result
        yield f"data: {json.dumps({'type': 'transcription', 'transcript': transcript, 'raw_transcript': stt_result.get('raw_transcript', transcript), 'transcription_latency_ms': stt_result['transcription_latency_ms'], 'audio_duration_seconds': stt_result['duration_seconds'], 'language': stt_result['language'], 'confidence': stt_result['confidence'], 's3_corrections': stt_result.get('s3_corrections', []), 'hopfield_corrections': stt_result.get('hopfield_corrections', []), 'flowedit_memory_applied': stt_result.get('flowedit_memory_applied', False)})}\n\n"

        final_bot_message = ""
        # Stream SLM turn processing
        for chunk in bot.process_turn_stream(session_id=session_id, candidate_answer=transcript):
            if chunk.get("type") == "result":
                final_bot_message = chunk.get("bot_message", "")
            elif chunk.get("type") == "token":
                final_bot_message += chunk.get("token", "")
            yield f"data: {json.dumps(chunk)}\n\n"

        # Synthesize the final bot message into audio (Michael voice)
        if final_bot_message:
            try:
                synth_res = audio_synthesizer.synthesize(final_bot_message, voice=target_voice)
                audio_payload = {
                    "type": "audio",
                    "audio_base64": synth_res["audio_base64"],
                    "audio_format": synth_res["audio_format"],
                    "duration_seconds": synth_res["duration_seconds"],
                    "tts_latency_ms": synth_res["tts_latency_ms"],
                    "voice": synth_res["voice"],
                    "s3_applied": synth_res.get("s3_applied", []),
                    "normalized_text": synth_res.get("normalized_text", final_bot_message)
                }
                yield f"data: {json.dumps(audio_payload)}\n\n"
            except Exception as e:
                logger.warning(f"Streaming TTS generation error: {e}")

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        audio_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "close",
            "Access-Control-Allow-Origin": "*"
        }
    )


@app.post("/api/audio/synthesize", tags=["Audio Pipeline"])
async def synthesize_speech(req: SynthesizeRequest):
    """
    **Standalone Text-to-Speech Synthesis (FlowEdit TTS, Default: Michael)**

    Synthesizes arbitrary text into natural speech with zero-shot speaker conditioning
    and brand-aware oncology pronunciation normalizations.
    """
    target_voice = req.voice or DEFAULT_TTS_VOICE
    synth_res = await audio_synthesizer.synthesize_async(req.text, voice=target_voice)

    return {
        "text": req.text,
        "normalized_text": synth_res.get("normalized_text", req.text),
        "s3_applied": synth_res.get("s3_applied", []),
        "hopfield_active": synth_res.get("hopfield_active", False),
        "voice": synth_res["voice"],
        "engine": synth_res["engine"],
        "audio_format": synth_res["audio_format"],
        "audio_base64": synth_res["audio_base64"],
        "duration_seconds": synth_res["duration_seconds"],
        "tts_latency_ms": synth_res["tts_latency_ms"]
    }


@app.get("/api/audio/voices", tags=["Audio Pipeline"])
def list_available_voices():
    """List available voice personas and defaults."""
    return {
        "default_voice": DEFAULT_TTS_VOICE,
        "available_voices": list(audio_synthesizer.preset_voices.keys()),
        "voice_details": {
            k: {
                "name": v["name"],
                "gender": v["gender"],
                "has_reference_wav": bool(v.get("speaker_wav") and os.path.exists(v["speaker_wav"]))
            }
            for k, v in audio_synthesizer.preset_voices.items()
        }
    }

# Static Web Application Mounting
if os.path.exists(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def serve_ui():
        return FileResponse(os.path.join(WEB_DIR, "index.html"))

    @app.get("/style.css", include_in_schema=False)
    def serve_css():
        return FileResponse(os.path.join(WEB_DIR, "style.css"))

    @app.get("/app.js", include_in_schema=False)
    def serve_js():
        return FileResponse(os.path.join(WEB_DIR, "app.js"))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    print("\n============================================================")
    print(" Johnson & Johnson Commercial Oncology API & Swagger Live!")
    print(f" Swagger Documentation : http://0.0.0.0:{port}/docs")
    print(f" ReDoc Documentation   : http://0.0.0.0:{port}/redoc")
    print(f" Web User Interface    : http://0.0.0.0:{port}")
    print("============================================================\n")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

