# FlowEdit Real-Time Speech Synthesis (TTS) Streaming API Guide

**Server Base URL**: `http://<SERVER_HOST>:<PORT>` (e.g., `http://10.225.67.250:8008` or local `http://127.0.0.1:8000`)

This guide details the dedicated **Real-Time Speech Synthesis (TTS Streaming)** endpoints powered by FlowEdit and Hopfield Memory for lifelong pronunciation adaptation.

---

## Table of Contents
1. [Overview & Architecture](#1-overview--architecture)
2. [Speech Synthesis Streaming APIs](#2-speech-synthesis-streaming-apis)
   - [POST `/api/synthesize/stream` (Chunked WAV)](#21-post-apisynthesizestream---chunked-wav)
   - [POST `/api/synthesize/stream` (Server-Sent Events / SSE)](#22-post-apisynthesizestream---server-sent-events-sse)
   - [WebSocket `/api/ws/synthesize` (Real-Time Bi-Directional Stream)](#23-websocket-apiwssynthesize)
3. [Hopfield Memory Brand Grounding during Streaming](#3-hopfield-memory-brand-grounding-during-streaming)
4. [Verification & Test Results](#4-verification--test-results)
5. [cURL and Python Integration Examples](#5-curl-and-python-integration-examples)

---

## 1. Overview & Architecture

FlowEdit TTS Streaming enables low-latency, real-time speech generation with on-the-fly pronunciation corrections:

```
┌──────────────────────────────────────────────────────────────┐
│                  Modern Hopfield Memory                      │
│        (Learned Brand Names: Rybrevant, Inlexzo, etc.)       │
└──────────────────────────────┬───────────────────────────────┘
                               │
               Refined Target Embeddings Δ*
                               ▼
┌─────────────────────────┐        ┌─────────────────────────┐
│ Client (Browser/Device) │ <────  │ FlowEdit TTS Streaming  │
│   Plays Chunks Live     │        │ (Zero-delay Chunk Yield)│
└─────────────────────────┘        └─────────────────────────┘
```

- **Immediate Audio Playback**: Chunked audio/wav streaming prepends an open 44-byte streaming WAV header, allowing media players (`ffplay`, HTML5 `<audio>`, VLC) to begin playing speech immediately before full generation completes.
- **Diagnostics & Metadata**: SSE mode streams base64 audio frames alongside per-chunk latency, sample rate, and Hopfield memory gating telemetry.
- **Full-Duplex WebSockets**: Ultra-low-latency real-time synthesis over persistent WebSocket connections.

---

## 2. Speech Synthesis Streaming APIs

### 2.1. POST `/api/synthesize/stream` - Chunked WAV
Streams raw PCM 16-bit 24kHz audio chunks with an indefinite 44-byte WAV header.

- **URL**: `/api/synthesize/stream`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data` or `application/x-www-form-urlencoded`
- **Response Format**: `audio/wav` (Transfer-Encoding: chunked)

#### Parameters:
| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `text` | string | **Yes** | - | Text to synthesize. |
| `speaker_name` | string | No | `"female"` | Preset voice name (`"female"`, `"male"`). |
| `speaker_wav` | file | No | `None` | Reference WAV file for zero-shot voice cloning. |
| `stream_format` | string | No | `"wav"` | Output format: `"wav"` or `"sse"`. |
| `correction_scale` | float | No | `1.25` | Amplification scale for Hopfield memory perturbations. |
| `language` | string | No | `"en"` | Language code. |

#### cURL Example:
```bash
curl -N -X POST "http://localhost:8000/api/synthesize/stream" \
  -F "text=Evaluating Rybrevant efficacy in patients with non-small cell lung cancer." \
  -F "stream_format=wav" \
  --output live_stream.wav
```

---

### 2.2. POST `/api/synthesize/stream` - Server-Sent Events (SSE)
Emits `text/event-stream` chunks containing Base64-encoded audio and chunk-level diagnostics.

- **URL**: `/api/synthesize/stream`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data` or `application/x-www-form-urlencoded`
- **Response Format**: `text/event-stream`

#### cURL Example:
```bash
curl -N -X POST "http://localhost:8000/api/synthesize/stream" \
  -F "text=INLEXZO demonstrates high response rates in clinical trials." \
  -F "stream_format=sse"
```

#### SSE Event Payload:
```json
data: {
  "type": "chunk",
  "chunk_index": 0,
  "audio_b64": "UklGRi...",
  "sample_rate": 24000,
  "samples": 8000,
  "is_final": false,
  "is_modified": true,
  "latency_ms": 142.50,
  "diagnostics": {
    "matched_spans": [{"word": "INLEXZO", "score": 0.94, "gate": 0.98}]
  }
}

data: {"type": "final", "status": "SUCCESS"}
data: [DONE]
```

---

### 2.3. WebSocket `/api/ws/synthesize`
Bi-directional real-time speech synthesis connection.

- **URL**: `ws://localhost:8000/api/ws/synthesize`

#### Client Request (JSON):
```json
{
  "text": "Initiating real-time synthesis for Rybrevant.",
  "speaker_name": "female",
  "correction_scale": 1.25
}
```

#### Server Streaming Responses:
```json
{"type": "chunk", "chunk_index": 0, "audio_b64": "...", "sample_rate": 24000, "is_final": false, "is_modified": true}
{"type": "chunk", "chunk_index": 1, "audio_b64": "...", "sample_rate": 24000, "is_final": true, "is_modified": true}
{"type": "done"}
```

---

## 3. Hopfield Memory Brand Grounding during Streaming

During streaming synthesis:
1. When input text is received, `FlowEditInference.synthesize_stream` calculates base token embeddings.
2. It queries `HopfieldMemory` for brand names and specialized terminology.
3. If an active memory entry is matched (e.g. *Rybrevant*, *Inlexzo*), the optimal perturbation $\Delta^*$ is injected into the target embeddings on-the-fly.
4. The neural backbone generates speech chunks with the exact target pronunciation, without re-training or slowing down generation latency.

---

## 4. Verification & Test Results

The test suite [tests/test_tts_streaming_api.py](file:///c:/Users/AniruddhaJoshi/OneDrive%20-%20Info%20Origin%20Technologies%20Pvt%20Ltd/Desktop/flow_edit/Flowedit/tests/test_tts_streaming_api.py) verifies all TTS streaming components:

```
2026-09-04 18:07:47,109 - httpx2 - INFO - HTTP Request: POST http://testserver/api/synthesize/stream "HTTP/1.1 200 OK"
2026-09-04 18:07:47,121 - httpx2 - INFO - HTTP Request: POST http://testserver/api/synthesize/stream "HTTP/1.1 200 OK"
2026-09-04 18:07:47,129 - flowedit.api - INFO - Synthesis WebSocket client disconnected.

[SUCCESS] ALL TTS STREAMING TESTS (HTTP CHUNKED WAV, SSE, AND WEBSOCKET) PASSED PERFECTLY!
```

---

## 5. cURL and Python Integration Examples

### Python Client Streaming Example:
```python
import httpx

url = "http://localhost:8000/api/synthesize/stream"
data = {
    "text": "Rybrevant demonstrated significant efficacy in non-small cell lung cancer.",
    "stream_format": "wav",
    "speaker_name": "female",
}

with httpx.stream("POST", url, data=data) as response:
    with open("streamed_output.wav", "wb") as f:
        for chunk in response.iter_bytes():
            f.write(chunk)
            print(f"Received chunk: {len(chunk)} bytes")
```
