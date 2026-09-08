"""
Audio Transcription Engine (Whisper STT)
=========================================
Wraps faster-whisper (CTranslate2-optimized OpenAI Whisper) for
high-speed audio-to-text transcription.

Supports:
  - Multiple audio formats: WAV, MP3, WebM, OGG, FLAC, M4A, etc.
  - Auto-conversion to WAV via pydub/ffmpeg for non-WAV inputs
  - Configurable model size via WHISPER_MODEL_SIZE env var (default: base)
  - GPU-accelerated inference with CTranslate2

Usage:
    from engine.audio_transcriber import AudioTranscriber

    transcriber = AudioTranscriber()
    result = transcriber.transcribe("/path/to/audio.wav")
    print(result["transcript"])
"""

import os
import sys
import time
import math
import tempfile
import logging
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

# Supported audio extensions that can be directly read or converted
SUPPORTED_EXTENSIONS = {
    ".wav", ".mp3", ".webm", ".ogg", ".flac", ".m4a",
    ".aac", ".wma", ".opus", ".mp4", ".mpeg", ".mpga"
}

# Ensure FlowEdit is discoverable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
JNJ_COMMERCIAL_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(JNJ_COMMERCIAL_DIR)
FLOWEDIT_DIR = os.path.join(PROJECT_ROOT, "Flowedit")

for p in [FLOWEDIT_DIR, "/home/rsurya/projects/flow_edit/Flowedit", "/home/rsurya/projects/flow_edit"]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)


class AudioTranscriber:
    """
    Whisper-based Speech-to-Text engine enhanced with FlowEdit Hopfield Memory
    and S3 Spelling Store corrections.
    
    Loads a CTranslate2-optimized Whisper model for fast local inference,
    biases decoding vocabulary using Hopfield associative memory, and
    post-corrects transcripts using S3 phonetic dictionary and Hopfield memory.
    """

    def __init__(
        self,
        model_size: Optional[str] = None,
        compute_type: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the in-process Whisper transcription engine directly integrated with
        FlowEdit S3 Spelling Store and Hopfield Associative Memory.

        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large-v3').
                        Defaults to WHISPER_MODEL_SIZE env var or 'base'.
            compute_type: CTranslate2 compute type ('float16', 'int8', 'float32').
                          Defaults to WHISPER_COMPUTE_TYPE env var or auto-detect.
            device: Device to run on ('cuda', 'cpu'). Auto-detected if not specified.
        """
        self.model_size = model_size or os.getenv("WHISPER_MODEL_SIZE", "base")
        self.default_language = os.getenv("WHISPER_LANGUAGE", "en")
        self.device = device
        self.compute_type = compute_type or os.getenv("WHISPER_COMPUTE_TYPE", "")
        self._model = None
        self._hopfield_memory = None
        self._last_memory_mtime = 0.0

        # Load Whisper model in-process
        try:
            self._load_model()
        except ImportError as e:
            print(f"[AudioTranscriber] NOTE: {e}")

        # Connect FlowEdit Hopfield associative memory in-process
        self._load_hopfield_memory()

    def _find_memory_file(self) -> Optional[str]:
        """Find the Hopfield memory corrections.pt file across project locations."""
        cands = [
            os.getenv("FLOWEDIT_MEMORY_PATH", ""),
            "/home/rsurya/projects/flow_edit/corrections.pt",
            "/home/rsurya/projects/flow_edit/Flowedit/corrections.pt",
            os.path.join(FLOWEDIT_DIR, "corrections.pt"),
            os.path.join(PROJECT_ROOT, "corrections.pt"),
            "./corrections.pt",
        ]
        for c in cands:
            if c and os.path.isfile(c) and os.path.getsize(c) > 50:
                return os.path.abspath(c)
        return None

    def _load_hopfield_memory(self):
        """Load or reload FlowEdit Hopfield associative memory from corrections.pt."""
        mem_file = self._find_memory_file()
        if not mem_file:
            return
        try:
            mtime = os.path.getmtime(mem_file)
            if self._hopfield_memory is None or mtime > self._last_memory_mtime:
                from flowedit.memory.hopfield_memory import HopfieldMemory
                from flowedit.config import FlowEditConfig
                cfg = FlowEditConfig()
                self._hopfield_memory = HopfieldMemory(cfg.memory)
                self._hopfield_memory.load(mem_file)
                self._last_memory_mtime = mtime
                print(f"[AudioTranscriber] ✓ Connected FlowEdit Hopfield Memory: {self._hopfield_memory.num_entries} corrections loaded from {mem_file}")
        except Exception as e:
            logger.debug(f"[AudioTranscriber] Hopfield memory load notice: {e}")

    def _load_model(self):
        """Load the faster-whisper model with appropriate device configuration."""
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "\n[Error] 'faster-whisper' is required for audio transcription.\n"
                "Install it using:\n"
                "    pip install faster-whisper\n"
            )

        # Auto-detect device and compute type
        if not self.device:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"

        if not self.compute_type:
            self.compute_type = "float16" if self.device == "cuda" else "int8"

        # Check for model path or size: env var or explicit
        model_name_or_path = self.model_size
        env_model_path = os.getenv("WHISPER_MODEL_PATH")
        if env_model_path and (os.path.isdir(env_model_path) or os.path.isfile(env_model_path)):
            model_name_or_path = env_model_path
        else:
            for cand in [
                os.getenv("WHISPER_MODEL_DIR", ""),
                "/home/rsurya/projects/flow_edit/Flowedit/model/whisper",
                "/home/rsurya/projects/flow_edit/model/whisper",
            ]:
                if cand and os.path.isdir(cand) and (
                    os.path.isfile(os.path.join(cand, "model.bin")) or
                    os.path.isfile(os.path.join(cand, "model.safetensors"))
                ):
                    model_name_or_path = cand
                    break

        print(f"[AudioTranscriber] Loading Whisper '{model_name_or_path}' model "
              f"(device={self.device}, compute_type={self.compute_type}, default_lang={self.default_language})...")

        t0 = time.time()
        self._model = WhisperModel(
            model_name_or_path,
            device=self.device,
            compute_type=self.compute_type
        )
        load_time = time.time() - t0
        print(f"[AudioTranscriber] Whisper model loaded in {load_time:.2f}s")

    def _convert_to_wav(self, audio_path: str) -> str:
        """
        Convert non-WAV audio to WAV format using pydub.
        Returns the path to the converted WAV file (temp file).
        If already WAV, returns the original path.
        """
        ext = os.path.splitext(audio_path)[1].lower()
        if ext == ".wav":
            return audio_path

        try:
            from pydub import AudioSegment
        except ImportError:
            # faster-whisper natively decodes WebM, MP3, OGG, FLAC via PyAV without needing pydub
            logger.debug(f"[AudioTranscriber] pydub not installed; passing {ext} directly to faster-whisper PyAV decoder.")
            return audio_path

        logger.info(f"[AudioTranscriber] Converting {ext} -> WAV...")

        # Map extensions to pydub format strings
        format_map = {
            ".mp3": "mp3",
            ".webm": "webm",
            ".ogg": "ogg",
            ".flac": "flac",
            ".m4a": "m4a",
            ".aac": "aac",
            ".wma": "wma",
            ".opus": "opus",
            ".mp4": "mp4",
            ".mpeg": "mpeg",
            ".mpga": "mp3",
        }
        fmt = format_map.get(ext, ext.lstrip("."))

        audio = AudioSegment.from_file(audio_path, format=fmt)
        # Ensure 16kHz mono for Whisper
        audio = audio.set_frame_rate(16000).set_channels(1)

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()
        audio.export(tmp_path, format="wav")
        return tmp_path

    def get_vocabulary_prompt(self) -> str:
        """Extract vocabulary biasing prompt from Hopfield Memory and S3 Spelling Store."""
        prompt_parts = []
        if self._hopfield_memory and self._hopfield_memory.num_entries > 0:
            hop_prompt = self._hopfield_memory.get_vocabulary_prompt()
            if hop_prompt:
                prompt_parts.append(hop_prompt)

        try:
            from flowedit.memory.s3_storage import s3_spelling_store
            s3_prompt = s3_spelling_store.get_vocabulary_prompt()
            if s3_prompt:
                prompt_parts.append(s3_prompt)
        except Exception:
            pass

        return " ".join(prompt_parts).strip()

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        beam_size: int = 5
    ) -> Dict[str, Any]:
        """
        Transcribe an audio file to text.

        Args:
            audio_path: Path to the audio file.
            language: Language code (e.g. 'en'). Auto-detected if None.
            beam_size: Beam search width for decoding.

        Returns:
            dict with keys:
                - transcript (str): Full transcribed text
                - language (str): Detected or specified language
                - confidence (float): Average segment confidence (0.0-1.0)
                - duration_seconds (float): Audio duration in seconds
                - transcription_latency_ms (float): Processing time in ms
                - segments (list): Individual segment details
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        ext = os.path.splitext(audio_path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported audio format '{ext}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        if self._model is None:
            self._load_model()

        # Convert to WAV if necessary
        converted_path = self._convert_to_wav(audio_path)
        is_temp = converted_path != audio_path

        # Resolve target language: default to configured language ('en') to prevent foreign hallucinations
        target_lang = language if language is not None else self.default_language
        if target_lang and str(target_lang).lower() in ("auto", "none", ""):
            target_lang = None

        # Reload Hopfield memory if updated on disk (in-process hot-reload)
        self._load_hopfield_memory()

        initial_prompt = self.get_vocabulary_prompt() or None
        if initial_prompt:
            logger.debug(f"[AudioTranscriber] Biasing Whisper with vocabulary prompt: '{initial_prompt}'")

        try:
            t_start = time.perf_counter()

            transcribe_kwargs = {
                "language": target_lang,
                "beam_size": beam_size,
                "vad_filter": True,
                "vad_parameters": dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=200
                )
            }
            if initial_prompt:
                transcribe_kwargs["initial_prompt"] = initial_prompt

            segments_gen, info = self._model.transcribe(
                converted_path,
                **transcribe_kwargs
            )

            # Collect all segments
            segments = []
            full_text_parts = []
            total_confidence = 0.0

            for seg in segments_gen:
                segments.append({
                    "start": round(seg.start, 3),
                    "end": round(seg.end, 3),
                    "text": seg.text.strip(),
                    "avg_logprob": round(seg.avg_logprob, 4),
                    "no_speech_prob": round(seg.no_speech_prob, 4)
                })
                full_text_parts.append(seg.text.strip())
                # Convert log-prob to approximate confidence
                total_confidence += math.exp(seg.avg_logprob)

            raw_transcript = " ".join(full_text_parts).strip()
            transcript = raw_transcript
            s3_applied = []
            hopfield_applied = []

            # ─────────────────────────────────────────────────────────────
            # FlowEdit Autonomous Correction Stage 1: S3 Spelling Store
            # ─────────────────────────────────────────────────────────────
            try:
                from flowedit.memory.s3_storage import s3_spelling_store
                transcript, s3_applied = s3_spelling_store.apply_corrections_to_transcript(transcript)
                if s3_applied:
                    print(f"[AudioTranscriber] ✓ S3 Spelling correction applied: {s3_applied}")
            except Exception as e:
                logger.debug(f"[AudioTranscriber] S3 spelling correction notice: {e}")

            # ─────────────────────────────────────────────────────────────
            # FlowEdit Autonomous Correction Stage 2: Hopfield Memory
            # ─────────────────────────────────────────────────────────────
            if self._hopfield_memory and self._hopfield_memory.num_entries > 0:
                try:
                    transcript, hopfield_applied = self._hopfield_memory.correct_transcript(transcript)
                    if hopfield_applied:
                        print(f"[AudioTranscriber] ✓ Hopfield Memory correction applied: {hopfield_applied}")
                except Exception as e:
                    logger.debug(f"[AudioTranscriber] Hopfield transcript correction notice: {e}")

            avg_confidence = (total_confidence / len(segments)) if segments else 0.0
            latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

            result = {
                "transcript": transcript,
                "raw_transcript": raw_transcript,
                "language": info.language,
                "language_probability": round(info.language_probability, 4),
                "confidence": round(min(avg_confidence, 1.0), 4),
                "duration_seconds": round(info.duration, 2),
                "transcription_latency_ms": latency_ms,
                "segments": segments,
                "s3_corrections": s3_applied,
                "hopfield_corrections": hopfield_applied,
                "flowedit_memory_applied": bool(s3_applied or hopfield_applied),
                "hopfield_entries_count": self._hopfield_memory.num_entries if self._hopfield_memory else 0
            }

            logger.info(
                f"[AudioTranscriber] Transcribed {info.duration:.1f}s audio "
                f"in {latency_ms:.0f}ms -> \"{transcript[:80]}...\""
            )
            print(
                f"[AudioTranscriber] Transcription complete: "
                f"{info.duration:.1f}s audio -> {len(transcript)} chars "
                f"({latency_ms:.0f}ms, lang={info.language}, "
                f"conf={avg_confidence:.2f}, memory_applied={bool(s3_applied or hopfield_applied)})"
            )

            return result

        finally:
            # Clean up temp file if we created one
            if is_temp and os.path.exists(converted_path):
                try:
                    os.unlink(converted_path)
                except OSError:
                    pass

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio from raw bytes (e.g. from a file upload).

        Args:
            audio_bytes: Raw audio file bytes.
            filename: Original filename for format detection.
            language: Language code. Auto-detected if None.

        Returns:
            Same dict structure as transcribe().
        """
        ext = os.path.splitext(filename)[1].lower() or ".wav"
        tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        tmp_path = tmp.name
        try:
            tmp.write(audio_bytes)
            tmp.close()
            return self.transcribe(tmp_path, language=language)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
