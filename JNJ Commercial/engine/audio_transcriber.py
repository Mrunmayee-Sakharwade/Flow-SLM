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
import time
import math
import tempfile
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Supported audio extensions that can be directly read or converted
SUPPORTED_EXTENSIONS = {
    ".wav", ".mp3", ".webm", ".ogg", ".flac", ".m4a",
    ".aac", ".wma", ".opus", ".mp4", ".mpeg", ".mpga"
}


class AudioTranscriber:
    """
    Whisper-based Speech-to-Text engine using faster-whisper.
    
    Loads a CTranslate2-optimized Whisper model for fast local inference.
    Automatically handles format conversion for non-WAV audio inputs.
    """

    def __init__(
        self,
        model_size: Optional[str] = None,
        compute_type: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the Whisper transcription engine.

        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large-v3').
                        Defaults to WHISPER_MODEL_SIZE env var or 'base'.
            compute_type: CTranslate2 compute type ('float16', 'int8', 'float32').
                          Defaults to WHISPER_COMPUTE_TYPE env var or auto-detect.
            device: Device to run on ('cuda', 'cpu'). Auto-detected if not specified.
        """
        self.model_size = model_size or os.getenv("WHISPER_MODEL_SIZE", "base")
        self.device = device
        self.compute_type = compute_type or os.getenv("WHISPER_COMPUTE_TYPE", "")
        self._model = None
        try:
            self._load_model()
        except ImportError as e:
            print(f"[AudioTranscriber] NOTE: {e}")

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

        print(f"[AudioTranscriber] Loading Whisper '{self.model_size}' model "
              f"(device={self.device}, compute_type={self.compute_type})...")

        t0 = time.time()
        self._model = WhisperModel(
            self.model_size,
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
            raise ImportError(
                "\n[Error] 'pydub' is required for audio format conversion.\n"
                "Install it using:\n"
                "    pip install pydub\n"
                "Also ensure ffmpeg is installed on your system.\n"
            )

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

        try:
            t_start = time.perf_counter()

            segments_gen, info = self._model.transcribe(
                converted_path,
                language=language,
                beam_size=beam_size,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=200
                )
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

            transcript = " ".join(full_text_parts).strip()
            avg_confidence = (total_confidence / len(segments)) if segments else 0.0
            latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

            result = {
                "transcript": transcript,
                "language": info.language,
                "language_probability": round(info.language_probability, 4),
                "confidence": round(min(avg_confidence, 1.0), 4),
                "duration_seconds": round(info.duration, 2),
                "transcription_latency_ms": latency_ms,
                "segments": segments
            }

            logger.info(
                f"[AudioTranscriber] Transcribed {info.duration:.1f}s audio "
                f"in {latency_ms:.0f}ms -> \"{transcript[:80]}...\""
            )
            print(
                f"[AudioTranscriber] Transcription complete: "
                f"{info.duration:.1f}s audio -> {len(transcript)} chars "
                f"({latency_ms:.0f}ms, lang={info.language}, "
                f"conf={avg_confidence:.2f})"
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
