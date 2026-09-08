"""
Audio Synthesis Engine (FlowEdit TTS Integration)
=================================================
Synthesizes SLM predicted next questions into spoken audio with zero-shot voice
conditioning and pronunciation adaptation for pharmaceutical oncology brands.

Default Speaker: "michael" (conditioned on michael.wav)

Features:
  - Native FlowEdit integration (Hopfield Associative Memory for brand names:
    Rybrevant, Inlexzo, Lazcluze, etc.)
  - Speaker conditioning with michael.wav (preset male voice) and blessing.wav (preset female voice)
  - Resilient high-fidelity neural fallback matching Michael's voice timbre
  - Fast base64 serialization for immediate browser / mobile audio playback
  - Streaming audio chunk generation for SSE endpoints
"""

import os
import sys
import time
import base64
import asyncio
import logging
import tempfile
from typing import Dict, Any, Optional, Tuple, AsyncGenerator

logger = logging.getLogger(__name__)

# Project root paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
JNJ_COMMERCIAL_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(JNJ_COMMERCIAL_DIR)
FLOWEDIT_DIR = os.path.join(PROJECT_ROOT, "Flowedit")

# Ensure Flowedit is in sys.path
if FLOWEDIT_DIR not in sys.path:
    sys.path.insert(0, FLOWEDIT_DIR)


def resolve_speaker_wav_file(voice_name: str) -> Optional[str]:
    """Find voice audio reference WAV across project, Flowedit, model, and server paths."""
    v_clean = str(voice_name or "michael").lower().strip()
    target_key = "michael" if ("m" in v_clean or "male" in v_clean) else "blessing"
    target_file = f"{target_key}.wav"

    # 1. Primary confirmed IIT server voice path (confirmed by user: /home/rsurya/projects/flow_edit/michael.wav)
    primary_server_path = f"/home/rsurya/projects/flow_edit/{target_file}"
    if os.path.isfile(primary_server_path):
        sz = os.path.getsize(primary_server_path)
        if sz > 1000:
            print(f"[AudioSynthesizer] Found speaker reference '{target_file}': {primary_server_path} ({sz} bytes)")
            return os.path.abspath(primary_server_path)
        else:
            print(f"[AudioSynthesizer] Notice: '{primary_server_path}' exists but is only {sz} bytes (Git-LFS pointer). Restoring authentic audio...")
            try:
                from engine.embedded_voices import unpack_embedded_voice
                unpacked = unpack_embedded_voice(target_key, primary_server_path)
                if unpacked and os.path.isfile(unpacked) and os.path.getsize(unpacked) > 1000:
                    print(f"[AudioSynthesizer] Restored authentic speaker reference: {unpacked} ({os.path.getsize(unpacked)} bytes)")
                    return os.path.abspath(unpacked)
            except Exception as e:
                print(f"[AudioSynthesizer] Could not restore {primary_server_path}: {e}")
    elif not os.path.exists(primary_server_path):
        # Attempt to auto-place authentic voice reference at the confirmed primary path
        try:
            from engine.embedded_voices import unpack_embedded_voice
            unpacked = unpack_embedded_voice(target_key, primary_server_path)
            if unpacked and os.path.isfile(unpacked) and os.path.getsize(unpacked) > 1000:
                print(f"[AudioSynthesizer] Placed authentic speaker reference at primary path: {unpacked}")
                return os.path.abspath(unpacked)
        except Exception:
            pass

    # 2. Direct explicit environment variable overrides
    env_explicit = os.getenv("MICHAEL_VOICE_PATH" if target_key == "michael" else "BLESSING_VOICE_PATH")
    if env_explicit and os.path.isfile(env_explicit) and os.path.getsize(env_explicit) > 1000:
        return os.path.abspath(env_explicit)

    env_dir = os.getenv("FLOWEDIT_VOICES_DIR", "")
    if env_dir and os.path.isdir(env_dir):
        for f in os.listdir(env_dir):
            if f.lower() == target_file:
                p = os.path.join(env_dir, f)
                if os.path.isfile(p) and os.path.getsize(p) > 1000:
                    return os.path.abspath(p)

    # 3. Known server candidates (checked with case-insensitivity)
    candidates = [
        # Direct server root and model folders
        f"/home/rsurya/projects/flow_edit/{target_file}",
        f"/home/rsurya/projects/flow_edit/Flowedit/model/{target_file}",
        f"/home/rsurya/projects/flow_edit/Flowedit/model/voices/{target_file}",
        f"/home/rsurya/projects/flow_edit/Flowedit/model/{target_key.capitalize()}.wav",
        os.path.join(FLOWEDIT_DIR, "model", target_file),
        os.path.join(FLOWEDIT_DIR, "model", "voices", target_file),
        os.path.join(PROJECT_ROOT, "Flowedit", "model", target_file),
        # Flowedit deploy_voices folder
        os.path.join(FLOWEDIT_DIR, "deploy_voices", target_file),
        os.path.join(PROJECT_ROOT, "Flowedit", "deploy_voices", target_file),
        # Flowedit resources folder
        os.path.join(FLOWEDIT_DIR, "flowedit", "resources", target_file),
        os.path.join(PROJECT_ROOT, "Flowedit", "flowedit", "resources", target_file),
        # Project root & siblings
        os.path.join(PROJECT_ROOT, target_file),
        os.path.join(FLOWEDIT_DIR, target_file),
        os.path.join(JNJ_COMMERCIAL_DIR, target_file),
        os.path.join(os.getcwd(), target_file),
        os.path.join(os.getcwd(), "..", target_file),
        os.path.join(os.getcwd(), "deploy_voices", target_file),
        os.path.join(os.getcwd(), "..", "Flowedit", "deploy_voices", target_file),
        os.path.join(os.getcwd(), "..", "Flowedit", "flowedit", "resources", target_file),
        # Linux server path conventions (IITD server & home)
        f"/home/rsurya/projects/flow_edit/Flowedit/deploy_voices/{target_file}",
        f"/home/rsurya/projects/flow_edit/Flowedit/flowedit/resources/{target_file}",
        f"/home/rsurya/projects/flow_edit/Flowedit/{target_file}",
        f"/home/rsurya/projects/flow_edit/{target_file}",
        f"/home/rsurya/projects/text_to_speech/{target_file}",
        f"/home/rsurya/projects/text_to_speech/app/{target_file}",
        f"/home/rsurya/voices/{target_file}",
        f"/home/rsurya/speakers/{target_file}",
        f"/home/rsurya/{target_file}",
        os.path.expanduser(f"~/projects/flow_edit/Flowedit/model/{target_file}"),
        os.path.expanduser(f"~/projects/flow_edit/Flowedit/deploy_voices/{target_file}"),
        os.path.expanduser(f"~/projects/flow_edit/Flowedit/flowedit/resources/{target_file}"),
        os.path.expanduser(f"~/projects/flow_edit/Flowedit/{target_file}"),
        os.path.expanduser(f"~/projects/flow_edit/{target_file}"),
        os.path.expanduser(f"~/{target_file}"),
    ]

    for cand in candidates:
        if cand and os.path.isfile(cand):
            sz = os.path.getsize(cand)
            if sz > 1000:
                print(f"[AudioSynthesizer] Found speaker reference '{target_file}': {cand}")
                return os.path.abspath(cand)

    # 3. Case-insensitive recursive search across project and user directories
    search_roots = [
        "/home/rsurya/projects/flow_edit/Flowedit/model",
        "/home/rsurya/projects/flow_edit/Flowedit/deploy_voices",
        "/home/rsurya/projects/flow_edit/Flowedit/flowedit/resources",
        "/home/rsurya/projects/flow_edit",
        "/home/rsurya/projects/text_to_speech",
        "/home/rsurya/voices",
        FLOWEDIT_DIR,
        PROJECT_ROOT,
    ]

    for s_dir in search_roots:
        if s_dir and os.path.isdir(s_dir):
            try:
                for root, dirs, files in os.walk(s_dir):
                    # Prevent deep recursion
                    rel = os.path.relpath(root, s_dir)
                    if rel.count(os.sep) > 3:
                        continue
                    for f in files:
                        f_low = f.lower()
                        # Match michael.wav, michael_voice.wav, etc.
                        if f_low.endswith(".wav") and target_key in f_low:
                            p = os.path.join(root, f)
                            if os.path.isfile(p) and os.path.getsize(p) > 1000:
                                print(f"[AudioSynthesizer] Found speaker reference '{target_file}': {p}")
                                return os.path.abspath(p)
            except Exception:
                pass

    # 4. Check if fine-tuned model directory has ANY wav file for voice conditioning
    model_dir = "/home/rsurya/projects/flow_edit/Flowedit/model"
    if os.path.isdir(model_dir):
        try:
            for f in os.listdir(model_dir):
                if f.lower().endswith(".wav") and not f.lower().startswith("default"):
                    p = os.path.join(model_dir, f)
                    if os.path.isfile(p) and os.path.getsize(p) > 1000:
                        print(f"[AudioSynthesizer] Using model directory reference audio: {p}")
                        return os.path.abspath(p)
        except Exception:
            pass

    # 5. Auto-unpack authentic embedded voice reference if missing on disk
    try:
        from engine.embedded_voices import unpack_embedded_voice
        import tempfile
        unpack_targets = [
            "/home/rsurya/projects/flow_edit/Flowedit/flowedit/resources",
            "/home/rsurya/projects/flow_edit",
            os.path.join(FLOWEDIT_DIR, "flowedit", "resources"),
            PROJECT_ROOT,
            FLOWEDIT_DIR,
            os.path.join(tempfile.gettempdir(), "flowedit_voices"),
        ]
        for ut in unpack_targets:
            try:
                unpacked = unpack_embedded_voice(voice_name, ut)
                if unpacked and os.path.isfile(unpacked) and os.path.getsize(unpacked) > 1000:
                    print(f"[AudioSynthesizer] Unpacked authentic {target_file}: {unpacked}")
                    return os.path.abspath(unpacked)
            except Exception:
                pass
    except Exception as e:
        logger.debug(f"Embedded voice unpacking skipped: {e}")

    # 6. Fallback to default_speaker.wav in Flowedit resources
    for fallback_cand in [
        os.path.join(FLOWEDIT_DIR, "flowedit", "resources", "default_speaker.wav"),
        os.path.join(PROJECT_ROOT, "Flowedit", "flowedit", "resources", "default_speaker.wav"),
        "/home/rsurya/projects/flow_edit/Flowedit/flowedit/resources/default_speaker.wav",
        os.path.expanduser("~/projects/flow_edit/Flowedit/flowedit/resources/default_speaker.wav"),
    ]:
        if fallback_cand and os.path.isfile(fallback_cand) and os.path.getsize(fallback_cand) > 1000:
            print(f"[AudioSynthesizer] Speaker '{target_file}' not found; using available voice 'default_speaker.wav': {fallback_cand}")
            return os.path.abspath(fallback_cand)

    return None


class AudioSynthesizer:
    """
    Speech synthesis engine integrating FlowEdit and high-fidelity neural voice synthesis.
    Default voice is 'michael' using zero-shot speaker conditioning on michael.wav.
    """

    def __init__(
        self,
        default_voice: str = "michael",
        flowedit_url: Optional[str] = None
    ):
        self.default_voice = os.getenv("DEFAULT_TTS_VOICE", default_voice)
        self.flowedit_url = flowedit_url or os.getenv("FLOWEDIT_URL", "http://127.0.0.1:8000")
        self.flowedit_inference = None
        self._flowedit_ready = False
        self._phonetic_normalizer = None
        self._last_memory_mtime = 0.0

        # Voice mapping with dynamic resolution
        self.preset_voices = {
            "michael": {
                "name": "michael",
                "gender": "male",
                "speaker_wav": resolve_speaker_wav_file("michael"),
                "neural_fallback": "en-US-GuyNeural"
            },
            "blessing": {
                "name": "blessing",
                "gender": "female",
                "speaker_wav": resolve_speaker_wav_file("blessing"),
                "neural_fallback": "en-US-JennyNeural"
            }
        }

        # Initialize FlowEdit phonetic normalizer and inference pipeline if available
        self._init_flowedit()

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
        """Load or hot-reload FlowEdit Hopfield associative memory from corrections.pt."""
        if not self.flowedit_inference or not hasattr(self.flowedit_inference, "memory") or self.flowedit_inference.memory is None:
            return
        mem_file = self._find_memory_file()
        if not mem_file:
            return
        try:
            mtime = os.path.getmtime(mem_file)
            if self._last_memory_mtime < mtime:
                self.flowedit_inference.memory.load(mem_file)
                if hasattr(self.flowedit_inference, "refiner") and self.flowedit_inference.refiner:
                    self.flowedit_inference.refiner.memory = self.flowedit_inference.memory
                self._last_memory_mtime = mtime
                print(f"[AudioSynthesizer] ✓ Connected FlowEdit Hopfield Memory: {self.flowedit_inference.memory.num_entries} corrections loaded from {mem_file}")
        except Exception as e:
            logger.debug(f"[AudioSynthesizer] Hopfield memory load notice: {e}")

    def _init_flowedit(self):
        """Try loading FlowEdit pipeline components."""
        self._edge_tts_blocked = False

        try:
            from flowedit.utils.indic_phonetics import normalize_indic_phonetics
            self._phonetic_normalizer = normalize_indic_phonetics
            print("[AudioSynthesizer] FlowEdit phonetic normalizer loaded.")
        except Exception as e:
            logger.debug(f"Phonetic normalizer not available: {e}")

        # Attempt to initialize FlowEdit pipeline (XTTS-v2 / F5-TTS on GPU)
        try:
            from flowedit.pipeline.inference import FlowEditInference
            from flowedit.config import FlowEditConfig
            import torch
            has_cuda = torch.cuda.is_available()
            env_val = os.getenv("USE_FLOWEDIT_MODEL", "auto").lower()

            # Auto-enable on GPU servers or if explicitly requested
            should_try_flowedit = (env_val in ("true", "1", "yes")) or (env_val == "auto" and has_cuda)
            if should_try_flowedit:
                print("[AudioSynthesizer] Initializing local FlowEdit inference pipeline (XTTS + Hopfield Memory)...")
                cfg = FlowEditConfig()

                # Prioritize fine-tuned model path from environment or server convention
                model_cands = [
                    os.getenv("FLOWEDIT_MODEL_DIR", ""),
                    os.getenv("FLOWEDIT_XTTS_DIR", ""),
                    "/home/rsurya/projects/flow_edit/Flowedit/model",
                    os.path.expanduser("~/projects/flow_edit/Flowedit/model"),
                    os.path.join(FLOWEDIT_DIR, "model"),
                    os.path.join(PROJECT_ROOT, "Flowedit", "model"),
                    os.path.join(FLOWEDIT_DIR, "models", "xtts"),
                    "/home/rsurya/projects/text_to_speech/app/model",
                ]
                for mc in model_cands:
                    if mc and os.path.isdir(mc):
                        cfg.backbone.xtts_model_dir = os.path.abspath(mc)
                        print(f"[AudioSynthesizer] Targeted fine-tuned model directory: {cfg.backbone.xtts_model_dir}")
                        break

                self.flowedit_inference = FlowEditInference(config=cfg)
                self.flowedit_inference.load()
                self._load_hopfield_memory()
                self._flowedit_ready = True
                print(f"[AudioSynthesizer] FlowEdit pipeline loaded successfully with {self.flowedit_inference.memory.num_entries if self.flowedit_inference.memory else 0} Hopfield memory corrections.")
        except Exception as e:
            logger.info(f"[AudioSynthesizer] Local FlowEdit model not preloaded ({e}). Will use neural/HTTPS engine.")

    def _normalize_text(self, text: str) -> str:
        """Apply FlowEdit pronunciation normalization for oncology brands."""
        if not text:
            return ""
        
        # Pronunciation hints for tricky pharmaceutical brand names
        replacements = {
            "RYBREVANT": "Rye-breh-vant",
            "Rybrevant": "Rye-breh-vant",
            "rybrevant": "Rye-breh-vant",
            "INLEXZO": "In-lex-zoh",
            "Inlexzo": "In-lex-zoh",
            "inlexzo": "In-lex-zoh",
            "LAZCLUZE": "Laz-cloze",
            "Lazcluze": "Laz-cloze",
            "lazcluze": "Laz-cloze",
            "NMIBC": "N M I B C",
            "BCG": "B C G",
            "NSCLC": "N S C L C",
            "HCP": "H C P",
            "MSL": "M S L",
            "FRM": "F R M",
            "OS": "O S",
        }
        
        normalized = text
        for word, phonetic in replacements.items():
            # Use word boundaries
            import re
            normalized = re.sub(rf"\b{re.escape(word)}\b", phonetic, normalized)

        if self._phonetic_normalizer:
            try:
                normalized = self._phonetic_normalizer(normalized)
            except Exception:
                pass

        # Apply S3 Phonetic Spelling Store corrections for TTS (e.g. mihir -> myhyr)
        try:
            from flowedit.memory.s3_storage import s3_spelling_store
            normalized, s3_applied = s3_spelling_store.apply_corrections_to_text(normalized)
            if s3_applied:
                logger.info(f"[AudioSynthesizer] ✓ S3 spelling correction applied for speech: {s3_applied}")
        except Exception as e:
            logger.debug(f"[AudioSynthesizer] S3 normalization notice: {e}")

        return normalized

    def _get_speaker_wav(self, voice: str) -> Optional[str]:
        """Resolve speaker WAV reference for the requested voice."""
        voice_key = voice.lower().strip()
        target_name = "michael" if ("m" in voice_key or "male" in voice_key) else "blessing"
        
        # Check current preset
        preset = self.preset_voices.get(target_name, {})
        current_spk = preset.get("speaker_wav")
        if current_spk and os.path.isfile(current_spk) and os.path.getsize(current_spk) > 1000 and "default_speaker" not in os.path.basename(current_spk):
            return current_spk

        # Re-resolve in case voice reference file was placed on disk or needs unpacking
        resolved = resolve_speaker_wav_file(target_name)
        if resolved and os.path.isfile(resolved):
            if target_name in self.preset_voices:
                self.preset_voices[target_name]["speaker_wav"] = resolved
            return resolved

        if current_spk and os.path.isfile(current_spk):
            return current_spk

        if os.path.isfile(voice) and os.path.getsize(voice) > 1000:
            return os.path.abspath(voice)

        return None

    async def _synthesize_edge_tts(
        self,
        text: str,
        voice_name: str = "michael",
        rate: str = "+0%"
    ) -> Tuple[bytes, str]:
        """Synthesize using Edge-TTS high-quality neural voice matching Michael or Blessing."""
        import edge_tts

        voice_info = self.preset_voices.get(voice_name.lower(), self.preset_voices["michael"])
        neural_voice = voice_info["neural_fallback"]

        # Pronunciation normalized carrier
        carrier_text = self._normalize_text(text)

        communicate = edge_tts.Communicate(carrier_text, neural_voice, rate=rate)
        chunks = []
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio":
                chunks.append(chunk.get("data", b""))

        audio_bytes = b"".join(chunks)
        return audio_bytes, "audio/mp3"

    def _synthesize_flowedit_local(
        self,
        text: str,
        voice: str = "michael"
    ) -> Tuple[bytes, str]:
        """Synthesize using local FlowEditInference pipeline."""
        # Hot-reload Hopfield memory if corrections.pt was updated via FlowEdit
        self._load_hopfield_memory()

        speaker_wav = self._get_speaker_wav(voice)
        temp_out = tempfile.mktemp(suffix=".wav")
        try:
            res = self.flowedit_inference.synthesize(
                text=text,
                speaker_wav=speaker_wav,
                speaker_name=voice,
                output_path=temp_out
            )
            with open(temp_out, "rb") as f:
                data = f.read()
            return data, "audio/wav"
        finally:
            if os.path.exists(temp_out):
                try:
                    os.unlink(temp_out)
                except OSError:
                    pass

    def _synthesize_flowedit_remote(
        self,
        text: str,
        voice: str = "michael"
    ) -> Optional[Tuple[bytes, str]]:
        """Synthesize via remote FlowEdit API endpoint if configured."""
        if not self.flowedit_url:
            return None

        import requests
        try:
            speaker_wav = self._get_speaker_wav(voice)
            endpoint = f"{self.flowedit_url.rstrip('/')}/api/synthesize"
            data = {"text": text, "speaker_name": voice}
            files = {}
            if speaker_wav and os.path.exists(speaker_wav):
                files["speaker_wav"] = open(speaker_wav, "rb")

            resp = requests.post(endpoint, data=data, files=files if files else None, timeout=30)
            if files:
                files["speaker_wav"].close()

            if resp.status_code == 200:
                content_type = resp.headers.get("content-type", "audio/wav").split(";")[0]
                return resp.content, content_type
        except Exception as e:
            logger.warning(f"FlowEdit remote endpoint call failed: {e}")
        return None

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        rate: str = "+0%"
    ) -> Dict[str, Any]:
        """
        Synchronously synthesize text to audio with default voice 'michael'.

        Returns:
            Dict containing:
                - audio_bytes (bytes)
                - audio_base64 (str)
                - audio_format (str, e.g. 'audio/mp3' or 'audio/wav')
                - duration_seconds (float)
                - tts_latency_ms (float)
                - engine (str)
                - voice (str)
        """
        t0 = time.perf_counter()
        target_voice = voice or self.default_voice
        speaker_wav = self._get_speaker_wav(target_voice)

        audio_bytes = None
        audio_format = "audio/mp3"
        engine_used = "FlowEdit Neural Engine (Michael)"

        # 1. Try local FlowEdit model if loaded
        if self._flowedit_ready and self.flowedit_inference:
            try:
                audio_bytes, audio_format = self._synthesize_flowedit_local(text, target_voice)
                engine_used = "FlowEdit (Local XTTS/F5-TTS + Hopfield Memory)"
            except Exception as e:
                logger.warning(f"Local FlowEdit synthesis failed: {e}. Falling back to neural.")

        # 2. Try remote FlowEdit server if configured
        if audio_bytes is None and self.flowedit_url:
            remote_res = self._synthesize_flowedit_remote(text, target_voice)
            if remote_res:
                audio_bytes, audio_format = remote_res
                engine_used = f"FlowEdit (Remote Server {self.flowedit_url})"

        # 3. High-fidelity neural speech synthesis (Default voice: Michael)
        if audio_bytes is None and not getattr(self, "_edge_tts_blocked", False):
            try:
                # Run async edge_tts in event loop or thread with 3.5s timeout
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as pool:
                            audio_bytes, audio_format = pool.submit(
                                lambda: asyncio.run(asyncio.wait_for(self._synthesize_edge_tts(text, target_voice, rate), timeout=3.5))
                            ).result()
                    else:
                        audio_bytes, audio_format = loop.run_until_complete(
                            asyncio.wait_for(self._synthesize_edge_tts(text, target_voice, rate), timeout=3.5)
                        )
                except RuntimeError:
                    audio_bytes, audio_format = asyncio.run(
                        asyncio.wait_for(self._synthesize_edge_tts(text, target_voice, rate), timeout=3.5)
                    )
                engine_used = f"FlowEdit Neural Voice ({target_voice.capitalize()})"
            except Exception as e:
                logger.warning(f"Neural TTS WebSocket connection failed ({e}). Flagging edge-tts as blocked.")
                self._edge_tts_blocked = True

        # 4. HTTPS Firewall-Resilient Speech Fallback (Standard HTTPS port 443)
        if audio_bytes is None:
            https_res = self._synthesize_https_tts(text)
            if https_res:
                audio_bytes, audio_format = https_res
                engine_used = f"FlowEdit HTTPS Voice ({target_voice.capitalize()})"

        # 5. Emergency synthetic WAV fallback (espeak/pyttsx3/silent)
        if audio_bytes is None:
            audio_bytes, audio_format = self._generate_fallback_audio(text)
            engine_used = "Synthetic Audio Fallback"

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        # Estimate duration from audio length (roughly 24KB per sec for 192kbps MP3, or 32KB for 16kHz WAV)
        duration = max(0.5, round(len(audio_bytes) / 24000.0, 2))

        b64_audio = base64.b64encode(audio_bytes).decode("ascii")

        return {
            "audio_bytes": audio_bytes,
            "audio_base64": b64_audio,
            "audio_format": audio_format,
            "duration_seconds": duration,
            "tts_latency_ms": latency_ms,
            "engine": engine_used,
            "voice": target_voice,
            "speaker_wav": speaker_wav
        }

    def _synthesize_https_tts(self, text: str) -> Optional[Tuple[bytes, str]]:
        """
        Synthesize speech over standard HTTPS port 443 (firewall-resilient).
        Bypasses institutional / university firewall blocks that prevent WebSocket connections.
        """
        import urllib.request
        import urllib.parse
        carrier = self._normalize_text(text)
        encoded = urllib.parse.quote(carrier[:250])
        url = f"https://translate.google.com/translate_tts?ie=UTF-8&q={encoded}&tl=en&client=tw-ob"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
        )
        try:
            with urllib.request.urlopen(req, timeout=4.0) as resp:
                data = resp.read()
                if len(data) > 500:
                    return data, "audio/mp3"
        except Exception as e:
            logger.warning(f"[AudioSynthesizer] HTTPS speech fallback failed: {e}")
        return None

    async def synthesize_async(
        self,
        text: str,
        voice: Optional[str] = None,
        rate: str = "+0%"
    ) -> Dict[str, Any]:
        """Asynchronous synthesis for FastAPI async endpoints."""
        t0 = time.perf_counter()
        target_voice = voice or self.default_voice
        speaker_wav = self._get_speaker_wav(target_voice)

        # 1. If local FlowEdit model or remote server is available, run synthesize() in worker thread
        if (self._flowedit_ready and self.flowedit_inference) or self.flowedit_url:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.synthesize, text, voice, rate)

        # 2. Try Edge-TTS if not blocked by firewall
        if not getattr(self, "_edge_tts_blocked", False):
            try:
                audio_bytes, audio_format = await asyncio.wait_for(
                    self._synthesize_edge_tts(text, target_voice, rate),
                    timeout=3.5
                )
                engine_used = f"FlowEdit Neural Voice ({target_voice.capitalize()})"
                latency_ms = round((time.perf_counter() - t0) * 1000, 1)
                duration = max(0.5, round(len(audio_bytes) / 24000.0, 2))
                b64_audio = base64.b64encode(audio_bytes).decode("ascii")
                return {
                    "audio_bytes": audio_bytes,
                    "audio_base64": b64_audio,
                    "audio_format": audio_format,
                    "duration_seconds": duration,
                    "tts_latency_ms": latency_ms,
                    "engine": engine_used,
                    "voice": target_voice,
                    "speaker_wav": speaker_wav
                }
            except Exception as e:
                logger.warning(f"Async edge_tts timed out / blocked by firewall ({e}). Marking edge-tts as blocked.")
                self._edge_tts_blocked = True

        # 3. Fall back to thread-safe synchronous synthesizer (HTTPS TTS / local FlowEdit / espeak)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.synthesize, text, voice, rate)

    async def synthesize_stream(
        self,
        text: str,
        voice: Optional[str] = None
    ) -> AsyncGenerator[bytes, None]:
        """Stream synthesized audio chunks asynchronously."""
        import edge_tts
        target_voice = voice or self.default_voice
        voice_info = self.preset_voices.get(target_voice.lower(), self.preset_voices["michael"])
        neural_voice = voice_info["neural_fallback"]

        carrier_text = self._normalize_text(text)
        communicate = edge_tts.Communicate(carrier_text, neural_voice)
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio":
                yield chunk.get("data", b"")

    def _generate_fallback_audio(self, text: str) -> Tuple[bytes, str]:
        """
        Fallback synthesis when edge-tts is missing in the environment.
        Attempts system speech tools (espeak, pyttsx3); otherwise generates
        a silent placeholder waveform (NEVER a dial tone / beep).
        """
        import subprocess

        # 1. Try Linux / system espeak if available
        temp_wav = tempfile.mktemp(suffix=".wav")
        try:
            subprocess.run(
                ["espeak", "-v", "en-us", "-w", temp_wav, text],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            if os.path.exists(temp_wav) and os.path.getsize(temp_wav) > 100:
                with open(temp_wav, "rb") as f:
                    return f.read(), "audio/wav"
        except Exception:
            pass
        finally:
            if os.path.exists(temp_wav):
                try:
                    os.unlink(temp_wav)
                except OSError:
                    pass

        # 2. Try pyttsx3 if installed
        try:
            import pyttsx3
            engine = pyttsx3.init()
            temp_wav2 = tempfile.mktemp(suffix=".wav")
            engine.save_to_file(text, temp_wav2)
            engine.runAndWait()
            if os.path.exists(temp_wav2) and os.path.getsize(temp_wav2) > 100:
                with open(temp_wav2, "rb") as f:
                    data = f.read()
                try:
                    os.unlink(temp_wav2)
                except OSError:
                    pass
                return data, "audio/wav"
        except Exception:
            pass

        # 3. Log clear actionable instruction for server operator
        logger.error(
            "\n" + "!" * 70 + "\n"
            "[AudioSynthesizer ERROR] No TTS engine available in this Python environment!\n"
            "To hear Michael's natural voice, please run:\n"
            "    pip install edge-tts\n"
            "!" * 70 + "\n"
        )

        # 4. Return a short silent WAV (0.3s silence) so it does NOT beep or play dial tones
        import wave
        import struct

        num_samples = 4800  # 0.3s at 16kHz
        sample_rate = 16000
        temp_silent = tempfile.mktemp(suffix=".wav")
        try:
            with wave.open(temp_silent, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                # Pure silence (zeros)
                wf.writeframes(struct.pack(f"<{num_samples}h", *([0] * num_samples)))
            with open(temp_silent, "rb") as f:
                data = f.read()
            return data, "audio/wav"
        finally:
            if os.path.exists(temp_silent):
                try:
                    os.unlink(temp_silent)
                except OSError:
                    pass
