import asyncio
import queue
import sounddevice as sd
import numpy as np
import time
import aiohttp
import json
import threading
import os
import re
import logging # Ajout pour une meilleure journalisation
import soundfile as sf # Utilisé pour sauvegarder l'audio TTS si besoin
import functools
from typing import Any, Generator, List, Optional, Tuple, Union, Dict
# import gc # Décommentez si vous voulez tester le garbage collection manuel

# --- Imports spécifiques aux services ---
# STT (Faster Whisper)
from faster_whisper import WhisperModel

# TTS (MLX Audio Kokoro Pipeline)
# Assurez-vous que mlx_audio est installé et que les dépendances nécessaires sont présentes
# pip install mlx-audio soundfile
# pip install --upgrade mlx-audio # Pour obtenir les derniers correctifs (comme PR #66)
try:
    # Supposant que l'utilisateur utilise la structure de mlx-audio
    # Ajustez ces imports si votre structure locale est différente
    from mlx_audio.tts.models.kokoro import KokoroPipeline, Model as KokoroModel # ModelConfig n'est pas utilisé directement ici
    from mlx_audio.tts.utils import load_model # Nécessaire pour charger le modèle Kokoro
    MLX_AUDIO_AVAILABLE = True
except ImportError as e:
    MLX_AUDIO_AVAILABLE = False
    print(f"WARN: mlx-audio not found or failed to import ({e}). TTS functionality will be disabled.")
    # Define dummy classes/functions if needed for the script to run without TTS
    class KokoroPipeline: pass
    class KokoroModel: pass
    # class ModelConfig: pass # Pas directement utilisé
    def load_model(*args, **kwargs): pass


# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s - %(name)s - %(message)s')
# Pour plus de détails pendant le debug, changez INFO en DEBUG
# logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# STT Configuration (Faster Whisper)
WHISPER_MODEL_SIZE = "small" # Options: "tiny", "base", "small", "medium", "large-v3". 'small' est un bon compromise.
FASTER_WHISPER_DEVICE = "cpu" # 'cpu' est souvent stable et performant avec la quantization. 'cuda' if available.
FASTER_WHISPER_COMPUTE_TYPE = "int8" # 'int8' pour la vitesse et faible usage RAM. 'float16' si plus de précision est nécessaire et supportée.
STT_LANGUAGE = 'fr' # Forcer le français pour la transcription

# LLM Configuration
LLM_API_ENDPOINT = "http://localhost:1234/v1/chat/completions"
LLM_API_KEY = None # Mettre votre clé API si nécessaire (souvent 'Not needed' pour LM Studio)
LLM_SYSTEM_PROMPT = "Vous êtes un assistant conversationnel utile et concis, répondant en français."
LLM_MODEL_NAME = "local-model" # Nom du modèle utilisé par votre API LLM (ex: "LM Studio Community/Meta-Llama-3-8B-Instruct-GGUF")

# TTS Configuration (MLX Audio Kokoro)
# Utilisation du modèle 4-bit quantizé pour la vitesse
KOKORO_MODEL_ID = 'mlx-community/Kokoro-82M-4bit'
KOKORO_VOICE = 'ff_siwis' # Voix FR (SIWIS female) - Vérifiez si cette voix existe pour le modèle/langue
KOKORO_LANG_CODE = 'f' # Code langue Kokoro pour Français ('f')
TTS_SPEECH_SPEED = 1.2 # Vitesse de parole (ex: 1.0 = normal, 1.2 = plus rapide)
TTS_OUTPUT_FILENAME_PREFIX = "response_audio" # Préfixe pour les fichiers audio générés (si sauvegarde activée)
SAVE_TTS_AUDIO = False # Mettre à True pour sauvegarder chaque réponse TTS
# Découpage du texte pour TTS (utilisé par KokoroPipeline)
TTS_SPLIT_PATTERN = r'[.!?\n]+' # Découpe par phrase/retour ligne, peut être ajusté

# Audio Input/Output Configuration
SAMPLE_RATE_INPUT = 16000 # Whisper préfère 16kHz
SAMPLE_RATE_OUTPUT = 24000 # Kokoro utilise 24kHz
CHANNELS_INPUT = 1
CHANNELS_OUTPUT = 1
CHUNK_DURATION_MS = 200 # Durée des chunks audio traités (ms)
CHUNK_SAMPLES_INPUT = int(SAMPLE_RATE_INPUT * CHUNK_DURATION_MS / 1000)
INPUT_DEVICE = None # Laisse sounddevice choisir le défaut
OUTPUT_DEVICE = None # Laisse sounddevice choisir le défaut

# Buffering STT Configuration (Ajuster pour latence vs complétude)
BUFFER_DURATION_S = 2.5 # Durée max du buffer avant transcription forcée (Réduit de 3.0s)
BUFFER_MAX_SAMPLES = int(BUFFER_DURATION_S * SAMPLE_RATE_INPUT)
SILENCE_THRESHOLD = 0.01 # Seuil pour détecter le silence (à ajuster selon micro/environnement)
SILENCE_CHUNKS_NEEDED = 4 # Nb de chunks silencieux consécutifs pour déclencher la transcription (Réduit de 5)

# Queue Configuration
AUDIO_INPUT_QUEUE_MAXSIZE = 50
STT_OUTPUT_QUEUE_MAXSIZE = 10 # Queue pour les transcriptions texte
LLM_OUTPUT_QUEUE_MAXSIZE = 10 # Queue pour les réponses texte du LLM
SPEAK_QUEUE_MAXSIZE = 10      # Queue pour les textes à envoyer au TTS
TTS_FINAL_AUDIO_QUEUE_MAXSIZE = 100 # Queue pour les *numpy arrays* audio finaux générés par TTS

# --- Queues ---
audio_input_queue = asyncio.Queue(maxsize=AUDIO_INPUT_QUEUE_MAXSIZE)
stt_output_queue = asyncio.Queue(maxsize=STT_OUTPUT_QUEUE_MAXSIZE)
llm_output_queue = asyncio.Queue(maxsize=LLM_OUTPUT_QUEUE_MAXSIZE)
speak_queue = asyncio.Queue(maxsize=SPEAK_QUEUE_MAXSIZE)
tts_final_audio_queue = asyncio.Queue(maxsize=TTS_FINAL_AUDIO_QUEUE_MAXSIZE)

# --- Événements pour la synchronisation et l'interruption ---
playback_active = asyncio.Event() # True si le playback est en cours
interrupt_playback_event = asyncio.Event() # Mis à True par STT si nouvelle parole pendant playback


# --- Services ---

class SpeechToTextService:
    """Module STT utilisant faster-whisper. Contient la logique de transcription."""
    def __init__(self, model_size: str, device: str, compute_type: str):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        logger.info(f"[STT Service] Initializing Faster Whisper model: {model_size} (Device: {device}, Compute: {compute_type})...")
        try:
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
            logger.info("[STT Service] Faster Whisper model initialized.")
        except Exception as e:
            logger.exception(f"[STT Service] Failed to load Faster Whisper model: {e}")
            raise
        self.audio_buffer = np.array([], dtype=np.float32)
        self.silent_chunks_count = 0

    async def _transcribe_specific_buffer(self, buffer_data: np.ndarray):
        """Appelle faster-whisper pour transcrire le buffer audio donné et met le résultat dans stt_output_queue."""
        if len(buffer_data) < SAMPLE_RATE_INPUT * 0.3:
            logger.debug("[STT Service Task] Buffer too short, ignoring.")
            return
        if np.max(np.abs(buffer_data)) < SILENCE_THRESHOLD * 1.2:
            logger.debug("[STT Service Task] Buffer below silence threshold, ignoring.")
            return

        logger.info(f"[STT Service Task] Transcribing buffer of {len(buffer_data) / SAMPLE_RATE_INPUT:.2f}s...")
        loop = asyncio.get_running_loop()
        transcription_result = None
        try:
            segments, info = await loop.run_in_executor(
                None,
                lambda: self.model.transcribe(buffer_data, beam_size=5, language=STT_LANGUAGE, vad_filter=True, vad_parameters={"min_silence_duration_ms": 500})
            )
            transcription_result = "".join(segment.text for segment in segments).strip()

            if transcription_result:
                logger.info(f"[STT Service Task] Transcription: '{transcription_result}' (Lang: {info.language}, Prob: {info.language_probability:.2f})")
                # Mettre la transcription dans la queue de sortie STT
                await stt_output_queue.put(transcription_result)
            else:
                logger.info("[STT Service Task] Ignored empty transcription.")
                # Mettre une chaîne vide ou None pour signaler la fin du traitement ?
                # Pour l'instant, on ne met rien si vide.
                # await stt_output_queue.put("") # Optionnel

        except Exception as e:
            logger.exception(f"[STT Service Task] Error during transcription: {e}")
            # En cas d'erreur, on pourrait mettre None ou une indication d'erreur
            # await stt_output_queue.put(None) # Optionnel

    async def process_audio_chunk(self, audio_chunk: np.ndarray):
        """Ajoute un chunk au buffer STT et déclenche la transcription si nécessaire."""
        if audio_chunk.ndim > 1 and audio_chunk.shape[1] > 1:
            audio_chunk_flat = audio_chunk[:, 0]
        else:
            audio_chunk_flat = audio_chunk.flatten()

        is_silent_chunk = np.max(np.abs(audio_chunk_flat)) < SILENCE_THRESHOLD

        if not is_silent_chunk:
            self.silent_chunks_count = 0
            self.audio_buffer = np.concatenate((self.audio_buffer, audio_chunk_flat))
        else:
            self.silent_chunks_count += 1
            if len(self.audio_buffer) > 0:
                self.audio_buffer = np.concatenate((self.audio_buffer, audio_chunk_flat))

        buffer_length_seconds = len(self.audio_buffer) / SAMPLE_RATE_INPUT
        should_transcribe = False

        if buffer_length_seconds >= BUFFER_DURATION_S:
            if np.max(np.abs(self.audio_buffer)) >= SILENCE_THRESHOLD:
                logger.info(f"[STT Service] Buffer duration reached ({buffer_length_seconds:.2f}s), triggering transcription.")
                should_transcribe = True
            else:
                logger.debug(f"[STT Service] Buffer duration reached but seems silent, resetting buffer.")
                self.audio_buffer = np.array([], dtype=np.float32)
                self.silent_chunks_count = 0

        elif self.silent_chunks_count >= SILENCE_CHUNKS_NEEDED and buffer_length_seconds > 0.5:
            logger.info(f"[STT Service] Silence detected after {buffer_length_seconds:.2f}s, triggering transcription.")
            should_transcribe = True

        if should_transcribe:
            buffer_to_transcribe = self.audio_buffer.copy()
            self.audio_buffer = np.array([], dtype=np.float32)
            self.silent_chunks_count = 0
            # Lance la transcription en tâche de fond. Le résultat ira dans stt_output_queue.
            asyncio.create_task(self._transcribe_specific_buffer(buffer_to_transcribe))


class LLMService:
    """Module LLM - Gère les appels API en streaming SSE."""
    def __init__(self, api_endpoint: str, model_name: str, system_prompt: str, api_key: str = None):
        self.api_endpoint = api_endpoint
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.session = None
        logger.info(f"[LLM Service] Initialized for endpoint: {api_endpoint}")

    async def start_session(self):
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=None, connect=20, sock_connect=20, sock_read=None)
            self.session = aiohttp.ClientSession(headers=self.headers, timeout=timeout)
            logger.info("[LLM Service] aiohttp session started.")

    async def close_session(self):
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("[LLM Service] aiohttp session closed.")
            self.session = None

    async def get_response_stream(self, text: str):
        """Appelle l'API LLM en mode streaming SSE et place la réponse complète dans llm_output_queue."""
        await self.start_session()
        if not text or not text.strip():
            logger.warning("[LLM Service] Received empty text for LLM, ignoring.")
            return

        logger.info(f"[LLM Service] Sending text to LLM API: '{text}'")
        payload = {
            "model": self.model_name,
            "messages": [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": text}],
            "temperature": 0.7, "stream": True,
        }
        full_response_content = ""
        stream_finished_sentinel = "[DONE]"

        try:
            async with self.session.post(self.api_endpoint, json=payload) as response:
                response.raise_for_status()
                start_time = time.time()
                first_token_received = False
                async for line_bytes in response.content:
                    if not line_bytes:
                        continue
                    line = line_bytes.decode('utf-8', errors='ignore').strip()
                    if not first_token_received and line.startswith("data:"):
                        logger.info(f"[LLM Service] Time to first token: {time.time() - start_time:.2f}s")
                        first_token_received = True
                    if line.startswith("data:"):
                        data_json = line[len("data:"):].strip()
                        if data_json == stream_finished_sentinel:
                            break
                        try:
                            if data_json:
                                chunk_data = json.loads(data_json)
                                delta = chunk_data.get("choices", [{}])[0].get("delta", {})
                                content_piece = delta.get("content")
                                if content_piece:
                                    full_response_content += content_piece
                                finish_reason = chunk_data.get("choices", [{}])[0].get("finish_reason")
                                if finish_reason:
                                    logger.info(f"[LLM Service] Stream finished with reason: {finish_reason}")
                        except (json.JSONDecodeError, IndexError) as json_e:
                            logger.warning(f"[LLM Service] Error processing stream chunk: {json_e} - Line: '{line}'")
                        except Exception as e:
                            logger.exception(f"[LLM Service] Unexpected error processing stream chunk: {e} - JSON: {data_json}")
                    elif line.startswith("error:"):
                        logger.error(f"[LLM Service] Received error line from server: {line}")

            if full_response_content:
                logger.info(f"[LLM Service] Full response received ({len(full_response_content)} chars). Putting to queue.")
                await llm_output_queue.put(full_response_content)
            else:
                logger.warning("[LLM Service] Stream ended without generating content.")

        except aiohttp.ClientResponseError as e:
            logger.error(f"[LLM Service] HTTP Error: {e.status} {e.message}")
        except aiohttp.ClientConnectionError as e:
            logger.error(f"[LLM Service] Connection Error: {e}")
        except asyncio.TimeoutError:
            logger.error(f"[LLM Service] Request timed out.")
        except Exception as e:
            logger.exception(f"[LLM Service] Unexpected error during LLM stream: {e}")


class KokoroTTSService:
    """Module TTS utilisant KokoroPipeline avec initialisation async."""
    def __init__(self, model_id: str, lang_code: str):
        self.model_id = model_id
        self.lang_code = lang_code
        self.kokoro_model: Optional[KokoroModel] = None
        self.pipeline: Optional[KokoroPipeline] = None
        if not MLX_AUDIO_AVAILABLE:
            logger.error("[Kokoro TTS] mlx-audio not available. TTS disabled.")
        else:
            logger.info(f"[Kokoro TTS] Service created (Model ID: {model_id}, Lang: {lang_code}). Call initialize().")

    async def initialize(self):
        """Charge le modèle Kokoro et initialise la KokoroPipeline."""
        if not MLX_AUDIO_AVAILABLE or self.pipeline:
            return
        logger.info(f"[Kokoro TTS] Async Initializing - Loading model '{self.model_id}'...")
        loop = asyncio.get_running_loop()
        loaded_model_obj = None
        try:
            try:
                logger.debug(f"[Kokoro TTS Debug] Calling load_model with id: {self.model_id}...")
                loaded_model_obj = await loop.run_in_executor(None, load_model, self.model_id)
                logger.info(f"[Kokoro TTS] Model '{self.model_id}' loaded successfully.")
                if loaded_model_obj is None:
                    raise RuntimeError(f"load_model returned None.")
                self.kokoro_model = loaded_model_obj
            except Exception as load_error:
                logger.error(f"[Kokoro TTS] !!! Exception caught DURING load_model execution !!!")
                logger.exception(f"[Kokoro TTS] The actual error from load_model is: {load_error}")
                raise RuntimeError(f"Failed to load model '{self.model_id}'.") from load_error

            pipeline_init_partial = functools.partial(
                KokoroPipeline, lang_code=self.lang_code, model=self.kokoro_model, repo_id=self.model_id
            )
            logger.debug("[Kokoro TTS Debug] Initializing KokoroPipeline...")
            self.pipeline = await loop.run_in_executor(None, pipeline_init_partial)
            # --- Ajout du log de warning Espeak ---
            # Ce warning est normalement loggué par KokoroPipeline.__init__ lui-même
            # mais on peut le dupliquer ici si besoin.
            if self.lang_code not in "abjz": # Si ce n'est pas anglais, japonais ou chinois
                logger.warning(f"[Kokoro TTS] Using EspeakG2P for language '{self.lang_code}'. Ensure espeak-ng is installed.")
            # --- Fin Ajout ---
            logger.info("[Kokoro TTS] Pipeline initialized.")
        except Exception as e:
            logger.exception(f"[Kokoro TTS] Error during async initialization: {e}")
            raise

    async def synthesize_speech_stream(self, text: str, voice: str, speed: float = 1.0):
        """Génère les chunks audio via Kokoro et les place dans tts_final_audio_queue."""
        if not self.pipeline:
            logger.error("[Kokoro TTS] Pipeline not initialized. Cannot synthesize.")
            await tts_final_audio_queue.put(None)
            return

        logger.info(f"[Kokoro TTS] Starting synthesis (Voice: '{voice}', Speed: {speed}): '{text[:60]}...'")
        loop = asyncio.get_running_loop()
        processed_chunks_count = 0
        start_time = time.time()
        all_audio_data_for_saving = []

        try:
            def generate_sync():
                return self.pipeline(text, voice=voice, speed=speed, split_pattern=TTS_SPLIT_PATTERN)

            result_generator = await loop.run_in_executor(None, generate_sync)

            while True:
                try:
                    next_result = await loop.run_in_executor(None, lambda: next(result_generator, StopIteration))
                    if next_result is StopIteration:
                        logger.info(f"[Kokoro TTS] End of pipeline generator reached after {processed_chunks_count} chunk(s).")
                        break

                    processed_chunks_count += 1
                    logger.info(f"[Kokoro TTS] Processing pipeline chunk {processed_chunks_count} (Text: '{next_result.graphemes[:30]}...')...")
                    audio_data = next_result.audio
                    if audio_data is None:
                        logger.warning(f"[Kokoro TTS] Chunk {processed_chunks_count} returned None audio.")
                        continue

                    try:
                        # Correction: Utilisation de lambda pour passer copy=True à np.array
                        audio_np = await loop.run_in_executor(None, lambda: np.array(audio_data, copy=True))
                        audio_np = np.squeeze(audio_np)
                    except Exception as convert_e:
                        logger.exception(f"[Kokoro TTS] Error converting audio for chunk {processed_chunks_count}: {convert_e}")
                        continue

                    if audio_np.ndim != 1:
                        logger.warning(f"[Kokoro TTS] Unexpected audio dimension {audio_np.ndim} for chunk {processed_chunks_count}.")
                        continue

                    await tts_final_audio_queue.put(audio_np)
                    if SAVE_TTS_AUDIO:
                        all_audio_data_for_saving.append(audio_np)

                except StopIteration:
                    break
                except Exception as e:
                    logger.exception(f"[Kokoro TTS] Error processing chunk {processed_chunks_count}: {e}")
                    break

            synthesis_duration = time.time() - start_time
            logger.info(f"[Kokoro TTS] Finished processing generator. Total Chunks: {processed_chunks_count}, Duration: {synthesis_duration:.2f}s")

            if SAVE_TTS_AUDIO and all_audio_data_for_saving:
                try:
                    full_audio = np.concatenate(all_audio_data_for_saving)
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"{TTS_OUTPUT_FILENAME_PREFIX}_{timestamp}.wav"
                    logger.info(f"[Kokoro TTS] Saving audio ({len(full_audio)/SAMPLE_RATE_OUTPUT:.2f}s) to {filename}...")
                    await loop.run_in_executor(None, sf.write, filename, full_audio, SAMPLE_RATE_OUTPUT)
                    logger.info(f"[Kokoro TTS] Saved audio to {filename}")
                except sf.LibsndfileError as sf_err:
                    logger.error(f"[Kokoro TTS] Soundfile save error: {sf_err}")
                except Exception as e_save:
                    logger.exception(f"[Kokoro TTS] Failed to save audio: {e_save}")

        except Exception as e_synth:
            logger.exception(f"[Kokoro TTS] Error during synthesis stream: {e_synth}")
        finally:
            logger.debug("[Kokoro TTS] Sending end signal (None) to final audio queue.")
            await tts_final_audio_queue.put(None)


# --- Audio Input Handling ---
sd_input_queue = queue.Queue()
def input_callback(indata, frames, time_info, status):
    if status:
        logger.warning(f"[Audio Input Callback] Status: {status}")
    try:
        sd_input_queue.put_nowait(indata.copy())
    except queue.Full:
        logger.warning("[Audio Input Callback] Input queue full. Dropping chunk.")


# --- Tâches Async ---

async def process_audio_input_to_stt():
    """Lit depuis sd_input_queue et met dans audio_input_queue."""
    logger.info("[Input Processor] Started.")
    loop = asyncio.get_running_loop()
    while True:
        try:
            indata = await loop.run_in_executor(None, sd_input_queue.get, True, 0.5)
            await audio_input_queue.put(indata)
        except queue.Empty:
            await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            logger.info("[Input Processor] Cancelled.")
            break
        except Exception as e:
            logger.exception(f"[Input Processor] Error: {e}")
            await asyncio.sleep(0.1)

async def run_stt_tasks(stt_service: SpeechToTextService):
    """Traite audio_input_queue, lance STT, et gère l'interruption."""
    logger.info("[STT Task Runner] Started.")
    while True:
        processed_transcription = False # Flag pour savoir si on a traité une transcription dans cette itération
        try:
            # Traiter d'abord l'audio entrant
            audio_chunk = await audio_input_queue.get()
            await stt_service.process_audio_chunk(audio_chunk) # Lance _transcribe_specific_buffer en tâche de fond
            audio_input_queue.task_done()

            # Vérifier si une transcription est prête DANS CETTE BOUCLE
            # Utiliser queue.get_nowait() pour ne pas bloquer si rien n'est prêt
            try:
                transcription = stt_output_queue.get_nowait()
                processed_transcription = True # On a eu une transcription
                if transcription:
                    logger.info(f"[STT Task Runner] Received transcription: '{transcription}'")
                    # --- Logique d'interruption ---
                    if playback_active.is_set():
                        logger.warning("[STT Task Runner] Playback active. Signaling interruption!")
                        interrupt_playback_event.set()
                        await asyncio.sleep(0.05) # Laisser le temps à playback_task de réagir
                    # --- Fin Logique d'interruption ---
                    await llm_output_queue.put(transcription) # Envoyer au LLM
                else:
                    # Gérer le cas d'une transcription vide si _transcribe_specific_buffer en mettait
                    logger.debug("[STT Task Runner] Received empty/None transcription.")

                stt_output_queue.task_done() # Marquer comme traité

            except asyncio.QueueEmpty:
                # Pas de transcription prête, c'est normal, on continue
                pass

        except asyncio.CancelledError:
            logger.info("[STT Task Runner] Cancelled."); break
        except Exception as e:
            logger.exception(f"[STT Task Runner] Error: {e}")
            # Nettoyer les queues en cas d'erreur ?
            if not audio_input_queue.empty():
                try:
                    audio_input_queue.task_done()
                except ValueError:
                    pass
            if processed_transcription: # Si on a sorti une transcription avant l'erreur
                if not stt_output_queue.empty():
                    try:
                        stt_output_queue.task_done()
                    except ValueError:
                        pass
            await asyncio.sleep(0.1)


async def run_llm_tasks(llm_service: LLMService):
    """Prend texte de stt_output_queue, appelle LLM, met résultat dans llm_output_queue."""
    logger.info("[LLM Task Runner] Started.")
    while True:
        try:
            # Note: La logique d'interruption est gérée dans run_stt_tasks AVANT de mettre ici
            # Correction: Lire de stt_output_queue, mettre dans llm_output_queue est fait par get_response_stream
            # Donc, cette tâche lit de stt_output_queue et LANCE get_response_stream

            # *** CORRECTION MAJEURE DE LOGIQUE ICI ***
            # Cette tâche doit lire stt_output_queue, pas llm_output_queue
            transcribed_text = await stt_output_queue.get() # LIRE ICI

            if transcribed_text:
                # Lancer la requête LLM en tâche de fond
                # Le résultat sera mis dans llm_output_queue par get_response_stream
                asyncio.create_task(llm_service.get_response_stream(transcribed_text))
            else:
                logger.info("[LLM Task Runner] Received empty transcription, skipping LLM call.")

            stt_output_queue.task_done() # Marquer la tâche STT comme faite

        except asyncio.CancelledError:
            logger.info("[LLM Task Runner] Cancelled. Closing LLM session.")
            await llm_service.close_session()
            break
        except Exception as e:
            logger.exception(f"[LLM Task Runner] Error: {e}")
            if not stt_output_queue.empty():
                try:
                    stt_output_queue.task_done()
                except ValueError:
                    pass
            await asyncio.sleep(0.1)

async def run_tts_tasks():
    """Prend réponse de llm_output_queue, met dans speak_queue."""
    logger.info("[TTS Task Runner] Started.")
    while True:
        try:
            llm_response = await llm_output_queue.get()
            if llm_response:
                logger.info(f"[TTS Task Runner] Received LLM response, putting to speak queue: '{llm_response[:50]}...'")
                await speak_queue.put(llm_response)
            else:
                logger.info("[TTS Task Runner] Received empty LLM response, skipping TTS.")
            llm_output_queue.task_done()
        except asyncio.CancelledError:
            logger.info("[TTS Task Runner] Cancelled.")
            break
        except Exception as e:
            logger.exception(f"[TTS Task Runner] Error: {e}")
            if not llm_output_queue.empty():
                try:
                    llm_output_queue.task_done()
                except ValueError:
                    pass
            await asyncio.sleep(0.1)


# --- Playback Task (Avec Interruption) ---

async def playback_task(kokoro_service: KokoroTTSService, kokoro_voice: str):
    """Lance la synthèse TTS et gère la lecture, AVEC INTERRUPTION."""
    if not MLX_AUDIO_AVAILABLE:
        logger.warning("[Playback Task] Cannot start - mlx-audio not available.")
        return

    logger.info(f"[Playback Task] Started (Voice: {kokoro_voice}). Waiting for text...")
    loop = asyncio.get_running_loop()

    callback_data = {
        "buffer": np.array([], dtype=np.float32), "lock": threading.Lock(),
        "stream_finished_event": asyncio.Event(), "playback_complete_event": asyncio.Event(),
        "generation_done": False, "loop": loop,
        "output_stream": None, "interrupted": False
    }

    def audio_callback(outdata: np.ndarray, frames: int, time_info, status: sd.CallbackFlags):
        # (Callback inchangé par rapport à la version précédente avec interruption)
        if status:
            logger.warning(f"[Audio Callback] Output Status: {status}")
        try:
            if callback_data["interrupted"]:
                outdata.fill(0.0)
                return
            with callback_data["lock"]:
                needed = len(outdata)
                available = len(callback_data["buffer"])
                chunk_to_play = callback_data["buffer"][: min(needed, available)]
                callback_data["buffer"] = callback_data["buffer"][min(needed, available) :]
                outdata[:available, 0] = chunk_to_play
                outdata[available:, 0] = 0.0
                if not callback_data["interrupted"] and callback_data["generation_done"] and len(callback_data["buffer"]) == 0:
                    callback_data["loop"].call_soon_threadsafe(callback_data["playback_complete_event"].set)
        except Exception as e:
            logger.exception(f"[Audio Callback] Error: {e}")
            outdata.fill(0.0)

    # --- Boucle Principale du Playback Task ---
    while True:
        synth_task = None
        current_speak_item_processed = False # Flag pour savoir si task_done a été appelé

        try:
            # 0. Vérifier interruption avant d'attendre
            if interrupt_playback_event.is_set():
                logger.info("[Playback Task] Interruption signal detected before getting new text.")
                interrupt_playback_event.clear()
                continue

            # 1. Attendre un texte
            logger.debug("[Playback Task] Waiting for text on speak_queue...")
            text_to_speak = await speak_queue.get()
            # --- Appel task_done() TÔT ---
            speak_queue.task_done()
            current_speak_item_processed = True # Marquer comme traité pour éviter double appel dans finally
            # --- Fin Appel task_done() TÔT ---


            if text_to_speak is None:
                logger.info("[Playback Task] Stop signal received.")
                break
            if not text_to_speak.strip():
                logger.info("[Playback Task] Received empty text, skipping.")
                continue

            logger.info(f"[Playback Task] Received text: '{text_to_speak[:50]}...'. Starting...")
            playback_active.set()
            callback_data["interrupted"] = False
            callback_data["stream_finished_event"].clear()
            callback_data["playback_complete_event"].clear()
            callback_data["generation_done"] = False
            with callback_data["lock"]:
                callback_data["buffer"] = np.array([], dtype=np.float32)
            while not tts_final_audio_queue.empty():
                try:
                    tts_final_audio_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

            # 3. Démarrer Synthèse TTS
            synth_task = asyncio.create_task(
                kokoro_service.synthesize_speech_stream(text_to_speak, kokoro_voice, speed=TTS_SPEECH_SPEED),
                name=f"SynthTask_{time.time():.0f}"
            )

            # 4. Configurer & Démarrer Stream Audio
            stream_context = None # Pour gérer le 'with' manuellement en cas d'interruption
            try:
                callback_data["output_stream"] = sd.OutputStream(
                    samplerate=SAMPLE_RATE_OUTPUT, channels=CHANNELS_OUTPUT, dtype='float32',
                    device=OUTPUT_DEVICE, callback=audio_callback,
                    finished_callback=callback_data["stream_finished_event"].set
                )
                stream_context = callback_data["output_stream"] # Référence pour le finally
                stream_context.start() # Démarrer manuellement
                logger.info("[Playback Task] Audio stream started. Consuming final audio chunks...")

                # 5. Boucle Consommation Chunks Audio
                while True:
                    # 5a. Vérifier Interruption
                    if interrupt_playback_event.is_set():
                        logger.warning("[Playback Task] Interruption signal detected during chunk consumption!")
                        interrupt_playback_event.clear()
                        callback_data["interrupted"] = True

                        if synth_task and not synth_task.done():
                            logger.info("[Playback Task] Cancelling synthesis task...")
                            synth_task.cancel()

                        logger.info("[Playback Task] Clearing remaining TTS audio chunks...")
                        while not tts_final_audio_queue.empty():
                            try:
                                tts_final_audio_queue.get_nowait()
                            except asyncio.QueueEmpty:
                                break

                        logger.info("[Playback Task] Stopping audio stream due to interruption...")
                        # Pas besoin de stream_context.stop() ici, le finally du bloc externe s'en chargera

                        with callback_data["lock"]:
                            callback_data["buffer"] = np.array([], dtype=np.float32)

                        logger.info("[Playback Task] Interruption handled. Breaking inner loop.")
                        break # Sortir boucle consommation

                    # 5b. Attendre prochain chunk
                    try:
                        audio_chunk_np = await asyncio.wait_for(tts_final_audio_queue.get(), timeout=0.1)
                    except asyncio.TimeoutError:
                        if callback_data["generation_done"]:
                            break # Fin normale + timeout
                        elif callback_data["interrupted"]:
                            break # Interrompu + timeout
                        else:
                            continue # Timeout normal, on reboucle

                    # 5c. Traiter chunk
                    if audio_chunk_np is None: # Fin normale TTS
                        logger.info("[Playback Task] End of synthesis stream (None). Generation done.")
                        with callback_data["lock"]:
                            callback_data["generation_done"] = True
                        with callback_data["lock"]:
                            if len(callback_data["buffer"]) == 0:
                                logger.debug("[Playback Task] Gen done & buffer empty.")
                                callback_data["loop"].call_soon_threadsafe(callback_data["playback_complete_event"].set)
                        break # Sortir boucle consommation
                    else: # Chunk audio normal
                        with callback_data["lock"]:
                            callback_data["buffer"] = np.concatenate((callback_data["buffer"], audio_chunk_np))
                # --- Fin Boucle Consommation Chunks ---

                # 6. Attendre Fin Playback (si pas interrompu)
                if not callback_data["interrupted"]:
                    logger.info("[Playback Task] Waiting for actual playback completion...")
                    await callback_data["playback_complete_event"].wait()
                    logger.info("[Playback Task] Playback definitively complete.")
                    # task_done a déjà été appelé au début

            # Gestion erreurs stream/portaudio
            except sd.PortAudioError as pae:
                logger.exception(f"[Playback Task] PortAudioError: {pae}")
                # task_done déjà appelé
                await asyncio.sleep(0.5)
            except Exception as e_stream:
                logger.exception(f"[Playback Task] Error during audio stream context: {e_stream}")
                # task_done déjà appelé
                await asyncio.sleep(0.5)
            finally:
                # Assurer l'arrêt et la fermeture du stream audio
                if stream_context:
                    logger.debug("[Playback Task] Ensuring audio stream is stopped and closed.")
                    try:
                        if stream_context.active:
                            stream_context.stop()
                        stream_context.close()
                    except sd.PortAudioError as pae_stop:
                        logger.error(f"Error stopping/closing stream in finally: {pae_stop}")
                    except Exception as e_finally:
                        logger.error(f"Generic error stopping/closing stream in finally: {e_finally}")
                callback_data["output_stream"] = None # Nettoyer la référence


        # Gestion erreurs boucle externe / Annulation
        except asyncio.CancelledError:
            logger.info("[Playback Task] Cancelled.")
            if synth_task and not synth_task.done():
                synth_task.cancel()
            break
        except Exception as e:
            logger.exception(f"[Playback Task] Error in outer playback loop: {e}")
            if synth_task and not synth_task.done():
                synth_task.cancel()
            # task_done a normalement déjà été appelé au début
            await asyncio.sleep(1)

        finally:
            logger.debug("[Playback Task] Outer loop cleanup.")
            playback_active.clear()
            interrupt_playback_event.clear()
            callback_data["interrupted"] = False
            # gc.collect() # Optionnel

    logger.info("[Playback Task] Finished.")


# --- Main Execution ---
async def main():
    logger.info("[Main] Initializing services...")
    stt_service = None
    llm_service = None
    kokoro_tts_service = None
    input_stream = None
    tasks = []

    try:
        stt_service = SpeechToTextService(WHISPER_MODEL_SIZE, FASTER_WHISPER_DEVICE, FASTER_WHISPER_COMPUTE_TYPE)
        llm_service = LLMService(LLM_API_ENDPOINT, LLM_MODEL_NAME, LLM_SYSTEM_PROMPT, LLM_API_KEY)
        if MLX_AUDIO_AVAILABLE:
            kokoro_tts_service = KokoroTTSService(KOKORO_MODEL_ID, KOKORO_LANG_CODE)
            await kokoro_tts_service.initialize()
        else:
            logger.warning("[Main] Skipping TTS service initialization.")

    except Exception as e:
        logger.exception(f"[Main] Critical error during service initialization: {e}")
        if llm_service:
            await llm_service.close_session()
        return

    tasks = [
        asyncio.create_task(process_audio_input_to_stt(), name="InputProcessor"),
        asyncio.create_task(run_stt_tasks(stt_service), name="STTRunner"), # Modifié pour interruption
        asyncio.create_task(run_llm_tasks(llm_service), name="LLMRunner"), # Corrigé pour lire bonne queue
    ]
    if MLX_AUDIO_AVAILABLE and kokoro_tts_service and kokoro_tts_service.pipeline:
        tasks.extend([
            asyncio.create_task(run_tts_tasks(), name="TTSRunner"),
            asyncio.create_task(playback_task(kokoro_tts_service, KOKORO_VOICE), name="Playback") # Modifié pour interruption
        ])
    else:
        logger.info("[Main] TTS Runner and Playback tasks not started.")

    try:
        logger.info("[Main] Starting audio input stream...")
        input_stream = sd.InputStream(
            samplerate=SAMPLE_RATE_INPUT, channels=CHANNELS_INPUT, dtype='float32',
            blocksize=CHUNK_SAMPLES_INPUT, device=INPUT_DEVICE, callback=input_callback
        )
        input_stream.start()
        logger.info("-" * 30 + "\n[Main] Pipeline running. Parlez en français...\nAppuyez sur Ctrl+C pour arrêter.\n" + "-" * 30)

        # Attendre qu'une tâche se termine (normalement ou par erreur) ou Ctrl+C
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            try:
                await task # Propager l'exception si une tâche a échoué
            except Exception as task_exc:
                logger.error(f"[Main] Task {task.get_name()} failed: {task_exc}")

    except KeyboardInterrupt:
        logger.info("\n[Main] KeyboardInterrupt received, stopping...")
    except sd.PortAudioError as pae:
        logger.exception(f"[Main] PortAudio Error: {pae}")
    except Exception as e:
        logger.exception(f"[Main] Unexpected error in main: {e}")
    finally:
        logger.info("[Main] Initiating shutdown sequence...")
        if input_stream:
            try:
                if input_stream.active:
                    input_stream.stop()
                    input_stream.close()
                    logger.info("[Main] Audio input stopped.")
            except Exception as e_stop:
                logger.error(f"[Main] Error stopping audio input: {e_stop}")

        playback_task_instance = next((t for t in tasks if t.get_name() == "Playback"), None)
        if playback_task_instance and not playback_task_instance.done():
            logger.debug("[Main] Sending stop signal to playback task...")
            try:
                await speak_queue.put(None)
                await asyncio.sleep(0.2)
            except Exception as e_q:
                logger.error(f"[Main] Error putting None to speak_queue: {e_q}")

        if llm_service:
            await llm_service.close_session()

        logger.info("[Main] Cancelling pending tasks...")
        all_current_tasks = asyncio.all_tasks()
        current_main_task = asyncio.current_task()
        tasks_to_cancel = [t for t in all_current_tasks if t is not current_main_task and not t.done()]
        if tasks_to_cancel:
            for task in tasks_to_cancel:
                task.cancel()
            results = await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
            for i, result in enumerate(results):
                task_name = tasks_to_cancel[i].get_name()
                if isinstance(result, asyncio.CancelledError):
                    logger.info(f"[Main] Task {task_name} cancelled.")
                elif isinstance(result, Exception):
                    logger.error(f"[Main] Task {task_name} error during shutdown: {result}")
        else:
            logger.info("[Main] No tasks needed cancellation.")

        logger.info("[Main] Shutdown complete.")

if __name__ == "__main__":
    try:
        logger.info(f"Sounddevice version: {sd.__version__}")
        logger.info("Attempting to query audio devices...")
        logger.info(f"Default input device: {sd.default.device[0]}, Default output device: {sd.default.device[1]}")
        logger.info("Sounddevice check seems OK.")
        sf.check_format('WAV', 'PCM_16')
        logger.info("Soundfile/libsndfile check seems OK.")
    except Exception as e:
        logger.exception("Error during initial audio checks.")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n[Main - Outer] KeyboardInterrupt caught.")
    except Exception as e:
        print(f"\n[FATAL ERROR] Could not run asyncio main loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nApplication finished.")