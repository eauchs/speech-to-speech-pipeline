# Pipeline Conversationnel Local Temps Réel (STT-LLM-TTS)

Ce projet implémente un pipeline conversationnel en temps réel qui s'exécute localement sur votre machine (optimisé pour macOS avec Apple Silicon). Il capture votre voix, la transcrit en texte, interroge un Modèle de Langage Large (LLM) local pour obtenir une réponse, synthétise cette réponse en parole et la joue, tout en gérant les interruptions (barge-in) pour une interaction fluide. L'objectif principal est d'atteindre une faible latence.

## Fonctionnalités Clés

* **Pipeline Complet :** Speech-to-Text -> LLM -> Text-to-Speech.
* **Temps Réel & Faible Latence :** Utilisation d'`asyncio`, de queues, de chunking audio et de modèles optimisés (quantifiés).
* **STT :** Utilise [Faster Whisper](https://github.com/guillaumekln/faster-whisper) (modèle `small` quantifié `int8` par défaut) pour une transcription rapide et efficace sur CPU.
* **LLM :** Interagit avec une API LLM locale compatible OpenAI (ex: [LM Studio](https://lmstudio.ai/), [Ollama](https://ollama.com/)) via streaming SSE.
* **TTS :** Utilise [MLX Audio](https://github.com/ml-explore/mlx-audio) avec le modèle **Kokoro** (version 4-bit par défaut), optimisé pour l'accélération matérielle (MPS) sur les puces Apple Silicon.
* **Interruption (Barge-in) :** La parole de l'assistant est automatiquement coupée si l'utilisateur recommence à parler.
* **Configurable :** Les modèles, appareils audio, seuils de silence, etc., peuvent être ajustés via les variables au début du script.
* **Multi-plateforme (avec limitations) :** Principalement conçu et optimisé pour macOS/Apple Silicon (grâce à MLX). Le STT et le LLM peuvent fonctionner sur d'autres plateformes, mais le TTS MLX ne fonctionnera pas sans Apple Silicon.

## Technologies Utilisées

* Python 3.10+
* Asyncio, Threading, Queue
* Sounddevice (Entrée/Sortie audio)
* NumPy (Manipulation audio)
* Aiohttp (Client HTTP asynchrone pour LLM)
* Faster Whisper (STT)
* MLX Audio & Kokoro (TTS - *Nécessite macOS/Apple Silicon*)
* PyTorch (Inférence Whisper & potentiellement Kokoro si non-MLX)
* SoundFile (Sauvegarde audio optionnelle)

## Prérequis

* **Système :** macOS (fortement recommandé, surtout Apple Silicon pour le TTS MLX). Fonctionnement partiel possible sur d'autres OS (sans TTS MLX).
* **Python :** Version 3.10 ou supérieure.
* **FFmpeg :** Nécessaire pour `torchaudio` (utilisé par `mlx-audio` et potentiellement `faster-whisper`). Installez via Homebrew sur Mac : `brew install ffmpeg`.
* **Git :** Pour cloner le dépôt.
* **Compte Hugging Face :** Pour télécharger les modèles Whisper et Kokoro (via `huggingface-cli login`).
* **Serveur LLM Local :** Une instance compatible API OpenAI tournant localement (ex: LM Studio, Ollama) et configurée pour utiliser un modèle conversationnel (ex: Llama 3 Instruct).

## Installation

1.  **Cloner le dépôt :**
    ```bash
    git clone https://github.com/eauchs/speech-to-speech-pipeline.git
    cd speech-to-speech-pipeline
    ```
2.  **Créer et activer un environnement virtuel :**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
3.  **Installer les dépendances :**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note : L'installation de `mlx-audio` et de ses dépendances peut nécessiter des étapes spécifiques sur Mac. Si `pip install mlx-audio` échoue, consultez leur documentation.)*
4.  **Se connecter à Hugging Face :**
    ```bash
    huggingface-cli login
    ```
    *(Suivez les instructions pour coller votre token d'accès).*
5.  **Configurer votre LLM Local :**
    * Lancez votre serveur LLM (LM Studio, Ollama...).
    * Assurez-vous qu'il expose une API compatible OpenAI sur `http://localhost:1234` (ou l'adresse configurée dans le script).
    * Chargez un modèle conversationnel approprié (ex: Meta-Llama-3-8B-Instruct-GGUF).
    * Mettez à jour `LLM_MODEL_NAME` dans le script si nécessaire pour correspondre au nom du modèle chargé dans votre serveur local.

## Configuration

Ajustez les variables dans la section `# --- Configuration ---` au début du script `main_pipeline.py` selon vos besoins :

* `WHISPER_MODEL_SIZE`, `FASTER_WHISPER_DEVICE`, `FASTER_WHISPER_COMPUTE_TYPE`, `STT_LANGUAGE`: Paramètres pour le STT Faster Whisper.
* `LLM_API_ENDPOINT`, `LLM_MODEL_NAME`, `LLM_SYSTEM_PROMPT`: Paramètres pour l'API LLM locale.
* `KOKORO_MODEL_ID`, `KOKORO_VOICE`, `KOKORO_LANG_CODE`, `TTS_SPEECH_SPEED`: Paramètres pour le TTS Kokoro MLX.
* `SAMPLE_RATE_INPUT`, `SAMPLE_RATE_OUTPUT`, `CHUNK_DURATION_MS`, `INPUT_DEVICE`, `OUTPUT_DEVICE`: Paramètres audio.
* `BUFFER_DURATION_S`, `SILENCE_THRESHOLD`, `SILENCE_CHUNKS_NEEDED`: Paramètres pour le buffering STT et la détection de silence (cruciaux pour la réactivité).
* `SAVE_TTS_AUDIO`: Mettre à `True` pour sauvegarder les fichiers audio générés par le TTS.

## Utilisation

1.  Assurez-vous que votre serveur LLM local est lancé et configuré.
2.  Activez votre environnement virtuel (`source .venv/bin/activate`).
3.  Exécutez le script principal :
    ```bash
    python main_pipeline.py
    ```
4.  Attendez le message `[Main] Pipeline running. Parlez en français...`. Les modèles (Whisper, Kokoro) seront téléchargés lors du premier lancement si nécessaire (cela peut prendre du temps).
5.  Parlez dans votre microphone. L'assistant devrait répondre après un court délai.
6.  Pour interrompre l'assistant pendant qu'il parle, commencez simplement à parler.
7.  Appuyez sur `Ctrl+C` dans le terminal pour arrêter le script proprement.

## Compatibilité

* Ce script est optimisé pour **macOS avec Apple Silicon** en raison de l'utilisation de `mlx-audio` pour le TTS.
* Le STT (Faster Whisper sur CPU) et l'appel LLM devraient fonctionner sur d'autres plateformes (Linux, Windows).
* Le TTS ne fonctionnera probablement pas en l'état sur des plateformes non-Apple Silicon sans modification pour utiliser une autre bibliothèque TTS (ex: Coqui TTS, Piper, etc.).
* L'entrée/sortie audio via `sounddevice` devrait être multiplateforme, mais les noms/index des périphériques peuvent varier.

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## Remerciements

* [Faster Whisper](https://github.com/guillaumekln/faster-whisper) pour le STT efficace.
* [MLX Audio](https://github.com/ml-explore/mlx-audio) et la communauté MLX pour le TTS Kokoro optimisé.
* [Sounddevice](https://python-sounddevice.readthedocs.io/) pour l'accès audio multiplateforme.
* La communauté open-source des LLMs, en particulier MLX.
