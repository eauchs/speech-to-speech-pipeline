

# Local Real-Time Conversational Pipeline (STT-LLM-TTS)

This project implements a real-time conversational pipeline that runs locally on your machine (optimized for macOS with Apple Silicon). It captures your voice, transcribes it to text, queries a local Large Language Model (LLM) for a response, synthesizes that response back into speech, and plays it—all while managing interruptions (barge-in) for a fluid interaction. The primary goal is to achieve minimal latency.

## Key Features

  * **Full Pipeline:** Speech-to-Text -\> LLM -\> Text-to-Speech.
  * **Real-Time & Low Latency:** Uses `asyncio`, `queues`, audio chunking, and optimized (quantized) models.
  * **STT:** Uses **Faster Whisper** (default: `small` `int8` quantized model) for fast and efficient CPU-based transcription.
  * **LLM:** Interacts with any local OpenAI-compatible API (e.g., **LM Studio**, **Ollama**) via SSE streaming.
  * **TTS:** Uses **MLX Audio** with the **Kokoro** model (default: 4-bit version), optimized for hardware acceleration (MPS) on Apple Silicon chips.
  * **Barge-in (Interruption):** The assistant's speech is automatically cut off if the user begins speaking again.
  * **Configurable:** Models, audio devices, silence thresholds, etc., can be adjusted via variables at the top of the script.
  * **Multi-platform (with limitations):** Primarily designed and optimized for macOS/Apple Silicon (due to MLX). The STT and LLM components can run on other platforms, but the MLX TTS will not function without Apple Silicon.

## Technologies Used

  * Python 3.10+
  * `Asyncio`, `Threading`, `Queue`
  * `Sounddevice` (Audio I/O)
  * `NumPy` (Audio manipulation)
  * `Aiohttp` (Async HTTP client for LLM)
  * `Faster Whisper` (STT)
  * `MLX Audio` & `Kokoro` (TTS - **Requires macOS/Apple Silicon**)
  * `PyTorch` (Whisper inference & potentially Kokoro if non-MLX)
  * `SoundFile` (Optional audio saving)

## Prerequisites

  * **System:** macOS (strongly recommended, especially Apple Silicon for MLX TTS). Partial functionality is possible on other OSes (without MLX TTS).
  * **Python:** Version 3.10 or higher.
  * **FFmpeg:** Required for `torchaudio` (used by `mlx-audio` and potentially `faster-whisper`). Install via Homebrew on Mac: `brew install ffmpeg`.
  * **Git:** To clone the repository.
  * **Hugging Face Account:** To download the Whisper and Kokoro models (requires `huggingface-cli login`).
  * **Local LLM Server:** An OpenAI-compatible API instance running locally (e.g., LM Studio, Ollama) and configured with a conversational model (e.g., Llama 3 Instruct).

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/eauchs/speech-to-speech-pipeline.git
    cd speech-to-speech-pipeline
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    *(Note: Installing `mlx-audio` and its dependencies may have specific steps on Mac. If `pip install mlx-audio` fails, consult their documentation.)*

4.  **Log in to Hugging Face:**

    ```bash
    huggingface-cli login
    ```

    *(Follow the prompts to paste your access token).*

5.  **Configure Your Local LLM:**

      * Launch your LLM server (LM Studio, Ollama...).
      * Ensure it exposes an OpenAI-compatible API on `http://localhost:1234` (or the address configured in the script).
      * Load a suitable conversational model (e.g., `Meta-Llama-3-8B-Instruct-GGUF`).
      * Update `LLM_MODEL_NAME` in the script if needed to match the model name loaded in your local server.

## Configuration

Adjust the variables in the `# --- Configuration ---` section at the top of `main_pipeline.py` as needed:

  * `WHISPER_MODEL_SIZE`, `FASTER_WHISPER_DEVICE`, `FASTER_WHISPER_COMPUTE_TYPE`, `STT_LANGUAGE`: Settings for Faster Whisper STT.
  * `LLM_API_ENDPOINT`, `LLM_MODEL_NAME`, `LLM_SYSTEM_PROMPT`: Settings for the local LLM API.
  * `KOKORO_MODEL_ID`, `KOKORO_VOICE`, `KOKORO_LANG_CODE`, `TTS_SPEECH_SPEED`: Settings for Kokoro MLX TTS.
  * `SAMPLE_RATE_INPUT`, `SAMPLE_RATE_OUTPUT`, `CHUNK_DURATION_MS`, `INPUT_DEVICE`, `OUTPUT_DEVICE`: Audio parameters.
  * `BUFFER_DURATION_S`, `SILENCE_THRESHOLD`, `SILENCE_CHUNKS_NEEDED`: STT buffering and silence detection settings (crucial for responsiveness).
  * `SAVE_TTS_AUDIO`: Set to `True` to save the generated TTS audio files.

## Usage

1.  Ensure your local LLM server is running and configured.
2.  Activate your virtual environment (`source .venv/bin/activate`).
3.  Run the main script:
    ```bash
    python main_pipeline.py
    ```
4.  Wait for the message `[Main] Pipeline running. Parlez en français....` (The models will be downloaded on the first run, which may take time).
5.  Speak into your microphone. The assistant should respond after a short delay.
6.  To interrupt the assistant while it's speaking, simply start speaking.
7.  Press `Ctrl+C` in the terminal to stop the script cleanly.

## Compatibility

  * This script is optimized for **macOS with Apple Silicon** due to its use of `mlx-audio` for TTS.
  * The STT (Faster Whisper on CPU) and LLM call should work on other platforms (Linux, Windows).
  * The TTS will likely not work on non-Apple Silicon platforms without modification to use a different TTS library (e.g., Coqui TTS, Piper, etc.).
  * Audio I/O via `sounddevice` should be cross-platform, but device names/indices may vary.

## License

This project is licensed under the MIT License. See the `LICENSE.md` file for details.

## Acknowledgements

  * **Faster Whisper** for efficient STT.
  * **MLX Audio** and the MLX community for the optimized Kokoro TTS.
  * **Sounddevice** for cross-platform audio access.
  * The open-source LLM community, especially MLX.
