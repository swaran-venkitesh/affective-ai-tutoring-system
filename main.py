import time
import queue
import threading
import io

import sounddevice as sd
import soundfile as sf

from stt_model_v2 import STTClient, STTConfig
from LLM_QEWN_v2 import QwenChat, QwenConfig
from qwen_tts_client_v2 import tts_request_bytes, DEFAULT_INSTRUCT

CHUNK_MARK = "<CHUNK>"


def play_wav_bytes(wav_bytes: bytes):
    wav, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
    sd.play(wav, sr, blocking=True)


def split_ready_chunks(buf: str):
    """
    Returns (chunks_ready, remainder).
    Splits on <CHUNK>.
    """
    parts = buf.split(CHUNK_MARK)
    if len(parts) == 1:
        return [], buf
    chunks = [p.strip() for p in parts[:-1] if p.strip()]
    rem = parts[-1]
    return chunks, rem


def main():
    llm = QwenChat(QwenConfig())

    text_q: "queue.Queue[str]" = queue.Queue(maxsize=50)
    print_lock = threading.Lock()

    def on_status(msg: str):
        with print_lock:
            print(msg, flush=True)

    def on_text(txt: str):
        # only accept STT when NOT paused (extra safety)
        try:
            text_q.put_nowait(txt)
        except queue.Full:
            pass

    # ===================== STT CONFIG =====================
    stt_cfg = STTConfig(
        api_url="http://172.16.13.91:8009/stt",
        fs=16000,
        frame_ms=30,
        vad_mode=2,
        rms_thresh=30,
        end_silence_ms=900,     # ✅ important for endpointing
        min_utter_sec=0.7,
        min_speech_ratio=0.35,
        preroll_frames=12,
        latency="high",
    )

    stt = STTClient(on_text=on_text, config=stt_cfg, on_status=on_status)

    # ===================== LLM + TTS PIPELINE =====================
    def llm_worker():
        with print_lock:
            print("\n✅ TUTOR PIPELINE STARTED (HALF-DUPLEX)")
            print("🎤 Speak → STT → Qwen (chunked) → Speak chunk-by-chunk")
            print("🔒 While tutor speaks: mic is PAUSED (no random inputs)")
            print("-" * 60)

        while True:
            user_txt = text_q.get()
            if user_txt is None:
                break

            # ✅ HALF-DUPLEX: pause mic immediately when we start responding
            stt.pause()

            # ✅ drop any accidental queued user texts (echo/noise)
            try:
                while True:
                    text_q.get_nowait()
            except Exception:
                pass

            with print_lock:
                print(f"\n🧑 You (STT): {user_txt}")
                print("-" * 60)
                print("🧠 Analyzing…", flush=True)

            # Per-turn processing inside try/finally so we ALWAYS resume STT
            try:
                chunk_text_q: "queue.Queue[tuple[int,str]]" = queue.Queue(maxsize=20)
                audio_q: "queue.Queue[tuple[int,str,bytes]]" = queue.Queue(maxsize=20)

                stop = threading.Event()

                # ---- TTS worker (prefetch) ----
                def tts_worker():
                    while not stop.is_set():
                        item = chunk_text_q.get()
                        if item is None:
                            break
                        idx, chunk_text = item
                        try:
                            wav_bytes, _sr = tts_request_bytes(chunk_text, instruct=DEFAULT_INSTRUCT)
                            audio_q.put((idx, chunk_text, wav_bytes))
                        except Exception as e:
                            audio_q.put((idx, f"[TTS ERROR: {e}] {chunk_text}", b""))
                        finally:
                            try:
                                chunk_text_q.task_done()
                            except Exception:
                                pass

                t_tts = threading.Thread(target=tts_worker, daemon=True)
                t_tts.start()

                # ---- LLM producer ----
                def llm_producer():
                    buf = ""
                    idx = 0
                    for tok in llm.stream_tokens(user_txt):
                        buf += tok
                        ready, buf = split_ready_chunks(buf)
                        for c in ready:
                            # keep only a few chunks ahead (real-time feel)
                            while chunk_text_q.qsize() >= 3 and not stop.is_set():
                                time.sleep(0.02)
                            chunk_text_q.put((idx, c))
                            idx += 1

                    # flush remainder
                    rem = buf.strip()
                    if rem:
                        chunk_text_q.put((idx, rem))

                    chunk_text_q.put(None)

                t_prod = threading.Thread(target=llm_producer, daemon=True)
                t_prod.start()

                # ---- Playback/UI consumer (in order) ----
                expected = 0
                pending = {}

                while True:
                    try:
                        idx, chunk_text, wav_bytes = audio_q.get(timeout=0.25)
                    except queue.Empty:
                        # exit when producer finished + queues drained
                        if (not t_prod.is_alive()) and chunk_text_q.empty() and len(pending) == 0:
                            break
                        continue

                    pending[idx] = (chunk_text, wav_bytes)

                    while expected in pending:
                        ctext, wb = pending.pop(expected)

                        with print_lock:
                            # First real content replaces "Analyzing..." feel
                            print("\n" + ctext, flush=True)

                        if wb:
                            play_wav_bytes(wb)

                        expected += 1

                stop.set()

                with print_lock:
                    print("\n" + "-" * 60)

            finally:
                # ✅ HALF-DUPLEX: resume mic ONLY after tutor is fully done speaking
                stt.resume()

    # ===================== START THREADS =====================
    t_llm = threading.Thread(target=llm_worker, daemon=True)
    t_llm.start()

    stt.start()

    try:
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        on_status("\n🛑 Stopping...")

    finally:
        stt.stop()
        try:
            text_q.put_nowait(None)
        except queue.Full:
            pass


if __name__ == "__main__":
    main()
