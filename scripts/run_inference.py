# scripts/run_inference.py — Live microphone demo for VoxGate
import sys, os, argparse, time
sys.path.insert(0, 'src/')

import numpy as np
import sounddevice as sd
import torch

from inference_engine import SpeechDisentanglementEngine


def main():
    parser = argparse.ArgumentParser(description="VoxGate live demo")
    parser.add_argument("--speaker_id", type=str, default="user_1")
    parser.add_argument("--target_word", type=str, default="yes")
    parser.add_argument("--kws_model_path", type=str,
                        default="models/trained/kws_best.pt")
    parser.add_argument("--sv_model_path", type=str,
                    default="models/trained/sv_distilled_robust.pt")
    parser.add_argument("--profile_path", type=str,
                    default="profiles/user_1_robust.pt")
    parser.add_argument("--chunk_ms", type=int, default=100,
                        help="Audio chunk size in ms (smaller = lower latency)")
    parser.add_argument("--cooldown_ms", type=int, default=1500,
                        help="Min ms between accepted triggers")
    args = parser.parse_args()

    sample_rate = 16000
    chunk_samples = int(sample_rate * args.chunk_ms / 1000)

    # ===== Build engine =====
    print("Loading VoxGate engine...")
    engine = SpeechDisentanglementEngine(
        kws_model_path=args.kws_model_path,
        sv_model_path=args.sv_model_path,
        speaker_profile_path=args.profile_path,
        speaker_id=args.speaker_id,
        target_word=args.target_word,
        device='cpu',
    )
    print("Engine ready.\n")

    # ===== Counters & cooldown =====
    last_trigger_ms = 0
    chunk_count = 0
    total_kws_runs = 0
    total_sv_runs = 0
    slow_chunks = 0
    latencies = []

    print("=" * 60)
    print(f"VoxGate is listening...")
    print(f"  Target word:    '{args.target_word}'")
    print(f"  Target speaker: '{args.speaker_id}'")
    print(f"  Press Ctrl+C to stop.")
    print("=" * 60 + "\n")

    # ===== Audio callback =====
    # We use a queue-style flag-based approach to keep the callback fast
    # (no model inference inside the audio callback itself).
    import queue
    audio_queue = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"[audio status] {status}", flush=True)
        audio_queue.put(indata[:, 0].copy())

    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype='float32',
        blocksize=chunk_samples,
        callback=audio_callback,
    )

    try:
        # Accumulate audio across 2 chunks (200ms) before processing
        accumulator = []
        with stream:
            while True:
                chunk = audio_queue.get()
                accumulator.append(chunk)

                # Process every 2 chunks (200ms effective hop)
                if len(accumulator) < 2:
                    continue

                combined = np.concatenate(accumulator)
                accumulator = []
                chunk_count += 1

                t0 = time.perf_counter()
                result = engine.process_chunk(combined)
                latency_ms = (time.perf_counter() - t0) * 1000
                latencies.append(latency_ms)

                if latency_ms > 100:
                    slow_chunks += 1

                # Track which stages ran
                total_kws_runs += 1
                # Heuristic: SV likely ran if KWS prob was above the cascade gate
                if result.kws_score >= 0.50:
                    total_sv_runs += 1

                # === Decision handling ===
                now_ms = time.time() * 1000
                in_cooldown = (now_ms - last_trigger_ms) < args.cooldown_ms

                if result.accepted and not in_cooldown:
                    last_trigger_ms = now_ms
                    print(f"\n🟢 ACCEPT  | "
                          f"KWS={result.kws_score:.3f}  "
                          f"SV={result.sv_score:.3f}  "
                          f"reason={result.reason}  "
                          f"latency={latency_ms:.1f}ms")
                elif result.kws_score > 0.15 and not in_cooldown:
                    print(f"🔴 reject  | "
                        f"KWS={result.kws_score:.3f}  "
                        f"SV={result.sv_score:.3f}  "
                        f"reason={result.reason}",
                        flush=True)

                # Periodic stats line
                if chunk_count % 50 == 0:
                    avg_lat = np.mean(latencies[-50:])
                    p95_lat = np.percentile(latencies[-50:], 95)
                    print(f"[stats] chunks={chunk_count} "
                          f"avg_lat={avg_lat:.1f}ms  "
                          f"p95={p95_lat:.1f}ms  "
                          f"SV_calls={total_sv_runs}/{total_kws_runs}",
                          flush=True)

    except KeyboardInterrupt:
        print("\n\n=== Stopping ===")

    # ===== Final stats =====
    if latencies:
        print(f"\nTotal chunks:       {chunk_count}")
        print(f"Mean latency:       {np.mean(latencies):.2f} ms")
        print(f"P95 latency:        {np.percentile(latencies, 95):.2f} ms")
        print(f"P99 latency:        {np.percentile(latencies, 99):.2f} ms")
        print(f"Slow chunks (>100ms): {slow_chunks}")
        print(f"SV cascade rate:    {total_sv_runs}/{total_kws_runs} "
              f"({100*total_sv_runs/max(1,total_kws_runs):.1f}%)")

        # xRT calculation
        real_audio_ms = chunk_count * args.chunk_ms
        total_compute_ms = sum(latencies)
        xrt = total_compute_ms / real_audio_ms
        print(f"xRT:                {xrt:.4f}  (target < 0.2)")
        if xrt < 0.2:
            print("✅ Meets xRT KPI")
        else:
            print("⚠️  xRT exceeds 0.2 — see tuning suggestions")


if __name__ == "__main__":
    main()

    