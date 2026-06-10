# evaluate.py — Formal KPI evaluation across SNR levels and impostor sets
import sys, argparse, time, random
sys.path.insert(0, 'src/')

import numpy as np
import torch
import torchaudio
from pathlib import Path

from inference_engine import SpeechDisentanglementEngine


def load_wav(path, target_sr=16000):
    wav, sr = torchaudio.load(str(path))
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.mean(dim=0).numpy()


def mix_at_snr(clean, noise, snr_db):
    """Mix clean + noise to achieve target SNR in dB."""
    # Match length
    if len(noise) < len(clean):
        reps = (len(clean) // len(noise)) + 1
        noise = np.tile(noise, reps)
    noise = noise[:len(clean)]

    clean_rms = np.sqrt(np.mean(clean ** 2) + 1e-10)
    noise_rms = np.sqrt(np.mean(noise ** 2) + 1e-10)

    # Scale noise to achieve target SNR
    target_noise_rms = clean_rms / (10 ** (snr_db / 20))
    noise_scaled = noise * (target_noise_rms / noise_rms)

    mixed = clean + noise_scaled
    # Avoid clipping
    peak = np.abs(mixed).max()
    if peak > 0.99:
        mixed = mixed * (0.99 / peak)
    return mixed.astype(np.float32)


def run_one(engine, audio):
    """Feed audio chunk-by-chunk; return whether system accepted + scores."""
    engine.audio_buffer.clear()
    engine.fusion.kws_history.clear()
    engine.fusion.sv_history.clear()
    engine.fusion.frames_since_accept = engine.fusion.cooldown_frames + 1

    # Pad to at least 1 sec
    if len(audio) < 16000:
        audio = np.pad(audio, (0, 16000 - len(audio)))

    chunk_size = 1600
    accept = False
    max_kws, max_sv = 0.0, 0.0
    latencies = []

    for i in range(0, len(audio) - chunk_size, chunk_size):
        chunk = audio[i:i + chunk_size]
        t0 = time.perf_counter()
        result = engine.process_chunk(chunk)
        latencies.append((time.perf_counter() - t0) * 1000)
        max_kws = max(max_kws, result.kws_score)
        if result.sv_score > 0:
            max_sv = max(max_sv, result.sv_score)
        if result.accepted:
            accept = True

    return accept, max_kws, max_sv, latencies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_self_per_snr", type=int, default=30,
                       help="Augmentations per enrollment clip per SNR")
    parser.add_argument("--n_impostor_per_snr", type=int, default=50)
    parser.add_argument("--snrs", type=int, nargs="+",
                       default=[30, 20, 10, 5, 0, -5])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # ===== Build engine =====
    print("Loading engine...")
    engine = SpeechDisentanglementEngine(
        kws_model_path="models/trained/kws_best.pt",
        sv_model_path="models/trained/sv_distilled.pt",
        speaker_profile_path="profiles/user_1.pt",
        speaker_id="user_1",
        target_word="yes",
        device='cpu',
    )
    print("Engine ready.\n")

    # ===== Gather audio =====
    self_yes_files = sorted(Path("profiles/user_1_yes_clips").glob("*.wav"))
    if not self_yes_files:
        print("ERROR: No self 'yes' clips found in profiles/user_1_yes_clips/")
        print("Run: python scripts/record_self_yes.py first.")
        return

    impostor_files = list(Path("data/speech_commands/yes").glob("*.wav"))
    random.shuffle(impostor_files)

    # MUSAN noise files
    noise_files = list(Path("data/musan/noise").rglob("*.wav"))
    if not noise_files:
        # Fallback to any musan subfolder
        noise_files = list(Path("data/musan").rglob("*.wav"))
    if not noise_files:
        print("ERROR: No MUSAN noise files found.")
        return

    print(f"Self 'yes' clips:   {len(self_yes_files)}")
    print(f"Impostor 'yes':     {len(impostor_files)}")
    print(f"MUSAN noise files:  {len(noise_files)}")
    print(f"SNR levels:         {args.snrs}\n")

    # ===== Evaluation =====
    results = {}   # results[snr] = {'ta': x, 'fa': y, ...}
    all_latencies = []

    for snr in args.snrs:
        print(f"\n{'=' * 60}")
        print(f"  SNR = {snr} dB")
        print('=' * 60)

        # --- TA: self clips + noise ---
        ta_accepts = 0
        ta_total = 0
        for self_file in self_yes_files:
            clean = load_wav(self_file)
            # Mix with N different noise files
            for _ in range(args.n_self_per_snr // len(self_yes_files) + 1):
                if ta_total >= args.n_self_per_snr:
                    break
                noise = load_wav(random.choice(noise_files))
                mixed = mix_at_snr(clean, noise, snr) if snr < 30 else clean
                accepted, _, _, lats = run_one(engine, mixed)
                if accepted:
                    ta_accepts += 1
                ta_total += 1
                all_latencies.extend(lats)

        # --- FA: impostor clips + noise ---
        fa_accepts = 0
        fa_total = 0
        for imp_file in impostor_files[:args.n_impostor_per_snr]:
            clean = load_wav(imp_file)
            noise = load_wav(random.choice(noise_files))
            mixed = mix_at_snr(clean, noise, snr) if snr < 30 else clean
            accepted, _, _, lats = run_one(engine, mixed)
            if accepted:
                fa_accepts += 1
            fa_total += 1
            all_latencies.extend(lats)

        ta_rate = ta_accepts / max(1, ta_total)
        fa_rate = fa_accepts / max(1, fa_total)
        results[snr] = {
            'ta': ta_rate, 'ta_count': f"{ta_accepts}/{ta_total}",
            'fa': fa_rate, 'fa_count': f"{fa_accepts}/{fa_total}",
        }
        print(f"  TA: {ta_accepts}/{ta_total} = {ta_rate:.1%}")
        print(f"  FA: {fa_accepts}/{fa_total} = {fa_rate:.1%}")

    # ===== Summary =====
    print(f"\n{'=' * 70}")
    print("  FINAL KPI REPORT")
    print('=' * 70)
    print(f"{'SNR':>6} | {'TA Rate':>15} | {'FA Rate':>15} | {'TA KPI':>8} | {'FA KPI':>8}")
    print('-' * 70)
    for snr in args.snrs:
        r = results[snr]
        ta_kpi = '✅' if (snr >= 20 and r['ta'] >= 0.99) or (snr < 20 and r['ta'] >= 0.90) else '❌'
        # FA: < 1/hr — roughly < 1% on this sample size
        fa_kpi = '✅' if r['fa'] < 0.01 else ('⚠️' if r['fa'] < 0.05 else '❌')
        print(f"{snr:>5}dB | {r['ta_count']:>9} ({r['ta']:>4.0%}) | "
              f"{r['fa_count']:>9} ({r['fa']:>4.0%}) | {ta_kpi:>8} | {fa_kpi:>8}")

    # FA per hour calc
    total_impostor_tests = sum(int(results[snr]['fa_count'].split('/')[1]) for snr in args.snrs)
    total_fa = sum(int(results[snr]['fa_count'].split('/')[0]) for snr in args.snrs)
    # Each test is ~1 sec, so total_impostor_tests = seconds
    fa_per_hour = (total_fa / total_impostor_tests) * 3600 if total_impostor_tests > 0 else 0
    print(f"\nEstimated FA per hour: {fa_per_hour:.2f}  (KPI: < 1)")

    # xRT
    if all_latencies:
        total_compute_ms = sum(all_latencies)
        total_audio_ms = len(all_latencies) * 100  # each chunk = 100ms
        xrt = total_compute_ms / total_audio_ms
        print(f"\nxRT: {xrt:.4f}  (KPI: < 0.2)  "
              f"{'✅' if xrt < 0.2 else '❌'}")
        print(f"  Mean latency/chunk: {np.mean(all_latencies):.1f}ms")
        print(f"  P95: {np.percentile(all_latencies, 95):.1f}ms")

    # Total params
    kws_params = sum(p.numel() for p in engine.kws_model.parameters())
    sv_params = sum(p.numel() for p in engine.sv_model.parameters())
    total = kws_params + sv_params
    print(f"\nModel size: {total:,} ({total/1e6:.2f}M)  KPI: < 3M  "
          f"{'✅' if total < 3e6 else '❌'}")
    print(f"  KWS: {kws_params:,}  SV: {sv_params:,}")


if __name__ == "__main__":
    main()