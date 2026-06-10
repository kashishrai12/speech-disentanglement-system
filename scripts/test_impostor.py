# scripts/test_impostor.py — Direct impostor test (no speakers/mic involved)
import sys, argparse
sys.path.insert(0, 'src/')

import torch
import torchaudio
import numpy as np
from pathlib import Path

from inference_engine import SpeechDisentanglementEngine


def test_file(engine, wav_path, label):
    wav, sr = torchaudio.load(str(wav_path))
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav.mean(dim=0).numpy()

    # Pad to at least 1 second for the engine's buffer
    if len(wav) < 16000:
        wav = np.pad(wav, (0, 16000 - len(wav)))

    # Reset buffer
    engine.audio_buffer.clear()

    # Reset fusion state between files so they're judged independently
    engine.fusion.kws_history.clear()
    engine.fusion.sv_history.clear()
    engine.fusion.frames_since_accept = engine.fusion.cooldown_frames + 1

    # Feed in 100ms chunks like the real demo does
    chunk_size = 1600
    last_result = None
    max_kws = 0
    max_sv = 0
    accept_count = 0

    for i in range(0, len(wav) - chunk_size, chunk_size):
        chunk = wav[i:i + chunk_size]
        result = engine.process_chunk(chunk)
        max_kws = max(max_kws, result.kws_score)
        if result.sv_score > 0:
            max_sv = max(max_sv, result.sv_score)
        if result.accepted:
            accept_count += 1
        last_result = result

    verdict = "🟢 ACCEPTED" if accept_count > 0 else "🔴 rejected"
    print(f"  {label:30s} | max_KWS={max_kws:.3f}  max_SV={max_sv:.3f}  → {verdict}")
    return accept_count > 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_impostor", type=int, default=20)
    parser.add_argument("--n_self", type=int, default=7)
    args = parser.parse_args()

    print("Loading engine...")
    engine = SpeechDisentanglementEngine(
        kws_model_path="models/trained/kws_best.pt",
        sv_model_path="models/trained/sv_distilled.pt",
        speaker_profile_path="profiles/user_1.pt",
        speaker_id="user_1",
        target_word="yes",
        device='cpu',
    )
    print("Ready.\n")

    # ===== Test 1: Impostors saying "yes" =====
    print("=" * 70)
    print(f"IMPOSTOR TEST — {args.n_impostor} 'yes' clips from Speech Commands")
    print("=" * 70)
    impostor_files = list(Path("data/speech_commands/yes").glob("*.wav"))
    np.random.shuffle(impostor_files)
    impostor_files = impostor_files[:args.n_impostor]

    false_accepts = 0
    for f in impostor_files:
        if test_file(engine, f, f.name):
            false_accepts += 1

    fa_rate = false_accepts / len(impostor_files)
    print(f"\n  FA rate on impostor 'yes': {false_accepts}/{len(impostor_files)} = {fa_rate:.1%}")
    print(f"  KPI target: < 1 per hour (~< 5% on short clips)\n")

    # ===== Test 2: Your own recorded enrollment clips =====
    print("=" * 70)
    print(f"SELF TEST — your own enrollment clips")
    print("=" * 70)
    self_files = sorted(Path("profiles/user_1_yes_clips").glob("*.wav"))[:args.n_self]

    true_accepts = 0
    for f in self_files:
        if test_file(engine, f, f.name):
            true_accepts += 1

    ta_rate = true_accepts / max(1, len(self_files))
    print(f"\n  TA rate on your clips: {true_accepts}/{len(self_files)} = {ta_rate:.1%}")
    print(f"  KPI target: ≥ 99% on clean audio")
    print(f"\n  NOTE: enrollment clips say sentences, not 'yes' — so TA may be 0%.")
    print(f"  This is just to verify the SV recognizes you. Look at max_SV scores above.")


if __name__ == "__main__":
    main()