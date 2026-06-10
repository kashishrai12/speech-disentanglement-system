# scripts/eval_robust.py — Run evaluate.py logic but with the robust model
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
    if len(noise) < len(clean):
        noise = np.tile(noise, (len(clean) // len(noise)) + 1)
    noise = noise[:len(clean)]
    clean_rms = np.sqrt(np.mean(clean ** 2) + 1e-10)
    noise_rms = np.sqrt(np.mean(noise ** 2) + 1e-10)
    target = clean_rms / (10 ** (snr_db / 20))
    mixed = clean + noise * (target / noise_rms)
    peak = np.abs(mixed).max()
    if peak > 0.99:
        mixed = mixed * (0.99 / peak)
    return mixed.astype(np.float32)


def run_one(engine, audio):
    engine.audio_buffer.clear()
    engine.fusion.kws_history.clear()
    engine.fusion.sv_history.clear()
    engine.fusion.frames_since_accept = engine.fusion.cooldown_frames + 1
    if len(audio) < 16000:
        audio = np.pad(audio, (0, 16000 - len(audio)))
    accept = False
    for i in range(0, len(audio) - 1600, 1600):
        result = engine.process_chunk(audio[i:i + 1600])
        if result.accepted:
            accept = True
    return accept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sv_model", default="models/trained/sv_distilled_robust.pt")
    ap.add_argument("--snrs", type=int, nargs="+", default=[30, 10, 0, -5])
    ap.add_argument("--n_self", type=int, default=15)
    ap.add_argument("--n_imp", type=int, default=25)
    args = ap.parse_args()

    random.seed(42)
    np.random.seed(42)

    print(f"Using SV model: {args.sv_model}")
    engine = SpeechDisentanglementEngine(
        kws_model_path="models/trained/kws_best.pt",
        sv_model_path=args.sv_model,
        speaker_profile_path="profiles/user_1_robust.pt" if "robust" in args.sv_model else "profiles/user_1.pt",
        speaker_id="user_1",
        target_word="yes",
        device='cpu',
    )

    self_files = sorted(Path("profiles/user_1_yes_clips").glob("*.wav"))
    imp_files = list(Path("data/speech_commands/yes").glob("*.wav"))
    random.shuffle(imp_files)
    noise_files = list(Path("data/musan").rglob("*.wav"))

    print(f"\n{'SNR':>5} | {'TA':>10} | {'FA':>10}")
    print("-" * 35)

    for snr in args.snrs:
        # TA
        ta = 0
        for k in range(args.n_self):
            clean = load_wav(self_files[k % len(self_files)])
            mixed = mix_at_snr(clean, load_wav(random.choice(noise_files)), snr) if snr < 30 else clean
            if run_one(engine, mixed):
                ta += 1

        # FA
        fa = 0
        for f in imp_files[:args.n_imp]:
            clean = load_wav(f)
            mixed = mix_at_snr(clean, load_wav(random.choice(noise_files)), snr) if snr < 30 else clean
            if run_one(engine, mixed):
                fa += 1

        print(f"{snr:>4}dB | {ta}/{args.n_self} ({100*ta/args.n_self:>3.0f}%) | "
              f"{fa}/{args.n_imp} ({100*fa/args.n_imp:>3.0f}%)")


if __name__ == "__main__":
    main()