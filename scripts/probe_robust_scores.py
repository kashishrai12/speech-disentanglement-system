# scripts/probe_robust_scores.py — Find impostor & self score distributions for robust model
import sys, random
sys.path.insert(0, 'src/')

import numpy as np
import torch
import torchaudio
from pathlib import Path

from inference_engine import SpeechDisentanglementEngine

random.seed(42)
np.random.seed(42)

print("Loading robust engine...")
engine = SpeechDisentanglementEngine(
    kws_model_path="models/trained/kws_best.pt",
    sv_model_path="models/trained/sv_distilled_robust.pt",
    speaker_profile_path="profiles/user_1_robust.pt",
    speaker_id="user_1",
    target_word="yes",
    device='cpu',
)


def load_wav(path):
    wav, sr = torchaudio.load(str(path))
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav.mean(dim=0).numpy()


def mix_snr(clean, noise, snr):
    if len(noise) < len(clean):
        noise = np.tile(noise, (len(clean) // len(noise)) + 1)
    noise = noise[:len(clean)]
    crms = np.sqrt(np.mean(clean ** 2) + 1e-10)
    nrms = np.sqrt(np.mean(noise ** 2) + 1e-10)
    target = crms / (10 ** (snr / 20))
    m = clean + noise * (target / nrms)
    p = np.abs(m).max()
    if p > 0.99: m = m * (0.99 / p)
    return m.astype(np.float32)


def get_max_sv(audio):
    engine.audio_buffer.clear()
    engine.fusion.kws_history.clear()
    engine.fusion.sv_history.clear()
    engine.fusion.frames_since_accept = engine.fusion.cooldown_frames + 1
    if len(audio) < 16000:
        audio = np.pad(audio, (0, 16000 - len(audio)))
    max_sv = 0.0
    for i in range(0, len(audio) - 1600, 1600):
        r = engine.process_chunk(audio[i:i + 1600])
        if r.sv_score > max_sv:
            max_sv = r.sv_score
    return max_sv


self_files = sorted(Path("profiles/user_1_yes_clips").glob("*.wav"))
imp_files = list(Path("data/speech_commands/yes").glob("*.wav"))
random.shuffle(imp_files)
imp_files = imp_files[:100]
noise_files = list(Path("data/musan").rglob("*.wav"))

print("\n=== Score distributions (max SV per clip) ===\n")
for snr in [30, 10, 0, -5]:
    self_scores = []
    for i in range(30):
        clean = load_wav(self_files[i % len(self_files)])
        x = mix_snr(clean, load_wav(random.choice(noise_files)), snr) if snr < 30 else clean
        self_scores.append(get_max_sv(x))

    imp_scores = []
    for f in imp_files[:50]:
        clean = load_wav(f)
        x = mix_snr(clean, load_wav(random.choice(noise_files)), snr) if snr < 30 else clean
        imp_scores.append(get_max_sv(x))

    self_arr = np.array(self_scores)
    imp_arr = np.array(imp_scores)
    print(f"SNR {snr:>3}dB | "
          f"Self  mean={self_arr.mean():.3f}  p10={np.percentile(self_arr, 10):.3f}  p50={np.percentile(self_arr, 50):.3f} "
          f"| Imp  mean={imp_arr.mean():.3f}  max={imp_arr.max():.3f}  p95={np.percentile(imp_arr, 95):.3f}")