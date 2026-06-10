#tune_threshold.py - Analyze model scores to select a good threshold for keyword spotting
import sys, torch
sys.path.insert(0, 'src/')

import torchaudio
from pathlib import Path
from audio_frontend import AudioFrontend
from kws_model import TCResNetKWS
import numpy as np

device = torch.device('cpu')
frontend = AudioFrontend()
model = TCResNetKWS(n_mels=40, num_classes=2).to(device)
model.load_state_dict(torch.load("models/trained/kws_best.pt", map_location=device, weights_only=False))
model.eval()

# Collect scores for positive and negative samples
pos_scores = []
neg_scores = []

data_path = Path("data/speech_commands")
print("Scoring samples...")

with torch.no_grad():
    # Score 'yes' samples
    yes_files = list((data_path / "yes").glob("*.wav"))[:500]
    for f in yes_files:
        try:
            w, sr = torchaudio.load(str(f))
            if sr != 16000:
                w = torchaudio.functional.resample(w, sr, 16000)
            feat = frontend.extract_features(w).unsqueeze(0)
            prob = torch.softmax(model(feat), dim=1)[0, 1].item()
            pos_scores.append(prob)
        except:
            continue

    # Score negative samples (sample from a few words)
    for word in ["no", "up", "down", "stop", "go"]:
        neg_files = list((data_path / word).glob("*.wav"))[:100]
        for f in neg_files:
            try:
                w, sr = torchaudio.load(str(f))
                if sr != 16000:
                    w = torchaudio.functional.resample(w, sr, 16000)
                feat = frontend.extract_features(w).unsqueeze(0)
                prob = torch.softmax(model(feat), dim=1)[0, 1].item()
                neg_scores.append(prob)
            except:
                continue

pos_scores = np.array(pos_scores)
neg_scores = np.array(neg_scores)

print(f"\nPositive samples scored: {len(pos_scores)}")
print(f"Negative samples scored: {len(neg_scores)}")
print(f"\nPos score distribution:")
print(f"  Mean:   {pos_scores.mean():.3f}")
print(f"  Median: {np.median(pos_scores):.3f}")
print(f"  >0.3:   {(pos_scores > 0.3).mean():.1%}")
print(f"  >0.5:   {(pos_scores > 0.5).mean():.1%}")

print(f"\nThreshold analysis:")
print(f"{'Threshold':>10} | {'TA (Recall)':>12} | {'FA Rate':>10}")
print("-" * 40)
for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    ta = (pos_scores >= thresh).mean()
    fa = (neg_scores >= thresh).mean()
    marker = " <-- good" if ta >= 0.90 and fa <= 0.05 else ""
    print(f"{thresh:>10.1f} | {ta:>12.1%} | {fa:>10.1%}{marker}")