# scripts/enroll_robust.py — Build a profile using the robust SV model
import sys
sys.path.insert(0, 'src/')

import torch
import torchaudio
from pathlib import Path

from audio_frontend import AudioFrontend
from speaker_verifier import MiniECAPATDNN

device = 'cpu'
sample_rate = 16000

# Load ROBUST SV model
print("Loading robust SV model...")
ck = torch.load('models/trained/sv_distilled_robust.pt',
               map_location=device, weights_only=False)
sv_model = MiniECAPATDNN(in_channels=40, embedding_dim=192)
sv_model.load_state_dict(ck['model_state'])
sv_model.eval()
print(f"  Epoch {ck['epoch']}, loss {ck['avg_loss']:.4f}")

frontend = AudioFrontend(sample_rate=sample_rate)

# Use the SAME enrollment clips you already recorded
clips_dir = Path("profiles/user_1_clips")
files = sorted(clips_dir.glob("utt_*.wav"))
print(f"\nUsing {len(files)} existing clips from {clips_dir}")

embeddings = []
for f in files:
    wav, sr = torchaudio.load(str(f))
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    wav = wav.mean(dim=0)
    features = frontend.extract_features_fast(wav)
    with torch.no_grad():
        emb = sv_model.extract_embedding(features)
    embeddings.append(emb)
    print(f"  {f.name}: emb norm {emb.norm(dim=1).item():.4f}")

all_emb = torch.cat(embeddings, dim=0)
profile = all_emb.mean(dim=0, keepdim=True)
profile = torch.nn.functional.normalize(profile, dim=1)

# Pairwise consistency
sims = torch.nn.functional.cosine_similarity(
    all_emb.unsqueeze(1), all_emb.unsqueeze(0), dim=2)
mask = torch.triu(torch.ones_like(sims), diagonal=1).bool()
pairwise = sims[mask]
print(f"\nProfile pairwise mean: {pairwise.mean().item():.4f}")
print(f"  (Clean model's profile pairwise was ~0.50)")

# Save with DIFFERENT name
out = Path("profiles/user_1_robust.pt")
torch.save(profile, out)
print(f"\nSaved: {out}")