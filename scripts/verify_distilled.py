# scripts/verify_distilled.py — sanity check the distilled SV model
import sys
sys.path.insert(0, 'src/')

import torch
from speaker_verifier import MiniECAPATDNN

m = MiniECAPATDNN(in_channels=40, embedding_dim=192)
ck = torch.load('models/trained/sv_distilled.pt', map_location='cpu', weights_only=False)
m.load_state_dict(ck['model_state'])
m.eval()

print(f"Loaded. Epoch: {ck['epoch']}, Avg Loss: {ck['avg_loss']:.4f}")

# Forward-pass test
x = torch.randn(1, 1, 40, 401)
with torch.no_grad():
    emb = m.extract_embedding(x)
print(f"Embedding shape: {emb.shape}")
print(f"Embedding norm:  {emb.norm(dim=1).item():.4f}  (should be ~1.0)")

# NaN check
has_nan = any(torch.isnan(v).any() for v in m.state_dict().values()
              if torch.is_tensor(v) and v.is_floating_point())
print(f"Has NaN weights: {has_nan}  (should be False)")

print("\nOK" if not has_nan and abs(emb.norm(dim=1).item() - 1.0) < 0.01 else "\nFAIL")