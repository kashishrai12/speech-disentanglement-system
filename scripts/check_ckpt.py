import sys
sys.path.insert(0, 'src/')
import torch
from speaker_verifier import MiniECAPATDNN

ck = torch.load('models/trained/sv_distilled_robust.pt',
               map_location='cpu', weights_only=False)
print("Keys in checkpoint:", list(ck.keys()))
print(f"Epoch: {ck['epoch']}")
print(f"Avg loss: {ck['avg_loss']:.4f}")

m = MiniECAPATDNN(in_channels=40, embedding_dim=192)
m.load_state_dict(ck['model_state'])
print("Model loads cleanly.")

has_nan = any(torch.isnan(v).any() for v in ck['model_state'].values()
              if torch.is_tensor(v) and v.is_floating_point())
print(f"Has NaN weights: {has_nan}")