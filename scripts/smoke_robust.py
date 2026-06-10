import sys; sys.path.insert(0,'src/')
import torch
import random
random.seed(0)

# Import & set up everything from the main script
exec(open('scripts/distill_sv_robust.py').read().split("# ===== Dataset")[0])

# Test augmentation
wav_batch = torch.randn(4, 64000) * 0.1
print(f"\nOriginal max: {wav_batch.abs().max():.4f}")

aug = augment_batch(wav_batch)
print(f"Augmented shape: {aug.shape}, max: {aug.abs().max():.4f}")
print(f"Augmented is finite: {torch.isfinite(aug).all().item()}")

# Run a single teacher+student forward
with torch.no_grad():
    t_emb = teacher.encode_batch(aug).squeeze(1)
    t_emb = torch.nn.functional.normalize(t_emb, dim=1)
print(f"Teacher embedding shape: {t_emb.shape}")

feats = frontend.extract_features_fast(aug)
s_emb = student.extract_embedding(feats)
print(f"Student embedding shape: {s_emb.shape}")

loss = (1 - torch.nn.functional.cosine_similarity(s_emb, t_emb)).mean()
print(f"Loss: {loss.item():.4f}")
print("OK")