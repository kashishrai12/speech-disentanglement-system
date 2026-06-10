import os, torch
from huggingface_hub import hf_hub_download

os.makedirs("models/pretrained/ecapa_tdnn", exist_ok=True)

print("Downloading pretrained ECAPA-TDNN from HuggingFace...")

# Download the model files directly — no SpeechBrain needed
files_to_download = [
    "embedding_model.ckpt",
    "mean_var_norm_emb.ckpt",
    "hyperparams.yaml",
]

for filename in files_to_download:
    print(f"  Downloading {filename}...")
    path = hf_hub_download(
        repo_id="speechbrain/spkrec-ecapa-voxceleb",
        filename=filename,
        local_dir="models/pretrained/ecapa_tdnn"
    )
    print(f"  Saved to {path}")

# Load and re-save just the embedding model weights
print("\nExtracting embedding weights...")
ckpt = torch.load(
    "models/pretrained/ecapa_tdnn/embedding_model.ckpt",
    map_location="cpu"
)

# The checkpoint is a dict of tensors — save it directly
torch.save(ckpt, "models/pretrained/ecapa_full_weights.pt")
print("Done — saved to models/pretrained/ecapa_full_weights.pt")

# Verify
print(f"\nCheckpoint keys (first 5): {list(ckpt.keys())[:5]}")
print(f"Total layers: {len(ckpt.keys())}")