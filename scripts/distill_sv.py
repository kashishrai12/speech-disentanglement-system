# distill_sv.py — Knowledge distillation: SpeechBrain teacher → MiniECAPATDNN student
# CPU-optimized for Windows. NaN-hardened.
import sys, os, time, random
sys.path.insert(0, 'src/')

import torch
import torch.nn.functional as F
import torchaudio
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from audio_frontend import AudioFrontend
from speaker_verifier import MiniECAPATDNN
from speechbrain.pretrained import EncoderClassifier

# ===== Setup =====
device = torch.device('cpu')
torch.set_num_threads(max(1, os.cpu_count() - 1))
random.seed(42)
torch.manual_seed(42)

# ===== Teacher =====
print("Loading SpeechBrain teacher...")
teacher = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="models/pretrained/ecapa_tdnn",
    run_opts={"device": "cpu"},
)
teacher.eval()
for p in teacher.mods.parameters():
    p.requires_grad = False
print("Teacher ready.")

# ===== Student =====
student = MiniECAPATDNN(in_channels=40, embedding_dim=192).to(device)
n_params = sum(p.numel() for p in student.parameters())
print(f"Student params: {n_params:,}  ({n_params/1e6:.2f}M)")

# ===== Frontend (fast batched version for training) =====
frontend = AudioFrontend()

# ===== Dataset =====
class LibriDataset(Dataset):
    def __init__(self, root, max_seconds=4.0):
        self.files = list(Path(root).rglob("*.flac"))
        random.shuffle(self.files)
        self.max_samples = int(16000 * max_seconds)
        print(f"Found {len(self.files)} utterances in {root}")
        if len(self.files) == 0:
            raise RuntimeError(f"No .flac files under {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            wav, sr = torchaudio.load(str(self.files[idx]))
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            wav = wav.mean(dim=0, keepdim=True)
            if wav.shape[1] > self.max_samples:
                start = random.randint(0, wav.shape[1] - self.max_samples)
                wav = wav[:, start:start + self.max_samples]
            else:
                wav = F.pad(wav, (0, self.max_samples - wav.shape[1]))
            wav = wav.squeeze(0)
            # FIX 3: tiny noise floor prevents zero-variance frames → NaN in pooling
            wav = wav + torch.randn_like(wav) * 1e-5
            return wav
        except Exception as e:
            print(f"Warning: failed to load {self.files[idx]}: {e}")
            return torch.zeros(self.max_samples)

dataset = LibriDataset("data/librispeech/LibriSpeech/train-clean-100")
loader = DataLoader(
    dataset, batch_size=16, shuffle=True,
    num_workers=0,   # Windows: 0 is most reliable
    pin_memory=False,
)

# ===== Resume support =====
ckpt_path = Path("models/trained/sv_distilled.pt")
log_path  = Path("models/trained/distill_log.txt")
log_path.parent.mkdir(parents=True, exist_ok=True)

# FIX 4: lower LR for stability (was 1e-3)
optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)

start_epoch = 0
if ckpt_path.exists():
    print(f"Resuming from {ckpt_path}")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Safety: don't resume from a NaN-corrupted checkpoint
    has_nan = any(torch.isnan(v).any() for v in ck['model_state'].values()
                  if torch.is_tensor(v) and v.is_floating_point())
    if has_nan:
        print("WARNING: checkpoint contains NaN — ignoring it and starting fresh.")
    else:
        student.load_state_dict(ck['model_state'])
        optimizer.load_state_dict(ck['optimizer_state'])
        start_epoch = ck['epoch']
        print(f"Resumed at epoch {start_epoch}")

NUM_EPOCHS = 4

with open(log_path, "a") as logf:
    logf.write(f"\n=== Run start: {time.ctime()} ===\n")
    logf.write(f"Student params: {n_params:,}\n")

# ===== Training loop =====
for epoch in range(start_epoch, NUM_EPOCHS):
    student.train()
    epoch_loss = 0.0
    n_batches = 0
    skipped = 0
    t0 = time.time()

    for step, wav_batch in enumerate(loader):
        wav_batch = wav_batch.to(device)  # [B, T]

        # === Teacher (no grad) ===
        with torch.no_grad():
            teacher_emb = teacher.encode_batch(wav_batch).squeeze(1)  # [B, 192]
            teacher_emb = F.normalize(teacher_emb, dim=1)

        # === Student ===
        feats = frontend.extract_features_fast(wav_batch)   # [B, 1, 40, T]
        student_emb = student.extract_embedding(feats)      # [B, 192]

        # === Cosine distillation loss ===
        loss = (1.0 - F.cosine_similarity(student_emb, teacher_emb)).mean()

        # FIX 4: NaN guard — skip bad batches instead of poisoning weights
        if not torch.isfinite(loss):
            skipped += 1
            optimizer.zero_grad()
            if skipped <= 5 or skipped % 25 == 0:
                print(f"  WARNING: non-finite loss at step {step}, skipping (total skipped: {skipped})")
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

        if step % 25 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (step + 1) * len(loader)
            msg = (f"Epoch {epoch+1}/{NUM_EPOCHS} | "
                   f"Step {step}/{len(loader)} | "
                   f"Loss {loss.item():.4f} | "
                   f"Elapsed {elapsed/60:.1f}min | "
                   f"ETA {eta/60:.1f}min")
            print(msg)
            with open(log_path, "a") as logf:
                logf.write(msg + "\n")

    scheduler.step()
    avg_loss = epoch_loss / max(1, n_batches)
    msg = (f"\n=== Epoch {epoch+1} | "
           f"Avg loss {avg_loss:.4f} | "
           f"Skipped {skipped} batches | "
           f"Time {(time.time()-t0)/60:.1f}min ===\n")
    print(msg)
    with open(log_path, "a") as logf:
        logf.write(msg)

    # FIX: only save if weights are finite
    state = student.state_dict()
    if any(torch.isnan(v).any() for v in state.values()
           if torch.is_tensor(v) and v.is_floating_point()):
        print("ERROR: model has NaN weights — NOT saving this epoch. Stopping.")
        break

    torch.save({
        'epoch': epoch + 1,
        'model_state': state,
        'optimizer_state': optimizer.state_dict(),
        'avg_loss': avg_loss,
    }, ckpt_path)
    print(f"Saved {ckpt_path}")

print("\nDistillation complete.")