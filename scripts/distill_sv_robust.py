# distill_sv_robust.py — Noise-augmented distillation: SpeechBrain → MiniECAPATDNN
# Outputs: models/trained/sv_distilled_robust.pt (does not overwrite original)
import sys, os, time, random
sys.path.insert(0, 'src/')

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
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
np.random.seed(42)

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

frontend = AudioFrontend()

# ===== Pre-load noise & RIR file lists =====
print("Indexing noise and RIR files...")
musan_noise_root = Path("data/musan")
noise_files = list(musan_noise_root.rglob("*.wav"))
print(f"  MUSAN files: {len(noise_files)}")

rir_root = Path("data/rir_noises")
rir_files = list(rir_root.rglob("*.wav"))
print(f"  RIR files:   {len(rir_files)}")

# ===== Audio augmentation =====
# Pre-cache a subset of noise files in RAM for speed (avoids disk I/O per batch)
NOISE_CACHE_SIZE = 100
RIR_CACHE_SIZE = 50

def load_audio(path, target_sr=16000, max_samples=64000):
    try:
        wav, sr = torchaudio.load(str(path))
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        wav = wav.mean(dim=0)
        if len(wav) > max_samples:
            start = random.randint(0, len(wav) - max_samples)
            wav = wav[start:start + max_samples]
        return wav
    except Exception:
        return None

print(f"Pre-loading {NOISE_CACHE_SIZE} noise files into RAM...")
noise_cache = []
random.shuffle(noise_files)
for f in noise_files[:NOISE_CACHE_SIZE * 2]:
    w = load_audio(f)
    if w is not None and len(w) > 8000:
        noise_cache.append(w)
    if len(noise_cache) >= NOISE_CACHE_SIZE:
        break
print(f"  Cached {len(noise_cache)} noise clips.")

print(f"Pre-loading {RIR_CACHE_SIZE} RIR files into RAM...")
rir_cache = []
random.shuffle(rir_files)
for f in rir_files[:RIR_CACHE_SIZE * 2]:
    w = load_audio(f, max_samples=8000)  # RIRs are short
    if w is not None:
        # Normalize RIR to prevent gain explosion
        w = w / (w.abs().max() + 1e-8)
        rir_cache.append(w)
    if len(rir_cache) >= RIR_CACHE_SIZE:
        break
print(f"  Cached {len(rir_cache)} RIR clips.")


def add_noise(speech, noise, snr_db):
    """Add noise at target SNR. speech and noise: [T] tensors."""
    if len(noise) < len(speech):
        reps = (len(speech) // len(noise)) + 1
        noise = noise.repeat(reps)
    # Random crop noise to match speech length
    start = random.randint(0, max(0, len(noise) - len(speech)))
    noise = noise[start:start + len(speech)]

    speech_rms = (speech ** 2).mean().sqrt() + 1e-8
    noise_rms = (noise ** 2).mean().sqrt() + 1e-8
    target_noise_rms = speech_rms / (10 ** (snr_db / 20))
    return speech + noise * (target_noise_rms / noise_rms)


def apply_rir(speech, rir):
    """Convolve speech with RIR. Keep same length."""
    rir = rir / (rir.abs().max() + 1e-8)
    out = F.conv1d(speech.view(1, 1, -1), rir.view(1, 1, -1), padding=len(rir) // 2)
    out = out.squeeze()[:len(speech)]
    # Normalize back to original loudness
    out = out * (speech.abs().max() / (out.abs().max() + 1e-8))
    return out


def augment_batch(wav_batch):
    """Apply random noise and RIR augmentation to a batch [B, T]."""
    augmented = []
    for wav in wav_batch:
        # 25% chance of RIR (simulates distance/reverb)
        if rir_cache and random.random() < 0.25:
            wav = apply_rir(wav, random.choice(rir_cache))

        # 50% chance of noise (simulates background)
        if noise_cache and random.random() < 0.5:
            snr_db = random.uniform(-5, 20)   # cover the KPI SNR range
            wav = add_noise(wav, random.choice(noise_cache), snr_db)

        # Re-normalize to prevent clipping in extreme cases
        peak = wav.abs().max()
        if peak > 0.99:
            wav = wav * (0.99 / peak)
        augmented.append(wav)
    return torch.stack(augmented)


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
            wav = wav + torch.randn_like(wav) * 1e-5
            return wav
        except Exception as e:
            print(f"Warning: failed to load {self.files[idx]}: {e}")
            return torch.zeros(self.max_samples)


dataset = LibriDataset("data/librispeech/LibriSpeech/train-clean-100")
loader = DataLoader(
    dataset, batch_size=16, shuffle=True,
    num_workers=0, pin_memory=False,
)

# ===== Save paths (DIFFERENT from original) =====
ckpt_path = Path("models/trained/sv_distilled_robust.pt")
log_path  = Path("models/trained/distill_robust_log.txt")
log_path.parent.mkdir(parents=True, exist_ok=True)

# ===== Optimizer / scheduler =====
# Slightly lower LR for noise-augmented training (noisier gradient)
optimizer = torch.optim.AdamW(student.parameters(), lr=2e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

# ===== Warm start from existing clean-distilled model =====
# This dramatically speeds up convergence — student starts from a
# good clean-audio baseline, only needs to learn noise invariance.
clean_ckpt = Path("models/trained/sv_distilled.pt")
start_epoch = 0
if ckpt_path.exists():
    print(f"Resuming from {ckpt_path}")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    has_nan = any(torch.isnan(v).any() for v in ck['model_state'].values()
                  if torch.is_tensor(v) and v.is_floating_point())
    if has_nan:
        print("WARNING: existing checkpoint has NaN, ignoring.")
    else:
        student.load_state_dict(ck['model_state'])
        optimizer.load_state_dict(ck['optimizer_state'])
        start_epoch = ck['epoch']
        print(f"Resumed at epoch {start_epoch}")
elif clean_ckpt.exists():
    print(f"Warm-starting from clean model: {clean_ckpt}")
    ck = torch.load(clean_ckpt, map_location=device, weights_only=False)
    student.load_state_dict(ck['model_state'])
    print(f"  Started from clean epoch {ck.get('epoch', '?')}, loss {ck.get('avg_loss', 0):.4f}")

NUM_EPOCHS = 5   # warm-started, fewer epochs needed

with open(log_path, "a") as logf:
    logf.write(f"\n=== Robust run: {time.ctime()} ===\n")
    logf.write(f"Student params: {n_params:,}\n")
    logf.write(f"Noise cache: {len(noise_cache)}  RIR cache: {len(rir_cache)}\n")

# ===== Training loop =====
for epoch in range(start_epoch, NUM_EPOCHS):
    student.train()
    epoch_loss = 0.0
    n_batches = 0
    skipped = 0
    t0 = time.time()

    for step, wav_batch in enumerate(loader):
        wav_batch = wav_batch.to(device)

        # === Augmentation: apply noise+RIR ===
        wav_aug = augment_batch(wav_batch)

        # === Teacher sees CLEAN audio — target embeddings represent the speaker ===
        with torch.no_grad():
            teacher_emb = teacher.encode_batch(wav_batch).squeeze(1)
            teacher_emb = F.normalize(teacher_emb, dim=1)

        # === Student sees NOISY audio — must learn invariance ===
        feats = frontend.extract_features_fast(wav_aug)
        student_emb = student.extract_embedding(feats)

        # === Cosine distillation loss ===
        loss = (1.0 - F.cosine_similarity(student_emb, teacher_emb)).mean()

        if not torch.isfinite(loss):
            skipped += 1
            optimizer.zero_grad()
            if skipped <= 5 or skipped % 25 == 0:
                print(f"  WARNING: non-finite loss at step {step}, skipping (total: {skipped})")
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
           f"Skipped {skipped} | "
           f"Time {(time.time()-t0)/60:.1f}min ===\n")
    print(msg)
    with open(log_path, "a") as logf:
        logf.write(msg)

    state = student.state_dict()
    if any(torch.isnan(v).any() for v in state.values()
           if torch.is_tensor(v) and v.is_floating_point()):
        print("ERROR: NaN weights — NOT saving. Stopping.")
        break

    torch.save({
        'epoch': epoch + 1,
        'model_state': state,
        'optimizer_state': optimizer.state_dict(),
        'avg_loss': avg_loss,
    }, ckpt_path)
    print(f"Saved {ckpt_path}")

print("\nRobust distillation complete.")