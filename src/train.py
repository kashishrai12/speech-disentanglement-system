import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchaudio
import random
import numpy as np
from pathlib import Path
import sys
import os
import time
import logging
from datetime import datetime, timedelta

sys.path.insert(0, 'src/')
from audio_frontend import AudioFrontend
from kws_model import TCResNetKWS


def setup_logger(log_path):
    logger = logging.getLogger('kws_training')
    logger.setLevel(logging.INFO)
    logger.handlers = []

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console)

    fh = logging.FileHandler(log_path, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s',
                                       datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(fh)
    return logger


class KWSDataset(Dataset):
    def __init__(self, data_dir, target_word, frontend,
                 noise_dir=None, rir_dir=None,
                 augment=True, snr_range=(-5, 30)):
        self.frontend = frontend
        self.target_word = target_word.lower()
        self.augment = augment
        self.snr_range = snr_range
        self.samples = []

        # ── Load audio file paths ───────────────────────────────────
        data_path = Path(data_dir)
        for word_dir in data_path.iterdir():
            if not word_dir.is_dir():
                continue
            if word_dir.name.startswith('_') or word_dir.name.startswith('.'):
                continue
            label = 1 if word_dir.name == self.target_word else 0
            for audio_file in word_dir.glob("*.wav"):
                self.samples.append((str(audio_file), label))

        pos = sum(1 for _, l in self.samples if l == 1)
        neg = sum(1 for _, l in self.samples if l == 0)
        print(f"  Positive ('{target_word}'): {pos}")
        print(f"  Negative (other words):   {neg}")
        print(f"  Total:                    {len(self.samples)}")

        # ── Pre-load noise files into RAM (eliminates disk I/O per sample) ──
        self.noise_files = []
        self.noise_cache = []
        if noise_dir:
            noise_paths = list(Path(noise_dir).rglob("*.wav"))
            noise_paths = random.sample(noise_paths, min(100, len(noise_paths)))
            print(f"  Pre-loading {len(noise_paths)} noise files into RAM...")
            for p in noise_paths:
                try:
                    w, sr = torchaudio.load(str(p))
                    if sr != 16000:
                        w = torchaudio.functional.resample(w, sr, 16000)
                    if w.shape[0] > 1:
                        w = w.mean(0, keepdim=True)
                    self.noise_cache.append(w)
                except Exception:
                    continue
            print(f"  Noise files cached: {len(self.noise_cache)}")

        # ── RIR files (paths only — too large to cache fully) ───────
        self.rir_files = []
        if rir_dir:
            self.rir_files = list(Path(rir_dir).rglob("*.wav"))
            print(f"  RIR files loaded:         {len(self.rir_files)}")

    def _add_noise(self, waveform):
        if not self.noise_cache:
            return waveform
        try:
            noise = random.choice(self.noise_cache).clone()
            T = waveform.shape[-1]
            if noise.shape[-1] < T:
                noise = noise.repeat(1, T // noise.shape[-1] + 1)
            noise = noise[..., :T]
            snr     = random.uniform(*self.snr_range)
            sig_p   = waveform.pow(2).mean() + 1e-8
            noise_p = noise.pow(2).mean() + 1e-8
            scale   = (sig_p / (noise_p * 10 ** (snr / 10))) ** 0.5
            return waveform + scale * noise
        except Exception:
            return waveform

    def _apply_rir(self, waveform):
        # 25% of samples use RIR — reduced from 50% for speed,
        # still sufficient for 0.5m–5m distance robustness
        if not self.rir_files or random.random() > 0.75:
            return waveform
        try:
            rir, sr = torchaudio.load(random.choice(self.rir_files))
            if sr != 16000:
                rir = torchaudio.functional.resample(rir, sr, 16000)
            if rir.shape[0] > 1:
                rir = rir.mean(0, keepdim=True)
            out = torchaudio.functional.fftconvolve(waveform, rir)
            return out[..., :waveform.shape[-1]]
        except Exception:
            return waveform

    def _spec_augment(self, features):
        features = features.clone()
        _, n_mels, T = features.shape
        for _ in range(2):
            f0 = random.randint(0, max(0, n_mels - 10))
            features[:, f0:f0 + random.randint(1, 10), :] = 0
        for _ in range(2):
            max_t = max(1, int(T * 0.2))
            t0 = random.randint(0, max(0, T - max_t))
            features[:, :, t0:t0 + random.randint(1, max_t)] = 0
        return features

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            waveform, sr = torchaudio.load(path)
        except Exception:
            waveform, sr = torch.zeros(1, 16000), 16000

        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
        if waveform.shape[-1] < 16000:
            waveform = torch.nn.functional.pad(
                waveform, (0, 16000 - waveform.shape[-1]))

        if self.augment:
            if random.random() > 0.3:
                waveform = self._add_noise(waveform)
            if random.random() > 0.5:
                waveform = self._apply_rir(waveform)

        features = self.frontend.extract_features(waveform)

        if self.augment:
            features = self._spec_augment(features)

        return features, label


def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))


def train_kws_model(data_dir, target_word, noise_dir, rir_dir,
                    save_path, num_epochs=30, batch_size=128, lr=1e-3,
                    resume_from=None):

    # ── Logging ────────────────────────────────────────────────────
    log_dir  = os.path.dirname(save_path)
    log_path = os.path.join(log_dir, "training_log.txt")
    logger   = setup_logger(log_path)

    logger.info("=" * 65)
    logger.info("KWS TRAINING STARTED")
    logger.info(f"Target word : '{target_word}'")
    logger.info(f"Epochs      : {num_epochs}")
    logger.info(f"Batch size  : {batch_size}")
    logger.info(f"Save path   : {save_path}")
    logger.info(f"Log file    : {log_path}")
    logger.info("=" * 65)

    # ── Device ─────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU   : {torch.cuda.get_device_name(0)}")

    # ── Dataset ────────────────────────────────────────────────────
    logger.info("\nLoading dataset...")
    frontend = AudioFrontend()
    dataset  = KWSDataset(
        data_dir, target_word, frontend,
        noise_dir=noise_dir,
        rir_dir=rir_dir,
        augment=True
    )

    train_size = int(0.9 * len(dataset))
    val_size   = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    logger.info(f"Train: {train_size} | Val: {val_size}")

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_set,   batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)

    # ── Model ──────────────────────────────────────────────────────
    model = TCResNetKWS(n_mels=40, num_classes=2).to(device)
    logger.info(f"\nModel parameters: {model.count_parameters():,}")

    # ── Resume from checkpoint ─────────────────────────────────────
    start_epoch  = 0
    best_val_acc = 0.0
    checkpoint_path = save_path.replace('.pt', '_checkpoint.pt')

    # ── Loss + Optimizer ───────────────────────────────────────────
    class FocalLoss(nn.Module):
        def __init__(self, gamma=2.0, alpha=0.25):
            super().__init__()
            self.gamma = gamma
            self.alpha = alpha
        def forward(self, logits, targets):
            ce = nn.functional.cross_entropy(logits, targets, reduction='none')
            p  = torch.exp(-ce)
            return (self.alpha * (1 - p) ** self.gamma * ce).mean()

    criterion = FocalLoss(gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    if resume_from and os.path.exists(resume_from):
        logger.info(f"\nResuming from checkpoint: {resume_from}")
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer'])  # must load BEFORE scheduler
        start_epoch  = ckpt['epoch']
        best_val_acc = ckpt['best_val_acc']
        logger.info(f"Resuming from epoch {start_epoch + 1}, "
                    f"best acc so far = {best_val_acc:.4f}")
    else:
        logger.info("\nStarting fresh — no checkpoint found.")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, last_epoch=start_epoch - 1
    )

    # ── Training loop ──────────────────────────────────────────────
    logger.info("\nStarting training...\n")
    epoch_times = []

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()

        # -- Train --
        model.train()
        train_loss    = 0.0
        train_correct = 0
        train_total   = 0
        batches       = len(train_loader)

        for batch_idx, (features, labels) in enumerate(train_loader):
            features = features.to(device)
            labels   = labels.to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss    += loss.item()
            preds          = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += labels.size(0)

            # Live batch progress on same line
            batch_acc = train_correct / train_total
            print(
                f"\r  Epoch {epoch+1:3d}/{num_epochs} | "
                f"Batch {batch_idx+1:4d}/{batches} | "
                f"Loss: {loss.item():.4f} | "
                f"Train acc: {batch_acc:.3f}",
                end='', flush=True
            )

        print()  # newline after epoch completes
        scheduler.step()

        # -- Validate --
        model.eval()
        val_correct = 0
        val_total   = 0
        val_tp = val_fp = val_fn = 0

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels   = labels.to(device)
                preds    = model(features).argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)
                val_tp += ((preds == 1) & (labels == 1)).sum().item()
                val_fp += ((preds == 1) & (labels == 0)).sum().item()
                val_fn += ((preds == 0) & (labels == 1)).sum().item()

        # -- Metrics --
        epoch_time  = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_loss    = train_loss / batches
        train_acc   = train_correct / train_total
        val_acc     = val_correct / val_total
        precision   = val_tp / (val_tp + val_fp + 1e-8)
        recall      = val_tp / (val_tp + val_fn + 1e-8)
        f1          = 2 * precision * recall / (precision + recall + 1e-8)
        avg_epoch_t = sum(epoch_times) / len(epoch_times)
        remaining   = avg_epoch_t * (num_epochs - epoch - 1)
        eta         = datetime.now() + timedelta(seconds=remaining)

        saved = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            saved = "  [BEST SAVED]"

        # Save checkpoint every epoch for crash recovery
        torch.save({
            'epoch':        epoch + 1,
            'model_state':  model.state_dict(),
            'best_val_acc': best_val_acc,
            'optimizer':    optimizer.state_dict(),
        }, checkpoint_path)

        logger.info(
            f"Epoch {epoch+1:3d}/{num_epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Train: {train_acc:.4f} | "
            f"Val: {val_acc:.4f} | "
            f"P: {precision:.3f} R: {recall:.3f} F1: {f1:.3f} | "
            f"Time: {format_time(epoch_time)} | "
            f"ETA: {eta.strftime('%H:%M:%S')} "
            f"({format_time(remaining)} left)"
            f"{saved}"
        )

    logger.info("\n" + "=" * 65)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Best val accuracy : {best_val_acc:.4f}")
    logger.info(f"Total time        : {format_time(sum(epoch_times))}")
    logger.info(f"Model saved to    : {save_path}")
    logger.info("=" * 65)