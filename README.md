# VoxGate — Speaker-Specific Custom Word Detection

A real-time cascaded keyword spotting and speaker verification system that triggers **only when a specific enrolled speaker says a specific target word**, even in noisy environments. Built on CPU, designed for edge deployment.

> **Origin:** Designed for the Aural-04 hackathon problem statement. Submitted late, then completed and refined as a personal project.

---

## What it does

Microphone → [VAD + Mel] → KWS ("yes"?) ──no──→ ignore 

│ yes 
↓ 

SV (user_1?) ────no──→ ignore 

│ yes 
↓ 

Decision Fusion → 🟢 ACCEPT 


VoxGate listens continuously, but only fires when **both** stages agree: the keyword spotter recognizes the target word AND the speaker verifier matches the enrolled voice. The two-stage cascade is critical — KWS alone false-accepts anyone saying "yes", SV alone false-accepts the right person saying anything.

---

## Final Results

Tested against the original KPI spec on a Windows 11 laptop, CPU only (no GPU).

| KPI | Target | Result | Status |
|---|---|---|---|
| **Total parameters** | < 3M | **1.42M** | ✅ 53% under budget |
| **xRT** (real-time factor) | < 0.20 | **0.090** | ✅ 55% under target |
| **False Accepts/hour** | < 1 | **0 across 100 impostor tests** | ✅ |
| **True Accept @ 30dB SNR** | ≥ 99% | 60% | ⚠️ partial |
| **True Accept @ 10dB SNR** | ≥ 90% | 60% | ⚠️ partial |
| **True Accept @ 0dB SNR** | ≥ 90% | 47% | ⚠️ partial |
| **True Accept @ -5dB SNR** | ≥ 90% | 33% | ⚠️ partial |

**Residual:** TA falls short of the spec at all noise levels. This is a fundamental capacity limit of distilling a 6M-parameter teacher into a 1.2M student under a hard parameter budget. Documented as a known trade-off.

---

## Architecture

## Home Page
![ta_vs_snr_clean_vs_robust)](images/ta_vs_snr_clean_vs_robust.png)

## Login Page
![Login Page](images/login.png)

## Dashboard
![Dashboard](images/dashboard.png)
### Stage 1 — Audio Frontend
- Pre-emphasis (high-frequency boost for consonant clarity)
- 40-band log-Mel filterbank, 25 ms window, 10 ms hop
- Per-utterance CMVN normalization
- **Vectorized PyTorch implementation** (~3 ms per 1-sec window)

### Stage 2 — Keyword Spotting (KWS)
- **TC-ResNet** with depthwise-separable convolutions and multi-head attention pooling
- **193K parameters**
- Trained on Google Speech Commands v2 (105K utterances, 35 words) with **Focal Loss** (γ=2.0) to handle class imbalance
- Achieves 93.4% recall at 0.8% false-alarm rate (threshold tuned post-training)

### Stage 3 — Speaker Verification (SV)
- **Mini ECAPA-TDNN** — Res2Net blocks (scale=4), attentive statistics pooling, 192-dim L2-normalized embeddings
- **1.23M parameters**
- Trained via **knowledge distillation** from SpeechBrain ECAPA-TDNN (6M params, trained on VoxCeleb 1+2) → cosine-similarity matching loss
- **Noise-augmented:** MUSAN noise (-5 to +20 dB SNR, 50% of batches) + RIR convolution (25% of batches) applied during distillation for noise invariance

### Stage 4 — Decision Fusion
- **Geometric mean fusion** of KWS probability and SV cosine similarity
- Three-gate decision: KWS prob ≥ 0.40 AND SV cosine ≥ 0.60 AND combined ≥ 0.55
- Score smoothing over 3-frame window
- 500ms cooldown to prevent double-triggers from one utterance

### Stage 5 — Enrollment
- 7 short utterances of the target word from the speaker
- Mean embedding stored as L2-normalized 192-dim vector
- Pairwise consistency check warns on low-quality enrollment

---

## Key Design Decisions

| Decision | Why |
|---|---|
| **Distillation over from-scratch SV training** | CPU-only training of a 1.2M ECAPA on LibriSpeech-100 would take days; distillation from a VoxCeleb-trained teacher gets there in ~14 hours |
| **Cascaded inference (KWS gates SV)** | SV is the expensive stage; only run it when the KWS already thinks it heard the word. Cuts compute by ~4x |
| **Silence skipping in frontend** | Audio with peak amplitude < 0.01 skips the entire pipeline. Drops xRT from 0.27 to 0.09 |
| **Cosine similarity, not raw distance** | L2-normalized embeddings + cosine is the standard for speaker verification; behaves well across noise levels |
| **Noise-augmented distillation, not noise-augmented from-scratch** | Two-stage: first distill to clean baseline (loss 0.29), then re-distill with augmentation (loss 0.40). The clean baseline acts as a warm start |
| **Two separate models + profiles** | Clean and robust models produce different embedding geometries; each needs its own enrollment profile. Robust model used by default |

---

## Repository Structure
```text
speech_disentanglement/
├── src/
│   ├── audio_frontend.py        # DSP + log-Mel features (two paths: inference & training)
│   ├── kws_model.py             # TC-ResNet KWS architecture
│   ├── speaker_verifier.py      # Mini ECAPA-TDNN SV architecture + ArcFace loss
│   ├── enrollment.py            # Speaker enrollment & verification logic
│   ├── decision_fusion.py       # Cascaded decision logic with smoothing & cooldown
│   ├── inference_engine.py      # Real-time streaming inference engine
│   └── train.py                 # KWS training script
├── scripts/
│   ├── train_kws.py             # Train KWS model
│   ├── tune_threshold.py        # Find optimal KWS threshold post-training
│   ├── distill_sv.py            # Distill SpeechBrain ECAPA → Mini ECAPA (clean)
│   ├── distill_sv_robust.py     # Same, with MUSAN + RIR augmentation
│   ├── enroll.py                # Record 7 utterances → speaker profile
│   ├── enroll_robust.py         # Build profile using robust SV model
│   ├── run_inference.py         # Live mic demo with xRT measurement
│   ├── test_impostor.py         # Quick impostor rejection sanity check
│   ├── eval_robust.py           # Per-SNR TA/FA evaluation
│   └── probe_robust_scores.py   # Score distribution analysis for threshold tuning
├── evaluate.py                   # Full formal KPI evaluation across SNR levels
├── models/
│   ├── pretrained/              # SpeechBrain ECAPA teacher (downloaded on first run)
│   └── trained/
│       ├── kws_best.pt          # Trained KWS model
│       ├── sv_distilled.pt      # Clean-audio distilled SV (kept as baseline)
│       └── sv_distilled_robust.pt  # Noise-augmented SV (default for inference)
├── profiles/
│   ├── user_1.pt                # Profile for clean SV
│   ├── user_1_robust.pt         # Profile for robust SV (default)
│   └── user_1_yes_clips/        # Reference recordings
└── data/                        # Datasets (gitignored — see Setup)
```

---

## Setup

### Requirements
- Python 3.10+
- 8+ GB RAM
- ~30 GB disk space for datasets

### Install
```bash
conda create -n speech_env python=3.10
conda activate speech_env
pip install torch torchaudio numpy scipy sounddevice speechbrain
```

### Datasets
- **LibriSpeech train-clean-100** (~6.3 GB) → `data/librispeech/LibriSpeech/train-clean-100/`
- **Google Speech Commands v2** (~2.4 GB) → `data/speech_commands/`
- **MUSAN** (~11 GB) → `data/musan/`
- **RIR Noises (OpenSLR-28)** (~1 GB) → `data/rir_noises/`

---

## How to Run

### 1. Train KWS (~6 hours CPU)
```bash
python scripts/train_kws.py
python scripts/tune_threshold.py   # Pick optimal threshold
```

### 2. Distill SV (~14-16 hours each)
```bash
# Step 1: Clean distillation
python scripts/distill_sv.py

# Step 2: Noise-augmented distillation (warm-started from clean)
python scripts/distill_sv_robust.py
```

### 3. Enroll your voice
```bash
python scripts/enroll_robust.py   # Records 7 short clips
```

### 4. Run the live demo
```bash
python scripts/run_inference.py
```

Speak "yes" → 🟢 ACCEPT. Someone else says "yes" → 🔴 reject. Wrong word → silent.

### 5. Evaluate against KPIs
```bash
python evaluate.py
```

---

## Lessons Learned

1. **CPU-only deep learning is harder than it looks.** A 14-hour training run for a 1.2M-param model is not a small commitment. Smoke tests before launch saved me from wasted overnight runs at least twice.

2. **Distillation has a capacity ceiling.** Compressing 6M parameters into 1.2M loses ~0.15 cosine similarity gap between same-speaker and different-speaker pairs. No amount of training fixes this — it's an architectural limit.

3. **Frontend mismatch destroys distilled models.** The student was trained on clean log-Mel features; my first inference path applied spectral subtraction and VAD, distorting the input. SV cosine scores collapsed until I switched to the same vectorized frontend for both training and inference.

4. **Noise augmentation is a trade.** The robust model improved TA at 0 dB from 13% to 47%, but only after I also re-enrolled the speaker against the robust model. Mismatched profiles destroyed the gain initially.

5. **Score normalization mistakes are silent.** A single line `sv_score = (sv_similarity + 1) / 2` mapped cosine similarity from [-1,1] to [0,1] — but lifted all impostor scores above 0.5, making the fusion threshold meaningless. Removing this one line dropped FA from 15% to 0%.

6. **Real-time means watching your hot path.** The biggest xRT win wasn't model optimization — it was skipping feature extraction entirely on silent chunks. One `numpy.max` call dropped xRT from 0.27 to 0.09.

---

## Future Work

- Quantization (int8) to halve the parameter memory footprint
- Per-SNR threshold tuning (adaptive based on estimated noise level)
- Larger student (2.5M params, still under budget) with longer noise-augmented distillation — likely closes most of the TA gap
- ONNX export for cross-platform deployment

---

## Acknowledgments

- **SpeechBrain** for the ECAPA-TDNN teacher model
- **Google Research** for Speech Commands v2
- **OpenSLR** for LibriSpeech, MUSAN, and RIR Noises
- Original ECAPA-TDNN paper: Desplanques, Thienpondt, Demuynck (2020)

---

## License

MIT 
