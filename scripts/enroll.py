# scripts/enroll.py — Record N utterances and create a speaker profile (resumable)
import sys, os, argparse
sys.path.insert(0, 'src/')

import torch
import torchaudio
import numpy as np
import sounddevice as sd
from pathlib import Path

from audio_frontend import AudioFrontend
from speaker_verifier import MiniECAPATDNN


def record_clip(seconds: float, sample_rate: int = 16000) -> np.ndarray:
    print(f"  Recording {seconds}s... speak now.")
    sd.default.samplerate = sample_rate
    sd.default.channels = 1
    audio = sd.rec(int(seconds * sample_rate), dtype='float32')
    sd.wait()
    print("  Done.")
    return audio.flatten()


def record_with_retry(seconds, sample_rate, prompt, clip_path):
    """Record, check quality, ask user to retry if bad. Saves on accept."""
    while True:
        print(f'\n  Say: "{prompt}"')
        input("  Press ENTER to start recording...")
        audio = record_clip(seconds, sample_rate)

        rms = float(np.sqrt(np.mean(audio ** 2)))
        peak = float(np.abs(audio).max())
        print(f"  RMS: {rms:.4f}  Peak: {peak:.4f}")

        warn = []
        if rms < 0.005:
            warn.append("very quiet")
        if rms > 0.20:
            warn.append("very loud")
        if peak > 0.98:
            warn.append("clipping")

        if warn:
            print(f"  ⚠️  WARNING: {', '.join(warn)}.")
            choice = input("  Redo this clip? [Y/n]: ").strip().lower()
            if choice in ("", "y", "yes"):
                continue

        # Save and return
        torchaudio.save(str(clip_path),
                       torch.FloatTensor(audio).unsqueeze(0),
                       sample_rate)
        print(f"  Saved {clip_path}")
        return audio


def main():
    parser = argparse.ArgumentParser(description="Enroll a speaker for VoxGate (resumable)")
    parser.add_argument("--speaker_id", type=str, default="user_1")
    parser.add_argument("--num_utterances", type=int, default=7)
    parser.add_argument("--clip_seconds", type=float, default=2.0)
    parser.add_argument("--sv_model_path", type=str, default="models/trained/sv_distilled.pt")
    parser.add_argument("--profile_dir", type=str, default="profiles")
    parser.add_argument("--fresh", action="store_true",
                       help="Delete existing clips and start over")
    parser.add_argument("--redo", type=int, default=None,
                       help="Re-record just this clip number (1-indexed)")
    args = parser.parse_args()

    device = 'cpu'
    sample_rate = 16000

    # ===== Load SV model =====
    print(f"Loading SV model from {args.sv_model_path}...")
    ck = torch.load(args.sv_model_path, map_location=device, weights_only=False)
    state = ck['model_state'] if isinstance(ck, dict) and 'model_state' in ck else ck
    sv_model = MiniECAPATDNN(in_channels=40, embedding_dim=192)
    sv_model.load_state_dict(state)
    sv_model.eval()
    print("  SV model loaded.")

    frontend = AudioFrontend(sample_rate=sample_rate)

    profile_dir = Path(args.profile_dir)
    profile_dir.mkdir(exist_ok=True)
    clips_dir = profile_dir / f"{args.speaker_id}_clips"
    clips_dir.mkdir(exist_ok=True)

    # ===== Fresh start? =====
    if args.fresh:
        for f in clips_dir.glob("utt_*.wav"):
            f.unlink()
        print(f"Deleted existing clips in {clips_dir}")

    prompts = [
        "yes",
        "yes please yes",
        "yeah yes okay",
        "yes I agree yes",
        "yes hello yes",
        "okay sure yes yes",
        "yes yes yes yes yes",
    ]

    # ===== Recording phase =====
    if args.redo is not None:
        # Re-record only one specific clip
        i = args.redo
        if i < 1 or i > args.num_utterances:
            print(f"--redo {i} out of range [1, {args.num_utterances}]")
            return
        clip_path = clips_dir / f"utt_{i:02d}.wav"
        prompt = prompts[(i - 1) % len(prompts)]
        print(f"\n--- Re-recording utterance {i} ---")
        record_with_retry(args.clip_seconds, sample_rate, prompt, clip_path)
        print(f"\nRe-record done. Now run without --redo to rebuild the profile.")
        return

    # Normal/resume: check what's already saved
    existing = sorted(clips_dir.glob("utt_*.wav"))
    if existing:
        print(f"\nFound {len(existing)} existing clip(s). Resuming.")
        print("(Use --fresh to start over, or --redo N to redo a specific clip.)")

    for i in range(1, args.num_utterances + 1):
        clip_path = clips_dir / f"utt_{i:02d}.wav"
        if clip_path.exists():
            print(f"\n--- Utterance {i}/{args.num_utterances} (already recorded, skipping) ---")
            continue

        prompt = prompts[(i - 1) % len(prompts)]
        print(f"\n--- Utterance {i}/{args.num_utterances} ---")
        record_with_retry(args.clip_seconds, sample_rate, prompt, clip_path)

    # ===== Embedding extraction =====
    print("\n=== Extracting embeddings ===")
    embeddings = []
    for i in range(1, args.num_utterances + 1):
        clip_path = clips_dir / f"utt_{i:02d}.wav"
        wav, sr = torchaudio.load(str(clip_path))
        if sr != sample_rate:
            wav = torchaudio.functional.resample(wav, sr, sample_rate)
        wav = wav.mean(dim=0)  # mono, [T]
        features = frontend.extract_features_fast(wav)
        with torch.no_grad():
            emb = sv_model.extract_embedding(features)
        embeddings.append(emb)
        print(f"  utt_{i:02d}: emb norm {emb.norm(dim=1).item():.4f}")

    # ===== Build profile =====
    all_emb = torch.cat(embeddings, dim=0)
    profile = all_emb.mean(dim=0, keepdim=True)
    profile = torch.nn.functional.normalize(profile, dim=1)

    # Pairwise consistency
    sims = torch.nn.functional.cosine_similarity(
        all_emb.unsqueeze(1), all_emb.unsqueeze(0), dim=2)
    mask = torch.triu(torch.ones_like(sims), diagonal=1).bool()
    pairwise = sims[mask]
    print(f"\n=== Profile quality ===")
    print(f"  Pairwise cosine similarity:")
    print(f"    Mean: {pairwise.mean().item():.4f}")
    print(f"    Min:  {pairwise.min().item():.4f}")
    print(f"    Max:  {pairwise.max().item():.4f}")

    # Show per-clip outlier scores so you know which to --redo
    print(f"\n  Per-clip avg similarity to others:")
    for i in range(len(embeddings)):
        others = torch.cat([embeddings[j] for j in range(len(embeddings)) if j != i], dim=0)
        avg_sim = torch.nn.functional.cosine_similarity(
            embeddings[i], others.mean(dim=0, keepdim=True)).item()
        marker = "  ← outlier" if avg_sim < pairwise.mean() - 0.10 else ""
        print(f"    utt_{i+1:02d}: {avg_sim:.4f}{marker}")

    if pairwise.mean() > 0.75:
        print("\n  ✅  Excellent consistency.")
    elif pairwise.mean() > 0.60:
        print("\n  ✅  Good consistency.")
    elif pairwise.mean() > 0.45:
        print("\n  ⚠️   Borderline. Consider --redo N for outliers above.")
    else:
        print("\n  ❌  Poor consistency. Re-enroll in a quieter setting.")

    # Save profile
    profile_path = profile_dir / f"{args.speaker_id}.pt"
    torch.save(profile, profile_path)
    print(f"\n✅ Profile saved: {profile_path}")


if __name__ == "__main__":
    main()