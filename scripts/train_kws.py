import sys, os
sys.path.insert(0, 'src/')
os.makedirs("models/trained", exist_ok=True)

from train import train_kws_model

train_kws_model(
    data_dir="data/speech_commands",
    target_word="yes",
    noise_dir="data/musan/noise",
    rir_dir="data/rir_noises/RIRS_NOISES/simulated_rirs",
    save_path="models/trained/kws_best.pt",
    num_epochs=30,          # reduced from 50 — 30 is enough for convergence
    batch_size=128,
    lr=1e-3,
    resume_from="models/trained/kws_best_checkpoint.pt",  # picks up from epoch 4
)