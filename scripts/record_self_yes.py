# scripts/record_self_yes.py
import sys
sys.path.insert(0,'src/')
import sounddevice as sd
import torchaudio, torch
from pathlib import Path

out_dir = Path("profiles/user_1_yes_clips")
out_dir.mkdir(exist_ok=True)

print("Recording 5 clips of you saying 'yes' (2s each)...")
for i in range(5):
    input(f"  Clip {i+1}/5 — press ENTER, then say 'yes' clearly: ")
    audio = sd.rec(int(2 * 16000), samplerate=16000, channels=1, dtype='float32')
    sd.wait()
    torchaudio.save(str(out_dir / f"yes_{i+1:02d}.wav"),
                   torch.FloatTensor(audio.flatten()).unsqueeze(0), 16000)
    print(f"    saved.")
print(f"\nDone. Files in {out_dir}")