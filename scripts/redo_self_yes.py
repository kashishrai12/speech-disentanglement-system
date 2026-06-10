# scripts/redo_self_yes.py — Re-record a single yes clip
import sys, argparse
sys.path.insert(0, 'src/')
import sounddevice as sd
import torchaudio, torch
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--clip_num", type=int, required=True, help="1-5")
args = parser.parse_args()

out_dir = Path("profiles/user_1_yes_clips")
out_dir.mkdir(exist_ok=True)

clip_path = out_dir / f"yes_{args.clip_num:02d}.wav"
input(f"  Re-recording {clip_path.name} — press ENTER, then say 'yes' clearly: ")
audio = sd.rec(int(2 * 16000), samplerate=16000, channels=1, dtype='float32')
sd.wait()
torchaudio.save(str(clip_path),
               torch.FloatTensor(audio.flatten()).unsqueeze(0), 16000)
print(f"  Saved {clip_path}")