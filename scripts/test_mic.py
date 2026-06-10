# scripts/test_mic.py — quick mic test
import sounddevice as sd
import numpy as np

print("Recording 3 seconds — speak now...")
audio = sd.rec(int(3 * 16000), samplerate=16000, channels=1, dtype='float32')
sd.wait()
audio = audio.flatten()

rms = np.sqrt(np.mean(audio ** 2))
peak = np.abs(audio).max()
print(f"\nRMS:  {rms:.4f}   (good: 0.02–0.10)")
print(f"Peak: {peak:.4f}   (good: 0.3–0.8)")

if rms < 0.005:
    print("⚠️  Too quiet — increase mic input level in Windows sound settings")
elif peak > 0.99:
    print("⚠️  Clipping — lower input or speak softer")
elif rms < 0.02:
    print("ℹ️  A bit quiet but usable. Try a slightly louder voice.")
else:
    print("✅ Levels look good. Proceed with enrollment.")