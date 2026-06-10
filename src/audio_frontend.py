# audio_frontend.py
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from scipy.signal import lfilter

class AudioFrontend:
    """
    Production audio frontend pipeline:
    1. Voice Activity Detection (VAD) - silences non-speech frames
    2. Pre-emphasis filter - boosts high frequencies (speech clarity)
    3. Noise reduction via spectral subtraction
    4. Log-Mel filterbank feature extraction
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 40,
        win_length_ms: int = 25,
        hop_length_ms: int = 10,
        n_fft: int = 512,
        pre_emphasis: float = 0.97,
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.win_length = int(sample_rate * win_length_ms / 1000)  # 400 samples
        self.hop_length = int(sample_rate * hop_length_ms / 1000)  # 160 samples
        self.n_fft = n_fft
        self.pre_emphasis = pre_emphasis

        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=n_mels,
            f_min=20,
            f_max=8000,
            power=2.0,
            normalized=True,
        )
        self.amplitude_to_db = T.AmplitudeToDB(stype='power', top_db=80)

        # WebRTC VAD — runs in 10ms frames
        try:
            import webrtcvad
            self.vad = webrtcvad.Vad(2)  # aggressiveness 0-3
        except ImportError:
            self.vad = None  # fallback: energy VAD

    def apply_pre_emphasis(self, signal: np.ndarray) -> np.ndarray:
        """Boost high frequencies to improve consonant detection."""
        return lfilter([1, -self.pre_emphasis], [1], signal)

    def spectral_subtraction(self, signal: np.ndarray, noise_frames: int = 10) -> np.ndarray:
        """
        Simple spectral subtraction for noise reduction.
        Estimate noise PSD from first N frames (assumed to be noise-only).
        This is a simplified version — see MMSE-STSA or Wiener filter for production.
        """
        frame_len = self.win_length
        hop = self.hop_length

        # STFT
        frames = []
        for i in range(0, len(signal) - frame_len, hop):
            frames.append(signal[i:i + frame_len] * np.hanning(frame_len))

        stft = np.fft.rfft(np.array(frames), n=self.n_fft)
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        # Estimate noise from first N frames
        noise_psd = np.mean(magnitude[:noise_frames] ** 2, axis=0)
        noise_psd = np.maximum(noise_psd, 1e-8)

        # Spectral subtraction with flooring (prevents musical noise)
        alpha = 2.0   # over-subtraction factor
        beta = 0.01   # spectral floor
        clean_magnitude = np.maximum(
            magnitude ** 2 - alpha * noise_psd,
            beta * noise_psd
        ) ** 0.5

        # Reconstruct signal
        clean_stft = clean_magnitude * np.exp(1j * phase)
        clean_frames = np.fft.irfft(clean_stft, n=self.n_fft)[:, :frame_len]

        # Overlap-add reconstruction
        output = np.zeros(len(signal))
        for i, frame in enumerate(clean_frames):
            start = i * hop
            output[start:start + frame_len] += frame * np.hanning(frame_len)

        return output

    def energy_vad(self, signal: np.ndarray, threshold_db: float = -40) -> np.ndarray:
        """Fallback VAD using energy thresholding."""
        frame_len = self.win_length
        hop = self.hop_length
        mask = np.zeros(len(signal))

        for i in range(0, len(signal) - frame_len, hop):
            frame = signal[i:i + frame_len]
            energy_db = 10 * np.log10(np.mean(frame**2) + 1e-10)
            if energy_db > threshold_db:
                mask[i:i + frame_len] = 1.0

        return signal * mask

    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Full feature extraction pipeline.
        Input:  waveform tensor [1, T] at 16kHz
        Output: log-Mel spectrogram [1, n_mels, time_frames]
        """
        signal = waveform.squeeze().numpy()

        # 1. Pre-emphasis
        signal = self.apply_pre_emphasis(signal)

        # 2. Noise reduction
        signal = self.spectral_subtraction(signal)

        # 3. VAD (mute non-speech frames)
        signal = self.energy_vad(signal)

        # 4. Normalize to [-1, 1]
        signal = signal / (np.abs(signal).max() + 1e-8)

        # 5. Log-Mel filterbank
        waveform_clean = torch.FloatTensor(signal).unsqueeze(0)
        mel_spec = self.mel_transform(waveform_clean)        # [1, n_mels, T]
        log_mel = self.amplitude_to_db(mel_spec)             # [1, 40, T]

        # 6. Global CMVN (Cepstral Mean and Variance Normalization)
        # Critical for microphone distance robustness
        mean = log_mel.mean(dim=-1, keepdim=True)
        std = log_mel.std(dim=-1, keepdim=True) + 1e-5
        log_mel = (log_mel - mean) / std

        return log_mel  # [1, 40, time_frames]
    
    def extract_features_fast(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Fast batched feature extraction for training only.
        No DSP cleanup — used when input is already clean (LibriSpeech).
        
        Input:  [B, T] or [T] raw waveform
        Output: [B, 1, 40, time_frames] log-Mel features
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)        # [1, T]

        # Pre-emphasis as a 1D conv (vectorized, batched)
        we = torch.cat([
            waveform[:, :1],
            waveform[:, 1:] - self.pre_emphasis * waveform[:, :-1]
        ], dim=1)

        # Log-Mel (already batched in torchaudio)
        mel_spec = self.mel_transform(we)               # [B, 40, T]
        log_mel = self.amplitude_to_db(mel_spec)        # [B, 40, T]

        # Per-utterance CMVN
        mean = log_mel.mean(dim=-1, keepdim=True)
        std  = log_mel.std(dim=-1, keepdim=True) + 1e-5
        log_mel = (log_mel - mean) / std

        return log_mel.unsqueeze(1)                     # [B, 1, 40, T]