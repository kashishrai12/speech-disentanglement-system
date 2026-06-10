# enrollment.py
import torch
import torchaudio        
import numpy as np
from pathlib import Path
from audio_frontend import AudioFrontend
from speaker_verifier import MiniECAPATDNN


class SpeakerEnrollment:
    """
    Enrollment creates a speaker profile from N utterances (typically 5–10).
    The profile is the mean of all enrollment embeddings — this averaging
    improves robustness and reduces noise in the embedding.
    
    Store the profile on disk so re-enrollment is not needed per session.
    """
    def __init__(self, model: MiniECAPATDNN, frontend: AudioFrontend, device: str = 'cpu'):
        self.model = model.eval()
        self.frontend = frontend
        self.device = device
        self.profiles = {}

    @torch.no_grad()
    def enroll_speaker(self, speaker_id: str, audio_paths: list) -> torch.Tensor:
        """
        Enroll a speaker from multiple audio files.
        Returns the mean embedding (speaker profile).
        """
        embeddings = []
        for path in audio_paths:
            waveform, sr = torchaudio.load(path)
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            # Use same frontend as distillation training for consistency
            features = self.frontend.extract_features_fast(waveform.squeeze(0))
            features = features.to(self.device)    # [1, 1, 40, T]
            emb = self.model.extract_embedding(features)        # [1, embedding_dim]
            embeddings.append(emb)

        profile = torch.mean(torch.cat(embeddings, dim=0), dim=0, keepdim=True)
        profile = torch.nn.functional.normalize(profile, dim=1)
        self.profiles[speaker_id] = profile
        return profile

    def save_profile(self, speaker_id: str, path: str):
        torch.save(self.profiles[speaker_id], path)

    def load_profile(self, speaker_id: str, path: str):
        self.profiles[speaker_id] = torch.load(path, map_location=self.device)

    @torch.no_grad()
    def verify_speaker(self, speaker_id: str, features: torch.Tensor) -> float:
        """
        Returns cosine similarity score [0, 1].
        Higher = more likely to be the enrolled speaker.
        """
        emb = self.model.extract_embedding(features.to(self.device))
        profile = self.profiles[speaker_id].to(self.device)
        similarity = torch.nn.functional.cosine_similarity(emb, profile).item()
        return similarity