# inference_engine.py
import torch
import torchaudio
import numpy as np
import time
from collections import deque
from audio_frontend import AudioFrontend
from kws_model import TCResNetKWS
from speaker_verifier import MiniECAPATDNN
from enrollment import SpeakerEnrollment
from decision_fusion import AdaptiveThresholdFusion, DecisionResult


class SpeechDisentanglementEngine:
    """
    Real-time inference engine that ties all stages together.
    Uses a sliding window over incoming audio chunks.
    
    Designed for streaming: call process_chunk() with each 10ms audio frame.
    The engine maintains internal buffers and context.
    """
    def __init__(
        self,
        kws_model_path: str,
        sv_model_path: str,
        speaker_profile_path: str,
        speaker_id: str,
        target_word: str,
        device: str = 'cpu',
        window_ms: int = 1000,     # 1-second sliding window
        step_ms: int = 10,         # 10ms hop (one frame at a time)
    ):
        self.device = torch.device(device)
        self.target_word = target_word
        self.speaker_id = speaker_id
        self.sample_rate = 16000

        # Initialize components
        self.frontend = AudioFrontend(sample_rate=self.sample_rate)

        # Helper: handles both raw state_dicts and full checkpoint dicts
        def _load_weights(path):
            ck = torch.load(path, map_location=device, weights_only=False)
            if isinstance(ck, dict) and 'model_state' in ck:
                return ck['model_state']
            return ck

        # Load KWS model
        self.kws_model = TCResNetKWS(n_mels=40, num_classes=2)
        self.kws_model.load_state_dict(_load_weights(kws_model_path))
        self.kws_model = self.kws_model.to(self.device).eval()

        # Load Speaker Verification model
        self.sv_model = MiniECAPATDNN(in_channels=40, embedding_dim=192)
        self.sv_model.load_state_dict(_load_weights(sv_model_path))
        self.sv_model = self.sv_model.to(self.device).eval()

        # Load speaker profile
        self.enrollment = SpeakerEnrollment(self.sv_model, self.frontend, device)
        self.enrollment.load_profile(speaker_id, speaker_profile_path)

        # Decision fusion
        self.fusion = AdaptiveThresholdFusion(
            kws_threshold=0.40,
            sv_threshold=0.60,
            combined_threshold=0.55,
        )

        # Audio buffer: holds last `window_ms` of audio
        window_samples = int(self.sample_rate * window_ms / 1000)
        self.audio_buffer = deque(maxlen=window_samples)

        self.kws_trigger_threshold = 0.50   # low gate: let SV double-check anything plausible

    @torch.no_grad()
    def process_chunk(self, audio_chunk: np.ndarray) -> DecisionResult:
        """
        Process one audio chunk (typically 10ms = 160 samples at 16kHz).
        Returns a DecisionResult every call.
        
        This is the hot path — must complete in < 5ms per frame to hit xRT < 0.2s
        for a 1-second window processed in real-time.
        """
        t0 = time.perf_counter()

        # 1. Add chunk to rolling buffer
        self.audio_buffer.extend(audio_chunk)
        if len(self.audio_buffer) < 400:   # minimum 25ms
            return DecisionResult(False, 0, 0, 0, "buffer_filling")
        
        # Fast silence skip — avoid feature extraction on quiet chunks.
        # Threshold 0.01 catches typical room silence; real speech is >0.02.
        if len(audio_chunk) > 0 and np.abs(audio_chunk).max() < 0.01:
            return DecisionResult(False, 0.0, 0.0, 0.0, "silence")

        # 2. Extract features from current window
        waveform = torch.FloatTensor(list(self.audio_buffer)).unsqueeze(0)
        # Use the fast frontend (no spectral subtraction / Python loops)
        # waveform is [1, T] → squeeze to [T]
        features = self.frontend.extract_features_fast(waveform.squeeze(0))
        # extract_features_fast returns [1, 1, 40, T]
        features = features.to(self.device)

        # 3. KWS inference (always runs)
        kws_logits = self.kws_model(features).squeeze(0)     # [2]
        kws_prob = torch.softmax(kws_logits, dim=-1)[1].item()

        # 4. SV inference (only when KWS is promising — saves ~40% compute)
        if kws_prob >= self.kws_trigger_threshold:
            sv_score = self.enrollment.verify_speaker(speaker_id=self.speaker_id, features=features)
        else:
            sv_score = 0.0

        # 5. Fusion decision
        result = self.fusion.combine_scores(kws_logits, sv_score)

        elapsed = (time.perf_counter() - t0) * 1000
        if elapsed > 80:
            print(f"Warning: chunk processing took {elapsed:.1f}ms (target < 80ms)")

        return result