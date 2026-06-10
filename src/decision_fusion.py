# decision_fusion.py
import torch
import numpy as np
from dataclasses import dataclass
from collections import deque


@dataclass
class DecisionResult:
    accepted: bool
    kws_score: float
    sv_score: float
    combined_score: float
    reason: str


class AdaptiveThresholdFusion:
    """
    Bayesian-inspired score fusion with adaptive thresholds.
    
    Key insight: KWS and SV scores are combined multiplicatively
    (both must be high simultaneously), not additively (which allows
    one weak signal to compensate for the other).
    
    FA suppression: a cooldown window prevents repeated triggers
    from the same utterance (a common source of spurious FAs).
    """
    def __init__(
        self,
        kws_threshold: float = 0.4,
        sv_threshold: float = 0.65,
        combined_threshold: float = 0.55,   # ← changed from 0.70
        cooldown_frames: int = 50,   # ~500ms cooldown between accepts
        history_len: int = 3,        # look at last 3 frames for smoothing
    ):
        self.kws_threshold = kws_threshold
        self.sv_threshold = sv_threshold
        self.combined_threshold = combined_threshold
        self.cooldown_frames = cooldown_frames
        self.frames_since_accept = cooldown_frames + 1

        # Score smoothing buffers
        self.kws_history = deque(maxlen=history_len)
        self.sv_history  = deque(maxlen=history_len)

    def _smooth(self, history: deque, new_score: float) -> float:
        history.append(new_score)
        # Weighted mean: recent frames have higher weight
        weights = np.arange(1, len(history) + 1, dtype=float)
        return float(np.average(list(history), weights=weights))

    def combine_scores(self, kws_logits: torch.Tensor, sv_similarity: float) -> DecisionResult:
        """
        kws_logits: [2] raw logits from KWS model [other_class, target_class]
        sv_similarity: cosine similarity in [-1, 1]
        """
        self.frames_since_accept += 1

        # KWS probability via softmax
        kws_prob = torch.softmax(kws_logits, dim=-1)[1].item()

        # Smooth over recent frames
        kws_smooth = self._smooth(self.kws_history, kws_prob)

        # Use raw cosine similarity directly — for normalized speaker
        # embeddings, this is already in ~[0, 1] for real voices and
        # mapping (x+1)/2 destroys discrimination by lifting impostors
        # into the accept zone.
        sv_smooth = self._smooth(self.sv_history, float(sv_similarity))

        # Geometric mean fusion — both must be strong
        combined = (kws_smooth * sv_smooth) ** 0.5

        # Cooldown check: prevent re-triggering during the same utterance
        if self.frames_since_accept < self.cooldown_frames:
            return DecisionResult(
                accepted=False,
                kws_score=kws_smooth, sv_score=sv_smooth,
                combined_score=combined, reason="cooldown"
            )

        # Decision gate: all three conditions must be met
        if (kws_smooth >= self.kws_threshold and
            sv_smooth  >= self.sv_threshold and
            combined   >= self.combined_threshold):

            self.frames_since_accept = 0
            return DecisionResult(
                accepted=True,
                kws_score=kws_smooth, sv_score=sv_smooth,
                combined_score=combined, reason="accepted"
            )

        reason = "kws_low" if kws_smooth < self.kws_threshold else \
                 "sv_low"  if sv_smooth  < self.sv_threshold  else "combined_low"

        return DecisionResult(
            accepted=False,
            kws_score=kws_smooth, sv_score=sv_smooth,
            combined_score=combined, reason=reason
        )