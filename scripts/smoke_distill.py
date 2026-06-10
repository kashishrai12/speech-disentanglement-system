import sys; sys.path.insert(0,'src/')
import torch, torch.nn.functional as F
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy
from audio_frontend import AudioFrontend
from speaker_verifier import MiniECAPATDNN

teacher = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="models/pretrained/ecapa_tdnn",
    run_opts={"device":"cpu"},
    local_strategy=LocalStrategy.COPY)
student = MiniECAPATDNN(in_channels=40, embedding_dim=192)
frontend = AudioFrontend()

wav = torch.randn(2, 64000)
with torch.no_grad():
    t_emb = teacher.encode_batch(wav).squeeze(1)
print("Teacher:", t_emb.shape)

feats = frontend.extract_features_fast(wav)
print("Features:", feats.shape)

s_emb = student.extract_embedding(feats)
print("Student:", s_emb.shape)

loss = (1 - F.cosine_similarity(s_emb, t_emb)).mean()
print("Loss (random init):", loss.item())
print("OK")