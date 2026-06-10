# verify_teacher.py - sanity check the teacher model
import sys, torch
sys.path.insert(0, 'src/')

device = torch.device('cpu')

# Load the SpeechBrain ECAPA weights
weights = torch.load("models/pretrained/ecapa_full_weights.pt", 
                     map_location=device, weights_only=False)

print("Type:", type(weights))
if isinstance(weights, dict):
    print(f"Keys: {len(weights)} layers")
    for k in list(weights.keys())[:10]:
        if hasattr(weights[k], 'shape'):
            print(f"  {k}: {weights[k].shape}")