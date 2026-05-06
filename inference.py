import torch
import numpy as np
import os
import cv2
from PIL import Image
import clip
import sys  # ✅ IMPORTANT

# ✅ Get video path from Flask
video_path = sys.argv[1]
video_file = os.path.basename(video_path)
video_id = os.path.splitext(video_file)[0]

OUTPUT_FILE = "submission.txt"
MODEL_PATH = "saved_models/model_epoch_210.pth"

TEXT_QUERY = ["detect fighting", "detect fire", "detect falling", "detect vehicle collision"]

CLIP_FRAME_NUM = 16
CLIP_STRIDE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CATEGORY_TO_ID = {
    "normal": 0,
    "fighting": 1,
    "fire": 2,
    "fall": 3,
    "vehicle_collision": 4
}
ID_TO_CATEGORY = {v: k for k, v in CATEGORY_TO_ID.items()}


class CrossModalModel(torch.nn.Module):
    def __init__(self, video_dim=512, text_dim=512, num_categories=len(CATEGORY_TO_ID)):
        super().__init__()
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(video_dim + text_dim + 1, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU()
        )
        self.attn = torch.nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
        self.anomaly_head = torch.nn.Linear(256, 1)
        self.category_head = torch.nn.Linear(256, num_categories)
        self.loc_head = torch.nn.Linear(256, 2)

    def forward(self, video_feat, text_feats, sim_score):
        text_feat = text_feats.mean(dim=1)
        fused = torch.cat([video_feat, text_feat, sim_score], dim=1)
        fused = self.fusion(fused).unsqueeze(1)
        attn_out, _ = self.attn(fused, fused, fused)
        attn_out = attn_out.squeeze(1)
        return self.anomaly_head(attn_out), self.category_head(attn_out), self.loc_head(attn_out)


# ✅ Load model
model = CrossModalModel().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)


# 🚀 SPEED OPTIMIZATION: limit frames
def extract_video_clip_feats(video_path, max_frames=64):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret or count >= max_frames:
            break
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frames.append(clip_preprocess(pil_img).unsqueeze(0))
        count += 1

    cap.release()

    feats = []
    frame_mapping = []

    for i in range(0, len(frames) - CLIP_FRAME_NUM + 1, CLIP_STRIDE):
        clip_imgs = torch.cat(frames[i:i+CLIP_FRAME_NUM], dim=0).to(DEVICE)
        with torch.no_grad():
            f = clip_model.encode_image(clip_imgs).mean(dim=0)
            f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu().numpy())
        frame_mapping.append([i, i + CLIP_FRAME_NUM - 1])

    return np.array(feats), np.array(frame_mapping), len(frames)


def extract_text_feats(text_list):
    text_tokens = clip.tokenize(text_list).to(DEVICE)
    with torch.no_grad():
        feats = clip_model.encode_text(text_tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()


# ✅ Process ONLY uploaded video
text_feats_np = extract_text_feats(TEXT_QUERY)

video_feats, frame_mapping, total_frames = extract_video_clip_feats(video_path)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:

    if len(video_feats) == 0:
        f_out.write(f"{video_id} -1 -1 normal\n")

    else:
        sim_matrix = video_feats @ text_feats_np.T
        sim_scores = sim_matrix.mean(axis=1, keepdims=True)

        video_feats = torch.FloatTensor(video_feats).to(DEVICE)
        text_feats = torch.FloatTensor(text_feats_np).unsqueeze(0).repeat(len(video_feats), 1, 1).to(DEVICE)
        sim_scores = torch.FloatTensor(sim_scores).to(DEVICE)

        with torch.no_grad():
            anomaly_logits, category_logits, loc_offsets = model(video_feats, text_feats, sim_scores)

        anomaly_probs = torch.sigmoid(anomaly_logits).cpu().numpy().flatten()
        category_preds = torch.argmax(category_logits, dim=1).cpu().numpy()
        loc_offsets = loc_offsets.cpu().numpy()

        results = []
        for i in range(len(video_feats)):
            if anomaly_probs[i] > 0.5:
                start = max(0, int(frame_mapping[i, 0] + loc_offsets[i, 0]))
                end = min(int(frame_mapping[i, 1] + loc_offsets[i, 1]), total_frames - 1)
                category = ID_TO_CATEGORY[category_preds[i]]
                results.append([start, end, category])

        if results:
            r = results[0]  # take first detection
            f_out.write(f"{video_id} {r[0]} {r[1]} {r[2]}\n")
        else:
            f_out.write(f"{video_id} -1 -1 normal\n")


print(f"Done: {video_id}")