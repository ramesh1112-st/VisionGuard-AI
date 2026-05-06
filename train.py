import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

# ==================== CONFIG ====================
VIDEO_FEAT_DIR = "preprocessed_data/video_features"
TEXT_FEAT_DIR  = "preprocessed_data/text_features"
SIM_MATRIX_DIR = "preprocessed_data/sim_matrices"
PSEUDO_LABEL_DIR = "pseudo_labels"  # pseudo label folder
SAVE_MODEL_DIR = "saved_models"
os.makedirs(SAVE_MODEL_DIR, exist_ok=True)

# Category mapping
CATEGORY_TO_ID = {
    "normal": 0,
    "fighting": 1,
    "fire": 2,
    "fall": 3,
    "vehicle_collision": 4
}

BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 500  # number of training epochs
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCH_SAVE = 10  # save model every 10 epochs

# ==================== Dataset (load pseudo labels) ====================
class AnomalyDataset(Dataset):
    def __init__(self, video_ids, label_dict):
        self.samples = []
        for vid in video_ids:
            # Load features
            video_feats = np.load(os.path.join(VIDEO_FEAT_DIR, f"{vid}_feats.npy"))
            text_feats  = np.load(os.path.join(TEXT_FEAT_DIR, f"{vid}_text_feats.npy"))
            sim_matrix  = np.load(os.path.join(SIM_MATRIX_DIR, f"{vid}_sim_matrix.npy"))
            frame_mapping = np.load(os.path.join(VIDEO_FEAT_DIR, f"{vid}_frame_mapping.npy"))
            
            # Video label
            category = label_dict[vid]["category"]
            category_id = CATEGORY_TO_ID[category]
            label = 0 if category == "normal" else 1  # 0=normal, 1=anomaly
            
            # Load pseudo labels (one txt per video)
            pseudo_label_path = os.path.join(PSEUDO_LABEL_DIR, f"{vid}.txt")
            events = []
            if os.path.exists(pseudo_label_path):
                with open(pseudo_label_path, "r") as f:
                    for line in f:
                        start, end = map(int, line.strip().split())
                        events.append([start, end])
            
            # Each clip is one sample
            for i in range(len(video_feats)):
                clip_start, clip_end = frame_mapping[i]
                start_offset = 0
                end_offset = 0

                # Check overlap with anomaly event
                for (gt_start, gt_end) in events:
                    if not (clip_end < gt_start or clip_start > gt_end):
                        # Calculate offset
                        start_offset = gt_start - clip_start
                        end_offset = gt_end - clip_end
                        break
                
                self.samples.append({
                    "video_feat": video_feats[i],
                    "text_feats": text_feats,
                    "sim_score": sim_matrix[i].mean(),
                    "label": label,
                    "category_id": category_id,
                    "frame_range": frame_mapping[i],
                    "start_offset": start_offset,
                    "end_offset": end_offset
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            torch.FloatTensor(s["video_feat"]),
            torch.FloatTensor(s["text_feats"]),
            torch.FloatTensor([s["sim_score"]]),
            torch.LongTensor([s["label"]]),
            torch.LongTensor([s["category_id"]]),
            torch.LongTensor(s["frame_range"]),
            torch.FloatTensor([s["start_offset"]]),
            torch.FloatTensor([s["end_offset"]])
        )

# ==================== Model ====================
class CrossModalModel(nn.Module):
    def __init__(self, video_dim=512, text_dim=512, num_categories=len(CATEGORY_TO_ID)):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(video_dim + text_dim + 1, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.attn = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
        
        self.anomaly_head = nn.Linear(256, 1)
        self.category_head = nn.Linear(256, num_categories)
        self.loc_head = nn.Linear(256, 2)  # predict start/end frame offset

    def forward(self, video_feat, text_feats, sim_score):
        text_feat = text_feats.mean(dim=1)
        fused = torch.cat([video_feat, text_feat, sim_score], dim=1)
        fused = self.fusion(fused).unsqueeze(1)
        attn_out, _ = self.attn(fused, fused, fused)
        attn_out = attn_out.squeeze(1)
        return self.anomaly_head(attn_out), self.category_head(attn_out), self.loc_head(attn_out)

# ==================== Training ====================
def train():
    # Video labels
    label_dict = {
        "car 01": {"category": "vehicle_collision"},
        "car 02": {"category": "vehicle_collision"},
        "car 03": {"category": "vehicle_collision"},
        "car 04": {"category": "vehicle_collision"},
        "car 05": {"category": "vehicle_collision"},
        "car 06": {"category": "vehicle_collision"},
        "car 07": {"category": "vehicle_collision"},
        "car 08": {"category": "vehicle_collision"},
        "car 10": {"category": "vehicle_collision"},
        "car 12": {"category": "vehicle_collision"},
        "normal_1": {"category": "normal"},
        "normal_2": {"category": "normal"},
        "normal_3": {"category": "normal"},
        "normal_4": {"category": "normal"},
        "normal_5": {"category": "normal"},
        "normal_6": {"category": "normal"},
        "normal_7": {"category": "normal"},
        "normal_8": {"category": "normal"},
        "normal_9": {"category": "normal"},
        "normal_10": {"category": "normal"},
    }

    video_ids = list(label_dict.keys())
    
    dataset = AnomalyDataset(video_ids, label_dict)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = CrossModalModel().to(DEVICE)
    criterion_anomaly = nn.BCEWithLogitsLoss()
    criterion_category = nn.CrossEntropyLoss()
    criterion_loc = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch in dataloader:
            video_feat, text_feats, sim_score, label, category_id, frame_range, start_offset, end_offset = [
                x.to(DEVICE) for x in batch
            ]

            optimizer.zero_grad()

            anomaly_logit, category_logit, loc_offset = model(video_feat, text_feats, sim_score)

            loss_anomaly = criterion_anomaly(anomaly_logit, label.float())

            mask = (label == 1).squeeze()
            if mask.float().sum() > 0:
                loss_category = criterion_category(
                    category_logit[mask],
                    category_id[mask].flatten()
                )
            else:
                loss_category = torch.tensor(0.0).to(DEVICE)

            loss_loc = torch.tensor(0.0).to(DEVICE)
            if mask.float().sum() > 0:
                pred_start = loc_offset[mask, 0].flatten()
                pred_end = loc_offset[mask, 1].flatten()
                gt_start = start_offset[mask].flatten()
                gt_end = end_offset[mask].flatten()

                loss_loc = criterion_loc(
                    torch.stack([pred_start, pred_end], dim=1),
                    torch.stack([gt_start, gt_end], dim=1)
                )

            loss = loss_anomaly + loss_category + 0.5 * loss_loc
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader):.4f}")

        if (epoch + 1) % EPOCH_SAVE == 0:
            save_path = os.path.join(SAVE_MODEL_DIR, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved: {save_path}")

if __name__ == "__main__":
    train()