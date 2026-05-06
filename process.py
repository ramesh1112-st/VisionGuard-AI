import torch
import clip
import cv2
from PIL import Image
import numpy as np
import os
from typing import List, Tuple

# -------------------------- 1. Basic Configuration --------------------------
class Config:
    # Model parameters
    MODEL_NAME = "ViT-B/32"   # CLIP model type (can also use ViT-L/14)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    CLIP_FRAME_NUM = 16   # Number of frames per video clip
    CLIP_STRIDE = 8       # Sliding window stride

    # Save paths
    VIDEO_FEAT_SAVE_DIR = "preprocessed_data/video_features"
    TEXT_FEAT_SAVE_DIR  = "preprocessed_data/text_features"
    SIM_MATRIX_SAVE_DIR = "preprocessed_data/sim_matrices"

    # Create directories if not exist
    for path in [VIDEO_FEAT_SAVE_DIR, TEXT_FEAT_SAVE_DIR, SIM_MATRIX_SAVE_DIR]:
        os.makedirs(path, exist_ok=True)

# Load CLIP model
model, preprocess = clip.load(Config.MODEL_NAME, device=Config.DEVICE)


# -------------------------- 2. Video Feature Extraction --------------------------
def extract_video_clip_feat(video_path: str) -> Tuple[np.ndarray, np.ndarray]:

    cap = cv2.VideoCapture(video_path)
    frame_list = []

    # 1. Read all frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR → RGB and preprocess for CLIP
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = preprocess(pil_img).unsqueeze(0)
        frame_list.append(img_tensor)

    cap.release()

    # 2. Split into clips and extract features
    clip_num = max(1, len(frame_list) - Config.CLIP_FRAME_NUM + 1)

    video_feats = []
    clip_frame_mapping = []

    for i in range(clip_num):

        start_frame = i * Config.CLIP_STRIDE
        end_frame = start_frame + Config.CLIP_FRAME_NUM

        # Handle last clip
        if end_frame > len(frame_list):
            end_frame = len(frame_list)
            start_frame = max(0, end_frame - Config.CLIP_FRAME_NUM)

        # Stack frames
        clip_imgs = torch.cat(frame_list[start_frame:end_frame], dim=0).to(Config.DEVICE)

        with torch.no_grad():
            clip_feat = model.encode_image(clip_imgs)  # (frames, 512)

        # Mean pooling → clip feature
        clip_feat = clip_feat.mean(dim=0)

        # L2 normalization
        clip_feat = clip_feat / clip_feat.norm(dim=-1, keepdim=True)

        video_feats.append(clip_feat.cpu().numpy())

        # Store frame index mapping
        clip_frame_mapping.append([start_frame, end_frame - 1])

    return np.array(video_feats), np.array(clip_frame_mapping)


# -------------------------- 3. Text Feature Extraction --------------------------
def extract_text_feat(text_list: List[str]) -> np.ndarray:
    """
    Extract text features (supports augmented text)
    Args:
        text_list: list of descriptions
    Returns:
        text_feats: (num_text, 512)
    """

    text_tokens = clip.tokenize(text_list).to(Config.DEVICE)

    with torch.no_grad():
        text_feats = model.encode_text(text_tokens)

    # Normalize
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    return text_feats.cpu().numpy()


# -------------------------- 4. Cross-modal Similarity --------------------------
def calculate_cross_modal_sim(
    video_feats: np.ndarray,
    text_feats: np.ndarray
) -> np.ndarray:
    """
    Compute cosine similarity between video clips and text
    """

    # Since normalized → dot product = cosine similarity
    sim_matrix = np.matmul(video_feats, text_feats.T)

    return sim_matrix


# -------------------------- 5. Full Pipeline --------------------------
def run_preprocess_pipeline(
    video_dir: str,
    text_desc_dict: dict
) -> None:

    for video_filename in os.listdir(video_dir):

        if not video_filename.endswith((".mp4", ".avi")):
            continue

        video_name = os.path.splitext(video_filename)[0]
        video_path = os.path.join(video_dir, video_filename)

        print(f"Processing video: {video_name}")

        # 1. Extract video features
        video_feats, clip_mapping = extract_video_clip_feat(video_path)

        np.save(
            os.path.join(Config.VIDEO_FEAT_SAVE_DIR, f"{video_name}_feats.npy"),
            video_feats
        )

        np.save(
            os.path.join(Config.VIDEO_FEAT_SAVE_DIR, f"{video_name}_frame_mapping.npy"),
            clip_mapping
        )

        # 2. Extract text features
        if video_name not in text_desc_dict:
            print(f"Warning: No text for {video_name}, skipping...")
            continue

        text_list = text_desc_dict[video_name]
        text_feats = extract_text_feat(text_list)

        np.save(
            os.path.join(Config.TEXT_FEAT_SAVE_DIR, f"{video_name}_text_feats.npy"),
            text_feats
        )

        # 3. Compute similarity
        sim_matrix = calculate_cross_modal_sim(video_feats, text_feats)

        np.save(
            os.path.join(Config.SIM_MATRIX_SAVE_DIR, f"{video_name}_sim_matrix.npy"),
            sim_matrix
        )

        print(f"Finished processing {video_name}\n")


# -------------------------- 6. Example Run --------------------------
if __name__ == "__main__":

    TEST_VIDEO_DIR = "./video"

    TEST_TEXT_DESC_DICT = {
        "car 01": ["detect collision", "vehicle collision in video", "identify accident", "traffic accident detected"],
        "car 02": ["rear-end collision", "vehicle scratch detected", "identify collision", "collision event present"],
        "car 03": ["traffic accident analysis", "pedestrian hits bike", "vehicle overturn", "road accident"],
        "car 04": ["detect accident", "vehicle collision in video", "identify accident", "traffic accident detected"],
        "car 05": ["rear-end accident", "two vehicles collide", "identify collision", "collision event present"],
        "car 06": ["detect accident", "vehicle collision in video", "identify accident", "traffic accident detected"],
        "car 07": ["rear-end accident", "vehicle scratch detected", "identify collision", "collision event present"],
        "car 08": ["traffic accident analysis", "vehicle collision", "vehicle overturn", "road accident"],
        "car 10": ["detect collision", "vehicle collision in video", "identify accident", "traffic accident detected"],
        "car 12": ["rear-end accident", "vehicle scratch detected", "identify collision", "collision event present"],

        "normal_1": ["normal", "no abnormal event", "normal state", "nothing unusual"],
        "normal_2": ["normal", "no abnormal event", "normal state", "nothing unusual"],
        "normal_3": ["normal", "no abnormal event", "normal state", "nothing unusual"],
        "normal_4": ["normal", "no abnormal event", "normal state", "nothing unusual"],
        "normal_5": ["normal", "no abnormal event", "normal state", "nothing unusual"],
        "normal_6": ["normal", "no abnormal event", "normal state", "nothing unusual"],
        "normal_7": ["normal", "no abnormal event", "normal state", "nothing unusual"],
        "normal_8": ["normal", "no abnormal event", "normal state", "nothing unusual"],
        "normal_9": ["normal", "no abnormal event", "normal state", "nothing unusual"],
        "normal_10": ["normal", "no abnormal event", "normal state", "nothing unusual"]
    }

    run_preprocess_pipeline(
        video_dir=TEST_VIDEO_DIR,
        text_desc_dict=TEST_TEXT_DESC_DICT
    )