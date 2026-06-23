"""Extract 50 driving frames from the CulturalDrive dataset videos (+ stills)
into autolabeling/frames/ and write a manifest. Uses OpenCV (no ffmpeg needed).
Run in the `dllm` conda env (has cv2)."""
import json
import os
import glob

import cv2

DATASET = os.environ.get("DATASET_DIR",
    "/weka/home/ext-yingzima/CulturalDrive/dataset/videos")  # dir of .mp4 / .png / .jpg
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (parent of utils/)
OUT_DIR = os.environ.get("FRAMES_DIR", os.path.join(ROOT, "frames"))
N_TARGET = 50


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    videos = sorted(glob.glob(os.path.join(DATASET, "*.mp4")))
    stills = sorted(glob.glob(os.path.join(DATASET, "*.png")) +
                    glob.glob(os.path.join(DATASET, "*.jpg")))
    print(f"{len(videos)} videos, {len(stills)} stills")

    manifest = []
    fid = 0

    # Reserve the stills first (real distinct scenes), then fill from videos.
    n_from_video = N_TARGET - len(stills)
    per_video = max(1, -(-n_from_video // max(1, len(videos))))  # ceil division

    for v in videos:
        cap = cv2.VideoCapture(v)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            continue
        # Evenly spaced frames across the clip.
        idxs = [int(total * (k + 0.5) / per_video) for k in range(per_video)]
        for j in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(j, total - 1))
            ok, frame = cap.read()
            if not ok:
                continue
            name = f"frame_{fid:03d}.jpg"
            path = os.path.join(OUT_DIR, name)
            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            manifest.append({"id": fid, "frame_path": path,
                             "source": os.path.basename(v), "kind": "video"})
            fid += 1
        cap.release()

    for s in stills:
        img = cv2.imread(s)
        if img is None:
            continue
        name = f"frame_{fid:03d}.jpg"
        path = os.path.join(OUT_DIR, name)
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        manifest.append({"id": fid, "frame_path": path,
                         "source": os.path.basename(s), "kind": "still"})
        fid += 1

    manifest = manifest[:N_TARGET]
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {len(manifest)} frames -> {OUT_DIR}/manifest.json")


if __name__ == "__main__":
    main()
