"""Sky masks per photo (run in nerfstudio's venv with the RADIO packages on PYTHONPATH: transformers; see
modelforge/sky.py). SegFormer-B5 trained on ADE20K (class 2 = sky) on each photo, the logits upsampled to the photo's
size; sky pixels -> 0, the rest 255 (nerfstudio's mask convention: 0 = left out of the losses).

python sky_masks.py JOBS.json OUT_STATS.json [--live LIVE.json]
JOBS.json: {"model": id, "long_side": px, "dilate": share of the width the sky is widened by, "jobs": [{"src": photo, "dst": png, "w": width, "h": height}]}
(w, h: the mask size, the training images' size; the photo is resized to it first)."""
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

SKY = 2  # ADE20K "sky"


def main(jobs_path, stats_path, live=None):
    spec = json.load(open(jobs_path))
    from transformers import SegformerForSemanticSegmentation
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegformerForSemanticSegmentation.from_pretrained(spec["model"]).to(dev).eval().half()
    assert model.config.id2label.get(SKY, model.config.id2label.get(str(SKY))) == "sky", model.config.id2label
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    stats, t0 = {}, time.time()
    jobs = spec["jobs"]
    for k, j in enumerate(jobs):
        W, H = int(j["w"]), int(j["h"])
        f = min(1.0, spec["long_side"] / max(W, H))
        iw, ih = round(W * f), round(H * f)  # inference size
        img = Image.open(j["src"])
        img.draft("RGB", (iw, ih))  # JPEG DCT scaling while decoding
        x = torch.from_numpy(np.asarray(img.convert("RGB").resize((iw, ih), Image.BILINEAR), dtype=np.float32)
                             / 255.0).to(dev)
        x = ((x.permute(2, 0, 1)[None] - mean) / std).half()
        with torch.no_grad():
            logits = model(pixel_values=x).logits  # 1/4 of the input
            logits = torch.nn.functional.interpolate(logits.float(), size=(H, W), mode="bilinear",
                                                     align_corners=False)
            sky = logits.argmax(1)[0] == SKY
            r = max(1, round(spec.get("dilate", 0.0) * W))  # widened: no sky colour learned at the edges
            if spec.get("dilate"):
                sky = torch.nn.functional.max_pool2d(sky[None, None].float(), 2 * r + 1, 1, r)[0, 0] > 0.5
            sky = sky.cpu().numpy()
        os.makedirs(os.path.dirname(j["dst"]), exist_ok=True)
        Image.fromarray(np.where(sky, 0, 255).astype(np.uint8)).save(j["dst"] + ".part.png")
        os.replace(j["dst"] + ".part.png", j["dst"])
        stats[os.path.basename(j["dst"])] = round(float(sky.mean()), 4)
        print(f"sky masks: photo {k + 1} / {len(jobs)} · {os.path.basename(j['src'])} · sky {100 * sky.mean():.0f} %",
              flush=True)
    json.dump({"sky_share": stats, "seconds": round(time.time() - t0, 1)}, open(stats_path, "w"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
