"""Ask a local vision-language model about images (Qwen2.5-VL, offline; modelforge/rig_vlm.py). Runs in UniRig's
Python (build/unirig/venv: torch 2.7.1, transformers 4.51, flash-attn), weights from HF_HOME.

    python vlm_ask.py REQUEST.json ANSWER

REQUEST.json: {"model", "max_pixels"?, and either "images" + "prompt" (+ "max_new_tokens") -> ANSWER is the text,
or "queries": [{"images", "prompt", "max_new_tokens"?}, ...] -> ANSWER is JSON [{"answer", "sizes"}], one model load
for all; "sizes" = [width, height] of each image as the model saw it (Qwen2.5-VL's point and box coordinates are
pixels of that resized image)}. Greedy decoding.
"""

import json
import sys

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def main():
    req = json.load(open(sys.argv[1]))
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        req["model"], torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="cuda")
    proc = AutoProcessor.from_pretrained(req["model"], max_pixels=req.get("max_pixels", 1024 * 28 * 28))
    queries = req.get("queries") or [{"images": req["images"], "prompt": req["prompt"],
                                      "max_new_tokens": req.get("max_new_tokens", 1500)}]
    out = []
    for k, q in enumerate(queries):
        images = [Image.open(p).convert("RGB") for p in q["images"]]
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": q["prompt"]}]
        text = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                        add_generation_prompt=True)
        inputs = proc(text=[text], images=images, return_tensors="pt").to("cuda")
        sizes = [[int(w) * 14, int(h) * 14] for _, h, w in inputs["image_grid_thw"].tolist()]
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=int(q.get("max_new_tokens", 1500)), do_sample=False)
        answer = proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        out.append({"answer": answer, "sizes": sizes})
        print(f"[vlm] query {k + 1} / {len(queries)}: {len(answer)} characters", flush=True)
    if req.get("queries"):
        json.dump(out, open(sys.argv[2], "w"))
    else:
        open(sys.argv[2], "w").write(out[0]["answer"])


if __name__ == "__main__":
    main()
