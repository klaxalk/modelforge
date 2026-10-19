"""Text embeddings for semantic search: a worker in nerfstudio's venv (with the RADIO packages) that loads
C-RADIOv3's SigLIP2 adaptor (its text model) once and answers queries over stdin / stdout, so the GUI
pays the multi-GB load once. modelforge.semantic.TextEncoder starts and talks to it.

    python semantic_text.py VERSION ADAPTOR
stdin:  one JSON list of strings per line
stdout: "ready" once loaded, then one JSON list of L2-normalized embeddings (lists of floats) per line
"""

import json
import sys

import torch


def main():
    version, adaptor = sys.argv[1], sys.argv[2]
    model = torch.hub.load("NVlabs/RADIO", "radio_model", version=version, adaptor_names=[adaptor], progress=False,
                           skip_validation=True, trust_repo=True)
    ad = model.adaptors[adaptor]
    text_model = ad.text_model.cuda().eval().half() if torch.cuda.is_available() else ad.text_model.eval()
    ad.text_model = text_model
    del model  # the vision backbone is not needed here
    dev = next(text_model.parameters()).device
    print("ready", flush=True)
    for line in sys.stdin:
        texts = json.loads(line)
        tok = ad.tokenizer(texts).to(dev)
        with torch.no_grad():
            e = ad.encode_text(tok, normalize=True).float().cpu().numpy()
        print(json.dumps([[round(float(x), 6) for x in row] for row in e]), flush=True)


if __name__ == "__main__":
    main()
