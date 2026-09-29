#!/usr/bin/env python3
"""Model the device bytes a Flash-Next 125B-A6B decode step reads, from an artifact's directory.

Reads only the artifact directory (logical bindings and stored object sizes), never the weight
payloads, so it takes seconds on the full artifact and follows whatever formats the artifact
records (BF16, FP8 row-scaled, NVFP4). Prints one JSON object with the per-step components:

- dense_bytes: text weights read in full every step (projections, norms, HyperConnection,
  router, shared expert, output head);
- expert_bank_bytes: the 48 routed NVFP4 expert banks, of which a step reads the selected
  experts (10 of 512 per token);
- embedding_row_bytes: one embedding row per token;
- gdn_state_bytes: the FP32 GDN recurrent state, read and written once per step
  (36 layers x 48 heads x 128 x 128);
- mtp_dense_bytes / mtp_expert_bank_bytes / draft_head_bytes: what one MTP draft step reads.

Not modeled: QSA KV and indexer reads (context-dependent, small at the 1K prompt of the step 2
power runs), the file-mapped PLE rows (a few KB per token, read from host pages), activations.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.artifact.reader import Artifact  # noqa: E402
from tools.artifact.schema import binding_parts  # noqa: E402

EXPERTS = 512
TOP_K = 10
GDN_STATE_BYTES = 36 * 48 * 128 * 128 * 4
TEXT_EXPERT_BANK = re.compile(r"^model\.language_model\.layers\.\d+\.mlp\.experts\.(gate_up|down)$")
MTP_EXPERT_BANK = re.compile(r"^mtp\.layers\.\d+\.mlp\.experts\.(gate_up_proj|down_proj)$")


def binding_bytes(artifact: Artifact, name: str, binding: object) -> float:
    total = 0.0
    for object_id, begin, end in binding_parts(binding, artifact.by_id, name):
        obj = artifact.object(object_id)
        elements = 1
        for extent in obj.shape:
            elements *= extent
        total += obj.bytes * (end - begin) / elements
    return total


def model(path: Path) -> dict:
    parts = {key: 0.0 for key in ("dense_bytes", "expert_bank_bytes", "embedding_row_bytes",
                                  "mtp_dense_bytes", "mtp_expert_bank_bytes", "draft_head_bytes")}
    with Artifact(path) as artifact:
        for name, binding in artifact.directory.bindings.items():
            if name.startswith("vision") or name.endswith("ple_embedding.ngram_embedding.weight"):
                continue
            size = binding_bytes(artifact, name, binding)
            if name == "model.language_model.embed_tokens.weight":
                rows = 248320
                parts["embedding_row_bytes"] += size / rows
            elif name.startswith("ninfer.optimized_proposal_head."):
                parts["draft_head_bytes"] += size
            elif MTP_EXPERT_BANK.match(name):
                parts["mtp_expert_bank_bytes"] += size
            elif name.startswith("mtp."):
                parts["mtp_dense_bytes"] += size
            elif TEXT_EXPERT_BANK.match(name):
                parts["expert_bank_bytes"] += size
            else:
                parts["dense_bytes"] += size
        recipe = artifact.directory.provenance.get("recipe")
    parts["gdn_state_bytes"] = 2.0 * GDN_STATE_BYTES
    parts["recipe"] = recipe
    return parts


def bytes_per_token(parts: dict, draft_tokens: int = 0, accepted_per_round: float = 1.0) -> float:
    """Modeled bytes per emitted token. Without speculation a step emits one token. With MTP a
    round verifies draft_tokens + 1 tokens in one pass (dense weights and state once, the union
    of their experts), runs draft_tokens MTP steps, and emits accepted_per_round tokens. The
    expert union assumes independent uniform routing, an upper bound on real (correlated)
    routing."""
    width = draft_tokens + 1
    union = EXPERTS * (1.0 - (1.0 - TOP_K / EXPERTS) ** width)
    round_bytes = (parts["dense_bytes"] + parts["gdn_state_bytes"]
                   + parts["expert_bank_bytes"] * union / EXPERTS
                   + parts["embedding_row_bytes"] * width)
    if draft_tokens:
        round_bytes += draft_tokens * (parts["mtp_dense_bytes"] + parts["draft_head_bytes"]
                                       + parts["mtp_expert_bank_bytes"] * TOP_K / EXPERTS)
    return round_bytes / max(accepted_per_round, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    parts = model(args.artifact)
    parts["bytes_per_token_no_speculation"] = bytes_per_token(parts)
    print(json.dumps(parts, indent=2))


if __name__ == "__main__":
    main()
