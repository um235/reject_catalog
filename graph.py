import json
from typing import Tuple

import numpy as np

from config import SIM_THRESHOLD_NEW_ATTR, TOPK_ATTR
from db import DB
from ai import embed, cosine, llm_json
import logging
from logging_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)
ARBITER_SYS = """
You are an ontology arbiter. Given a candidate attribute label and a list of existing nodes (label + examples), decide:
- best_match: index or -1 if new node is required
- relation: "SIMILAR_TO" or "ALIAS_OF" when match found
Return JSON: {"best_index": int, "relation": "SIMILAR_TO|ALIAS_OF|NONE"}
"""

def ensure_attr_node(db: DB, label: str) -> int:
  rows = db.get_attr_nodes()
  if not rows:
    vec = embed([label])[0]
    return db.create_attr_node(label, vec, label)
  labels = [r["label"] for r in rows]
  centroids = [json.loads(r["centroid"]) for r in rows]
  qv = embed([label])[0]
  sims = [cosine(qv, c) for c in centroids]
  best_i = int(np.argmax(np.array(sims)))
  best_score = sims[best_i]
  if best_score >= SIM_THRESHOLD_NEW_ATTR:
    # attach to existing and update centroid (simple avg)
    new_centroid = ((np.array(centroids[best_i]) + np.array(qv))/2.0).tolist()
    db.update_attr_node(rows[best_i]["attr_id"], new_centroid, label)
    return rows[best_i]["attr_id"]
  # ask LLM arbiter against TOPK candidates
  order = np.argsort(np.array(sims))[::-1][:min(TOPK_ATTR, len(rows))]
  candidates = [{"label":labels[i], "examples":json.loads(rows[i]["examples"])} for i in order]
  j = llm_json(ARBITER_SYS, json.dumps({"candidate":label, "existing":candidates}, ensure_ascii=False))
  idx = int(j.get("best_index", -1))
  if idx>=0:
    idx = int(order[idx])
    new_centroid = ((np.array(centroids[idx]) + np.array(qv))/2.0).tolist()
    db.update_attr_node(rows[idx]["attr_id"], new_centroid, label)
    return rows[idx]["attr_id"]
  # create new
  return db.create_attr_node(label, qv, label)