import uuid
import json
import logging
from logging_config import setup_logging
from db import DB
from ai import extract_pairs
from graph import ensure_attr_node
from filter_stats import match_pred
from config import DATABASE_PATH
# Setup logging
setup_logging()

# Initialize logger
logger = logging.getLogger(__name__)


def start(title: str, desc: str):
  try:
    logger.info("Starting classification process.")
    db = DB(DATABASE_PATH)
    item_id = f"tmp-{uuid.uuid4().hex[:8]}"
    db.upsert_item(item_id, title, desc)
    logger.debug("Item inserted into the database.")

    pairs = extract_pairs(title, desc)
    logger.debug(f"Extracted pairs: {pairs}")

    occ_rows = []
    for p in pairs:
      ensure_attr_node(db, p['name'])
      num = float(p.get('number')) if p.get('number') is not None else None
      occ_rows.append((item_id, p['name'], str(p.get('value')), p.get('unit'), num, f"{title} {desc}"))

    db.add_occurrences(occ_rows)
    logger.info(f"Occurrences added for item ID {item_id}.")

    filters = db.load_filters()
    logger.debug(f"Loaded filters: {filters}")

    best = None
    score = -1
    for f in filters:
      must_ok = all(match_pred(db, item_id, p) for p in f['plan'].get('must', []))
      must_not_ok = all(not match_pred(db, item_id, p) for p in f['plan'].get('must_not', []))
      s = 1.0 if (must_ok and must_not_ok) else 0.0
      if s > score:
        best = f
        score = s

    logger.info(f"Assigned best filter: {best}")
    logger.debug(json.dumps({"item_id": item_id, "assigned_filter": best}, ensure_ascii=False, indent=2))

  except Exception as e:
    logger.exception("An error occurred in the `start` function.")