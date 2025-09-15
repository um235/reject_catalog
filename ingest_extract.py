import argparse, csv, uuid, json, numpy as np
from db import DB
from ai import extract_pairs
from graph import ensure_attr_node
from config import DATABASE_PATH
import logging
from logging_config import setup_logging
setup_logging()

logger = logging.getLogger(__name__)
def extract():
  db = DB(DATABASE_PATH)
  with open("data/materials_example.csv", 'r', encoding='utf-8') as f:
    rdr = csv.DictReader(f)
    for row in rdr:
      item_id = row.get('item_id') or uuid.uuid4().hex[:8]
      title = row.get('title',''); desc = row.get('description','')
      db.upsert_item(item_id, title, desc)
      pairs = extract_pairs(title, desc)
      occ_rows = []
      for p in pairs:
        ensure_attr_node(db, p['name'])  # creates/attaches graph nodes via embeddings+LLM
        num = float(p.get('number')) if p.get('number') is not None else None
        occ_rows.append((item_id, p['name'], str(p.get('value')), p.get('unit'), num, f"{title} {desc}"))
      if occ_rows:
        db.add_occurrences(occ_rows)
  print(f"Ingest OK → data.db")

extract()