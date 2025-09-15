from __future__ import annotations
import json, sqlite3, math
from typing import Any, Dict, Iterable, List, Tuple

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS items (
  item_id TEXT PRIMARY KEY,
  title   TEXT,
  description TEXT,
  created_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS occurrences (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  item_id TEXT,
  name_text TEXT,
  value_text TEXT,
  unit_text TEXT,
  number_value REAL,
  context_text TEXT
);
CREATE TABLE IF NOT EXISTS attr_nodes (
  attr_id INTEGER PRIMARY KEY AUTOINCREMENT,
  label TEXT,
  centroid JSON,           -- embedding vector (JSON array)
  examples JSON            -- list of strings
);
CREATE TABLE IF NOT EXISTS edges (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  src_attr_id INTEGER,
  dst_attr_id INTEGER,
  type TEXT,               -- SIMILAR_TO | ALIAS_OF
  weight REAL,
  payload JSON
);
CREATE TABLE IF NOT EXISTS saved_filters (
  filter_id TEXT PRIMARY KEY,
  name TEXT,
  plan_json TEXT,
  created_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

class DB:
  def __init__(self, path: str):
    self.conn = sqlite3.connect(path)
    self.conn.row_factory = sqlite3.Row
    for stmt in SCHEMA.split(';'):
      s = stmt.strip()
      if s: self.conn.execute(s)
    self.conn.commit()

  def upsert_item(self, item_id: str, title: str, description: str):
    self.conn.execute("INSERT OR REPLACE INTO items(item_id,title,description) VALUES(?,?,?)", (item_id,title,description))
    self.conn.commit()

  def add_occurrences(self, rows: Iterable[Tuple[str,str,str,str,float,str]]):
    self.conn.executemany(
      "INSERT INTO occurrences(item_id,name_text,value_text,unit_text,number_value,context_text) VALUES(?,?,?,?,?,?)",
      rows
    ); self.conn.commit()

  def get_attr_nodes(self) -> List[sqlite3.Row]:
    return self.conn.execute("SELECT * FROM attr_nodes").fetchall()

  def create_attr_node(self, label: str, centroid: List[float], example: str) -> int:
    cur = self.conn.execute(
      "INSERT INTO attr_nodes(label,centroid,examples) VALUES(?,?,?)",
      (label, json.dumps(centroid,ensure_ascii=False), json.dumps([example],ensure_ascii=False))
    ); self.conn.commit(); return cur.lastrowid

  def update_attr_node(self, attr_id: int, new_centroid: List[float], new_example: str):
    row = self.conn.execute("SELECT centroid,examples FROM attr_nodes WHERE attr_id=?", (attr_id,)).fetchone()
    ex = json.loads(row["examples"]) if row and row["examples"] else []
    if new_example and new_example not in ex: ex.append(new_example)
    self.conn.execute(
      "UPDATE attr_nodes SET centroid=?, examples=? WHERE attr_id=?",
      (json.dumps(new_centroid), json.dumps(ex,ensure_ascii=False), attr_id)
    ); self.conn.commit()

  def save_filter(self, fid: str, name: str, plan: Dict[str,Any]):
    self.conn.execute("INSERT OR REPLACE INTO saved_filters(filter_id,name,plan_json) VALUES(?,?,?)",
                      (fid, name, json.dumps(plan, ensure_ascii=False)) ); self.conn.commit()

  def load_filters(self) -> List[Dict[str,Any]]:
    rows = self.conn.execute("SELECT * FROM saved_filters").fetchall()
    return [{"filter_id":r["filter_id"], "name":r["name"], "plan":json.loads(r["plan_json"])} for r in rows]

  def fetch_items(self):
    return self.conn.execute("SELECT * FROM items").fetchall()

  def fetch_occ_by_item(self, item_id: str):
    return self.conn.execute("SELECT * FROM occurrences WHERE item_id=?", (item_id,)).fetchall()
