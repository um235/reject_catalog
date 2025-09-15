import argparse, json, sqlite3
from db import DB
from ai import plan_from_query, convert_value
from config import DATABASE_PATH
import logging
from logging_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

def match_pred(db: DB, item_id: str, pred: dict) -> bool:


  attr = pred['attr']; op = pred['op']; val = pred['value']; unit = pred.get('unit')
  occs = db.fetch_occ_by_item(item_id)
  cands = [o for o in occs if o['name_text'].lower()==attr.lower()]
  if not cands: return False if op in ('eq','gte','lte','range','contains') else True
  if op in ('eq','gte','lte','range'):
    for o in cands:
      num = o['number_value']
      if num is None: continue
      from_u = o['unit_text']; v = float(val) if isinstance(val,(int,float,str)) else None
      v = float(val) if isinstance(val,(int,float)) else (float(val) if isinstance(val,str) and val.replace('.','',1).isdigit() else None)
      if unit: num = convert_value(num, from_u, unit)
      if op=='eq' and v is not None and abs(num - v) < 1e-6: return True
      if op=='gte' and v is not None and num >= v: return True
      if op=='lte' and v is not None and num <= v: return True
      if op=='range' and (float(val['gte']) <= num <= float(val['lte'])): return True
    return False
  if op=='contains':
    t = str(val).lower(); return any(t in (o['value_text'] or '').lower() for o in cands)
  if op=='eq':
    t = str(val).lower(); return any(t == (o['value_text'] or '').lower() for o in cands)
  if op=='neq':
    t = str(val).lower(); return all(t != (o['value_text'] or '').lower() for o in cands)
  if op=='in':
    arr = [str(v).lower() for v in val]; return any((o['value_text'] or '').lower() in arr for o in cands)
  return False

def stats(db: DB, items: list[str]) -> dict:
  out = {"count": len(items)}
  # add per-attribute numeric summaries
  from collections import defaultdict
  acc = defaultdict(list)
  for iid in items:
    for o in db.fetch_occ_by_item(iid):
      if o['number_value'] is not None:
        acc[o['name_text'].lower()].append(float(o['number_value']))
  for k, arr in acc.items():
    out[f"num::{k}"] = {"min": min(arr), "max": max(arr), "avg": sum(arr)/len(arr)}
  return out

def main(query:str, save_as,plan=None):

  db = DB(DATABASE_PATH)
  plan = json.load(open(plan,'r',encoding='utf-8')) if plan else plan_from_query(query)
  logger.info(plan)
  items = [row['item_id'] for row in db.fetch_items()]
  matched = []
  for iid in items:
    must_ok = all(match_pred(db, iid, p) for p in plan.get('must',[]))
    must_not_ok = all(not match_pred(db, iid, p) for p in plan.get('must_not',[]))
    if must_ok and must_not_ok:
      matched.append(iid)

  out = {"matched_items": matched, "stats": stats(db, matched)}
  logger.info(json.dumps(out, ensure_ascii=False, indent=2))

  if save_as:
    db.save_filter(save_as, save_as, plan)

main("кабель медный 3x2.5 мм², напряжение ≥ 450В, не ПВХ" , "copper_3x2_450");''