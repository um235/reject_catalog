

from __future__ import annotations
from typing import List, Dict, Any, Optional
import os
import json
import logging

import httpx
import numpy as np
from openai import OpenAI

from config import EMBED_MODEL, CHAT_MODEL, PROXY
from logging_config import setup_logging

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
setup_logging()
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# OpenAI Client
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Embeddings + Cosine
# ------------------------------------------------------------------------------
def embed(texts: List[str]) -> List[List[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def cosine(a: List[float], b: List[float]) -> float:
    va, vb = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(np.dot(va, vb) / denom) if denom else 0.0

# ------------------------------------------------------------------------------
# Structured Output Helper
# ------------------------------------------------------------------------------
def llm_structured(schema: Dict[str, Any], system: str, user: str, temperature: float = 0) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema.get("name", "Schema"),
                "strict": True,
                "schema": schema["schema"],
            },
        },
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    content = resp.choices[0].message.content
    logger.debug("LLM structured content: %s", content)
    try:
        return json.loads(content)
    except Exception as e:
        logger.exception("Failed to parse structured content as JSON: %s", e)
        return {}

# ------------------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------------------

EXTRACT_SCHEMA: Dict[str, Any] = {
    "name": "ExtractPairs",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "pairs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name":   {"type": "string"},
                        "value":  {"anyOf": [{"type": "string"}, {"type": "number"}]},
                        "unit":   {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "number": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                    },
                    "required": ["name", "value", "unit", "number"],
                },
            }
        },
        "required": ["pairs"],
    },
}

# ✅ FIXED: use top-level $defs + $ref
PLAN_SCHEMA: Dict[str, Any] = {
    "name": "FilterPlan",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "$defs": {
            "Clause": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "attr": {"type": "string"},
                    "op":   {"type": "string", "enum": ["eq", "neq", "in", "gte", "lte", "range", "contains"]},
                    "value": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "number"},
                            {"type": "array", "items": {"anyOf": [{"type": "string"}, {"type": "number"}]}},
                        ]
                    },
                    "unit": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                },
                "required": ["attr", "op", "value", "unit"],
            }
        },
        "properties": {
            "must":     {"type": "array", "items": {"$ref": "#/$defs/Clause"}},
            "must_not": {"type": "array", "items": {"$ref": "#/$defs/Clause"}},
            "should":   {"type": "array", "items": {"$ref": "#/$defs/Clause"}},
        },
        "required": ["must", "must_not", "should"],
    },
}

CONVERT_SCHEMA: Dict[str, Any] = {
    "name": "UnitConvert",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "ok":    {"type": "boolean"},
            "value": {"anyOf": [{"type": "number"}, {"type": "null"}]},
            "unit":  {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
        "required": ["ok", "value", "unit"],
    },
}

# ------------------------------------------------------------------------------
# System Prompts
# ------------------------------------------------------------------------------
EXTRACT_SYS = """
Ты извлекаешь характеристики материалов из русских названий/описаний.
Верни JSON с массивом `pairs`, каждый элемент:
{ "name": string, "value": string|number, "unit": string|null, "number": number|null }.
Если видишь `3x2.5 мм²` → {name:"жилы", value:3, number:3} и {name:"сечение", value:2.5, unit:"мм2", number:2.5}.
Если напряжение `450/750 В` → возьми минимум (450) как {name:"номинальное напряжение", unit:"В", number:450, value:450}.
Для отрицания `не ПВХ` → {name:"материал оболочки", value:"!ПВХ"}.
Стандарты включай как {name:"ГОСТ/стандарт", value:"ГОСТ 31996-2012"}.
Используй русские названия характеристик и значений.
"""

PLAN_SYS = """
You are a procurement assistant. Build a FILTER PLAN from a natural-language query.
Prefer numeric comparisons with units when possible. Attribute names may include:
material, cores, cross section, rated voltage, sheath material, standard.
Return JSON with keys: must, must_not, should.
"""

CONVERT_SYS = """
You convert numeric values between units. Respond with
{"ok":true, "value": number, "unit": string} or {"ok":false, "value": null, "unit": null}.
Supported units: common SI and electrical (mm2, mm, m, V, A, kg, g).
If source and target are equivalent, return the same number.
"""

# ------------------------------------------------------------------------------
# Public API (Structured)
# ------------------------------------------------------------------------------
def extract_pairs(title: str, desc: str) -> List[Dict[str, Any]]:
    j = llm_structured(
        schema=EXTRACT_SCHEMA,
        system=EXTRACT_SYS,
        user=f"TITLE: {title}\nDESC: {desc}",
        temperature=0,
    )
    return j.get("pairs", [])

def plan_from_query(query: str) -> Dict[str, Any]:
    return llm_structured(
        schema=PLAN_SCHEMA,
        system=PLAN_SYS,
        user=query,
        temperature=0,
    )

def convert_value(value: float, from_unit: Optional[str], to_unit: Optional[str]) -> float:
    if not to_unit or not from_unit or from_unit.lower() == to_unit.lower():
        return value
    j = llm_structured(
        schema=CONVERT_SCHEMA,
        system=CONVERT_SYS,
        user=f"value={value}, from={from_unit}, to={to_unit}",
        temperature=0,
    )
    if j.get("ok") and j.get("value") is not None:
        try:
            return float(j["value"])
        except (TypeError, ValueError):
            logger.warning("Conversion returned non-float value: %r", j["value"])
    return value

# ------------------------------------------------------------------------------
# Optional: generic JSON helper (no schema)
# ------------------------------------------------------------------------------
def llm_json(system: str, user: str, temperature: float = 0) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    content = resp.choices[0].message.content
    logger.debug("LLM json content: %s", content)
    try:
        return json.loads(content)
    except Exception as e:
        logger.exception("Failed to parse JSON content: %s", e)
        return {}

# ------------------------------------------------------------------------------
# Demo
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    sample_title = "Кабель силовой ВВГнг 3x2.5, медь; 450/750 В; не ПВХ; ГОСТ 31996-2012"
    sample_desc  = "Для внутренней прокладки. Плоский. Сечение 2.5 мм²."
    try:
        pairs = extract_pairs(sample_title, sample_desc)
        print("EXTRACTED PAIRS:", json.dumps(pairs, ensure_ascii=False, indent=2))
    except Exception as e:
        logger.error("extract_pairs failed: %s", e)

    try:
        plan = plan_from_query("Найди медный кабель 3 жилы, сечение 2.5 мм2, напряжение не ниже 450 В, стандарт ГОСТ")
        print("FILTER PLAN:", json.dumps(plan, ensure_ascii=False, indent=2))
    except Exception as e:
        logger.error("plan_from_query failed: %s", e)

    try:
        conv = convert_value(2.5, "mm2", "mm2")
        print("CONVERTED VALUE:", conv)
    except Exception as e:
        logger.error("convert_value failed: %s", e)
