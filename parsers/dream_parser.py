"""LLM-based dream → JSON graph. Uses OpenAI when OPENAI_API_KEY is set."""

from __future__ import annotations

import json
import os
import re

FEATURE_ORDER = [
    "emotional_valence",
    "scale",
    "agency",
    "familiarity",
    "threat_level",
    "transformation_likelihood",
    "centrality",
    "boundary_nature",
]

SYSTEM_PROMPT = """You are a symbolic graph extractor for dream analysis.

Extract a structured symbolic graph from dream reports. Focus on FUNCTIONAL and SYMBOLIC roles, not literal descriptions.

Abstract upward:
- "a tall building I couldn't escape" → enclosed_space node with features indicating overwhelming scale and entrapment
- "my boss but it wasn't really him" → authority node with low familiarity
- "a door I couldn't open" → threshold node with obstacle quality

Return ONLY valid JSON, no preamble, no markdown, exactly this structure:
{
  "nodes": [
    {
      "id": "n0",
      "label": "short symbolic label",
      "raw_description": "what user actually said",
      "features": {
        "emotional_valence": float (-1 to 1),
        "scale": float (0 to 1),
        "agency": float (0 to 1),
        "familiarity": float (0 to 1),
        "threat_level": float (0 to 1),
        "transformation_likelihood": float (0 to 1),
        "centrality": float (0 to 1),
        "boundary_nature": float (0 to 1)
      }
    }
  ],
  "edges": [
    {
      "from": "n0",
      "to": "n1",
      "relationship": "verb phrase describing relationship",
      "confidence": float (0 to 1)
    }
  ]
}

Only include edges with confidence > 0.5.
Minimum 2 nodes, maximum 12 nodes per dream.
Focus on symbols that carry emotional or structural weight."""


def _stub_graph(raw_text: str) -> dict:
    return {
        "nodes": [
            {
                "id": "n0",
                "label": "threshold",
                "raw_description": raw_text[:120],
                "features": {k: 0.55 for k in FEATURE_ORDER},
            },
            {
                "id": "n1",
                "label": "enclosed_space",
                "raw_description": raw_text[:120],
                "features": {k: 0.45 for k in FEATURE_ORDER},
            },
        ],
        "edges": [
            {
                "from": "n0",
                "to": "n1",
                "relationship": "leads into",
                "confidence": 0.9,
            }
        ],
    }


def _normalize_parsed(parsed: dict, raw_text: str) -> dict:
    for node in parsed.get("nodes", []):
        feat_dict = node.get("features") or {}
        if not isinstance(feat_dict, dict):
            feat_dict = {}
        node["features"] = [float(feat_dict.get(k, 0.5)) for k in FEATURE_ORDER]
        node["emotional_valence"] = float(feat_dict.get("emotional_valence", 0.0))
    parsed["raw_text"] = raw_text
    return parsed


def _extract_json_text(text: str) -> str:
    text = text.strip()
    fence = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", text, re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    return text


def parse_dream(raw_text: str) -> dict:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return _normalize_parsed(_stub_graph(raw_text), raw_text)

    try:
        from openai import OpenAI
    except ImportError:
        # Key is set but the SDK is missing — still run the pipeline on the stub graph.
        return _normalize_parsed(_stub_graph(raw_text), raw_text)

    model = os.environ.get("OPENAI_DREAM_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    client = OpenAI(api_key=key)

    completion = client.chat.completions.create(
        model=model,
        max_tokens=2000,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Dream report: {raw_text}"},
        ],
    )
    message = completion.choices[0].message
    response_text = (message.content or "").strip()
    response_text = _extract_json_text(response_text)

    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError:
        parsed = _stub_graph(raw_text)

    if "nodes" not in parsed:
        parsed = _stub_graph(raw_text)
    return _normalize_parsed(parsed, raw_text)
