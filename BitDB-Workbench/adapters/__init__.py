"""
Prototype Adapter Factory & Discovery
"""

from typing import Dict
from config import PROTOTYPE_DEFS
from .base import BasePrototypeAdapter
from .prototype4_adapter import Prototype4Adapter
from .prototype3_adapter import Prototype3Adapter
from .legacy_adapter import LegacyTreeAdapter

def get_adapter(prototype_key: str) -> BasePrototypeAdapter:
    if prototype_key not in PROTOTYPE_DEFS:
        raise ValueError(f"Unknown prototype: {prototype_key}")
    info = PROTOTYPE_DEFS[prototype_key]
    ptype = info.get("type", "")
    if ptype == "columnar_er2":
        return Prototype4Adapter(prototype_key, info)
    elif ptype == "avalanche_chain":
        return Prototype3Adapter(prototype_key, info)
    else:
        return LegacyTreeAdapter(prototype_key, info)

def get_all_adapters() -> Dict[str, BasePrototypeAdapter]:
    adapters = {}
    for key in PROTOTYPE_DEFS:
        adapters[key] = get_adapter(key)
    return adapters
