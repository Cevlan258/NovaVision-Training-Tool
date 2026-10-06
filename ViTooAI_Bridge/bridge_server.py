# -*- coding: utf-8 -*-
"""将兼容入口转交给 App_NovaVision 下的桥接实现。"""
import os
import runpy
import sys

_TARGET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "App_NovaVision", "ViTooAI_Bridge"))
_TARGET = os.path.join(_TARGET_DIR, os.path.basename(__file__))
if not os.path.isfile(_TARGET):
    raise FileNotFoundError(f"桥接实现不存在: {_TARGET}")
if _TARGET_DIR not in sys.path:
    sys.path.insert(0, _TARGET_DIR)
_namespace = runpy.run_path(_TARGET, run_name=__name__)
globals().update({
    key: value for key, value in _namespace.items()
    if key not in {"__name__", "__file__", "__package__", "__loader__", "__spec__", "__builtins__"}
})
