from __future__ import annotations

from . import *  # noqa
import glob
import os
import importlib

modules = glob.glob(f"{os.path.dirname(__file__)}/**/*.py", recursive=True)
for m in modules:
    m = os.path.relpath(m)
    if os.path.basename(m) == os.path.basename(__file__):
        continue
    importlib.import_module(m.replace("/", ".").replace(".py", ""))
del modules