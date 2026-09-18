from __future__ import annotations
import json,platform,sys
from pathlib import Path

def runtime_snapshot():
    return {"python":sys.version,"platform":platform.platform(),"machine":platform.machine(),"processor":platform.processor()}

def write_runtime_snapshot(path:str|Path):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(runtime_snapshot(),indent=2)+"\n")
