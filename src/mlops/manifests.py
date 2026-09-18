from __future__ import annotations
import hashlib,json
from pathlib import Path

def file_sha256(path:str|Path)->str:
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def write_dataset_manifest(path:str|Path,payload:dict)->None:
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")

def read_manifest(path:str|Path)->dict:
    return json.loads(Path(path).read_text())
