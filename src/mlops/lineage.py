from __future__ import annotations
import os,platform,subprocess
from dataclasses import dataclass,asdict
from datetime import datetime,timezone

@dataclass(frozen=True)
class Lineage:
    git_commit:str
    dataset_id:str
    feature_version:str
    label_version:str
    validation_version:str
    python_version:str
    created_at:str
    @classmethod
    def create(cls,**kwargs):
        try:
            git=subprocess.check_output(["git","rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
        except Exception:
            git=os.getenv("GIT_COMMIT","unknown")
        return cls(git,kwargs.get("dataset_id","unknown"),kwargs.get("feature_version","unknown"),
                   kwargs.get("label_version","unknown"),kwargs.get("validation_version","unknown"),
                   platform.python_version(),datetime.now(timezone.utc).isoformat())
    def to_dict(self): return asdict(self)
