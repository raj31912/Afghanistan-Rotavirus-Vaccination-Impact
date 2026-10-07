#!/usr/bin/env python3
"""Create SHA-256 manifests for derived outputs and release files."""
from pathlib import Path
import hashlib, pandas as pd
ROOT=Path(__file__).resolve().parent

def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def main():
    rows=[]
    for base in [ROOT/"outputs",ROOT/"figures"]:
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.name!="release_sha256.csv":
                rows.append({"relative_path":str(p.relative_to(ROOT)),"bytes":p.stat().st_size,"sha256":sha256(p)})
    out=ROOT/"outputs"/"release"/"release_sha256.csv"; out.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(out,index=False)
if __name__=="__main__": main()
