"""Validate paper/supplementary.tex on its own.

It is a standalone document (its own \\documentclass and \\bibliography), so
check_tex_structure.py -- which follows manuscript.tex's \\input chain -- never sees it. Its refs
must resolve within itself and its \\cite keys must exist in the shared references.bib.
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

ROOT = Path("c:/Users/ojasw/Documents/Data_collection_setup/paper")
src = io.open(ROOT / "supplementary.tex", encoding="utf-8").read()
body = re.sub(r"(?m)^\s*%.*$", "", src)

labels = set(re.findall(r"\\label\{([^}]*)\}", body))
refs = set(re.findall(r"\\ref\{([^}]*)\}", body))
cites: set[str] = set()
for m in re.findall(r"\\cite\{([^}]*)\}", body):
    cites.update(k.strip() for k in m.split(","))

bib = io.open(ROOT / "references.bib", encoding="utf-8", errors="ignore").read()
bibkeys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib))

ok = True
print("labels :", sorted(labels))
print("refs   :", sorted(refs))
unresolved = sorted(refs - labels)
print("unresolved refs :", unresolved or "none")
ok &= not unresolved

print("cites  :", sorted(cites))
missing = sorted(c for c in cites if c not in bibkeys)
print("cites not in references.bib :", missing or "none")
ok &= not missing

nb = body.count("{") - body.count("}")
print(f"braces : {body.count('{')} open / {body.count('}')} close ->", "OK" if nb == 0 else "MISMATCH")
ok &= nb == 0

ctrl = [i for i, c in enumerate(src) if ord(c) < 32 and ord(c) not in (9, 10)]
print("control characters :", len(ctrl))
ok &= not ctrl

dollars = body.count("$")
print(f"inline math $ count : {dollars} ->", "OK" if dollars % 2 == 0 else "ODD -- unclosed")
ok &= dollars % 2 == 0

print()
print("SUPPLEMENTARY OK" if ok else "SUPPLEMENTARY HAS PROBLEMS")
sys.exit(0 if ok else 1)
