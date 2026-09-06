"""Structural sanity checks on the manuscript sources. No LaTeX available locally, so this is
the only pre-compile check we get: brace balance, math-mode balance, and label/ref pairing.
"""
import io
import re
from pathlib import Path

ROOT = Path("c:/Users/ojasw/Documents/Data_collection_setup/paper")
STRIP = re.compile(r"(?<!\\)%.*")

ok = True
for f in sorted(ROOT.glob("section*.tex")) + [ROOT / "manuscript.tex"]:
    raw = io.open(f, encoding="utf-8").read()
    t = STRIP.sub("", raw)
    o, c = t.count("{"), t.count("}")
    d = t.count("$")
    bad = (o != c) or (d % 2)
    ok = ok and not bad
    print(f"  {f.name:<28} braces {o:>4}/{c:<4} {'OK ' if o == c else 'MISMATCH'}"
          f"   $ {d:>3} {'OK' if d % 2 == 0 else 'ODD'}")

print("\nlabel / ref pairing across the whole manuscript:")
alltex = "".join(STRIP.sub("", io.open(f, encoding="utf-8").read())
                 for f in list(ROOT.glob("section*.tex")) + [ROOT / "manuscript.tex"])
labels = set(re.findall(r"\\label\{([^}]*)\}", alltex))
refs = set(re.findall(r"\\(?:eq)?ref\{([^}]*)\}", alltex))
missing = sorted(refs - labels)
print(f"  {len(labels)} labels, {len(refs)} distinct refs")
if missing:
    ok = False
    print(f"  DANGLING REFS (will render as ??): {missing}")
else:
    print("  every \\ref resolves to a \\label")
unused = sorted(labels - refs)
if unused:
    print(f"  labels never referenced (harmless): {unused}")

print("\ncollapsed-escape corruption (a shell heredoc turning \\n, \\t, \\r into whitespace):")
# This class of bug destroyed four \num{} macros in Section 3 and NOTHING caught it: braces stay
# balanced, every \ref still resolves, and the text reads fine to a skim -- it would simply have
# typeset as "um1.31". The signature is a LaTeX control word that has lost its backslash and now
# begins a line, so scan for known macro tails orphaned at a line start.
ORPHAN = re.compile(r"^\s*(um|umrange|ocite|ewcommand|ewline|abel|ef|eqref|ind|ot|oindent|"
                    r"extbf|extit|imes|iny|able|extwidth)\{", re.M)
dirty = []
for f in sorted(ROOT.glob("*.tex")):
    for m in ORPHAN.finditer(io.open(f, encoding="utf-8").read()):
        dirty.append(f"{f.name}: line begins '{m.group(1)}{{' -- lost its backslash")
if dirty:
    ok = False
    for d in dirty:
        print(f"  !! {d}")
else:
    print("  clean")

print("\nblocklisted numbers from the manuscript skill:")
BLOCK = ["13.6", "41.4", "36.5", "79.5", "0.480", "0.350"]
for b in BLOCK:
    hits = [f.name for f in ROOT.glob("section*.tex")
            if b in STRIP.sub("", io.open(f, encoding="utf-8").read())]
    if hits:
        ok = False
        print(f"  !! {b} appears in {hits}")
print("  clean" if ok else "  SEE ABOVE")

# ---------------------------------------------------------------------------------------------
# Submission placeholders. The author name, repository DOI and acknowledgements are all RENDERED
# INTO THE PDF, so an unfilled one ships as visible text rather than as a comment nobody sees.
# They are defined as macros in manuscript.tex; this fails the gate while any still says so.
# ---------------------------------------------------------------------------------------------
print("\nsubmission placeholders (must be filled before the paper leaves the building):")
man = STRIP.sub("", io.open(ROOT / "manuscript.tex", encoding="utf-8").read())
unfilled = [ln.strip() for ln in man.splitlines() if "TO BE ADDED" in ln]
if unfilled:
    ok = False
    for u in unfilled:
        print(f"  !! {u[:96]}")
    print(f"  {len(unfilled)} placeholder(s) still unfilled -- the PDF will print them verbatim")
else:
    print("  clean")

# ---------------------------------------------------------------------------------------------
# Control characters. A shell heredoc silently turns \n \a \b \f \r \v into the control byte, so
# \num{1.31} ships as um{1.31} and \acknowledgementtext as <BEL>cknowledgementtext. Both have
# happened in this project. Every C0 byte except tab and newline is a macro that lost its
# backslash.
# ---------------------------------------------------------------------------------------------
print("\ncontrol characters (a heredoc that ate a backslash):")
CTRL = {c for c in range(0x20) if c not in (0x09, 0x0A)}
LOST = {0x07: "a", 0x08: "b", 0x0C: "f", 0x0B: "v", 0x0D: "r", 0x1B: "e"}
found = False
for f in sorted(ROOT.glob("*.tex")):
    raw = io.open(f, encoding="utf-8").read()
    for i, ch in enumerate(raw):
        if ord(ch) in CTRL:
            guess = LOST.get(ord(ch))
            note = f" (was backslash-{guess})" if guess else ""
            print(f"  !! {f.name}: 0x{ord(ch):02X} at offset {i}{note}"
                  f"  ...{raw[max(0, i - 25):i + 25]!r}...")
            found = True
            ok = False
            break
if not found:
    print("  clean")

print(f"\n{'ALL STRUCTURAL CHECKS PASSED' if ok else 'PROBLEMS FOUND'}")
