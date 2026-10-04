"""Move embedded base64 images out of a notebook into image files.

Usage (from repo root, on the session branch):
    python externalize_images.py wise2627/notebooks/01_geschichte_und_use_cases.ipynb session-01

- Writes images to wise2627/notebooks/img/<notebook-stem>/<alt-slug>.<ext>
- Replaces data:image/...;base64,... in markdown cells with raw.githubusercontent URLs
- Leaves code-cell outputs untouched; keeps a .bak copy of the notebook
"""
import base64, json, re, shutil, sys, unicodedata
from pathlib import Path

REPO = "kevisback/bda-course"
nb_path, branch = Path(sys.argv[1]), sys.argv[2]
img_dir = nb_path.parent / "img" / nb_path.stem
img_dir.mkdir(parents=True, exist_ok=True)
shutil.copy(nb_path, nb_path.with_suffix(".ipynb.bak"))

nb = json.loads(nb_path.read_text(encoding="utf-8"))
tag = re.compile(r'<img\b[^>]*?src="data:image/(\w+);base64,([^"]+)"[^>]*>')
used = set()

def slug(text):
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", text.lower())[:40].strip("-") or "bild"

def repl(m):
    ext = {"jpeg": "jpg"}.get(m.group(1), m.group(1))
    alt = re.search(r'alt="([^"]*)"', m.group(0))
    name = slug(alt.group(1) if alt else "bild")
    base, i = name, 2
    while name in used:
        name, i = f"{base}-{i}", i + 1
    used.add(name)
    (img_dir / f"{name}.{ext}").write_bytes(base64.b64decode(m.group(2)))
    rel = (img_dir / f"{name}.{ext}").as_posix()
    url = f"https://raw.githubusercontent.com/{REPO}/{branch}/{rel}"
    print(f"  {name}.{ext}")
    return m.group(0).replace(m.group(0)[m.group(0).index("data:image"):m.group(0).index('"', m.group(0).index("data:image"))], url)

for cell in nb["cells"]:
    if cell["cell_type"] != "markdown":
        continue
    src = "".join(cell["source"])
    new = tag.sub(repl, src)
    if new != src:
        cell["source"] = new.splitlines(keepends=True)

nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Done. Images in {img_dir}, backup at {nb_path.with_suffix('.ipynb.bak')}")
