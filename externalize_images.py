"""Move embedded base64 images out of a notebook into image files.

Usage (from repo root, on the session branch):
    python externalize_images.py wise2627/notebooks/01_geschichte_und_use_cases.ipynb session-01

- Handles HTML <img src="data:..."> AND markdown ![alt](data:...)
- Handles png, jpeg, gif, webp and svg+xml
- Writes images to wise2627/notebooks/img/<notebook-stem>/<alt-slug>.<ext>
- Replaces the data URI with a raw.githubusercontent URL
- Leaves code-cell outputs untouched; keeps a .bak copy of the notebook
- Safe to re-run: already externalized images are skipped
"""
import base64, json, re, shutil, sys, unicodedata
from pathlib import Path

REPO = "kevisback/bda-course"
nb_path, branch = Path(sys.argv[1]), sys.argv[2]
img_dir = nb_path.parent / "img" / nb_path.stem
img_dir.mkdir(parents=True, exist_ok=True)
shutil.copy(nb_path, nb_path.with_suffix(".ipynb.bak"))

nb = json.loads(nb_path.read_text(encoding="utf-8"))
DATA = r'data:image/([\w.+-]+);base64,([A-Za-z0-9+/=\s]+)'
html_img = re.compile(r'(<img\b[^>]*?src=")' + DATA + r'("[^>]*>)')
md_img = re.compile(r'(!\[([^\]]*)\]\()' + DATA + r'(\))')
EXT = {"jpeg": "jpg", "svg+xml": "svg"}
used = {p.stem for p in img_dir.iterdir()}

def slug(text):
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", text.lower())[:40].strip("-") or "bild"

def save(alt, mime, b64):
    name, base, i = slug(alt), slug(alt), 2
    while name in used:
        name, i = f"{base}-{i}", i + 1
    used.add(name)
    path = img_dir / f"{name}.{EXT.get(mime, mime)}"
    path.write_bytes(base64.b64decode(re.sub(r"\s", "", b64)))
    print(f"  {path.name}")
    return f"https://raw.githubusercontent.com/{REPO}/{branch}/{path.as_posix()}"

def repl_html(m):
    alt = re.search(r'alt="([^"]*)"', m.group(0))
    return m.group(1) + save(alt.group(1) if alt else "bild", m.group(2), m.group(3)) + m.group(4)

def repl_md(m):
    return m.group(1) + save(m.group(2) or "bild", m.group(3), m.group(4)) + m.group(5)

for cell in nb["cells"]:
    if cell["cell_type"] != "markdown":
        continue
    src = "".join(cell["source"])
    new = md_img.sub(repl_md, html_img.sub(repl_html, src))
    if new != src:
        cell["source"] = new.splitlines(keepends=True)

nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Done. Images in {img_dir}, backup at {nb_path.with_suffix('.ipynb.bak')}")
