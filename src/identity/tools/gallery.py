"""Build gallery.md showing every DFM identity (with info) and every source image (image only)."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metadata import DFM_IDENTITIES, SOURCE_IDENTITIES, age_today, build_source_id_map

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from identity.tools.pairings import DFM_DIR, PREVIEWS_DIR, SRC_DIR

OUT = Path(__file__).resolve().parent / "gallery.md"
THUMB_W = 200


def _dfm_thumb(stem):
    for ext in ('.jpg', '.png'):
        p = PREVIEWS_DIR / f"{stem}{ext}"
        if p.exists():
            return p.relative_to(OUT.parent).as_posix()
    return None


def _grid(cells, cols=4):
    rows = []
    for i in range(0, len(cells), cols):
        chunk = cells[i:i + cols]
        if len(chunk) < cols:
            chunk = chunk + [''] * (cols - len(chunk))
        rows.append("| " + " | ".join(chunk) + " |")
    head = "| " + " | ".join([" "] * cols) + " |"
    sep = "| " + " | ".join(["---"] * cols) + " |"
    return "\n".join([head, sep] + rows)


def _dfm_cell(stem, meta):
    thumb = _dfm_thumb(stem)
    age = age_today(meta.get('birth_year'))
    gender = meta.get('gender') or '?'
    dfm_id = meta.get('dfm_id')
    id_tag = f" **({dfm_id})**" if dfm_id else ""
    line2 = gender if age is None else f"{gender}, age {age}"
    info = f"**{stem}**{id_tag}<br/>{line2}"
    img = f'<img src="{thumb}" width="{THUMB_W}"/>' if thumb else '_(no preview)_'
    return f"{img}<br/>{info}"


def _by_dfm_id(items):
    def key(kv):
        dfm_id = kv[1].get('dfm_id') or ''
        m = re.match(r'^dfm(\d+)$', dfm_id)
        return (0, int(m.group(1))) if m else (1, dfm_id)
    return sorted(items, key=key)


_groups = {'male': [], 'female': [], 'other': []}
for stem, meta in DFM_IDENTITIES.items():
    _groups[meta.get('gender') if meta.get('gender') in ('male', 'female') else 'other'].append((stem, meta))

lines = ["# Identity gallery", ""]
for label, key in [("DFM swap models — male", 'male'),
                   ("DFM swap models — female", 'female'),
                   ("DFM swap models — other", 'other')]:
    items = _by_dfm_id(_groups[key])
    if not items:
        continue
    lines += [f"## {label}", "", _grid([_dfm_cell(s, m) for s, m in items], cols=4), ""]

lines += ["## Source images", ""]
src_id_map = build_source_id_map(SOURCE_IDENTITIES)
src_cells = []
for fname in SOURCE_IDENTITIES:
    p = SRC_DIR / fname
    if not p.exists():
        continue
    src_id = src_id_map.get(fname)
    id_tag = f' **({src_id})**' if src_id else ''
    src_cells.append(f'<img src="{p.as_posix()}" width="{THUMB_W}"/><br/><sub>{fname}{id_tag}</sub>')
lines.append(_grid(src_cells, cols=4))
lines.append("")

OUT.write_text("\n".join(lines), encoding='utf-8')
print(f"Wrote {OUT}")
