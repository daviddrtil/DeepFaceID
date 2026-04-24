import re
from pathlib import Path

ANALYZERS = ('spatial', 'frequency', 'temporal')
_FOLDER_RE = re.compile(r'^(.+?)(?:_(real|fake))?_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$')
_KV_RE = re.compile(r'(\w+)=\s*(\S+)')
_QUOTED_RE = re.compile(r"(\w+)='([^']*)'")
_CHALLENGE_RE = re.compile(r'challenge=(\d+)/(\d+)')


def _parse_val(val):
    if val == 'None':
        return None
    for conv in (int, float):
        try:
            return conv(val)
        except ValueError:
            continue
    return val


def parse_folder_name(name):
    m = _FOLDER_RE.match(name)
    return (m.group(1), m.group(2), m.group(3)) if m else (name, None, None)


def parse_stats_line(line):
    sections = line.split('|')
    if len(sections) < 5:
        return {}

    result = {}
    for key, val in _KV_RE.findall('|'.join(sections[:4])):
        result[key] = _parse_val(val)

    for field in ('face', 'hand', 'overlap'):
        if field in result:
            result[field] = result[field] == 1

    m = _CHALLENGE_RE.search(sections[4])
    if m:
        result['challenge_index'] = int(m.group(1))
        result['challenge_total'] = int(m.group(2))
    for key, val in _QUOTED_RE.findall(sections[4]):
        result[key] = None if val == 'None' else val

    return result


def load_session(stats_file):
    frames, summary = [], {}
    in_summary = False
    with open(stats_file, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith('--- SUMMARY'):
                in_summary = True
                continue
            if in_summary:
                if '=' in line:
                    key, val = line.split('=', 1)
                    key = key.strip()
                    if key in ('label', 'final_decision'):
                        summary[key] = val.strip()
                for a in ANALYZERS:
                    m = re.search(rf'{a}=(\S+?)\((\d+)\)', line)
                    if m:
                        raw = m.group(1)
                        summary[f'{a}_avg'] = None if raw == 'N/A' else float(raw)
                        summary[f'{a}_count'] = int(m.group(2))
            else:
                parsed = parse_stats_line(line)
                if parsed:
                    frames.append(parsed)
    return frames, summary


def find_sessions(outputs_dir):
    path = Path(outputs_dir)
    if not path.exists():
        return []
    sessions = []
    for folder in path.iterdir():
        stats_file = folder / 'stats.txt'
        if not folder.is_dir() or not stats_file.exists():
            continue
        name, label, ts = parse_folder_name(folder.name)
        sessions.append({
            'folder': folder, 'stats_file': stats_file,
            'session_name': name, 'label': label, 'timestamp': ts,
        })
    return sorted(sessions, key=lambda s: s['timestamp'] or '')
