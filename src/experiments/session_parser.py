import re
import pandas as pd
from pathlib import Path


# Folder naming convention:
# real:  subject<N>_real_<YYYY-MM-DD_HH-MM-SS>
# fake:  subject<N>_<generator>[_<attack_model>]_fake_<YYYY-MM-DD_HH-MM-SS>
# generator examples: dfl, ff
# attack_model examples: dfm, insight, default
_TIMESTAMP_RE = re.compile(r'_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$')


def parse_session_name(folder_name):
    m = _TIMESTAMP_RE.search(folder_name)
    timestamp = m.group(1) if m else None
    head = folder_name[:m.start()] if m else folder_name
    tokens = head.split('_') if head else []
    if not tokens:
        return {'subject_id': None, 'deepfake_label': None, 'generator': None, 'attack_model': None, 'timestamp': timestamp}
    subject_id = tokens[0] or None
    label = tokens[-1] if tokens[-1] in ('real', 'fake') else None
    middle = tokens[1:-1] if label else tokens[1:]
    if label != 'fake' or not middle:
        generator, attack_model = (None, None)
    elif len(middle) == 1:
        generator, attack_model = (middle[0], None)
    else:
        generator, attack_model = (middle[0], '_'.join(middle[1:]))
    return {'subject_id': subject_id, 'deepfake_label': label, 'generator': generator, 'attack_model': attack_model, 'timestamp': timestamp}


def _load_csv_frames(path):
    df = pd.read_csv(path)
    for col in ('pose', 'occlusions', 'expressions'):
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x.split('|') if isinstance(x, str) and x else [])
    return df.where(pd.notna(df), None).to_dict('records')


def _load_csv_summary(path):
    df = pd.read_csv(path)
    if df.empty:
        return {}
    row = df.iloc[0]
    return {k: (None if pd.isna(v) else (v.item() if hasattr(v, 'item') else v)) for k, v in row.items()}


def _load_csv_actions(path):
    df = pd.read_csv(path)
    return df.where(pd.notna(df), None).to_dict('records')


def load_session(stats_file):
    stats_file = Path(stats_file)
    frames = _load_csv_frames(stats_file)
    summary_file = stats_file.with_name('summary.csv')
    summary = _load_csv_summary(summary_file) if summary_file.exists() else {}
    actions_file = stats_file.with_name('actions.csv')
    actions = _load_csv_actions(actions_file) if actions_file.exists() else []
    return frames, summary, actions


def find_sessions(outputs_dir):
    path = Path(outputs_dir)
    if not path.exists():
        return []
    sessions = []
    for folder in sorted(path.iterdir()):
        if not (folder.is_dir() and (folder / 'stats.csv').exists()):
            continue
        meta = parse_session_name(folder.name)
        sessions.append({
            'folder': folder,
            'stats_file': folder / 'stats.csv',
            'session_name': folder.name,
            **meta,
        })
    return sessions

