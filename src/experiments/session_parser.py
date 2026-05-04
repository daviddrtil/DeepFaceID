import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.session_metadata import parse_session_name


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

