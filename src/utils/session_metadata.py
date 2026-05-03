"""
Shared parser for the experimental session folder naming convention.

Folder format:
    real:  subject<N>_real_<YYYY-MM-DD_HH-MM-SS>
    fake:  subject<N>_<generator>[_<attack_model>]_fake_<YYYY-MM-DD_HH-MM-SS>

Examples:
    subject1_real_2026-05-03_00-13-59
    subject1_dfl_dfm_fake_2026-05-04_10-30-00
    subject2_ff_default_fake_2026-05-04_10-30-00
"""
import re

_TIMESTAMP_RE = re.compile(r'_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$')


def parse_session_name(folder_name):
    folder_name = folder_name or ''
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
    return {
        'subject_id': subject_id,
        'deepfake_label': label,
        'generator': generator,
        'attack_model': attack_model,
        'timestamp': timestamp,
    }
