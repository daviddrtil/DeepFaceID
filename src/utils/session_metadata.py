"""
Shared parser for the experimental session folder naming convention.

Folder format:
    real:  subject<N>_<gender>_real_<YYYY-MM-DD_HH-MM-SS>
    fake:  subject<N>_<gender>_<generator>[_<attack_model>]_fake_<YYYY-MM-DD_HH-MM-SS>

Examples:
    subject1_male_real_2026-05-04_10-30-00
    subject1_male_dfl_dfm_fake_2026-05-04_10-30-00
    subject2_female_ff_default_fake_2026-05-04_10-30-00
"""
import re

_TIMESTAMP_RE = re.compile(r'_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$')
_GENDERS = {'male', 'female'}
_LABELS = {'real', 'fake'}


def parse_session_name(folder_name):
    folder_name = folder_name or ''
    m = _TIMESTAMP_RE.search(folder_name)
    timestamp = m.group(1) if m else None
    head = folder_name[:m.start()] if m else folder_name
    tokens = head.split('_') if head else []
    blank = {'subject_id': None, 'gender': None, 'deepfake_label': None,
             'generator': None, 'attack_model': None, 'timestamp': timestamp}
    if not tokens:
        return blank

    subject_id = tokens[0] or None
    label = tokens[-1] if tokens[-1] in _LABELS else None
    middle = tokens[1:-1] if label else tokens[1:]

    gender = middle[0] if middle and middle[0] in _GENDERS else None
    rest = middle[1:] if gender else middle

    if label != 'fake' or not rest:
        generator, attack_model = (None, None)
    elif len(rest) == 1:
        generator, attack_model = (rest[0], None)
    else:
        generator, attack_model = (rest[0], '_'.join(rest[1:]))

    return {
        'subject_id': subject_id,
        'gender': gender,
        'deepfake_label': label,
        'generator': generator,
        'attack_model': attack_model,
        'timestamp': timestamp,
    }
