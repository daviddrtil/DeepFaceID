"""Identity metadata for DFM swap models and FF (FaceFusion / inswapper) sources used in experiments.

DFM_IDENTITIES — keyed by .dfm filename stem (matches DeepFaceLive userdata layout).
FF_IDENTITIES  — keyed by source-image group key (output of `source_identity_key`),
                 since each FF identity has several source photos in `sources/`.
"""
import re


DFM_IDENTITIES = {
    # --- Male (dfm1–dfm6) ---
    'Bryan_Greynolds':       {'name': 'Bryan Greynolds',   'gender': 'male',   'birth_year': None, 'wiki': None,                'dfm_id': 'dfm1'},
    'Tim_Norland':           {'name': 'Tim Norland',       'gender': 'male',   'birth_year': None, 'wiki': None,                'dfm_id': 'dfm2'},
    'Dean_Wiesel':           {'name': 'Dean Wiesel',       'gender': 'male',   'birth_year': None, 'wiki': None,                'dfm_id': 'dfm3'},
    'Ewon_Spice':            {'name': 'Ewon Spice',        'gender': 'male',   'birth_year': None, 'wiki': None,                'dfm_id': 'dfm4'},
    'Tim_Chrys':             {'name': 'Tom Cruise',        'gender': 'male',   'birth_year': 1962, 'wiki': 'Tom_Cruise',        'dfm_id': 'dfm5'},
    'Daniel_Radcliffe_224':  {'name': 'Daniel Radcliffe',  'gender': 'male',   'birth_year': 1989, 'wiki': 'Daniel_Radcliffe',  'dfm_id': 'dfm6'},
    # --- Female (dfm11–dfm16) ---
    'Emily_Winston':         {'name': 'Emily Winston',     'gender': 'female', 'birth_year': None, 'wiki': None,                'dfm_id': 'dfm11'},
    'Millie_Park':           {'name': 'Millie Park',       'gender': 'female', 'birth_year': None, 'wiki': None,                'dfm_id': 'dfm12'},
    'Matilda_Bobbie':        {'name': 'Matilda Bobbie',    'gender': 'female', 'birth_year': None, 'wiki': None,                'dfm_id': 'dfm13'},
    'Natasha_Former':        {'name': 'Natasha Former',    'gender': 'female', 'birth_year': None, 'wiki': None,                'dfm_id': 'dfm14'},
    'Albica_Johns':          {'name': 'Albica Johns',      'gender': 'female', 'birth_year': None, 'wiki': None,                'dfm_id': 'dfm15'},
    'Natalie_Fatman':        {'name': 'Natalie Fatman',    'gender': 'female', 'birth_year': None, 'wiki': None,                'dfm_id': 'dfm16'},
}


FF_IDENTITIES = {
    # --- Male (ff1–ff6) ---
    'Tom_Holland':           {'name': 'Tom Holland',          'gender': 'male',   'birth_year': 1996, 'wiki': 'Tom_Holland_(actor)',   'ff_id': 'ff1'},
    'Ted_Mosby':             {'name': 'Josh Radnor (Ted Mosby)', 'gender': 'male', 'birth_year': 1974, 'wiki': 'Josh_Radnor',        'ff_id': 'ff2'},
    'Elon_Musk':             {'name': 'Elon Musk',            'gender': 'male',   'birth_year': 1971, 'wiki': 'Elon_Musk',             'ff_id': 'ff3'},
    'Vin_Diesel':            {'name': 'Vin Diesel',           'gender': 'male',   'birth_year': 1967, 'wiki': 'Vin_Diesel',            'ff_id': 'ff4'},
    'Tom_Cruise':            {'name': 'Tom Cruise',           'gender': 'male',   'birth_year': 1962, 'wiki': 'Tom_Cruise',            'ff_id': 'ff5'},
    'Neil_Patrick_Harris':   {'name': 'Neil Patrick Harris',  'gender': 'male',   'birth_year': 1973, 'wiki': 'Neil_Patrick_Harris',   'ff_id': 'ff6'},
    # --- Female (ff11–ff16) ---
    'Kristyna_Leichtova':    {'name': 'Kristýna Leichtová',   'gender': 'female', 'birth_year': 1985, 'wiki': 'Krist%C3%BDna_Leichtov%C3%A1', 'ff_id': 'ff11'},
    'Emilia_Clarke':         {'name': 'Emilia Clarke',        'gender': 'female', 'birth_year': 1986, 'wiki': 'Emilia_Clarke',         'ff_id': 'ff12'},
    'Jennifer_Lawrence':     {'name': 'Jennifer Lawrence',    'gender': 'female', 'birth_year': 1990, 'wiki': 'Jennifer_Lawrence',     'ff_id': 'ff13'},
    'Sydney_Sweeney':        {'name': 'Sydney Sweeney',       'gender': 'female', 'birth_year': 1997, 'wiki': 'Sydney_Sweeney',        'ff_id': 'ff14'},
    'Natalie_Portman':       {'name': 'Natalie Portman',      'gender': 'female', 'birth_year': 1981, 'wiki': 'Natalie_Portman',       'ff_id': 'ff15'},
    'Emma_Roberts':          {'name': 'Emma Roberts',         'gender': 'female', 'birth_year': 1991, 'wiki': 'Emma_Roberts',          'ff_id': 'ff16'},
}


def age_today(birth_year, today_year=2026):
    return None if birth_year is None else today_year - birth_year


def source_identity_key(filename):
    """Group key for a source image filename: strip extension, cut at the first digit.

    Examples:
        'Tom_Holland1.jpg'  -> 'Tom_Holland'
        'Ted_Mosby7.png'    -> 'Ted_Mosby'
        'Elon_Musk1.png'    -> 'Elon_Musk'
    """
    stem = filename.rsplit('.', 1)[0]
    m = re.match(r'^([^\d]*)', stem)
    key = m.group(1) if m else stem
    return key.rstrip('_').strip() or stem


def ff_id_for_filename(filename):
    """Resolve ff_id (e.g. 'ff3') from a source-image filename, or None if not in FF_IDENTITIES."""
    meta = FF_IDENTITIES.get(source_identity_key(filename))
    return meta['ff_id'] if meta else None
