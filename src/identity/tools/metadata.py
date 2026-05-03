DFM_IDENTITIES = {
    'Alice_Eve_320':         {'name': 'Alice Eve',         'gender': 'female', 'birth_year': 1982, 'wiki': 'Alice_Eve'},
    'Brett_Cooper_224':      {'name': 'Brett Cooper',      'gender': 'female', 'birth_year': 2001, 'wiki': 'Brett_Cooper_(commentator)'},
    'Daniel_Radcliffe_224':  {'name': 'Daniel Radcliffe',  'gender': 'male',   'birth_year': 1989, 'wiki': 'Daniel_Radcliffe'},
    'Emma_Roberts_224':      {'name': 'Emma Roberts',      'gender': 'female', 'birth_year': 1991, 'wiki': 'Emma_Roberts'},
    'IU_asian_224':          {'name': 'Lee Ji-eun',        'gender': 'female', 'birth_year': 1993, 'wiki': 'IU_(singer)'},
    'Jennifer_Lawrence_448': {'name': 'Jennifer Lawrence', 'gender': 'female', 'birth_year': 1990, 'wiki': 'Jennifer_Lawrence'},
    'Keanu_Reeves':          {'name': 'Keanu Reeves',      'gender': 'male',   'birth_year': 1964, 'wiki': 'Keanu_Reeves'},
    'Kim_Jarrey':            {'name': 'Jim Carrey',        'gender': 'male',   'birth_year': 1962, 'wiki': 'Jim_Carrey'},
    'Love_Quinn_224':        {'name': 'Love Quinn',        'gender': 'female', 'birth_year': 1995, 'wiki': 'Victoria_Pedretti'},
    'Melissa_Benoist_224':   {'name': 'Melissa Benoist',   'gender': 'female', 'birth_year': 1988, 'wiki': 'Melissa_Benoist'},
    'Rob_Doe':               {'name': 'Robert Downey Jr.', 'gender': 'male',   'birth_year': 1965, 'wiki': 'Robert_Downey_Jr.'},
    'Tim_Chrys':             {'name': 'Tom Cruise',        'gender': 'male',   'birth_year': 1962, 'wiki': 'Tom_Cruise'},
    'Viggo_Mortinnsen_384':  {'name': 'Viggo Mortensen',   'gender': 'male',   'birth_year': 1958, 'wiki': 'Viggo_Mortensen'},
}

SOURCE_IDENTITIES = {
    'asian_guy.jpg':          {'name': 'Asian male 1', 'gender': 'male'},
    'asian_guy2.jpg':         {'name': 'Asian male 2', 'gender': 'male'},
    'asian_guy3.jpg':         {'name': 'Asian male 3', 'gender': 'male'},
    'asian_guy4.jpg':         {'name': 'Asian male 4', 'gender': 'male'},
    'asian_guy5.jpg':         {'name': 'Asian male 5', 'gender': 'male'},
    'asian_guy6.jpg':         {'name': 'Asian male 6', 'gender': 'male'},
    'asian_guy7.jpg':         {'name': 'Asian male 7', 'gender': 'male'},
    'asian_guy8.jpg':         {'name': 'Asian male 8', 'gender': 'male'},
    'Biden.png':              {'name': 'Joe Biden',                'gender': 'male'},
    'Elon_Musk.png':          {'name': 'Elon Musk',                'gender': 'male'},
    'Elon_Musk_blue_bg.png':  {'name': 'Elon Musk',                'gender': 'male'},
    'firc.jpg':               {'name': 'Anton Firc',               'gender': 'male'},
    'Kim Chen Yin.png':       {'name': 'Kim Jong Un',              'gender': 'male'},
    'Lukashenko.png':         {'name': 'Alexander Lukashenko',     'gender': 'male'},
    'Putin.png':              {'name': 'Vladimir Putin',           'gender': 'male'},
    'Putin2.png':             {'name': 'Vladimir Putin',           'gender': 'male'},
    'rdj.jpg':                {'name': 'Robert Downey Jr.',        'gender': 'male'},
    'ted_mosby.jpg':          {'name': 'Josh Radnor (Ted Mosby)',  'gender': 'male'},
    'ted_mosby.png':          {'name': 'Josh Radnor (Ted Mosby)',  'gender': 'male'},
    'ted_mosby1.jpg':         {'name': 'Josh Radnor (Ted Mosby)',  'gender': 'male'},
    'ted_mosby1_1.jpg':       {'name': 'Josh Radnor (Ted Mosby)',  'gender': 'male'},
    'ted_mosby2.jpg':         {'name': 'Josh Radnor (Ted Mosby)',  'gender': 'male'},
    'ted_mosby3.jpg':         {'name': 'Josh Radnor (Ted Mosby)',  'gender': 'male'},
    'ted_mosby4.jpg':         {'name': 'Josh Radnor (Ted Mosby)',  'gender': 'male'},
    'ted_mosby5.jpg':         {'name': 'Josh Radnor (Ted Mosby)',  'gender': 'male'},
    'ted_mosby6.jpg':         {'name': 'Josh Radnor (Ted Mosby)',  'gender': 'male'},
    'tom_cruise.webp':        {'name': 'Tom Cruise',               'gender': 'male'},
}


def age_today(birth_year, today_year=2026):
    return None if birth_year is None else today_year - birth_year
