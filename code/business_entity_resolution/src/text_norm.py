"""Text normalisation for business names and addresses.

Every record is turned into a few normalised views:
  name_norm  canonical tokens (abbreviations unified, legal words kept)
  core       name tokens with legal suffixes / stopwords removed
  skel       consonant skeleton of each core token (robust to typos and
             transliteration, e.g. 'raam maarketting' and 'ram marketing'
             both become 'rm mrktng')
  concat     core tokens joined without spaces (matches 'wilfordhancock.com'
             against 'Wilford Hancock LLC')
  addr_norm  canonical address tokens
  addr_tok   informative address words (street / area / city names)
  postcode   5 or 6 digit code found after the first address component
  house      first number in the first address component
"""
import re
from multiprocessing import Pool

from unidecode import unidecode

NAME_CANON = {
    "limited": "ltd", "private": "pvt", "corporation": "corp", "incorporated": "inc",
    "company": "co", "brothers": "bros", "international": "intl", "enterprises": "ent",
    "enterprise": "ent", "associates": "assoc", "services": "svc", "service": "svc",
    "technologies": "tech", "technology": "tech", "solutions": "soln", "solution": "soln",
    "industries": "ind", "industry": "ind", "manufacturing": "mfg", "centre": "center",
    "saint": "st", "mount": "mt", "doing": "dba",
}
LEGAL_STOP = {
    "ltd", "pvt", "llc", "inc", "corp", "co", "llp", "plc", "lp", "pllc", "pc", "gmbh",
    "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "the", "and", "of", "dba", "a",
    "de", "la", "le", "les", "des", "du", "et", "l", "d", "com", "net", "org", "www",
    "in", "fr", "us", "opc", "praa", "pra", "li", "pvtltd", "limted",
}
ADDR_CANON = {
    "road": "rd", "street": "st", "avenue": "ave", "av": "ave", "boulevard": "blvd",
    "drive": "dr", "lane": "ln", "court": "ct", "place": "pl", "highway": "hwy",
    "suite": "ste", "apartment": "apt", "building": "bldg", "bldg": "bldg", "floor": "flr",
    "fl": "flr", "near": "nr", "opposite": "opp", "opp": "opp", "sector": "sec",
    "north": "n", "south": "s", "east": "e", "west": "w", "parkway": "pkwy",
    "circle": "cir", "terrace": "ter", "square": "sq", "number": "no", "house": "h",
    "hno": "h", "nagar": "ngr", "colony": "col", "chowk": "chk", "marg": "mg",
    # US states -> postal codes
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv", "ohio": "oh",
    "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa", "wisconsin": "wi",
    "wyoming": "wy",
}
# Multi-word US states handled before tokenising
ADDR_PHRASES = [
    ("new hampshire", "nh"), ("new jersey", "nj"), ("new mexico", "nm"), ("new york", "ny"),
    ("north carolina", "nc"), ("north dakota", "nd"), ("rhode island", "ri"),
    ("south carolina", "sc"), ("south dakota", "sd"), ("west virginia", "wv"),
]
ADDR_STOP = set(ADDR_CANON.values()) | {
    "no", "h", "flat", "unit", "shop", "plot", "khasra", "kh", "gali", "block", "phase",
    "india", "usa", "france", "united", "states", "rue", "avenue", "bis", "ter", "and",
    "the", "of", "de", "la", "le", "du", "des", "main", "cross", "stage", "tower", "wing",
    "ground", "first", "second", "third", "1st", "2nd", "3rd", "4th", "po", "ps", "dist",
    "district", "tehsil", "taluka", "village", "vill", "post", "office", "market", "bazar",
}

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_VOWELS = re.compile(r"[aeiouhy]")
_REPEAT = re.compile(r"(.)\1+")
_POSTCODE = re.compile(r"\b(\d{3}\s\d{3}|\d{6}|\d{5})\b")
_NUMBER = re.compile(r"\d+")


def _tokens(s):
    """Lowercase, transliterate non-ASCII to Latin, split on non-alphanumerics."""
    if not s:
        return []
    if not s.isascii():
        s = unidecode(s)
    s = s.lower().replace("&", " and ")
    return _NON_ALNUM.sub(" ", s).split()


def skeleton(tok):
    """Consonant skeleton of a token: keep first letter, drop vowels, h and y, merge repeats."""
    if tok.isdigit():
        return tok
    # v3: extra folds seen in back-transliterated Indian names in the training data
    # ('praaibhett' ~ 'private', 'lkssmii' ~ 'laxmi', 'mainejmentt' ~ 'management')
    tok = (tok.replace("ph", "f").replace("bh", "v").replace("ck", "k").replace("x", "ks")
           .replace("q", "k").replace("c", "k").replace("z", "s").replace("j", "g"))
    return _REPEAT.sub(r"\1", tok[0] + _VOWELS.sub("", tok[1:]))


LEGAL_SKEL = {skeleton(t) for t in ("limited", "private", "pvt", "ltd", "corporation", "incorporated", "llc")}


def normalize_name(name):
    """Return (name_norm, core, skel, concat) for a business name."""
    toks = [NAME_CANON.get(t, t) for t in _tokens(name)]
    core = [t for t in toks if t not in LEGAL_STOP and skeleton(t) not in LEGAL_SKEL]
    if not core:                      # name made only of legal words: keep them
        core = toks
    return " ".join(toks), " ".join(core), " ".join(skeleton(t) for t in core), "".join(core)


def normalize_address(addr):
    """Return (addr_norm, addr_tok, postcode, house) for an address."""
    if not addr:
        return "", "", "", ""
    first, _, rest = addr.partition(",")
    pcs = _POSTCODE.findall(rest)
    postcode = pcs[-1].replace(" ", "") if pcs else ""
    # house number: first number in the address that is not the postcode
    # (searched over the whole address because components are sometimes reordered)
    nums = [n for n in _NUMBER.findall(addr) if n != postcode and len(n) <= 5]
    house = nums[0].lstrip("0") if nums else ""
    low = unidecode(addr).lower() if not addr.isascii() else addr.lower()
    for phrase, code in ADDR_PHRASES:
        if phrase in low:
            low = low.replace(phrase, code)
    toks = [ADDR_CANON.get(t, t) for t in _NON_ALNUM.sub(" ", low).split()]
    informative = [t for t in toks if len(t) >= 3 and t.isalpha() and t not in ADDR_STOP]
    # v3: informative address words are stored as skeletons so transliterated place names
    # line up ('mhaaraassttr' ~ 'maharashtra', 'krnaattk' ~ 'karnataka')
    informative = [skeleton(t) for t in informative]
    return " ".join(toks), " ".join(dict.fromkeys(t for t in informative if len(t) >= 2)), postcode, house


def _normalize_batch(args):
    names, addrs = args
    out = []
    for n, a in zip(names, addrs):
        out.append(normalize_name(n) + normalize_address(a))
    return out


def normalize_records(names, addrs, workers=4, batch=50_000):
    """Normalise parallel lists of names and addresses using a process pool.

    Returns a list of 8-tuples in the column order of COLUMNS.
    """
    jobs = [(names[i:i + batch], addrs[i:i + batch]) for i in range(0, len(names), batch)]
    with Pool(workers) as pool:
        parts = pool.map(_normalize_batch, jobs)
    return [row for part in parts for row in part]


COLUMNS = ["name_norm", "core", "skel", "concat", "addr_norm", "addr_tok", "postcode", "house"]
