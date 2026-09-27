"""v5.1 normalisation: v4 views plus canonical forms for the documented noise families.

Names (core/skel/concat and the folded view): trade-name markers keep the name after
`aka`/`a/k/a`/`dba`/`d/b/a`/`trading as`; digits inside words are read as letters
(`g0mez`, `a1l`, `8ili`); honorifics are dropped; repeated tokens are merged.
Addresses (addr_norm, addr_tok): canonical street types and ordinals, US/Indian state
names to codes, mailbox/unit/`n/a`/house-number labels removed, leading zeros dropped.
Every rule is a fixed string transformation; no labels or external data are used.
"""
from __future__ import annotations

import polars as pl

from .text_norm import LEGAL, LEGAL_SKEL, ADDR_STOP, _WEB, _clean, _drop_words, skeleton, to_ascii

HONORIFIC = {"smt", "shri", "sri", "shree", "mr", "mrs", "ms", "dr", "m s"}
ALIAS = r"^.*\b(?:aka|a k a|dba|d b a|fka|f k a|also known as|doing business as|trading as|t a)\b\s+"
LEET = {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b"}
STREET = {"street": "st", "saint": "st", "str": "st", "road": "rd", "drive": "dr", "avenue": "ave", "av": "ave",
          "court": "ct", "place": "pl", "lane": "ln", "boulevard": "blvd", "parkway": "pkwy", "highway": "hwy",
          "circle": "cir", "terrace": "ter", "square": "sq", "trail": "trl", "point": "pt", "mount": "mt",
          "north": "n", "south": "s", "east": "e", "west": "w", "first": "1st", "second": "2nd", "third": "3rd",
          "fourth": "4th", "fifth": "5th", "sixth": "6th", "seventh": "7th", "eighth": "8th", "ninth": "9th",
          "tenth": "10th", "nagar": "ngr", "marg": "mrg", "sector": "sec", "sec": "sec"}
STATE_PHRASES = {
    "north carolina": "nc", "south carolina": "sc", "north dakota": "nd", "south dakota": "sd",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny", "rhode island": "ri",
    "west virginia": "wv", "district of columbia": "dc", "tamil nadu": "tn", "west bengal": "wb",
    "madhya pradesh": "mp", "uttar pradesh": "up", "andhra pradesh": "ap", "himachal pradesh": "hp",
    "arunachal pradesh": "ar", "jammu and kashmir": "jk", "jammu kashmir": "jk"}
STATE_WORDS = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv", "ohio": "oh",
    "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "tennessee": "tn", "texas": "tx", "utah": "ut",
    "vermont": "vt", "virginia": "va", "washington": "wa", "wisconsin": "wi", "wyoming": "wy",
    "maharashtra": "mh", "delhi": "dl", "karnataka": "ka", "telangana": "tg", "ts": "tg", "gujarat": "gj",
    "rajasthan": "rj", "kerala": "kl", "haryana": "hr", "punjab": "pb", "bihar": "br", "odisha": "od",
    "orissa": "od", "assam": "as", "jharkhand": "jh", "chhattisgarh": "cg", "uttarakhand": "uk", "goa": "ga",
    "chandigarh": "ch", "puducherry": "py", "pondicherry": "py", "bangalore": "bengaluru",
    "bombay": "mumbai", "calcutta": "kolkata", "madras": "chennai", "gurgaon": "gurugram"}
ADDR_DROP = {"city", "of", "township", "town", "village", "na", "n a", "none", "null"}


def _map_tokens(c: pl.Expr, mapping: dict[str, str], drop: set[str] = frozenset()) -> pl.Expr:
    return (c.str.split(" ")
            .list.eval(pl.element().filter((pl.element() != "") & ~pl.element().is_in(list(drop)))
                       .replace(mapping))
            .list.join(" "))


def _deleet(c: pl.Expr) -> pl.Expr:
    """Digits inside mostly-alphabetic name tokens become letters; ordinals stay."""
    token = pl.element()
    swapped = token
    for digit, letter in LEET.items():
        swapped = swapped.str.replace_all(digit, letter, literal=True)
    letters = token.str.count_matches(r"[a-z]")
    digits = token.str.count_matches(r"[0-9]")
    eligible = (digits > 0) & (letters >= 2) & (letters >= digits) & ~token.str.contains(r"^\d+(st|nd|rd|th)$")
    return c.str.split(" ").list.eval(pl.when(eligible).then(swapped).otherwise(token)).list.join(" ")


def _dedupe(c: pl.Expr) -> pl.Expr:
    """Keep the first occurrence of each token (`bili s bili s analytics` -> `bili s analytics`)."""
    return (c.str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
            .list.unique(maintain_order=True).list.join(" "))


def canonical_address(c: pl.Expr) -> pl.Expr:
    c = (c.str.replace_all(r"\b(?:pmb|p o box|po box|box|unit|suite|ste|apt|apartment|room|rm)\s+\w*\d\w*\b", " ")
         .str.replace_all(r"\b(?:h no|hno|house no|door no|flat no|plot no|shop no|no|n a)\b", " ")
         .str.replace_all(r"\b(\d+) 1 2\b", "$1")
         .str.replace_all(r"\b0+(\d)", "$1"))
    for phrase, code in STATE_PHRASES.items():
        c = c.str.replace_all(rf"\b{phrase}\b", code)
    c = _map_tokens(c, {**STREET, **STATE_WORDS}, ADDR_DROP)
    return c.str.replace_all(r"\s+", " ").str.strip_chars()


def canonical_core(name_norm: pl.Expr) -> pl.Expr:
    c = name_norm.str.replace_all(ALIAS, "")
    c = _deleet(c)
    c = _drop_words(c, LEGAL | HONORIFIC)
    return _dedupe(c)


def normalise(df: pl.DataFrame) -> pl.DataFrame:
    """Same columns as text_norm.normalise, with canonical core and address views."""
    name = to_ascii(df["business_name"]).str.to_lowercase()
    addr = to_ascii(df["business_address"]).str.to_lowercase()
    df = df.with_columns(name.alias("_n"), addr.alias("_a"))
    name_norm = _clean(pl.col("_n").str.replace_all(r"\(\s*id\s*:?\s*\d+\s*\)", " ")
                       .str.replace_all(_WEB, " ").str.replace_all(r"[&+]", " and "))
    df = df.with_columns(name_norm.alias("name_norm"), canonical_address(_clean(pl.col("_a"))).alias("addr_norm"))
    df = df.with_columns(canonical_core(pl.col("name_norm")).alias("core"))
    df = df.with_columns(_drop_words(skeleton(pl.col("core")), LEGAL_SKEL).alias("skel"),
                         pl.col("core").str.replace_all(" ", "").alias("concat"))
    addr_tok = (pl.col("addr_norm").str.split(" ")
                .list.eval(pl.element().filter(pl.element().str.contains(r"^[a-z]{3,}$")
                                               & ~pl.element().is_in(list(ADDR_STOP))))
                .list.join(" "))
    after_comma = pl.col("_a").str.splitn(",", 2).struct.field("field_1")
    postcode = after_comma.str.extract(r"\b(\d{5,6})\b", 1).str.strip_chars_start("0")
    nums = (pl.col("_a").str.replace_all(r"\b(?:pmb|p\.?\s*o\.?\s*box|box|unit|suite|ste|apt)\s*#?\s*\d+", " ")
            .str.extract_all(r"\d+")
            .list.eval(pl.element().str.strip_chars_start("0"))
            .list.eval(pl.element().filter((pl.element() != "") & (pl.element().str.len_chars() <= 5))))
    df = df.with_columns(addr_tok.alias("addr_tok"), postcode.alias("postcode"), nums.alias("_nums"))
    h0 = pl.col("_nums").list.get(0, null_on_oob=True)
    h1 = pl.col("_nums").list.get(1, null_on_oob=True)
    house = pl.when(pl.col("postcode").is_not_null() & (h0 == pl.col("postcode"))).then(h1).otherwise(h0)
    df = df.with_columns(house.alias("house"))
    return df.drop("_n", "_a", "_nums")
