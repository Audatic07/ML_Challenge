"""Text normalisation. Each record gets these views:

name_norm  lowercase ASCII name, punctuation removed
core       name_norm without legal/filler words
skel       consonant skeleton of each core token, legal skeletons removed
concat     core without spaces
addr_norm  lowercase ASCII address, punctuation removed
addr_tok   informative address words (alphabetic, len >= 3, not generic street words)
postcode   5-6 digit number found after the first comma (so a leading house number is not a ZIP)
house      first number (<= 5 digits) anywhere in the address that is not the postcode
"""
from __future__ import annotations

import unicodedata

import polars as pl
from unidecode import unidecode

LEGAL = {
    "inc", "incorporated", "llc", "corp", "corporation", "co", "company", "ltd", "limited",
    "pvt", "private", "llp", "lp", "plc", "pllc", "pc", "the", "and",
    "sarl", "sas", "sasu", "eurl", "sa", "sci", "snc",
}

ADDR_STOP = {
    "street", "road", "avenue", "ave", "drive", "lane", "boulevard", "blvd", "court",
    "place", "circle", "way", "highway", "hwy", "parkway", "pkwy", "suite", "ste", "unit",
    "apt", "floor", "plot", "near", "opp", "opposite", "behind", "house", "building", "bldg",
    "block", "sector", "null", "none", "box", "pmb", "rue", "chemin", "impasse", "allee",
    "bis", "ter", "des", "les", "the", "and", "flat", "shop", "road", "main", "cross",
}

_WEB = r"\b(www\.)|\.(com|net|org|in|co\.in|fr|biz|info)\b"


def to_ascii(s: pl.Series) -> pl.Series:
    """Transliterate non-ASCII values (Devanagari, accents, ...) with unidecode.

    Runs only on distinct non-ASCII strings for speed.
    """
    s = s.fill_null("")
    uniq = s.unique()
    non = uniq.filter(~uniq.str.contains(r"^[\x00-\x7F]*$"))
    if len(non) == 0:
        return s
    old = non.to_list()
    new = [unidecode(unicodedata.normalize("NFKC", x)) for x in old]
    return s.replace(old, new)


def _clean(c: pl.Expr) -> pl.Expr:
    return c.str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars()


def _drop_words(c: pl.Expr, words: set[str]) -> pl.Expr:
    return (c.str.split(" ")
            .list.eval(pl.element().filter((pl.element() != "") & ~pl.element().is_in(list(words))))
            .list.join(" "))


def skeleton(c: pl.Expr) -> pl.Expr:
    """Keep first letter of each token, drop later vowels and h, map c/q->k, z->s,
    ph->f, ck->k, then merge repeated letters. `raam maarketting` -> `rm mrktng`."""
    c = (c.str.replace_all("ph", "f").str.replace_all("ck", "k")
         .str.replace_all("[cq]", "k").str.replace_all("z", "s"))
    c = c.str.replace_all(r"\B[aeiouh]", "")
    for ch in "abcdefghijklmnopqrstuvwxyz":
        c = c.str.replace_all(ch + ch + "+", ch)
    return c


def _legal_skel() -> set[str]:
    s = pl.DataFrame({"w": sorted(LEGAL)}).select(skeleton(pl.col("w")))["w"].to_list()
    return {x for x in s if len(x) >= 3}


LEGAL_SKEL = _legal_skel()


def normalise(df: pl.DataFrame) -> pl.DataFrame:
    """df has business_name, business_address; returns it with the views added."""
    name = to_ascii(df["business_name"]).str.to_lowercase()
    addr = to_ascii(df["business_address"]).str.to_lowercase()
    df = df.with_columns(name.alias("_n"), addr.alias("_a"))

    name_norm = _clean(pl.col("_n").str.replace_all(r"\(\s*id\s*:?\s*\d+\s*\)", " ")
                       .str.replace_all(_WEB, " ").str.replace_all(r"[&+]", " and "))
    df = df.with_columns(name_norm.alias("name_norm"), _clean(pl.col("_a")).alias("addr_norm"))
    df = df.with_columns(_drop_words(pl.col("name_norm"), LEGAL).alias("core"))
    df = df.with_columns(_drop_words(skeleton(pl.col("core")), LEGAL_SKEL).alias("skel"),
                         pl.col("core").str.replace_all(" ", "").alias("concat"))

    addr_tok = (pl.col("addr_norm").str.split(" ")
                .list.eval(pl.element().filter(pl.element().str.contains(r"^[a-z]{3,}$")
                                               & ~pl.element().is_in(list(ADDR_STOP))))
                .list.join(" "))
    after_comma = pl.col("_a").str.splitn(",", 2).struct.field("field_1")
    postcode = after_comma.str.extract(r"\b(\d{5,6})\b", 1).str.strip_chars_start("0")
    nums = (pl.col("_a").str.extract_all(r"\d+")
            .list.eval(pl.element().str.strip_chars_start("0"))
            .list.eval(pl.element().filter((pl.element() != "") & (pl.element().str.len_chars() <= 5))))
    df = df.with_columns(addr_tok.alias("addr_tok"), postcode.alias("postcode"), nums.alias("_nums"))
    h0 = pl.col("_nums").list.get(0, null_on_oob=True)
    h1 = pl.col("_nums").list.get(1, null_on_oob=True)
    house = pl.when(pl.col("postcode").is_not_null() & (h0 == pl.col("postcode"))).then(h1).otherwise(h0)
    df = df.with_columns(house.alias("house"))
    return df.drop("_n", "_a", "_nums")
