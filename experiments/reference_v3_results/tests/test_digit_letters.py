import polars as pl

from src.text_norm import derive, fix_digit_letters


def fix(names):
    return pl.DataFrame({"n": names}).select(fix_digit_letters(pl.col("n")))["n"].to_list()


def test_digit_typos_inside_words_become_letters():
    assert fix(["mi1ler spruce llc", "8ozarth creative", "6reen township", "br0thers 5chool"]) == [
        "miller spruce llc", "bozarth creative", "green township", "brothers school"]


def test_pure_numbers_and_unmapped_digits_are_kept():
    assert fix(["4515 midland", "council 9630615796", "3rd 24 hours", ""]) == [
        "4515 midland", "council 9630615796", "3rd 24 hours", ""]


def test_derive_fixes_views_and_is_idempotent():
    df = pl.DataFrame({"name_norm": ["mi1ler spruce llc", "miller spruce llc"],
                       "addr_norm": ["4515 midland trail va", "4515 midland trl va"]})
    once = derive(df)
    assert once["name_norm"].to_list() == ["miller spruce llc", "miller spruce llc"]
    assert once["core"][0] == once["core"][1] == "miller spruce"
    assert once["skel"][0] == once["skel"][1]
    assert derive(once).equals(once)
