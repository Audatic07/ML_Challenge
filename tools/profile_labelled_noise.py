"""Local-only labelled-pair diagnostic; exports aggregate statistics, never records.

This is not a retriever, classifier, validation score, or estimate of test performance.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re
import time
import unicodedata
import zipfile


def rows(archive: zipfile.ZipFile, member: str):
    with archive.open(member) as raw, io.TextIOWrapper(raw, encoding="utf-8-sig") as stream:
        columns = stream.readline().rstrip("\r\n").split("\t")
        for line in stream:
            values = line.rstrip("\r\n").split("\t")
            if len(values) != len(columns):
                raise ValueError(f"Unexpected number of columns in {member}")
            yield dict(zip(columns, values))


def normal(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(re.findall(r"[^\W_]+", text, flags=re.UNICODE))


def view(row: dict) -> dict:
    name, address = normal(row["business_name"]), normal(row["business_address"])
    padded = " " + name + " "
    return {"name": name, "address": address, "country": row["country"],
            "tokens": set(name.split()),
            "grams": {padded[i:i + 3] for i in range(max(0, len(padded) - 2))},
            "address_tokens": set(address.split()),
            "numbers": set(re.findall(r"\d+", address))}


def signals(left: dict, right: dict) -> dict:
    union = left["grams"] | right["grams"]
    similarity = len(left["grams"] & right["grams"]) / len(union) if union else 0.0
    token = bool(left["tokens"] & right["tokens"])
    addr = bool(left["address_tokens"] & right["address_tokens"])
    return {"name_exact": left["name"] == right["name"],
            "address_exact_nonempty": bool(left["address"]) and left["address"] == right["address"],
            "target_address_missing": not right["address"],
            "name_token_overlap": token,
            "name_trigram_jaccard_ge_025": similarity >= 0.25,
            "name_trigram_jaccard_ge_050": similarity >= 0.50,
            "no_name_overlap_and_trigram_lt_025": not token and similarity < 0.25,
            "address_token_overlap": addr,
            "no_name_or_address_token_overlap": not token and not addr,
            "both_have_numbers": bool(left["numbers"] and right["numbers"]),
            "both_have_numbers_but_disjoint": bool(left["numbers"] and right["numbers"]) and not (left["numbers"] & right["numbers"]),
            "cross_country": left["country"] != right["country"],
            "diagnostic_edge": token or similarity >= 0.25}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="6ab10eb3b23ba_student_resource.zip")
    parser.add_argument("--output", default="output/hackathon/labelled_noise_profile.json")
    parser.add_argument("--modulus", type=int, default=100)
    args = parser.parse_args()
    if args.modulus < 1:
        parser.error("modulus must be positive")
    start = time.monotonic()
    prefix = "student_resource/dataset/train/"
    truth, refs, targets = {}, {}, {}
    with zipfile.ZipFile(args.archive) as archive:
        for row in rows(archive, prefix + "train_ground_truth.tsv"):
            entity = row["source1_entity_id"]
            digest = hashlib.blake2b(("noise-profile-v1:" + entity).encode(), digest_size=8).digest()
            if int.from_bytes(digest, "big") % args.modulus == 0:
                truth[entity] = row["matched_entity_ids"].split(",") if row["matched_entity_ids"] else []
        wanted = {entity for values in truth.values() for entity in values}
        print(f"Sampled {len(truth)} S1; locating {len(wanted)} target IDs", flush=True)
        for row in rows(archive, prefix + "train_source1.tsv"):
            if row["entity_id"] in truth:
                refs[row["entity_id"]] = view(row)
        for source in (2, 3):
            for row in rows(archive, prefix + f"train_source{source}.tsv"):
                if row["entity_id"] in wanted:
                    targets[row["entity_id"]] = view(row)
            print(f"Scanned S{source}; found {len(targets)} target records", flush=True)
    if set(refs) != set(truth) or set(targets) != wanted:
        raise ValueError("Sampled labels contain a missing source/target record")
    counts, country, groups = Counter(), Counter(), Counter()
    by_slice = {}
    oracle_direct = 0.0
    oracle_connected = 0.0
    for entity, matches in truth.items():
        reference = refs[entity]
        country[reference["country"]] += 1
        if not matches:
            groups["singletons"] += 1
            oracle_direct += 1.0
            oracle_connected += 1.0
            continue
        variants = [targets[target] for target in matches]
        direct, reached = set(), {0}
        for index, (target, variant) in enumerate(zip(matches, variants), 1):
            s = signals(reference, variant)
            counts["pairs"] += 1
            counts.update({key: int(value) for key, value in s.items()})
            key = reference["country"] + "/" + target[:2]
            bucket = by_slice.setdefault(key, Counter())
            bucket["pairs"] += 1
            bucket.update({name: int(value) for name, value in s.items()})
            if s["diagnostic_edge"]:
                direct.add(index)
                reached.add(index)
        all_views = [reference, *variants]
        changed = True
        while changed:
            changed = False
            for index in range(1, len(all_views)):
                if index not in reached and any(signals(all_views[j], all_views[index])["diagnostic_edge"] for j in reached):
                    reached.add(index)
                    changed = True
        direct_n, connected_n, g = len(direct), len(reached) - 1, len(matches)
        oracle_direct += 5 * direct_n / (4 * direct_n + g)
        oracle_connected += 5 * connected_n / (4 * connected_n + g)
        groups["nonsingletons"] += 1
        groups["all_direct_edges"] += direct_n == g
        groups["all_connected_with_true_labels"] += connected_n == g
        groups["no_direct_edge"] += direct_n == 0
        groups["positive_links_recovered_via_true_label_bridge"] += connected_n - direct_n
    result = {
        "schema_version": 1, "archive": Path(args.archive).name,
        "sampling": f"blake2b64('noise-profile-v1:' + S1_ID) modulo {args.modulus} == 0",
        "sample_s1": len(truth), "country_counts": country,
        "positive_pair_counts": counts, "positive_pair_fractions": {k: v / counts["pairs"] for k, v in counts.items() if k != "pairs"},
        "group_counts": groups, "by_country_source": by_slice,
        "diagnostic_name_edge_oracle_macro": oracle_direct / len(truth),
        "diagnostic_true_label_bridge_oracle_macro": oracle_connected / len(truth),
        "limitations": [
            "Positive-only sample; no negative precision, actual candidate recall, or model score measured.",
            "Diagnostic edge is any shared normalized name token OR name trigram Jaccard >= 0.25; common tokens included.",
            "Bridge statistic uses known truth to connect variants. It is an optimistic diagnostic, not an inference procedure.",
            "No actual full-pool retrieval, ranking, posting-list limit or candidate cap is simulated.",
            "No test labels, model training, cloud jobs or external identity lookups used."
        ], "elapsed_seconds": round(time.monotonic() - start, 2)
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "s1": len(truth), "pairs": counts["pairs"], "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
