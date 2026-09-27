"""Publish the retrieval benchmark queue and lower memory floors on the main queue."""
import json
import boto3
import numpy as np
B = "amazon-sagemaker-580857071542-ap-south-1-dn0trrxacr3ddt"
MAIN = "shared/er-v5-20260927"
BENCH = "shared/er-v5-bench1"
s3 = boto3.client("s3", region_name="ap-south-1")


def put(key, value):
    s3.put_object(Bucket=B, Key=key, Body=json.dumps(value, indent=1).encode(), ServerSideEncryption="AES256")


def get(key):
    return json.loads(s3.get_object(Bucket=B, Key=key)["Body"].read())


plan = get(f"{MAIN}/plan.json")
# 1) main queue: c-class workers (15.3 GB) can take US train and India test shards
changed = 0
for page in s3.get_paginator("list_objects_v2").paginate(Bucket=B, Prefix=f"{MAIN}/tasks/feat-"):
    for obj in page.get("Contents", []):
        name = obj["Key"].rsplit("/", 1)[1]
        if name.startswith(("feat-train-US", "feat-test-India")):
            task = get(obj["Key"])
            if task["min_mem_gb"] > 13:
                task["min_mem_gb"] = 13.0
                put(obj["Key"], task)
                changed += 1
print("main tasks relaxed:", changed)
# 2) benchmark queue with v5 normalisation
bench_plan = dict(plan, norm="v5", anchors=False, plan_sha="bench1", task_count=2)
put(f"{BENCH}/plan.json", bench_plan)
tune = np.array(plan["tune_rows"])
country = np.array(plan["tune_country"])
rng = np.random.default_rng(11)
char = {"analyzer": "char_wb", "ngram": [3, 4], "max_df": 0.03}
word = {"analyzer": "word", "ngram": [1, 2], "max_df": 0.05}
channels = {
    "name_char": {"view": "core", **char, "top_k": 40, "ks": [20, 40]},
    "address_char": {"view": "addr_norm", **char, "top_k": 40, "ks": [20, 40]},
    "name_fold": {"view": "name_fold", **word, "top_k": 20, "ks": [10, 20]},
    "address_fold": {"view": "address_fold", **word, "top_k": 20, "ks": [10, 20]},
    "combo": {"view": "combo", **char, "top_k": 60, "ks": [20, 40, 60]},
    "combo_word": {"view": "combo", **word, "top_k": 40, "ks": [20, 40]},
}
base = ["name_char@20", "address_char@20", "name_fold@20", "address_fold@20"]
unions = [
    base,
    base + ["combo@20"],
    base + ["combo@40"],
    ["name_char@20", "address_char@40", "name_fold@10", "address_fold@10", "combo@40"],
    ["name_char@20", "address_char@40", "name_fold@10", "address_fold@10", "combo@40", "combo_word@20"],
    ["name_char@20", "address_char@20", "combo@40"],
    ["name_char@40", "address_char@40", "name_fold@20", "address_fold@20", "combo@60", "combo_word@40"],
]
for c in ("India", "US"):
    rows = np.sort(rng.choice(tune[country == c], 4000, replace=False)).tolist()
    put(f"{BENCH}/tasks/bench-{c}.json", {"id": f"bench-{c}", "kind": "bench", "split": "train", "country": c,
                                          "rows": rows, "channels": channels, "unions": unions, "norm": "v5",
                                          "priority": 0, "min_mem_gb": 8})
print("bench tasks published")
