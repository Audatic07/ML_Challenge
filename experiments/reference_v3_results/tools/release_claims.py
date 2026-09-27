"""Release unfinished main-queue claims held by named workers (they moved to another queue)."""
import json, sys
import boto3
B = "amazon-sagemaker-580857071542-ap-south-1-dn0trrxacr3ddt"
import os
P = os.environ.get("ER_PREFIX", "shared/er-v51-20260927")
workers = set(sys.argv[1:])
s3 = boto3.client("s3", region_name="ap-south-1")


def names(folder):
    out = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=B, Prefix=f"{P}/{folder}/"):
        out += [o["Key"].rsplit("/", 1)[1] for o in page.get("Contents", [])]
    return out


finished = {n[:-5] for n in names("done")} | {n[:-5] for n in names("failed")}
for claim in names("claims"):
    if claim in finished:
        continue
    body = json.loads(s3.get_object(Bucket=B, Key=f"{P}/claims/{claim}")["Body"].read())
    if body.get("worker") in workers:
        s3.put_object(Bucket=B, Key=f"{P}/attempts/{claim}--{body['worker']}.json", Body=b'{"released": "moved queue"}',
                      ServerSideEncryption="AES256")
        s3.delete_object(Bucket=B, Key=f"{P}/claims/{claim}")
        print("released", claim, "from", body["worker"])
