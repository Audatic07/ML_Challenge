"""One-shot, read-only S3 snapshot. No credentials are created or printed.

Only HeadObject/GetObject are used. Copies go to a NEW private local directory
outside Git. Exit 2 means absent artifacts or missing credentials; do not infer
job completion from this tool. No polling, task queue or SageMaker calls.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, NoCredentialsError


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bucket", default="amazon-sagemaker-580857071542-ap-south-1-dn0trrxacr3ddt")
    p.add_argument("--prefix", default="shared/er-v5-20260927")
    p.add_argument("--profile", help="Existing read-only AWS profile; no secrets in arguments")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    out = args.out.resolve()
    if any((d / ".git").exists() for d in (out, *out.parents)):
        p.error("Choose a private local directory outside Git")
    if out.exists():
        p.error("--out must be new; snapshots are never overwritten")
    session = boto3.Session(profile_name=args.profile, region_name="ap-south-1")
    if session.get_credentials() is None:
        print(json.dumps({"status": "not_accessible", "reason": "No configured AWS credentials"}))
        return 2
    s3 = session.client("s3", config=Config(connect_timeout=10, read_timeout=60,
                                           retries={"max_attempts": 2, "mode": "standard"}))
    names = ["tune_per_query.parquet", "tune_scores.parquet", "model_manifest.json"]
    objects = {}
    for name in names:
        key = args.prefix.rstrip("/") + "/model/" + name
        try:
            head = s3.head_object(Bucket=args.bucket, Key=key)
        except ClientError as exc:
            print(json.dumps({"status": "not_ready_or_not_accessible", "object": name,
                              "code": exc.response["Error"]["Code"]}))
            return 2
        objects[name] = {"key": key, "etag": head["ETag"], "bytes": head["ContentLength"],
                         "version_id": head.get("VersionId"), "modified": head["LastModified"].isoformat()}
    out.mkdir(parents=True, exist_ok=False)
    for name, info in objects.items():
        kwargs = {"Bucket": args.bucket, "Key": info["key"], "IfMatch": info["etag"]}
        if info["version_id"]:
            kwargs["VersionId"] = info["version_id"]
        response = s3.get_object(**kwargs)
        h = hashlib.sha256()
        total = 0
        with response["Body"] as body, (out / name).open("xb") as stream:
            for data in iter(lambda: body.read(8 << 20), b""):
                stream.write(data)
                h.update(data)
                total += len(data)
        if total != info["bytes"]:
            raise ValueError("Downloaded size differs from snapshot")
        info["sha256"] = h.hexdigest()
    for name, info in objects.items():
        current = s3.head_object(Bucket=args.bucket, Key=info["key"])
        if current["ETag"] != info["etag"] or current.get("VersionId") != info["version_id"]:
            raise ValueError("Source changed during snapshot; discard this incomplete local snapshot")
    record = {"at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "bucket": args.bucket, "prefix": args.prefix, "objects": objects,
              "s3_operations": ["HeadObject", "GetObject"], "complete": True}
    (out / "snapshot.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"status": "downloaded", "objects": len(objects), "bytes": sum(x["bytes"] for x in objects.values())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
