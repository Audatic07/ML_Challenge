import collections, datetime, json, sys
import boto3
B = "amazon-sagemaker-580857071542-ap-south-1-dn0trrxacr3ddt"
import os
P = os.environ.get("ER_PREFIX", "shared/er-v51-20260927")
s3 = boto3.client("s3", region_name="ap-south-1")


def ls(folder):
    out, token = [], None
    while True:
        kw = {"Bucket": B, "Prefix": f"{P}/{folder}/"}
        if token:
            kw["ContinuationToken"] = token
        page = s3.list_objects_v2(**kw)
        out += page.get("Contents", [])
        if not page.get("IsTruncated"):
            return out
        token = page["NextContinuationToken"]


def name(obj):
    return obj["Key"].rsplit("/", 1)[1]


def kind(task):
    return task.split("-")[0] if not task.startswith(("train", "final")) else task


def group(task):
    parts = task.split("-")
    return "-".join(parts[:3]) if parts[0] in ("feat", "score") else task


now = datetime.datetime.now(datetime.timezone.utc)
done = {name(o)[:-5]: o for o in ls("done")}
failed = {name(o)[:-5] for o in ls("failed")}
claims = {name(o) for o in ls("claims")}
tasks = {name(o)[:-5] for o in ls("tasks")}
stats = collections.defaultdict(lambda: collections.Counter())
for t in tasks:
    g = group(t)
    stats[g]["total"] += 1
    if t in done:
        stats[g]["done"] += 1
    elif t in failed:
        stats[g]["failed"] += 1
    elif t in claims:
        stats[g]["running"] += 1
print(now.strftime("%H:%M:%SZ"), f"tasks={len(tasks)} done={len(done)} failed={len(failed)} claimed={len(claims)}")
for g in sorted(stats):
    c = stats[g]
    print(f"  {g:28s} {c['done']:3d}/{c['total']:3d} running={c['running']} failed={c['failed']}")
for o in ls("logs"):
    n = name(o)
    if n.endswith(".json"):
        hb = json.loads(s3.get_object(Bucket=B, Key=o["Key"])["Body"].read())
        age = (now - o["LastModified"]).total_seconds()
        print(f"  HB {n[:-5]:12s} task={hb.get('task')} avail={hb.get('mem_avail_gb', 0):.1f}/{hb.get('mem_total_gb', 0):.1f}GB age={age:.0f}s")
if failed:
    for t in sorted(failed):
        body = json.loads(s3.get_object(Bucket=B, Key=f"{P}/failed/{t}.json")["Body"].read())
        print("FAILED", t, body.get("worker"), body.get("error"))
        if "-v" in sys.argv:
            print(body.get("traceback"))
recent = sorted(done.values(), key=lambda o: o["LastModified"])[-int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else -3:]
for o in recent:
    body = json.loads(s3.get_object(Bucket=B, Key=o["Key"])["Body"].read())
    print("DONE", name(o)[:-5], body["worker"], f"{body['seconds']:.0f}s", json.dumps(body.get("info"))[:400])
