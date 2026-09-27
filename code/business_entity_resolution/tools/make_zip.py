"""Fill measured numbers into the docs, validate outputs, build the ZIP and re-validate its members."""
import hashlib, json, subprocess, sys, zipfile
from pathlib import Path

PKG = Path(r"C:\Ml_challenge_submission\pkg")
DATA = Path(r"C:\Ml_challenge\student_resource\dataset")
VALIDATOR = Path(r"C:\Ml_challenge\student_resource\utils\validate_submission.py")
values = json.loads(Path(sys.argv[1]).read_text())
zip_path = Path(sys.argv[2])

doc = PKG / "Documentation_template.md"
text = doc.read_text(encoding="utf-8")
for key, value in values.items():
    text = text.replace("{" + key + "}", str(value))
missing = [part.split("}")[0] for part in text.split("{")[1:] if part.split("}")[0].isupper()]
assert not missing, f"unfilled placeholders: {missing}"
doc.write_text(text, encoding="utf-8")

out = PKG / "output"
run = subprocess.run([sys.executable, str(VALIDATOR), "--matching", str(out / "matching_results.tsv"),
                      "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir", str(DATA / "test")],
                     capture_output=True, text=True, encoding="utf-8", errors="replace")
print(run.stdout[-800:])
assert run.returncode == 0, "official validator failed"

members = sorted(p for p in PKG.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
    for path in members:
        archive.write(path, path.relative_to(PKG).as_posix())
print("zip", zip_path, zip_path.stat().st_size, "bytes,", len(members), "files")

# re-validate the TSVs as extracted from the archive
check = zip_path.parent / "zip_check"
with zipfile.ZipFile(zip_path) as archive:
    assert archive.testzip() is None
    for name in ("output/matching_results.tsv", "output/candidate_pairs.tsv"):
        archive.extract(name, check)
run = subprocess.run([sys.executable, str(VALIDATOR), "--matching", str(check / "output/matching_results.tsv"),
                      "--candidate", str(check / "output/candidate_pairs.tsv"), "--test-dir", str(DATA / "test")],
                     capture_output=True, text=True, encoding="utf-8", errors="replace")
print("extracted:", run.stdout.strip().splitlines()[-1])
assert run.returncode == 0

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()

manifest = {"zip": zip_path.name, "zip_sha256": sha(zip_path), "zip_bytes": zip_path.stat().st_size,
            "matching_results_sha256": sha(out / "matching_results.tsv"),
            "candidate_pairs_sha256": sha(out / "candidate_pairs.tsv"), **values}
(zip_path.parent / "release_manifest.json").write_text(json.dumps(manifest, indent=1))
print(json.dumps(manifest, indent=1))
