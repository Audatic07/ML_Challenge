"""V8 step 2 (GPU notebook): SC-Block supervised contrastive fine-tuning of all-MiniLM-L6-v2 (Apache-2.0) on the fit-only
groups (two members per group per batch, other groups as negatives, temperature 0.07, mean pooling, L2 norm), then
cosine similarity for every tune and test candidate pair. Tune scores are uploaded first so the CPU gate can start.
Run: ~/SageMaker/scblock-env/bin/python gpu.py   (inputs in ~/SageMaker/v8, S3 prefix in V8_S3)
"""
import os
import subprocess
import time

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

V = os.path.expanduser("~/SageMaker/v8")
S3 = os.environ["V8_S3"]
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAXLEN, T, G = 40, 0.07, 256
TRAIN_SECONDS = float(os.environ.get("V8_TRAIN_SECONDS", 300))
torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL).cuda()


def embed(texts):
    b = tok(texts, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to("cuda")
    h = model(**b).last_hidden_state
    m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
    return F.normalize((h * m).sum(1) / m.sum(1).clamp(min=1), dim=-1)


groups = pl.read_parquet(f"{V}/train_groups.parquet").group_by("g").agg(pl.col("text")).filter(pl.col("text").list.len() >= 2)
texts = groups["text"].to_list()
rng = np.random.default_rng(0)
order = rng.permutation(len(texts))
opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
model.train()
start, step = time.time(), 0
while time.time() - start < TRAIN_SECONDS and (step + 1) * G <= len(order):
    a, b = [], []
    for i in order[step * G:(step + 1) * G]:
        x, y = rng.choice(len(texts[i]), 2, replace=False)
        a.append(texts[i][x]); b.append(texts[i][y])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        z = embed(a + b)
    z = z.float()
    logits = z @ z.T / T
    logits.fill_diagonal_(-1e9)
    n = len(a)
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).cuda()
    loss = F.cross_entropy(logits, target)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    step += 1
    if step % 50 == 0:
        print(f"step {step} loss {loss.item():.4f} t={time.time() - start:.0f}s", flush=True)
print(f"trained {step} steps ({step * G:,} groups) in {time.time() - start:.0f}s", flush=True)
model.eval()


@torch.no_grad()
def table(path, key):
    df = pl.read_parquet(path)
    txt = df["text"].to_list()
    order = np.argsort([len(s) for s in txt])          # length-sorted batches pad less
    out = torch.empty((len(txt), model.config.hidden_size), dtype=torch.float16, device="cuda")
    for i in range(0, len(txt), 4096):
        idx = order[i:i + 4096]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out[torch.as_tensor(idx, device="cuda")] = embed([txt[j] for j in idx]).half()
    return df.select(key).with_row_index("_i"), out


@torch.no_grad()
def cosines(split):
    t0 = time.time()
    qk, qe = table(f"{V}/{split}_q.parquet", "s1")
    tk, te = table(f"{V}/{split}_t.parquet", "t")
    pairs = (pl.read_parquet(f"{V}/{split}_pairs.parquet").with_row_index("_p")
             .join(qk.rename({"_i": "qi"}), on="s1").join(tk.rename({"_i": "ti"}), on="t").sort("_p"))
    qi, ti = torch.as_tensor(pairs["qi"].to_numpy().astype(np.int64), device="cuda"), torch.as_tensor(pairs["ti"].to_numpy().astype(np.int64), device="cuda")
    cos = torch.cat([(qe[qi[i:i + 1_000_000]].float() * te[ti[i:i + 1_000_000]].float()).sum(-1) for i in range(0, len(qi), 1_000_000)])
    out = pairs.select("s1", "t").with_columns(pl.Series("cos", cos.cpu().numpy().astype(np.float32)))
    path = f"{V}/cos_{split}.parquet"
    out.write_parquet(path)
    subprocess.run(["aws", "s3", "cp", "--only-show-errors", path, f"{S3}/cos_{split}.parquet"], check=True)
    print(f"{split}: {len(out):,} pair cosines, mean {out['cos'].mean():.3f}, uploaded, t={time.time() - t0:.0f}s", flush=True)


cosines("tune")
print("TUNE_UPLOADED", flush=True)
cosines("test")
print("GPU_DONE", flush=True)
