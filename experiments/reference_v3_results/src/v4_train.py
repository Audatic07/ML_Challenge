"""Full-catalog v4 experiment with fixed fit/stop/tune/audit partitions.

Example: python -m src.v4_train --data /data/dataset --work /work/v4 \
    --fit 400000 --stop 10000 --tune 100000 --rounds 2200 --workers 8
Use --stage oracle for a retrieval-only diagnostic on canonical tuning queries.
The audit partition is reserved and is never evaluated here.
"""
from __future__ import annotations
import argparse
import json
import os
import platform
import time
from pathlib import Path
import importlib.metadata
import lightgbm as lgb
import numpy as np
import polars as pl
from .config import LGB_PARAMS
from .metric import per_entity, f05_macro, unique_assign
from .splits import make_split, split_manifest
from .v4_retrieval import (SparseCatalog, prepare_catalog, load_queries, raw_batches,
                          atomic_json, sha256, digest, CHANNELS)
from .v4_features import build_features, FEATURES


def partitions(n, fit_size, stop_size, tune_size):
    if n >= 310000:
        split = make_split(n, min(fit_size + 10000, n - 100000 - n // 10), tune_size)
        perm = np.random.default_rng(42).permutation(n)
        stop_reserve = perm[300000:310000].astype(np.uint32)
        if len(stop_reserve) != 10000 or stop_size > 10000:
            raise ValueError("Need a full fixed early-stopping reserve; stop <= 10000")
        excluded = np.concatenate([split.held_out, stop_reserve])
        fit = np.sort(perm[~np.isin(perm, excluded)][:fit_size]).astype(np.uint32)
        stop = np.sort(stop_reserve[:stop_size])
    else:
        # Preserve a locked 10% audit and use 10% for full-catalog tuning,
        # then spend the remaining population on fit/early stop.
        audit_size = int(n * 0.10)
        tune_size = min(tune_size, max(1, n // 10))
        validation_start = n - audit_size - tune_size
        if validation_start <= 0:
            raise ValueError("Dataset is too small for separate tune and audit partitions")
        split = make_split(n, validation_start, tune_size,
                           validation_start=validation_start, validation_size=tune_size)
        max_stop = min(10000, max(1, validation_start // 20))
        stop_size = min(stop_size, max_stop)
        stop_reserve = split.train[:max_stop]
        fit_size = min(fit_size, len(split.train) - max_stop)
        fit = split.train[max_stop:max_stop + fit_size]
        stop = stop_reserve[:stop_size]
        excluded = np.concatenate([split.held_out, stop_reserve])
    return split, fit, stop, split.validation, np.unique(excluded)


def labels(data_dir, ids, queries, heldout_rows):
    heldout_ids = ids["entity_id"].gather(pl.Series(heldout_rows)).rename("source1_entity_id")
    qmap = queries.select(pl.col("id").alias("source1_entity_id"), pl.col("row").alias("s1"))
    truth_parts, excluded_parts = [], []
    for batch in raw_batches(Path(data_dir) / "train" / "train_ground_truth.tsv"):
        selected = batch.join(qmap, on="source1_entity_id", how="inner")
        selected = (selected.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
                    .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
                    .select("s1", pl.col("matched_entity_ids").alias("target_id")))
        truth_parts.append(selected)
        held = batch.filter(pl.col("source1_entity_id").is_in(heldout_ids.implode()))
        excluded_parts.append(held.select(pl.col("matched_entity_ids").fill_null("").str.split(","))
            .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
            .rename({"matched_entity_ids": "target_id"}))
    truth = pl.concat(truth_parts)
    if truth.n_unique() != len(truth):
        raise ValueError("Duplicate labelled query/target pairs")
    return truth, pl.concat(excluded_parts).unique()


def metric_truth(truth, candidates):
    # Evaluation uses string target IDs, including positives never retrieved.
    return truth.rename({"target_id": "t"}), candidates.select("s1", pl.col("target_id").alias("t"))


def report(rows, truth, feats, queries, scores=None, threshold=None):
    selected = feats.filter(pl.col("s1").is_in(pl.Series(rows).implode()))
    gt = truth.filter(pl.col("s1").is_in(pl.Series(rows).implode())).rename({"target_id": "t"})
    candidate = selected.select("s1", pl.col("target_id").alias("t"))
    hit = candidate.join(gt, on=["s1", "t"])
    per = per_entity(rows, gt, hit).rename({"f": "oracle_f05"})
    if scores is not None:
        pred = scores.filter(pl.col("s1").is_in(pl.Series(rows).implode()) & (pl.col("p") >= threshold))
        pred = unique_assign(pred)
        actual = per_entity(rows, gt, pred).select("s1", pl.col("f").alias("macro_f05"))
        per = per.join(actual, on="s1")
    per = per.join(queries.select(pl.col("row").alias("s1"), "country"), on="s1")
    metrics = {"queries": len(rows), "truth_pairs": len(gt), "candidate_pairs": len(candidate),
        "link_recall": len(hit) / max(1, len(gt)), "oracle_f05": per["oracle_f05"].mean(),
        "zero_candidates": len(rows) - candidate["s1"].n_unique(),
        "candidate_count_quantiles": candidate.group_by("s1").len()["len"].quantile(0.95),
        "partition": "development; full target catalog; audit unopened"}
    aggregate = [pl.len().alias("queries"), pl.col("oracle_f05").mean()]
    if scores is not None:
        metrics["macro_f05"] = per["macro_f05"].mean()
        metrics["matching_gap"] = metrics["oracle_f05"] - metrics["macro_f05"]
        aggregate.append(pl.col("macro_f05").mean())
    metrics["country"] = per.group_by("country").agg(aggregate).to_dicts()
    return metrics, per


def train_model(feats, truth, excluded, fit, stop, tune, queries, work, rounds, workers, pins):
    labelled = feats.join(truth.with_columns(pl.lit(1, pl.Int8).alias("y")), on=["s1", "target_id"], how="left")
    labelled = labelled.with_columns(pl.col("y").fill_null(0))
    fitframe = labelled.filter(pl.col("s1").is_in(pl.Series(fit).implode()))
    if fitframe.filter(pl.col("y") == 1).join(excluded, on="target_id", how="semi").height:
        raise ValueError("A labelled target crosses fit/held-out businesses")
    fitframe = fitframe.join(excluded, on="target_id", how="anti")
    stopframe = labelled.filter(pl.col("s1").is_in(pl.Series(stop).implode()))
    tuneframe = labelled.filter(pl.col("s1").is_in(pl.Series(tune).implode()))
    if len(fitframe) == 0 or fitframe['y'].n_unique() < 2 or len(stopframe) == 0:
        raise ValueError("Pilot lacks training classes/early-stopping candidates")
    params = dict(LGB_PARAMS, num_threads=workers, deterministic=True, force_col_wise=True)
    dtrain = lgb.Dataset(fitframe.select(FEATURES).to_numpy(), label=fitframe['y'].to_numpy(), feature_name=FEATURES)
    dstop = lgb.Dataset(stopframe.select(FEATURES).to_numpy(), label=stopframe['y'].to_numpy(), reference=dtrain)
    booster = lgb.train(params, dtrain, num_boost_round=rounds, valid_sets=[dstop], valid_names=['stop'],
        callbacks=[lgb.early_stopping(75), lgb.log_evaluation(100)])
    score = booster.predict(tuneframe.select(FEATURES).to_numpy(), num_threads=workers)
    scores = tuneframe.select('s1', pl.col('target_id').alias('t')).with_columns(pl.Series('p', score))
    assigned = unique_assign(scores)
    gt = truth.filter(pl.col('s1').is_in(pl.Series(tune).implode())).rename({'target_id':'t'})
    grid = [(float(t), f05_macro(tune, gt, assigned.filter(pl.col('p') >= t)))
            for t in np.unique(np.r_[np.arange(.2,.961,.025), [.975,.99,.995]])]
    threshold, score = max(grid, key=lambda pair: pair[1])
    metrics, per = report(tune, truth, feats, queries, scores, threshold)
    per.write_parquet(work / 'tune_per_query.parquet')
    scores.write_parquet(work / 'tune_scores.parquet')
    booster.save_model(str(work / 'model.txt'))
    meta = {**pins, 'features':FEATURES, 'threshold':threshold, 'threshold_grid':grid,
        'model_sha256':sha256(work/'model.txt'), 'params':params, 'best_iteration':booster.best_iteration,
        'metrics':metrics, 'fit_pairs':len(fitframe), 'heldout_targets_excluded':len(excluded),
        'model_license':'MIT (LightGBM); no external pretrained weights',
        'tree_count':booster.num_trees(), 'audit_opened':False}
    # Tree ensemble size: leaves and split thresholds, well below an 8B scalar cap.
    dumped = booster.dump_model()['tree_info']
    leaves = sum(t['num_leaves'] for t in dumped)
    meta['parameter_count'] = 2 * leaves - len(dumped)
    atomic_json(work / 'model_manifest.json', meta)
    print('TRAINING_COMPLETE ' + json.dumps(metrics), flush=True)
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--fit', type=int, default=400000)
    parser.add_argument('--stop', type=int, default=10000)
    parser.add_argument('--tune', type=int, default=100000)
    parser.add_argument('--rounds', type=int, default=2200)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--max-df', type=float, default=.1)
    parser.add_argument('--dimensions', type=int, default=2**19)
    parser.add_argument('--stage', choices=['oracle','train'], default='train')
    args = parser.parse_args()
    started = time.monotonic()
    args.work.mkdir(parents=True, exist_ok=True)
    cache = args.cache or args.work / 'catalog'
    ids = pl.read_csv(args.data/'train'/'train_source1.tsv', separator='\t', quote_char=None,
                      columns=['entity_id'], infer_schema=False)
    if ids['entity_id'].n_unique() != len(ids):
        raise ValueError('Duplicate S1 IDs')
    split, fit, stop, tune, heldout = partitions(len(ids), args.fit, args.stop, args.tune)
    rows = tune if args.stage == 'oracle' else np.sort(np.concatenate([fit,stop,tune]))
    pin = {'split':split_manifest(split,ids['entity_id']),
           'selected_rows_sha':digest(rows.tolist()), 'fit_rows_sha':digest(fit.tolist()),
           'stop_rows_sha':digest(stop.tolist()), 'tune_rows_sha':digest(tune.tolist()),
           'fit_queries':len(fit), 'stop_queries':len(stop), 'tune_queries':len(tune),
           'early_stop_reserve':('permutation[300000:310000]' if len(ids) >= 310000
                                 else 'fixed prefix of fit pool; selected row hashes pinned'),
           'repeated_fingerprint_grouping':'not audited; no duplicate S1 ID; shared-target check enforced',
           'data_source1_sha':sha256(args.data/'train'/'train_source1.tsv'),
           'ground_truth_sha':sha256(args.data/'train'/'train_ground_truth.tsv'),
           'source_code_sha':{p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},
           'versions':{n:importlib.metadata.version(n) for n in ['numpy','polars','lightgbm','scikit-learn','scipy','sparse-dot-topn','rapidfuzz','Unidecode']},
           'python':platform.python_version()}
    atomic_json(args.work/'run_manifest.json',pin)
    queries = load_queries(args.data,rows)
    truth, excluded = labels(args.data,ids,queries,heldout)
    del ids
    manifest = prepare_catalog(args.data,cache)
    retriever = SparseCatalog(cache,manifest,top_k=args.top_k,workers=args.workers,
                             dimensions=args.dimensions,max_df=args.max_df)
    candidates = retriever.retrieve(queries,args.work/'retrieval')
    feats = build_features(queries,candidates,cache,manifest,args.work/'features',args.workers)
    for name, selected in [('tune',tune)]:
        metrics, per = report(selected,truth,feats,queries)
        atomic_json(args.work/f'{name}_retrieval_metrics.json',metrics)
        print('RETRIEVAL_COMPLETE '+json.dumps(metrics),flush=True)
    pin.update(retrieval=retriever.policy,candidate_sha=sha256(args.work/'retrieval'/'candidates.parquet'))
    if args.stage == 'train':
        train_model(feats,truth,excluded,fit,stop,tune,queries,args.work,args.rounds,args.workers,pin)
    atomic_json(args.work/'complete.json',{'stage':args.stage,'seconds':time.monotonic()-started,'pins':digest(pin)})


if __name__ == '__main__':
    main()
