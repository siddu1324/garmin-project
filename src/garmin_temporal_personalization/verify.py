"""Independent reference calculations, split/transform replays and preservation."""

import json
from pathlib import Path
from collections import defaultdict
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from garmin_comprehensive.extract import REFERENCE as OLD_REFERENCE
from garmin_inventory.privacy import sha256
from garmin_preprocessing.common import verify_bundle, write_json
from .history import HISTORY, online_baseline
from .models import cohort, nested


def verify(root):
    root = Path(root)
    ptr = json.loads((root / "private/temporal_personalization/current.json").read_text())
    run, features = root / ptr["run"], root / ptr["features"]
    verify_bundle(features)
    if (run / "manifest.json").exists():
        sealed = verify_bundle(run)
        assert all(sha256(run / p) == h for p, h in sealed["job_manifest_sha256"].items())
    f = pd.read_parquet(features / "development.parquet")
    assert len(f) == 3706 and f.n_ema_id.is_unique and f.n_partition.eq("development").all()
    assert f[HISTORY].isna().all().all() and f.q_verified_history_count.eq(0).all()
    assert online_baseline(f, set(f.n_participant), 5).empty
    cfg = json.loads((run / "identity.json").read_text())["config"]
    identity = json.loads((run / "identity.json").read_text())
    assert identity["feature_manifest"] == sha256(features / "manifest.json")
    for n, h in identity["code"].items():
        assert sha256(Path(__file__).with_name(n)) == h
    fi = json.loads((features / "identity.json").read_text())
    for n, h in fi["source"].items():
        assert sha256(Path(__file__).with_name(n)) == h
    assert (
        fi["specification"]
        == identity["specification"]
        == sha256(root / "reports/temporal_personalization/specification.md")
    )
    source_hashes = json.loads((features / "source_hashes.json").read_text())
    assert all(sha256(root / p) == h for p, h in source_hashes.items())
    replays = inner_checks = fresh = candidates = nonconverged = 0
    failures = []
    members = {}
    for path in sorted((run / "jobs").glob("*/summary.json")):
        verify_bundle(path.parent)
        info = json.loads(path.read_text())
        s, t, k = info["spec"], info["task"], info["fold"]
        if info["status"] != "complete":
            failures.append(dict(task=t, model=s["name"], fold=k))
            continue
        g = cohort(f, s["cohort"])
        train, valid = g[g.n_fold.ne(k)], g[g.n_fold.eq(k)]
        assert not set(train.n_participant) & set(valid.n_participant)
        fitted = joblib.load(path.parent / "estimator.joblib")
        transform = fitted.transform.base
        pd.testing.assert_series_equal(transform.medians, train[transform.kept].median())
        if transform.scaler is not None:
            np.testing.assert_allclose(
                transform.scaler.mean_, train[transform.kept].fillna(transform.medians).mean()
            )
        for role, z in [("train", train), ("validation", valid)]:
            p = pd.read_parquet(path.parent / f"{role}.parquet")
            assert p.n_ema_id.tolist() == z.n_ema_id.tolist()
            np.testing.assert_array_equal(p.target, z[f"target_{t}"].to_numpy(int))
            with threadpool_limits(limits=1):
                np.testing.assert_allclose(
                    p.probability, fitted.probability(z), atol=1e-12, rtol=1e-12
                )
        membership = pd.read_parquet(path.parent / "inner_membership.parquet")
        key = (t, s["cohort"], k)
        if key in members:
            pd.testing.assert_frame_equal(members[key], membership)
        else:
            members[key] = membership
        for _, part in membership.groupby("inner_fold"):
            a, b = part[part.role.eq("train")], part[part.role.eq("validation")]
            assert not set(a.n_participant) & set(b.n_participant)
            assert set(a.n_ema_id) | set(b.n_ema_id) == set(train.n_ema_id)
            inner_checks += 1
        trials = json.loads((path.parent / "trials.json").read_text())
        candidates += len(trials)
        nonconverged += sum(not all(x["converged"]) for x in trials)
        if k == 0 and s["name"] in ["personal", "hgb_personal", "physiology"]:
            with threadpool_limits(limits=1):
                freshfit, _, _, _ = nested(train, t, s, cfg)
                np.testing.assert_allclose(
                    freshfit.probability(valid), fitted.probability(valid), atol=1e-12, rtol=1e-12
                )
            fresh += 1
        assert not any(
            n.startswith(("item_", "target_", "n_participant", "previous_", "historical_"))
            for n in fitted.transform.names
        )
        replays += 1
    # Independent Python loop over actual accepted records for one EMA per sampled person.
    people = json.loads(
        (root / cfg["normalized_bundle"] / "participant_directories.json").read_text()
    )
    trace = 0
    for person in people[:8]:
        z = f[f.n_participant.eq(person["participant"])].sort_values("n_actual_epoch")
        if z.empty:
            continue
        # Latest observed response makes multi-day history feasible without consulting outcomes.
        r = z.iloc[-1]
        c = float(r.n_cutoff_epoch)
        path = (
            root
            / cfg["ledger_bundle"]
            / "reading_dispositions"
            / person["directory"]
            / "hr.parquet"
        )
        g = pd.read_parquet(path)
        mins = defaultdict(list)
        days = set()
        for row in g.itertuples():
            if not row.a_first or not row.a_acceptable or not np.isfinite(row.n_epoch_s):
                continue
            t = float(row.n_epoch_s)
            if c - 7 * 86400 <= t < c - 3600:
                delta = abs(t % 86400 - c % 86400)
                if min(delta, 86400 - delta) <= 3600:
                    mins[int(t // 60)].append(float(row.n_value))
                    days.add(int(t // 86400))
        expected = (
            float(np.mean([sum(v) / len(v) for v in mins.values()]))
            if len(mins) >= 30 and len(days) >= 3
            else np.nan
        )
        np.testing.assert_allclose(expected, r.clock_reference_mean, equal_nan=True)
        pairs = pd.read_parquet(
            root / cfg["ledger_bundle"] / "counter_pairs" / person["directory"] / "steps.parquet"
        )
        rates = []
        for start, end in [(c - 172800, c - 86400), (c - 86400, c)]:
            seconds = added = 0.0
            for p in pairs.itertuples():
                if (
                    p.a_pair_accepted
                    and float(p.a_pair_start) >= start
                    and float(p.n_epoch_s) < end
                ):
                    seconds += p.a_pair_duration
                    added += p.a_pair_delta
            rates.append(60 * added / seconds if seconds >= 3600 else np.nan)
        np.testing.assert_allclose(rates[1] - rates[0], r.step_block_rate_change, equal_nan=True)
        trace += 1
    before = json.loads((root / "private/temporal_personalization/before.json").read_text())
    assert all(sha256(root / p) == h for p, h in before["files"].items())
    assert all(sha256(root / p) == h for p, h in before["manifests"].items())
    baseline = json.loads((root / "private/integrity_baseline.json").read_text())
    raw_checks = 0
    for label, folder in [("data", root / "data"), ("reference", OLD_REFERENCE)]:
        for name, record in baseline[label].items():
            st = (folder / name).stat()
            assert st.st_size == record["size"] and st.st_mtime_ns == record["mtime_ns"]
            raw_checks += 1
    result = dict(
        validation_replays=replays,
        training_replays=replays,
        inner_split_checks=inner_checks,
        independent_fresh_refits=fresh,
        candidate_evaluations=candidates,
        nonconverged_candidates=nonconverged,
        failed_jobs=failures,
        independent_ledger_traces=trace,
        source_partition_hashes=len(source_hashes),
        prior_files_preserved=len(before["files"]),
        prior_manifests_preserved=len(before["manifests"]),
        raw_metadata_checks=raw_checks,
        accounted_responses=len(f),
        real_history_prediction_rows=0,
        final_test_evaluated=False,
    )
    out = root / "reports/temporal_personalization" / run.name
    write_json(root / "private/temporal_personalization/verification.json", result)
    if not (out / "manifest.json").exists():
        write_json(out / "verification.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(verify(Path.cwd()), indent=2))
