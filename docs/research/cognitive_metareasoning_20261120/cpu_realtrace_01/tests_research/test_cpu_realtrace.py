"""COG01B CPU real-trace: public behavior and leakage guards."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpu_realtrace import build_records, crossfit, summarize


def fake_dataset(n_positions=6):
    traces, assays, priors = [], [], {}
    positions = [{"id": f"P{i:03d}"} for i in range(n_positions)]
    for i in range(n_positions):
        pid = f"P{i:03d}"
        ck = "C01_gen_5"
        # Changing choices across budgets, with both beneficial and detrimental flips.
        decisions = [0, i % 2, 1, 0]
        policies = [
            [0.8, 0.2] if a == 0 else [0.2, 0.8]
            for a in decisions
        ]
        trace = {
            "checkpoint_id": ck, "position_id": pid,
            "trace_budgets": [8, 16, 32, 64],
            "trace_policies": policies, "trace_source": "fresh",
        }
        traces.append(trace)
        prior = [0.9, 0.1] if i % 2 == 0 else [0.1, 0.9]
        priors[(ck, pid)] = {"policy": prior, "value": 0.1}
        for b, p in zip((8, 16, 32, 64), policies):
            assays.append({
                "checkpoint_id": ck, "position_id": pid,
                "budget": b, "system": "A4",
                "search_continuation": "independent_restarts",
                "argmax_effective": p.index(max(p)),
                "argmax_prior": prior.index(max(prior)),
                "reference_best": i % 2,
            })
    return traces, assays, priors, positions


class CpuRealTraceTests(unittest.TestCase):
    def test_signed_changes_and_current_only_features(self):
        t, a, nn, pos = fake_dataset(2)
        records = build_records(t, a, nn, pos)
        self.assertEqual(len(records), 6)
        first = [x for x in records if x["position_id"] == "P001"]
        self.assertEqual([x["delta_reference"] for x in first], [1, 0, -1])
        self.assertTrue(all(x["search_continuation"] == "independent_restarts" for x in records))
        old_f = copy.deepcopy(first[0]["features"])
        changed = copy.deepcopy(t)
        changed[1]["trace_policies"][1] = [0.8, 0.2]
        changed_assays = copy.deepcopy(a)
        for assay in changed_assays:
            if assay["position_id"] == "P001" and assay["budget"] == 16:
                assay["argmax_effective"] = 0
        changed_records = build_records(changed, changed_assays, nn, pos)
        self.assertEqual(
            next(x for x in changed_records if x["position_id"] == "P001" and x["budget"] == 8)["features"],
            old_f,
        )

    def test_cannot_replicate_posthoc_systems(self):
        t, a, nn, pos = fake_dataset(2)
        dup = copy.deepcopy(a[0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_records(t, a + [dup], nn, pos)

    def test_wrong_trace_policy_agreement_rejected(self):
        t, a, nn, pos = fake_dataset(2)
        a[0]["argmax_effective"] = 99
        with self.assertRaisesRegex(ValueError, "argmax"):
            build_records(t, a, nn, pos)

    def test_missing_checkpoint_identity_rejected(self):
        t, a, nn, pos = fake_dataset(2)
        nn.pop(("C01_gen_5", "P001"))
        with self.assertRaisesRegex(ValueError, "prior"):
            build_records(t, a, nn, pos)

    def test_future_label_cannot_leak_into_oof_predictor(self):
        t, a, nn, pos = fake_dataset(9)
        rows = build_records(t, a, nn, pos)
        preds = crossfit(rows, folds=3)
        assert set(preds) == {"M0", "M1", "M2"}
        test_positions = sorted({r["position_id"] for r in rows})[::3]
        modified = copy.deepcopy(rows)
        for r in modified:
            if r["position_id"] in test_positions:
                r["delta_reference"] = 999
        altered = crossfit(modified, folds=3)
        for i, r in enumerate(rows):
            if r["position_id"] in test_positions:
                for method in preds:
                    self.assertAlmostEqual(preds[method][i], altered[method][i], places=12)

    def test_outputs_have_fixed_budget_levels_and_counts(self):
        t, a, nn, pos = fake_dataset(12)
        rows = build_records(t, a, nn, pos)
        preds = crossfit(rows, folds=3)
        summary = summarize(rows, preds, folds=3, bootstraps=100, seed=77)
        self.assertEqual(summary["n_rows"], 36)
        self.assertEqual(summary["n_positions"], 12)
        self.assertEqual(summary["n_checkpoints"], 1)
        self.assertEqual(set(summary["model_mse"]), {"M0", "M1", "M2"})
        self.assertEqual(sum(summary["signed_counts"].values()), 36)
        self.assertEqual(sorted(summary["budgets"], key=int), ["8", "16", "32"])

    def test_nonfinite_and_invalid_probabilities_rejected(self):
        t, a, nn, pos = fake_dataset(2)
        t[0]["trace_policies"][0][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "policy"):
            build_records(t, a, nn, pos)
        t, a, nn, pos = fake_dataset(2)
        t[0]["trace_policies"][0] = [0.8, -0.2]
        with self.assertRaisesRegex(ValueError, "policy"):
            build_records(t, a, nn, pos)


if __name__ == "__main__":
    unittest.main()
