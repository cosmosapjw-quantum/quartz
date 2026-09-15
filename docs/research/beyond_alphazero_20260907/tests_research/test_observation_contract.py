"""Observable W1 invariants, without an engine or a fitted posterior."""

from dataclasses import replace
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import observation_contract as contract
except ModuleNotFoundError as error:
    if error.name != "observation_contract":
        raise
    contract = None


class ObservationContractTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(contract, "W1 observation contract is not implemented")
        self.identity = contract.EpochIdentity("root", 0, 1, 1, 1)
        self.store = contract.BeliefStore(self.identity)

    def key(self, evaluation="eval-1", **changes):
        base = contract.ObservationKey(
            root_hash="root", state_hash="parent", action_id="left",
            leaf_hash="leaf", model_epoch=1, representation_epoch=1,
            target_epoch=1, evaluation_id=evaluation, evaluator_id="fixed-nn",
        )
        return replace(base, **changes)

    def enqueue(self, key, job="job-1", identity=None):
        pending = contract.PendingJob(job, identity or self.identity, key, 10)
        self.store.enqueue(pending)
        return pending

    def block(self, key, values=(1.0,), **changes):
        block = contract.CompletedBlock(
            key, contract.Moments.from_values(values), 11, 20,
            provenance="declared test generating process",
        )
        return replace(block, **changes)

    def finish(self, key, job="job-1", values=(1.0,)):
        self.enqueue(key, job)
        return self.store.complete(job, self.block(key, values))

    def test_exact_retransmission_is_idempotent(self):
        key = self.key()
        self.enqueue(key)
        block = self.block(key)
        first = self.store.complete("job-1", block)
        self.assertEqual(first.status, "accepted")
        self.assertEqual(self.store.complete("job-1", block), first)
        self.assertEqual(len(self.store.history), 1)
        self.assertEqual(self.store.counters.completed, 1)
        self.assertEqual(self.store.moments.n, 1)

    def test_transposition_and_reevaluation_do_not_create_evidence(self):
        self.finish(self.key())
        key = self.key("eval-2", state_hash="other-parent", action_id="right")
        self.assertEqual(self.finish(key, "job-2").status, "duplicate")
        self.assertEqual(self.store.counters.attempted, 2)
        self.assertEqual(self.store.counters.completed, 2)
        self.assertEqual(self.store.counters.unique_observations, 1)
        self.assertEqual(self.store.moments.n, 1)

    def test_out_of_order_results_only_count_after_completion(self):
        first, second = self.key(), self.key("eval-2", leaf_hash="leaf-2")
        self.enqueue(first)
        self.enqueue(second, "job-2")
        self.assertEqual(self.store.counters.pending, 2)
        self.assertEqual(self.store.moments.n, 0)
        self.store.complete("job-2", self.block(second, (3.0,)))
        self.assertEqual(self.store.moments.mean, 3.0)
        self.store.complete("job-1", self.block(first))
        self.assertEqual(self.store.counters.pending, 0)
        self.assertEqual(self.store.moments, contract.Moments.from_values((1.0, 3.0)))

    def test_root_change_quarantines_late_completion_and_preserves_history(self):
        first, late = self.key(), self.key("late", leaf_hash="late-leaf")
        self.finish(first)
        self.enqueue(late, "late-job")
        new = replace(self.identity, root_hash="successor", candidate_generation=1)
        self.store.invalidate("root advanced", self.identity, new)
        self.assertEqual(self.store.moments.n, 0)
        receipt = self.store.complete("late-job", self.block(late))
        self.assertEqual(receipt.status, "quarantined")
        self.assertEqual([x.status for x in self.store.history], ["accepted", "quarantined"])
        self.assertEqual(self.store.counters.pending, 0)

    def test_every_epoch_and_candidate_generation_invalidates_visible_evidence(self):
        for field in ("model_epoch", "representation_epoch", "target_epoch", "candidate_generation"):
            with self.subTest(field=field):
                self.store = contract.BeliefStore(self.identity)
                key = self.key()
                self.finish(key)
                late = self.key("late", leaf_hash="late-leaf")
                self.enqueue(late, "late-job")
                new = replace(self.identity, **{field: getattr(self.identity, field) + 1})
                self.store.invalidate(field, self.identity, new)
                self.assertEqual(self.store.moments.n, 0)
                self.assertEqual(self.store.complete("late-job", self.block(late)).status, "quarantined")
                self.assertEqual(len(self.store.history), 2)

    def test_new_epoch_accepts_a_new_evaluator_result(self):
        self.finish(self.key())
        new = replace(self.identity, model_epoch=2)
        self.store.invalidate("model replaced", self.identity, new)
        key = self.key("eval-new", model_epoch=2)
        self.enqueue(key, "job-new", new)
        self.assertEqual(self.store.complete("job-new", self.block(key, (2.0,))).status, "accepted")
        self.assertEqual(self.store.moments.mean, 2.0)
        self.assertEqual(len(self.store.history), 2)

    def test_identity_roundtrip_cannot_revalidate_an_old_pending_job(self):
        key = self.key()
        self.enqueue(key)
        interim = replace(self.identity, root_hash="interim-root")
        self.store.invalidate("advance", self.identity, interim)
        self.store.invalidate("return", interim, self.identity)
        self.assertEqual(self.store.complete("job-1", self.block(key)).status, "quarantined")
        self.assertEqual(self.store.moments.n, 0)

    def test_raw_evidence_is_not_automatically_replayed_after_root_change(self):
        self.finish(self.key())
        new = replace(self.identity, root_hash="new-root", candidate_generation=1)
        self.store.invalidate("advance", self.identity, new)
        key = self.key("eval-new", root_hash="new-root")
        self.enqueue(key, "job-new", new)
        self.assertEqual(self.store.complete("job-new", self.block(key)).status, "duplicate")
        self.assertEqual(self.store.moments.n, 0)
        self.assertEqual(len(self.store.history), 2)

    def test_conflicting_redelivery_preserves_both_raw_receipts_without_recounting(self):
        key = self.key()
        self.finish(key)
        conflict = self.block(key, (9.0,))
        receipt = self.store.complete("job-1", conflict)
        self.assertEqual(receipt.status, "rejected")
        self.assertEqual(self.store.complete("job-1", conflict), receipt)
        self.assertEqual(len(self.store.history), 2)
        self.assertEqual(self.store.counters.completed, 1)
        self.assertEqual(self.store.moments.mean, 1.0)

    def test_independent_noisy_repeats_require_distinct_declared_blocks(self):
        for index, value in enumerate((1.0, 2.0, 3.0)):
            key = self.key(
                str(index), observation_kind="stochastic",
                independent_block_id=f"block-{index}", rng_block_id=f"rng-{index}",
                generating_policy_id="fixed-rollout-v1",
            )
            self.finish(key, str(index), (value,))
        self.assertEqual(self.store.counters.unique_observations, 3)
        self.assertEqual(self.store.moments, contract.Moments(3, 2.0, 2.0))

    def test_renaming_rng_block_does_not_certify_independence(self):
        key = self.key(observation_kind="stochastic", independent_block_id="b1",
                       rng_block_id="rng-shared", generating_policy_id="rollout")
        self.finish(key)
        renamed = replace(key, evaluation_id="eval-2", independent_block_id="b2")
        self.assertEqual(self.finish(renamed, "job-2").status, "duplicate")
        reused = replace(key, evaluation_id="eval-3", rng_block_id="rng-other")
        self.assertEqual(self.finish(reused, "job-3").status, "rejected")
        self.assertEqual(self.store.moments.n, 1)

    def test_rng_block_alias_cannot_change_meaning_in_either_arrival_order(self):
        cases = (
            (("r1", "b1", "accepted"), ("r1", "b2", "duplicate"), ("r2", "b2", "rejected")),
            (("r1", "b1", "accepted"), ("r2", "b2", "accepted"), ("r1", "b2", "rejected")),
        )
        for case in cases:
            with self.subTest(case=case):
                self.store = contract.BeliefStore(self.identity)
                for index, (rng, block, expected) in enumerate(case):
                    key = self.key(
                        f"eval-{index}", observation_kind="stochastic",
                        independent_block_id=block, rng_block_id=rng,
                        generating_policy_id="fixed-rollout",
                    )
                    self.assertEqual(self.finish(key, f"job-{index}").status, expected)
                self.assertEqual(self.store.counters.completed, 3)
                self.assertEqual(self.store.counters.unique_observations,
                                 sum(expected == "accepted" for _, _, expected in case))

    def test_stochastic_observation_requires_declared_process_metadata(self):
        for missing in ("independent_block_id", "rng_block_id", "generating_policy_id"):
            metadata = dict(independent_block_id="b", rng_block_id="r", generating_policy_id="p")
            metadata[missing] = None
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                self.key(observation_kind="stochastic", **metadata)

    def test_inconsistent_duplicate_and_evaluation_id_collision_are_rejected(self):
        self.finish(self.key())
        same_leaf = self.key("eval-2")
        self.assertEqual(self.finish(same_leaf, "job-2", (9.0,)).status, "rejected")
        collision = self.key(leaf_hash="different-leaf")
        self.assertEqual(self.finish(collision, "job-3").status, "rejected")
        self.assertEqual(self.store.moments.mean, 1.0)

    def test_same_evaluation_retransmitted_through_another_job_is_duplicate(self):
        self.finish(self.key())
        key = self.key(state_hash="another-parent", action_id="another-action")
        self.assertEqual(self.finish(key, "job-2").status, "duplicate")
        self.assertEqual(self.store.counters.unique_observations, 1)

    def test_result_must_match_the_pending_observation(self):
        key = self.key()
        self.enqueue(key)
        wrong = replace(key, leaf_hash="not-requested")
        self.assertEqual(self.store.complete("job-1", self.block(wrong)).status, "rejected")
        self.assertEqual(self.store.moments.n, 0)
        self.assertEqual(self.store.counters.completed, 1)

    def test_cancel_and_failure_do_not_become_completed_evidence(self):
        self.enqueue(self.key())
        self.enqueue(self.key("eval-2"), "job-2")
        self.store.cancel("job-1", "no longer needed")
        self.store.fail("job-2", "evaluator failure")
        self.assertEqual(self.store.counters.pending, 0)
        self.assertEqual(self.store.counters.attempted, 2)
        self.assertEqual(self.store.counters.completed, 0)
        self.assertEqual(self.store.counters.cancelled, 1)
        self.assertEqual(self.store.counters.failed, 1)
        late = self.store.complete("job-1", self.block(self.key()))
        self.assertEqual(late.status, "quarantined")
        self.assertEqual(self.store.counters.completed, 1)
        self.assertEqual(self.store.moments.n, 0)

    def test_duplicate_enqueue_is_idempotent_but_job_id_collision_is_not(self):
        job = self.enqueue(self.key())
        self.store.enqueue(job)
        self.assertEqual(self.store.counters.attempted, 1)
        with self.assertRaises(ValueError):
            self.store.enqueue(replace(job, expected_observation=self.key("other")))

    def test_deterministic_block_cannot_hide_repeated_backups(self):
        with self.assertRaises(ValueError):
            self.block(self.key(), (1.0,) * 64)

    def test_block_merge_matches_serial_welford_and_large_offset_reference(self):
        values = (1e9 + 1, 1e9 + 4, 1e9 - 2, 1e9 + 6, 1e9 - 4, 1e9 + 2)
        serial = contract.Moments.from_values(values)
        merged = contract.Moments.from_values(values[:2]).merge(
            contract.Moments.from_values(values[2:5])).merge(
            contract.Moments.from_values(values[5:]))
        expected_mean = math.fsum(values) / len(values)
        expected_m2 = math.fsum((x - expected_mean) ** 2 for x in values)
        for moments in (serial, merged):
            self.assertEqual(moments.n, len(values))
            self.assertAlmostEqual(moments.mean, expected_mean, places=6)
            self.assertAlmostEqual(moments.m2, expected_m2, places=5)
        self.assertEqual(serial.merge(contract.Moments()), serial)
        self.assertEqual(contract.Moments().merge(serial), serial)

    def test_nonfinite_or_impossible_moments_and_times_are_rejected(self):
        for values in ((float("nan"),), (float("inf"),)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                contract.Moments.from_values(values)
        for args in ((-1, 0.0, 0.0), (1, 1.0, -1.0), (1, 1.0, 1.0), (0, 1.0, 0.0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                contract.Moments(*args)
        with self.assertRaises(ValueError):
            self.block(self.key(), started_ns=30, completed_ns=20)


if __name__ == "__main__":
    unittest.main()
