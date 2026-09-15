"""W1 single-owner observation reducer; no engine integration or posterior.

An evaluation attempt, a completed job, a unique evidence block, and a raw
value count are different units. Aggregate moments are bookkeeping over the
visible blocks, potentially from different latent targets; W2 must group by
target/operator before fitting a likelihood. Neither ``Moments.n`` nor block
IDs certify independence, effective sample size, or epistemic confidence.

Leaf hashes must identify complete evaluator inputs, including player/value
perspective and any relevant history. Epoch and evaluator IDs are supplied by
the caller. Deterministic transpositions share evidence regardless of paths,
root, action or evaluation ID. Stochastic metadata declares the generating
process and RNG block; it does not establish that blocks are independent.
Within-block dependence is retained as unspecified: its moments cannot be
interpreted as n independent observations without a separate likelihood.

Invalidation empties the visible view and preserves raw receipt history.
There is deliberately no automatic evidence transport/replay across roots.
This reference requires one owner/reducer and is not thread safe.
"""

from dataclasses import dataclass
import math
from typing import Iterable


@dataclass(frozen=True)
class EpochIdentity:
    root_hash: str
    candidate_generation: int
    model_epoch: int
    representation_epoch: int
    target_epoch: int


@dataclass(frozen=True)
class ObservationKey:
    root_hash: str
    state_hash: str
    action_id: str
    leaf_hash: str
    model_epoch: int
    representation_epoch: int
    target_epoch: int
    evaluation_id: str
    evaluator_id: str
    observation_kind: str = "deterministic"
    independent_block_id: str | None = None
    rng_block_id: str | None = None
    generating_policy_id: str | None = None

    def __post_init__(self):
        if not all((self.root_hash, self.state_hash, self.action_id, self.leaf_hash,
                    self.evaluation_id, self.evaluator_id)):
            raise ValueError("observation identity fields must be nonempty")
        metadata = (self.independent_block_id, self.rng_block_id, self.generating_policy_id)
        if self.observation_kind == "deterministic":
            if any(value is not None for value in metadata):
                raise ValueError("deterministic evidence cannot declare stochastic blocks")
        elif self.observation_kind == "stochastic":
            if not all(metadata):
                raise ValueError("stochastic evidence needs block, RNG and generating-policy IDs")
        else:
            raise ValueError("unknown observation kind")

    @property
    def evidence_identity(self):
        """Canonical evaluator input; stochastic identity includes the RNG block."""
        base = (self.leaf_hash, self.evaluator_id, self.model_epoch,
                self.representation_epoch, self.target_epoch, self.observation_kind)
        if self.observation_kind == "stochastic":
            return base + (self.generating_policy_id, self.rng_block_id)
        return base

    def matches_epoch(self, identity):
        return (self.root_hash, self.model_epoch, self.representation_epoch,
                self.target_epoch) == (identity.root_hash, identity.model_epoch,
                                       identity.representation_epoch, identity.target_epoch)


@dataclass(frozen=True)
class Moments:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def __post_init__(self):
        if not isinstance(self.n, int) or self.n < 0:
            raise ValueError("moment count must be a nonnegative integer")
        if not math.isfinite(self.mean) or not math.isfinite(self.m2) or self.m2 < 0:
            raise ValueError("moments must be finite with nonnegative centered sum")
        if (self.n == 0 and (self.mean != 0 or self.m2 != 0)) or (self.n == 1 and self.m2 != 0):
            raise ValueError("empty/singleton moments are inconsistent")

    @classmethod
    def from_values(cls, values: Iterable[float]):
        n, mean, m2 = 0, 0.0, 0.0
        for value in values:
            if not math.isfinite(value):
                raise ValueError("observations must be finite")
            n += 1
            delta = value - mean
            mean += delta / n
            m2 += delta * (value - mean)
        return cls(n, mean, m2)

    def merge(self, other):
        """Chan/Welford merge; arithmetic counts are not statistical ESS."""
        if self.n == 0:
            return other
        if other.n == 0:
            return self
        n = self.n + other.n
        delta = other.mean - self.mean
        return Moments(n, self.mean + delta * other.n / n,
                       self.m2 + other.m2 + delta * delta * self.n * other.n / n)

    @property
    def sum_value(self):
        return self.n * self.mean


@dataclass(frozen=True)
class CompletedBlock:
    key: ObservationKey
    moments: Moments
    started_ns: int
    completed_ns: int
    depth: int = 0
    provenance: str = ""

    def __post_init__(self):
        if self.moments.n < 1:
            raise ValueError("a completed block must contain a value")
        if self.key.observation_kind == "deterministic" and self.moments.n != 1:
            raise ValueError("repeated deterministic backups are one evidence item")
        if not 0 <= self.started_ns <= self.completed_ns or self.depth < 0:
            raise ValueError("invalid completion timing or depth")

    @property
    def n_completed(self):
        return self.moments.n

    @property
    def sum_value(self):
        return self.moments.sum_value

    @property
    def centered_sum_sq(self):
        return self.moments.m2


@dataclass(frozen=True)
class PendingJob:
    job_id: str
    identity: EpochIdentity
    expected_observation: ObservationKey
    enqueued_ns: int


@dataclass(frozen=True)
class Receipt:
    job_id: str
    block: CompletedBlock
    status: str
    reason: str


@dataclass(frozen=True)
class Counters:
    attempted: int
    completed: int
    pending: int
    unique_observations: int  # Visible unique blocks, not raw values or ESS.
    cancelled: int
    failed: int


class BeliefStore:
    """Single-owner reducer; the name does not imply a fitted belief model.

    ``completed`` counts the first delivered result for a known job, including
    rejected/quarantined results. Retransmission does not add another job.
    ``attempted`` counts enqueues, not hardware starts. Cancellation/failure
    counts may overlap completed when a late result subsequently arrives.
    """

    def __init__(self, identity: EpochIdentity):
        self.identity = identity
        self._jobs = {}
        self._pending = set()
        self._invalidated_jobs = set()
        self._terminal = {}
        self._receipts = {}
        self._history = []
        self._evidence = {}
        self._evaluation_ids = {}
        self._independent_blocks = {}
        self._visible = {}
        self.invalidations = []

    def enqueue(self, job: PendingJob):
        previous = self._jobs.get(job.job_id)
        if previous is not None:
            if previous != job:
                raise ValueError("job ID reused with a different request")
            return
        if job.identity != self.identity or not job.expected_observation.matches_epoch(job.identity):
            raise ValueError("pending job does not match the current identity")
        if not job.job_id or job.enqueued_ns < 0:
            raise ValueError("invalid job identity or enqueue time")
        self._jobs[job.job_id] = job
        self._pending.add(job.job_id)

    def _finish_without_result(self, job_id, status, reason):
        if job_id not in self._jobs:
            raise KeyError(job_id)
        if job_id in self._receipts:
            raise ValueError("job already completed")
        previous = self._terminal.get(job_id)
        if previous is not None and previous != (status, reason):
            raise ValueError("job already terminated differently")
        self._pending.discard(job_id)
        self._terminal[job_id] = (status, reason)

    def cancel(self, job_id, reason):
        self._finish_without_result(job_id, "cancelled", reason)

    def fail(self, job_id, reason):
        self._finish_without_result(job_id, "failed", reason)

    def complete(self, job_id: str, block: CompletedBlock):
        job = self._jobs[job_id]
        previous = self._receipts.get(job_id)
        if previous is not None:
            if previous.block == block:
                return previous
            # Keep the first receipt; conflicting redelivery is also raw history.
            receipt = Receipt(job_id, block, "rejected", "inconsistent retransmission")
            if receipt not in self._history:
                self._history.append(receipt)
            return receipt

        self._pending.discard(job_id)
        key = block.key
        canonical = key.evidence_identity
        declared_block = (canonical[:-1] + (key.independent_block_id,)
                          if key.observation_kind == "stochastic" else None)
        signature = (canonical, block.moments)
        status, reason = "accepted", "new evidence block"
        if key != job.expected_observation or block.started_ns < job.enqueued_ns:
            status, reason = "rejected", "result does not match pending request"
        elif job_id in self._terminal:
            status, reason = "quarantined", "result arrived after cancellation/failure"
        elif job_id in self._invalidated_jobs or job.identity != self.identity:
            status, reason = "quarantined", "result belongs to an invalidated identity"
        elif key.evaluation_id in self._evaluation_ids and self._evaluation_ids[key.evaluation_id] != signature:
            status, reason = "rejected", "evaluation ID collision"
        elif (declared_block in self._independent_blocks
              and self._independent_blocks[declared_block] != canonical):
            status, reason = "rejected", "independent block reused with different RNG"
        elif canonical in self._evidence:
            if self._evidence[canonical].moments == block.moments:
                status, reason = "duplicate", "canonical evidence already recorded"
            else:
                status, reason = "rejected", "inconsistent canonical duplicate"
        receipt = Receipt(job_id, block, status, reason)
        self._receipts[job_id] = receipt
        self._history.append(receipt)
        if status in ("accepted", "duplicate"):
            self._evaluation_ids[key.evaluation_id] = signature
            if declared_block is not None:
                self._independent_blocks[declared_block] = canonical
        if status == "accepted":
            self._evidence[canonical] = block
            self._visible[canonical] = block
        return receipt

    def invalidate(self, reason: str, old_identity: EpochIdentity, new_identity: EpochIdentity):
        if old_identity != self.identity:
            raise ValueError("invalidation starts from a stale identity")
        if new_identity == old_identity or not reason:
            raise ValueError("invalidation needs a changed identity and a reason")
        self.invalidations.append((reason, old_identity, new_identity))
        self._invalidated_jobs.update(self._pending)
        self.identity = new_identity
        self._visible.clear()

    @property
    def visible_blocks(self):
        return tuple(self._visible.values())

    @property
    def history(self):
        return tuple(self._history)

    @property
    def moments(self):
        result = Moments()
        for block in self._visible.values():
            result = result.merge(block.moments)
        return result

    @property
    def counters(self):
        return Counters(len(self._jobs), len(self._receipts), len(self._pending),
                        len(self._visible),
                        sum(status == "cancelled" for status, _ in self._terminal.values()),
                        sum(status == "failed" for status, _ in self._terminal.values()))
