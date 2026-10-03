"""A preference model fitted to the raters' choices.

Every other scorer in the comparison predates the rating study. This one is fitted to it,
and answers a different question: how much of the raters' agreement is reachable from the
quantities the other scorers compute. It is an upper bound, and it holds only while the
model never sees the pair it is scored on. Three properties enforce that.

Scoring is antisymmetric. A pair scores as `w . (f(a) - f(b))` with no intercept. An
intercept would make the decision depend on which description was labelled `a`, so the same
pair swapped could be preferred either way.

Folds are disjoint by lecture rather than by pair. Two pairs from one lecture share a slide
deck, a lecturer and often a moment, so splitting them across folds leaves the answer in the
training half. Every pair is predicted by a fit that saw no pair from its lecture.

One rater is held out entirely. The fit pools the remaining raters; the held-out labels are
read once, against frozen weights. `freeze` and `score_frozen` are that path and cannot
refit.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..core.errors import ContractError
from ..core.ids import lecture_of

#: The sides a label may take; anything else is not a choice between the two.
SIDES = ("a", "b")


@dataclass(frozen=True)
class FeaturedPair:
    """One pair, with a number per feature on each side."""

    pair_id: str
    lecture: str
    a: Mapping[str, float]
    b: Mapping[str, float]

    def difference(self, features: Sequence[str]) -> list[float] | None:
        """`f(a) - f(b)`, or None when either side is missing a feature.

        Missing is not zero: a feature absent on one side would otherwise read as
        the two sides being equal on it, which is a measurement the pair does not
        carry.
        """
        out = []
        for name in features:
            left, right = self.a.get(name), self.b.get(name)
            if left is None or right is None:
                return None
            out.append(float(left) - float(right))
        return out


@dataclass(frozen=True)
class Head:
    """Fitted weights, one per feature, scored without an intercept."""

    features: tuple[str, ...]
    weights: tuple[float, ...]
    spread: tuple[float, ...] = ()
    folds: int = 0

    def __post_init__(self) -> None:
        if len(self.features) != len(self.weights):
            raise ContractError(
                f"{len(self.features)} features and {len(self.weights)} weights"
            )

    def margin(self, difference: Sequence[float]) -> float:
        if len(difference) != len(self.weights):
            raise ContractError(
                f"a pair with {len(difference)} differences against {len(self.weights)} weights"
            )
        return sum(weight * value for weight, value in zip(self.weights, difference, strict=True))

    def prefers(self, difference: Sequence[float]) -> str:
        """Which side it prefers. Antisymmetric, so the swapped pair gives the other."""
        return "a" if self.margin(difference) > 0 else "b"

    def held(self, difference: Sequence[float], name: str, at: float = 0.0) -> list[float]:
        """One pair's differences with one feature fixed, for a counterfactual.

        Used to ask what the head prefers between two descriptions of equal length.
        The feature stays in the fit, where it absorbs what it explains, and is held
        only when the question is asked: dropping it from the fit instead would push
        what it explains into whichever feature correlates with it.
        """
        if name not in self.features:
            raise ContractError(f"no feature {name!r}; known: {list(self.features)}")
        out = list(difference)
        out[self.features.index(name)] = at
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "features": list(self.features),
            "weights": [round(weight, 6) for weight in self.weights],
            "spread": [round(value, 6) for value in self.spread],
            "folds": self.folds,
        }


def logistic_weights(
    rows: Sequence[Sequence[float]],
    outcomes: Sequence[int],
    counts: Sequence[float] | None = None,
    *,
    strength: float = 1.0,
    steps: int = 200,
    tolerance: float = 1e-8,
) -> list[float]:
    """Weights of a logistic model with an intercept, fitted by Newton's method.

    `strength` is the weight of the ridge term on the coefficients. The intercept is
    fitted, because leaving it out during the fit would push a position preference
    into the coefficients, and dropped at scoring, which is what makes the score
    antisymmetric.
    """
    if not rows:
        raise ContractError("there is nothing to fit on")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ContractError("the rows do not all carry the same features")
    if len(outcomes) != len(rows):
        raise ContractError(f"{len(rows)} rows and {len(outcomes)} outcomes")
    weight_of = list(counts) if counts is not None else [1.0] * len(rows)
    if len(weight_of) != len(rows):
        raise ContractError(f"{len(rows)} rows and {len(weight_of)} counts")

    # The intercept rides as a constant column and is left out of the ridge term.
    design = [[1.0, *row] for row in rows]
    size = width + 1
    beta = [0.0] * size
    for _ in range(steps):
        gradient = [0.0] * size
        hessian = [[0.0] * size for _ in range(size)]
        for row, outcome, count in zip(design, outcomes, weight_of, strict=True):
            linear = sum(value * coefficient for value, coefficient in zip(row, beta, strict=True))
            chance = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, linear))))
            residual = count * (outcome - chance)
            curve = count * chance * (1.0 - chance)
            for index, value in enumerate(row):
                gradient[index] += residual * value
                for other in range(index, size):
                    hessian[index][other] += curve * value * row[other]
        for index in range(1, size):
            gradient[index] -= strength * beta[index]
            hessian[index][index] += strength
        for index in range(size):
            for other in range(index):
                hessian[index][other] = hessian[other][index]
        # A ridge on the intercept only, small enough not to move it, so that a
        # separable fold still has one answer instead of failing to solve.
        hessian[0][0] += 1e-8
        step = _solve(hessian, gradient)
        beta = [value + move for value, move in zip(beta, step, strict=True)]
        if max(abs(move) for move in step) < tolerance:
            break
    return beta[1:]


def _solve(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    """Gauss elimination with partial pivoting."""
    size = len(vector)
    rows = [[*row, value] for row, value in zip(matrix, vector, strict=True)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda index: abs(rows[index][column]))
        if abs(rows[pivot][column]) < 1e-12:
            raise ContractError("the features are collinear; the fit has no single answer")
        rows[column], rows[pivot] = rows[pivot], rows[column]
        for below in range(column + 1, size):
            factor = rows[below][column] / rows[column][column]
            for across in range(column, size + 1):
                rows[below][across] -= factor * rows[column][across]
    out = [0.0] * size
    for column in reversed(range(size)):
        total = rows[column][size] - sum(
            rows[column][across] * out[across] for across in range(column + 1, size)
        )
        out[column] = total / rows[column][column]
    return out


def sign_stable(
    pairs: Mapping[str, FeaturedPair],
    labels: Mapping[str, Mapping[str, str]],
    features: Sequence[str],
    *,
    keep: Sequence[str] = (),
) -> tuple[list[str], list[str]]:
    """The features every rater pulls the same way, and the ones they do not.

    A feature two raters weight in opposite directions describes one rater rather
    than the preference, and pooling the two averages it towards nothing while
    leaving it free to fit noise. Names in `keep` are kept whatever their signs, so
    that a baseline feature stays in and its instability is visible instead.
    """
    signs: dict[str, list[float]] = {name: [] for name in features}
    for rater in sorted(labels):
        rows, outcomes = [], []
        for pair_id, side in labels[rater].items():
            pair = pairs.get(pair_id)
            if side not in SIDES or pair is None:
                continue
            difference = pair.difference(features)
            if difference is None:
                continue
            rows.append(difference)
            outcomes.append(1 if side == "a" else 0)
        if not rows:
            continue
        for name, weight in zip(features, logistic_weights(rows, outcomes), strict=True):
            if weight != 0:
                signs[name].append(math.copysign(1.0, weight))
    stable = [name for name in features if len(set(signs[name])) <= 1]
    dropped = [name for name in features if name not in stable]
    for name in keep:
        if name in features and name not in stable:
            stable.append(name)
            dropped.remove(name)
    return [name for name in features if name in stable], dropped


def _training_rows(
    pairs: Mapping[str, FeaturedPair],
    labels: Mapping[str, Mapping[str, str]],
    features: Sequence[str],
    pair_ids: Sequence[str],
) -> tuple[list[list[float]], list[int], list[float]]:
    """One row per side a rater chose, counted by how many raters chose it.

    Counting rather than repeating keeps a pair three raters agree on three times as
    heavy as one a single rater saw, without letting the number of raters who happened
    to answer change what a pair means.
    """
    rows: list[list[float]] = []
    outcomes: list[int] = []
    counts: list[float] = []
    for pair_id in pair_ids:
        pair = pairs.get(pair_id)
        if pair is None:
            continue
        difference = pair.difference(features)
        if difference is None:
            continue
        votes = {side: 0 for side in SIDES}
        for rater in labels:
            side = labels[rater].get(pair_id)
            if side in votes:
                votes[side] += 1
        for side, count in votes.items():
            if count:
                rows.append(difference)
                outcomes.append(1 if side == "a" else 0)
                counts.append(float(count))
    return rows, outcomes, counts


def fit(
    pairs: Mapping[str, FeaturedPair],
    labels: Mapping[str, Mapping[str, str]],
    features: Sequence[str],
    *,
    pair_ids: Sequence[str] | None = None,
    strength: float = 1.0,
) -> Head:
    """One head on the pairs given, pooling the raters."""
    chosen = list(pair_ids) if pair_ids is not None else sorted(pairs)
    rows, outcomes, counts = _training_rows(pairs, labels, features, chosen)
    if not rows:
        raise ContractError("no pair carries every feature and a rater's choice")
    weights = logistic_weights(rows, outcomes, counts, strength=strength)
    return Head(tuple(features), tuple(weights), folds=1)


@dataclass
class OutOfFold:
    """What each pair was predicted by a fit that never saw its lecture."""

    features: tuple[str, ...]
    preferred: dict[str, str] = field(default_factory=dict)
    margin: dict[str, float] = field(default_factory=dict)
    fold_of: dict[str, str] = field(default_factory=dict)
    weights: tuple[float, ...] = ()
    spread: tuple[float, ...] = ()
    folds: int = 0
    skipped: int = 0

    def head(self) -> Head:
        """The average of the folds' weights, to freeze and carry to a held-out rater."""
        return Head(self.features, self.weights, self.spread, self.folds)

    def as_dict(self) -> dict[str, Any]:
        return {
            "features": list(self.features),
            "pairs": len(self.preferred),
            "skipped_for_missing_features": self.skipped,
            "folds": self.folds,
            "fold_disjoint_on": "lecture",
            "mean_weights": {
                name: round(weight, 4)
                for name, weight in zip(self.features, self.weights, strict=True)
            },
            "weight_spread": {
                name: round(value, 4)
                for name, value in zip(self.features, self.spread, strict=True)
            },
        }


def out_of_fold(
    pairs: Mapping[str, FeaturedPair],
    labels: Mapping[str, Mapping[str, str]],
    features: Sequence[str],
    *,
    strength: float = 1.0,
    fold_of: Callable[[FeaturedPair], str] | None = None,
) -> OutOfFold:
    """Predict every pair from a fit that saw no pair of its lecture."""
    name_of = fold_of or (lambda pair: pair.lecture)
    folds: dict[str, list[str]] = {}
    for pair_id, pair in pairs.items():
        folds.setdefault(name_of(pair), []).append(pair_id)
    if len(folds) < 2:
        raise ContractError(
            "out-of-fold prediction needs at least two lectures; "
            f"these pairs come from {sorted(folds)}"
        )
    found = OutOfFold(tuple(features))
    per_fold: list[Sequence[float]] = []
    for fold, held in sorted(folds.items()):
        elsewhere = [pair_id for pair_id in pairs if pair_id not in set(held)]
        rows, outcomes, counts = _training_rows(pairs, labels, features, elsewhere)
        if not rows:
            continue
        weights = logistic_weights(rows, outcomes, counts, strength=strength)
        per_fold.append(weights)
        head = Head(tuple(features), tuple(weights))
        for pair_id in held:
            difference = pairs[pair_id].difference(features)
            if difference is None:
                found.skipped += 1
                continue
            found.preferred[pair_id] = head.prefers(difference)
            found.margin[pair_id] = head.margin(difference)
            found.fold_of[pair_id] = fold
    if not per_fold:
        raise ContractError("no fold could be fitted")
    found.folds = len(per_fold)
    found.weights = tuple(
        sum(fold[index] for fold in per_fold) / len(per_fold) for index in range(len(features))
    )
    found.spread = tuple(
        math.sqrt(
            sum((fold[index] - found.weights[index]) ** 2 for fold in per_fold) / len(per_fold)
        )
        for index in range(len(features))
    )
    return found


def freeze(
    pairs: Mapping[str, FeaturedPair],
    labels: Mapping[str, Mapping[str, str]],
    features: Sequence[str],
    *,
    held_out_rater: str,
    strength: float = 1.0,
) -> Head:
    """Weights fitted without one rater, to be tested on that rater once.

    The rater is removed here, at the fit, so that the held-out test cannot be run
    against weights that saw its answers.
    """
    if held_out_rater not in labels:
        raise ContractError(
            f"no rater {held_out_rater!r}; known: {sorted(labels)}"
        )
    remaining = {name: sides for name, sides in labels.items() if name != held_out_rater}
    if not remaining:
        raise ContractError("holding out that rater leaves no one to fit on")
    found = out_of_fold(pairs, remaining, features, strength=strength)
    return found.head()


def score_frozen(
    head: Head, pairs: Mapping[str, FeaturedPair], labels: Mapping[str, str]
) -> dict[str, Any]:
    """The frozen head against one rater's choices. It fits nothing."""
    agreed = disagreed = 0
    for pair_id, side in labels.items():
        pair = pairs.get(pair_id)
        if side not in SIDES or pair is None:
            continue
        difference = pair.difference(head.features)
        if difference is None:
            continue
        if head.prefers(difference) == side:
            agreed += 1
        else:
            disagreed += 1
    total = agreed + disagreed
    from ..stats.intervals import wilson

    interval = wilson(agreed, total)
    return {
        "n": total,
        "agreed": agreed,
        "agreement": round(agreed / total, 4) if total else None,
        "interval": interval.as_dict(),
        "refitted": False,
    }


def features_from_scores(
    pair_rows: Sequence[Mapping[str, Any]],
    per_side: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> dict[str, FeaturedPair]:
    """Featured pairs from a table of per-side numbers.

    `per_side` is `{pair_id: {"a": {feature: value}, "b": {...}}}`, which is what the
    scorers write out when they are run on the two sides of a pair.
    """
    out = {}
    for row in pair_rows:
        pair_id = str(row["pair_id"])
        sides = per_side.get(pair_id)
        if not sides or "a" not in sides or "b" not in sides:
            continue
        lecture = str(row.get("lecture") or lecture_of(str(row.get("moment_id") or pair_id)))
        out[pair_id] = FeaturedPair(pair_id, lecture, dict(sides["a"]), dict(sides["b"]))
    return out
