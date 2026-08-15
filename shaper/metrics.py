"""Full-catalog ranking metrics and evaluation utilities (spec A.3, A.8).

- NDCG@k (primary), HR@k, MRR@k (secondary; Recall@10 == HR@10 with one
  relevant target and is omitted), and full-catalog target NLL (prespecified
  smooth secondary metric).
- Deterministic ties: break score ties by ascending item ID.
- Prefix filtering: every item occurring in the available prefix is excluded;
  if the ground-truth target also occurs earlier in the prefix it is RETAINED
  as a candidate and counted in the repeated-target rate.
- Per-user utilities are stored so the game decomposes linearly over users.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .backbone import SASRecWithProjection


@dataclass
class UserRankingResult:
    rank: int                 # 1-based rank of the target (n_items+1 if excluded)
    ndcg: float
    hr: float
    mrr: float
    nll: float
    repeated_target: bool


def rank_one_user(
    scores: torch.Tensor,  # [n_items] raw scores
    target: int,
    exclusion: Sequence[int],
    k: int,
    retain_repeated_target: bool = True,
    tie_break: str = "item_id_ascending",
) -> UserRankingResult:
    """Deterministic full-catalog ranking for one user.

    Ties on score are broken by ascending item ID. Excluded items are
    removed unless the target itself repeats earlier in the prefix.
    """
    n_items = scores.numel()
    if target in set(int(e) for e in exclusion):
        repeated = True
        if not retain_repeated_target:
            raise ValueError("repeated target excluded but retain_repeated_target=False")
        exclusion = [e for e in exclusion if e != target]
    else:
        repeated = False
    s = scores.clone()
    for e in exclusion:
        s[int(e) - 1] = -math.inf
    # deterministic ordering: descending score, ties -> ascending item id
    order = torch.argsort(-s, stable=True)
    rank = int((order == target - 1).nonzero()[0].item()) + 1
    ndcg = 1.0 / math.log2(rank + 1) if rank <= k else 0.0
    hr = 1.0 if rank <= k else 0.0
    mrr = 1.0 / rank
    # full-catalog target NLL
    logits = s - s.max()  # stable softmax; -inf excluded items -> p = 0
    logp = logits[target - 1] - torch.logsumexp(logits, dim=0)
    nll = float(-logp.item())
    return UserRankingResult(rank=rank, ndcg=ndcg, hr=hr, mrr=mrr, nll=nll, repeated_target=repeated)


@dataclass
class EvaluationResult:
    ndcg: float = 0.0
    hr: float = 0.0
    mrr: float = 0.0
    nll: float = 0.0
    repeated_target_rate: float = 0.0
    n_users: int = 0
    per_user_ndcg: List[float] = field(default_factory=list)
    per_user_nll: List[float] = field(default_factory=list)
    per_user_hr: List[float] = field(default_factory=list)
    per_user_ranks: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ndcg": self.ndcg,
            "hr": self.hr,
            "mrr": self.mrr,
            "nll": self.nll,
            "repeated_target_rate": self.repeated_target_rate,
            "n_users": self.n_users,
        }


def evaluate_model(
    model: SASRecWithProjection,
    inputs: Dict[str, Any],
    k: int,
    batch_size: int = 512,
    device: Optional[str] = None,
    tie_break: str = "item_id_ascending",
    retain_repeated_target: bool = True,
) -> EvaluationResult:
    """Evaluate a trained coalition model on a frozen evaluation set.

    `inputs` comes from FrozenData.eval_inputs / test_inputs:
        prefix [U, T], target [U], exclusion [list per user], users [U].
    """
    model.eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    prefix = inputs["prefix"].to(device)
    targets = inputs["target"].numpy()
    exclusions = inputs["exclusion"]
    res = EvaluationResult()
    backbone = getattr(model, "backbone", None)
    n_items = getattr(getattr(backbone, "cfg", None), "n_items", None)
    if n_items is None:
        n_items = getattr(getattr(model, "cfg", None), "n_items", None)
    if n_items is None:
        raise ValueError("cannot derive n_items from the model (backbone.cfg / cfg)")
    repeated = 0
    with torch.no_grad():
        for start in range(0, prefix.shape[0], batch_size):
            batch = prefix[start : start + batch_size]
            scores = model.rank_scores(batch).cpu()
            for i in range(scores.shape[0]):
                u = start + i
                r = rank_one_user(
                    scores[i],
                    int(targets[u]),
                    exclusions[u],
                    k,
                    retain_repeated_target=retain_repeated_target,
                    tie_break=tie_break,
                )
                res.per_user_ndcg.append(r.ndcg)
                res.per_user_hr.append(r.hr)
                res.per_user_nll.append(r.nll)
                res.per_user_ranks.append(r.rank)
                if r.repeated_target:
                    repeated += 1
    res.n_users = len(res.per_user_ndcg)
    res.ndcg = float(torch.tensor(res.per_user_ndcg).mean())
    res.hr = float(torch.tensor(res.per_user_hr).mean())
    res.mrr = float(torch.tensor([1.0 / r for r in res.per_user_ranks]).mean())
    res.nll = float(torch.tensor(res.per_user_nll).mean())
    res.repeated_target_rate = repeated / res.n_users if res.n_users else 0.0
    return res


def ndcg_at_k(ranks: Sequence[int], k: int) -> List[float]:
    """NDCG@k for a list of 1-based ranks (one relevant target each)."""
    out = []
    for rank in ranks:
        out.append(1.0 / math.log2(rank + 1) if rank <= k else 0.0)
    return out


__all__ = [
    "UserRankingResult",
    "EvaluationResult",
    "evaluate_model",
    "rank_one_user",
    "ndcg_at_k",
]
