import torch
import pytrec_eval

from typing import List, Dict, Tuple



@torch.no_grad()
def accuracy(output: torch.tensor, target: torch.tensor, topk=(1,)) -> List[float]:
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res


@torch.no_grad()
def batch_mrr(output: torch.tensor, target: torch.tensor) -> float:
    assert len(output.shape) == 2
    assert len(target.shape) == 1
    sorted_score, sorted_indices = torch.sort(output, dim=-1, descending=True)
    _, rank = torch.nonzero(sorted_indices.eq(target.unsqueeze(-1)).long(), as_tuple=True)
    assert rank.shape[0] == output.shape[0]

    rank = rank + 1
    mrr = torch.sum(100 / rank.float()) / rank.shape[0]
    return mrr.item()
