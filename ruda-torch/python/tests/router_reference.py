"""Independent PyTorch reference, TESTS ONLY. Never imported by production."""
import torch


def weights_reference(logits, indices, softmax, renormalize, scale, *, promote=True):
    x = logits.float() if promote else logits
    p = x.softmax(-1) if softmax else x.sigmoid()
    selected = p.gather(1, indices.long())
    if renormalize:
        selected = selected / selected.sum(-1, keepdim=True)
    return selected * scale


def analytical_vjp(logits, indices, grad, softmax, renormalize, scale, *, promote=True):
    """Algebraic VJP. Dense seed is a REFERENCE convenience, not kernel storage."""
    x = logits.float() if promote else logits
    p = x.softmax(-1) if softmax else x.sigmoid()
    grad = grad.to(p.dtype)
    picked = p.gather(1, indices.long())
    s = picked.sum(-1, keepdim=True)
    dot = (picked * grad).sum(-1, keepdim=True)
    seed = torch.zeros_like(p).scatter_add(1, indices.long(), grad)
    counts = torch.zeros_like(p).scatter_add(1, indices.long(), torch.ones_like(grad))
    if renormalize:
        dp = scale * (seed - counts * dot / s) / s
        result = p * dp if softmax else p * (1-p) * dp
    elif softmax:
        result = scale * p * (seed - dot)
    else:
        result = scale * p * (1-p) * seed
    return result.to(logits.dtype)
