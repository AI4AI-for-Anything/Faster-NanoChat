"""XLA-compatible replacements for nanochat's fused optimizer steps.

Upstream keeps hyperparameters in 0-D CPU tensors so that torch.compile does not
recompile when their values change. XLA breaks on that in two ways:

  * a CPU tensor cannot be mixed into device math
    (RuntimeError: Expected XLA tensor. Got: torch.FloatTensor), and
  * Tensor.add_(..., alpha=<tensor>) needs a Python Number; an XLA tensor there
    forces a host sync, which torch_xla.compile forbids
    (RuntimeError: Unexpected execution happens inside the compiled function).

These replacements take plain Python numbers for every hyperparameter and keep
the arithmetic identical. They are applied by assigning over the module
attributes, so the upstream checkout is never modified.
"""
import torch

from nanochat.common import COMPUTE_DTYPE

# Constants for Polar Express, copied from nanochat.optim (num_iters=5).
POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.7724841532176850, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def adamw_step_xla(p, grad, exp_avg, exp_avg_sq, step, lr, beta1, beta2, eps, wd):
    """Fused AdamW step with Python-number hyperparameters."""
    # Callers may hand over _Scalar shims or plain numbers; normalise first.
    step, lr, beta1, beta2, eps, wd = (float(x) for x in (step, lr, beta1, beta2, eps, wd))
    p32 = p.float()
    exp_avg32 = exp_avg.float()
    exp_avg_sq32 = exp_avg_sq.float()
    grad32 = grad.float()

    p32.mul_(1 - lr * wd)
    exp_avg32.lerp_(grad32, 1 - beta1)
    exp_avg_sq32.lerp_(grad32.square(), 1 - beta2)

    bias1 = 1 - beta1 ** step
    bias2 = 1 - beta2 ** step
    denom = (exp_avg_sq32 / bias2).sqrt() + eps
    step_size = lr / bias1

    p32.add_(exp_avg32 / denom, alpha=-step_size)

    p.copy_(p32)
    exp_avg.copy_(exp_avg32)
    exp_avg_sq.copy_(exp_avg_sq32)


def muon_step_xla(stacked_grads, stacked_params, momentum_buffer,
                  second_momentum_buffer, momentum, lr, wd, beta2,
                  ns_steps, red_dim):
    """Fused Muon step with Python-number hyperparameters."""
    momentum, lr, wd, beta2 = (float(x) for x in (momentum, lr, wd, beta2))
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    X = g.bfloat16() if COMPUTE_DTYPE == torch.bfloat16 else g

    target = X.float().norm(dim=(-2, -1), keepdim=True) / (X.size(-2) ** 0.5)
    row_norm = X.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
    X = X * (target / row_norm).to(X.dtype)

    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X

    g = X.to(stacked_params.dtype)

    target_norm = min(g.size(-2), g.size(-1)) ** 0.5
    current_norm = g.float().norm(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    g = g * (target_norm / current_norm).to(g.dtype)

    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)

    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class _Scalar:
    """Stand-in for the optimizer's 0-D hyperparameter tensors.

    MuonAdamW's call sites do `self._adamw_lr_t.fill_(value)` before each step and
    then hand the object to the fused function. On XLA the object must not be a
    device tensor (mixing it into device math fails, and an XLA tensor as
    `add_(alpha=...)` forces a host sync). This holds a plain Python float,
    accepts the same fill_() call, and converts with float().
    """

    __slots__ = ("value",)

    def __init__(self, value=0.0):
        self.value = float(value)

    def fill_(self, value):
        self.value = float(value)
        return self

    def __float__(self):
        return self.value

    def __repr__(self):
        return f"_Scalar({self.value})"


def patch_optimizer_for_xla(optimizer):
    """Make a MuonAdamW instance take Python-number hyperparameters on XLA.

    Two things change: the fused step functions are swapped for the XLA-safe
    versions above, and the optimizer's 0-D hyperparameter tensors are replaced
    by _Scalar shims. The call sites inside MuonAdamW keep working unchanged.
    """
    from nanochat import optim as O

    O.adamw_step_fused = adamw_step_xla
    O.muon_step_fused = muon_step_xla

    for name in ("_adamw_step_t", "_adamw_lr_t", "_adamw_beta1_t", "_adamw_beta2_t",
                 "_adamw_eps_t", "_adamw_wd_t", "_muon_momentum_t", "_muon_lr_t",
                 "_muon_wd_t", "_muon_beta2_t"):
        if hasattr(optimizer, name):
            old = getattr(optimizer, name)
            value = float(old.detach().cpu()) if torch.is_tensor(old) else float(old)
            setattr(optimizer, name, _Scalar(value))
    return optimizer
