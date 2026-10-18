"""
SAM: Sharpness-Aware Minimization
==================================

SAM simultaneously minimizes loss value AND sharpness (sensitivity to perturbations).
It does a two-pass optimization each step:

  Pass 1 (ascent): Perturb weights in the direction of the gradient to find a
                   "worst-case neighbor" in weight space.
  Pass 2 (descent): Compute gradient at the perturbed point, then step the base
                    optimizer from the ORIGINAL weights using this new gradient.

This finds flat minima — parameters where the loss landscape is broad and smooth —
which generalize dramatically better than sharp minima found by vanilla SGD/Adam.

In our context, this is particularly relevant for small-data satellite imagery fine-tuning:
the LoRA adapter has only 6.7M trainable parameters on a complex visual-linguistic
task, making it prone to sharp overfitting. SAM mitigates this.

Reference:
  Foret et al. (2021), "Sharpness-Aware Minimization for Efficiently Improving Generalization"
  https://arxiv.org/abs/2010.01412

This is the second optimizer in our dual-optimization scheme:
  Optimizer 1: AdamW  — adaptive moment estimation for fast convergence
  Optimizer 2: SAM    — geometry-aware perturbation for flat-minima generalization
"""

import torch


class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization wrapped around any base optimizer.

    Usage in training loop:
        sam = SAM(model.get_trainable_params(), base_optimizer=torch.optim.AdamW, lr=lr, rho=0.05)

        # pass 1
        loss.backward()
        sam.first_step(zero_grad=True)

        # pass 2: recompute loss at perturbed weights
        output2 = model(...)
        loss2 = criterion(...)
        loss2.backward()
        sam.second_step(zero_grad=True)
    """

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"rho must be non-negative, got {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)

        # base optimizer operates on the same param groups
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """Perturb weights toward the gradient ascent direction (worst-case neighbor)."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                # adaptive SAM scales perturbation per-parameter by param norm
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # perturb: w = w + e_w
                self.state[p]["e_w"] = e_w  # store for reverting

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """Revert perturbation, then apply base optimizer step using the perturbed gradient."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # revert: w = w - e_w

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        # standard optimizer interface: not used in our training loop directly
        assert closure is not None, "SAM requires closure in step() — use first_step/second_step instead"
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        """Compute the global gradient norm across all parameter groups."""
        # pull all tensors to the same device (the device of the first grad)
        shared_device = self.param_groups[0]["params"][0].device
        norms = [
            ((torch.abs(p) if group.get("adaptive") else 1.0) * p.grad).norm(p=2).to(shared_device)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        if not norms:
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(norms), p=2)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
