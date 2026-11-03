"""Exact objective conventions and explicitly approximate numerical samplers."""
import math
import torch

VARIANTS = ["A0", "A1", "A2", "B0", "B1", "B2", "B3", "B0_anchor", "B3_anchor"]
DIFFUSION = {"A1", "B3", "B3_anchor"}
INTERPOLANTS = {"B1", "B2"}
ANCHORED = {"B2", "B0_anchor", "B3_anchor"}


def alpha(t):
    s = .008
    return (torch.cos((t+s)/(1+s)*math.pi/2)/math.cos(s/(1+s)*math.pi/2)).square()


def bridge(x0, x1, t, noise, sigma):
    q = t[:, None, None]
    g = sigma*torch.sin(math.pi*q)
    dg = sigma*math.pi*torch.cos(math.pi*q)
    return (1-q)*x0+q*x1+g*noise, x1-x0+dg*noise, g


def source(coarse, cfg):
    # The same small source jitter distribution is used by every B variant.
    return coarse+cfg["source_jitter"]*torch.randn_like(coarse)


def objective(model, variant, target, context, coarse, cfg):
    """Return loss pieces and a clean-endpoint SURROGATE for auxiliary rendering."""
    b = target.shape[0]; device = target.device
    t = cfg["time_eps"]+(1-2*cfg["time_eps"])*torch.rand(b, device=device)
    noise = torch.randn_like(target)
    if variant == "A0":
        pred, _ = model(torch.zeros_like(target), torch.zeros_like(t), context)
        return {"main": (pred-target).square().mean(), "score": target.new_zeros(())}, pred
    if variant in DIFFUSION:
        # B3 has the same forward training objective as A1; only its sampler is warm-started.
        ab = alpha(t)[:, None, None]
        xt = ab.sqrt()*target+(1-ab).sqrt()*noise
        pred, _ = model(xt, t, context)
        clean = (xt-(1-ab).sqrt()*pred)/ab.sqrt().clamp_min(1e-3)
        return {"main": (pred-noise).square().mean(), "score": target.new_zeros(())}, clean
    x0 = torch.randn_like(target) if variant == "A2" else source(coarse, cfg)
    sigma = cfg["sigma"] if variant in INTERPOLANTS else 0.
    xt, velocity, g = bridge(x0, target, t, noise, sigma)
    pred, predicted_noise = model(xt, t, context)
    score_loss = ((predicted_noise-noise).square().mean()
                  if variant in INTERPOLANTS else target.new_zeros(()))
    # In the SI case v is conditional-mean velocity, so this is NOT an exact clean state.
    clean = xt+(1-t[:, None, None])*pred
    return {"main": (pred-velocity).square().mean(), "score": score_loss}, clean


def rollout_for_anchor(model, variant, coarse, context, cfg, end_time):
    """Differentiable proposal: anchor actual learned predictions, not fixed Xt.

    B2/B0: short ODE transport. B3: short deterministic DDIM trajectory from a
    coarse-corrupted source. The penalties share data and renderer count, but
    paths differ; report this limitation instead of claiming an exact theorem.
    """
    x = source(coarse, cfg)
    steps = cfg["anchor_steps"]
    if variant in DIFFUSION:
        start = x.new_full((x.shape[0],), cfg["coarse_noise_time"])
        ab = alpha(start)[:, None, None]
        x = ab.sqrt()*x+(1-ab).sqrt()*torch.randn_like(x)
        # Random fraction of the reverse trajectory, including near-clean states.
        times = torch.linspace(0., 1., steps+1, device=x.device)
        for j in range(steps):
            t = start*(1-end_time*times[j]); nxt = start*(1-end_time*times[j+1])
            pred, _ = model(x, t, context)
            a = alpha(t)[:, None, None]; an = alpha(nxt)[:, None, None]
            clean = (x-(1-a).sqrt()*pred)/a.sqrt().clamp_min(1e-3)
            x = an.sqrt()*clean+(1-an).sqrt()*pred
        return x
    dt = end_time/steps
    for j in range(steps):
        t = x.new_full((x.shape[0],), j*dt)
        v, _ = model(x, t, context)
        x = x+dt*v
    return x


@torch.no_grad()
def sample(model, variant, context, shape, cfg, nfe, coarse=None):
    if nfe < 1: raise ValueError("nfe must be positive")
    device = context.device
    if variant == "A0":
        return model(torch.zeros(shape, device=device), torch.zeros(shape[0], device=device), context)[0], 1
    if variant.startswith("B") and coarse is None: raise ValueError("B requires input-only coarse source")
    if variant in DIFFUSION:
        start = .999 if variant == "A1" else cfg["coarse_noise_time"]
        x = torch.randn(shape, device=device)
        if variant != "A1":
            c = source(coarse, cfg)
            a = alpha(x.new_full((shape[0],), start))[:, None, None]
            x = a.sqrt()*c+(1-a).sqrt()*x
        grid = torch.linspace(start, 0., nfe+1, device=device)
        for j in range(nfe):
            t = grid[j].expand(shape[0]); nxt = grid[j+1].expand(shape[0])
            eps, _ = model(x, t, context)
            a = alpha(t)[:, None, None]; an = alpha(nxt)[:, None, None]
            clean = (x-(1-a).sqrt()*eps)/a.sqrt().clamp_min(1e-3)
            x = an.sqrt()*clean+(1-an).sqrt()*eps
        return x, nfe
    x = torch.randn(shape, device=device) if variant == "A2" else source(coarse, cfg)
    dt = 1/nfe
    for j in range(nfe):
        # Midpoint coefficient avoids a zero gamma division. Euler uses this field
        # approximation at the current state; endpoint discretization remains approximate.
        u = (j+.5)*dt
        t = x.new_full((shape[0],), u)
        v, noise_pred = model(x, t, context)
        if variant in INTERPOLANTS and cfg["sde_strength"]>0:
            g = cfg["sigma"]*math.sin(math.pi*u)
            a = cfg["sde_strength"]*g*g
            # s=-noise_pred/g and a*s=-strength*g*noise_pred, no singular division.
            x = x+dt*(v-cfg["sde_strength"]*g*noise_pred)+math.sqrt(2*a*dt)*torch.randn_like(x)
        else:
            x = x+dt*v
    return x, nfe
