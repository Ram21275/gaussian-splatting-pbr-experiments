# Implemented mathematics and experiment boundaries

## Common state

An object has N persistent anchor identities and 13 unconstrained channels per anchor. Channels decode to position (3), isotropic radius (1), approximate normal (3), albedo (3), roughness (1), metallic (1), opacity (1). All transport operates on channel-standardized values using training-object statistics only. Decoder, anchors and normalizer are identical across variants. The decoder is analytic rather than learned and cannot represent arbitrary assets outside its bounded family.

A small shared token transformer receives the noisy joint state, time and a three-view camera-aware RGB descriptor. It outputs a main vector and a Gaussian-noise vector. No ground-truth target features enter `conditions()` or `coarse_prediction()`.

## A: diffusion versus rectified flow

A1 trains a VP noise predictor with x_t = sqrt(alpha_bar(t)) x_1 + sqrt(1-alpha_bar(t)) epsilon and cosine alpha_bar. Deterministic DDIM samples from t=0.999 toward 0. The terminal Gaussian is an approximation at finite terminal time. B3 reuses A1 weights with a corrupted coarse initialization at t=0.5; it is a warm-start DDIM control inspired by SDEdit, not a complete SDEdit reproduction.

A2 constructs x_t=(1-t)x_0+t x_1, x_0~N(0,I), and regresses velocity x_1-x_0. Euler sampling uses exactly one shared-trunk call per requested NFE. Auxiliary RGB loss uses the surrogate clean endpoint x_t+(1-t)v_theta. That expression is exact for an individual linear bridge velocity and approximate for the learned conditional-mean field. It is not an unbiased clean endpoint estimator in every setting.

## B: stochastic interpolant

The frozen A0 model constructs a coarse estimate from input RGB/cameras/light only. The source X0 adds isotropic Gaussian jitter of standard deviation `source_jitter` in standardized latent coordinates. That same source distribution is used by B0/B1/B2 and anchor controls.

For B1/B2, gamma(t)=sigma sin(pi t), with:

```
X_t = (1-t)X0 + t X1 + gamma(t) xi
velocity target = X1-X0 + gamma'(t) xi
noise target = xi
score estimate = -noise_theta(X_t,t,c)/gamma(t)
```

The noise-head MSE is a gamma-squared weighted score-regression objective. Its population conditional optimum is the same score at interior times, but finite-model optimization differs from unweighted score regression. Training excludes t within `time_eps` of either endpoint. B0 uses gamma=0 and no score loss.

The SI sampler uses a(t)=`sde_strength` gamma(t)^2:

```
dX = [b_theta + a(t) score_theta] dt + sqrt(2 a(t)) dW
```

The implemented correction is `-sde_strength * gamma * predicted_noise`, avoiding direct division by gamma near endpoints. Scalar coefficients are evaluated at each time interval's midpoint, with an Euler state update. This is a finite-step numerical approximation, not exact sampling. With exact fields, the added Fokker–Planck terms cancel because rho*s=grad(rho); arbitrary extra noise without this correction changes the marginal law. The included analytic Gaussian test detects that sign/scaling error. It does not validate reconstruction performance.

## Correction: the anchor must depend on the learned model

The previous experiment prompt wrote an observed-view loss on paired X_t sampled from a fixed bridge. Once X0, X1 and gamma are fixed, that X_t has no model-parameter dependency. Its renderer loss cannot train velocity/score parameters. Do not implement that expression and assume it anchors the trained dynamics.

This starter instead integrates the learned velocity for `anchor_steps` differentiable Euler steps from the input-only source to a randomly selected time u in [0.1,1]. Let the resulting proposal be Y_theta(u). It uses:

```
L_anchor = max(0, E_obs(D(Y_theta(u))) - E_obs(D(coarse)) - tolerance)
E_obs = mean input-view linear-RGB MSE
```

This penalty has a real gradient into the model. It is a proposed short-rollout regularizer, not an exact implementation of the earlier fixed-X_t expression and not a theorem that all states remain evidence-consistent. It acts on ODE rollouts for B0/B1/B2; B2 inference remains stochastic. Training-to-SDE path transfer must be checked empirically. A noisy SDE path loss could be a later, explicitly separate ablation.

B1 and B2 execute the same rollout/render graph and differ only in its loss weight, preserving their renderer-work budget. B0 also executes the zero-weight graph. B0_anchor tests the same active penalty on deterministic coarse flow. B3_anchor trains noise prediction plus an observed-view penalty on a short differentiable coarse-start DDIM rollout. These latter trajectories differ, so their equal renderer count does not imply equal total arithmetic or identical regularization strength. Report actual time, gradients and quality. The exact-marginal SI statement no longer guarantees the learned distribution after auxiliary regularization.

## Hypothesis evidence

First show A2 works on truly held-out identities and compare with A0/A1 using a common decoder/compute accounting. Next compare B2 against B0, B1, B3, B0_anchor and B3_anchor at matching NFE/K and visibility masks. A positive B2-vs-B0 result alone cannot identify a stochastic-interpolant contribution. Evaluate three training seeds for the shortlisted contrast. Bootstrap object differences within a seed; show seed spread separately.

Procedural images are generated and evaluated with the same approximate renderer. These results cannot establish DreamLifting parity, real-asset PBR reconstruction or publication novelty. The package intentionally reports those boundaries alongside any measurements.

## Primary references

- Stochastic interpolants: https://arxiv.org/abs/2303.08797
- Data-dependent couplings: https://arxiv.org/abs/2310.03725
- SDEdit: https://arxiv.org/abs/2108.01073
- DreamLifting: https://arxiv.org/abs/2509.07435 and https://github.com/ZX-Yin/DreamLifting
- DiffSplat: https://arxiv.org/abs/2501.16764
- SplatFlow, multiview rectified flow: https://arxiv.org/abs/2411.16443
- PyTorch reproducibility: https://docs.pytorch.org/docs/stable/notes/randomness.html
