import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from gs_meeting.data import Dataset, conditions, make_data, fingerprint, audit_data
from gs_meeting.scene import decode, camera, render, observed_energy
from gs_meeting.model import Core
from gs_meeting.methods import bridge, sample, rollout_for_anchor, objective
from gs_meeting.engine import train


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.cfg = json.loads(Path("configs/smoke.json").read_text())
        cls.cfg.update(data_root=str(Path(cls.tmp.name)/"data"),
                       output_root=str(Path(cls.tmp.name)/"runs"))
        make_data(cls.cfg)
    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def test_disjoint_and_conditions_no_target_leakage(self):
        ds = Dataset(self.cfg["data_root"], "test")
        d = ds[0]; before = conditions(d)
        altered = copy.deepcopy(d)
        for key in altered:
            if key.startswith("target_") or key in {"state", "z", "visible_fraction"}:
                altered[key] = torch.randn_like(altered[key])
        self.assertTrue(torch.equal(before, conditions(altered)))
        self.assertEqual(len(set(sum(ds.splits.values(), []))), sum(map(len, ds.splits.values())))
        with np.load(Path(self.cfg["data_root"])/"stats.npz") as stats:
            training = np.concatenate([np.load(Path(self.cfg["data_root"])/f"{i}.npz")["state"] for i in ds.splits["train"]])
            np.testing.assert_allclose(stats["mean"], training.mean(0), atol=1e-6)

    def test_decoder_render_gradients_and_ceiling(self):
        d = Dataset(self.cfg["data_root"], "train")[0]
        raw = d["state"].clone().requires_grad_(True)
        attrs = decode(raw)
        self.assertTrue((attrs["scale"]>0).all())
        r = render(raw, d["target_cameras"][0], d["target_lights"][0], self.cfg["size"])
        torch.testing.assert_close(r["rgb"], d["target_rgb"][0, 0], atol=1e-6, rtol=1e-6)
        r["rgb"].mean().backward()
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertGreater(float(raw.grad.abs().sum()), 0)

    def test_completion_masks_have_heldout_pixels(self):
        result = audit_data(self.cfg)
        for split in ["train", "val", "test"]:
            self.assertGreater(result["splits"][split]["mean_disoccluded_foreground_fraction"], .01)

    def test_bridge_endpoints_and_finite_velocity(self):
        x0, x1, xi = [torch.randn(2, 16, 13) for _ in range(3)]
        y, _, _ = bridge(x0, x1, torch.zeros(2), xi, .05)
        torch.testing.assert_close(y, x0)
        y, velocity, _ = bridge(x0, x1, torch.ones(2), xi, .05)
        torch.testing.assert_close(y, x1)
        self.assertTrue(torch.isfinite(velocity).all())

    def test_every_objective_and_learned_anchor_gradients(self):
        ds = Dataset(self.cfg["data_root"], "train"); d = ds[0]
        model = Core(self.cfg); c = conditions(d)[None]
        coarse = torch.zeros_like(d["z"])[None]
        for variant in ["A0", "A1", "A2", "B0", "B1", "B2", "B3", "B0_anchor", "B3_anchor"]:
            model.zero_grad(set_to_none=True)
            pieces, endpoint = objective(model, variant, d["z"][None], c, coarse, self.cfg)
            loss = sum(pieces.values()); loss.backward()
            self.assertTrue(torch.isfinite(loss))
            for p in model.parameters():
                if p.grad is not None: self.assertTrue(torch.isfinite(p.grad).all())
        for variant in ["B2", "B0_anchor", "B3_anchor"]:
            model.zero_grad(set_to_none=True)
            y = rollout_for_anchor(model, variant, coarse, c, self.cfg, .6)
            y.square().mean().backward()
            self.assertGreater(float(model.main.weight.grad.abs().sum()), 0)
        # Contrast: the original fixed paired bridge has no parameter dependency.
        fixed, _, _ = bridge(coarse, d["z"][None], torch.tensor([.5]), torch.randn_like(coarse), .05)
        self.assertFalse(fixed.requires_grad)
        # The actual renderer penalty (not merely latent magnitude) trains the core.
        model.zero_grad(set_to_none=True)
        true_coarse = d["z"][None]
        y = rollout_for_anchor(model, "B2", true_coarse, c, self.cfg, .8)
        e = observed_energy(ds.raw(y[0]), d["input_rgb"], d["input_cameras"], d["input_lights"], self.cfg["size"])
        e.backward()
        self.assertGreater(float(model.main.weight.grad.abs().sum()), 0)

    def test_corrected_sde_preserves_known_gaussian_variance(self):
        class ExactGaussian(torch.nn.Module):
            def forward(self, x, t, context):
                q = t[:, None, None]; sigma = .4
                g = sigma*torch.sin(math.pi*q); dg = sigma*math.pi*torch.cos(math.pi*q)
                r = (1-q).square()+q.square()+g.square()
                rp = -2*(1-q)+2*q+2*g*dg
                return .5*rp/r*x, g/r*x
        cfg = self.cfg|dict(sigma=.4, source_jitter=1., sde_strength=5.)
        torch.manual_seed(19)
        x, nfe = sample(ExactGaussian(), "B1", torch.zeros(4000, 1, 1), (4000, 1, 13),
                        cfg, 256, torch.zeros(4000, 1, 13))
        self.assertEqual(nfe, 256)
        self.assertLess(abs(float(x.var())-1), .06)

    def test_resume_matches_uninterrupted_updates(self):
        full = self.cfg|dict(steps=2, rgb_weight=0., output_root=str(Path(self.tmp.name)/"full"))
        paused = full|dict(steps=1, output_root=str(Path(self.tmp.name)/"paused"))
        a = train(full, "A0", 13, "cpu")
        b = train(paused, "A0", 13, "cpu")
        train(paused|dict(steps=2), "A0", 13, "cpu", resume=True)
        ca = torch.load(a, weights_only=False); cb = torch.load(b, weights_only=False)
        for key in ca["model"]: self.assertTrue(torch.equal(ca["model"][key], cb["model"][key]), key)


if __name__=="__main__": unittest.main()
