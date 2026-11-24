"""Test unitari per il contratto del batch (src/contract.py)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

from src.config import PROJECT_ROOT, load_config
from src.contract import (
    FEATURE_NAMES,
    N_DEFAULT_FEATURES,
    N_FUTURE,
    N_PAST,
    N_QUANTILES,
    N_TARGETS,
    QUANTILES,
    TARGET_NAMES,
    TARGET_SHORT_NAMES,
    concat_history_future,
    make_dummy_batch,
    move_batch_to_device,
    validate_batch,
    validate_predictions,
)


class TestBatchContract(unittest.TestCase):
    def test_constants_aligned_with_config_and_data(self) -> None:
        cfg = load_config()
        self.assertEqual(tuple(cfg["task"]["targets"]), TARGET_SHORT_NAMES)
        self.assertEqual(tuple(cfg["task"]["quantiles"]), QUANTILES)
        self.assertEqual(cfg["task"]["n_past"], N_PAST)
        self.assertEqual(cfg["task"]["n_future"], N_FUTURE)
        self.assertEqual(len(FEATURE_NAMES), N_DEFAULT_FEATURES)

        train_dir = PROJECT_ROOT / cfg["data"]["root"] / cfg["data"]["splits"]["train"]
        if train_dir.exists():
            meta_files = [p for p in sorted(train_dir.glob("*.json")) if p.name != "scaler.json"]
            self.assertGreater(len(meta_files), 0)
            with open(meta_files[0], "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.assertEqual(tuple(meta["feature_names"]), FEATURE_NAMES)
            self.assertEqual(tuple(meta["target_names"]), TARGET_NAMES)

    def test_dummy_batch_valid_default_and_ablated(self) -> None:
        batch = make_dummy_batch(batch_size=8, seed=123)
        validate_batch(batch, expected_num_features=21)

        batch_no_wind = make_dummy_batch(batch_size=4, num_features=20, seed=456)
        validate_batch(batch_no_wind, expected_num_features=20)

    def test_validate_batch_rejects_missing_key(self) -> None:
        batch = dict(make_dummy_batch(batch_size=4))
        del batch["last_obs"]
        with self.assertRaises(ValueError):
            validate_batch(batch)

    def test_validate_batch_rejects_nan_values(self) -> None:
        batch = make_dummy_batch(batch_size=4)
        batch["history"][0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            validate_batch(batch)

    def test_validate_batch_rejects_inverted_padding_mask(self) -> None:
        batch = make_dummy_batch(batch_size=4, min_hist_len=11, max_hist_len=20)
        batch["history_padding_mask"] = ~batch["history_padding_mask"]
        with self.assertRaises(ValueError):
            validate_batch(batch)

    def test_validate_batch_rejects_unmasked_padding_features(self) -> None:
        batch = make_dummy_batch(batch_size=4, min_hist_len=11, max_hist_len=20)
        # Forza un campione ad avere padding reale
        if batch["history_padding_mask"].any():
            b_idx, t_idx = torch.nonzero(batch["history_padding_mask"], as_tuple=True)
            batch["history_mask"][b_idx[0], t_idx[0], 0] = False
            with self.assertRaises(ValueError):
                validate_batch(batch)

    def test_validate_batch_rejects_bad_target_positions(self) -> None:
        batch = make_dummy_batch(batch_size=4)
        batch["future_target_positions"][0] = torch.tensor([5, 5, 10], dtype=torch.int64)
        with self.assertRaises(ValueError):
            validate_batch(batch)

        batch2 = make_dummy_batch(batch_size=4)
        batch2["future_target_positions"][0, 2] = batch2["future_len"][0]  # out of bounds
        with self.assertRaises(ValueError):
            validate_batch(batch2)

    def test_validate_batch_rejects_wrong_delta_t(self) -> None:
        batch = make_dummy_batch(batch_size=4)
        batch["target_delta_t"][0, 0] += 1.0
        with self.assertRaises(ValueError):
            validate_batch(batch)

    def test_validate_predictions(self) -> None:
        B = 5
        q50 = torch.zeros((B, N_FUTURE, N_TARGETS), dtype=torch.float32)
        preds = torch.stack([q50 - 0.1, q50, q50 + 0.1], dim=-1)
        validate_predictions(preds, batch_size=B, require_monotonic_quantiles=True)

        # Shape errata
        with self.assertRaises(ValueError):
            validate_predictions(preds.reshape(B, -1), batch_size=B)

        # Quantile crossing
        preds_crossed = torch.stack([q50 + 0.2, q50, q50 + 0.1], dim=-1)
        validate_predictions(preds_crossed, batch_size=B, require_monotonic_quantiles=False)
        with self.assertRaises(ValueError):
            validate_predictions(preds_crossed, batch_size=B, require_monotonic_quantiles=True)

    def test_concat_history_future_contiguity(self) -> None:
        batch = make_dummy_batch(batch_size=6, min_hist_len=11, max_hist_len=25, min_fut_len=15, max_fut_len=35)
        cat = concat_history_future(batch)

        B = 6
        expected_len = batch["history_len"] + batch["future_len"]
        self.assertTrue(torch.equal(cat["seq_len"], expected_len))

        for b in range(B):
            lh = int(batch["history_len"][b].item())
            lf = int(batch["future_len"][b].item())
            lt = lh + lf

            # Nessun padding interno: validi esattamente 0..lt-1
            self.assertFalse(cat["padding_mask"][b, :lt].any().item())
            if lt < cat["seq"].shape[1]:
                self.assertTrue(cat["padding_mask"][b, lt:].all().item())
                self.assertTrue(cat["mask"][b, lt:].all().item())

            # Contiguità temporale: delta_t passa da -(lh-1) .. 0 .. +1 .. +lf
            dt_valid = cat["delta_t"][b, :lt]
            expected_dt = torch.arange(-(lh - 1), lf + 1, dtype=torch.float32)
            self.assertTrue(torch.allclose(dt_valid, expected_dt))

            # Flag is_future: False su history, True su future, False su pad
            self.assertFalse(cat["is_future"][b, :lh].any().item())
            self.assertTrue(cat["is_future"][b, lh:lt].all().item())
            if lt < cat["seq"].shape[1]:
                self.assertFalse(cat["is_future"][b, lt:].any().item())

            # target_positions punta esattamente ai 3 target futuri
            t_pos = cat["target_positions"][b]
            self.assertTrue(torch.allclose(cat["delta_t"][b, t_pos], batch["target_delta_t"][b]))

    def test_persistence_and_pinball_loss_broadcasting(self) -> None:
        batch = move_batch_to_device(make_dummy_batch(batch_size=4), "cpu")
        preds = batch["last_obs"][:, None, :, None].expand(-1, N_FUTURE, -1, N_QUANTILES)
        validate_predictions(preds, batch_size=4, require_monotonic_quantiles=True)

        quantiles = torch.tensor(QUANTILES, dtype=preds.dtype)
        errors = batch["target"].unsqueeze(-1) - preds
        pinball = torch.maximum(quantiles * errors, (quantiles - 1.0) * errors)
        valid = (~batch["target_mask"]).unsqueeze(-1).expand_as(pinball)
        loss = pinball[valid].mean()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreaterEqual(loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
