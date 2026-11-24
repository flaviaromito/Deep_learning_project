"""Contratto formale del batch e delle predizioni del modello.

Questo modulo centralizza:
- le costanti canoniche del task (nomi feature, target, quantili, dimensioni);
- il tipo `Batch` (TypedDict) condiviso tra preprocessing, modelli, training e eval;
- i validatori runtime `validate_batch` e `validate_predictions`;
- l'utility `move_batch_to_device` per trasferire un batch su CPU/GPU/MPS;
- il generatore `make_dummy_batch` per testare modelli/loss/metriche prima o senza
  caricare i file .npz da disco.

Specifica completa in `docs/batch_contract.md`.
"""

from __future__ import annotations

from typing import Any, Mapping, TypedDict

import torch

# -----------------------------------------------------------------------------
# Costanti canoniche (allineate a DATA/*.json e config.yaml)
# -----------------------------------------------------------------------------

METEO_FEATURES: tuple[str, ...] = (
    "wind",
    "humidity",
    "solar_irradiance",
    "rainfall",
    "pressure",
    "avg_temperature",
    "min_temperature",
    "max_temperature",
)

CLOUD_FEATURES: tuple[str, ...] = (
    "thick_clouds",
    "thin_clouds",
    "clouds_shadows",
)

METEO_CLOUD_FEATURES: tuple[str, ...] = METEO_FEATURES + CLOUD_FEATURES

TIME_FEATURES: tuple[str, ...] = (
    "day_of_year",
    "day_of_year_fourier_sin_1",
    "day_of_year_fourier_cos_1",
    "day_of_year_fourier_sin_2",
    "day_of_year_fourier_cos_2",
    "day_of_year_fourier_sin_3",
    "day_of_year_fourier_cos_3",
)

TARGET_HISTORY_FEATURES: tuple[str, ...] = (
    "target_history_avg_NDVI_clear_sky",
    "target_history_avg_GNDVI_clear_sky",
    "target_history_avg_EVI_clear_sky",
)

FEATURE_NAMES: tuple[str, ...] = (
    METEO_FEATURES + CLOUD_FEATURES + TIME_FEATURES + TARGET_HISTORY_FEATURES
)

TARGET_NAMES: tuple[str, ...] = (
    "avg_NDVI_clear_sky",
    "avg_GNDVI_clear_sky",
    "avg_EVI_clear_sky",
)

TARGET_SHORT_NAMES: tuple[str, ...] = ("NDVI", "GNDVI", "EVI")

QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)

N_DEFAULT_FEATURES: int = len(FEATURE_NAMES)  # 21
N_PAST: int = 3
N_FUTURE: int = 3
N_TARGETS: int = len(TARGET_NAMES)  # 3
N_QUANTILES: int = len(QUANTILES)  # 3

# Convenzione immutabile: True = mancante/padding, False = valido
MASK_TRUE_IS_MISSING: bool = True

TENSOR_BATCH_KEYS: tuple[str, ...] = (
    "history",
    "history_mask",
    "history_padding_mask",
    "history_len",
    "history_delta_t",
    "future",
    "future_mask",
    "future_padding_mask",
    "future_len",
    "future_delta_t",
    "future_target_positions",
    "target_delta_t",
    "last_obs",
    "target",
    "target_raw",
    "target_mask",
    "sample_idx",
)

METADATA_BATCH_KEYS: tuple[str, ...] = (
    "area",
    "climate",
)

REQUIRED_BATCH_KEYS: tuple[str, ...] = TENSOR_BATCH_KEYS + METADATA_BATCH_KEYS


class Batch(TypedDict):
    """Dizionario restituito dal DataLoader ad ogni iterazione."""

    # Sequenza storica (right-padded a T_h)
    history: torch.Tensor                  # (B, T_h, F) float32
    history_mask: torch.Tensor             # (B, T_h, F) bool (True = mancante/pad)
    history_padding_mask: torch.Tensor     # (B, T_h)    bool (True = pad)
    history_len: torch.Tensor              # (B,)        int64
    history_delta_t: torch.Tensor          # (B, T_h)    float32 (giorni <= 0)

    # Sequenza futura (right-padded a T_f)
    future: torch.Tensor                   # (B, T_f, F) float32
    future_mask: torch.Tensor              # (B, T_f, F) bool (True = mancante/pad)
    future_padding_mask: torch.Tensor      # (B, T_f)    bool (True = pad)
    future_len: torch.Tensor               # (B,)        int64
    future_delta_t: torch.Tensor           # (B, T_f)    float32 (giorni >= 1)

    # Posizioni e anchor temporali/residui
    future_target_positions: torch.Tensor  # (B, 3)      int64 (0-based in future)
    target_delta_t: torch.Tensor           # (B, 3)      float32 (giorni > 0)
    last_obs: torch.Tensor                 # (B, 3)      float32 (stessa scala di target)

    # Target
    target: torch.Tensor                   # (B, 3, 3)   float32 (B, H, C)
    target_raw: torch.Tensor               # (B, 3, 3)   float32 (B, H, C) scala originale
    target_mask: torch.Tensor              # (B, 3, 3)   bool (True = mancante)

    # Metadati (SOLO diagnostica, mai input del modello)
    area: list[str]                        # len B
    climate: list[str]                     # len B (100% "unknown" su ood_s)
    sample_idx: torch.Tensor               # (B,)        int64


def move_batch_to_device(
    batch: Mapping[str, Any],
    device: torch.device | str,
    *,
    non_blocking: bool = False,
) -> dict[str, Any]:
    """Sposta tutti i tensori del batch sul device indicato lasciando invariati i metadati."""
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device=device, non_blocking=non_blocking)
        else:
            moved[key] = value
    return moved


def validate_batch(
    batch: Mapping[str, Any],
    *,
    expected_num_features: int | None = None,
    check_delta_t_values: bool = True,
) -> None:
    """Verifica tutte le invarianti del contratto sul dizionario `batch`.

    Solleva `ValueError` o `TypeError` alla prima violazione rilevata.
    """
    missing = [k for k in REQUIRED_BATCH_KEYS if k not in batch]
    if missing:
        raise ValueError(f"Chiavi mancanti nel batch: {missing}")

    # 1. Controllo tipi base
    for key in TENSOR_BATCH_KEYS:
        if not isinstance(batch[key], torch.Tensor):
            raise TypeError(f"batch['{key}'] deve essere un torch.Tensor, trovato {type(batch[key])}")

    # 2. Controllo dimensioni principali
    history = batch["history"]
    future = batch["future"]
    if history.ndim != 3:
        raise ValueError(f"batch['history'] deve avere 3 dim (B, T_h, F), trovato {tuple(history.shape)}")
    if future.ndim != 3:
        raise ValueError(f"batch['future'] deve avere 3 dim (B, T_f, F), trovato {tuple(future.shape)}")

    B, T_h, F = history.shape
    B_f, T_f, F_f = future.shape
    if B == 0:
        raise ValueError("Batch vuoto (B == 0) non ammesso.")
    if B_f != B:
        raise ValueError(f"Batch size incoerente tra history ({B}) e future ({B_f}).")
    if F_f != F:
        raise ValueError(f"Numero feature incoerente tra history ({F}) e future ({F_f}).")
    if expected_num_features is not None and F != expected_num_features:
        raise ValueError(f"Attese {expected_num_features} feature, trovate {F}.")

    # 3. Controllo shape e dtype esatti per ogni tensore
    expected_specs: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
        "history": ((B, T_h, F), torch.float32),
        "history_mask": ((B, T_h, F), torch.bool),
        "history_padding_mask": ((B, T_h), torch.bool),
        "history_len": ((B,), torch.int64),
        "history_delta_t": ((B, T_h), torch.float32),
        "future": ((B, T_f, F), torch.float32),
        "future_mask": ((B, T_f, F), torch.bool),
        "future_padding_mask": ((B, T_f), torch.bool),
        "future_len": ((B,), torch.int64),
        "future_delta_t": ((B, T_f), torch.float32),
        "future_target_positions": ((B, N_FUTURE), torch.int64),
        "target_delta_t": ((B, N_FUTURE), torch.float32),
        "last_obs": ((B, N_TARGETS), torch.float32),
        "target": ((B, N_FUTURE, N_TARGETS), torch.float32),
        "target_raw": ((B, N_FUTURE, N_TARGETS), torch.float32),
        "target_mask": ((B, N_FUTURE, N_TARGETS), torch.bool),
        "sample_idx": ((B,), torch.int64),
    }

    for key, (exp_shape, exp_dtype) in expected_specs.items():
        t = batch[key]
        if tuple(t.shape) != exp_shape:
            raise ValueError(f"batch['{key}'] shape {tuple(t.shape)} != attesa {exp_shape}")
        if t.dtype != exp_dtype:
            raise TypeError(f"batch['{key}'] dtype {t.dtype} != atteso {exp_dtype}")

    # 4. Metadati testuali
    for key in METADATA_BATCH_KEYS:
        val = batch[key]
        if not isinstance(val, (list, tuple)) or len(val) != B:
            raise ValueError(f"batch['{key}'] deve essere una sequenza di lunghezza {B}")
        if not all(isinstance(x, str) for x in val):
            raise TypeError(f"Tutti gli elementi di batch['{key}'] devono essere str")

    # 5. Assenza di NaN / Inf in tutti i tensori float (i NaN devono essere stati sostituiti)
    float_keys = (
        "history",
        "history_delta_t",
        "future",
        "future_delta_t",
        "target_delta_t",
        "last_obs",
        "target",
        "target_raw",
    )
    for key in float_keys:
        if not torch.isfinite(batch[key]).all():
            raise ValueError(f"batch['{key}'] contiene valori NaN o Inf.")

    # 6. Coerenza lunghezze <-> padding_mask <-> feature_mask (convenzione True = mancante/pad)
    h_len = batch["history_len"]
    f_len = batch["future_len"]
    if (h_len < 1).any() or (h_len > T_h).any():
        raise ValueError(f"history_len fuori range [1, {T_h}]: {h_len.tolist()}")
    if (f_len < N_FUTURE).any() or (f_len > T_f).any():
        raise ValueError(f"future_len fuori range [{N_FUTURE}, {T_f}]: {f_len.tolist()}")

    h_steps = torch.arange(T_h, device=h_len.device).unsqueeze(0)  # (1, T_h)
    expected_h_pad = h_steps >= h_len.unsqueeze(1)
    if not torch.equal(batch["history_padding_mask"], expected_h_pad):
        raise ValueError("history_padding_mask incoerente con history_len (atteso True per t >= len).")

    f_steps = torch.arange(T_f, device=f_len.device).unsqueeze(0)  # (1, T_f)
    expected_f_pad = f_steps >= f_len.unsqueeze(1)
    if not torch.equal(batch["future_padding_mask"], expected_f_pad):
        raise ValueError("future_padding_mask incoerente con future_len (atteso True per t >= len).")

    # Sui timestep di padding tutte le feature devono risultare mascherate (True)
    h_pad_mask = batch["history_padding_mask"]
    if h_pad_mask.any() and not batch["history_mask"][h_pad_mask].all():
        raise ValueError("Trovati elementi False in history_mask su posizioni di padding.")
    f_pad_mask = batch["future_padding_mask"]
    if f_pad_mask.any() and not batch["future_mask"][f_pad_mask].all():
        raise ValueError("Trovati elementi False in future_mask su posizioni di padding.")

    # 7. Coerenza future_target_positions: 0 <= pos_0 < pos_1 < pos_2 < future_len
    pos = batch["future_target_positions"]
    if (pos < 0).any() or (pos >= f_len.unsqueeze(1)).any():
        raise ValueError("future_target_positions contiene indici fuori da [0, future_len - 1].")
    if (pos[:, 1:] <= pos[:, :-1]).any():
        raise ValueError("future_target_positions deve essere strettamente crescente.")

    # 8. Coerenza valori delta_t (se abilitati e non azzerati da add_delta_t=False)
    if check_delta_t_values:
        h_dt = batch["history_delta_t"]
        f_dt = batch["future_delta_t"]
        t_dt = batch["target_delta_t"]

        expected_h_dt = torch.where(
            expected_h_pad,
            torch.zeros(1, dtype=torch.float32, device=h_dt.device),
            (h_steps - (h_len.unsqueeze(1) - 1)).to(torch.float32),
        )
        expected_f_dt = torch.where(
            expected_f_pad,
            torch.zeros(1, dtype=torch.float32, device=f_dt.device),
            (f_steps + 1).to(torch.float32),
        )
        expected_t_dt = (pos + 1).to(torch.float32)

        if not torch.allclose(h_dt, expected_h_dt):
            raise ValueError("history_delta_t non corrisponde a [-(L_h-1)..0] sui passi validi.")
        if not torch.allclose(f_dt, expected_f_dt):
            raise ValueError("future_delta_t non corrisponde a [+1..+L_f] sui passi validi.")
        if not torch.allclose(t_dt, expected_t_dt):
            raise ValueError("target_delta_t non corrisponde a future_target_positions + 1.")


def validate_predictions(
    preds: torch.Tensor,
    batch_size: int,
    *,
    require_monotonic_quantiles: bool = False,
) -> None:
    """Verifica che l'output di un modello rispetti il contratto `(B, 3, 3, 3)`."""
    if not isinstance(preds, torch.Tensor):
        raise TypeError(f"Le predizioni devono essere un torch.Tensor, trovato {type(preds)}")

    expected_shape = (batch_size, N_FUTURE, N_TARGETS, N_QUANTILES)
    if tuple(preds.shape) != expected_shape:
        raise ValueError(f"Shape predizioni {tuple(preds.shape)} != attesa {expected_shape}")

    if not preds.is_floating_point():
        raise TypeError(f"Le predizioni devono avere dtype floating point, trovato {preds.dtype}")

    if not torch.isfinite(preds).all():
        raise ValueError("Le predizioni contengono NaN o Inf.")

    if require_monotonic_quantiles:
        if (preds[..., 0] > preds[..., 1]).any() or (preds[..., 1] > preds[..., 2]).any():
            raise ValueError("Quantile crossing rilevato: richiesto q0.1 <= q0.5 <= q0.9.")


def make_dummy_batch(
    batch_size: int = 4,
    *,
    num_features: int = N_DEFAULT_FEATURES,
    min_hist_len: int = 11,
    max_hist_len: int = 26,
    min_fut_len: int = 15,
    max_fut_len: int = 30,
    seed: int = 42,
    device: torch.device | str = "cpu",
) -> Batch:
    """Genera un batch sintetico pienamente conforme al contratto per test rapidi."""
    if min_hist_len < N_PAST or min_hist_len > max_hist_len:
        raise ValueError("Intervallo hist_len non valido.")
    if min_fut_len < N_FUTURE or min_fut_len > max_fut_len:
        raise ValueError("Intervallo fut_len non valido.")

    gen = torch.Generator(device="cpu").manual_seed(seed)

    h_len = torch.randint(min_hist_len, max_hist_len + 1, (batch_size,), generator=gen, dtype=torch.int64)
    f_len = torch.randint(min_fut_len, max_fut_len + 1, (batch_size,), generator=gen, dtype=torch.int64)
    T_h = int(h_len.max().item())
    T_f = int(f_len.max().item())

    h_steps = torch.arange(T_h).unsqueeze(0)
    f_steps = torch.arange(T_f).unsqueeze(0)
    h_pad = h_steps >= h_len.unsqueeze(1)
    f_pad = f_steps >= f_len.unsqueeze(1)

    history = torch.randn((batch_size, T_h, num_features), generator=gen, dtype=torch.float32)
    future = torch.randn((batch_size, T_f, num_features), generator=gen, dtype=torch.float32)

    # Maschere realistiche: 5% missing meteo + pad = True
    history_mask = torch.rand((batch_size, T_h, num_features), generator=gen) < 0.05
    future_mask = torch.rand((batch_size, T_f, num_features), generator=gen) < 0.05

    # Se ci sono almeno 3 colonne target alla fine, simula i 3 clear-sky in history e 100% NaN in future
    if num_features >= N_TARGETS:
        t_slice = slice(num_features - N_TARGETS, num_features)
        history_mask[:, :, t_slice] = True
        for b in range(batch_size):
            lh = int(h_len[b].item())
            mid = lh // 2
            history_mask[b, 0, t_slice] = False
            history_mask[b, mid, t_slice] = False
            history_mask[b, lh - 1, t_slice] = False
        future_mask[:, :, t_slice] = True

    history_mask = history_mask | h_pad.unsqueeze(-1)
    future_mask = future_mask | f_pad.unsqueeze(-1)
    history = history.masked_fill(history_mask, 0.0)
    future = future.masked_fill(future_mask, 0.0)

    history_delta_t = torch.where(h_pad, torch.zeros(1), (h_steps - (h_len.unsqueeze(1) - 1)).float())
    future_delta_t = torch.where(f_pad, torch.zeros(1), (f_steps + 1).float())

    pos = torch.zeros((batch_size, N_FUTURE), dtype=torch.int64)
    for b in range(batch_size):
        lf = int(f_len[b].item())
        p1 = max(0, lf // 3 - 1)
        p2 = max(p1 + 1, (2 * lf) // 3 - 1)
        p3 = lf - 1
        pos[b] = torch.tensor([p1, p2, p3], dtype=torch.int64)

    target_delta_t = (pos + 1).float()
    target_raw = torch.rand((batch_size, N_FUTURE, N_TARGETS), generator=gen, dtype=torch.float32) * 0.8
    target = torch.randn((batch_size, N_FUTURE, N_TARGETS), generator=gen, dtype=torch.float32)
    target_mask = torch.zeros((batch_size, N_FUTURE, N_TARGETS), dtype=torch.bool)
    last_obs = torch.randn((batch_size, N_TARGETS), generator=gen, dtype=torch.float32)

    batch: Batch = {
        "history": history,
        "history_mask": history_mask,
        "history_padding_mask": h_pad,
        "history_len": h_len,
        "history_delta_t": history_delta_t,
        "future": future,
        "future_mask": future_mask,
        "future_padding_mask": f_pad,
        "future_len": f_len,
        "future_delta_t": future_delta_t,
        "future_target_positions": pos,
        "target_delta_t": target_delta_t,
        "last_obs": last_obs,
        "target": target,
        "target_raw": target_raw,
        "target_mask": target_mask,
        "area": [f"DUMMY_AREA_{i}" for i in range(batch_size)],
        "climate": ["Csa"] * batch_size,
        "sample_idx": torch.arange(batch_size, dtype=torch.int64),
    }
    return move_batch_to_device(batch, device)  # type: ignore[return-value]


def concat_history_future(batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Concatena history e future senza padding interno (utile per modelli single-sequence).

    Per ogni campione `b` concatena i primi `history_len[b]` giorni di `history`
    con i primi `future_len[b]` giorni di `future` in una sequenza giornaliera
    contigua di lunghezza `L_tot = L_h + L_f`, con right-padding fino a `max(L_tot)`.

    Restituisce un dizionario con:
      - `seq`: (B, T_tot, F) float32
      - `mask`: (B, T_tot, F) bool (True = mancante/pad)
      - `padding_mask`: (B, T_tot) bool (True = pad)
      - `delta_t`: (B, T_tot) float32
      - `is_future`: (B, T_tot) bool (True sui passi validi di future)
      - `seq_len`: (B,) int64
      - `target_positions`: (B, 3) int64 (indici 0-based dei 3 target dentro `seq`)
    """
    history = batch["history"]
    history_mask = batch["history_mask"]
    history_dt = batch["history_delta_t"]
    h_len = batch["history_len"]

    future = batch["future"]
    future_mask = batch["future_mask"]
    future_dt = batch["future_delta_t"]
    f_len = batch["future_len"]

    B, _, F = history.shape
    device = history.device
    seq_len = h_len + f_len
    T_tot = int(seq_len.max().item())

    seq = torch.zeros((B, T_tot, F), dtype=history.dtype, device=device)
    mask = torch.ones((B, T_tot, F), dtype=torch.bool, device=device)
    padding_mask = torch.ones((B, T_tot), dtype=torch.bool, device=device)
    delta_t = torch.zeros((B, T_tot), dtype=history_dt.dtype, device=device)
    is_future = torch.zeros((B, T_tot), dtype=torch.bool, device=device)

    for b in range(B):
        lh = int(h_len[b].item())
        lf = int(f_len[b].item())
        lt = lh + lf

        seq[b, :lh] = history[b, :lh]
        seq[b, lh:lt] = future[b, :lf]

        mask[b, :lh] = history_mask[b, :lh]
        mask[b, lh:lt] = future_mask[b, :lf]

        padding_mask[b, :lt] = False

        delta_t[b, :lh] = history_dt[b, :lh]
        delta_t[b, lh:lt] = future_dt[b, :lf]

        is_future[b, lh:lt] = True

    target_positions = h_len.unsqueeze(1) + batch["future_target_positions"]

    return {
        "seq": seq,
        "mask": mask,
        "padding_mask": padding_mask,
        "delta_t": delta_t,
        "is_future": is_future,
        "seq_len": seq_len,
        "target_positions": target_positions,
    }
