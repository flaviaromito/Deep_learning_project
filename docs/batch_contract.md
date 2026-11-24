# Contratto del Batch e Interfaccia Modello–Pipeline

Questo documento definisce l'**interfaccia unica e vincolante** tra:

- **Pipeline dati (`src/preprocessing.py`)** — caricamento, normalizzazione, masking, padding dinamico e `DataLoader`;
- **Modelli (`src/models/`)** — qualsiasi architettura (baseline, RNN/LSTM/GRU, TCN, Transformer, Encoder–Decoder, ecc.);
- **Training, Loss e Valutazione (`src/train.py`, `src/eval.py`, `src/metrics.py`)**.

La specifica è implementata e validata automaticamente in [`src/contract.py`](../src/contract.py).

---

## 1. Convenzioni Fondamentali (Invarianti Globali)

1. **Convenzione unica delle maschere (`mask_true_is_missing: true`)**:
   - `True` = **mancante** (`NaN` originale, feature oscurata da ablation, oppure timestep di padding).
   - `False` = **valido** (valore osservato e reale).
   - Questa convenzione coincide al 100% sia con i file `.npz` originali sia con `key_padding_mask` / `src_key_padding_mask` di PyTorch (`nn.MultiheadAttention`, `nn.Transformer`).
2. **Nessuna interpolazione dei `NaN`**:
   - I valori mancanti non vengono mai interpolati.
   - Dopo la normalizzazione `standardization_arcsinh`, tutte le posizioni con `mask == True` vengono riempite con `nan_fill_value = 0.0` (corrispondente alla media standardizzata `arcsinh(0) = 0`), accompagnate dalle maschere booleane esplicite.
   - **Nessun tensore float del `batch` contiene mai `NaN` o `Inf`** (evita gradienti `NaN` in backward).
3. **Right-padding dinamico (`dynamic_padding: true`)**:
   - Sia `history` sia `future` sono paddati **a destra** fino alla lunghezza massima nel batch (`T_h = max(history_len)`, `T_f = max(future_len)`).
   - Compatibile nativamente con `pack_padded_sequence(..., batch_first=True, enforce_sorted=False)`, Transformer con `key_padding_mask` e indicizzazione diretta tramite `future_target_positions`.
   - Se `dynamic_padding: false`, il padding avviene a lunghezza fissa `max_seq_len` (default `150`, superiore al massimo osservato di `126` su `ood_s`).
   - I modelli **non devono mai assumere una lunghezza fissa** né usare embedding posizionali appresi tarati solo sul massimo del train (`70` gg): devono supportare sequenze di almeno `150+` giorni (es. positional encoding sinusoidale/continuo su `delta_t`).
4. **Zero Data Leakage**:
   - Lo scaler (`DATA/train_multi_output_clear_sky/scaler.json`) è stimato **solo sul `train`** e applicato identicamente a `val`, `id`, `ood_s`, `ood_t` (e split nascosto).
   - `future_noise` non viene **mai** inserito nel `batch` né sottratto a `future`.
   - Le colonne dei 3 indici vegetativi (`target_history_avg_*`) in `future` sono sempre mascherate (`future_mask[..., target_cols] = True`) e poste a `0.0`.
   - `climate` non deve **mai** essere usato come feature di input del modello (100% `"unknown"` su `ood_s`): è esposto nel `batch` solo per diagnostica e analisi errori.
5. **Contiguità temporale giornaliera (verificata su tutti i 140.122 campioni)**:
   - `history` è una serie giornaliera contigua di lunghezza $L_h$ che termina al giorno $t_0$ (`Δt = 0`), con le 3 acquisizioni *clear-sky* sempre presenti (prima al giorno `0`, terza al giorno $L_h - 1$).
   - `future` è una serie giornaliera contigua di lunghezza $L_f$ che inizia a $t_0 + 1$ (`Δt = +1 .. +L_f`).
   - `future_target_positions` sono indici `0`-based in `future` (con l'ultimo sempre pari a $L_f - 1$), quindi l'orizzonte in giorni rispetto a $t_0$ è esattamente `future_target_positions + 1`.

---

## 2. Schema del Dizionario `batch`

Ogni iterazione del `DataLoader` restituisce un dizionario `batch: dict[str, Any]` (`src.contract.Batch`) con le seguenti chiavi obbligatorie:

| Chiave | Tipo / Shape | Dtype | Semantica | Uso |
|---|---|---|---|---|
| `history` | `Tensor (B, T_h, F)` | `float32` | Feature giornaliere storiche normalizzate (`arcsinh`), con posizioni mancanti/pad riempite a `0.0`. | Input modello |
| `history_mask` | `Tensor (B, T_h, F)` | `bool` | Maschera element-wise: `True` = `NaN` originale o padding, `False` = valore valido. | Input modello (canali di presenza `(~mask).float()`) |
| `history_padding_mask` | `Tensor (B, T_h)` | `bool` | Maschera per timestep: `True` per `t >= history_len`, `False` per giorni reali. | `key_padding_mask` / pooling |
| `history_len` | `Tensor (B,)` | `int64` | Lunghezza reale $L_h \ge 11$ di ciascuna sequenza storica. | `pack_padded_sequence` / indicizzazione |
| `history_delta_t` | `Tensor (B, T_h)` | `float32` | Giorni relativi a $t_0$: `[-(L_h-1), ..., -1, 0]` sui passi validi, `0.0` sul padding. | Encoding temporale continuo |
| `future` | `Tensor (B, T_f, F)` | `float32` | Feature giornaliere future (meteo rumoroso + calendario/Fourier) normalizzate; colonne target e posizioni mancanti/pad a `0.0`. | Input modello |
| `future_mask` | `Tensor (B, T_f, F)` | `bool` | Maschera element-wise: `True` = `NaN` originale (sempre `True` sulle colonne target e sul meteo futuro se `use_future_meteo: false`) o padding. | Input modello |
| `future_padding_mask` | `Tensor (B, T_f)` | `bool` | Maschera per timestep: `True` per `t >= future_len`, `False` per giorni reali. | `key_padding_mask` / pooling |
| `future_len` | `Tensor (B,)` | `int64` | Lunghezza reale $L_f \ge 15$ di ciascuna sequenza futura. | `pack_padded_sequence` / indicizzazione |
| `future_delta_t` | `Tensor (B, T_f)` | `float32` | Giorni relativi a $t_0$: `[+1, +2, ..., +L_f]` sui passi validi, `0.0` sul padding. | Encoding temporale continuo |
| `future_target_positions` | `Tensor (B, 3)` | `int64` | Indici `0`-based dentro `future` dei 3 passi *clear-sky* da prevedere (`0 <= pos < future_len`, ordinati crescenti). | Gather delle rappresentazioni ai 3 target |
| `target_delta_t` | `Tensor (B, 3)` | `float32` | Orizzonte in giorni dei 3 target rispetto a $t_0$, uguale a `(future_target_positions + 1).float()`. | Query di orizzonte / teste dirette |
| `last_obs` | `Tensor (B, 3)` | `float32` | Ultima osservazione *clear-sky* storica di `[NDVI, GNDVI, EVI]` (al giorno $t_0$, indice `L_h - 1`), **nella stessa scala di `target`**. | Residual forecasting & baseline persistenza |
| `target` | `Tensor (B, 3, 3)` | `float32` | Ground truth `(B, H=3, C=3)` = `(batch, step_futuro, indice)`. Normalizzato con lo scaler del train se `normalize_targets: true`, altrimenti grezzo. | Training loss (Pinball / CRPS) |
| `target_raw` | `Tensor (B, 3, 3)` | `float32` | Ground truth `(B, H=3, C=3)` sempre sulla **scala originale `[-1, 1]`**. | Valutazione metriche finali |
| `target_mask` | `Tensor (B, 3, 3)` | `bool` | Maschera target: `True` = mancante, `False` = valido (tutti `False` nei 4 split forniti; salvaguardia per lo split nascosto). | Mascheramento loss e metriche |
| `area` | `list[str]` (len `B`) | `str` | Codice area geografica (es. `"29SND"` o fascia latitudine su `ood_s`). | Solo diagnostica / analisi errori |
| `climate` | `list[str]` (len `B`) | `str` | Classe climatica di Köppen (es. `"Csa"`, `"unknown"` su `ood_s`). | **Solo diagnostica — vietato come input al modello** |
| `sample_idx` | `Tensor (B,)` | `int64` | Indice deterministico del campione nello split. | Tracciabilità / debugging |

> **Nota sulla dimensione feature `F`**:
> Di default `F = 21` (ordine canonico definito in `src.contract.FEATURE_NAMES`). Se `preprocessing.drop_features` è non vuoto (es. `["wind"]` in ablation), `F = 21 - len(drop_features)`. I modelli devono leggere dinamicamente il numero di feature dalla configurazione / metadati del dataset (o da `batch["history"].shape[-1]`), mai hardcodare `21`.

### Ordine Canonico delle Feature (`F = 21` di default)

| Indici | Gruppo | Feature |
|---|---|---|
| `0..7` | Meteo (8) | `wind`, `humidity`, `solar_irradiance`, `rainfall`, `pressure`, `avg_temperature`, `min_temperature`, `max_temperature` |
| `8..10` | Nuvole (3) | `thick_clouds`, `thin_clouds`, `clouds_shadows` |
| `11..17` | Calendario / Fourier (7) | `day_of_year`, `day_of_year_fourier_sin_{1,2,3}`, `day_of_year_fourier_cos_{1,2,3}` |
| `18..20` | Indici storici (3) | `target_history_avg_NDVI_clear_sky`, `target_history_avg_GNDVI_clear_sky`, `target_history_avg_EVI_clear_sky` |

### Ordine Canonico degli Assi di Target e Predizioni

- **Asse `H` (step futuri, dim = 3)**: `[0: step_1, 1: step_2, 2: step_3]` (ordinati temporalmente).
- **Asse `C` (indici vegetativi, dim = 3)**: `[0: NDVI, 1: GNDVI, 2: EVI]`.
- **Asse `Q` (quantili, dim = 3)**: `[0: q0.1, 1: q0.5, 2: q0.9]`.

---

## 3. Contratto dei Modelli (`src/models/`)

Ogni modello eredita da `torch.nn.Module` ed espone una firma uniforme:

```python
preds = model(batch)
```

- **Input**: il dizionario `batch` (già spostato sul device corretto tramite helper `move_batch_to_device(batch, device)`).
- **Output (`preds`)**:
  - **Shape**: `(B, 3, 3, 3)` corrispondente a `(B, H=3, C=3, Q=3)` = `(batch, step_futuro, indice, quantile)`.
  - **Dtype**: `torch.float32`, privo di `NaN`/`Inf`.
  - **Scala**: la **stessa scala di `batch["target"]`** (spazio `standardization_arcsinh` se `normalize_targets: true`, scala originale se `false`).
  - **Monotonicità dei quantili**: fortemente raccomandata per costruzione architetturale (`q0.1 <= q0.5 <= q0.9`), ad esempio prevedendo mediana $\hat{q}_{0.5}$ e scostamenti positivi tramite `softplus`:
    ```python
    q50 = raw_med
    q10 = q50 - F.softplus(raw_low)
    q90 = q50 + F.softplus(raw_high)
    preds = torch.stack([q10, q50, q90], dim=-1)  # (B, 3, 3, 3)
    ```

### Pattern d'Uso Supportati dal Contratto

1. **Previsione Residua (consigliata per OOD-S)**:
   ```python
   # delta_preds: (B, 3, 3, 3), last_obs: (B, 3) -> (B, 1, 3, 1)
   preds = batch["last_obs"][:, None, :, None] + delta_preds
   ```
2. **Baseline di Persistenza (zero parametri)**:
   ```python
   preds = batch["last_obs"][:, None, :, None].expand(-1, 3, -1, 3)
   ```
3. **Modelli Encoder–Decoder (LSTM/GRU/Transformer/TFT)**:
   - Encoder su `(batch["history"], batch["history_mask"], batch["history_delta_t"], batch["history_padding_mask"])`.
   - Decoder su `(batch["future"], batch["future_mask"], batch["future_delta_t"], batch["future_padding_mask"])`.
   - Lettura ai 3 passi target tramite `torch.gather` con `batch["future_target_positions"]`:
     ```python
     # dec_out: (B, T_f, D), pos: (B, 3)
     idx = batch["future_target_positions"].unsqueeze(-1).expand(-1, -1, dec_out.size(-1))
     target_repr = dec_out.gather(dim=1, index=idx)  # (B, 3, D)
     ```
4. **Modelli Single-Sequence (Encoder-only Transformer / BiGRU / TCN)**:
   - Usare direttamente `src.contract.concat_history_future(batch)`, che concatena i passi validi di `history` e `future` **senza padding interno** restituendo `seq`, `mask`, `padding_mask`, `delta_t`, `is_future`, `seq_len` e `target_positions` (già traslati di `history_len` per fare `torch.gather` direttamente su `seq`).

---

## 4. Contratto Loss e Valutazione

### Training Loss (spazio di `batch["target"]`)

Grazie alla convenzione `(B, H, C, Q)`, la **Pinball Loss** si calcola con broadcasting diretto:

```python
# preds: (B, 3, 3, 3), target: (B, 3, 3), target_mask: (B, 3, 3)
quantiles = torch.tensor([0.1, 0.5, 0.9], device=preds.device, dtype=preds.dtype)
errors = batch["target"].unsqueeze(-1) - preds                      # (B, 3, 3, 3)
pinball = torch.maximum(quantiles * errors, (quantiles - 1) * errors)
valid = (~batch["target_mask"]).unsqueeze(-1)                       # (B, 3, 3, 1)
loss = pinball[valid.expand_as(pinball)].mean()
```

### Valutazione (sempre sulla scala originale)

1. Se `cfg["preprocessing"]["normalize_targets"]` è `true`, si applica l'inversa dello scaler del train a `preds`:
   $$\hat{x} = \sinh(\hat{y}) \cdot \sqrt{\sigma^2_{\text{target}} + \varepsilon} + \mu_{\text{target}}$$
   ottenendo `preds_raw` di shape `(B, 3, 3, 3)` sulla scala originale.
2. Tutte le metriche richieste dalla traccia (**RMSE, MAE, WMAPE, MASE** su `q0.5 = preds_raw[..., 1]`; **CRPS, Pinball Loss, Coverage 80%** su `[preds_raw[..., 0], preds_raw[..., 2]]`) si calcolano confrontando `preds_raw` con `batch["target_raw"]` (rispettando `~batch["target_mask"]`).

---

## 5. Effetto dei Parametri `preprocessing` in `config.yaml`

| Parametro `config.yaml` | Default | Comportamento nel `batch` |
|---|---|---|
| `normalize_features` | `true` | Applica `standardization_arcsinh` (scaler del train) a `history` e `future` prima del fill `0.0`. |
| `normalize_targets` | `true` | Applica `standardization_arcsinh` (statistiche `target_mean`/`target_var` del train) a `target` e `last_obs`. `target_raw` resta sempre grezzo. |
| `nan_fill_value` | `0.0` | Valore scritto nelle posizioni con `mask == True` dopo la normalizzazione. |
| `mask_true_is_missing` | `true` | Fisso a `true` (errore esplicito se impostato a `false`, per impedire inversioni silenti). |
| `max_seq_len` | `150` | Lunghezza fissa usata se `dynamic_padding: false`; capacità minima richiesta ai modelli. |
| `dynamic_padding` | `true` | `true`: padding alla max lunghezza nel batch; `false`: padding fisso a `max_seq_len`. |
| `use_climate` | `false` | Fisso a `false` come input del modello; `batch["climate"]` è solo diagnostico. |
| `use_future_meteo` | `true` | Se `false`, maschera (`True`) e azzera (`0.0`) tutte le colonne meteo e nuvole (`0..10`) in `future`, mantenendo le feature temporali/Fourier e `future_delta_t`. |
| `drop_features` | `[]` | Rimuove le colonne elencate per nome da `history`/`history_mask` e `future`/`future_mask` (riducendo `F`). `last_obs` viene sempre estratto prima della rimozione. |
| `add_delta_t` | `true` | Popola `history_delta_t`, `future_delta_t`, `target_delta_t` (se `false`, i tensori restano comunque presenti ma riempiti con `0.0` per non rompere il contratto delle chiavi). |

---

## 6. Validazione Automatica e Dummy Batch per il Team

Il modulo [`src/contract.py`](../src/contract.py) mette a disposizione tre funzioni pronte all'uso:

- `validate_batch(batch, expected_num_features=None)`: controlla chiavi, shape, dtype, assenza di `NaN`/`Inf` e coerenza tra maschere, lunghezze, `future_target_positions` e `delta_t`.
- `validate_predictions(preds, batch_size)`: controlla che l'output di un modello rispetti `(B, 3, 3, 3)`, `float32` e sia finito.
- `make_dummy_batch(batch_size=4, ...)`: genera un batch sintetico 100% conforme al contratto per sviluppare e testare modelli, loss e metriche in modo indipendente dai file `.npz`.
