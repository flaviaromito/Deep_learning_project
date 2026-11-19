# Deep_learning_project

Progetto del corso di **Deep Learning** – Laurea Magistrale in Ingegneria dei Sistemi Intelligenti, Università Campus Bio-Medico di Roma.

**Autori:** Tommaso Giannini, Flavia Romito, Erika Oliveri

---

## Indice

1. [La traccia del progetto](#1-la-traccia-del-progetto)
   - [1.1 Obiettivo](#11-obiettivo)
   - [1.2 Indici di vegetazione da prevedere](#12-indici-di-vegetazione-da-prevedere)
   - [1.3 Dataset](#13-dataset)
   - [1.4 Nuvole e campionamento irregolare](#14-nuvole-e-campionamento-irregolare)
   - [1.5 Split dei dati e generalizzazione](#15-split-dei-dati-e-generalizzazione)
   - [1.6 Parametri fissati ed iperparametri](#16-Parametri-fissati-ed-iperparametri)
   - [1.7 Output probabilistico e funzione di perdita](#17-output-probabilistico-e-funzione-di-perdita)
   - [1.8 Metriche di valutazione](#18-metriche-di-valutazione)
   - [1.9 Consegna](#19-consegna)

---

## 1. La traccia del progetto

### 1.1 Obiettivo

Il progetto riguarda la **previsione dello stato di salute della vegetazione** in Europa a partire da serie temporali satellitari e da dati meteorologici.

Bisogna prevedere **tre indici di vegetazione** (NDVI, GNDVI, EVI) **circa 14 giorni in avanti**, cioè per le **prossime 3 acquisizioni a cielo sereno** del satellite. Come input si usano le osservazioni passate a cielo sereno e le covariate meteo giornaliere (passate e future).

```
  storia (3 acquisizioni clear-sky passate) + meteo giornaliero
                          │
                          ▼
                    [ nostro modello ]
                          │
                          ▼
  3 indici × 3 passi futuri × 3 quantili (0.1, 0.5, 0.9)
```

### 1.2 Indici di vegetazione da prevedere

Gli indici si calcolano dalle bande spettrali di Sentinel-2 (Blue, Green, Red, Near-Infrared):

| Indice | Formula | Significato |
|---|---|---|
| **NDVI** – Normalized Difference Vegetation Index | $\dfrac{NIR - Red}{NIR + Red}$ | Indicatore standard del verde della chioma (*canopy greenness*) |
| **GNDVI** – Green NDVI | $\dfrac{NIR - Green}{NIR + Green}$ | Più sensibile al contenuto di clorofilla |
| **EVI** – Enhanced Vegetation Index | $2.5 \cdot \dfrac{NIR - Red}{NIR + 6\,Red - 7.5\,Blue + 1}$ | Satura meno sulla vegetazione densa e riduce le interferenze atmosferiche |

I valori stanno nell'intervallo $[-1, 1]$: valori vicini a $-1$ indicano suolo nudo, acqua, rocce o vegetazione morta, mentre valori che crescono verso $1$ indicano vegetazione viva, sana e in crescita.

### 1.3 Dataset

I dati provengono da **GreenEarthNet**:

- **Minicubi Sentinel-2**: 128 × 128 pixel a 20 m di risoluzione (circa 2.56 km di lato), con bande Blue, Green, Red e NIR, anni **2017–2022**.
- **Target clear-sky**: gli indici sono mediati sui pixel di coltura validi. Esistono solo per le date senza nuvole, quindi la serie è campionata in modo irregolare.
- **Meteo giornaliero (E-OBS)**: vento, umidità, radiazione solare, pioggia, pressione al livello del mare, temperatura (minima, media, massima).
- **Area geografica**: tutta Europa, dai climi mediterranei a quelli continentali e boreali (classi di Köppen-Geiger diverse). Prevalgono frumento, mais e altri cereali.

Le **21 feature** di input comprendono: meteo, copertura nuvolosa, termini di Fourier del giorno dell'anno (*day-of-year*) e i valori clear-sky di NDVI, GNDVI ed EVI.

### 1.4 Nuvole e campionamento irregolare

Quando passa sopra un'area nuvolosa il satellite non vede il suolo, e gli indici calcolati vengono sbagliati (vicini a $-1$). Se la copertura è parziale, la media sui pixel risulta falsata verso il basso. Per questo:

- **Vanno usati solo i target clear-sky**, sia in addestramento sia in valutazione. Nel dataset ci sono anche target calcolati su tutte le date (es. `NDVI all`), ma **non vanno usati**.
- Sentinel-2 passa nominalmente ogni 5 giorni, ma nella pratica ogni 3–7 giorni. Togliendo le date nuvolose, la serie dei target diventa **sparsa e irregolare**. Il meteo invece è **giornaliero e regolare**.

**Vincoli sul preprocessing:**

1. **Niente interpolazione**: i buchi restano `NaN`. Il modello li deve gestire con un **masking** o con un'architettura che accetta valori mancanti.
2. **Finestre già tagliate**: 3 acquisizioni passate (*lookback* = 3) → 3 acquisizioni future (*horizon* = 3, circa 14 giorni), con **stride 4**.
3. **Meteo futuro rumoroso**: alle covariate meteo future è già stato aggiunto rumore, per simulare una vera previsione meteo.
4. **Normalizzazione calcolata solo sul train**: viene fornito uno `scaler.json` (media, varianza, trasformazione `arcsinh`) stimato sul train ma **non ancora applicato**. Possiamo usarlo oppure definirne uno nostro, ma va sempre stimato solo sul train e poi applicato senza modifiche agli altri split.

### 1.5 Split dei dati e generalizzazione

Si addestra **una sola volta** e poi si misura quanto il modello generalizza nello **spazio** e nel **tempo**:

| Split | Ruolo | Dove | Quando | Campioni |
|---|---|---|---|---|
| `train` | Addestramento | Regioni di training | 2017–2019 | 55.338 |
| `id` | Test in-distribution | Vicino alle località di training | 2020 | 3.336 |
| `ood-s` | Test con spostamento spaziale | **Regioni mai viste** | 2017–2019 | 25.619 |
| `ood-t` | Test con spostamento temporale | Stesse località di `id` | **2021–2022** | 6.512 |
| *hidden* | Valutazione finale | Tenuto dai docenti, mai rilasciato | – | – |

Regole:
- Si addestra **solo sullo split `train`**.
- `id`, `ood-s` e `ood-t` servono **solo per la valutazione**.
- Gli iperparametri si scelgono su un **sottoinsieme di validazione ricavato dal `train`**.

> Conta soprattutto **quanto il modello generalizza a regioni e anni mai visti**.

### 1.6 Cosa è fisso e cosa scegliamo noi

| Fisso per tutti i gruppi | A nostra scelta |
|---|---|
| Finestre: 3 passi passati → 3 passi futuri clear-sky, stride 4 | **Normalizzazione**: lo scaler fornito oppure uno nostro |
| Target: NDVI, GNDVI ed EVI, tutti e tre | **Valori mancanti**: masking, o modelli che accettano `NaN` |
| 21 feature (meteo, nuvole, Fourier del giorno dell'anno, indici clear-sky) | **Feature**: selezione, embedding, feature derivate |
| Buchi lasciati come `NaN` con maschere, senza interpolazione | **Architettura**: statistica, ricorrente (RNN/LSTM), convoluzionale, Transformer, foundation model, … |
| Meteo futuro già perturbato con rumore | **Loss e training**: qualsiasi obiettivo che produca i tre quantili |
| `scaler.json` stimato sul train, non applicato | |
| Addestramento solo sullo split `train` | |
| Output: quantili 0.1, 0.5, 0.9 per ogni indice e ogni passo | |
| Valutazione su `id`, `ood-s`, `ood-t` e split nascosto | |

### 1.7 Output probabilistico e funzione di perdita

Il modello deve fare una **previsione probabilistica**: non basta una stima puntuale, servono i **quantili 0.1, 0.5 (mediana) e 0.9** per ognuno dei 3 indici e dei 3 passi futuri, quindi **3 × 3 × 3 = 27 valori** per campione.

Una scelta naturale per la loss è la **pinball loss** (o *quantile loss*), che per un quantile $\tau \in \{0.1, 0.5, 0.9\}$ vale:

$$
L_\tau(y, \hat{q}_\tau) = \max\big(\tau\,(y - \hat{q}_\tau),\; (\tau - 1)\,(y - \hat{q}_\tau)\big)
$$

In alternativa si può usare il **CRPS** (*Continuous Ranked Probability Score*).

### 1.8 Metriche di valutazione

Il modello viene confrontato con il **modello di riferimento dei docenti**, usando gli stessi split, lo stesso orizzonte e metriche calcolate sulla **scala originale degli indici** (cioè dopo aver annullato la normalizzazione).

- **Accuratezza puntuale** sulla mediana ($q = 0.5$), per ogni indice e ogni split:
  - RMSE – *Root Mean Squared Error*
  - MAE – *Mean Absolute Error*
  - WMAPE – *Weighted Mean Absolute Percentage Error*
  - MASE – *Mean Absolute Scaled Error*
- **Incertezza** sui quantili 0.1, 0.5, 0.9:
  - CRPS
  - Pinball loss media
  - **Copertura** empirica dell'intervallo all'80% $[\hat{q}_{0.1}, \hat{q}_{0.9}]$

### 1.9 Consegna

- Il progetto vale il **25% del voto finale**.
- **Presentazione**: 12 minuti esatti, seguiti da domande. I gruppi sono gli stessi della presentazione del paper.
- **Repository del codice**:
  - codice pulito, commentato ed eseguibile dall'inizio alla fine;
  - `README.md` con setup, istruzioni per addestramento e valutazione, e risultati;
  - `requirements.txt` con versioni fissate.
- **Slide**: il file `.pptx` usato nella presentazione.
- **Scadenza**: da comunicare.
- Nella valutazione contano le metriche sugli split OOD, ma anche il **rigore metodologico**, la **pulizia del codice** e la **motivazione teorica delle scelte architetturali**.

---

## 2. Setup

```bash
# creare e attivare l'ambiente virtuale (testato con Python 3.14)
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# installare le dipendenze
pip install -r requirements.txt
```

## 3. Struttura del repository

*Da completare.*

## 4. Addestramento e valutazione

*Da completare.*

## 5. Risultati

*Da completare.*
