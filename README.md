# 🇹🇳 EUR/TND Exchange Rate Forecasting — Deep Learning

> **Prévision du taux de change Euro / Dinar Tunisien par Deep Learning**  
> *Ben Selma Hibe & Cherchir Aya — 2025*

---

## 📌 Objectif

Prédire le taux de change **EUR/TND du lendemain** en comparant 5 architectures Deep Learning entraînées sur des données macroéconomiques et de marché (2015–2025).

---

## 🏆 Résultats Finaux

| Rang | Modèle | MAE (TND) | RMSE (TND) | MAPE (%) | Δ vs Naïf |
|:---:|---|---:|---:|---:|---:|
| 🥇 | **LSTM** | **0.006736** | **0.009452** | **0.2015%** | **+31.34%** |
| 🥈 | CNN-LSTM | 0.007520 | 0.010440 | 0.2249% | +23.34% |
| 🥉 | Bidirectional LSTM | 0.007977 | 0.010779 | 0.2385% | +18.69% |
| 4 | MLP (Baseline DL) | 0.009479 | 0.014763 | 0.2837% | +3.38% |
| — | Référence naïve (lag-1) | 0.009811 | 0.017180 | 0.2935% | 0.00% |
| 5 | Transformer | 0.009955 | 0.017172 | 0.2978% | -1.47% |

> **Le LSTM surpasse le modèle naïf de 31.34%** avec une MAE de seulement **0.0067 TND** (~0.20% d'erreur relative).

---

## 📁 Structure du Projet

```
📦 projet-dl-eurtnd/
├── 📓 02_Modeling_DL.ipynb          # Notebook principal (12 sections)
├── 📊 macro_market_merged.csv        # Dataset brut (2015–2025)
├── 📊 dataset_features.csv           # Dataset après feature engineering
├── 📈 model_comparison_final_dl.csv  # Métriques comparatives
├── 📈 walkforward_predictions_dl.csv # Prédictions walk-forward
├── 🖼️ feature_importance_corr.png
├── 🖼️ train_test_split.png
├── 🖼️ model_comparison_metrics.png
├── 🖼️ actual_vs_predicted_all.png
├── 🖼️ learning_curves_all.png
├── 🖼️ val_loss_comparison.png
├── 🖼️ walkforward_best_dl.png
└── 🖼️ forecast_tomorrow_dl.png
```

---

## 📊 Dataset

| Propriété | Valeur |
|---|---|
| **Source** | BCT + Yahoo Finance (`yfinance`) |
| **Période** | 02/01/2015 → 01/01/2025 |
| **Fréquence** | Journalière |
| **Observations** | ~3 650 jours |
| **Features brutes** | 10 variables |
| **Features engineered** | ~70 variables |

### Variables macroéconomiques

| Variable | Description |
|---|---|
| `Balance Commerciale` | Balance commerciale TN (mensuelle interpolée) |
| `PIB` | Produit Intérieur Brut (trimestriel interpolé) |
| `Taux Intérêt` | Taux directeur BCT |
| `Inflation Daily` | Inflation quotidienne interpolée |

### Variables de marché

| Variable | Description |
|---|---|
| `eur_usd` | Taux EUR/USD |
| `brent_oil` | Prix du pétrole Brent (USD/baril) |
| `gold` | Prix de l'or (USD/once) |
| `eur_gbp` | Taux EUR/GBP |
| `us_10y_yield` | Rendement obligataire US 10 ans |
| `msci_em` | Indice MSCI Marchés Émergents |

---

## ⚙️ Feature Engineering

À partir des 10 variables brutes, on construit ~70 features :

| Type | Description | Exemple |
|---|---|---|
| **Lags** | Valeurs décalées (1, 3, 7, 30 jours) | `eurtnd_lag7` |
| **Rolling stats** | Moyenne et volatilité glissantes | `eurtnd_ma30`, `eurtnd_std14` |
| **Log-returns** | Rendements logarithmiques (1, 7, 30j) | `eurtnd_ret1` |
| **Spreads** | Taux réel, déviation vs MA30 | `rate_inflation_spread` |
| **Calendrier** | Jour, mois, trimestre | `day_of_week`, `quarter` |

---

## 🧠 Architectures Deep Learning

### 🔵 MLP — Baseline DL
```
Input (n_features — dernier timestep)
        ↓
  Dense(128) + BatchNorm + Dropout(0.3)
        ↓
  Dense(64)  + Dropout(0.2)
        ↓
  Dense(32)
        ↓
  Dense(1) → Prédiction
```

### 🟦 LSTM *(🏆 Meilleur)*
```
Input Sequence (30 jours × n_features)
        ↓
   LSTM(128, return_sequences=True) + L2
        ↓
   LSTM(64,  return_sequences=False)
        ↓
   Dropout(0.3)
        ↓
   Dense(32) → Dense(1) → Prédiction
```

### 🟢 Bidirectional LSTM
```
        Input Sequence
       /               \
BiLSTM(64, fwd)    BiLSTM(64, bwd)
       \               /
        Concatenation (128)
              ↓
        BiLSTM(32) → Dropout(0.3)
              ↓
        Dense(32) → Dense(1) → Prédiction
```

### 🟡 CNN-LSTM
```
Input Sequence
      ↓
Conv1D(64, k=3, causal) → Conv1D(32, k=3, causal)
      ↓
MaxPooling1D → Dropout(0.1)
      ↓
LSTM(64) → Dropout(0.2)
      ↓
Dense(32) → Dense(1) → Prédiction
```

### 🔴 Transformer
```
Input Sequence
      ↓
Dense(32) + Positional Embedding
      ↓
[MultiHeadAttention(2 heads) + LayerNorm] × 2
      ↓
GlobalAveragePooling1D → Dropout(0.2)
      ↓
Dense(32) → Dense(1) → Prédiction
```

---

## 🔑 Décision Clé — Prédire les Log-Returns

Le choix le plus important du projet est de **prédire les log-returns** (variations) plutôt que le prix brut.

**Pourquoi ?**

```
❌ Prix brut   : [3.100, 3.101, 3.100, 3.102 ...]
                  Signal journalier : ~0.001 TND sur une plage de 0.5 TND
                  → Modèle apprend la moyenne → MAE catastrophique (-500%)

✅ Log-return  : [+0.0003, -0.0003, +0.0006 ...]
                  Signal centré en 0, variance stable
                  → Apprentissage efficace → MAE compétitive (+31%)
```

**Reconstruction du prix :**
```python
log_return_pred  = model.predict(sequence)
price_tomorrow   = price_today × exp(log_return_pred)
```

---

## 🛠️ Paramètres d'Entraînement

| Paramètre | Valeur |
|---|---|
| **Fenêtre temporelle** | 30 jours |
| **Séparation train/test** | 80% / 20% (chronologique) |
| **Epochs max** | 150 |
| **Batch size** | 32 (64 pour Transformer) |
| **Optimizer** | Adam + gradient clipping (`clipnorm=1.0`) |
| **Learning rate** | 5e-4 (LSTM/BiLSTM/MLP), 3e-4 (CNN-LSTM), 2e-4 (Transformer) |
| **Early stopping** | patience=25, `restore_best_weights=True` |
| **ReduceLROnPlateau** | factor=0.5, patience=10, min_lr=1e-7 |
| **Régularisation** | L2 (1e-4) sur couches LSTM, Dropout |
| **Scaler X** | `StandardScaler` |
| **Scaler y** | `StandardScaler` (log-returns) |

---

## 📐 Métriques d'Évaluation

| Métrique | Formule | Interprétation |
|---|---|---|
| **MAE** | mean\|actual - pred\| | Erreur absolue moyenne en TND |
| **RMSE** | √mean(actual - pred)² | Pénalise les grandes erreurs |
| **MAPE** | mean\|actual - pred\| / actual × 100 | Erreur relative (%) |
| **Δ vs naïf** | (1 - MAE_model / MAE_naïf) × 100 | Gain sur le modèle de référence |

> Le **modèle naïf** prédit `price_t = price_{t-1}` (taux d'hier = taux d'aujourd'hui).  
> C'est le benchmark standard pour les séries financières.

---

## 🔄 Validation Walk-Forward

En plus du test set fixe, une **validation walk-forward** par fenêtres de 30 jours simule les conditions réelles : le modèle prédit par blocs successifs sans réentraînement, mesurant la stabilité des performances dans le temps.

---

## 💻 Environnement

```bash
Python        >= 3.10
TensorFlow    >= 2.13
scikit-learn  >= 1.3
pandas        >= 2.0
numpy         >= 1.24
matplotlib    >= 3.7
yfinance      >= 0.2
```

### Installation

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib yfinance
```

### Lancement

```bash
jupyter notebook 02_Modeling_DL.ipynb
```

> ⚠️ **Important** : placer `macro_market_merged.csv` dans le même dossier que le notebook avant de lancer.

---

## 📝 Interprétation des Résultats

**LSTM (+31.34%)** est le meilleur modèle car il capture naturellement les **dépendances temporelles à long terme** du taux EUR/TND, une devise gérée par la BCT avec des tendances persistantes.

**CNN-LSTM (+23.34%)** est second : les Conv1D détectent des **patterns locaux** (semaine, mois) que le LSTM mémorise ensuite.

**Transformer (-1.47%)** ne bat pas le naïf car les mécanismes d'attention nécessitent **beaucoup plus de données** (~100K samples) pour converger. Sur 3 650 jours, le LSTM reste supérieur.

---

## ⚠️ Limites

- Prévision limitée à **1 jour** (horizon court)
- Performances dépendantes du **régime de marché** (crises, décisions BCT imprévues)
- Le Transformer nécessiterait 10× plus de données pour exprimer son potentiel
- Pas de réentraînement automatique en production

---

*Projet réalisé dans le cadre d'un cours de Deep Learning appliqué à la Finance.*
