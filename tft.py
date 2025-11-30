# =========================================
# 0. INSTALLS (uncomment in Colab if needed)
# =========================================
# !pip install -q pytorch-forecasting==1.5.0
# !pip install -q lightning pandas numpy yfinance matplotlib seaborn

import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.preprocessing import StandardScaler

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping

import pytorch_forecasting as pf
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

# ---- debug probe (optional, for version checks) ----
print("PF ver :", pf.__version__)
print("PL ver :", pl.__version__)
print("TFT MRO:", TemporalFusionTransformer.__mro__)
# ----------------------------------------------------


# =========================================
# 1. DOWNLOAD PRICE DATA
# =========================================
tickers = [
    "AAPL", "MSFT", "GOOG", "AMZN", "TSLA",
    "META", "NVDA", "JPM", "V", "JNJ"
]

# You can silence the FutureWarning by setting auto_adjust explicitly
prices = yf.download(
    tickers,
    start="2018-01-01",
    end="2024-01-01",
    auto_adjust=True,  # or False, but keep it consistent
)["Close"]

# Convert MultiIndex columns -> tidy dataframe: [date, ticker, price]
prices = prices.stack().reset_index()
prices.columns = ["date", "ticker", "price"]


# =========================================
# 2. FEATURE ENGINEERING
# =========================================
def add_quantamental(dF: pd.DataFrame) -> pd.DataFrame:
    dF = dF.sort_values(["ticker", "date"])

    # calendar feature
    dF["day"] = dF["date"].dt.dayofweek

    # daily return
    dF["ret"] = dF.groupby("ticker")["price"].pct_change()

    # crude RSI-like feature: rolling mean / rolling std of returns
    dF["rsi"] = dF.groupby("ticker")["ret"].transform(
        lambda x: x.rolling(14).mean() / x.rolling(14).std()
    )

    # placeholder "fundamentals" (you should replace with real fundamentals later)
    dF["pe"] = np.random.lognormal(2.2, 0.3, len(dF))
    dF["roe"] = np.random.normal(0.15, 0.05, len(dF))

    # dummy macro driver (sin wave over ~1 year)
    dF["usd"] = np.sin(np.arange(len(dF)) * 2 * np.pi / 252)

    # integer time index
    dF["time_idx"] = (dF["date"] - dF["date"].min()).dt.days

    return dF


df = add_quantamental(prices)

# =========================================
# 3. DEFINE TARGET (NEXT-DAY LOG-RETURN)
# =========================================
df["target"] = np.log1p(df["ret"]).shift(-1)  # predict next-day log-return
df = df.dropna().reset_index(drop=True)

print("Data shape after feature eng. and target:", df.shape)


# =========================================
# 4. VARIABLE TYPES FOR TFT
# =========================================
static_cats = ["ticker"]
time_varying_known_reals = ["day", "usd"]
time_varying_unknown_reals = ["price", "rsi", "pe", "roe", "ret"]

# (Optional) add more technical indicators here, e.g.:
# df["ret_5d"] = df.groupby("ticker")["ret"].transform(lambda x: x.rolling(5).sum())
# df["vol_10d"] = df.groupby("ticker")["ret"].transform(lambda x: x.rolling(10).std())
# time_varying_unknown_reals += ["ret_5d", "vol_10d"]


# =========================================
# 5. SCALE FEATURES (BUT **NOT** TARGET)
# =========================================
# We let GroupNormalizer handle target per ticker.
# Only scale feature columns to avoid double-normalization.

feature_cols = time_varying_known_reals + time_varying_unknown_reals
scalers = {col: StandardScaler() for col in feature_cols}

for col, scaler in scalers.items():
    df[col] = scaler.fit_transform(df[[col]]).flatten()

print("Feature scaling done for:", feature_cols)


# =========================================
# 6. BUILD TimeSeriesDataSet (TRAIN & VAL)
# =========================================
max_encoder_len = 60   # look-back window
max_prediction_len = 1 # 1-step ahead forecast

# Better temporal split: ~90% train, 10% validation
training_cutoff = int(df["time_idx"].quantile(0.9))
print("Training cutoff time_idx:", training_cutoff)

training_df = df[df.time_idx <= training_cutoff].copy()

dataset = TimeSeriesDataSet(
    training_df,
    time_idx="time_idx",
    target="target",
    group_ids=["ticker"],

    min_encoder_length=max_encoder_len // 2,
    max_encoder_length=max_encoder_len,
    min_prediction_length=max_prediction_len,
    max_prediction_length=max_prediction_len,

    static_categoricals=static_cats,
    time_varying_known_reals=time_varying_known_reals,
    time_varying_unknown_reals=time_varying_unknown_reals,

    # IMPORTANT: single normalizer for target (no external scaling)
    target_normalizer=GroupNormalizer(groups=["ticker"]),

    add_relative_time_idx=True,
    add_target_scales=True,
    allow_missing_timesteps=True,
    add_encoder_length=True,
)

# Validation dataset from full df (pytorch_forecasting infers which parts to predict)
val_dataset = TimeSeriesDataSet.from_dataset(
    dataset,
    df,
    predict=True,
    stop_randomization=True,
)

train_loader = dataset.to_dataloader(
    batch_size=128,
    shuffle=True,
    num_workers=0  # keep 0 on Windows to avoid issues
)
val_loader = val_dataset.to_dataloader(
    batch_size=256,
    shuffle=False,
    num_workers=0
)

print(f"Number of training samples: {len(dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")


# =========================================
# 7. TRAINER WITH EARLY STOPPING
# =========================================
pl.seed_everything(42, workers=True)

early_stop_cb = EarlyStopping(
    monitor="val_loss",  # metric logged by TFT
    patience=5,
    mode="min"
)

trainer = pl.Trainer(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    max_epochs=1 0,                # more than 1 epoch now
    gradient_clip_val=0.1,
    logger=False,                 # no logger, so no LR monitor
    enable_progress_bar=True,
    callbacks=[early_stop_cb],
    log_every_n_steps=10,
)


# =========================================
# 8. BUILD TFT MODEL (UPGRADED CONFIG)
# =========================================
tft = TemporalFusionTransformer.from_dataset(
    dataset,
    learning_rate=3e-4,          # lower LR for stability
    hidden_size=128,             # larger model
    attention_head_size=4,
    dropout=0.2,                 # a bit more regularization
    reduce_on_plateau_patience=3,
    # loss = pf.metrics.MAE(),   # You can switch to MAE or QuantileLoss if desired
)

print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")


# =========================================
# 9. TRAIN MODEL
# =========================================
trainer.fit(tft, train_loader, val_loader)
print("Training finished.")

# Best validation loss seen during training (from EarlyStopping callback)
best_val_loss = early_stop_cb.best_score.item()
print(f"[UPGRADED MODEL] Best val_loss: {best_val_loss:.6f}")

# Validation loss in the last epoch
final_val_loss = trainer.callback_metrics["val_loss"].item()
print(f"[UPGRADED MODEL] Final epoch val_loss: {final_val_loss:.6f}")


# =========================================
# 10. PREDICTION & INTERPRETATION
# =========================================
# mode="raw" returns raw network output (for interpretability)
predictions = tft.predict(
    val_loader,
    mode="raw",
    return_x=True,
    trainer_kwargs={"accelerator": "cpu"}  # CPU is fine for inference
)

interpretation = tft.interpret_output(predictions.output, reduction="sum")

# Plot variable importances, attention, etc.
tft.plot_interpretation(interpretation)
plt.tight_layout()
plt.show()
