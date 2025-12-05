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

from sklearn.metrics import mean_absolute_error

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

prices = yf.download(
    tickers,
    start="2018-01-01",
    end="2024-01-01",
    auto_adjust=True,
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

    # placeholder "fundamentals"
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


# =========================================
# 5. SCALE FEATURES (BUT **NOT** TARGET)
# =========================================
feature_cols = time_varying_known_reals + time_varying_unknown_reals
scalers = {col: StandardScaler() for col in feature_cols}

for col, scaler in scalers.items():
    df[col] = scaler.fit_transform(df[[col]]).flatten()

print("Feature scaling done for:", feature_cols)


# =========================================
# 6. CREATE 70% / 15% / 15% SPLIT BY TIME
# =========================================
# dùng unique time_idx để chia theo thời gian, không shuffle
time_indices = np.sort(df["time_idx"].unique())
n_time = len(time_indices)

train_end_idx = time_indices[int(0.7 * n_time) - 1]   # mốc kết thúc train
val_end_idx   = time_indices[int(0.85 * n_time) - 1]  # mốc kết thúc val

train_df = df[df.time_idx <= train_end_idx].copy()
val_df   = df[(df.time_idx > train_end_idx) & (df.time_idx <= val_end_idx)].copy()
test_df  = df[df.time_idx > val_end_idx].copy()

print("Train period :",
      train_df["date"].min().date(), "->", train_df["date"].max().date())
print("Val period   :",
      val_df["date"].min().date(), "->", val_df["date"].max().date())
print("Test period  :",
      test_df["date"].min().date(), "->", test_df["date"].max().date())

print("Train rows:", len(train_df))
print("Val rows  :", len(val_df))
print("Test rows :", len(test_df))


# =========================================
# 7. BUILD TimeSeriesDataSet (TRAIN / VAL / TEST)
# =========================================
max_encoder_len = 60   # look-back window
max_prediction_len = 1 # 1-step ahead forecast

# 7.1 training dataset
training = TimeSeriesDataSet(
    train_df,
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

    target_normalizer=GroupNormalizer(groups=["ticker"]),

    add_relative_time_idx=True,
    add_target_scales=True,
    allow_missing_timesteps=True,
    add_encoder_length=True,
)

# 7.2 validation & test reuse cấu hình từ training
validation = TimeSeriesDataSet.from_dataset(
    training,
    val_df,
    predict=False,
    stop_randomization=True,
)

test = TimeSeriesDataSet.from_dataset(
    training,
    test_df,
    predict=False,
    stop_randomization=True,
)

train_loader = training.to_dataloader(
    batch_size=128,
    shuffle=True,
    num_workers=0
)
val_loader = validation.to_dataloader(
    batch_size=256,
    shuffle=False,
    num_workers=0
)
test_loader = test.to_dataloader(
    batch_size=256,
    shuffle=False,
    num_workers=0
)

print(f"Number of training samples  : {len(training)}")
print(f"Number of validation samples: {len(validation)}")
print(f"Number of test samples      : {len(test)}")


# =========================================
# 8. TRAINER WITH EARLY STOPPING
# =========================================
pl.seed_everything(42, workers=True)

early_stop_cb = EarlyStopping(
    monitor="val_loss",
    patience=5,
    mode="min"
)

trainer = pl.Trainer(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    max_epochs=30,               # bạn có thể chỉnh tùy ý
    gradient_clip_val=0.1,
    logger=False,
    enable_progress_bar=True,
    callbacks=[early_stop_cb],
    log_every_n_steps=10,
)


# =========================================
# 9. BUILD TFT MODEL
# =========================================
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=3e-4,
    hidden_size=128,
    attention_head_size=4,
    dropout=0.2,
    reduce_on_plateau_patience=3,
)

print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")


# =========================================
# 10. TRAIN MODEL
# =========================================
trainer.fit(tft, train_loader, val_loader)
print("Training finished.")

best_val_loss = early_stop_cb.best_score.item()
print(f"[MODEL] Best val_loss : {best_val_loss:.6f}")

final_val_loss = trainer.callback_metrics["val_loss"].item()
print(f"[MODEL] Final val_loss: {final_val_loss:.6f}")


# =========================================
# 11. EVALUATE ON TEST SET
# =========================================
tft.to("cpu")  # đảm bảo inference trên CPU

# collect predictions & ground truth
y_true_list = []
y_pred_list = []

with torch.no_grad():
    for x, y in iter(test_loader):
        # y: (batch, max_prediction_len)
        y_true_list.append(y[:, 0].cpu())
        pred = tft(x.to("cpu")).squeeze(-1)  # output: (batch, 1) -> (batch,)
        y_pred_list.append(pred[:, 0].cpu())

y_true = torch.cat(y_true_list).numpy()
y_pred = torch.cat(y_pred_list).numpy()

test_mae = mean_absolute_error(y_true, y_pred)
print(f"[MODEL] Test MAE on target (next-day log-return): {test_mae:.6f}")


# =========================================
# 12. PREDICTION & INTERPRETATION (OPTIONAL, USING TEST SET)
# =========================================
predictions = tft.predict(
    test_loader,
    mode="raw",
    return_x=True,
    trainer_kwargs={"accelerator": "cpu"},
)

interpretation = tft.interpret_output(predictions.output, reduction="sum")

tft.plot_interpretation(interpretation)
plt.tight_layout()
plt.show()
