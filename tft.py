# !pip install -q pytorch-forecasting 1.5.0
# 1.0+ works with torch≥2.0
# !pip install -q lightning pandas numpy yfinance matplotlib seaborn
import torch
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf

from sklearn.preprocessing import StandardScaler
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

# ---- debug probe ----
import lightning.pytorch as pl, pytorch_forecasting as pf
#from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
print("PF ver :", pf.__version__)
print("PL ver :", pl.__version__)
print("TFT MRO:", TemporalFusionTransformer.__mro__)
# ---------------------

tickers = ["AAPL","MSFT","GOOG","AMZN","TSLA","META","NVDA","JPM","V","JNJ"]
prices = yf.download(tickers, start="2018-01-01", end="2024-01-01")['Close']
prices = prices.stack().reset_index()
prices.columns = ["date","ticker","price"]

# create simple technical/fundamental placeholders
def add_quantamental(dF):
    dF = dF.sort_values(["ticker","date"])
    dF["day"]  = dF["date"].dt.dayofweek
    dF["ret"]  = dF.groupby("ticker")["price"].pct_change()
    dF["rsi"]  = dF.groupby("ticker")["ret"].transform(
                   lambda x: x.rolling(14).mean()/x.rolling(14).std())
    dF["pe"]   = np.random.lognormal(2.2,0.3,len(dF))   # <- replace with real
    dF["roe"]  = np.random.normal(0.15,0.05,len(dF))    #   fundamental feeds
    dF["usd"]  = np.sin(np.arange(len(dF))*2*np.pi/252) # dummy macro driver
    dF["time_idx"] = (dF["date"] - dF["date"].min()).dt.days
    return dF

df = add_quantamental(prices)

df["target"] = np.log1p(df["ret"]).shift(-1)   # next-day log-return
df = df.dropna()

static_cats   = ["ticker"]
time_varying_known_reals   = ["day","usd"]
time_varying_unknown_reals = ["price","rsi","pe","roe","ret"]

scalers = {col: StandardScaler() for col in time_varying_unknown_reals+["target"]}
for col in scalers:
    df[col] = scalers[col].fit_transform(df[[col]]).flatten()
    
max_encoder_len   = 60   # look-back window
max_prediction_len = 1   # 1-step ahead

training_cutoff = df["time_idx"].max() - 30  # keep last 30 days for validation

dataset = TimeSeriesDataSet(
    df[df.time_idx <= training_cutoff],
    time_idx        = "time_idx",
    target          = "target",
    group_ids       = ["ticker"],
    min_encoder_length = max_encoder_len//2,
    max_encoder_length = max_encoder_len,
    min_prediction_length = max_prediction_len,
    max_prediction_length = max_prediction_len,
    static_categoricals   = static_cats,
    time_varying_known_reals   = time_varying_known_reals,
    time_varying_unknown_reals = time_varying_unknown_reals,
    target_normalizer = GroupNormalizer(groups=["ticker"]),
    add_relative_time_idx = True,
    add_target_scales     = True,
    allow_missing_timesteps = True,
    add_encoder_length = True
)

val_dataset = TimeSeriesDataSet.from_dataset(dataset, df, predict=True, stop_randomization=True)
train_loader = dataset.to_dataloader(batch_size=128, shuffle=True)
val_loader   = val_dataset.to_dataloader(batch_size=256, shuffle=False)


pl.seed_everything(42)
trainer = pl.Trainer(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    max_epochs=1,
    gradient_clip_val=0.1,
    logger=False,
    enable_progress_bar=True,
)

tft = TemporalFusionTransformer.from_dataset(
    dataset,
    learning_rate=1e-3,
    hidden_size=64,
    attention_head_size=4,
    dropout=0.1,
    #logging_metrics=["MAE"],
    reduce_on_plateau_patience=3,
)

trainer.fit(tft, train_loader, val_loader)

predictions = tft.predict(
    val_loader, mode="raw",return_x=True, trainer_kwargs=dict(accelerator="cpu")
)
#predictions_vs_actuals = tft.calculate_prediction_actual_by_variable(
#    predictions.x, predictions.output
#)
#tft.plot_prediction_actual_by_variable(predictions_vs_actual)

interpretation = tft.interpret_output(predictions.output, reduction="sum")
tft.plot_interpretation(interpretation)
plt.show()