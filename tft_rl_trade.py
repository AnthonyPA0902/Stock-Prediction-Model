# All dependencies: `pip install torch torchrl pandas numpy matplotlib seaborn tqdm pytorch-forecasting==1.5.0`.

"""
End-to-end:  train TFT on daily quantamental panel → SAC agent learns position size
Single-name demo (AAPL) for brevity. Works with pytorch-forecasting 1.5.0
"""
import os, warnings, random, numpy as np, pandas as pd, yfinance as yf
import torch, torch.nn as nn
from tqdm import tqdm

warnings.filterwarnings("ignore"); torch.set_float32_matmul_precision("medium")

# ------------------------------------------------------------------ 0.  deps
# pip install torch torchrl pandas numpy matplotlib seaborn tqdm yfinance pytorch-forecasting==1.5.0
from pytorch_forecasting import TimeSeriesDataSet
from torchrl.envs import EnvBase, TransformedEnv, Compose, StepCounter, RewardScaling
from torchrl.data import BoundedTensorSpec
from torchrl.modules import ProbabilisticActor, TanhNormal
from torchrl.objectives import SACLoss
from torchrl.collectors import SyncDataCollector
from tensordict import TensorDict
from pytorch_forecasting.data import GroupNormalizer
import numpy as np
import torch
from torchrl.envs import EnvBase
from torchrl.data import BoundedTensorSpec, CompositeSpec
from tensordict import TensorDict

# ---- debug probe ----
import lightning.pytorch as pl, pytorch_forecasting as pf
from pytorch_forecasting.models import TemporalFusionTransformer
print("PF ver :", pf.__version__)
print("PL ver :", pl.__version__)
print("TFT MRO:", TemporalFusionTransformer.__mro__)
# ---------------------

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------ 1.  download daily data
TICKER = "AAPL"
df = yf.download(TICKER, start="2016-01-01", end="2024-01-01")[["Close"]]
df.columns = ["price"]
df["ret"] = df.price.pct_change()
df["log_ret"] = np.log1p(df.ret)

# -------------------- 2.  fake quantamental & macro -----------------
np.random.seed(SEED)
df["pe"]   = np.random.lognormal(2.2, 0.25, len(df))
df["roe"]  = np.random.normal(0.15, 0.04, len(df))
df["rsi"]  = df.ret.rolling(14).mean() / df.ret.rolling(14).std()
df["usd"]  = np.sin(np.arange(len(df)) * 2 * np.pi / 252)
df["vol"]  = df.ret.rolling(20).std()
df["spread"] = df.price * 0.0005   # 5 bp proxy
df = df.dropna().reset_index()
df["ticker"] = TICKER
df["time_idx"] = (df["Date"] - df["Date"].min()).dt.days
TARGET = "log_ret"
print("rows:", len(df))

# -------------------- 3.  build TimeSeriesDataSet -------------------
max_enc, max_pred = 60, 1
training_cutoff = df.time_idx.max() - 252   # last year for validation

training = TimeSeriesDataSet(
    df[df.time_idx <= training_cutoff],
    time_idx="time_idx",
    target=TARGET,
    group_ids=["ticker"],
    min_encoder_length=max_enc//2,
    max_encoder_length=max_enc,
    min_prediction_length=max_pred,
    max_prediction_length=max_pred,
    static_categoricals=["ticker"],
    time_varying_known_reals=["time_idx", "usd"],
    time_varying_unknown_reals=["price", "rsi", "pe", "roe", "ret", "vol"],
    target_normalizer=GroupNormalizer(groups=["ticker"]),
    add_relative_time_idx=True,
    add_target_scales=True,
    allow_missing_timesteps=True,
)
validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)
train_loader = training.to_dataloader(batch_size=128, shuffle=True)
val_loader   = validation.to_dataloader(batch_size=256, shuffle=False)

# -------------------- 4.  train TFT ---------------------------------
class TFTLightningModule(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = TemporalFusionTransformer(*args, **kwargs)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        return self.model.training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.model.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return self.model.configure_optimizers()

tft_model = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=1e-3,
    hidden_size=64,
    attention_head_size=2,
    dropout=0.1,
    reduce_on_plateau_patience=3,
)

print("is LM ?", isinstance(tft_model, pl.LightningModule))

trainer = pl.Trainer(
    max_epochs=1,
    accelerator=DEVICE,
    gradient_clip_val=0.1,
    enable_progress_bar=True,
    enable_model_summary=False,
)

trainer.fit(tft_model, train_loader, val_loader)

# -------------------- 5.  add TFT forecast & vol to df --------------
def add_tft_mu_sigma(df, tft, training_ds):
    df = df.copy()
    enc_len = training_ds.max_encoder_length

    # make TFT model eval-only
    tft.eval()

    # 1) build a full prediction dataset over the *entire* df
    predict_ds = TimeSeriesDataSet.from_dataset(
        training_ds,
        df,
        predict=True,
        stop_randomization=True,
    )

    # 2) get quantile predictions in one shot (no spammy progress bar)
    with torch.no_grad():
        q = tft.predict(
            predict_ds,
            mode="quantiles",
            trainer_kwargs=dict(
                accelerator="cpu",        # you don't have GPU
                logger=False,
                enable_progress_bar=False
            ),
        )

    # q shape: [n_windows, pred_len, n_quantiles]
    # here pred_len = 1 → squeeze dim 1 → [n_windows, n_quantiles]
    q = q.squeeze(1)

    # default TFT quantiles indices: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    q02_idx, q10_idx, q25_idx, q50_idx, q75_idx, q90_idx, q98_idx = range(7)

    median = q[:, q50_idx].cpu().numpy()
    spread = ((q[:, q90_idx] - q[:, q10_idx]) / 2.0).cpu().numpy()

    # 3) align predictions to df index
    # predict_ds starts making windows once we have enc_len history,
    # so number of windows should be len(df) - enc_len
    n_windows = len(median)
    start_idx = enc_len
    end_idx = start_idx + n_windows  # normally == len(df)

    df.loc[df.index[start_idx:end_idx], "mu"] = median
    df.loc[df.index[start_idx:end_idx], "sig"] = spread

    return df.dropna(subset=["mu", "sig"])

rl_df = add_tft_mu_sigma(df, tft_model, training)

# -------------------- 6.  TorchRL environment -----------------------
class TFTTradingEnv(EnvBase):
    def __init__(self, df, cost_bp=8, risk_pen=1e-4):
        super().__init__()
        self.df  = df.reset_index(drop=True)
        self.k   = cost_bp/1e4
        self.lam = risk_pen
        self._pos, self._t = 0.0, 0
        self.state_spec  = BoundedTensorSpec(low=-torch.inf, high=torch.inf, shape=(6,))
        self.action_spec = BoundedTensorSpec(low=-1.0, high=1.0, shape=(1,))

    def _reset(self, tensordict=None):
        self._t, self._pos = 0, 0.0
        return TensorDict({"observation": self._make_obs()}, batch_size=[])

    def _make_obs(self):
        row = self.df.iloc[self._t]
        obs = torch.tensor([row.mu, row.sig, self._pos,
                            row.ret, row.vol, row.spread], dtype=torch.float32)
        return obs

    def _step(self, tensordict):
        action = tensordict["action"].clip(-1, 1).item()
        row    = self.df.iloc[self._t]
        ret    = row.ret
        reward = self._pos * ret - self.k * abs(action - self._pos) \
                 - self.lam * (self._pos * row.vol)**2
        self._pos = action
        self._t  += 1
        done = self._t >= len(self.df)-1
        obs  = self._make_obs()
        return TensorDict({
            "observation": obs,
            "reward": torch.tensor(reward, dtype=torch.float32).view(1),
            "done": torch.tensor(done).view(1),
        }, batch_size=[])

    def _set_seed(self, seed): torch.manual_seed(seed)

# -------------------- 7.  SAC training ----------------------------
env = TFTTradingEnv(rl_df)
env = TransformedEnv(env, Compose(StepCounter(500), RewardScaling(scale=10.0)))

actor_net = nn.Sequential(nn.Linear(6,64), nn.Tanh(),
                          nn.Linear(64,64), nn.Tanh(),
                          nn.Linear(64,1)).to(DEVICE)
actor = ProbabilisticActor(module=actor_net, in_keys=["observation"],
                           distribution_class=TanhNormal, return_log_prob=True).to(DEVICE)

q1 = nn.Sequential(nn.Linear(6+1,64), nn.Tanh(),
                   nn.Linear(64,64), nn.Tanh(),
                   nn.Linear(64,1)).to(DEVICE)
q2 = nn.Sequential(nn.Linear(6+1,64), nn.Tanh(),
                   nn.Linear(64,64), nn.Tanh(),
                   nn.Linear(64,1)).to(DEVICE)

loss_mod = SACLoss(actor, q1, q2)
opt = torch.optim.Adam(loss_mod.parameters(), lr=3e-4)

collector = SyncDataCollector(env, policy=actor, frames_per_batch=2_000,
                              total_frames=100_000, device=DEVICE)

pbar = tqdm(collector)
for batch in pbar:
    loss_vals = loss_mod(batch)
    opt.zero_grad()
    loss_vals["loss"].backward()
    opt.step()
    pbar.set_postfix(loss=loss_vals["loss"].item())

# -------------------- 8.  evaluate -------------------------------
env.eval()
with torch.no_grad():
    roll = env.rollout(max_steps=len(rl_df)-1, policy=actor, auto_reset=True)
net_ret = roll["reward"].sum().item()
print("Net P&L over full history:", net_ret)

