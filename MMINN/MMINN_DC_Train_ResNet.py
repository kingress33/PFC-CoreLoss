# %% [markdown]
# # Import Package & Hyperparameter Configuration

# %%
# 清空所有變數
%reset -f
# # 強制 Python 回收記憶體
# import gc
# gc.collect()

# %% [markdown]
# ## Package
# 

# %%
import os
import torch
import numpy as np
import random
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
from datetime import datetime
import json
import pandas as pd
from pathlib import Path  #python用

try:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    print("Notebook 環境，跳過切換目錄")

# %% [markdown]
# ## Hyperparameter Config

# %%
# %%
# Unified Hyperparameter Configuration
class Config:
    SEED = 1
    NUM_EPOCHS_PHASE1 = 1000
    NUM_EPOCHS_PHASE2 = 1000

    BATCH_SIZE = 1024
    LEARNING_RATE = 0.002

    # --- Scheduler 相關參數  ---
    LR_DECAY_EPOCH = 200
    LR_DECAY_RATIO = 0.5
    # LR_SCHEDULER_GAMMA = 0.99 # Step Decay
    # --- Early Stopping ---
    EARLY_STOPPING_PATIENCE = 100
    # --- Model Hyperparameters ---
    HIDDEN_SIZE = 30
    OPERATOR_SIZE = 30
    MAXOUT_H = 1

    PLOT_INTERVAL = 100

    DOWNSAMPLE = 1024  # 波形降階點數


# Reproducibility
random.seed(Config.SEED)
np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# %% [markdown]
# ## Material & Number of Data

# %%
material = "CH467160"
fix_way = "MMINN_AC_ResNet_DC"
note = "bulid_model"
note_detail = "Dataset改成倒三角形(滿足PFC假設)、建立AC和DC的DataLoader、準備建立模型"
save_figure = True
timestamp = datetime.now().strftime("%Y%m%d")

result_dir = os.path.join("results",
                          f"{timestamp}_{fix_way}_{material}_{note}")
os.makedirs(result_dir, exist_ok=True)

# 定義保存模型的路徑
model_save_path = os.path.join(
    result_dir, f"{material}_{fix_way}_{note}_{timestamp}.pt")  # 定義模型保存檔名

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# # Data processing and data loader generate 

# %%
# %% Preprocess data into a data loader
def get_dataloader(data_B,
                   data_F,
                   data_T,
                   data_H,
                   data_N,
                   data_Hdc,
                   data_Duty_P,
                   data_Duty_N,
                   data_Pcv,
                   global_B_max,
                   global_H_max,
                   n_init=16,
                   norm=None):

    # Data pre-process

    # ── 0. 全域設定/降階設定 ──────────────────────────────
    eps = 1e-8  # 防止除以 0
    if Config.DOWNSAMPLE == 1024:
        seq_length = 1024  # 單筆波形點數 (不再 down-sample)
    else:
        seq_length = Config.DOWNSAMPLE
        cols = np.linspace(0, 1023, seq_length, dtype=int)
        data_B = data_B[:, cols]
        data_H = data_H[:, cols]

    # ── 1. 波形拼接 (補 n_init 點作初始磁化) ────
    data_length = seq_length + n_init
    data_B = np.hstack((data_B[:, -n_init:], data_B))  # (batch, data_length)
    data_H = np.hstack((data_H[:, -n_init:], data_H))

    print("B shape:", data_B.shape)
    print("H shape:", data_H.shape)
    print("F shape:", data_F.shape)
    print("T shape:", data_T.shape)
    print("Hdc shape:", data_Hdc.shape)
    print("N shape:", data_N.shape)
    print("Duty Pos shape:", data_Duty_P.shape)
    print("Duty Neg shape:", data_Duty_N.shape)
    print("Pcv shape:", data_Pcv.shape)

    # ── 2. 轉成 Tensor ───────────────────────────
    B = torch.from_numpy(data_B).view(-1, data_length, 1).float()  # (B,N,1)
    H = torch.from_numpy(data_H).view(-1, data_length, 1).float()
    F = torch.log10(torch.from_numpy(data_F).view(-1, 1).float())  # 純量
    T = torch.from_numpy(data_T).view(-1, 1).float()
    Hdc = torch.from_numpy(data_Hdc).view(-1, 1).float()
    N = torch.from_numpy(data_N).view(-1, 1).float()
    Duty_P = torch.from_numpy(data_Duty_P).view(-1, 1).float()
    Duty_N = torch.from_numpy(data_Duty_N).view(-1, 1).float()
    Pcv = torch.log10(torch.from_numpy(data_Pcv).view(-1, 1).float())

    # ── 3. 先計算導數，再除以 scale_B ─────────────
    dB = torch.diff(B, dim=1, prepend=B[:, :1])
    dB_dt = dB * (seq_length * F.view(-1, 1, 1))  # 真實斜率

    # ── 4. 計算二階導數 ─────────────────────────
    in_B = B / global_B_max
    out_H = H / global_H_max
    in_dB_dt = dB_dt / global_B_max

    # ── 5. 純量特徵：計算 z-score 參數 ─────────────
    # Data Normalization (套用 norm)
    in_F = (F - norm[0][0]) / norm[0][1]
    in_T = (T - norm[1][0]) / norm[1][1]
    in_Hdc = (Hdc - norm[2][0]) / norm[2][1]
    in_N = (N - norm[3][0]) / norm[3][1]
    in_Pcv = (Pcv - norm[4][0]) / norm[4][1]
    in_Duty_P = Duty_P
    in_Duty_N = Duty_N

    # ── 6. 產生初始 Preisach operator 狀態 s0 ──────
    max_B, _ = torch.max(in_B, dim=1)
    min_B, _ = torch.min(in_B, dim=1)
    # s0 = get_operator_init(in_B[:, 0] - dB[:, 0] / scale_B.squeeze(-1),
    #                        dB / scale_B, max_B, min_B)
    s0 = get_operator_init(in_B[:, 0] - dB[:, 0] / global_B_max,
                           dB / global_B_max, max_B, min_B)

    # ── 7. 組合 Dataset ───────────────────────────

    wave_inputs = torch.cat(
        (
            in_B,  # ① B
            dB / global_B_max,  # ② ΔB
            in_dB_dt  # ③ dB/dt
        ),
        dim=2)  #

    aux_features = torch.cat((in_F, in_T, in_Hdc, in_N, in_Duty_P, in_Duty_N),
                             dim=1)  # (B,4)

    amp_B = torch.full((len(B), 1), global_B_max, dtype=torch.float32)
    amp_H = torch.full((len(B), 1), global_H_max, dtype=torch.float32)
    amps = torch.cat((amp_B, amp_H), dim=1)  # 仍給 RNN2 用

    # 這裡把 Pcv（已 z-score）單獨拿出來當另一個 label
    target_Pcv = in_Pcv  # (B,1)

    full_dataset = torch.utils.data.TensorDataset(
        wave_inputs,  # 0  → 模型序列輸入
        aux_features,  # 1  → 4 個純量
        amps,  # 2  → 幅值係數
        s0,  # 3  → Preisach 初始狀態
        out_H,  # 4  → 目標 H  (已 scale_H)
        target_Pcv)  # 5  → 目標 Pcv (已 z-score)

    # ── 8. Train / Valid split & DataLoader ───────
    train_size = int(0.8 * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    train_set, valid_set = torch.utils.data.random_split(
        full_dataset, [train_size, valid_size],
        generator=torch.Generator().manual_seed(Config.SEED))

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=Config.BATCH_SIZE,
                                               shuffle=True,
                                               num_workers=0,
                                               pin_memory=True,
                                               collate_fn=filter_input)

    valid_loader = torch.utils.data.DataLoader(valid_set,
                                               batch_size=Config.BATCH_SIZE,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               collate_fn=filter_input)

    return train_loader, valid_loader, norm


# %% Predict the operator state at t0
def get_operator_init(B1,
                      dB,
                      Bmax,
                      Bmin,
                      max_out_H=Config.MAXOUT_H,
                      operator_size=Config.OPERATOR_SIZE):
    """Compute the initial state of hysteresis operators"""
    s0 = torch.zeros((dB.shape[0], operator_size))
    operator_thre = torch.from_numpy(
        np.linspace(max_out_H / operator_size, max_out_H,
                    operator_size)).view(1, -1)

    for i in range(dB.shape[0]):
        for j in range(operator_size):
            r = operator_thre[0, j]
            if (Bmax[i] >= r) or (Bmin[i] <= -r):
                if dB[i, 0] >= 0:
                    if B1[i] > Bmin[i] + 2 * r:
                        s0[i, j] = r
                    else:
                        s0[i, j] = B1[i] - (r + Bmin[i])
                else:
                    if B1[i] < Bmax[i] - 2 * r:
                        s0[i, j] = -r
                    else:
                        s0[i, j] = B1[i] + (r - Bmax[i])
    return s0


def filter_input(batch):
    inputs, features, amps, s0, target_H, target_Pcv = zip(*batch)

    inputs = torch.stack(inputs)
    features = torch.stack(features)
    amps = torch.stack(amps)
    s0 = torch.stack(s0)
    target_H = torch.stack(target_H)[:, -Config.DOWNSAMPLE:, :]  # 保留全長
    target_Pcv = torch.stack(target_Pcv)  # (B,1)

    return inputs, features, amps, s0, target_H, target_Pcv


# 溫度頻率不變加入微小的 epsilon
def safe_mean_std(tensor, eps=1e-8):
    m_tensor = torch.mean(tensor)  # 還是 Tensor
    s_tensor = torch.std(tensor)  # 還是 Tensor

    m_val = m_tensor.item()  # 第一次轉成 float
    s_val = s_tensor.item()
    if s_val < eps:
        s_val = 1.0
    return [m_val, s_val]  # 直接回傳 float


# %% [markdown]
# # Define Network Structure

# %%
# %% Magnetization mechansim-determined neural network
"""
    Total Parameters:
    - x: (batch, seq, 3) -> 1.B, 2.dB, 3.dB/dt
    - var: (batch, 6) -> 包含 F, T, Hdc, N... 
    
    MMINN(AC) Parameters:
    - hidden_size: number of eddy current slices (RNN neuron)
    - operator_size: number of operators
    - input_size: number of inputs (1.B 2.dB 3.dB/dt)
    - var_size: number of supplenmentary variables (1.F 2.T)        
    - output_size: number of outputs (1.H)
    
    ResNet Parameters:
    - 
    
"""


class MMINet(nn.Module):

    def __init__(self,
                 norm,
                 hidden_size=Config.HIDDEN_SIZE,
                 operator_size=Config.OPERATOR_SIZE,
                 input_size=3,
                 var_size=2,
                 output_size=1):
        super().__init__()
        self.input_size = input_size
        self.var_size = var_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.operator_size = operator_size
        self.norm = norm

        self.rnn1 = StopOperatorCell(self.operator_size)
        self.dnn1 = nn.Linear(self.operator_size + self.var_size, 1)
        # var_size (F, T) + 3 (B, dB/dt)
        self.rnn2 = EddyCell(var_size + 2, self.hidden_size, output_size)
        self.dnn2 = nn.Linear(self.hidden_size, 1)
        self.rnn2_hx = None

    def forward(self, x, var, amps, s0, n_init=16):
        """
        Parameters: 
        - x(batch,seq,input_size): Input features (1.B, 2.dB, 3.dB/dt)  
        - var(batch,var_size): Supplementary inputs (1.F 2.T) (取前2個，其餘未使用)
        - s0(batch,1): Operator inital states
        """
        batch_size = x.size(0)  # Batch size
        seq_size = x.size(1)  # Series length
        self.rnn1_hx = s0

        var_mminn = var[:, 0:2]  # Shape: (Batch, 2)

        # Initialize DNN2 input (1.B 3.dB/dt)
        x2 = torch.cat((x[:, :, 0:1], x[:, :, 2:3]), dim=2)

        for t in range(seq_size):
            # RNN1 input (dB,state)
            self.rnn1_hx = self.rnn1(x[:, t, 1:2], self.rnn1_hx)

            # DNN1 input (rnn1_hx,F,T)
            # dnn1_in = torch.cat((self.rnn1_hx, var), dim=1)
            dnn1_in = torch.cat((self.rnn1_hx, var_mminn), dim=1)

            # H hysteresis prediction
            H_hyst_pred = self.dnn1(dnn1_in)

            # DNN2 input (B,dB/dt,T,F)
            # rnn2_in = torch.cat((x2[:, t, :], var), dim=1)
            rnn2_in = torch.cat((x2[:, t, :], var_mminn), dim=1)

            # Initialize second rnn state
            if t == 0:
                H_eddy_init = x[:, t, 0:1] - H_hyst_pred
                buffer = x.new_ones(x.size(0), self.hidden_size)
                self.rnn2_hx = Variable(
                    (buffer / torch.sum(self.dnn2.weight, dim=1)) *
                    H_eddy_init)

            #rnn2_in = torch.cat((rnn2_in,H_hyst_pred),dim=1)
            self.rnn2_hx = self.rnn2(rnn2_in, self.rnn2_hx)

            # H eddy prediction
            H_eddy = self.dnn2(self.rnn2_hx)

            # H total
            H_total = (H_hyst_pred + H_eddy).view(batch_size, 1,
                                                  self.output_size)
            if t == 0:
                output = H_total
            else:
                output = torch.cat((output, H_total), dim=1)

        H = (output[:, n_init:, :])

        return H


class StopOperatorCell():

    def __init__(self, operator_size):
        self.operator_thre = torch.from_numpy(
            np.linspace(Config.MAXOUT_H / operator_size, Config.MAXOUT_H,
                        operator_size)).view(1, -1)

    def sslu(self, X):
        a = torch.ones_like(X)
        return torch.max(-a, torch.min(a, X))

    def __call__(self, dB, state):
        r = self.operator_thre.to(dB.device)
        output = self.sslu((dB + state) / r) * r
        return output.float()


class EddyCell(nn.Module):

    def __init__(self, input_size, hidden_size, output_size=1):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.x2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, hidden=None):
        hidden = self.x2h(x) + self.h2h(hidden)
        hidden = torch.sigmoid(hidden)
        return hidden


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class HARDCORECorrectionCNN(nn.Module):

    def __init__(self, scalar_dim, hidden_dim=16):
        # 論文建議參數極少，hidden_dim 設 16 或 8 效果最好且最快
        super().__init__()

        input_dim = 1 + scalar_dim

        self.net = nn.Sequential(
            # --- 第 1 層：捕捉特徵 ---
            # padding_mode='circular' 是靈魂！解決週期信號邊界問題
            # kernel_size=9 是論文推薦的甜蜜點
            nn.Conv1d(input_dim,
                      hidden_dim,
                      kernel_size=9,
                      padding=4,
                      padding_mode='circular'),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),

            # --- 第 2 層：擴大視野 (Context) ---
            # dilation=4, kernel_size=9
            # padding 計算: (kernel - 1) * dilation / 2 = (9-1)*4/2 = 16
            nn.Conv1d(hidden_dim,
                      hidden_dim,
                      kernel_size=9,
                      padding=16,
                      dilation=4,
                      padding_mode='circular'),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),

            # --- 第 3 層：再擴大 (Global Offset) ---
            # 這一層專門用來看整體的 DC 偏移
            # padding = (9-1)*8/2 = 32 (假設 dilation 加倍)
            nn.Conv1d(hidden_dim,
                      hidden_dim,
                      kernel_size=9,
                      padding=32,
                      dilation=8,
                      padding_mode='circular'),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),

            # --- 輸出層 ---
            nn.Conv1d(hidden_dim, 1, kernel_size=1))

    def forward(self, h_ac, scalars):
        # h_ac: (Batch, Seq, 1)
        # scalars: (Batch, Dim)

        batch, seq, _ = h_ac.shape
        x = h_ac.permute(0, 2, 1)  # (B, 1, Seq)

        # 擴充純量並拼接
        scalars_expand = scalars.unsqueeze(2).expand(-1, -1, seq)
        x = torch.cat([x, scalars_expand], dim=1)  # (B, 1+Dim, Seq)

        # 卷積運算
        out = self.net(x)

        return out.permute(0, 2, 1)  # (B, Seq, 1)


class HybridModel(nn.Module):

    def __init__(self, norm, config):
        super().__init__()

        # 1. 物理層 (MMINN) - 負責預測基礎 AC 波形
        # 注意：MMINet 內部會自己切片只取前 2 個特徵 (F, T)
        self.mminn = MMINet(
            norm=norm,
            hidden_size=config.HIDDEN_SIZE,
            operator_size=config.OPERATOR_SIZE,
            input_size=3,  # B, dB, dB/dt
            var_size=2,  # 這裡設定 2 沒錯，對應 MMINet 的結構
            output_size=1)

        # 2. 修正層 (CNN) - 負責預測 DC 造成的殘差
        # var 對應: 0:F, 1:T, 2:Hdc, 3:N, 4:DutyP, 5:DutyN
        # 定義 CNN 要看 var 中的欄位: F, T, Hdc (前三個)

        self.resnet_feature_indices = [0, 1, 2]
        scalar_dim = len(self.resnet_feature_indices)

        # self.resnet = SimpleCorrectionCNN(
        #     scalar_dim=scalar_dim,
        #     hidden_dim=32  # 這是 CNN 內部的隱藏層寬度
        # )

        self.resnet = HARDCORECorrectionCNN(scalar_dim=scalar_dim,
                                            hidden_dim=16)  # 輕量化，訓練速度會飛快

    def forward(self, x, var, amps, s0):
        # 1. MMINN 預測 H_ac (它內部會只看 F, T)
        H_ac = self.mminn(x, var, amps, s0)

        # 2. 準備 CNN 的輸入
        # 根據 indices 挑選變數 (挑 F, T, Hdc)
        var_resnet = var[:, self.resnet_feature_indices]

        # 3. CNN 預測修正量 H_dc_correction
        # 輸入: MMINN 算出來的 H_ac 波形 + 環境參數
        H_dc_correction = self.resnet(H_ac, var_resnet)

        # 4. 最終合成: AC 基底 + DC 修正
        H_total = H_ac + H_dc_correction

        return H_total, H_ac

    # --- 參數凍結工具 (Two-Stage Training ) ---
    def freeze_mminn(self):
        for param in self.mminn.parameters():
            param.requires_grad = False
        print("MMINN 參數已凍結 (Fixing Physics Layer)")

    def unfreeze_mminn(self):
        for param in self.mminn.parameters():
            param.requires_grad = True
        print("MMINN 參數已解凍 (Training Physics Layer)")

    def freeze_resnet(self):
        for param in self.resnet.parameters():
            param.requires_grad = False
        print("ResNet/CNN 參數已凍結 (Fixing Correction Layer)")

    def unfreeze_resnet(self):
        for param in self.resnet.parameters():
            param.requires_grad = True
        print("ResNet/CNN 參數已解凍 (Training Correction Layer)")


# %% [markdown]
# # Training the Model

# %% [markdown]
# ## Load Dataset

# %%
# %%
def load_dataset(material, base_path="./Data/"):

    in_file1 = f"{base_path}{material}/train/B_Field.csv"
    in_file2 = f"{base_path}{material}/train/Frequency.csv"
    in_file3 = f"{base_path}{material}/train/Temperature.csv"
    in_file4 = f"{base_path}{material}/train/H_Field.csv"
    in_file5 = f"{base_path}{material}/train/Volumetric_Loss.csv"
    in_file6 = f"{base_path}{material}/train/Hdc.csv"
    in_file7 = f"{base_path}{material}/train/Turns.csv"
    in_file8 = f"{base_path}{material}/train/Duty_P.csv"
    in_file9 = f"{base_path}{material}/train/Duty_N.csv"

    data_B = np.genfromtxt(in_file1, delimiter=',')  # N x 1024
    data_F = np.genfromtxt(in_file2, delimiter=',')  # N x 1
    data_T = np.genfromtxt(in_file3, delimiter=',')  # N x 1
    data_H = np.genfromtxt(in_file4, delimiter=',')  # N x 1024
    data_Pcv = np.genfromtxt(in_file5, delimiter=',')  # N x 1
    data_Hdc = np.genfromtxt(in_file6, delimiter=',')  # N x 1
    data_N = np.genfromtxt(in_file7, delimiter=',')  # N x 1
    data_Duty_P = np.genfromtxt(in_file8, delimiter=',')  # N x 1
    data_Duty_N = np.genfromtxt(in_file9, delimiter=',')  # N x 1

    return data_B, data_F, data_T, data_H, data_Pcv, data_Hdc, data_N, data_Duty_P, data_Duty_N


# %% [markdown]
# ## Train Logger

# %%
class TrainLogger:

    def __init__(self, exp_name, config_dict, result_dir):
        self.exp_name = exp_name
        self.result_dir = result_dir
        self.config = config_dict
        os.makedirs(self.result_dir, exist_ok=True)

        self._save_config()
        self._write_metadata()

    def _save_config(self):
        with open(os.path.join(self.result_dir, "config.json"), "w") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

    def _write_metadata(self):
        metadata = {
            "experiment_name": self.exp_name,
            "timestamp": datetime.now().isoformat()
        }
        with open(os.path.join(self.result_dir, "meta.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    def save_norm_params(self, norm, feature_names=["F", "T", "Hdc", "Pcv"]):
        """
        將標準化參數存成：
        {
          "CH467160": [
             [mean_F, std_F],
             [mean_T, std_T],
             [mean_Hdc, std_Hdc],
             [mean_N, std_N],
             [mean_Pcv, std_Pcv],
          ]
        }
        """
        # 從 exp_name 前半段取出 material
        material_key = self.exp_name.split('_')[0]

        # 直接把 norm (list of [mean, std]) 當成 value
        output = {material_key: norm}

        # 寫檔
        with open(os.path.join(self.result_dir, "norm_params.json"), "w") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)
        print(
            f"✅ Normalization parameters saved to {os.path.join(self.result_dir, 'norm_params.json')}"
        )

    def save_summary(self, best_epoch, best_val_loss, best_loss_H,
                     best_loss_Pcv, model_save_path, elapsed):
        summary = {
            "exp_name": self.exp_name,
            "timestamp": datetime.now().isoformat(),
            "duration_sec": elapsed,
            "config": self.config,
            "best_model": {
                "path": model_save_path,
                "epoch": best_epoch,
                "val_loss": best_val_loss,
                "loss_H": best_loss_H,
                "loss_Pcv": best_loss_Pcv
            },
            "note": note,
            "note detail": note_detail
        }
        with open(os.path.join(self.result_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

# %% [markdown]
# ## Train Code
# 

# %% [markdown]
# ### Tools

# %%
def clamp_learning_rate(optimizer, min_lr=1e-5):
    for param_group in optimizer.param_groups:
        if param_group['lr'] < min_lr:
            param_group['lr'] = min_lr


def calculate_nrmse(y_pred, y_true, eps=1e-9):
    """
    計算 H-field 的歸一化均方根誤差 (Normalized Root Mean Square Error)。
    這個指標用來評估波形「形狀」的相似度，數值越低越好。
    """
    # y_pred, y_true 的 shape 都是 (batch, seq_len, 1)
    error = torch.sqrt(torch.mean((y_pred - y_true)**2, dim=1))  # (batch, 1)
    norm = torch.sqrt(torch.mean(y_true**2, dim=1))  # (batch, 1)

    # 計算平均 NRMSE 並轉為百分比
    return torch.mean(error / (norm + eps)).item() * 100


def calculate_mape(y_pred, y_true, norm_params, eps=1e-9):
    """
    計算 Pcv 的平均絕對百分比誤差 (Mean Absolute Percentage Error)。
    這個指標直接反映了損耗預測值的「百分比誤差」，數值越低越好。
    """
    # y_pred, y_true 的 shape 都是 (batch, 1)，並且是經過 log10 和 z-score 處理的
    # 步驟 1: 將 z-score 還原成 log10(Pcv)
    # norm_params[4] 是 Pcv 的 [mean, std]
    pred_log = y_pred * norm_params[4][1] + norm_params[4][0]
    true_log = y_true * norm_params[4][1] + norm_params[4][0]

    # 步驟 2: 將 log10(Pcv) 還原成真實的 Pcv
    pred_real = 10**pred_log
    true_real = 10**true_log

    # 步驟 3: 計算 MAPE 並轉為百分比
    return torch.mean(torch.abs(
        (pred_real - true_real) / (true_real + eps))).item() * 100


# 輔助繪圖函數
def plot_comparison(model,
                    loader,
                    epoch,
                    phase_name,
                    save_dir,
                    device,
                    num_samples=2):
    model.eval()

    # 抓一個 Batch 出來畫
    with torch.no_grad():
        inputs, features, amps, s0, target_H, _ = next(iter(loader))
        inputs, features, amps, s0, target_H = inputs.to(device), features.to(
            device), amps.to(device), s0.to(device), target_H.to(device)

        # 根據階段決定輸出
        if "Phase1" in phase_name:
            _, pred_H = model(inputs, features, amps, s0)  # 取 H_ac
        else:
            pred_H, _ = model(inputs, features, amps, s0)  # 取 H_total

    # 轉回 Numpy
    pred_H_np = pred_H.cpu().numpy()  # (Batch, 1024, 1)
    target_H_np = target_H.cpu().numpy()  # (Batch, 1024, 1)

    # inputs 的 B 是 1040 點，要切成跟 H 一樣的長度 (1024))
    seq_len = pred_H_np.shape[1]

    # 取 inputs 的第 0 個通道 (B)，並且只取最後 seq_len 點
    B_np = inputs[:, -seq_len:, 0].cpu().numpy()  # (Batch, 1024)

    # --- 圖 1: H Waveform 比較 ---
    plt.figure(figsize=(12, 5))

    # 避免 batch size 小於 num_samples (例如只剩 1 筆)
    actual_samples = min(num_samples, inputs.size(0))

    for i in range(actual_samples):
        plt.subplot(1, 2, 1)
        plt.plot(target_H_np[i, :, 0],
                 'k-',
                 alpha=0.6,
                 label='Target' if i == 0 else "")
        plt.plot(pred_H_np[i, :, 0], '--', label='Pred' if i == 0 else "")
        plt.title(f"H Waveform (Ep {epoch})")
        plt.xlabel("Time Step")
        plt.ylabel("H (Normalized)")
        if i == 0: plt.legend()
        plt.grid(True, alpha=0.3)

        # --- 圖 2: B-H Loop 比較 ---
        plt.subplot(1, 2, 2)
        # 現在 X 軸 (H) 和 Y 軸 (B) 長度都是 1024 了，不會報錯
        plt.plot(target_H_np[i, :, 0],
                 B_np[i, :],
                 'k-',
                 alpha=0.6,
                 label='Target' if i == 0 else "")
        plt.plot(pred_H_np[i, :, 0],
                 B_np[i, :],
                 '--',
                 label='Pred' if i == 0 else "")
        plt.title(f"B-H Loop (Ep {epoch})")
        plt.xlabel("H (Normalized)")
        plt.ylabel("B (Normalized)")
        if i == 0: plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    # 存檔
    save_path = os.path.join(save_dir, f"{phase_name}_Ep{epoch}_Compare.svg")
    plt.savefig(save_path)
    plt.show()
    plt.close()  # 關閉圖表釋放記憶體
    print(f"[Plot] Saved comparison figure to {save_path}")


def get_time_str(start):
    elapsed = time.perf_counter() - start
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    return f"{mins}m {secs}s"


# %% [markdown]
# ### Main train 

# %%
def train_model(norm, train_loader_AC, valid_loader_AC, train_loader_DC,
                valid_loader_DC, logger):

    print("=== Start Train  ===")
    print(r"""⠀⠀⠀

        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣾⣷⣄⠀⠀⠀⣀⣤⣤⣤⡀⠀⠀⠀⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣶⠏⠀⠀⣿⠀⢀⡾⠛⠋⠀⣾⣿⠀⠀⠀⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡀⡏⠀⠀⠀⣿⢀⣾⠁⠀⣰⠆⢹⡿⠀⠀⠀⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠃⣧⠀⠀⢠⡟⢸⡇⠀⣰⠟⠀⣼⠃⠀⠀⠀⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣹⣆⢀⣸⣇⣸⠃⢠⡏⠀⣸⠋⠀⠀⠀⠀⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⣤⣴⣶⣶⣶⠾⠟⠛⠉⠉⠉⠈⠉⠉⠛⠁⢾⠁⣴⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣤⣤⣶⣶⠾⠟⠛⠛⣻⣿⣙⡁⠀⠀⢾⣶⣾⣷⣿⣶⣄⠀⠀⠀⠀⠰⢿⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⢀⣀⣀⣀⣠⣴⣶⣶⠾⠟⠛⠉⠉⠉⠀⠀⠀⠀⠀⣿⣻⣟⣻⣿⡦⠀⠘⣿⣿⣛⡿⢶⡇⠀⠀⠀⠀⠀⠀⢻⣆⠀⠀⠀⠀⠀⠀⠀⠀
        ⣠⣶⣶⣶⣾⣿⣿⣿⣿⣿⣿⣿⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡟⠙⣿⣿⡗⠀⠀⠿⠉⣿⣿⣿⣶⠀⠀⠀⠀⠀⠀⠈⢿⠀⠀⠀⠀⠀⠀⠀⠀
        ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⠳⣄⣿⡿⠁⠀⠀⠘⢦⣿⣿⠇⠟⠁⠀⠀⠀⠀⠀⠀⣸⡇⠀⠀⠀⠀⠀⠀⠀
        ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠁⡇⠀⠀⠀⠀⠀⠀⠀
        ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⣇⠀⠀⠀⠀⠀⠀⠀
        ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⣿⠀⠀⠀⠀⠀⠀⠀
        ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡏⠀⠀⠀⠀⠀⠀⠀
        ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀
        ⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣇⡇⠀⠀⠀⠀⠀⠀⠀
        ⠀⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢤⣤⡀⠀⠀⠀⠀⠀⠀⠀⣿⡇⠀⠀⠀⠀⠀⠀⠀
        ⠀⠀⠈⠙⢿⣿⣿⣿⣿⣿⣿⠟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣤⡾⠟⠛⠆⠀⠀⠀⠀⠀⢀⢻⡇⠀⠀⠀⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠈⠙⠿⣿⣭⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣤⣴⣶⠾⠟⠋⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣾⠇⠀⠀⠀⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠙⠛⠷⠶⢶⣶⣦⣤⣴⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣌⣿⠀⠀⠀⠀⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⣿⡄⠀⠀⠀⠀⠀⠀⠀⠀⠙⠛⠛⠛⠃⠀⠀⠀⠀⠀⠀⠀⣤⣴⣾⣿⣿⣿⣓⠀⠀⠀⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣼⣿⣷⣦⣄⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⣠⣤⣶⣾⣟⣯⣽⠟⠋⠀⠉⠳⣄⠀⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣾⢇⠀⠉⠛⠷⣮⣍⣩⡍⢻⡟⠉⣉⢹⡏⠉⣿⣹⣷⣦⣿⠿⠟⠉⠀⠀⠀⠀⠀⠀⠙⣆⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⠏⢸⠇⠀⠀⠀⠀⠀⠉⠉⠛⠛⠛⠛⠛⠛⠛⠋⠉⠉⠀⠀⠀⠀⠀⢠⣠⡶⠀⠀⠀⠀⠘⣧⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⡿⠀⣸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠟⠁⠀⠀⠀⠀⠀⠘⣆⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣾⠃⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣾⣇⡀⠀⠀⠀⠀⠀⠀⢹⡆⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣾⠀⣾⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢿⣥⢠⣤⠼⠇⠀⠀⠘⣿⡄
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⣽⡄⠈⢿⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣿⠿⠾⠷⠄⠀⠀⠀⢀⣿⠁
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣧⠀⠸⣷⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣤⣾⠋⠀⠀⠀⠀⠀⠀⢰⣾⡿⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⣦⣠⣿⣿⣶⣶⣤⣤⣄⣀⣀⣀⣀⠀⠀⠀⠀⠀⠀⠀⣀⣀⣠⣴⣿⣇⠀⠀⠀⠀⠀⠀⠀⣸⡟⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢻⣿⠀⠉⠛⢿⣿⣯⣿⡟⢿⠻⣿⢻⣿⢿⣿⣿⣿⣿⣿⠿⠟⠹⣟⢷⣄⠀⠀⠀⢀⣼⠟⠀⠀⠀             一切會順利的!
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣿⣄⠀⠀⠘⢷⣌⡻⠿⣿⣛⣿⣟⣛⣛⣋⣉⣉⣉⣀⡀⠀⠀⠈⠻⢿⣷⣶⣶⢛⣧⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⣏⠀⠀⠀⠀⠹⢯⣟⣛⢿⣿⣽⣅⣀⡀⠀⣀⡀⠀⠀⠀⠠⢦⣀⠰⡦⠀⢸⠀⣏⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢿⡀⠀⠀⠀⠀⠀⠀⠈⠉⢻⣿⡟⠛⠉⠉⠁⠀⠀⠀⠀⠀⠀⠈⠛⠷⠀⣸⠀⣿⡀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣧⠀⠀⠀⠀⠀⠀⠀⠀⠘⣿⣇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠀⣿⡇⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⢿⠀⠀⢦⡀⡀⠀⠀⠀⠀⢹⣿⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⡄⡏⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡄⠀⠈⠳⣝⠦⢄⠀⠀⠀⣟⣷⠀⠀⠀⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⣿⡇⡇⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣄⣷⡀⠀⠀⠈⠙⠂⠀⠀⠀⢸⣿⡄⠀⠀⠘⢦⡙⢦⡀⠀⠀⠀⠀⢰⣷⣷⡇⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⡿⢧⣤⣀⡀⠀⠀⠀⠀⠀⠀⢿⣷⣄⠀⠀⠀⠁⠋⠀⠀⠀⠀⠀⢸⣿⣿⣇⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⣷⡀⠈⠉⠛⠛⠛⠛⠛⠛⠛⠛⢿⡍⠛⠳⠶⣶⣤⣤⣤⣤⣤⣤⠼⠟⡟⢿⡇⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣷⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠰⣾⡇⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⣤⣴⣿⣷⣶⣶⣶⣶⣶⣶⣦⣀⣀⣀⣻⡀⠀⠀⠀⣀⣀⠀⡀⠀⠀⠀⢀⣼⣿⠇⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⠟⠉⠁⠀⠀⠈⠻⣿⡆⢹⣯⣽⣿⣿⠟⠋⠙⣿⣶⣿⣿⣿⣿⣾⣿⣿⣿⣟⠋⠉⣇⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡇⠀⠀⠀⡀⠀⠀⠀⠈⢻⣆⣿⠀⠀⠀⢁⣶⣿⠿⠟⠛⠷⣶⣽⣿⣿⣻⣏⠙⠃⣴⢻⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⣷⣀⠀⠀⠉⠀⠀⠀⠀⠀⢹⣿⠀⣀⣴⣿⠋⠀⠀⠀⠀⠀⠀⠉⠻⣿⣧⣿⢀⣰⣿⣿⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⢿⣶⣶⣤⣤⣤⣤⣤⣤⣾⣿⣟⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣿⣅⣾⢿⣵⠇⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⠛⠛⠛⠛⠛⠛⠛⠉⠉⠉⠁⢹⣜⠷⠦⠤⠤⠤⠤⠤⠴⠶⠛⣉⣱⠿⠁⠀⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠛⠿⠷⣦⣤⣤⣄⣠⣤⣤⡶⠟⠁⠀⠀⠀⠀⠀⠀⠀
                
                 
    """)

    start_time = time.perf_counter()

    # 1. 初始化混合模型
    model = HybridModel(norm=norm, config=Config).to(device)
    print(f"Hybrid Model Created. Total Params: {count_parameters(model)}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # 學習率調度
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=Config.LR_DECAY_EPOCH,
        gamma=Config.LR_DECAY_RATIO)

    # 記錄器初始化
    history = {
        "phase": [],
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_nrmse": []
    }

    # 繪圖用的固定樣本索引 (隨機抓 3 筆來畫)
    fixed_idx = None

    # =========================================================================
    # 🚀 Phase 1: 訓練 MMINN (只用 AC 數據)
    # =========================================================================
    print("\n" + "=" * 20)
    model.freeze_resnet()
    model.unfreeze_mminn()

    PHASE1_EPOCHS = Config.NUM_EPOCHS_PHASE1  # 總 Epochs

    best_phase1_nrmse = float('inf')
    for epoch in range(PHASE1_EPOCHS):
        model.train()
        train_loss = 0

        for inputs, features, amps, s0, target_H, _ in train_loader_AC:
            inputs, features, amps, s0, target_H = inputs.to(
                device), features.to(device), amps.to(device), s0.to(
                    device), target_H.to(device)
            optimizer.zero_grad()

            # Phase 1: 只取 H_ac (忽略 ResNet)
            _, H_ac = model(inputs, features, amps, s0)
            loss = criterion(H_ac, target_H)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()  # 更新 Learning Rate

        # Validation Phase 1
        model.eval()
        val_loss = 0
        val_nrmse_list = []
        with torch.no_grad():
            for inputs, features, amps, s0, target_H, _ in valid_loader_AC:
                inputs, features, amps, s0, target_H = inputs.to(
                    device), features.to(device), amps.to(device), s0.to(
                        device), target_H.to(device)
                _, H_ac = model(inputs, features, amps, s0)

                loss = criterion(H_ac, target_H)
                val_loss += loss.item()
                val_nrmse_list.append(calculate_nrmse(H_ac, target_H))

        avg_train_loss = train_loss / len(train_loader_AC)
        avg_val_loss = val_loss / len(valid_loader_AC)
        avg_val_nrmse = np.mean(val_nrmse_list)

        # 記錄 History
        history["phase"].append(1)
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_nrmse"].append(avg_val_nrmse)

        # 簡單 Log
        if (epoch + 1) % 10 == 0:
            current_lr = scheduler.get_last_lr()[0]
            time_str = get_time_str(start_time)
            print(
                f"[Phase 1] Ep {epoch+1}/{PHASE1_EPOCHS} | Time: {time_str} | Train: {avg_train_loss:.6f} | Val NRMSE: {avg_val_nrmse:.4f}% | LR: {current_lr:.6f}"
            )

        # 儲存 Phase 1 最佳模型 (基於 NRMSE)
        if avg_val_nrmse < best_phase1_nrmse:
            best_phase1_nrmse = avg_val_nrmse
            torch.save(model.state_dict(),
                       os.path.join(result_dir, "phase1_best.pt"))
            if (epoch + 1) % 10 == 0:
                print(
                    f"  --> ★ Phase 1 Best Model Saved! NRMSE: {best_phase1_nrmse:.4f}%"
                )

        # --- 繪圖 (H比較 & BH比較) ---
        if (epoch + 1) % Config.PLOT_INTERVAL == 0:
            plot_comparison(model, valid_loader_AC, epoch + 1, "Phase1",
                            result_dir, device)

    print(f"✅ Phase 1 Complete. Best AC NRMSE: {best_phase1_nrmse:.4f}%")

    # =========================================================================
    # 🚀 Phase 2: 訓練 ResNet (只用 DC 數據)
    # =========================================================================
    print("\n" + "=" * 60)
    print("   PHASE 2: Training ResNet on DC Data (MMINN Frozen)")
    print("=" * 60)

    model.freeze_mminn()
    model.unfreeze_resnet()

    # reset optimizer 和 scheduler
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=(Config.LEARNING_RATE) * 0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=Config.LR_DECAY_EPOCH,
        gamma=Config.LR_DECAY_RATIO)

    PHASE2_EPOCHS = Config.NUM_EPOCHS_PHASE2  # 總 Epochs
    best_val_nrmse = float('inf')
    patience_counter = 0

    for epoch in range(PHASE2_EPOCHS):
        model.train()
        train_loss = 0

        for inputs, features, amps, s0, target_H, _ in train_loader_DC:
            inputs, features, amps, s0, target_H = inputs.to(
                device), features.to(device), amps.to(device), s0.to(
                    device), target_H.to(device)
            optimizer.zero_grad()

            # Phase 2: 取 H_total
            H_total, _ = model(inputs, features, amps, s0)
            loss = criterion(H_total, target_H)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()  # 更新 Learning Rate

        # Validation Phase 2
        model.eval()
        val_loss = 0
        val_nrmse_list = []
        with torch.no_grad():
            for inputs, features, amps, s0, target_H, _ in valid_loader_DC:
                inputs, features, amps, s0, target_H = inputs.to(
                    device), features.to(device), amps.to(device), s0.to(
                        device), target_H.to(device)
                H_total, _ = model(inputs, features, amps, s0)

                loss = criterion(H_total, target_H)
                val_loss += loss.item()
                val_nrmse_list.append(calculate_nrmse(H_total, target_H))

        avg_train_loss = train_loss / len(train_loader_DC)
        avg_val_loss = val_loss / len(valid_loader_DC)
        avg_val_nrmse = np.mean(val_nrmse_list)

        history["phase"].append(2)
        history["epoch"].append(PHASE1_EPOCHS + epoch + 1)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_nrmse"].append(avg_val_nrmse)

        if (epoch + 1) % 10 == 0:
            current_lr = scheduler.get_last_lr()[0]
            time_str = get_time_str(start_time)
            print(
                f"[Phase 2] Ep {epoch+1}/{PHASE2_EPOCHS} | Time: {time_str} | Train: {avg_train_loss:.6f} | Val NRMSE: {avg_val_nrmse:.4f}% | LR: {current_lr:.6f}"
            )

        # --- Early Stopping (基於 NRMSE) ---
        if avg_val_nrmse < best_val_nrmse:
            best_val_nrmse = avg_val_nrmse
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"  --> ★ Best Model Saved! NRMSE: {best_val_nrmse:.4f}%")
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f" Early stopping triggered at epoch {epoch+1}")
            break

        # --- 繪圖 (H比較 & BH比較) ---
        if (epoch + 1) % Config.PLOT_INTERVAL == 0:
            plot_comparison(model, valid_loader_DC, PHASE1_EPOCHS + epoch + 1,
                            "Phase2", result_dir, device)
            if (epoch + 1) % 10 == 0:
                print(
                    f"  --> ★ Phase 2 Best Model Saved! NRMSE: {best_val_nrmse:.4f}%"
                )


# 結束訓練
    elapsed = time.perf_counter() - start_time
    hrs = int(elapsed // 3600)
    mins = int((elapsed % 3600) // 60)
    secs = elapsed % 60

    print("\n" + "=" * 60)
    print(f"🎉 Training Finished!")
    print(f"⏱️ Total Time: {hrs}h {mins}m {secs:.2f}s")
    print(f"🏆 Best Validation NRMSE: {best_val_nrmse:.4f}%")
    print("=" * 60)

    # 儲存 Summary
    logger.save_summary(
        best_epoch=len(history["epoch"]),  # 最後的 epoch
        best_val_loss=best_val_nrmse,  # 這裡存 NRMSE 代表 loss
        best_loss_H=best_val_nrmse,  # 這裡也是存 NRMSE
        best_loss_Pcv=0.0,  # 暫時沒有 Pcv loss
        model_save_path=model_save_path,
        elapsed=elapsed)

    # 儲存 History 到 JSON
    hist_df = pd.DataFrame(history)
    json_path = os.path.join(result_dir, "training_history.json")
    hist_df.to_json(json_path, orient="records", force_ascii=False, indent=2)
    print(f"✅ History saved to {json_path}")


# %% [markdown]
# ### Start train!

# %%
def main():
    # Python用
    # BASE_DIR = Path(__file__).resolve().parent
    # os.chdir(BASE_DIR)
    # print("👉 Switch CWD to script folder:", os.getcwd())

    # 1. 載入原始數據
    print("載入原始數據...")
    data_B, data_F, data_T, data_H, data_Pcv, data_Hdc, data_N, data_Duty_P, data_Duty_N = load_dataset(
        material)

    # 2.  全球最大值計算：必須基於 "所有數據" 來定義這個世界的邊界，這樣 AC 和 DC 才會在同一個比例尺下被處理
    print("正在計算全球最大值...")
    GLOBAL_B_MAX = np.abs(data_B).max()
    GLOBAL_H_MAX = np.abs(data_H).max()
    print(f"Global B Max: {GLOBAL_B_MAX}")
    print(f"Global H Max: {GLOBAL_H_MAX}")

    print("計算全域 Normalization 參數 (基於完整數據)...")

    # 這是你原本寫在 get_dataloader 裡面的 helper，搬出來用
    def safe_mean_std_np(array, eps=1e-8):
        m = np.mean(array)
        s = np.std(array)
        if s < eps: s = 1.0
        return [float(m), float(s)]

    # 算出這把 "全域的尺"
    global_norm = [
        safe_mean_std_np(np.log10(data_F)),  # F (記得 log10)
        safe_mean_std_np(data_T),  # T
        safe_mean_std_np(data_Hdc),  # Hdc <--- 這裡會算出正確的 Mean/Std
        safe_mean_std_np(data_N),  # N
        safe_mean_std_np(np.log10(data_Pcv))  # Pcv (記得 log10)
    ]

    print("Global Normalization parameters:")
    feature_names = ["F", "T", "Hdc", "N", "Pcv"]
    for i, name in enumerate(feature_names):
        mean, std = global_norm[i]
        print(f"  {name}: mean={mean:.6f}, std={std:.6f}")

    # 4. 數據分流 (AC vs DC)
    print("正在進行數據分流 (AC vs DC)...")
    is_ac_data = (np.abs(data_Hdc) < 1e-5).flatten()
    is_dc_data = ~is_ac_data

    def filter_arrays(indices, *arrays):
        return [arr[indices] for arr in arrays]

    # 製作 AC 數據變數
    (ac_B, ac_F, ac_T, ac_H, ac_Pcv, ac_Hdc, ac_N, ac_Duty_P,
     ac_Duty_N) = filter_arrays(is_ac_data, data_B, data_F, data_T, data_H,
                                data_Pcv, data_Hdc, data_N, data_Duty_P,
                                data_Duty_N)

    # 製作 DC 數據變數
    (dc_B, dc_F, dc_T, dc_H, dc_Pcv, dc_Hdc, dc_N, dc_Duty_P,
     dc_Duty_N) = filter_arrays(is_dc_data, data_B, data_F, data_T, data_H,
                                data_Pcv, data_Hdc, data_N, data_Duty_P,
                                data_Duty_N)

    print(f"原始數據總筆數: {len(data_B)}")
    print(f"AC 數據筆數 (Hdc=0): {len(ac_B)} -> 產生 train_loader_AC")
    print(f"DC 數據筆數 (Hdc!=0): {len(dc_B)} -> 產生 train_loader_DC")

    # 5. 建立 DataLoader
    print("\n=== 建立 AC DataLoader ===")
    train_loader_AC, valid_loader_AC, _ = get_dataloader(
        ac_B,
        ac_F,
        ac_T,
        ac_H,
        ac_N,
        ac_Hdc,
        ac_Duty_P,
        ac_Duty_N,
        ac_Pcv,
        GLOBAL_B_MAX,
        GLOBAL_H_MAX,
        norm=global_norm  # <--- 傳入全域標準
    )

    print("\n=== 建立 DC DataLoader ===")
    train_loader_DC, valid_loader_DC, _ = get_dataloader(
        dc_B,
        dc_F,
        dc_T,
        dc_H,
        dc_N,
        dc_Hdc,
        dc_Duty_P,
        dc_Duty_N,
        dc_Pcv,
        GLOBAL_B_MAX,
        GLOBAL_H_MAX,
        norm=global_norm  # <--- 傳入全域標準
    )

    # 6.Logger
    logger = TrainLogger(
        exp_name=f"{material}_{note}_{timestamp}",
        config_dict={
            k: getattr(Config, k)
            for k in dir(Config)
            if not k.startswith('__') and not callable(getattr(Config, k))
        },
        result_dir=result_dir)
    feature_names = ["F", "T", "Hdc", "N", "Pcv"]
    logger.save_norm_params(global_norm, feature_names)

    print("DataLoader 準備完成！")

    # 7. 呼叫訓練
    train_model(norm=global_norm,
                train_loader_AC=train_loader_AC,
                valid_loader_AC=valid_loader_AC,
                train_loader_DC=train_loader_DC,
                valid_loader_DC=valid_loader_DC,
                logger=logger)


# %%
if __name__ == "__main__":
    main()


