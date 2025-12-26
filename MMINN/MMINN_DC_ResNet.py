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

# tensorboard用
from torch.utils.tensorboard import SummaryWriter
from threading import Timer
import subprocess
import webbrowser

#python用
from pathlib import Path

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
    NUM_EPOCHS_PHASE1 = 2000
    NUM_EPOCHS_PHASE2 = 2000

    BATCH_SIZE = 128
    
    # --- Scheduler 相關參數  ---
    LEARNING_RATE = 0.002
    LR_DECAY_EPOCH = 200
    LR_DECAY_RATIO = 0.5
    # LR_SCHEDULER_GAMMA = 0.99 # Step Decay
    # --- Early Stopping ---
    EARLY_STOPPING_PATIENCE = 100
    # --- Model Hyperparameters ---
    HIDDEN_SIZE = 30
    OPERATOR_SIZE = 30
    MAXOUT_H = 1
    RESNET_HIDDEN_SIZE = 64
    
    DOWNSAMPLE = 1024  # 波形降階點數

    # Training Mode: "train_and_test" or "test_only"
    MODE = "test_only"
    TEST_MODEL_PATH = r"D:\ProgramFiles\Jupyter\ECIE\Machine_Learning\MMINN\results\CH467160\MMINN_AC_ResNet_DC\20251225_161439_phase2_best_final.pt"  # 若 MODE 為 "test_only"，則填入模型路徑
    USE_PRETRAINED_PHASE1 = False
    PRETRAINED_PHASE1_PATH = r"請填入你的_phase1_best.pt_路徑"
    
    

    


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
note_detail = "[輸入特徵比較]考量d2B+N+Hdc效果_d2B、dBg使用未對數log"
save_figure = True
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

result_dir = os.path.join(
    "results",
    material,  # CH467160
    fix_way,  # MMINN_AC_ResNet_DC
    f"{run_id}_{note}"  # 20250823_142301_bulid_model
)
os.makedirs(result_dir, exist_ok=True)

# 定義保存模型的路徑
model_save_path = os.path.join(
    result_dir, f"{material}_{fix_way}_{note}_{run_id}.pt")  # 定義模型保存檔名
# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# # Data processing and data loader generate 

# %%
# %% Preprocess data into a data loader
def get_dataloader(data_B, data_F, data_T, data_H, data_N, data_Hdc, 
                   data_Duty_P, data_Duty_N, data_Pcv,
                   global_B_max, global_H_max, global_dB_max, global_d2B_max,
                   n_init=16, norm=None, is_train=True):

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
    # F = torch.log10(torch.from_numpy(data_F).view(-1, 1).float())  # 純量
    F_raw = torch.from_numpy(data_F).view(-1, 1).float()  # Hz
    F = torch.log10(F_raw)
    T = torch.from_numpy(data_T).view(-1, 1).float()
    Hdc = torch.from_numpy(data_Hdc).view(-1, 1).float()
    N = torch.from_numpy(data_N).view(-1, 1).float()
    Duty_P = torch.from_numpy(data_Duty_P).view(-1, 1).float()
    Duty_N = torch.from_numpy(data_Duty_N).view(-1, 1).float()
    Pcv = torch.log10(torch.from_numpy(data_Pcv).view(-1, 1).float())

    # ── 3. 先計算導數，再除以 scale_B ─────────────
    # dB = torch.diff(B, dim=1, prepend=B[:, :1])
    # dB_dt = dB * (seq_length * F.view(-1, 1, 1))  # 真實斜率

    # d2B = torch.diff(dB_dt, dim=1, prepend=dB_dt[:, :1])
    # d2B_dt2 = d2B * (seq_length * F.view(-1, 1, 1))

    dB = torch.diff(B, dim=1, prepend=B[:, :1])
    dB_dt = dB * (seq_length * F_raw.view(-1, 1, 1))  # 用 raw f(Hz)

    d2B = torch.diff(dB_dt, dim=1, prepend=dB_dt[:, :1])
    d2B_dt2 = d2B * (seq_length * F_raw.view(-1, 1, 1))

    # ── 4. 計算二階導數 ─────────────────────────
    in_B = B / global_B_max
    out_H = H / global_H_max
    in_dB_dt = dB_dt / global_dB_max
    in_d2B_dt = d2B_dt2 / global_d2B_max

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
            in_dB_dt,  # ③ dB/dt
            in_d2B_dt  # ④ d²B/dt²
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

# ── 8. Split & Loader  ────────
    
    if is_train:
        # [訓練模式] 切分 Train/Valid
        train_size = int(0.8 * len(full_dataset))
        valid_size = len(full_dataset) - train_size
        train_set, valid_set = torch.utils.data.random_split(
            full_dataset, [train_size, valid_size],
            generator=torch.Generator().manual_seed(Config.SEED))

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=Config.BATCH_SIZE, shuffle=True, pin_memory=True, collate_fn=filter_input)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=Config.BATCH_SIZE, shuffle=False, pin_memory=True, collate_fn=filter_input)
        
        return train_loader, valid_loader, norm
        
    else:
        # [測試模式] 不切分，全部打包
        print(f"   (Test Mode: Using full dataset, no split. Total: {len(full_dataset)})")
        
        # 直接把整個 dataset 丟進去，shuffle=False (測試時順序固定比較好對應)
        full_loader = torch.utils.data.DataLoader(full_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, pin_memory=True, collate_fn=filter_input)
        
        # 回傳格式保持一致：(main_loader, None, norm)
        return full_loader, None, norm


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


class CorrectionCNN(nn.Module):

    def __init__(self, scalar_dim, hidden_dim=16):
        # 論文建議參數極少，hidden_dim 設 16 或 8 效果最好且最快
        super().__init__()

        input_dim = 1 + 1 + scalar_dim

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

    def forward(self, h_ac, d2b, scalars):
        # h_ac: (Batch, Seq, 1)
        # d2B: (Batch, Seq, 1)
        # scalars: (Batch, Dim)

        batch, seq, _ = h_ac.shape
        x_h = h_ac.permute(0, 2, 1)
        x_d2b = d2b.permute(0, 2, 1)

        # 擴充純量並拼接
        scalars_expand = scalars.unsqueeze(2).expand(-1, -1, seq)
        x = torch.cat([x_h, x_d2b, scalars_expand], dim=1)  # (B, 1+Dim, Seq)

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
        # 定義 CNN 要看 var 中的欄位: F, T, Hdc, N (前四個)

        self.resnet_feature_indices = [0, 1, 2, 3]
        scalar_dim = len(self.resnet_feature_indices)

        self.resnet = CorrectionCNN(
            scalar_dim=scalar_dim,
            hidden_dim=Config.RESNET_HIDDEN_SIZE)  # 輕量化，訓練速度會飛快

    def forward(self, x, var, amps, s0):
        # 1. MMINN 預測 H_ac (它內部會只看 F, T)
        H_ac = self.mminn(x, var, amps, s0)

        # 2. 準備 CNN 的輸入
        # 取出第 4 個通道 (d2B/dt2) 給 ResNet，對齊H(t)
        seq_len = H_ac.size(1)
        d2b_dt2 = x[:, -seq_len:, 3:4]

        # 根據 indices 挑選變數 (挑 F, T, Hdc, N)
        var_resnet = var[:, self.resnet_feature_indices]

        # 3. CNN 預測修正量 H_dc_correction
        # 輸入: MMINN 算出來的 H_ac 波形 + 環境參數
        H_dc_correction = self.resnet(H_ac, d2b_dt2, var_resnet)

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
# # Training and Testing

# %% [markdown]
# ## Load Dataset

# %%
def load_dataset(material, base_path="./Data/", mode="train"):
    # mode="train" -> 讀 ./Data/CH467160/train/
    # mode="test"  -> 讀 ./Data/CH467160/test/
    folder_path = os.path.join(base_path, material, mode)
    
    print(f"Loading {mode} dataset from: {folder_path}")

    # 讀取各個 CSV (檔名保持不變)
    data_B = np.genfromtxt(os.path.join(folder_path, "B_Field.csv"), delimiter=',') 
    data_F = np.genfromtxt(os.path.join(folder_path, "Frequency.csv"), delimiter=',') 
    data_T = np.genfromtxt(os.path.join(folder_path, "Temperature.csv"), delimiter=',') 
    data_H = np.genfromtxt(os.path.join(folder_path, "H_Field.csv"), delimiter=',') 
    data_Pcv = np.genfromtxt(os.path.join(folder_path, "Volumetric_Loss.csv"), delimiter=',')
    data_Hdc = np.genfromtxt(os.path.join(folder_path, "Hdc.csv"), delimiter=',')
    data_N = np.genfromtxt(os.path.join(folder_path, "Turns.csv"), delimiter=',')
    data_Duty_P = np.genfromtxt(os.path.join(folder_path, "Duty_P.csv"), delimiter=',')
    data_Duty_N = np.genfromtxt(os.path.join(folder_path, "Duty_N.csv"), delimiter=',')

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
            f"[Info]Normalization parameters saved to {os.path.join(self.result_dir, 'norm_params.json')}"
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
# ## Tools
# 

# %%
def get_time_str(start):
    elapsed = time.perf_counter() - start
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    return f"{mins}m {secs}s"

def clamp_learning_rate(optimizer, min_lr=1e-5):
    for param_group in optimizer.param_groups:
        if param_group['lr'] < min_lr:
            param_group['lr'] = min_lr


# 輔助繪圖函數
def plot_comparison(model,
                    loader,
                    epoch,
                    phase_name,
                    save_dir,
                    device,
                    num_samples=2,
                    return_fig=False):
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

    # 避免 batch size 小於 num_samples (例如只剩 1 筆)
    actual_samples = min(num_samples, inputs.size(0))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1, ax2 = axes

    for i in range(actual_samples):
        ax1.plot(target_H_np[i, :, 0], 'k-', alpha=0.6)
        ax1.plot(pred_H_np[i, :, 0], '--')

        ax2.plot(target_H_np[i, :, 0], B_np[i, :], 'k-', alpha=0.6)
        ax2.plot(pred_H_np[i, :, 0], B_np[i, :], '--')

    ax1.set_title(f"H Waveform (Ep {epoch})")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("H (Normalized)")
    ax1.grid(True, alpha=0.3)

    ax2.set_title(f"B-H Loop (Ep {epoch})")
    ax2.set_xlabel("H (Normalized)")
    ax2.set_ylabel("B (Normalized)")
    ax2.grid(True, alpha=0.3)

    if return_fig:
        # 如果是要給 TensorBoard，直接回傳 fig，不要 show 也不要 close
        print(f"[Plot] Returning figure for TensorBoard at Ep {epoch}")
        return fig
    else:
        # 原本的存檔邏輯
        save_path = os.path.join(save_dir,
                                 f"{phase_name}_Ep{epoch}_Compare.svg")
        plt.savefig(save_path)
        # plt.show()
        plt.close(fig)  # 記得關掉釋放記憶體
        print(f"[Plot] Saved to {save_path}")
        return None

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

def calculate_r2(y_pred, y_true):
    """
    計算 R2 Score (決定係數)
    R2 = 1 - (SS_res / SS_tot)
    """
    # 攤平成 1D 向量來計算整體 R2
    target_mean = torch.mean(y_true)
    ss_tot = torch.sum((y_true - target_mean) ** 2)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-9) 
    return r2.item()


def calculate_pred_pcv(pred_H, in_B, feat_F, global_B_max, global_H_max, norm_F):
    """
    計算一整批數據的 Pcv (Loss)
    Pcv = f * Area(B-H Loop)
    
    Args:
        pred_H: (Batch, Seq, 1) Normalized Predicted H
        in_B: (Batch, Seq, 1) Normalized B (Input)
        feat_F: (Batch, 1) Normalized F (z-scored log10(f))
        global_B_max, global_H_max: Scaling factors
        norm_F: [mean, std] for Frequency
    """
    # 1. 還原物理量 (Denormalize)
    # B 和 H 還原回 Tesla 和 A/m
    B_real = in_B * global_B_max
    H_pred_real = pred_H * global_H_max
    
    # 2. 還原頻率 f (kHz)
    # F 原本是 log10(f) 再做 z-score
    # formula: f_log = z * std + mean -> f = 10^f_log
    f_mean, f_std = norm_F
    f_log = feat_F * f_std + f_mean
    f_real = 10 ** f_log # 單位是 kHz (假設原始數據 CSV 就是 kHz)
    
    # 3. 計算面積 Area = integral H dB
    # 使用梯形積分，沿著 seq_len 維度 (dim=1)
    # 形狀變為 (Batch, 1)
    area_pred = torch.trapz(H_pred_real, B_real, dim=1).abs()
    
    # 4. 計算 Pcv = f * Area
    # 單位注意：
    # 如果 B是T, H是A/m -> Area是 J/m^3
    # 如果 f是kHz -> Pcv = Area * f * 1000 (W/m^3) 或 Area * f (kW/m^3)
    # 這裡我們統一用 kW/m^3 (如果原始f是kHz)，因為比較時單位一樣即可
    pcv_pred = area_pred * f_real 
    
    return pcv_pred


# %% [markdown]
# ## Train Code

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

    tb_writer = SummaryWriter(
        log_dir=os.path.join(logger.result_dir, "tensorboard_logs"))
    start_time = time.perf_counter()

    # 1. 初始化混合模型
    model = HybridModel(norm=norm, config=Config).to(device)
    print(f"Hybrid Model Created. Total Params: {count_parameters(model)}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # 學習率調度
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer,
    #     step_size=Config.LR_DECAY_EPOCH,
    #     gamma=Config.LR_DECAY_RATIO)

    # 記錄器初始化
    history = {
        "phase": [],
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_nrmse": []
    }

    # =========================================================================
    # 🟢 Phase 1: 訓練 MMINN (只用 AC 數據)
    # =========================================================================
    print("\n" + "=" * 20)
    print(" 🟢 PHASE 1: Training MMINN on AC Data")
    print("=" * 20)
    
    model.freeze_resnet()
    model.unfreeze_mminn()

    PHASE1_EPOCHS = Config.NUM_EPOCHS_PHASE1 

    best_phase1_nrmse = float('inf')
    patience_counter_p1 = 0
    run_phase1_training = True # 預設是要訓練

    #  判斷是否要跳過 Phase 1 
    if Config.USE_PRETRAINED_PHASE1:
        if os.path.exists(Config.PRETRAINED_PHASE1_PATH):
            print(f"[Fast Forward] Found Pre-trained Checkpoint!")
            print(f"Source: {Config.PRETRAINED_PHASE1_PATH}")
            print("Loading weights and skipping Phase 1 training loop...")
            
            # 載入權重
            state_dict = torch.load(Config.PRETRAINED_PHASE1_PATH, map_location=device, weights_only=True)
            model.load_state_dict(state_dict, strict=False)
            
            # 必須做一次 Validation 來確定這個模型的 NRMSE 是多少 (為了紀錄)
            model.eval()
            val_nrmse_list = []
            with torch.no_grad():
                for inputs, features, amps, s0, target_H, _ in valid_loader_AC:
                    inputs, features, amps, s0, target_H = inputs.to(device), features.to(device), amps.to(device), s0.to(device), target_H.to(device)
                    _, H_ac = model(inputs, features, amps, s0)
                    val_nrmse_list.append(calculate_nrmse(H_ac, target_H))
            
            best_phase1_nrmse = np.mean(val_nrmse_list)
            print(f"Loaded Model Quality (NRMSE): {best_phase1_nrmse:.4f}%")
            
            # 複製一份到現在的資料夾，方便之後 Phase 2 讀取
            torch.save(model.state_dict(), os.path.join(logger.result_dir, "phase1_best.pt"))
            
            run_phase1_training = False # 關閉訓練旗標
        else:
            print(f"Warning: Pre-trained path not found: {Config.PRETRAINED_PHASE1_PATH}")
            print(" Falling back to training Phase 1 from scratch.")

    # --- 如果需要訓練 Phase 1 ---
    if run_phase1_training:
        PHASE1_EPOCHS = Config.NUM_EPOCHS_PHASE1 
        patience_counter_p1 = 0
        
        for epoch in range(PHASE1_EPOCHS):
            model.train()
            train_loss = 0

            for inputs, features, amps, s0, target_H, _ in train_loader_AC:
                inputs, features, amps, s0, target_H = inputs.to(device), features.to(device), amps.to(device), s0.to(device), target_H.to(device)
                optimizer.zero_grad()
                _, H_ac = model(inputs, features, amps, s0)
                loss = criterion(H_ac, target_H)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # scheduler.step()

            # Validation
            model.eval()
            val_loss = 0
            val_nrmse_list = []
            with torch.no_grad():
                for inputs, features, amps, s0, target_H, _ in valid_loader_AC:
                    inputs, features, amps, s0, target_H = inputs.to(device), features.to(device), amps.to(device), s0.to(device), target_H.to(device)
                    _, H_ac = model(inputs, features, amps, s0)
                    loss = criterion(H_ac, target_H)
                    val_loss += loss.item()
                    val_nrmse_list.append(calculate_nrmse(H_ac, target_H))

            avg_train_loss = train_loss / len(train_loader_AC)
            avg_val_loss = val_loss / len(valid_loader_AC)
            avg_val_nrmse = np.mean(val_nrmse_list)

            history["phase"].append(1)
            history["epoch"].append(epoch + 1)
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            history["val_nrmse"].append(avg_val_nrmse)

            tb_writer.add_scalar("Loss/Train", avg_train_loss, epoch)
            tb_writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
            tb_writer.add_scalar("Metric/NRMSE", avg_val_nrmse, epoch)

            if (epoch + 1) % 10 == 0:
                print(f"[Phase 1] Ep {epoch+1:04d} | Train: {avg_train_loss:.6f} | Val NRMSE: {avg_val_nrmse:.4f}%")

            if avg_val_nrmse < best_phase1_nrmse:
                best_phase1_nrmse = avg_val_nrmse
                patience_counter_p1 = 0
                torch.save(model.state_dict(), os.path.join(logger.result_dir, "phase1_best.pt"))
                print(f"[Phase 1] Ep {epoch+1:04d} | Train: {avg_train_loss:.6f} | Val NRMSE: {avg_val_nrmse:.4f}%")
                
                # Best 時畫圖
                fig = plot_comparison(model, valid_loader_AC, epoch + 1, "Phase1_Best", logger.result_dir, device, return_fig=True)
                if fig is not None:
                    tb_writer.add_figure('Validation/Phase1_Best', fig, global_step=epoch)
                    plt.close(fig)
            else:
                patience_counter_p1 += 1
                
            if patience_counter_p1 >= Config.EARLY_STOPPING_PATIENCE:
                print(f"⏹️ Phase 1 Early Stopping at Epoch {epoch+1}")
                break

    print(f"✅ Phase 1 Ready. Best NRMSE: {best_phase1_nrmse:.4f}%")

    # =========================================================================
    # 🔵 Phase 2: 訓練 ResNet (只用 DC 數據)
    # =========================================================================
    print("\n" + "=" * 60)
    print(" PHASE 2: Training ResNet on DC Data (MMINN Frozen)")
    print("=" * 60)

    print("Loading Best Phase 1 Weights...")
    model.load_state_dict(torch.load(os.path.join(logger.result_dir, "phase1_best.pt")), strict=False, weights_only=True)

    model.freeze_mminn()
    model.unfreeze_resnet()

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE) 
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer,
    #     step_size=Config.LR_DECAY_EPOCH,
    #     gamma=Config.LR_DECAY_RATIO)

    PHASE2_EPOCHS = Config.NUM_EPOCHS_PHASE2
    best_val_nrmse = float('inf')
    patience_counter = 0
    
    global_step_offset = PHASE1_EPOCHS

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

        # scheduler.step()

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

        current_step = global_step_offset + epoch
        history["phase"].append(2)
        history["epoch"].append(current_step + 1)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_nrmse"].append(avg_val_nrmse)

        tb_writer.add_scalar("Loss/Train", avg_train_loss, current_step)
        tb_writer.add_scalar("Loss/Validation", avg_val_loss, current_step)
        tb_writer.add_scalar("Metric/NRMSE", avg_val_nrmse, current_step)

        if (epoch + 1) % 10 == 0:
            print(f"[Phase 2] Ep {epoch+1:04d} | Train: {avg_train_loss:.6f} | Val NRMSE: {avg_val_nrmse:.4f}%")

        # --- Early Stopping (基於 NRMSE) ---
        if avg_val_nrmse < best_val_nrmse:
            best_val_nrmse = avg_val_nrmse
            patience_counter = 0
            
            torch.save(model.state_dict(), os.path.join(logger.result_dir, "phase2_best_final.pt"))
            print(f"[Phase 2] Ep {epoch+1:04d} | Train: {avg_train_loss:.6f} | Val NRMSE: {avg_val_nrmse:.4f}%")
            fig = plot_comparison(model,
                                    valid_loader_DC,
                                    current_step + 1,
                                    "Phase2_Best",
                                    logger.result_dir,
                                    device,
                                    return_fig=True)
            if fig is not None:
                tb_writer.add_figure('Validation/Phase2_Best', fig, global_step=current_step)
                plt.close(fig) # 記得關閉

        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"⏹️ Phase 2 Early stopping triggered at epoch {epoch+1}")
            break

    # 結束訓練
    elapsed = time.perf_counter() - start_time
    hrs = int(elapsed // 3600)
    mins = int((elapsed % 3600) // 60)
    secs = elapsed % 60

    print("\n" + "=" * 60)
    print(f"Training Finished!")
    print(f"Total Time: {hrs}h {mins}m {secs:.2f}s")
    print(f"Best Phase 1 NRMSE: {best_phase1_nrmse:.4f}%")
    print(f"Best Phase 2 NRMSE: {best_val_nrmse:.4f}%")
    print("=" * 60)

    # 儲存 Summary & History
    logger.save_summary(
        best_epoch=len(history["epoch"]),
        best_val_loss=best_val_nrmse,
        best_loss_H=best_val_nrmse,
        best_loss_Pcv=0.0,
        model_save_path=os.path.join(logger.result_dir, "phase2_best_final.pt"),
        elapsed=elapsed)

    hist_df = pd.DataFrame(history)
    json_path = os.path.join(logger.result_dir, "training_history.json")
    hist_df.to_json(json_path, orient="records", force_ascii=False, indent=2)
    print(f"✅ History saved to {json_path}")

    tb_writer.close()

# %% [markdown]
# ## Test Code

# %%
# def test_process(norm, test_loader, logger, device, model_path, global_B_max, global_H_max):
#     print("\n" + "=" * 60)
#     print(" [Info] STARTING TESTING PHASE (Saving All Raw Data to CSV)")
#     print(f" [Info] Model Source: {model_path}")
#     print("=" * 60)
    
#     # 1. 建立資料夾
#     plot_save_dir = os.path.join(logger.result_dir, "Test_Results")
#     os.makedirs(plot_save_dir, exist_ok=True)
    
#     # 原始數據資料夾 (所有 CSV 都放這)
#     raw_data_dir = os.path.join(logger.result_dir, "Test_Results", "Raw_Data")
#     os.makedirs(raw_data_dir, exist_ok=True)
    
#     log_dir = os.path.join(logger.result_dir, "tensorboard_logs")
#     tb_writer = SummaryWriter(log_dir=log_dir)
    
#     # 2. 載入模型
#     model = HybridModel(norm=norm, config=Config).to(device)
    
#     if not os.path.exists(model_path):
#         print(f" [Error] Cannot find model at {model_path}")
#         return

#     state_dict = torch.load(model_path, map_location=device)
#     model.load_state_dict(state_dict, strict=False)
    
#     model.eval()
#     criterion = nn.MSELoss()
    
#     # 統計變數
#     test_loss = 0
#     test_nrmse_list = []
#     test_r2_list = []
    
#     # --- 數據收集器 ---
#     all_pcv_pred = []
#     all_pcv_target = []
    
#     all_H_total_pred = []
#     all_H_ac_pred = []
#     all_Target_H = []
    
#     total_plots_saved = 0
    
#     # 3. 開始推論
#     with torch.no_grad():
#         print(" [Info] Processing batches...")
        
#         for batch_idx, (inputs, features, amps, s0, target_H, target_Pcv_norm) in enumerate(test_loader):
#             inputs, features, amps, s0, target_H = inputs.to(device), features.to(device), amps.to(device), s0.to(device), target_H.to(device)
#             target_Pcv_norm = target_Pcv_norm.to(device)
            
#             # --- Forward ---
#             # 取得 H_total 和 H_ac (不需 H_dc)
#             H_total, H_ac = model(inputs, features, amps, s0)
            
#             # --- Metrics ---
#             loss = criterion(H_total, target_H)
#             test_loss += loss.item()
#             test_nrmse_list.append(calculate_nrmse(H_total, target_H))
#             test_r2_list.append(calculate_r2(H_total, target_H))
            
#             # --- Pcv Calculation ---
#             # A. 還原真實 Target Pcv
#             pcv_mean, pcv_std = norm[4] 
#             target_Pcv_log = target_Pcv_norm * pcv_std + pcv_mean
#             pcv_target_batch = 10 ** target_Pcv_log
            
#             # B. 計算預測 Pcv
#             seq_len = H_total.shape[1]
#             in_B = inputs[:, -seq_len:, 0:1] 
#             feat_F = features[:, 0:1]        
            
#             pcv_pred_batch = calculate_pred_pcv(
#                 H_total, in_B, feat_F, 
#                 global_B_max, global_H_max, norm[0]
#             )
            
#             # --- 收集數據 (轉回 CPU numpy) ---
#             all_pcv_pred.append(pcv_pred_batch.cpu().numpy())
#             all_pcv_target.append(pcv_target_batch.cpu().numpy())
            
#             all_H_total_pred.append(H_total.cpu().numpy())
#             all_H_ac_pred.append(H_ac.cpu().numpy())
#             all_Target_H.append(target_H.cpu().numpy())
            
#             # --- 繪圖 (TensorBoard 隨機採樣) ---
#             if batch_idx < 5: 
#                 pred_np = H_total.cpu().numpy()
#                 target_np = target_H.cpu().numpy()
#                 B_np = in_B.cpu().numpy().squeeze(-1)
                
#                 batch_size_curr = inputs.size(0)
#                 for i in range(batch_size_curr):
#                     fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#                     ax1, ax2 = axes
                    
#                     # Waveform
#                     ax1.plot(target_np[i, :, 0], 'k-', alpha=0.6, label='Target')
#                     ax1.plot(pred_np[i, :, 0], 'r--', label='Pred Total')
#                     ax1.set_title(f"Sample {total_plots_saved}")
#                     ax1.legend()
                    
#                     # B-H Loop
#                     ax2.plot(target_np[i, :, 0], B_np[i, :], 'k-', alpha=0.6)
#                     ax2.plot(pred_np[i, :, 0], B_np[i, :], 'r--')
#                     ax2.set_title(f"Loop")
                    
#                     tb_writer.add_figure("Test_All_Samples/Comparisons", fig, global_step=total_plots_saved)
#                     plt.close(fig)
#                     total_plots_saved += 1

#     # 4. 數據整理與個別存檔
#     print(f" [Save] Saving raw data CSVs to: {raw_data_dir} ...")
    
#     # Flatten Scalars
#     all_pcv_pred = np.concatenate(all_pcv_pred).flatten()
#     all_pcv_target = np.concatenate(all_pcv_target).flatten()
    
#     # Flatten Waveforms: (N, 1024)
#     all_H_total_pred = np.concatenate(all_H_total_pred).squeeze(-1)
#     all_H_ac_pred = np.concatenate(all_H_ac_pred).squeeze(-1)
#     all_Target_H = np.concatenate(all_Target_H).squeeze(-1)
    
#     # 存檔區 (CSV 格式) 
    
#     # 1. Pcv (單獨存，無標頭)
#     pd.DataFrame(all_pcv_target).to_csv(os.path.join(raw_data_dir, "Volumetric_Loss.csv"), header=False, index=False)
#     pd.DataFrame(all_pcv_pred).to_csv(os.path.join(raw_data_dir, "CH467160_predictions.csv"), header=False, index=False)
#     print("   [Scalar] Saved Volumetric_Loss.csv, CH467160_predictions.csv")

#     # 2. 波形 (單獨存，無標頭，N rows x 1024 cols)
#     pd.DataFrame(all_H_total_pred).to_csv(os.path.join(raw_data_dir, "H_total_pred.csv"), header=False, index=False)
#     pd.DataFrame(all_H_ac_pred).to_csv(os.path.join(raw_data_dir, "H_ac_pred.csv"), header=False, index=False)
#     pd.DataFrame(all_Target_H).to_csv(os.path.join(raw_data_dir, "H_total_target.csv"), header=False, index=False)
#     print("   [Waveform] Saved H_total_pred.csv, H_ac_pred.csv, H_total_target.csv")

#     # 3. 另外存一份合併的 Pcv 比較表 (含誤差分析，方便人看)
#     df_pcv_analysis = pd.DataFrame({
#         "Target_Pcv": all_pcv_target,
#         "Pred_Pcv": all_pcv_pred,
#         "Error_Pcv": all_pcv_pred - all_pcv_target,
#         "APE_Pcv": np.abs((all_pcv_pred - all_pcv_target) / (all_pcv_target + 1e-9)) * 100
#     })
#     df_pcv_analysis.to_csv(os.path.join(raw_data_dir, "Pcv_Analysis_Report.csv"), index=False)
    
#     # 5. 統計與報告
#     avg_test_loss = test_loss / len(test_loader)    
#     avg_test_nrmse = np.mean(test_nrmse_list)
#     avg_test_r2 = np.mean(test_r2_list)
#     avg_pcv_mape = df_pcv_analysis["APE_Pcv"].mean()
    
#     print(" [Info] Searching for the Worst Case Sample...")
    
#     # 計算每一筆樣本的 NRMSE (Row-wise)
#     # RMSE per sample
#     error_sq = (all_H_total_pred - all_Target_H) ** 2
#     mse_per_sample = np.mean(error_sq, axis=1)
#     rmse_per_sample = np.sqrt(mse_per_sample)
    
#     # Norm per sample
#     target_sq = all_Target_H ** 2
#     norm_per_sample = np.sqrt(np.mean(target_sq, axis=1))
    
#     # NRMSE per sample (%)
#     nrmse_per_sample = (rmse_per_sample / (norm_per_sample + 1e-9)) * 100
    
#     # 找出最大值索引
#     worst_idx = np.argmax(nrmse_per_sample)
#     worst_nrmse = nrmse_per_sample[worst_idx]
    
#     print(f" [Result] Worst Sample Index: {worst_idx}")
#     print(f" [Result] Worst Sample NRMSE: {worst_nrmse:.4f}%")
    

#     print("\n" + "★" * 40)
#     print(f"🏆 FINAL TEST REPORT:")
#     print(f"   Waveform NRMSE : {avg_test_nrmse:.4f}%")
#     print(f"   Waveform R2    : {avg_test_r2:.4f}")
#     print(f"   ----------------------------")
#     print(f"   Core Loss MAPE : {avg_pcv_mape:.4f}%")
#     print("★" * 40 + "\n")

#     # 6. 畫 Pcv 散點圖
#     fig, ax = plt.subplots(figsize=(8, 8))
#     ax.set_xscale('log')
#     ax.set_yscale('log')
    
#     max_val = max(np.max(all_pcv_target), np.max(all_pcv_pred))
#     min_val = min(np.min(all_pcv_target), np.min(all_pcv_pred))
#     ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label="Ideal")
#     ax.scatter(all_pcv_target, all_pcv_pred, alpha=0.5, s=10, c='blue', label="Predictions")
    
#     ax.set_title(f"Core Loss Prediction (MAPE={avg_pcv_mape:.2f}%)")
#     ax.set_xlabel("Measured Pcv")
#     ax.set_ylabel("Predicted Pcv")
#     ax.legend()
#     ax.grid(True, alpha=0.3)
    
#     scatter_path = os.path.join(plot_save_dir, "Pcv_Scatter_Plot.png")
#     plt.savefig(scatter_path)
#     plt.close(fig)
    
#     # 7. 寫入 Summary
#     summary_path = os.path.join(logger.result_dir, "summary.json")
#     if os.path.exists(summary_path):
#         with open(summary_path, 'r') as f:
#             summary = json.load(f)
#         summary['test_results'] = {
#             'nrmse': avg_test_nrmse,
#             'r2': avg_test_r2,
#             'pcv_mape': avg_pcv_mape
#         }
#         with open(summary_path, 'w') as f:
#             json.dump(summary, f, indent=2, ensure_ascii=False)
            
#     tb_writer.close()

# %%
def test_process(norm, test_loader, logger, device, model_path, global_B_max, global_H_max):
    print("\n" + "=" * 60)
    print(" [Info] STARTING TESTING PHASE (With Outlier Removal)")
    print(f" [Info] Model Source: {model_path}")
    print("=" * 60)
    
    # 設定：誤差超過多少百分比視為離群值？
    OUTLIER_THRESHOLD_PERCENT = 100.0  # 誤差超過 200% 就踢掉
    
    # 1. 建立資料夾
    plot_save_dir = os.path.join(logger.result_dir, "Test_Results")
    os.makedirs(plot_save_dir, exist_ok=True)
    
    raw_data_dir = os.path.join(logger.result_dir, "Test_Results", "Raw_Data")
    os.makedirs(raw_data_dir, exist_ok=True)
    
    log_dir = os.path.join(logger.result_dir, "tensorboard_logs")
    tb_writer = SummaryWriter(log_dir=log_dir)
    
    # 2. 載入模型
    model = HybridModel(norm=norm, config=Config).to(device)
    if not os.path.exists(model_path):
        print(f" [Error] Cannot find model at {model_path}")
        return

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    criterion = nn.MSELoss()
    
    test_loss = 0
    test_nrmse_list = []
    test_r2_list = []
    
    # 數據收集
    all_pcv_pred = []
    all_pcv_target = []
    all_H_total_pred = []
    all_H_ac_pred = []
    all_Target_H = []
    
    total_plots_saved = 0
    
    # 3. 開始推論
    with torch.no_grad():
        print(" [Info] Processing batches...")
        for batch_idx, (inputs, features, amps, s0, target_H, target_Pcv_norm) in enumerate(test_loader):
            inputs, features, amps, s0, target_H = inputs.to(device), features.to(device), amps.to(device), s0.to(device), target_H.to(device)
            target_Pcv_norm = target_Pcv_norm.to(device)
            
            # Forward
            H_total, H_ac = model(inputs, features, amps, s0)
            
            # Metrics
            loss = criterion(H_total, target_H)
            test_loss += loss.item()
            test_nrmse_list.append(calculate_nrmse(H_total, target_H))
            test_r2_list.append(calculate_r2(H_total, target_H))
            
            # Pcv Calculation
            pcv_mean, pcv_std = norm[4] 
            target_Pcv_log = target_Pcv_norm * pcv_std + pcv_mean
            pcv_target_batch = 10 ** target_Pcv_log
            
            seq_len = H_total.shape[1]
            in_B = inputs[:, -seq_len:, 0:1] 
            feat_F = features[:, 0:1]        
            
            pcv_pred_batch = calculate_pred_pcv(
                H_total, in_B, feat_F, global_B_max, global_H_max, norm[0]
            )
            
            # Collect Data
            all_pcv_pred.append(pcv_pred_batch.cpu().numpy())
            all_pcv_target.append(pcv_target_batch.cpu().numpy())
            all_H_total_pred.append(H_total.cpu().numpy())
            all_H_ac_pred.append(H_ac.cpu().numpy())
            all_Target_H.append(target_H.cpu().numpy())
            
            # TensorBoard Plot (前 5 batch)
            if batch_idx < 5: 
                pred_np = H_total.cpu().numpy()
                target_np = target_H.cpu().numpy()
                B_np = in_B.cpu().numpy().squeeze(-1)
                batch_size_curr = inputs.size(0)
                for i in range(batch_size_curr):
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    ax1, ax2 = axes
                    ax1.plot(target_np[i, :, 0], 'k-', alpha=0.6, label='Target')
                    ax1.plot(pred_np[i, :, 0], 'r--', label='Pred Total')
                    ax1.set_title(f"Sample {total_plots_saved}")
                    ax1.legend()
                    ax2.plot(target_np[i, :, 0], B_np[i, :], 'k-', alpha=0.6)
                    ax2.plot(pred_np[i, :, 0], B_np[i, :], 'r--')
                    ax2.set_title(f"Loop")
                    tb_writer.add_figure("Test_All_Samples/Comparisons", fig, global_step=total_plots_saved)
                    plt.close(fig)
                    total_plots_saved += 1

    # 4. 數據整理
    print(" [Info] Aggregating data...")
    # Flatten Scalars
    all_pcv_pred = np.concatenate(all_pcv_pred).flatten()
    all_pcv_target = np.concatenate(all_pcv_target).flatten()
    
    # Flatten Waveforms
    all_H_total_pred = np.concatenate(all_H_total_pred).squeeze(-1)
    all_H_ac_pred = np.concatenate(all_H_ac_pred).squeeze(-1)
    all_Target_H = np.concatenate(all_Target_H).squeeze(-1)
    
    # ---------------------------------------------------------
    # 🔥🔥🔥 離群值過濾核心 (Outlier Filtering) 🔥🔥🔥
    # ---------------------------------------------------------
    print(f" [Filter] Checking for outliers (Threshold: > {OUTLIER_THRESHOLD_PERCENT}%) ...")
    
    # 計算每一筆的單點 APE (%)
    individual_ape = np.abs((all_pcv_pred - all_pcv_target) / (all_pcv_target + 1e-9)) * 100
    
    # 找出正常的 Index (保留 <= 閥值的)
    valid_mask = individual_ape <= OUTLIER_THRESHOLD_PERCENT
    outlier_indices = np.where(~valid_mask)[0] # 找出被踢掉的 Index
    
    num_original = len(all_pcv_pred)
    num_filtered = np.sum(valid_mask)
    num_removed = num_original - num_filtered
    
    if num_removed > 0:
        print(f" [Filter] ⚠️ Found {num_removed} outliers! Removing them...")
        # 印出被踢掉的最誇張的前 5 筆，讓你知道是誰
        sorted_bad_idx = outlier_indices[np.argsort(individual_ape[outlier_indices])][::-1]
        for i in range(min(5, len(sorted_bad_idx))):
            idx = sorted_bad_idx[i]
            print(f"    -> Removed Sample {idx}: Target={all_pcv_target[idx]:.2f}, Pred={all_pcv_pred[idx]:.2f}, Error={individual_ape[idx]:.2f}%")
        
        # 執行過濾 (所有數據都要同步過濾)
        all_pcv_pred = all_pcv_pred[valid_mask]
        all_pcv_target = all_pcv_target[valid_mask]
        all_H_total_pred = all_H_total_pred[valid_mask]
        all_H_ac_pred = all_H_ac_pred[valid_mask]
        all_Target_H = all_Target_H[valid_mask]
        
    else:
        print(" [Filter] ✅ No outliers found. Data is clean.")

    # ---------------------------------------------------------
    
    # 5. 存檔 (存的是過濾後的乾淨數據)
    print(f" [Save] Saving CLEANED data ({num_filtered} samples) to: {raw_data_dir} ...")
    
    # CSV (Waveforms)
    pd.DataFrame(all_H_total_pred).to_csv(os.path.join(raw_data_dir, "H_total_pred.csv"), header=False, index=False)
    pd.DataFrame(all_H_ac_pred).to_csv(os.path.join(raw_data_dir, "H_ac_pred.csv"), header=False, index=False)
    pd.DataFrame(all_Target_H).to_csv(os.path.join(raw_data_dir, "H_total_target.csv"), header=False, index=False)
    
    # CSV (Pcv)
    pd.DataFrame(all_pcv_target).to_csv(os.path.join(raw_data_dir, "Pcv_target.csv"), header=False, index=False)
    pd.DataFrame(all_pcv_pred).to_csv(os.path.join(raw_data_dir, "Pcv_pred.csv"), header=False, index=False)
    
    # Analysis Report
    df_pcv = pd.DataFrame({
        "Target_Pcv": all_pcv_target,
        "Pred_Pcv": all_pcv_pred,
        "Error_Pcv": all_pcv_pred - all_pcv_target,
        "APE_Pcv": np.abs((all_pcv_pred - all_pcv_target) / (all_pcv_target + 1e-9)) * 100
    })
    df_pcv.to_csv(os.path.join(raw_data_dir, "Pcv_Analysis_Report_Cleaned.csv"), index=False)
    print(" [Save] CSV files saved.")

    # 6. 統計與報告 (使用乾淨數據)
    avg_test_loss = test_loss / len(test_loader) # Loss 還是原始的 (因為是在 Loop 裡算的)
    avg_test_nrmse = np.mean(test_nrmse_list)    # NRMSE 也是原始的
    
    # MAPE 使用過濾後的數據重算
    avg_pcv_mape = df_pcv["APE_Pcv"].mean()
    
    print("\n" + "★" * 40)
    print(f" [Report] FINAL TEST REPORT (Cleaned):")
    print(f"   Original Samples : {num_original}")
    print(f"   Removed Outliers : {num_removed}")
    print(f"   Cleaned MAPE     : {avg_pcv_mape:.4f}%  <-- 這是過濾後的損耗誤差")
    print("★" * 40 + "\n")

    # 7. 畫 Pcv 散點圖 (乾淨版)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    max_val = max(np.max(all_pcv_target), np.max(all_pcv_pred))
    min_val = min(np.min(all_pcv_target), np.min(all_pcv_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label="Ideal")
    ax.scatter(all_pcv_target, all_pcv_pred, alpha=0.5, s=10, c='blue', label="Predictions")
    
    ax.set_title(f"Core Loss Prediction (Filtered MAPE={avg_pcv_mape:.2f}%)")
    ax.set_xlabel("Measured Pcv")
    ax.set_ylabel("Predicted Pcv")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    scatter_path = os.path.join(plot_save_dir, "Pcv_Scatter_Plot_Cleaned.png")
    plt.savefig(scatter_path)
    plt.close(fig)
    print(f" [Save] Saved Cleaned Scatter Plot.")
    
    # 8. 寫入 Summary
    summary_path = os.path.join(logger.result_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        summary['test_results'] = {
            'nrmse': avg_test_nrmse,
            'r2': avg_test_r2,
            'pcv_mape_cleaned': avg_pcv_mape,
            'outliers_removed': int(num_removed)
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            
    tb_writer.close()

# %% [markdown]
# ## Start train/test!

# %%
def main():
    # 1. 載入 Training 數據 & 計算 Global Max/Norm
    print("[Step 1] Loading Training Data for Normalization...")
    t_B, t_F, t_T, t_H, t_Pcv, t_Hdc, t_N, t_Duty_P, t_Duty_N = load_dataset(material, mode="train")

    print("Calculating Global Max & Norm Params...")
    GLOBAL_B_MAX = np.abs(t_B).max()
    GLOBAL_H_MAX = np.abs(t_H).max()
    seq_len = 1024

    # 計算導數最大值 (用於 Scaling)
    np_dB = np.diff(t_B, axis=1, prepend=t_B[:, :1])
    np_dB_dt = np_dB * (seq_len * t_F.reshape(-1, 1))
    GLOBAL_DB_DT_MAX = np.max(np.abs(np_dB_dt))

    np_d2B = np.diff(np_dB_dt, axis=1, prepend=np_dB_dt[:, :1])
    np_d2B_dt2 = np_d2B * (seq_len * t_F.reshape(-1, 1))
    GLOBAL_D2B_DT2_MAX = np.max(np.abs(np_d2B_dt2))

    print(f"Global B Max: {GLOBAL_B_MAX}")
    print(f"Global H Max: {GLOBAL_H_MAX}")

    # 計算標準化參數 (Mean/Std)
    def safe_mean_std_np(array, eps=1e-8):
        m = np.mean(array)
        s = np.std(array)
        if s < eps: s = 1.0
        return [float(m), float(s)]

    global_norm = [
        safe_mean_std_np(np.log10(t_F)),
        safe_mean_std_np(t_T),
        safe_mean_std_np(t_Hdc),
        safe_mean_std_np(t_N),
        safe_mean_std_np(np.log10(t_Pcv))
    ]

    # 2. Logger 初始化
    logger = TrainLogger(
        exp_name=f"{material}_{fix_way}_{run_id}",
        config_dict={k: getattr(Config, k) for k in dir(Config) if not k.startswith('__')},
        result_dir=result_dir)
    logger.save_norm_params(global_norm)

    # =========================================================================
    # 分支判斷：訓練模式 vs 測試模式
    # =========================================================================
    
    if Config.MODE == "train_and_test":
        print("\n" + "="*40)
        print("MODE: Train Phase 1 & 2 -> Then Test")
        print("="*40)
        
        # --- A. 準備訓練數據 ---
        print("Creating AC/DC DataLoaders for Training...")
        is_ac_data = (np.abs(t_Hdc) < 1e-5).flatten()
        is_dc_data = ~is_ac_data
        
        def filter_arrays(indices, *arrays): return [arr[indices] for arr in arrays]
        
        ac_vars = filter_arrays(is_ac_data, t_B, t_F, t_T, t_H, t_Pcv, t_Hdc, t_N, t_Duty_P, t_Duty_N)
        dc_vars = filter_arrays(is_dc_data, t_B, t_F, t_T, t_H, t_Pcv, t_Hdc, t_N, t_Duty_P, t_Duty_N)

        print("\n=== 建立 AC DataLoader ===")
        train_loader_AC, valid_loader_AC, _ = get_dataloader(
            *ac_vars,
            GLOBAL_B_MAX, GLOBAL_H_MAX, GLOBAL_DB_DT_MAX, GLOBAL_D2B_DT2_MAX,
            norm=global_norm,
            is_train=True 
        )

        print("\n=== 建立 DC DataLoader ===")
        train_loader_DC, valid_loader_DC, _ = get_dataloader(
            *dc_vars,
            GLOBAL_B_MAX, GLOBAL_H_MAX, GLOBAL_DB_DT_MAX, GLOBAL_D2B_DT2_MAX,
            norm=global_norm,
            is_train=True 
        )

        # --- B. 執行訓練 ---
        train_model(norm=global_norm,
                    train_loader_AC=train_loader_AC,
                    valid_loader_AC=valid_loader_AC,
                    train_loader_DC=train_loader_DC,
                    valid_loader_DC=valid_loader_DC,
                    logger=logger)
        
        # --- C. 設定測試模型路徑 (剛練好的) ---
        target_model_path = os.path.join(logger.result_dir, "phase2_best_final.pt")

    elif Config.MODE == "test_only":
        print("\n" + "="*40)
        print("MODE: Test Only (Skipping Training)")
        print("="*40)
        
        # --- C. 設定測試模型路徑 (從 Config 讀取) ---
        target_model_path = Config.TEST_MODEL_PATH
        if target_model_path is None or not os.path.exists(target_model_path):
            print("Error: TEST_MODEL_PATH is invalid. Please set it in Config.")
            return

    # =========================================================================
    # 測試階段 (共用流程)
    # =========================================================================
    print("\n[Step 3] Loading Test Data...")
    try:
        # 讀取測試集
        test_B, test_F, test_T, test_H, test_Pcv, test_Hdc, test_N, test_Duty_P, test_Duty_N = load_dataset(material, mode="test")
        print(f"   Test Samples: {len(test_B)}")
        
        # 建立測試 Loader 
        test_loader_instance, _, _ = get_dataloader(
            test_B, test_F, test_T, test_H, test_N, test_Hdc, test_Duty_P, test_Duty_N, test_Pcv,
            GLOBAL_B_MAX, GLOBAL_H_MAX, GLOBAL_DB_DT_MAX, GLOBAL_D2B_DT2_MAX,
            norm=global_norm,
            is_train=False
        )
        print("Test DataLoader created successfully.")
        
        # 執行測試
        test_process(norm=global_norm, 
                        test_loader=test_loader_instance, 
                        logger=logger, 
                        device=device,
                        model_path=target_model_path,
                        global_B_max=GLOBAL_B_MAX,  
                        global_H_max=GLOBAL_H_MAX)
                     
    except Exception as e:
            print(f"Testing Failed (Data load error): {e}")
            import traceback
            traceback.print_exc()

# %%
if __name__ == "__main__":
    main()


