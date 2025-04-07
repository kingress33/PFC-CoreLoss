# %% [markdown]
# # Part 1: Network Training

# %% [markdown]
# ## Step0: Import Package & Hyperparameter Configuration

# %% [markdown]
# ### Package

# %%
%reset -f

import os
import torch
import numpy as np
import random
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# ### Hyperparameter Config

# %%
# %%
# Unified Hyperparameter Configuration
class Config:
    SEED = 1
    NUM_EPOCHS = 3000
    BATCH_SIZE = 128
    LEARNING_RATE = 0.002
    LR_SCHEDULER_GAMMA = 0.99
    DECAY_EPOCH = 200
    DECAY_RATIO = 0.5
    EARLY_STOPPING_PATIENCE = 500
    HIDDEN_SIZE = 30
    OPERATOR_SIZE = 30


# Reproducibility
random.seed(Config.SEED)
np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# %% [markdown]
# ### Material & Number of Data

# %%
material = "CH467160"
data_number = 300
downsample = 1024

# 定義保存模型的路徑
model_save_dir = "./Model/"
os.makedirs(model_save_dir, exist_ok=True)  # 如果路徑不存在，創建路徑

# 定義模型保存檔名
model_save_path = os.path.join(model_save_dir, f"{material}_{downsample}.pt")

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# ## Step1: Data processing and data loader generate 

# %%
# %% Preprocess data into a data loader
def get_dataloader(data_B,
                   data_F,
                   data_T,
                   data_H,
                   data_N,
                   data_Hdc,
                   data_Pcv,
                   norm,
                   n_init=16):
    """ #*(Date:250105)
    Process data and return DataLoader for training, validation, and testing.

    Parameters
    ----------
    data_B : np.array
        Magnetic flux density data.
    data_F : np.array
        Frequency data.
    data_T : np.array
        Temperature data.
    data_N : np.array
        Turns data.
    data_Hdc : np.array
        DC Magnetic field strength data.
    data_H : np.array
        AC Magnetic field strength data.
    data_Pcv : np.array
        Core loss data.
    norm : list
        Normalization parameters for the features.
    n_init : int
        Number of initial data points for magnetization.

    Returns
    -------
        train_loader, valid_loader : DataLoader
        Dataloaders for training, validation
        norm
    """

    # Data pre-process
    # 1. Down-sample to 128 points
    seq_length = downsample
    cols = range(
        0, 8192, int(8192 / seq_length)
    )  #range(start, stop, step) #*  Add  Down-sample: 8192 to 128 points (Date:241213)
    data_B = data_B[:, cols]
    data_H = data_H[:, cols]  #*  Add H Down-sample to 128 points (Date:241213)

    # 2. Add extra points for initial magnetization calculation
    data_length = seq_length + n_init
    data_B = np.hstack((data_B, data_B[:, :n_init]))
    data_H = np.hstack(
        (data_H, data_H[:, :n_init]))  #*(Date:241216) MMINN output似乎是128點
    #*(Date:250130) 原始MMINN H有包含n_init

    data_B = data_B - np.mean(data_B, axis=1,
                              keepdims=True)  #*  移除降階影響 (Date:250325)
    data_H = data_H - np.mean(data_H, axis=1, keepdims=True)

    # 3. Format data into tensors  #*(Date:241216) seq_length=128, data_length=144
    B = torch.from_numpy(data_B).view(-1, data_length, 1).float()
    H = torch.from_numpy(data_H).view(-1, data_length, 1).float()
    F = torch.log10(torch.from_numpy(data_F).view(-1, 1).float())
    T = torch.from_numpy(data_T).view(-1, 1).float()
    Hdc = torch.from_numpy(data_Hdc).view(-1, 1).float()
    N = torch.from_numpy(data_N).view(-1, 1).float()
    # Pcv = torch.log10(torch.from_numpy(data_Pcv).view(-1, 1).float())
    Pcv = torch.from_numpy(data_Pcv).view(-1, 1).float()

    # 原本在6. 因要先計算標準化故移至這
    dB = torch.diff(B, dim=1)
    dB = torch.cat((dB[:, 0:1], dB), dim=1)
    dB_dt = dB * (seq_length * F.view(-1, 1, 1))

    #  4. Compute normalization parameters (均值 & 標準差)**

    # 5. Data Normalization
    in_B = (B - norm[0][0]) / norm[0][1]  # B
    out_H = (H - norm[1][0]) / norm[1][1]  # H
    in_F = (F - norm[2][0]) / norm[2][1]  # F
    in_T = (T - norm[3][0]) / norm[3][1]  # T
    in_Pcv = (Pcv - norm[5][0]) / norm[5][1]  # Pcv
    in_Hdc = (Hdc - norm[6][0]) / norm[6][1]  # Hdc
    in_N = (N - norm[7][0]) / norm[7][1]  # N

    # 6. Extra features

    in_dB = torch.diff(B, dim=1)
    in_dB = torch.cat((in_dB[:, 0:1], in_dB), dim=1)

    in_dB_dt = (dB_dt - norm[4][0]) / norm[4][1]

    max_B, _ = torch.max(in_B, dim=1)
    min_B, _ = torch.min(in_B, dim=1)

    s0 = get_operator_init(in_B[:, 0] - in_dB[:, 0], in_dB, max_B, min_B)

    # 7. Create dataloader to speed up data processing
    test_dataset = torch.utils.data.TensorDataset(
        torch.cat((in_B, in_dB, in_dB_dt), dim=2),  # B 部分（144 點）
        torch.cat((in_F, in_T, in_N, in_Hdc, in_Pcv), dim=1),  # 輔助變量
        s0,  # 初始狀態
        out_H  # 目標值 H（128 點）
    )

    # Split dataset into train, validation, and test sets (60:20:20)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=Config.BATCH_SIZE,
                                              shuffle=False,
                                              num_workers=0,
                                              collate_fn=filter_input,
                                              drop_last=False)

    return test_loader


# %% Predict the operator state at t0
def get_operator_init(B1,
                      dB,
                      Bmax,
                      Bmin,
                      max_out_H=5,
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
    inputs, features, s0, target_H = zip(*batch)

    # 如果 inputs 是 tuple，先堆疊成張量
    inputs = torch.stack(inputs)  # B 的所有輸入部分（144 點）

    # 保留 in_B, in_dB, in_dB_dt 作為模型輸入
    inputs = inputs[:, :, :3]

    # 保留 features（包括 in_F 和 in_T）
    features = torch.stack(
        features
    )[:, :4]  #!(250317)保留 in_F, in_T, in_Hdc, in_N (排除 in_Pcv，in_Pcv要放在最面)

    # 保留目標值 H
    target_H = torch.stack(target_H)[:, -downsample:, :]  # 只取最後 128 點

    s0 = torch.stack(s0)  # 初始狀態

    return inputs, features, s0, target_H


# %% [markdown]
# ### Material normalization data

# %%
# %%
# Material normalization data (0:B,1:H,2:F,3:T,4:dB/dt,5:Pcv,6:Hdc,7:N)
normsDict = {
    "CH467160": [
        [0.00030208364478312433, 0.027951346710324287],
        [2.0709943771362305, 153.72361755371094],
        [2.0, 1.0],
        [25.0, 1.0],
        [0.025416461750864983, 0.452995628118515],
        [1.8068293333053589, 0.7426784038543701],
        [1200.8160400390625, 708.39208984375],
        [10.0, 1.0],
    ]
}
'''
# ALL data
    "CH467160": [
        [0.0012138759484514594, 0.028327999636530876],
        [7.418940544128418, 156.20217895507812],
        [2.0, 1.0],
        [25.0, 1.0],
        [0.04440629482269287, 0.45276743173599243],
        [1.8324720859527588, 0.7202332019805908],
        [1196.234375, 695.8637084960938],
        [10.0, 1.0],
        ]
#ALLData/128點/扣掉降階後bias  
    "CH467160": [
        [2.879451399540045e-10, 0.028311699628829956],
        [0.0, 156.0928192138672],
        [2.0, 1.0],
        [25.0, 1.0],
        [0.04440630227327347, 0.45276743173599243],
        [1.8324720859527588, 0.7202332019805908],
        [1196.234375, 695.8637084960938],
        [10.0, 1.0],
    ]
    
#ALLData/1024點/無扣掉降階後bias  
    "CH467160": [
        [-1.993466364202945e-10, 0.02925829403102398],
        [-2.449571354645741e-07, 160.9007568359375],
        [2.0, 1.0],
        [25.0, 1.0],
        [0.00668413657695055, 0.4514836370944977],
        [1.8324720859527588, 0.7202332019805908],
        [1196.234375, 695.8637084960938],
        [10.0, 1.0],
    ]
        
        

'''

# %% [markdown]
# ## Step2: Define Network Structure

# %%
# %% Magnetization mechansim-determined neural network
"""
    Parameters:
    - hidden_size: number of eddy current slices (RNN neuron)
    - operator_size: number of operators
    - input_size: number of inputs (1.B 2.dB 3.dB/dt)
# ! - var_size: number of supplenmentary variables (1.F 2.T 3.Hdc 4.N)        
    - output_size: number of outputs (1.H)
"""


class MMINet(nn.Module):
    def __init__(
            self,
            Material,  #*這裡改成從外部傳入 norm(250203)
            hidden_size=Config.HIDDEN_SIZE,
            operator_size=Config.OPERATOR_SIZE,
            input_size=3,
            var_size=4,
            output_size=1):
        super().__init__()
        self.input_size = input_size
        self.var_size = var_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.operator_size = operator_size
        self.norm = normsDict[Material]

        self.rnn1 = StopOperatorCell(self.operator_size)
        self.dnn1 = nn.Linear(self.operator_size + 4,
                              1)  #!250317更新：operator_size + 4
        self.rnn2 = EddyCell(
            6, self.hidden_size,
            output_size)  #!250317更新：4 (F, T, B, dB/dt ) + 2 (Hdc, N)
        self.dnn2 = nn.Linear(self.hidden_size, 1)
        self.rnn2_hx = None

    def forward(self, x, var, s0, n_init=16):
        """
         Parameters: 
          - x(batch,seq,input_size): Input features (1.B, 2.dB, 3.dB/dt)  
# !       - var(batch,var_size): Supplementary inputs (1.F 2.T 3.Hdc 4.N) 
          - s0(batch,1): Operator inital states
        """
        batch_size = x.size(0)  # Batch size
        seq_size = x.size(1)  # Ser
        self.rnn1_hx = s0

        # Initialize DNN2 input (1.B 2.dB/dt)
        x2 = torch.cat((x[:, :, 0:1], x[:, :, 2:3]), dim=2)

        for t in range(seq_size):
            # RNN1 input (dB,state)
            self.rnn1_hx = self.rnn1(x[:, t, 1:2], self.rnn1_hx)

            # DNN1 input (rnn1_hx,F,T,Hdc,N)
            dnn1_in = torch.cat((self.rnn1_hx, var), dim=1)

            # H hysteresis prediction
            H_hyst_pred = self.dnn1(dnn1_in)

            # DNN2 input (B,dB/dt,T,F)
            rnn2_in = torch.cat((x2[:, t, :], var), dim=1)

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

        B = (x[:, n_init:, 0:1] * self.norm[0][1] + self.norm[0][0])
        H = (output[:, n_init:, :] * self.norm[1][1] + self.norm[1][0])
        Pcv = torch.trapz(H, B, axis=1) * (10**(var[:, 0:1] * self.norm[2][1] +
                                                self.norm[2][0]))

        return torch.flatten(Pcv), H, B


class StopOperatorCell():
    def __init__(self, operator_size):
        self.operator_thre = torch.from_numpy(
            np.linspace(5 / operator_size, 5, operator_size)).view(1, -1)

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


# %% [markdown]
# ## Step3: Training the Model

# %% [markdown]
# ### Load Dataset

# %%
# Load Data
def load_dataset(material, base_path="./Data/"):

    in_file1 = f"{base_path}{material}/B_Field.csv"
    in_file2 = f"{base_path}{material}/Frequency.csv"
    in_file3 = f"{base_path}{material}/Temperature.csv"
    in_file4 = f"{base_path}{material}/H_Field.csv"
    in_file5 = f"{base_path}{material}/Volumetric_Loss.csv"
    in_file6 = f"{base_path}{material}/Hdc.csv"  # *250317新增：直流偏置磁場
    in_file7 = f"{base_path}{material}/Turns.csv"  # *250317新增：匝數

    data_B = np.genfromtxt(in_file1, delimiter=',')  # N x 1024
    data_F = np.genfromtxt(in_file2, delimiter=',')  # N x 1
    data_T = np.genfromtxt(in_file3, delimiter=',')  # N x 1
    data_H = np.genfromtxt(in_file4, delimiter=',')  # N x 1024  # *250317新增
    data_Pcv = np.genfromtxt(in_file5, delimiter=',')  # N x 1
    data_Hdc = np.genfromtxt(in_file6, delimiter=',')  # N x 1  # *250317新增
    data_N = np.genfromtxt(in_file7, delimiter=',')  # N x 1

    # 隨機選取 100 筆資料
    data_size = len(data_B)
    np.random.seed(Config.SEED)  # 設定隨機種子確保可復現
    indices = np.random.choice(data_size, data_number, replace=False)

    # 根據選取的索引提取資料
    selected_B = data_B[indices]
    selected_F = data_F[indices]
    selected_T = data_T[indices]
    selected_H = data_H[indices]
    selected_N = data_N[indices]
    selected_Hdc = data_Hdc[indices]
    selected_Pcv = data_Pcv[indices]

    # **將選取的索引與 Pcv 存入 DataFrame**
    df = pd.DataFrame({'Index': indices, 'Pcv': selected_Pcv.flatten()})

    # **將 DataFrame 匯出為 CSV**
    output_path = f"./Output/{material}_measured.csv"
    df.to_csv(output_path, index=False,
              header=False)  # `index=False` 避免 pandas 產生額外索引
    print(f"取出的資料已匯出至 {output_path}")

    return selected_B, selected_F, selected_T, selected_H, selected_Pcv, selected_Hdc, selected_N


# %% [markdown]
# ### Test Code

# %%
def test_model(test_loader, measured_Pcv, original_B, original_H):
    """
    test_loader: 測試集 DataLoader
    measured_Pcv: 原始測量的損耗 (Pcv)
    original_B, original_H: 原始未降階的 B 與 H（用於參考繪圖）
    """
    model = MMINet(Material=material).to(device)
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    yy_pred_list = []
    yy_pred_H_list = []
    yy_gt_H_list = []  # 測量的 H
    down_B_list = []  # 降階後的 B

    with torch.no_grad():
        for inputs, features, s0, target_H in test_loader:
            inputs, features, s0, target_H = inputs.to(device), features.to(
                device), s0.to(device), target_H.to(device)
            yy_pred, yy_pred_H, down_B = model(inputs, features, s0)
            yy_pred_list.append(yy_pred.cpu().numpy())
            yy_pred_H_list.append(yy_pred_H.cpu().numpy())
            down_B_list.append(down_B.cpu().numpy())
            yy_gt_H_list.append(target_H.cpu().numpy())

    yy_pred = np.concatenate(yy_pred_list, axis=0)
    yy_pred_H = np.concatenate(yy_pred_H_list,
                               axis=0).reshape(-1, yy_pred_H_list[0].shape[1])
    yy_pred_H = yy_pred_H - np.mean(yy_pred_H, axis=1,
                                    keepdims=True)  #!輸出扣掉平均值

    down_B = np.concatenate(down_B_list,
                            axis=0).reshape(-1, down_B_list[0].shape[1])
    yy_gt_H = np.concatenate(yy_gt_H_list,
                             axis=0).reshape(-1, yy_pred_H.shape[1])

    mean_H = model.norm[1][0]
    std_H = model.norm[1][1]
    yy_gt_H = yy_gt_H * std_H + mean_H
    measured_Pcv = measured_Pcv[:yy_pred.shape[0]]

    # 輸出損耗誤差資訊
    Error_re = abs(yy_pred - measured_Pcv) / abs(measured_Pcv) * 100

    print(f"Relative Error: {np.mean(Error_re):.8f}")
    print(f"AVG Error: {np.mean(Error_re):.8f}")
    print(f"95-PRCT Error: {np.percentile(Error_re, 95):.8f}")
    print(f"99th Percentile Error: {np.percentile(Error_re, 99):.8f}")
    print(f"MAX Error: {np.max(Error_re):.8f}")
    print(f"MIN Error: {np.min(Error_re):.8f}")

    plt.figure(figsize=(10, 5))
    plt.plot(yy_pred, label="Predicted Pcv", linestyle="-", marker='x')
    plt.plot(measured_Pcv, label="Actual Pcv", linestyle="--", marker='x')
    plt.legend()
    plt.xlabel("Test Samples")
    plt.ylabel("Pcv")
    plt.title("Comparison of Predicted and Actual Core Loss")
    plt.show()

    # log 空間誤差
    rel_err_log = np.abs(
        np.log10(yy_pred + 1e-12) - np.log10(measured_Pcv + 1e-12)) * 100
    print(f"Relative Log Error: {np.mean(rel_err_log):.8f}")
    print(f"AVG Log Error: {np.mean(rel_err_log):.8f}")
    print(f"95-PRCT Log Error: {np.percentile(rel_err_log, 95):.8f}")
    print(f"99th Percentile Log Error: {np.percentile(rel_err_log, 99):.8f}")
    print(f"MAX Log Error: {np.max(rel_err_log):.8f}")
    print(f"MIN Log Error: {np.min(rel_err_log):.8f}")

    # ================= 新增繪圖 =================
    # # (a) 原始資料：x 軸 B，y 軸 H
    # for i in range(original_B.shape[0]):
    #     plt.figure(figsize=(8, 4))
    #     plt.plot(original_B[i, :], original_H[i, :], marker='o', markersize=1, label="Original H")
    #     plt.xlabel("B")
    #     plt.ylabel("H")
    #     plt.title(f"Original Data Sample #{i} (B vs H)")
    #     plt.legend()
    #     plt.grid(alpha=0.3)
    #     plt.show()

    # # (b) 降階後資料（預測）：x 軸降階後的 B，y 軸預測 H
    # for i in range(down_B.shape[0]):
    #     plt.figure(figsize=(8, 4))
    #     plt.plot(down_B[i, :], yy_pred_H[i, :], marker='x', markersize=1, label="Predicted H")
    #     plt.xlabel("B")
    #     plt.ylabel("H")
    #     plt.title(f"Downsampled Data Sample #{i} - B vs Predicted H")
    #     plt.legend()
    #     plt.grid(alpha=0.3)
    #     plt.show()

    # # (c) 降階後資料（測量）：x 軸降階後的 B，y 軸測量 H
    # for i in range(down_B.shape[0]):
    #     plt.figure(figsize=(8, 4))
    #     plt.plot(down_B[i, :], yy_gt_H[i, :], marker='x', markersize=3, label="Measured H")
    #     plt.xlabel("B")
    #     plt.ylabel("H")
    #     plt.title(f"Downsampled Data Sample #{i} - B vs Measured H")
    #     plt.legend()
    #     plt.grid(alpha=0.3)
    #     plt.show()
    # ================= 新增繪圖結束 =================

    # 儲存預測結果 (保留原有部分)
    df = pd.DataFrame(yy_pred)
    df_H = pd.DataFrame(yy_pred_H)
    df_H_target = pd.DataFrame(yy_gt_H)
    output_path = f"./Output/{material}_predictions.csv"
    df.to_csv(output_path, index=False, header=False)
    outputH_path = f"./Output/{material}_predictions_H.csv"
    df_H_target.to_csv(outputH_path, index=False, header=False)
    meas_H_path = f"./Output/{material}_measured_H.csv"
    df_H.to_csv(meas_H_path, index=False, header=False)
    print(f"預測結果已匯出至 {output_path}")

    # ================= 修改區結束 =================

    # ================= 三種疊圖 (Overlay Plot: Original, Predicted, Measured) =================
    for i in range(original_B.shape[0]):
        plt.figure(figsize=(8, 4))
        # 原始未降階：直接使用 load_dataset 回傳的資料 (B, H)
        plt.plot(original_H[i, :],
                 original_B[i, :],
                 label="Original (Raw)",
                 alpha=0.5)
        # 降階後的：B 與預測 H
        plt.plot(yy_pred_H[i, :],
                 down_B[i, :],
                 marker='x',
                 markersize=1,
                 label="Predicted H (Downsampled)")
        # 降階後的：B 與測量 H
        plt.plot(yy_gt_H[i, :],
                 down_B[i, :],
                 marker='s',
                 markersize=1,
                 label="Measured H (Downsampled)",
                 alpha=0.7)

        plt.xlabel("H")
        plt.ylabel("B")
        plt.title(f"Overlay Plot Sample #{i}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()
    # ================= 三種疊圖結束 =================

# %% [markdown]
# ### Start Test!!!

# %%
if __name__ == "__main__":

    # 取得原始資料（未降階）
    data_B, data_F, data_T, data_H, data_Pcv, data_Hdc, data_N = load_dataset(
        material)

    norm = normsDict[material]

    test_loader = get_dataloader(data_B, data_F, data_T, data_H, data_N,
                                 data_Hdc, data_Pcv, norm)

    test_model(test_loader, data_Pcv, data_B, data_H)


