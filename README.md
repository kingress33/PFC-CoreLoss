# MMINN: Magnetization Mechanism-Inspired Neural Network for Core Loss Prediction

> **專案描述：** 本專案實現了 MMINN 模型，旨在預測電力電子轉換器（特別是 Boost PFC）中電感的磁芯損耗（Core Loss）。模型特別針對 **DC 偏壓 (DC Bias)** 與 **Duty Cycle 變化** 下的動態磁滯行為進行優化與驗證。

---

## 🛠️ 環境需求 (Requirements)

為了順利執行本專案的訓練與測試程式碼，請確保您的環境滿足以下要求。

### 核心依賴 (Core Dependencies)
本專案主要基於 **Python** 與 **PyTorch** 框架開發。

* **Python**: >= 3.8
* **PyTorch**: >= 1.10 (建議使用支援 CUDA 的版本以加速訓練)
* **Jupyter Lab / Notebook**: 用於執行 `.ipynb` 檔案

### 必要函式庫 (Libraries)
請使用 `pip` 安裝以下套件：

```bash
pip install numpy pandas matplotlib scipy tqdm torch
