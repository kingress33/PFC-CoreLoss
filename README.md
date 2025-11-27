## 🛠️ 環境需求 (Requirements)

本專案基於 **Python 3.11** 與 **PyTorch 2.5** 開發，並採用 **CUDA 12.4** 進行硬體加速。
為了確保重現性 (Reproducibility) 與 NumPy 2.0 的兼容性，請務必滿足以下版本要求。

### 核心系統 (Core System)
* **Python**: 3.11.x
* **CUDA Toolkit**: 12.4 (若使用 GPU 訓練)

### 必要函式庫 (Python Libraries)
請參考以下版本進行安裝，以避免相依性衝突：

| 套件名稱 | 建議版本 | 用途 |
| :--- | :--- | :--- |
| **torch** | `>= 2.5.1` | 深度學習核心框架 |
| **numpy** | `>= 2.0.1` | **注意：本專案使用 NumPy 2.0 新標準** |
| **pandas** | `>= 2.2.3` | 數據處理與 CSV 讀寫 |
| **scipy** | `>= 1.15.2` | 科學運算與積分 |
| **matplotlib** | `>= 3.10.0` | 靜態圖表繪製 |
| **tqdm** | `>= 4.67.1` | 訓練進度條顯示 |

### 進階功能與工具 (Advanced Tools)
本專案包含自動超參數調整與 Web 展示介面，需安裝以下套件：

* **optuna** (`>= 4.4.0`): 用於自動搜尋最佳模型參數 (Hyperparameter Optimization)。
* **streamlit** (`>= 1.51.0`): 用於啟動磁芯損耗預測的互動式 Web App。
* **plotly** (`>= 6.4.0`): 用於繪製互動式 B-H 迴線圖。

### 快速安裝 (Installation)
您可以建立一個 `requirements.txt` 檔案並貼上以下內容，然後執行 `pip install -r requirements.txt`：

```text
python_version>=3.11
torch>=2.5.1
numpy>=2.0.1
pandas>=2.2.3
scipy>=1.15.2
matplotlib>=3.10.0
tqdm>=4.67.1
optuna>=4.4.0
streamlit>=1.51.0
plotly>=6.4.0
scikit-learn>=1.7.1
openpyxl>=3.1.5
