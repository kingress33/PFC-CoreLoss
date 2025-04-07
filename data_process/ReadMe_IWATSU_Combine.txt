# 步驟 2：數據整合

## 目標
本步驟將所有 IWATSU 測試數據進行清理與整合，輸出為單一 CSV (`combined_{date}.csv`)，供後續機器學習使用。

## 使用方法

### 1. **準備資料**
- **確保 IWATSU 測試數據已放置於 `./summary/` 資料夾**
  - IWATSU 產出的 CSV 檔案需存放在 `summary/` 目錄下
  - 每個檔案包含不同測試條件的數據
  - 檔案範例：
    ```
    summary/
    ├── Test_1.csv
    ├── Test_2.csv
    ├── Test_3.csv
    ```

### 2. **執行 Python 腳本**
- 執行以下指令：
  ```bash
  python step2_combine_data.py
