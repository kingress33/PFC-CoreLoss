# %%
import numpy as np
import matplotlib.pyplot as plt
import os


# 載入數據的函數（從你的原始程式碼中提取）
def load_dataset(material, base_path="./Data/"):
    in_file1 = f"{base_path}{material}/train/B_Field.csv"
    in_file2 = f"{base_path}{material}/train/Frequency.csv"
    in_file3 = f"{base_path}{material}/train/Temperature.csv"
    in_file4 = f"{base_path}{material}/train/H_Field.csv"
    in_file5 = f"{base_path}{material}/train/Volumetric_Loss.csv"
    in_file6 = f"{base_path}{material}/train/Hdc.csv"
    in_file7 = f"{base_path}{material}/train/Turns.csv"

    data_B = np.genfromtxt(in_file1, delimiter=',')  # N x 1024
    data_F = np.genfromtxt(in_file2, delimiter=',')  # N x 1
    data_T = np.genfromtxt(in_file3, delimiter=',')  # N x 1
    data_H = np.genfromtxt(in_file4, delimiter=',')  # N x 1024
    data_Pcv = np.genfromtxt(in_file5, delimiter=',')  # N x 1
    data_Hdc = np.genfromtxt(in_file6, delimiter=',')  # N x 1
    data_N = np.genfromtxt(in_file7, delimiter=',')  # N x 1

    return data_B, data_F, data_T, data_H, data_Pcv, data_Hdc, data_N


# 執行相位位移並可視化
def check_phase_shift(material,
                      sample_idx=0,
                      save_figure=True,
                      base_path="./Data/"):
    # 載入數據
    data_B, data_F, data_T, data_H, data_Pcv, data_Hdc, data_N = load_dataset(
        material, base_path)

    # 選擇一筆樣本
    B_orig = data_B[sample_idx].copy()
    H_orig = data_H[sample_idx].copy()

    # 相位位移
    B_90 = np.roll(B_orig, shift=256)  # 90 度 = 1024/4 = 256 點
    H_90 = np.roll(H_orig, shift=256)
    B_180 = np.roll(B_orig, shift=512)  # 180 度 = 1024/2 = 512 點
    H_180 = np.roll(H_orig, shift=512)

    # 創建保存圖表的資料夾
    figure_save_base_path = f"./figure/phase_shift_check/{material}/"
    os.makedirs(figure_save_base_path, exist_ok=True)

    # 可視化波形
    plt.figure(figsize=(12, 8))

    # B 波形
    plt.subplot(2, 2, 1)
    plt.plot(B_orig, label="Original B", alpha=0.8)
    plt.plot(B_90, label="90° Shift B", linestyle="--", alpha=0.8)
    plt.plot(B_180, label="180° Shift B", linestyle=":", alpha=0.8)
    plt.title(f"B Waveform (Sample {sample_idx})")
    plt.xlabel("Time Index")
    plt.ylabel("B (T)")
    plt.legend()
    plt.grid(alpha=0.5)

    # H 波形
    plt.subplot(2, 2, 2)
    plt.plot(H_orig, label="Original H", alpha=0.8)
    plt.plot(H_90, label="90° Shift H", linestyle="--", alpha=0.8)
    plt.plot(H_180, label="180° Shift H", linestyle=":", alpha=0.8)
    plt.title(f"H Waveform (Sample {sample_idx})")
    plt.xlabel("Time Index")
    plt.ylabel("H (A/m)")
    plt.legend()
    plt.grid(alpha=0.5)

    # B-H 回線
    plt.subplot(2, 2, 3)
    plt.plot(B_orig, H_orig, label="Original B-H", alpha=0.8)
    plt.plot(B_90, H_90, label="90° Shift B-H", linestyle="--", alpha=0.8)
    plt.plot(B_180, H_180, label="180° Shift B-H", linestyle=":", alpha=0.8)
    plt.title(f"B-H Loop (Sample {sample_idx})")
    plt.xlabel("B (T)")
    plt.ylabel("H (A/m)")
    plt.legend()
    plt.grid(alpha=0.5)

    # # 放大 B-H 回線的一部分（可選）
    # plt.subplot(2, 2, 4)
    # plt.plot(B_orig, H_orig, label="Original B-H", alpha=0.8)
    # plt.plot(B_90, H_90, label="90° Shift B-H", linestyle="--", alpha=0.8)
    # plt.plot(B_180, H_180, label="180° Shift B-H", linestyle=":", alpha=0.8)
    # plt.title(f"B-H Loop Zoomed (Sample {sample_idx})")
    # plt.xlabel("B (T)")
    # plt.ylabel("H (A/m)")
    # plt.legend()
    # plt.grid(alpha=0.5)
    # # 設定放大範圍（可根據數據調整）
    # plt.xlim(np.min(B_orig) * 0.5, np.max(B_orig) * 0.5)
    # plt.ylim(np.min(H_orig) * 0.5, np.max(H_orig) * 0.5)

    plt.tight_layout()

    # 保存圖表
    if save_figure:
        figure_save_path = os.path.join(
            figure_save_base_path, f"phase_shift_sample_{sample_idx}.svg")
        plt.savefig(figure_save_path, format="svg")
        print(f"Figure saved at: {figure_save_path}")

    plt.show()

    # 檢查 B-H 回線面積（磁芯損耗相關）
    def calculate_loop_area(B, H):
        # 使用梯形積分計算 B-H 回線面積
        return np.abs(np.trapz(H, B))

    area_orig = calculate_loop_area(B_orig, H_orig)
    area_90 = calculate_loop_area(B_90, H_90)
    area_180 = calculate_loop_area(B_180, H_180)

    print(f"B-H Loop Area (Sample {sample_idx}):")
    print(f"Original: {area_orig:.6f}")
    print(f"90° Shift: {area_90:.6f}")
    print(f"180° Shift: {area_180:.6f}")
    print(f"Area Difference (90° vs Orig): {abs(area_90 - area_orig):.6f}")
    print(f"Area Difference (180° vs Orig): {abs(area_180 - area_orig):.6f}")


# 主程式
if __name__ == "__main__":
    material = "CH467160"  # 替換為你的材料名稱
    sample_idx = 25  # 選擇要檢查的樣本索引
    check_phase_shift(material, sample_idx=sample_idx, save_figure=True)
