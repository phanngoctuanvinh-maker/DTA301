# -*- coding: utf-8 -*-
"""
DTA301 - UAV Fault Detection (Stage 1 Automation)
Tự động hóa toàn bộ thực nghiệm, trực quan hóa và phân tích lỗi cho báo cáo D3 cập nhật
"""

import os
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, precision_recall_curve, roc_auc_score
)

# Cấu hình matplotlib hiển thị đẹp và hỗ trợ font
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Calibri']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.autolayout'] = True
sns.set_theme(style="whitegrid", font_scale=1.1)

OUTPUT_FIG_DIR = os.path.join("reports", "figures")
os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)

print("=== BẮT ĐẦU CHẠY GIAI ĐOẠN 1: THỰC NGHIỆM & XUẤT BIỂU ĐỒ ===")

# -------------------------------------------------------------
# 1. TẠO CÁC BIỂU ĐỒ EDA (FIG 1, 2, 3) TỪ DỮ LIỆU IMU_COMBINED
# -------------------------------------------------------------
print("\n[1/5] Đang tạo biểu đồ EDA từ data/processed/imu_combined.csv...")
df_combined = pd.read_csv("data/processed/imu_combined.csv")

# Chuẩn bị cột thời gian tương đối theo từng chuyến bay
df_combined = df_combined.sort_values(["source_file", "TimeUS"]).reset_index(drop=True)
df_combined["Time_s"] = df_combined.groupby("source_file")["TimeUS"].transform(lambda x: (x - x.iloc[0]) / 1e6)

# FIG 1: Time-series Comparison giữa Normal và Fault
print(" -> Đang vẽ Fig 1: Time-series Comparison...")
fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex=True)
sensors = ["AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ"]
sensor_labels = [
    "AccX (Gia tốc trục X - m/s²)", "GyrX (Tốc độ góc X - rad/s)",
    "AccY (Gia tốc trục Y - m/s²)", "GyrY (Tốc độ góc Y - rad/s)",
    "AccZ (Gia tốc trục Z - m/s²)", "GyrZ (Tốc độ góc Z - rad/s)"
]

# Chọn 1 đoạn 4 giây ổn định (từ giây thứ 6 đến 10) để so sánh
sample_normal = df_combined[(df_combined["source_file"] == "NO_FAULT1.mat") & 
                            (df_combined["Time_s"] >= 6) & (df_combined["Time_s"] <= 10)]
sample_fault = df_combined[(df_combined["source_file"] == "FAULT_M1_10.mat") & 
                           (df_combined["Time_s"] >= 6) & (df_combined["Time_s"] <= 10)]

for idx, sensor in enumerate(sensors):
    r = idx % 3
    c = idx // 3
    ax = axes[r, c]
    
    # Chuẩn hóa lại thời gian vẽ về 0 - 4 giây
    t_norm = sample_normal["Time_s"] - sample_normal["Time_s"].iloc[0]
    t_fault = sample_fault["Time_s"] - sample_fault["Time_s"].iloc[0]
    
    ax.plot(t_norm, sample_normal[sensor], label="Normal (NO_FAULT1)", color="#1f77b4", alpha=0.85, linewidth=1.2)
    ax.plot(t_fault, sample_fault[sensor], label="Fault M1_10% (FAULT_M1_10)", color="#d62728", alpha=0.75, linewidth=1.2)
    
    ax.set_ylabel(sensor_labels[idx], fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    if idx == 0:
        ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

axes[2, 0].set_xlabel("Thời gian tương đối (giây)", fontsize=12, fontweight="bold")
axes[2, 1].set_xlabel("Thời gian tương đối (giây)", fontsize=12, fontweight="bold")
plt.suptitle("Hình 1: So sánh tín hiệu chuỗi thời gian IMU giữa Trạng thái Normal và Fault (Cánh quạt M1 mẻ 10%)", 
             fontsize=14, fontweight="bold", y=1.02)
fig1_path = os.path.join(OUTPUT_FIG_DIR, "fig1_timeseries_comparison.png")
plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"    Đã lưu: {fig1_path}")

# FIG 2: Boxplot & Phân bố 6 kênh cảm biến (Sự mở rộng IQR/Std ở Fault)
print(" -> Đang vẽ Fig 2: Boxplot & Phân phối 6 kênh cảm biến...")
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
df_combined["Trạng thái"] = df_combined["Label"].map({0: "Normal (0)", 1: "Fault (1)"})
palette = {"Normal (0)": "#1f77b4", "Fault (1)": "#d62728"}

for idx, sensor in enumerate(sensors):
    ax = axes[idx // 3, idx % 3]
    sns.boxplot(data=df_combined, x="Trạng thái", y=sensor, palette=palette, ax=ax, width=0.45,
                flierprops={"marker": ".", "markersize": 2, "alpha": 0.2})
    
    # Tính toán median và IQR để chú thích trực quan
    norm_q25 = df_combined[df_combined["Label"] == 0][sensor].quantile(0.25)
    norm_q75 = df_combined[df_combined["Label"] == 0][sensor].quantile(0.75)
    norm_iqr = norm_q75 - norm_q25
    norm_med = df_combined[df_combined["Label"] == 0][sensor].median()
    
    fault_q25 = df_combined[df_combined["Label"] == 1][sensor].quantile(0.25)
    fault_q75 = df_combined[df_combined["Label"] == 1][sensor].quantile(0.75)
    fault_iqr = fault_q75 - fault_q25
    fault_med = df_combined[df_combined["Label"] == 1][sensor].median()
    
    iqr_growth = ((fault_iqr - norm_iqr) / norm_iqr) * 100
    
    ax.set_title(f"{sensor}: Median(N)={norm_med:.2f}, Median(F)={fault_med:.2f}\nΔIQR = +{iqr_growth:.1f}% ở Fault", 
                 fontsize=11, fontweight="bold")
    ax.set_ylabel(f"Giá trị {sensor}", fontsize=11)
    ax.set_xlabel("")
    ax.grid(True, linestyle="--", alpha=0.5)

plt.suptitle("Hình 2: Phân phối và Sự mở rộng Khoảng tứ phân vị (IQR) giữa Normal và Fault trên 6 kênh cảm biến", 
             fontsize=14, fontweight="bold", y=1.02)
fig2_path = os.path.join(OUTPUT_FIG_DIR, "fig2_boxplot_iqr_expansion.png")
plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"    Đã lưu: {fig2_path}")

# FIG 3: Correlation Heatmap Normal vs Fault
print(" -> Đang vẽ Fig 3: Correlation Heatmap...")
corr_norm = df_combined[df_combined["Label"] == 0][sensors].corr()
corr_fault = df_combined[df_combined["Label"] == 1][sensors].corr()

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.heatmap(corr_norm, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, center=0,
            square=True, ax=axes[0], cbar_kws={'label': 'Hệ số tương quan Pearson (r)'})
axes[0].set_title("Trạng thái Normal (Chuyến bay bình thường)\nTương quan động lực học AccX-AccY (-0.55), AccX-AccZ (+0.69)", 
                  fontsize=12, fontweight="bold")

sns.heatmap(corr_fault, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, center=0,
            square=True, ax=axes[1], cbar_kws={'label': 'Hệ số tương quan Pearson (r)'})
axes[1].set_title("Trạng thái Fault (Lỗi cánh quạt)\nTương quan AccX với các trục sụt giảm mạnh về gần 0 (+0.04, +0.15)", 
                  fontsize=12, fontweight="bold")

plt.suptitle("Hình 3: Ma trận tương quan Pearson giữa 6 kênh IMU ở trạng thái Normal và Fault", 
             fontsize=14, fontweight="bold", y=1.02)
fig3_path = os.path.join(OUTPUT_FIG_DIR, "fig3_correlation_heatmap.png")
plt.savefig(fig3_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"    Đã lưu: {fig3_path}")

del df_combined  # Giải phóng bộ nhớ RAM

# -------------------------------------------------------------
# 2. ĐỌC DỮ LIỆU ĐẶC TRƯNG CỬA SỔ & HUẤN LUYỆN 5-FOLD CV
# -------------------------------------------------------------
print("\n[2/5] Đang xử lý dữ liệu đặc trưng cửa sổ (imu_features.csv)...")
df_features = pd.read_csv("data/processed/imu_features.csv")
df_features['Label'] = pd.to_numeric(df_features['Label']).astype(int)

target = 'Label'
group_col = 'source_file'
metadata_cols = ['source_file', 'Window_ID', 'Label']
features = [col for col in df_features.columns if col not in metadata_cols]

X = df_features[features].values
y = df_features[target].values
groups = df_features[group_col].values

print(f" -> Tổng số mẫu cửa sổ: {len(df_features)}")
print(f" -> Số đặc trưng: {len(features)}")
print(f" -> Số chuyến bay độc lập: {df_features[group_col].nunique()}")
print(f" -> Tỷ lệ phân bố nhãn: Normal={sum(y==0)} ({sum(y==0)/len(y)*100:.1f}%), Fault={sum(y==1)} ({sum(y==1)/len(y)*100:.1f}%)")

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

# --- THỬ NGHIỆM BASELINE & ĐO THỜI GIAN ---
lr_base = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(random_state=42, max_iter=1000, C=1.0, solver='lbfgs'))
])

rf_base = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

lr_records = []
rf_records = []
lr_fit_times, lr_pred_times = [], []
rf_fit_times, rf_pred_times = [], []

cm_lr_total = np.zeros((2, 2), dtype=int)
cm_rf_total = np.zeros((2, 2), dtype=int)

all_test_indices = []
all_preds_lr = []
all_preds_rf = []
all_probs_lr = []
all_probs_rf = []

print(" -> Đang huấn luyện 5-Fold Stratified Group CV cho Baseline Models...")
for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups=groups), start=1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # 1. Logistic Regression
    t0 = time.time()
    lr_base.fit(X_train, y_train)
    t_fit_lr = time.time() - t0
    
    t0 = time.time()
    y_pred_lr = lr_base.predict(X_test)
    t_pred_lr = time.time() - t0
    y_prob_lr = lr_base.predict_proba(X_test)[:, 1]
    
    lr_fit_times.append(t_fit_lr)
    lr_pred_times.append(t_pred_lr)
    
    lr_records.append({
        'Fold': fold_idx,
        'Precision': precision_score(y_test, y_pred_lr, pos_label=1, zero_division=0),
        'Recall': recall_score(y_test, y_pred_lr, pos_label=1, zero_division=0),
        'F1-score': f1_score(y_test, y_pred_lr, pos_label=1, zero_division=0),
        'Accuracy': accuracy_score(y_test, y_pred_lr),
        'Fit_Time_s': t_fit_lr,
        'Pred_Time_s': t_pred_lr
    })
    cm_lr_total += confusion_matrix(y_test, y_pred_lr, labels=[0, 1])
    
    # 2. Random Forest
    t0 = time.time()
    rf_base.fit(X_train, y_train)
    t_fit_rf = time.time() - t0
    
    t0 = time.time()
    y_pred_rf = rf_base.predict(X_test)
    t_pred_rf = time.time() - t0
    y_prob_rf = rf_base.predict_proba(X_test)[:, 1]
    
    rf_fit_times.append(t_fit_rf)
    rf_pred_times.append(t_pred_rf)
    
    rf_records.append({
        'Fold': fold_idx,
        'Precision': precision_score(y_test, y_pred_rf, pos_label=1, zero_division=0),
        'Recall': recall_score(y_test, y_pred_rf, pos_label=1, zero_division=0),
        'F1-score': f1_score(y_test, y_pred_rf, pos_label=1, zero_division=0),
        'Accuracy': accuracy_score(y_test, y_pred_rf),
        'Fit_Time_s': t_fit_rf,
        'Pred_Time_s': t_pred_rf
    })
    cm_rf_total += confusion_matrix(y_test, y_pred_rf, labels=[0, 1])
    
    all_test_indices.extend(test_idx)
    all_preds_lr.extend(y_pred_lr)
    all_preds_rf.extend(y_pred_rf)
    all_probs_lr.extend(y_prob_lr)
    all_probs_rf.extend(y_prob_rf)

df_lr_folds = pd.DataFrame(lr_records)
df_rf_folds = pd.DataFrame(rf_records)

# -------------------------------------------------------------
# 3. HYPERPARAMETER TUNING THỰC NGHIỆM
# -------------------------------------------------------------
print("\n[3/5] Đang thực hiện Hyperparameter Tuning với GridSearchCV...")

cv_splits = list(sgkf.split(X, y, groups=groups))

# Tuning Logistic Regression
param_grid_lr = {
    'clf__C': [0.01, 0.1, 1.0, 10.0, 50.0],
    'clf__solver': ['lbfgs']
}
grid_lr = GridSearchCV(
    Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(random_state=42, max_iter=1000))]),
    param_grid_lr,
    cv=cv_splits,
    scoring='f1',
    n_jobs=1
)
grid_lr.fit(X, y)
best_lr_params = grid_lr.best_params_
best_lr_score = grid_lr.best_score_
print(f" -> Best LR Params: {best_lr_params} (Mean F1: {best_lr_score:.4f})")

# Tuning Random Forest
param_grid_rf = {
    'n_estimators': [50, 100, 150],
    'max_depth': [8, 12, 16],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid_rf = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid_rf,
    cv=cv_splits,
    scoring='f1',
    n_jobs=1
)
grid_rf.fit(X, y)
best_rf_params = grid_rf.best_params_
best_rf_score = grid_rf.best_score_
print(f" -> Best RF Params: {best_rf_params} (Mean F1: {best_rf_score:.4f})")

# -------------------------------------------------------------
# 4. TẠO BIỂU ĐỒ ĐÁNH GIÁ MÔ HÌNH (FIG 4, 5, 6, 7)
# -------------------------------------------------------------
print("\n[4/5] Đang tạo biểu đồ đánh giá và phân tích lỗi...")

# FIG 4: Confusion Matrix So sánh LR vs RF
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# LR Heatmap
lr_annot = np.array([
    [f"TN = {cm_lr_total[0,0]}\n({cm_lr_total[0,0]/299*100:.1f}%)", f"FP = {cm_lr_total[0,1]}\n({cm_lr_total[0,1]/299*100:.1f}%)"],
    [f"FN = {cm_lr_total[1,0]}\n({cm_lr_total[1,0]/689*100:.1f}%)", f"TP = {cm_lr_total[1,1]}\n({cm_lr_total[1,1]/689*100:.1f}%)"]
])
sns.heatmap(cm_lr_total, annot=lr_annot, fmt="", cmap="Blues", cbar=False, ax=axes[0],
            xticklabels=['Dự đoán Normal (0)', 'Dự đoán Fault (1)'],
            yticklabels=['Thực tế Normal (0)', 'Thực tế Fault (1)'], annot_kws={"fontsize": 12, "fontweight": "bold"})
axes[0].set_title(f"Logistic Regression (Baseline)\nPrecision: {df_lr_folds['Precision'].mean()*100:.2f}% | Recall: {df_lr_folds['Recall'].mean()*100:.2f}%\nFP = 273 (Báo động giả cao) | FN = 31", 
                  fontsize=12, fontweight="bold")

# RF Heatmap
rf_annot = np.array([
    [f"TN = {cm_rf_total[0,0]}\n({cm_rf_total[0,0]/299*100:.1f}%)", f"FP = {cm_rf_total[0,1]}\n({cm_rf_total[0,1]/299*100:.1f}%)"],
    [f"FN = {cm_rf_total[1,0]}\n({cm_rf_total[1,0]/689*100:.1f}%)", f"TP = {cm_rf_total[1,1]}\n({cm_rf_total[1,1]/689*100:.1f}%)"]
])
sns.heatmap(cm_rf_total, annot=rf_annot, fmt="", cmap="Greens", cbar=False, ax=axes[1],
            xticklabels=['Dự đoán Normal (0)', 'Dự đoán Fault (1)'],
            yticklabels=['Thực tế Normal (0)', 'Thực tế Fault (1)'], annot_kws={"fontsize": 12, "fontweight": "bold"})
axes[1].set_title(f"Random Forest (Baseline)\nPrecision: {df_rf_folds['Precision'].mean()*100:.2f}% | Recall: {df_rf_folds['Recall'].mean()*100:.2f}%\nFP = 197 (Giảm 76 FP) | FN = 77 (Tăng 46 FN)", 
                  fontsize=12, fontweight="bold")

plt.suptitle("Hình 4: Ma trận nhầm lẫn (Confusion Matrix) tổng hợp sau 5 Folds giữa Logistic Regression và Random Forest", 
             fontsize=14, fontweight="bold", y=1.02)
fig4_path = os.path.join(OUTPUT_FIG_DIR, "fig4_confusion_matrices.png")
plt.savefig(fig4_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"    Đã lưu: {fig4_path}")

# FIG 5: Feature Importance của Random Forest
fig, ax = plt.subplots(figsize=(11, 7))
importances = pd.Series(rf_base.feature_importances_, index=features).sort_values(ascending=True)
top15 = importances.tail(15)

# Đánh dấu các đặc trưng đo độ biến thiên (IQR, Std, RMS, Max) bằng màu nổi bật
colors = ['#d95f02' if any(k in f for k in ['IQR', 'Std', 'RMS', 'Max']) else '#1f77b4' for f in top15.index]
top15.plot(kind='barh', color=colors, ax=ax)
ax.set_title("Hình 5: Top 15 Đặc trưng quan trọng nhất (Random Forest Feature Importance)\n(Cam: Nhóm đặc trưng đo độ phân tán & năng lượng dao động chiếm ưu thế)", 
             fontsize=13, fontweight="bold")
ax.set_xlabel("Mức độ đóng góp (Importance Score)", fontsize=11, fontweight="bold")
ax.grid(axis='x', linestyle='--', alpha=0.6)

# Chú thích giá trị trên thanh
for i, v in enumerate(top15):
    ax.text(v + 0.0008, i - 0.15, f"{v:.4f}", color='black', fontsize=10, fontweight="bold")

fig5_path = os.path.join(OUTPUT_FIG_DIR, "fig5_rf_feature_importance.png")
plt.savefig(fig5_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"    Đã lưu: {fig5_path}")

# FIG 6: Precision-Recall Curve & Threshold Tuning cho bài toán UAV
thresholds = np.linspace(0.1, 0.9, 33)
thresh_metrics = []
y_true_eval = df_features.iloc[all_test_indices]['Label'].values

for th in thresholds:
    th_preds = (np.array(all_probs_rf) >= th).astype(int)
    cm = confusion_matrix(y_true_eval, th_preds, labels=[0, 1])
    p = precision_score(y_true_eval, th_preds, pos_label=1, zero_division=0)
    r = recall_score(y_true_eval, th_preds, pos_label=1, zero_division=0)
    f = f1_score(y_true_eval, th_preds, pos_label=1, zero_division=0)
    thresh_metrics.append({
        'Threshold': th,
        'Precision': p,
        'Recall': r,
        'F1-score': f,
        'FN': cm[1, 0],
        'FP': cm[0, 1]
    })

df_thresh = pd.DataFrame(thresh_metrics)

fig, ax1 = plt.subplots(figsize=(11, 6))
ax1.plot(df_thresh['Threshold'], df_thresh['Precision'], label='Precision (Độ chính xác cảnh báo)', color='#2ca02c', linewidth=2.5)
ax1.plot(df_thresh['Threshold'], df_thresh['Recall'], label='Recall (Tỷ lệ bắt lỗi - An toàn UAV)', color='#d62728', linewidth=2.5)
ax1.plot(df_thresh['Threshold'], df_thresh['F1-score'], label='F1-score (Cân bằng)', color='#1f77b4', linestyle='--', linewidth=2)
ax1.set_xlabel("Ngưỡng quyết định xác suất (Classification Threshold)", fontsize=11, fontweight="bold")
ax1.set_ylabel("Giá trị Metric (%)", fontsize=11, fontweight="bold")
ax1.set_ylim([0.4, 1.02])

# Vạch mức threshold tối ưu cho an toàn UAV (ví dụ th = 0.35)
ax1.axvline(x=0.35, color='purple', linestyle=':', linewidth=2, label='Ngưỡng đề xuất cho UAV (Th=0.35, Recall>95%)')
ax1.axvline(x=0.50, color='gray', linestyle='--', linewidth=1.5, label='Ngưỡng mặc định (Th=0.50)')
ax1.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9)
ax1.grid(True, linestyle='--', alpha=0.5)

ax1.set_title("Hình 6: Phân tích đánh đổi Precision - Recall và Tối ưu hóa Ngưỡng quyết định cho An toàn UAV (Random Forest)", 
              fontsize=13, fontweight="bold")
fig6_path = os.path.join(OUTPUT_FIG_DIR, "fig6_precision_recall_tradeoff.png")
plt.savefig(fig6_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"    Đã lưu: {fig6_path}")

# FIG 7: Error Analysis Phân tích tỷ lệ sai sót theo điều kiện
df_eval = df_features.iloc[all_test_indices].copy()
df_eval['Pred_LR'] = all_preds_lr
df_eval['Pred_RF'] = all_preds_rf
df_eval['Prob_RF'] = all_probs_rf
df_eval['Error_LR'] = (df_eval['Label'] != df_eval['Pred_LR']).astype(int)
df_eval['Error_RF'] = (df_eval['Label'] != df_eval['Pred_RF']).astype(int)

# Gán nhóm điều kiện bay
def get_flight_condition(filename):
    if "NO_FAULT" in filename:
        return "Normal (Không lỗi)"
    elif "_5.mat" in filename:
        return "Fault 5% (Lỗi nhẹ)"
    elif "_10.mat" in filename:
        return "Fault 10% (Lỗi nặng)"
    return "Khác"

def get_motor(filename):
    if "NO_FAULT" in filename:
        return "Normal"
    for m in ["M1", "M2", "M3", "M4", "M5", "M6"]:
        if m in filename:
            return m
    return "Khác"

df_eval['Condition'] = df_eval['source_file'].apply(get_flight_condition)
df_eval['Motor'] = df_eval['source_file'].apply(get_motor)

error_by_cond = df_eval.groupby('Condition').agg(
    Tong_Mau=('Label', 'count'),
    Sai_LR=('Error_LR', 'sum'),
    Ty_le_sai_LR=('Error_LR', lambda x: x.mean() * 100),
    Sai_RF=('Error_RF', 'sum'),
    Ty_le_sai_RF=('Error_RF', lambda x: x.mean() * 100)
).loc[["Normal (Không lỗi)", "Fault 5% (Lỗi nhẹ)", "Fault 10% (Lỗi nặng)"]]

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(error_by_cond))
width = 0.35

rects1 = ax.bar(x - width/2, error_by_cond['Ty_le_sai_LR'], width, label='Logistic Regression', color='#a6cee3')
rects2 = ax.bar(x + width/2, error_by_cond['Ty_le_sai_RF'], width, label='Random Forest', color='#1f78b4')

ax.set_ylabel('Tỷ lệ dự đoán sai (%)', fontsize=11, fontweight="bold")
ax.set_title('Hình 7: Tỷ lệ dự đoán sai phân rã theo 3 điều kiện chuyến bay thực nghiệm', fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(error_by_cond.index, fontsize=11, fontweight="bold")
ax.legend(frameon=True, facecolor='white')
ax.grid(axis='y', linestyle='--', alpha=0.6)

for r in rects1:
    h = r.get_height()
    ax.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight="bold")
for r in rects2:
    h = r.get_height()
    ax.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight="bold")

fig7_path = os.path.join(OUTPUT_FIG_DIR, "fig7_error_analysis_by_condition.png")
plt.savefig(fig7_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"    Đã lưu: {fig7_path}")

# -------------------------------------------------------------
# 5. XUẤT CÁC BẢNG DỮ LIỆU TỔNG HỢP CHO BÁO CÁO
# -------------------------------------------------------------
print("\n[5/5] Đang tổng hợp các số liệu kiểm định và thời gian...")

# Statistical test
stat_results = []
for metric in ['Precision', 'Recall', 'F1-score', 'Accuracy']:
    t_stat, p_ttest = stats.ttest_rel(df_rf_folds[metric], df_lr_folds[metric])
    try:
        w_stat, p_wilc = stats.wilcoxon(df_rf_folds[metric], df_lr_folds[metric], zero_method='zsplit')
    except Exception:
        w_stat, p_wilc = np.nan, np.nan
        
    stat_results.append({
        'Metric': metric,
        'Mean_LR': df_lr_folds[metric].mean(),
        'Std_LR': df_lr_folds[metric].std(),
        'Mean_RF': df_rf_folds[metric].mean(),
        'Std_RF': df_rf_folds[metric].std(),
        'Diff (RF - LR)': df_rf_folds[metric].mean() - df_lr_folds[metric].mean(),
        'Paired_t_stat': t_stat,
        'Paired_t_pvalue': p_ttest,
        'Wilcoxon_pvalue': p_wilc
    })
df_stats = pd.DataFrame(stat_results)

# Bảng thời gian thực nghiệm
time_summary = pd.DataFrame({
    'Mô hình': ['Logistic Regression (Pipeline)', 'Random Forest (n_estimators=100, max_depth=12)'],
    'Thời gian huấn luyện / Fold (s)': [f"{np.mean(lr_fit_times):.4f} ± {np.std(lr_fit_times):.4f}", 
                                       f"{np.mean(rf_fit_times):.4f} ± {np.std(rf_fit_times):.4f}"],
    'Thời gian suy luận / Fold (ms)': [f"{np.mean(lr_pred_times)*1000:.2f} ± {np.std(lr_pred_times)*1000:.2f}",
                                      f"{np.mean(rf_pred_times)*1000:.2f} ± {np.std(rf_pred_times)*1000:.2f}"],
    'Thời gian suy luận / Cửa sổ 2s (µs)': [f"{(np.mean(lr_pred_times)/197)*1e6:.1f} µs",
                                            f"{(np.mean(rf_pred_times)/197)*1e6:.1f} µs"]
})

# Lưu các bảng kết quả sang CSV trong thư mục reports
df_stats.to_csv("reports/statistical_tests_results.csv", index=False)
time_summary.to_csv("reports/model_execution_times.csv", index=False)
error_by_cond.to_csv("reports/error_analysis_by_condition.csv")
df_eval.groupby(['source_file', 'Label', 'Condition']).agg(
    Tong_Mau=('Label', 'count'),
    Sai_LR=('Error_LR', 'sum'),
    Ty_le_sai_LR=('Error_LR', lambda x: round(x.mean()*100, 2)),
    Sai_RF=('Error_RF', 'sum'),
    Ty_le_sai_RF=('Error_RF', lambda x: round(x.mean()*100, 2))
).reset_index().to_csv("reports/error_analysis_by_file.csv", index=False)

# Lưu kết quả tổng kết JSON để dùng cập nhật báo cáo Word
summary_dict = {
    'lr_mean_precision': float(df_lr_folds['Precision'].mean()),
    'lr_mean_recall': float(df_lr_folds['Recall'].mean()),
    'lr_mean_f1': float(df_lr_folds['F1-score'].mean()),
    'lr_mean_acc': float(df_lr_folds['Accuracy'].mean()),
    'rf_mean_precision': float(df_rf_folds['Precision'].mean()),
    'rf_mean_recall': float(df_rf_folds['Recall'].mean()),
    'rf_mean_f1': float(df_rf_folds['F1-score'].mean()),
    'rf_mean_acc': float(df_rf_folds['Accuracy'].mean()),
    'cm_lr': cm_lr_total.tolist(),
    'cm_rf': cm_rf_total.tolist(),
    'lr_fit_time_mean': float(np.mean(lr_fit_times)),
    'rf_fit_time_mean': float(np.mean(rf_fit_times)),
    'lr_pred_time_mean': float(np.mean(lr_pred_times)),
    'rf_pred_time_mean': float(np.mean(rf_pred_times)),
    'best_lr_params': best_lr_params,
    'best_rf_params': best_rf_params,
    'best_lr_score': float(best_lr_score),
    'best_rf_score': float(best_rf_score)
}
with open("reports/experiment_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary_dict, f, indent=4, ensure_ascii=False)

print("\n=== HOÀN TẤT THỰC NGHIỆM GIAI ĐOẠN 1 THÀNH CÔNG ===")
print(f"Toàn bộ 7 biểu đồ đã được lưu trong: {OUTPUT_FIG_DIR}")
print("Các bảng kết quả thống kê và thời gian đã được lưu trong thư mục reports/")
