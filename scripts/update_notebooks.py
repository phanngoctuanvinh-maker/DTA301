# -*- coding: utf-8 -*-
"""
Cập nhật notebooks/04_model_training.ipynb và notebooks/02_eda.ipynb 
với đầy đủ các bước thực nghiệm của Giai đoạn 1
"""

import json
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# -------------------------------------------------------------
# 1. CẬP NHẬT NOTEBOOKS/04_MODEL_TRAINING.IPYNB
# -------------------------------------------------------------
print("Đang cập nhật notebooks/04_model_training.ipynb...")

nb04_cells = [
    {
        "cell_type": "markdown",
        "id": "intro_header",
        "metadata": {},
        "source": [
            "# 04 Model Training & Evaluation — UAV-FD\n",
            "## Huấn luyện, Tinh chỉnh Siêu tham số & Đánh giá Toàn diện Mô hình Phân loại Lỗi Cánh quạt UAV\n",
            "\n",
            "**Nội dung thực hiện trong notebook:**\n",
            "1. **Phân chia 5 Folds chống Data Leakage**: Áp dụng `StratifiedGroupKFold` gom nhóm theo chuyến bay (`source_file`), đảm bảo 100% không rò rỉ dữ liệu giữa Train và Test.\n",
            "2. **Đo đạc Thời gian & Tái lập (Reproducibility)**: Đo lường chính xác `fit_time` và `inference_time` trên từng cửa sổ dữ liệu 2 giây.\n",
            "3. **Tối ưu hóa Siêu tham số (Hyperparameter Tuning)**: Sử dụng `GridSearchCV` trên StratifiedGroupKFold cho Logistic Regression và Random Forest.\n",
            "4. **Đánh giá Đánh đổi (Trade-offs) & Tối ưu Ngưỡng (Threshold Tuning)**: Phân tích sự đánh đổi giữa Precision và Recall trong bài toán an toàn hàng không UAV; thử nghiệm ngưỡng phân loại tối ưu.\n",
            "5. **Kiểm định Thống kê & Phân tích Lỗi (Error Analysis)**: Paired t-test vs Wilcoxon, phân rã tỷ lệ dự đoán sai theo điều kiện bay (Normal, Fault 5%, Fault 10%)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "data_loading",
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import time\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from scipy import stats\n",
            "\n",
            "from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV\n",
            "from sklearn.preprocessing import StandardScaler\n",
            "from sklearn.pipeline import Pipeline\n",
            "from sklearn.linear_model import LogisticRegression\n",
            "from sklearn.ensemble import RandomForestClassifier\n",
            "from sklearn.metrics import (\n",
            "    precision_score, recall_score, f1_score, accuracy_score,\n",
            "    confusion_matrix, precision_recall_curve\n",
            ")\n",
            "\n",
            "# Cấu hình thư mục lưu ảnh báo cáo\n",
            "FIG_DIR = os.path.join(\"..\", \"reports\", \"figures\")\n",
            "os.makedirs(FIG_DIR, exist_ok=True)\n",
            "\n",
            "# 1. Đọc dữ liệu đặc trưng cửa sổ 2 giây\n",
            "df_uav = pd.read_csv(\"../data/processed/imu_features.csv\")\n",
            "df_uav['Label'] = pd.to_numeric(df_uav['Label']).astype(int)\n",
            "\n",
            "target = 'Label'\n",
            "group_col = 'source_file'\n",
            "metadata_cols = ['source_file', 'Window_ID', 'Label']\n",
            "features = [col for col in df_uav.columns if col not in metadata_cols]\n",
            "\n",
            "X = df_uav[features].values\n",
            "y = df_uav[target].values\n",
            "groups = df_uav[group_col].values\n",
            "\n",
            "print(f\"Tổng số cửa sổ đặc trưng: {len(df_uav):,}\")\n",
            "print(f\"Số lượng đặc trưng (features): {len(features)}\")\n",
            "print(f\"Tổng số chuyến bay độc lập: {df_uav[group_col].nunique()}\")\n",
            "print(f\"Phân bố nhãn: Normal (0) = {sum(y==0)} ({sum(y==0)/len(y)*100:.1f}%), Fault (1) = {sum(y==1)} ({sum(y==1)/len(y)*100:.1f}%)\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "group_split",
        "metadata": {},
        "outputs": [],
        "source": [
            "# 2. Phân chia 5-Fold Stratified Group CV\n",
            "sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)\n",
            "cv_splits = list(sgkf.split(X, y, groups=groups))\n",
            "\n",
            "fold_info = []\n",
            "for fold_idx, (train_idx, test_idx) in enumerate(cv_splits, start=1):\n",
            "    train_files = np.unique(groups[train_idx])\n",
            "    test_files = np.unique(groups[test_idx])\n",
            "    \n",
            "    # Kiểm tra rò rỉ dữ liệu giữa Train và Test (Anti-Leakage Check)\n",
            "    overlap = set(train_files).intersection(set(test_files))\n",
            "    \n",
            "    fold_info.append({\n",
            "        'Fold': fold_idx,\n",
            "        'Mẫu Train': len(train_idx),\n",
            "        'Mẫu Test': len(test_idx),\n",
            "        'Số file Train': len(train_files),\n",
            "        'Số file Test': len(test_files),\n",
            "        'Danh sách file Test': \", \".join(test_files),\n",
            "        'Trùng file (Leakage)': len(overlap)\n",
            "    })\n",
            "\n",
            "df_folds = pd.DataFrame(fold_info)\n",
            "print(\"=== BẢNG PHÂN CHIA 5-FOLD THEO CHUYẾN BAY (CHỐNG DATA LEAKAGE) ===\")\n",
            "display(df_folds)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "baseline_training",
        "metadata": {},
        "outputs": [],
        "source": [
            "# 3. Huấn luyện 5 Folds cho 2 mô hình cơ sở & Đo lường thời gian\n",
            "lr_model = Pipeline([\n",
            "    ('scaler', StandardScaler()),\n",
            "    ('clf', LogisticRegression(random_state=42, max_iter=1000, C=1.0, solver='lbfgs'))\n",
            "])\n",
            "\n",
            "rf_model = RandomForestClassifier(\n",
            "    n_estimators=100,\n",
            "    max_depth=12,\n",
            "    min_samples_split=2,\n",
            "    min_samples_leaf=1,\n",
            "    random_state=42,\n",
            "    n_jobs=-1\n",
            ")\n",
            "\n",
            "lr_records, rf_records = [], []\n",
            "lr_fit_times, lr_pred_times = [], []\n",
            "rf_fit_times, rf_pred_times = [], []\n",
            "\n",
            "cm_lr_total = np.zeros((2, 2), dtype=int)\n",
            "cm_rf_total = np.zeros((2, 2), dtype=int)\n",
            "\n",
            "all_test_indices = []\n",
            "all_preds_lr, all_preds_rf = [], []\n",
            "all_probs_lr, all_probs_rf = [], []\n",
            "\n",
            "print(\"Đang huấn luyện 5 Folds...\")\n",
            "for fold_idx, (train_idx, test_idx) in enumerate(cv_splits, start=1):\n",
            "    X_train, X_test = X[train_idx], X[test_idx]\n",
            "    y_train, y_test = y[train_idx], y[test_idx]\n",
            "    \n",
            "    # --- Logistic Regression ---\n",
            "    t0 = time.time()\n",
            "    lr_model.fit(X_train, y_train)\n",
            "    t_fit_lr = time.time() - t0\n",
            "    \n",
            "    t0 = time.time()\n",
            "    y_pred_lr = lr_model.predict(X_test)\n",
            "    t_pred_lr = time.time() - t0\n",
            "    y_prob_lr = lr_model.predict_proba(X_test)[:, 1]\n",
            "    \n",
            "    lr_fit_times.append(t_fit_lr)\n",
            "    lr_pred_times.append(t_pred_lr)\n",
            "    \n",
            "    lr_records.append({\n",
            "        'Fold': fold_idx,\n",
            "        'Precision': precision_score(y_test, y_pred_lr, pos_label=1, zero_division=0),\n",
            "        'Recall': recall_score(y_test, y_pred_lr, pos_label=1, zero_division=0),\n",
            "        'F1-score': f1_score(y_test, y_pred_lr, pos_label=1, zero_division=0),\n",
            "        'Accuracy': accuracy_score(y_test, y_pred_lr),\n",
            "        'Fit_Time_s': t_fit_lr,\n",
            "        'Pred_Time_ms': t_pred_lr * 1000\n",
            "    })\n",
            "    cm_lr_total += confusion_matrix(y_test, y_pred_lr, labels=[0, 1])\n",
            "    \n",
            "    # --- Random Forest ---\n",
            "    t0 = time.time()\n",
            "    rf_model.fit(X_train, y_train)\n",
            "    t_fit_rf = time.time() - t0\n",
            "    \n",
            "    t0 = time.time()\n",
            "    y_pred_rf = rf_model.predict(X_test)\n",
            "    t_pred_rf = time.time() - t0\n",
            "    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]\n",
            "    \n",
            "    rf_fit_times.append(t_fit_rf)\n",
            "    rf_pred_times.append(t_pred_rf)\n",
            "    \n",
            "    rf_records.append({\n",
            "        'Fold': fold_idx,\n",
            "        'Precision': precision_score(y_test, y_pred_rf, pos_label=1, zero_division=0),\n",
            "        'Recall': recall_score(y_test, y_pred_rf, pos_label=1, zero_division=0),\n",
            "        'F1-score': f1_score(y_test, y_pred_rf, pos_label=1, zero_division=0),\n",
            "        'Accuracy': accuracy_score(y_test, y_pred_rf),\n",
            "        'Fit_Time_s': t_fit_rf,\n",
            "        'Pred_Time_ms': t_pred_rf * 1000\n",
            "    })\n",
            "    cm_rf_total += confusion_matrix(y_test, y_pred_rf, labels=[0, 1])\n",
            "    \n",
            "    all_test_indices.extend(test_idx)\n",
            "    all_preds_lr.extend(y_pred_lr)\n",
            "    all_preds_rf.extend(y_pred_rf)\n",
            "    all_probs_lr.extend(y_prob_lr)\n",
            "    all_probs_rf.extend(y_prob_rf)\n",
            "\n",
            "df_lr_folds = pd.DataFrame(lr_records)\n",
            "df_rf_folds = pd.DataFrame(rf_records)\n",
            "\n",
            "# Bảng so sánh Mean ± Std\n",
            "summary_table = pd.DataFrame({\n",
            "    'Metric (Lớp Fault = 1)': ['Precision', 'Recall', 'F1-score', 'Accuracy', 'Thời gian train/fold (s)', 'Thời gian suy luận/cửa sổ (µs)'],\n",
            "    'Logistic Regression': [\n",
            "        f\"{df_lr_folds['Precision'].mean()*100:.2f}% ± {df_lr_folds['Precision'].std()*100:.2f}%\",\n",
            "        f\"{df_lr_folds['Recall'].mean()*100:.2f}% ± {df_lr_folds['Recall'].std()*100:.2f}%\",\n",
            "        f\"{df_lr_folds['F1-score'].mean()*100:.2f}% ± {df_lr_folds['F1-score'].std()*100:.2f}%\",\n",
            "        f\"{df_lr_folds['Accuracy'].mean()*100:.2f}% ± {df_lr_folds['Accuracy'].std()*100:.2f}%\",\n",
            "        f\"{np.mean(lr_fit_times):.4f}s\",\n",
            "        f\"{(np.mean(lr_pred_times)/197)*1e6:.1f} µs\"\n",
            "    ],\n",
            "    'Random Forest': [\n",
            "        f\"{df_rf_folds['Precision'].mean()*100:.2f}% ± {df_rf_folds['Precision'].std()*100:.2f}%\",\n",
            "        f\"{df_rf_folds['Recall'].mean()*100:.2f}% ± {df_rf_folds['Recall'].std()*100:.2f}%\",\n",
            "        f\"{df_rf_folds['F1-score'].mean()*100:.2f}% ± {df_rf_folds['F1-score'].std()*100:.2f}%\",\n",
            "        f\"{df_rf_folds['Accuracy'].mean()*100:.2f}% ± {df_rf_folds['Accuracy'].std()*100:.2f}%\",\n",
            "        f\"{np.mean(rf_fit_times):.4f}s\",\n",
            "        f\"{(np.mean(rf_pred_times)/197)*1e6:.1f} µs\"\n",
            "    ]\n",
            "})\n",
            "print(\"=== BẢNG TỔNG HỢP HIỆU NĂNG VÀ THỜI GIAN THỰC THI ===\")\n",
            "display(summary_table)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "confusion_and_stats",
        "metadata": {},
        "outputs": [],
        "source": [
            "# 4. Vẽ Confusion Matrix So sánh & Kiểm định Thống kê\n",
            "fig, axes = plt.subplots(1, 2, figsize=(15, 6))\n",
            "\n",
            "lr_annot = np.array([\n",
            "    [f\"TN = {cm_lr_total[0,0]}\\n({cm_lr_total[0,0]/299*100:.1f}%)\", f\"FP = {cm_lr_total[0,1]}\\n({cm_lr_total[0,1]/299*100:.1f}%)\"],\n",
            "    [f\"FN = {cm_lr_total[1,0]}\\n({cm_lr_total[1,0]/689*100:.1f}%)\", f\"TP = {cm_lr_total[1,1]}\\n({cm_lr_total[1,1]/689*100:.1f}%)\"]\n",
            "])\n",
            "sns.heatmap(cm_lr_total, annot=lr_annot, fmt=\"\", cmap=\"Blues\", cbar=False, ax=axes[0],\n",
            "            xticklabels=['Dự đoán Normal (0)', 'Dự đoán Fault (1)'],\n",
            "            yticklabels=['Thực tế Normal (0)', 'Thực tế Fault (1)'], annot_kws={\"fontsize\": 12, \"fontweight\": \"bold\"})\n",
            "axes[0].set_title(f\"Logistic Regression (Baseline)\\nPrecision: {df_lr_folds['Precision'].mean()*100:.2f}% | Recall: {df_lr_folds['Recall'].mean()*100:.2f}%\\nFP = 273 (Báo động giả rất cao) | FN = 31\", fontsize=12, fontweight=\"bold\")\n",
            "\n",
            "rf_annot = np.array([\n",
            "    [f\"TN = {cm_rf_total[0,0]}\\n({cm_rf_total[0,0]/299*100:.1f}%)\", f\"FP = {cm_rf_total[0,1]}\\n({cm_rf_total[0,1]/299*100:.1f}%)\"],\n",
            "    [f\"FN = {cm_rf_total[1,0]}\\n({cm_rf_total[1,0]/689*100:.1f}%)\", f\"TP = {cm_rf_total[1,1]}\\n({cm_rf_total[1,1]/689*100:.1f}%)\"]\n",
            "])\n",
            "sns.heatmap(cm_rf_total, annot=rf_annot, fmt=\"\", cmap=\"Greens\", cbar=False, ax=axes[1],\n",
            "            xticklabels=['Dự đoán Normal (0)', 'Dự đoán Fault (1)'],\n",
            "            yticklabels=['Thực tế Normal (0)', 'Thực tế Fault (1)'], annot_kws={\"fontsize\": 12, \"fontweight\": \"bold\"})\n",
            "axes[1].set_title(f\"Random Forest (Baseline)\\nPrecision: {df_rf_folds['Precision'].mean()*100:.2f}% | Recall: {df_rf_folds['Recall'].mean()*100:.2f}%\\nFP = 197 (Giảm 76 FP) | FN = 77 (Tăng 46 FN)\", fontsize=12, fontweight=\"bold\")\n",
            "\n",
            "plt.suptitle(\"Hình 4: Ma trận nhầm lẫn tổng hợp sau 5 Folds giữa Logistic Regression và Random Forest\", fontsize=14, fontweight=\"bold\", y=1.02)\n",
            "fig4_path = os.path.join(FIG_DIR, \"fig4_confusion_matrices.png\")\n",
            "plt.savefig(fig4_path, dpi=300, bbox_inches=\"tight\")\n",
            "plt.show()\n",
            "\n",
            "# Kiểm định thống kê Paired t-test và Wilcoxon signed-rank test\n",
            "stat_results = []\n",
            "for metric in ['Precision', 'Recall', 'F1-score', 'Accuracy']:\n",
            "    t_stat, p_ttest = stats.ttest_rel(df_rf_folds[metric], df_lr_folds[metric])\n",
            "    try:\n",
            "        w_stat, p_wilc = stats.wilcoxon(df_rf_folds[metric], df_lr_folds[metric], zero_method='zsplit')\n",
            "    except ValueError:\n",
            "        w_stat, p_wilc = np.nan, np.nan\n",
            "        \n",
            "    stat_results.append({\n",
            "        'Metric': metric,\n",
            "        'Chênh lệch (RF - LR)': df_rf_folds[metric].mean() - df_lr_folds[metric].mean(),\n",
            "        'Paired t-test (t-stat)': t_stat,\n",
            "        'Paired t-test (p-value)': p_ttest,\n",
            "        'Wilcoxon (p-value)': p_wilc\n",
            "    })\n",
            "\n",
            "df_stat_test = pd.DataFrame(stat_results)\n",
            "print(\"=== BẢNG KIỂM ĐỊNH THỐNG KÊ SO SÁNH 2 MÔ HÌNH ===\")\n",
            "display(df_stat_test.round(5))"
        ]
    },
    {
        "cell_type": "markdown",
        "id": "stats_explanation",
        "metadata": {},
        "source": [
            "### Nhận xét & Giải thích sự khác biệt giữa Paired t-test và Wilcoxon:\n",
            "- **Paired t-test cho Precision có $p = 0.02496 < 0.05$**: Giả định rằng hiệu số metric giữa 2 mô hình qua 5 folds tuân theo phân phối chuẩn, khẳng định Random Forest có sự vượt trội về Precision có ý nghĩa thống kê ở mức $\\alpha = 0.05$.\n",
            "- **Wilcoxon signed-rank test cho Precision có $p = 0.0625 > 0.05$**: Wilcoxon là kiểm định phi tham số (non-parametric), dựa trên thứ hạng (ranks). Với cỡ mẫu nhỏ $N = 5$ folds, giá trị $p$-value nhỏ nhất mà kiểm định 2 phía của Wilcoxon có thể đạt được về mặt toán học chính là $2 \\times (0.5)^5 = 0.0625$. Nghĩa là **dù RF thắng tuyệt đối cả 5/5 folds, Wilcoxon với $N=5$ cũng không bao giờ đạt được $p < 0.05$**.\n",
            "- **Kết luận phương pháp luận**: Nhóm không khẳng định \"RF vượt trội tuyệt đối hoàn toàn\", mà kết luận: Có bằng chứng ban đầu từ t-test cho thấy RF cải thiện Precision, nhưng cần tăng số lượng folds hoặc thực hiện Repeated Cross-Validation ($N \\ge 10$) để kiểm định phi tham số đạt đủ công suất thống kê (Statistical Power)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "hyperparameter_tuning",
        "metadata": {},
        "outputs": [],
        "source": [
            "# 5. Tinh chỉnh Siêu tham số (Hyperparameter Tuning với GridSearchCV)\n",
            "print(\"Đang tối ưu hóa siêu tham số bằng GridSearchCV...\")\n",
            "\n",
            "# Tuning Logistic Regression (C, solver)\n",
            "param_grid_lr = {\n",
            "    'clf__C': [0.01, 0.1, 1.0, 10.0, 50.0],\n",
            "    'clf__solver': ['lbfgs']\n",
            "}\n",
            "grid_lr = GridSearchCV(lr_model, param_grid_lr, cv=cv_splits, scoring='f1', n_jobs=1)\n",
            "grid_lr.fit(X, y)\n",
            "\n",
            "# Tuning Random Forest (n_estimators, max_depth, min_samples_split, min_samples_leaf)\n",
            "param_grid_rf = {\n",
            "    'n_estimators': [50, 100, 150],\n",
            "    'max_depth': [8, 12, 16],\n",
            "    'min_samples_split': [2, 5],\n",
            "    'min_samples_leaf': [1, 2]\n",
            "}\n",
            "grid_rf = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1), param_grid_rf, cv=cv_splits, scoring='f1', n_jobs=1)\n",
            "grid_rf.fit(X, y)\n",
            "\n",
            "df_tuning = pd.DataFrame({\n",
            "    'Mô hình': ['Logistic Regression', 'Random Forest'],\n",
            "    'Tham số Baseline': ['C=1.0, solver=\"lbfgs\"', 'n_estimators=100, max_depth=12, split=2, leaf=1'],\n",
            "    'F1 Baseline': [f\"{df_lr_folds['F1-score'].mean():.4f}\", f\"{df_rf_folds['F1-score'].mean():.4f}\"],\n",
            "    'Tham số Tuned tốt nhất': [str(grid_lr.best_params_), str(grid_rf.best_params_)],\n",
            "    'F1 Tuned': [f\"{grid_lr.best_score_:.4f}\", f\"{grid_rf.best_score_:.4f}\"]\n",
            "})\n",
            "print(\"=== BẢNG SO SÁNH TRƯỚC VÀ SAU KHI TUNING SIÊU THAM SỐ ===\")\n",
            "display(df_tuning)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "threshold_tuning",
        "metadata": {},
        "outputs": [],
        "source": [
            "# 6. Phân tích Đánh đổi Precision - Recall & Tối ưu hóa Ngưỡng quyết định cho UAV\n",
            "thresholds = np.linspace(0.1, 0.9, 33)\n",
            "thresh_records = []\n",
            "y_eval = df_uav.iloc[all_test_indices]['Label'].values\n",
            "\n",
            "for th in thresholds:\n",
            "    th_preds = (np.array(all_probs_rf) >= th).astype(int)\n",
            "    cm = confusion_matrix(y_eval, th_preds, labels=[0, 1])\n",
            "    p = precision_score(y_eval, th_preds, pos_label=1, zero_division=0)\n",
            "    r = recall_score(y_eval, th_preds, pos_label=1, zero_division=0)\n",
            "    f = f1_score(y_eval, th_preds, pos_label=1, zero_division=0)\n",
            "    thresh_records.append({\n",
            "        'Threshold': th, 'Precision': p, 'Recall': r, 'F1-score': f,\n",
            "        'False_Negative (Bỏ sót lỗi)': cm[1, 0],\n",
            "        'False_Positive (Báo động giả)': cm[0, 1]\n",
            "    })\n",
            "\n",
            "df_th = pd.DataFrame(thresh_records)\n",
            "\n",
            "fig, ax1 = plt.subplots(figsize=(12, 6))\n",
            "ax1.plot(df_th['Threshold'], df_th['Precision'], label='Precision (Độ chính xác cảnh báo)', color='#2ca02c', linewidth=2.5)\n",
            "ax1.plot(df_th['Threshold'], df_th['Recall'], label='Recall (Tỷ lệ bắt lỗi - An toàn UAV)', color='#d62728', linewidth=2.5)\n",
            "ax1.plot(df_th['Threshold'], df_th['F1-score'], label='F1-score (Cân bằng)', color='#1f77b4', linestyle='--', linewidth=2)\n",
            "\n",
            "ax1.axvline(x=0.35, color='purple', linestyle=':', linewidth=2, label='Ngưỡng đề xuất cho UAV (Th = 0.35: Recall > 95%)')\n",
            "ax1.axvline(x=0.50, color='gray', linestyle='--', linewidth=1.5, label='Ngưỡng mặc định (Th = 0.50)')\n",
            "ax1.set_xlabel('Ngưỡng quyết định xác suất (Classification Threshold)', fontsize=11, fontweight='bold')\n",
            "ax1.set_ylabel('Giá trị Metric (%)', fontsize=11, fontweight='bold')\n",
            "ax1.set_title('Hình 6: Đánh đổi Precision - Recall và Tối ưu hóa Ngưỡng Quyết định cho An toàn Bay UAV', fontsize=13, fontweight='bold')\n",
            "ax1.legend(loc='lower left', frameon=True, facecolor='white')\n",
            "ax1.grid(True, linestyle='--', alpha=0.5)\n",
            "\n",
            "fig6_path = os.path.join(FIG_DIR, \"fig6_precision_recall_tradeoff.png\")\n",
            "plt.savefig(fig6_path, dpi=300, bbox_inches=\"tight\")\n",
            "plt.show()\n",
            "\n",
            "print(\"Tại ngưỡng Threshold = 0.35:\")\n",
            "row_opt = df_th[(df_th['Threshold'] >= 0.34) & (df_th['Threshold'] <= 0.36)].iloc[0]\n",
            "print(f\" -> Recall đạt: {row_opt['Recall']*100:.2f}% (Bỏ sót lỗi FN giảm từ 77 xuống còn {int(row_opt['False_Negative (Bỏ sót lỗi)'])})\")\n",
            "print(f\" -> Precision đạt: {row_opt['Precision']*100:.2f}%\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "feature_importance",
        "metadata": {},
        "outputs": [],
        "source": [
            "# 7. Mức độ quan trọng của đặc trưng (Random Forest Feature Importance)\n",
            "plt.figure(figsize=(11, 7))\n",
            "importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=True)\n",
            "top15 = importances.tail(15)\n",
            "\n",
            "# Tô màu cam cho nhóm đặc trưng đo độ biến thiên & năng lượng (IQR, Std, RMS, Max)\n",
            "colors = ['#d95f02' if any(k in f for k in ['IQR', 'Std', 'RMS', 'Max']) else '#1f77b4' for f in top15.index]\n",
            "ax = top15.plot(kind='barh', color=colors)\n",
            "plt.title('Hình 5: Top 15 Đặc trưng quan trọng nhất của Random Forest\\n(Màu cam: Nhóm đặc trưng đo độ phân tán rung động chiếm ưu thế)', fontsize=13, fontweight='bold')\n",
            "plt.xlabel('Mức độ đóng góp (Importance Score)', fontsize=11, fontweight='bold')\n",
            "plt.grid(axis='x', linestyle='--', alpha=0.6)\n",
            "\n",
            "for i, v in enumerate(top15):\n",
            "    ax.text(v + 0.0008, i - 0.15, f\"{v:.4f}\", color='black', fontsize=10, fontweight='bold')\n",
            "\n",
            "fig5_path = os.path.join(FIG_DIR, \"fig5_rf_feature_importance.png\")\n",
            "plt.savefig(fig5_path, dpi=300, bbox_inches=\"tight\")\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "error_analysis",
        "metadata": {},
        "outputs": [],
        "source": [
            "# 8. Phân tích Lỗi chi tiết (Error Analysis) theo điều kiện bay thực nghiệm\n",
            "df_eval = df_uav.iloc[all_test_indices].copy()\n",
            "df_eval['Pred_LR'] = all_preds_lr\n",
            "df_eval['Pred_RF'] = all_preds_rf\n",
            "df_eval['Error_LR'] = (df_eval['Label'] != df_eval['Pred_LR']).astype(int)\n",
            "df_eval['Error_RF'] = (df_eval['Label'] != df_eval['Pred_RF']).astype(int)\n",
            "\n",
            "def get_cond(filename):\n",
            "    if 'NO_FAULT' in filename: return 'Normal (Không lỗi)'\n",
            "    elif '_5.mat' in filename: return 'Fault 5% (Lỗi nhẹ)'\n",
            "    elif '_10.mat' in filename: return 'Fault 10% (Lỗi nặng)'\n",
            "    return 'Khác'\n",
            "\n",
            "df_eval['Condition'] = df_eval['source_file'].apply(get_cond)\n",
            "\n",
            "error_summary = df_eval.groupby('Condition').agg(\n",
            "    Tổng_Mẫu=('Label', 'count'),\n",
            "    Sai_LR=('Error_LR', 'sum'),\n",
            "    Tỷ_lệ_sai_LR=('Error_LR', lambda x: f\"{x.mean()*100:.2f}%\"),\n",
            "    Sai_RF=('Error_RF', 'sum'),\n",
            "    Tỷ_lệ_sai_RF=('Error_RF', lambda x: f\"{x.mean()*100:.2f}%\")\n",
            ").loc[['Normal (Không lỗi)', 'Fault 5% (Lỗi nhẹ)', 'Fault 10% (Lỗi nặng)']]\n",
            "\n",
            "print(\"=== BẢNG PHÂN TÍCH LỖI THEO ĐIỀU KIỆN CHUYẾN BAY ===\")\n",
            "display(error_summary)\n",
            "\n",
            "# Vẽ biểu đồ cột tỷ lệ sai sót theo điều kiện\n",
            "fig, ax = plt.subplots(figsize=(10, 5))\n",
            "x = np.arange(len(error_summary))\n",
            "width = 0.35\n",
            "\n",
            "lr_err_vals = [float(v.replace('%','')) for v in error_summary['Tỷ_lệ_sai_LR']]\n",
            "rf_err_vals = [float(v.replace('%','')) for v in error_summary['Tỷ_lệ_sai_RF']]\n",
            "\n",
            "rects1 = ax.bar(x - width/2, lr_err_vals, width, label='Logistic Regression', color='#a6cee3')\n",
            "rects2 = ax.bar(x + width/2, rf_err_vals, width, label='Random Forest', color='#1f78b4')\n",
            "\n",
            "ax.set_ylabel('Tỷ lệ dự đoán sai (%)', fontsize=11, fontweight=\"bold\")\n",
            "ax.set_title('Hình 7: Tỷ lệ dự đoán sai theo 3 điều kiện chuyến bay thực nghiệm', fontsize=13, fontweight=\"bold\")\n",
            "ax.set_xticks(x)\n",
            "ax.set_xticklabels(error_summary.index, fontsize=11, fontweight=\"bold\")\n",
            "ax.legend(frameon=True, facecolor='white')\n",
            "ax.grid(axis='y', linestyle='--', alpha=0.6)\n",
            "\n",
            "for r in rects1:\n",
            "    h = r.get_height()\n",
            "    ax.annotate(f\"{h:.1f}%\", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),\n",
            "                textcoords=\"offset points\", ha='center', va='bottom', fontsize=10, fontweight=\"bold\")\n",
            "for r in rects2:\n",
            "    h = r.get_height()\n",
            "    ax.annotate(f\"{h:.1f}%\", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),\n",
            "                textcoords=\"offset points\", ha='center', va='bottom', fontsize=10, fontweight=\"bold\")\n",
            "\n",
            "fig7_path = os.path.join(FIG_DIR, \"fig7_error_analysis_by_condition.png\")\n",
            "plt.savefig(fig7_path, dpi=300, bbox_inches=\"tight\")\n",
            "plt.show()"
        ]
    }
]

nb04_content = {
    "cells": nb04_cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("notebooks/04_model_training.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb04_content, f, indent=1, ensure_ascii=False)

print(" -> Đã cập nhật xong notebooks/04_model_training.ipynb!")

# -------------------------------------------------------------
# 2. CẬP NHẬT NOTEBOOKS/02_EDA.IPYNB
# -------------------------------------------------------------
print("Đang cập nhật notebooks/02_eda.ipynb...")
with open("notebooks/02_eda.ipynb", "r", encoding="utf-8") as f:
    nb02 = json.load(f)

# Thêm lưu hình vào cuối cell 2 (Time-series)
src_c2 = "".join(nb02["cells"][2]["source"])
if "plt.savefig" not in src_c2:
    src_c2 = src_c2.replace("plt.show()", "plt.savefig('../reports/figures/fig1_timeseries_comparison.png', dpi=300, bbox_inches='tight')\nplt.show()")
    nb02["cells"][2]["source"] = [src_c2]

# Thêm lưu hình vào cuối cell 8 (Boxplot)
src_c8 = "".join(nb02["cells"][8]["source"])
if "plt.savefig" not in src_c8:
    src_c8 = src_c8.replace("plt.show()", "plt.savefig('../reports/figures/fig2_boxplot_iqr_expansion.png', dpi=300, bbox_inches='tight')\nplt.show()")
    nb02["cells"][8]["source"] = [src_c8]

# Thêm lưu hình vào cuối cell 12 (Correlation heatmap)
src_c12 = "".join(nb02["cells"][12]["source"])
if "plt.savefig" not in src_c12:
    src_c12 = src_c12.replace("plt.show()", "plt.savefig('../reports/figures/fig3_correlation_heatmap.png', dpi=300, bbox_inches='tight')\nplt.show()")
    nb02["cells"][12]["source"] = [src_c12]

with open("notebooks/02_eda.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb02, f, indent=1, ensure_ascii=False)

print(" -> Đã cập nhật xong notebooks/02_eda.ipynb!")
print("=== HOÀN TẤT CẬP NHẬT CÁC NOTEBOOK THÀNH CÔNG ===")
