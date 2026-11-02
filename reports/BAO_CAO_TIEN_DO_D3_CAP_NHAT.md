# BÁO CÁO TIẾN ĐỘ D3 (BẢN HOÀN THIỆN THEO GÓP Ý GIẢNG VIÊN)
**Môn học:** DTA301 – Phân tích dữ liệu & RBL  
**Đề tài:** Phát hiện lỗi cánh quạt UAV từ dữ liệu cảm biến IMU bằng Học máy  
**Môi trường:** Python 3.14 | scikit-learn 1.8 | Pandas 2.2.3 | StratifiedGroupKFold (5 Folds)  

---

## 1. KẾT QUẢ PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EDA) VÀ ĐÍNH CHÍNH LẬP LUẬN

### 1.1. EDA 1 – Phân tích Tín hiệu Chuỗi thời gian (Time-series Visualization)
![Hình 1](figures/fig1_timeseries_comparison.png)
*Hình 1: So sánh dao động chuỗi thời gian 6 kênh IMU giữa Trạng thái Normal và Fault (Cánh quạt M1 mẻ 10%)*

- **Nhận xét & Ý nghĩa vật lý:** Trạng thái bình thường dao động biên độ hẹp quanh điểm cân bằng. Khi cánh quạt mẻ, lực khí động mất cân bằng sinh rung chấn tần số cao rõ rệt nhất trên hai trục gia tốc ngang (AccX, AccY) và trục quay Roll (GyrX). Trục AccZ duy trì thành phần trọng trường ~ -9.8 m/s² nhưng biên độ dao động nhấp nhô tăng mạnh.

### 1.2. EDA 2 – Phân phối & Sự mở rộng Khoảng tứ phân vị (Boxplot & IQR Expansion)
![Hình 2](figures/fig2_boxplot_iqr_expansion.png)
*Hình 2: Phân phối và sự mở rộng Khoảng tứ phân vị (IQR) giữa Normal và Fault trên 6 kênh cảm biến*

- **Nhận xét định lượng:** Cả 6 kênh cảm biến có giá trị trung vị gần như trùng khít nhau (GyrX, GyrY, GyrZ ~ 0; AccZ ~ -9.74 m/s²). Lỗi cơ khí tạo ra dao động đối xứng qua điểm 0 (Zero-crossing). Ngược lại, IQR mở rộng mạnh ở trạng thái Fault (AccX +34.5%, GyrX +30.8%, GyrY +30.0%, AccZ +21.8%, GyrZ +19.8%).
- **Đính chính lập luận:** Trong báo cáo trước, nhận định *"trung vị gần nhau nên siêu phẳng tuyến tính của Logistic Regression không phân loại hiệu quả"* là chưa chuẩn xác vì Logistic Regression phân loại trên không gian đa chiều. Nguyên nhân thực sự là do dao động đối xứng qua điểm 0 (phương sai đổi nhưng kỳ vọng không đổi). Khi chuyển sang trích xuất đặc trưng cửa sổ, sự phân tách phi tuyến vẫn tạo lợi thế lớn cho cơ chế chia nhánh của Random Forest.

### 1.3. EDA 3 – Ma trận Tương quan Động lực học (Correlation Heatmap)
![Hình 3](figures/fig3_correlation_heatmap.png)
*Hình 3: Ma trận tương quan Pearson giữa 6 kênh IMU ở trạng thái Normal và Fault*

- **Nhận xét định lượng:** Ở nhóm Normal, các trục gia tốc có tương quan động học mạnh: AccY-AccZ (r = -0.72), AccX-AccZ (r = +0.69), AccX-AccY (r = -0.55). Ở nhóm Fault, tương quan giữa AccX với các trục khác sụt giảm mạnh về gần 0 (AccX-AccY từ -0.55 lên +0.04; AccX-AccZ từ +0.69 sụt xuống +0.15). Lực đẩy bất đối xứng đã phá vỡ cấu trúc tương quan tuyến tính giữa các trục.
- **Đính chính lập luận:** Tương quan tuyến tính thấp (Pearson r gần 0) **KHÔNG đồng nghĩa với độc lập thống kê**, vì giữa các trục vẫn tồn tại các liên kết động lực học phi tuyến phức tạp của con quay hồi chuyển và phương trình chuyển động Euler. Diễn đạt chính xác: Các kênh cảm biến cung cấp thông tin đa chiều bổ trợ lẫn nhau và không bị đa cộng tuyến tuyến tính nghiêm trọng (|r| < 0.80).

---

## 2. QUÁ TRÌNH HUẤN LUYỆN, TUNING SIÊU THAM SỐ VÀ KHẢ NĂNG TÁI LẬP

### 2.1. Quy trình Tiền xử lý & Chống Data Leakage
- 18 file tương ứng 18 chuyến bay riêng biệt.
- Sử dụng `StratifiedGroupKFold` gom nhóm theo `source_file`, đảm bảo 100% cửa sổ của một chuyến bay chỉ nằm ở Train hoặc Test (Trùng lặp Leakage = 0).
- `StandardScaler` được đóng gói trong `Pipeline`, chỉ fit trên Train fold.

### 2.2. Siêu tham số & Kết quả Tuning (GridSearchCV)
| Mô hình | Cấu hình Baseline | F1 Baseline | Cấu hình Tuned tốt nhất | F1 Tuned |
|---|---|---|---|---|
| **Logistic Regression** | C = 1.0, solver = 'lbfgs', max_iter = 1000 | 0.8113 | C = 0.1, solver = 'lbfgs', max_iter = 1000 | 0.8181 |
| **Random Forest** | n_estimators=100, max_depth=12, split=2, leaf=1 | 0.8102 | n_estimators=150, max_depth=16, split=5, leaf=2 | 0.8185 |

### 2.3. Thời gian Huấn luyện và Suy luận
| Mô hình | Thời gian Train / Fold (s) | Thời gian Test / Fold (ms) | Suy luận / Cửa sổ 2s (µs) |
|---|---|---|---|
| **Logistic Regression** | 0.0218 ± 0.0057 s | 0.61 ± 0.32 ms | 3.1 µs |
| **Random Forest** | 0.1835 ± 0.0078 s | 28.55 ± 1.24 ms | 144.9 µs |

---

## 3. ĐÁNH GIÁ MÔ HÌNH, TRADE-OFF VÀ MẤT CÂN BẰNG DỮ LIỆU

### 3.1. Ma trận Nhầm lẫn (Confusion Matrix)
![Hình 4](figures/fig4_confusion_matrices.png)
*Hình 4: Ma trận nhầm lẫn tổng hợp sau 5 Folds giữa Logistic Regression và Random Forest*

### 3.2. Đánh đổi Precision vs Recall & Tối ưu hóa Ngưỡng cho An toàn UAV
![Hình 6](figures/fig6_precision_recall_tradeoff.png)
*Hình 6: Phân tích đánh đổi Precision - Recall và Tối ưu hóa Ngưỡng quyết định cho An toàn UAV*

- **False Positive (Báo động giả):** Khiến UAV hạ cánh khẩn cấp oan, gián đoạn nhiệm vụ.
- **False Negative (Bỏ sót lỗi):** Nguy cơ rơi máy bay thảm khốc (catastrophic crash).
- **Hạn chế của Logistic Regression:** Dù Recall 95.57%, LR đoán sai 91.3% tập Normal (chỉ đúng 26/299 mẫu), bị thiên lệch lớp đa số (Fault chiếm 69.7%).
- **Threshold Tuning cho Random Forest:** Khi hạ ngưỡng quyết định từ 0.50 xuống **Th = 0.35**, Random Forest nâng Recall lên **95.21%** (FN giảm từ 77 xuống còn 33), trong khi Precision vẫn đạt **72.8%**, vượt trội toàn diện so với LR.

### 3.3. Hạn chế của Metric Accuracy khi Dữ liệu Mất cân bằng
- Dữ liệu gồm 689 Fault (69.7%) và 299 Normal (30.3%).
- Một bộ phân loại tầm thường luôn đoán mọi mẫu là Fault vẫn đạt Accuracy = 69.7%! Do đó không được dùng Accuracy làm thước đo chính.

---

## 4. KIỂM ĐỊNH THỐNG KÊ (PAIRED T-TEST VS WILCOXON)

| Metric (Fault = 1) | Logistic Regression | Random Forest | Chênh lệch (RF - LR) | Paired t-test (p-value) | Wilcoxon (p-value) |
|---|---|---|---|---|---|
| **Precision** | 70.74% ± 6.55% | 75.66% ± 8.85% | +4.92% | **p = 0.02496** | **p = 0.0625** |
| **Recall** | 95.57% ± 1.72% | 87.89% ± 9.75% | -7.68% | p = 0.16235 | p = 0.3125 |
| **F1-score** | 81.13% ± 4.04% | 81.02% ± 7.53% | -0.11% | p = 0.95635 | p = 1.0000 |
| **Accuracy** | 69.25% ± 5.49% | 72.09% ± 9.29% | +2.84% | p = 0.25200 | p = 0.3750 |

- **Giải thích khoa học sự mâu thuẫn giữa 2 kiểm định:**
  - Paired t-test (kiểm định tham số) cho $p = 0.025 < 0.05$.
  - Wilcoxon (kiểm định phi tham số) dựa trên thứ hạng. Với $N = 5$ folds, $p$-value nhỏ nhất có thể đạt được về mặt toán học là $2 \times (0.5)^5 = 0.0625$. Nghĩa là dù RF thắng tuyệt đối cả 5 folds thì Wilcoxon cũng không thể nào $< 0.05$.
  - Do đó, nhóm kết luận thận trọng: Có bằng chứng ban đầu từ t-test, và sẽ mở rộng lên 10-Fold CV trong D4 để Wilcoxon đạt đủ lực thống kê.

---

## 5. PHÂN TÍCH LỖI (ERROR ANALYSIS) VÀ ĐẶC TRƯNG QUAN TRỌNG

### 5.1. Mức độ quan trọng của đặc trưng
![Hình 5](figures/fig5_rf_feature_importance.png)
*Hình 5: Top 15 Đặc trưng đóng góp lớn nhất trong Random Forest*
- Các đặc trưng đo độ biến thiên (IQR, Std, RMS, Max) của hai trục gia tốc ngang AccX và AccY chiếm ưu thế áp đảo.

### 5.2. Phân tích Lỗi theo Điều kiện chuyến bay
![Hình 7](figures/fig7_error_analysis_by_condition.png)
*Hình 7: Tỷ lệ dự đoán sai theo 3 điều kiện chuyến bay thực nghiệm*

| Điều kiện chuyến bay | Tổng mẫu | Sai LR | Tỷ lệ sai LR | Sai RF | Tỷ lệ sai RF |
|---|---|---|---|---|---|
| **Normal (Không lỗi)** | 299 | 273 | 91.30% | 197 | 65.89% |
| **Fault 5% (Lỗi nhẹ)** | 327 | 12 | 3.67% | 54 | 16.51% |
| **Fault 10% (Lỗi nặng)** | 362 | 19 | 5.25% | 23 | 6.35% |

- **Nguyên nhân sai sót:** Ở nhóm Fault 5%, độ rung chấn còn yếu gần với dao động tự nhiên khi bay gió giật nên RF nhầm 16.51% thành Normal. Khi lên Fault 10%, sai sót giảm mạnh còn 6.35%.

---

## 6. ĐỊNH HƯỚNG GIAI ĐOẠN D4
1. Tích hợp Threshold Tuning (Th = 0.35) vào hệ thống phát hiện lỗi thời gian thực.
2. Mở rộng lên 10-Fold CV / Repeated CV để tăng công suất thống kê.
3. Trích xuất thêm đặc trưng miền tần số (FFT, Spectral Energy).
4. Thử nghiệm XGBoost, LightGBM sau khi đã có baseline vững chắc.
