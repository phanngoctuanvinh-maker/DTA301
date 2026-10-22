# DTA301_UAV

Đề tài: Phát hiện lỗi cánh quạt UAV từ dữ liệu cảm biến IMU nhằm hỗ trợ công tác kiểm tra bảo trì.

## Cấu trúc
- `data/raw/normal/`: đặt 6 file `NO_FAULT1.mat` ... `NO_FAULT6.mat` (nhãn 0).
- `data/raw/fault/`: đặt 12 file `FAULT_M1_5.mat`, `FAULT_M1_10.mat`, ... `FAULT_M6_10.mat` (nhãn 1).
- `data/processed/`: dữ liệu đã làm sạch.
- `data/features/`: bảng đặc trưng theo cửa sổ thời gian.
- `notebooks/`: notebook phân tích và huấn luyện.
- `models/`: mô hình đã huấn luyện.
- `reports/`: biểu đồ, kết quả, báo cáo.

## Bắt đầu
1. Giải nén thư mục `DTA301_UAV` và mở thư mục này trong VS Code.
2. Tự chép 18 file `.mat` gốc vào đúng hai thư mục `data/raw/normal` và `data/raw/fault`. File ZIP không bao gồm dữ liệu UAV-FD.
3. Cài thư viện: `pip install -r requirements.txt`.
4. Mở `notebooks/01_data_understanding.ipynb`.

**Lưu ý:** Giữ nguyên dữ liệu gốc. Khi chia Train/Test, cần xác minh nhóm phiên ghi/chuyến bay độc lập để tránh rò rỉ dữ liệu; không chia ngẫu nhiên từng dòng IMU.
