# -*- coding: utf-8 -*-
"""
DTA301 - Tự động tạo Báo cáo Tiến độ D3 Cập nhật (Word .docx và Markdown .md)
Tích hợp đầy đủ các góp ý của giáo viên: 7 biểu đồ minh họa, bảng số liệu,
lập luận toán học - an toàn UAV chuẩn mực.
"""

import os
import sys
import json
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("Đang khởi tạo tạo Báo cáo Tiến độ D3 cập nhật...")

# Đọc kết quả thực nghiệm
with open("reports/experiment_summary.json", "r", encoding="utf-8") as f:
    summary = json.load(f)

df_stats = pd.read_csv("reports/statistical_tests_results.csv")
df_times = pd.read_csv("reports/model_execution_times.csv")
df_err_cond = pd.read_csv("reports/error_analysis_by_condition.csv")

# -------------------------------------------------------------
# 1. TẠO TÀI LIỆU WORD (.DOCX)
# -------------------------------------------------------------
doc = Document()

# Cấu hình lề 2cm (0.79 inch)
for section in doc.sections:
    section.top_margin = Inches(0.8)
    section.bottom_margin = Inches(0.8)
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)

# Định dạng style Normal
style_normal = doc.styles['Normal']
style_normal.font.name = 'Times New Roman'
style_normal.font.size = Pt(11.5)
style_normal.font.color.rgb = RGBColor(30, 30, 30)
style_normal.paragraph_format.line_spacing = 1.2
style_normal.paragraph_format.space_after = Pt(4)

def add_title(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.font.size = Pt(16)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0, 51, 102)
    p.paragraph_format.space_after = Pt(2)
    return p

def add_subtitle(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.font.size = Pt(13)
    run.font.bold = True
    run.font.color.rgb = RGBColor(70, 70, 70)
    p.paragraph_format.space_after = Pt(12)
    return p

def add_h1(text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.size = Pt(13.5)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0, 51, 102)
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    return p

def add_h2(text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.size = Pt(12)
    run.font.bold = True
    run.font.color.rgb = RGBColor(34, 102, 102)
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    return p

def add_p(text, bold_prefix=""):
    p = doc.add_paragraph()
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if bold_prefix:
        r_b = p.add_run(bold_prefix)
        r_b.font.bold = True
    p.add_run(text)
    p.paragraph_format.space_after = Pt(4)
    return p

def add_image_with_caption(img_path, caption_text):
    if os.path.exists(img_path):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(4)
        p_img.paragraph_format.space_after = Pt(2)
        doc.add_picture(img_path, width=Inches(5.9))
        
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p_cap.add_run(caption_text)
        run.font.size = Pt(10)
        run.font.italic = True
        run.font.bold = True
        run.font.color.rgb = RGBColor(80, 80, 80)
        p_cap.paragraph_format.space_after = Pt(8)

def format_table(table):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(table.rows):
        for cell in row.cells:
            # Padding
            tcPr = cell._tc.get_or_add_tcPr()
            tcMar = OxmlElement('w:tcMar')
            for m in ['top', 'bottom']:
                node = OxmlElement(f'w:{m}')
                node.set(qn('w:w'), '120')
                node.set(qn('w:type'), 'dxa')
                tcMar.append(node)
            for m in ['left', 'right']:
                node = OxmlElement(f'w:{m}')
                node.set(qn('w:w'), '160')
                node.set(qn('w:type'), 'dxa')
                tcMar.append(node)
            tcPr.append(tcMar)
            
            # Header color
            if i == 0:
                shading = parse_xml(r'<w:shd {} w:fill="003366"/>'.format(nsdecls('w')))
                cell._tc.get_or_add_tcPr().append(shading)
                for paragraph in cell.paragraphs:
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for run in paragraph.runs:
                        run.font.bold = True
                        run.font.color.rgb = RGBColor(255, 255, 255)
                        run.font.size = Pt(10.5)
            else:
                if i % 2 == 1:
                    shading = parse_xml(r'<w:shd {} w:fill="F4F6F9"/>'.format(nsdecls('w')))
                    cell._tc.get_or_add_tcPr().append(shading)
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.size = Pt(10)

# TIÊU ĐỀ BÁO CÁO
add_title("MÔN HỌC: DTA301 – PHÂN TÍCH DỮ LIỆU & RBL")
add_subtitle("BÁO CÁO TIẾN ĐỘ D3 (BẢN HOÀN THIỆN THEO GÓP Ý GIẢNG VIÊN)\nĐề tài: Phát hiện lỗi cánh quạt UAV từ dữ liệu cảm biến IMU bằng Học máy")

# METADATA & REPRODUCIBILITY
p_meta = doc.add_paragraph()
p_meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p_meta.add_run("Môi trường thực nghiệm: Python 3.14 | scikit-learn 1.8.dev0 | Pandas 2.2.3 | CPU Multi-core | OS: Windows 11\n"
                   "Tập dữ liệu: UAV-FD (Zenodo) – 18 chuyến bay độc lập (6 Normal, 12 Fault) | Cửa sổ: 2 giây | 36 Đặc trưng Thống kê")
r.font.size = Pt(9.5)
r.font.italic = True
p_meta.paragraph_format.space_after = Pt(10)

# MỤC 1: EDA VÀ TRỰC QUAN HÓA
add_h1("1. KẾT QUẢ PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EDA) VÀ ĐÍNH CHÍNH LẬP LUẬN")
add_p("Nhóm tiến hành phân tích khám phá dữ liệu chuyên sâu trên toàn bộ 18 file dữ liệu chuỗi thời gian IMU. Dưới đây là 3 phân tích trực quan cùng diễn giải vật lý và chỉnh lý các lập luận toán học theo đúng góp ý của giáo viên:")

add_h2("1.1. EDA 1 – Phân tích Tín hiệu Chuỗi thời gian (Time-series Visualization)")
add_p("Biểu đồ chuỗi thời gian so sánh dao động 6 trục giữa chuyến bay bình thường (NO_FAULT1) và chuyến bay lỗi cánh quạt M1 mẻ 10% (FAULT_M1_10) trong đoạn bay 4 giây ổn định.")
add_image_with_caption("reports/figures/fig1_timeseries_comparison.png", "Hình 1: So sánh dao động chuỗi thời gian 6 kênh IMU giữa Trạng thái Normal và Fault")
add_p("Ở trạng thái bình thường (Normal), các kênh gia tốc dao động biên độ hẹp quanh điểm cân bằng. Khi cánh quạt bị mẻ (Fault), lực khí động học mất cân bằng sinh ra rung chấn tần số cao thể hiện rõ nhất trên hai trục gia tốc ngang (AccX, AccY) và trục quay Roll (GyrX). Trục AccZ duy trì thành phần trọng trường ~ -9.8 m/s² nhưng biên độ dao động nhấp nhô tăng mạnh.", "Ý nghĩa vật lý đối với bài toán: ")

add_h2("1.2. EDA 2 – Phân tích Phân phối và Sự mở rộng Khoảng tứ phân vị (Boxplot & IQR Expansion)")
add_p("Phân tích phân phối thống kê đo lường Trung vị (Median), Khoảng tứ phân vị (IQR = Q75 - Q25) và Độ lệch chuẩn (Std) trên toàn bộ 18 chuyến bay.")
add_image_with_caption("reports/figures/fig2_boxplot_iqr_expansion.png", "Hình 2: Phân phối và sự mở rộng Khoảng tứ phân vị (IQR) giữa Normal và Fault trên 6 kênh cảm biến")
add_p("Cả 6 kênh cảm biến có giá trị trung vị giữa Normal và Fault gần như trùng khít nhau (GyrX, GyrY, GyrZ đều xấp xỉ 0; AccZ xấp xỉ -9.74 m/s²). Lỗi cơ khí tạo ra các dao động đối xứng hai chiều quanh điểm cân bằng (Zero-crossing). Điểm khác biệt mang tính quyết định nằm ở độ mở rộng của IQR: Fault làm IQR mở rộng mạnh trên cả 6 kênh (AccX +34.5%, GyrX +30.8%, GyrY +30.0%, AccZ +21.8%, GyrZ +19.8%).", "Nhận xét định lượng: ")
add_p("Trong báo cáo trước, nhóm nhận định 'trung vị gần nhau nên siêu phẳng tuyến tính của Logistic Regression không phân tách hiệu quả'. Nhận định này chưa chuẩn xác vì Logistic Regression phân loại trên không gian đa chiều, không xét đơn biến. Nguyên nhân thực sự khiến mô hình tuyến tính gặp khó khăn trên dữ liệu tức thời là do sự đối xứng của dao động qua điểm 0 (phương sai đổi nhưng kỳ vọng không đổi). Khi chuyển sang trích xuất đặc trưng cửa sổ (variance, IQR, energy), sự phân tách giữa các mức độ lỗi khác nhau vẫn mang tính phi tuyến, tạo lợi thế lớn cho cơ chế chia nhánh của Random Forest.", "Đính chính lập luận khoa học: ")

add_h2("1.3. EDA 3 – Ma trận Tương quan Động lực học (Correlation Heatmap)")
add_p("So sánh ma trận tương quan Pearson giữa 15 cặp cảm biến ở hai trạng thái để đánh giá sự thay đổi cấu trúc liên kết động học.")
add_image_with_caption("reports/figures/fig3_correlation_heatmap.png", "Hình 3: Ma trận tương quan Pearson giữa 6 kênh IMU ở trạng thái Normal và Fault")
add_p("Ở nhóm Normal, các trục gia tốc có tương quan động lực học rõ rệt: AccY - AccZ (r = -0.72), AccX - AccZ (r = +0.69), AccX - AccY (r = -0.55). Ở nhóm Fault, tương quan giữa AccX với các trục khác sụt giảm mạnh về gần 0 (AccX - AccY từ -0.55 lên +0.04; AccX - AccZ từ +0.69 sụt xuống +0.15). Lực đẩy bất đối xứng khi gãy cánh đã phá vỡ tương quan tuyến tính tự nhiên giữa các trục.", "Nhận xét định lượng: ")
add_p("Trong báo cáo trước, nhóm viết 'các cặp cảm biến có tương quan thấp chứng minh cả 6 kênh đều độc lập'. Theo góp ý của cô, tương quan tuyến tính thấp (Pearson r gần 0) KHÔNG đồng nghĩa với độc lập thống kê, vì giữa các trục vẫn tồn tại các liên kết động lực học phi tuyến phức tạp của con quay hồi chuyển và phương trình chuyển động Euler. Diễn đạt chính xác là: Các kênh cảm biến cung cấp thông tin đa chiều bổ trợ lẫn nhau (gia tốc đo rung động tịnh tiến, con quay đo dao động Roll-Pitch-Yaw) và không xảy ra hiện tượng đa cộng tuyến tuyến tính nghiêm trọng (|r| < 0.80).", "Đính chính lập luận khoa học: ")

# MỤC 2: MODEL TRAINING VÀ HYPERPARAMETERS
add_h1("2. QUÁ TRÌNH HUẤN LUYỆN, TUNING SIÊU THAM SỐ VÀ KHẢ NĂNG TÁI LẬP")
add_p("Để đảm bảo khả năng tái lập (Reproducibility), nhóm trình bày chi tiết quy trình chuẩn hóa, cấu hình siêu tham số và thời gian thực thi.")

add_h2("2.1. Quy trình Tiền xử lý & Chống rò rỉ dữ liệu (Anti-Leakage Pipeline)")
add_p("Tập dữ liệu gồm 18 file tương ứng với 18 chuyến bay/phiên thử nghiệm hoàn toàn riêng biệt. Sau bước chia cửa sổ 2 giây (~700 mẫu đo/cửa sổ) và lọc gián đoạn, thu được 988 cửa sổ hợp lệ (299 Normal, 689 Fault). Nhóm sử dụng StratifiedGroupKFold gom nhóm theo cột source_file (18 files), đảm bảo 100% cửa sổ của một chuyến bay chỉ nằm ở Train hoặc Test (Trùng lặp Leakage = 0). Phép chuẩn hóa StandardScaler được đóng gói trong Pipeline, chỉ fit trên tập Train của từng Fold và transform sang Test fold, ngăn chặn triệt để Data Leakage.")

add_h2("2.2. Chi tiết Siêu tham số & Kết quả Tinh chỉnh (Hyperparameter Tuning)")
add_p("Nhóm thực hiện GridSearchCV với 5-Fold Stratified Group CV, sử dụng F1-score làm metric tối ưu hóa. Bảng dưới đây so sánh cấu hình siêu tham số Baseline và Tuned:")

# Tạo bảng Tuning
t_tune = doc.add_table(rows=1, cols=5)
hdr = t_tune.rows[0].cells
hdr[0].text = "Mô hình"
hdr[1].text = "Cấu hình Baseline"
hdr[2].text = "F1 Baseline"
hdr[3].text = "Cấu hình Tuned tốt nhất"
hdr[4].text = "F1 Tuned"

row_lr = t_tune.add_row().cells
row_lr[0].text = "Logistic Regression"
row_lr[1].text = "C = 1.0, solver = 'lbfgs', max_iter = 1000"
row_lr[2].text = f"{summary['lr_mean_f1']:.4f}"
row_lr[3].text = "C = 0.1, solver = 'lbfgs', max_iter = 1000"
row_lr[4].text = f"{summary['best_lr_score']:.4f}"

row_rf = t_tune.add_row().cells
row_rf[0].text = "Random Forest"
row_rf[1].text = "n_estimators=100, max_depth=12, split=2, leaf=1"
row_rf[2].text = f"{summary['rf_mean_f1']:.4f}"
row_rf[3].text = "n_estimators=150, max_depth=16, split=5, leaf=2"
row_rf[4].text = f"{summary['best_rf_score']:.4f}"
format_table(t_tune)

add_h2("2.3. Thời gian Huấn luyện và Suy luận (Training & Inference Time)")
add_p("Đo lường thời gian thực thi trên CPU nhằm kiểm chứng tính khả thi khi triển khai trên vi điều khiển/máy tính nhúng của UAV:")

# Bảng thời gian
t_time = doc.add_table(rows=1, cols=4)
hdr = t_time.rows[0].cells
hdr[0].text = "Mô hình"
hdr[1].text = "Thời gian Train / Fold (s)"
hdr[2].text = "Thời gian Test / Fold (ms)"
hdr[3].text = "Suy luận / Cửa sổ 2s (µs)"

for _, r in df_times.iterrows():
    row = t_time.add_row().cells
    row[0].text = str(r.iloc[0])
    row[1].text = str(r.iloc[1])
    row[2].text = str(r.iloc[2])
    row[3].text = str(r.iloc[3])
format_table(t_time)

add_p("Thời gian suy luận của Random Forest chỉ mất xấp xỉ 144.9 µs cho một cửa sổ 2 giây, hoàn toàn đáp ứng yêu cầu giám sát thời gian thực trên UAV (vốn có chu kỳ quyết định tính bằng giây).")

# MỤC 3: ĐÁNH GIÁ MÔ HÌNH VÀ PHÂN TÍCH ĐÁNH ĐỔI
add_h1("3. ĐÁNH GIÁ MÔ HÌNH, TRADE-OFF PRECISION VS RECALL VÀ MẤT CÂN BẰNG DỮ LIỆU")

add_h2("3.1. Phân tích Ma trận Nhầm lẫn (Confusion Matrix)")
add_image_with_caption("reports/figures/fig4_confusion_matrices.png", "Hình 4: Ma trận nhầm lẫn tổng hợp sau 5 Folds giữa Logistic Regression và Random Forest")

add_h2("3.2. Phân tích Đánh đổi giữa Precision và Recall trong Bài toán An toàn UAV")
add_p("Kết quả thực nghiệm cho thấy sự đánh đổi rõ rệt giữa hai mô hình: Random Forest có Precision cao hơn (75.66% so với 70.74%), nhưng Logistic Regression lại có Recall cao hơn (95.57% so với 87.89%). Cụ thể, RF giảm được 76 ca False Positive (báo động giả) nhưng lại tăng thêm 46 ca False Negative (bỏ sót lỗi).")
add_p("• False Positive (FP - Báo động giả): Hệ thống cảnh báo lỗi cánh quạt khi UAV thực tế đang bay bình thường. Hệ quả: Bắt buộc UAV phải hủy nhiệm vụ, hạ cánh khẩn cấp không cần thiết, làm tăng chi phí vận hành và gián đoạn công tác kiểm tra.\n"
      "• False Negative (FN - Bỏ sót lỗi): Cánh quạt bị nứt/mẻ nhưng hệ thống không phát hiện. Hệ quả: Cực kỳ nguy hiểm, có thể dẫn tới gãy rời cánh quạt giữa không trung, mất cân bằng động lực học và rơi máy bay (catastrophic crash), phá hủy tải trọng và đe dọa an toàn con người bên dưới.\n"
      "• Đánh giá sâu sắc về Logistic Regression: Dù LR đạt Recall danh nghĩa rất cao (95.57%) nhưng thực chất mô hình bị suy biến thiên lệch về lớp đa số (Fault chiếm 69.7%). LR chỉ đoán đúng được vỏn vẹn 26/299 cửa sổ Normal, khiến tỷ lệ báo động giả trên tập Normal lên tới 91.3%! Một mô hình báo lỗi gần như toàn bộ thời gian bay không có giá trị ứng dụng thực tế.\n"
      "• Giải pháp Tối ưu hóa Ngưỡng (Threshold Tuning) cho Random Forest: Thay vì dùng ngưỡng mặc định 0.50, nhóm khảo sát biến thiên ngưỡng phân loại từ 0.1 đến 0.9 (Hình 6). Khi hạ ngưỡng quyết định xác suất xuống Th = 0.35, Random Forest nâng Recall lên 95.21% (FN giảm từ 77 xuống chỉ còn 33 ca), trong khi vẫn duy trì Precision ở mức 72.8%, vượt trội hơn hẳn so với Logistic Regression.", "Phân tích ý nghĩa hàng không và sự lựa chọn mô hình: ")

add_image_with_caption("reports/figures/fig6_precision_recall_tradeoff.png", "Hình 6: Phân tích đánh đổi Precision - Recall và Tối ưu hóa Ngưỡng quyết định cho An toàn UAV")

add_h2("3.3. Ảnh hưởng của Mất cân bằng Nhãn và Hạn chế của Metric Accuracy")
add_p("Tập dữ liệu có 689 cửa sổ Fault (69.7%) và 299 cửa sổ Normal (30.3%), tỷ lệ chênh lệch xấp xỉ 2.3 : 1. Trong phân bố mất cân bằng, Accuracy là chỉ số gây ngộ nhận nghiêm trọng: Một bộ phân loại tầm thường (dummy classifier) luôn luôn dự đoán mọi cửa sổ là Fault vẫn sẽ đạt Accuracy = 69.7%! Đó chính là lý do Logistic Regression dù đoán sai 91.3% tập Normal nhưng Accuracy tổng thể vẫn đạt 69.25%. Vì vậy, nhóm bắt buộc phải dựa vào Precision, Recall, F1-score của từng lớp và Macro-F1 để đánh giá công tâm năng lực phân tách của mô hình.")

# MỤC 4: KIỂM ĐỊNH THỐNG KÊ
add_h1("4. KIỂM ĐỊNH THỐNG KÊ VÀ GIẢI TRÌNH SỰ KHÁC BIỆT PAIRED T-TEST VS WILCOXON")
add_p("Bảng kết quả kiểm định ý nghĩa thống kê giữa Random Forest và Logistic Regression qua 5 Folds:")

# Bảng Statistical tests
t_stat = doc.add_table(rows=1, cols=6)
hdr = t_stat.rows[0].cells
hdr[0].text = "Metric (Fault = 1)"
hdr[1].text = "Logistic Regression"
hdr[2].text = "Random Forest"
hdr[3].text = "Chênh lệch (RF - LR)"
hdr[4].text = "Paired t-test (p-value)"
hdr[5].text = "Wilcoxon (p-value)"

for _, r in df_stats.iterrows():
    row = t_stat.add_row().cells
    row[0].text = str(r['Metric'])
    row[1].text = f"{r['Mean_LR']*100:.2f}% ± {r['Std_LR']*100:.2f}%"
    row[2].text = f"{r['Mean_RF']*100:.2f}% ± {r['Std_RF']*100:.2f}%"
    row[3].text = f"{r['Diff (RF - LR)']*100:+.2f}%"
    row[4].text = f"p = {r['Paired_t_pvalue']:.5f}"
    row[5].text = f"p = {r['Wilcoxon_pvalue']:.4f}"
format_table(t_stat)

add_p("Kiểm định Paired t-test cho Precision đạt p = 0.02496 < 0.05, nhưng Wilcoxon signed-rank test lại cho p = 0.0625 > 0.05. Sự khác biệt này bắt nguồn từ bản chất toán học của hai phương pháp:\n"
      "1. Paired t-test là kiểm định tham số (parametric), giả định hiệu số độ đo tuân theo phân phối chuẩn. Với mức tăng trung bình +4.92% Precision ở cả 5 folds, t-test khẳng định sự vượt trội có ý nghĩa ở mức α = 0.05.\n"
      "2. Wilcoxon là kiểm định phi tham số (non-parametric), không giả định phân phối chuẩn mà xếp hạng (ranking) hiệu số. Khi số lượng folds N = 5, không gian hoán vị thứ hạng cực kỳ nhỏ: Giá trị p-value tối thiểu mà kiểm định 2 phía của Wilcoxon có thể đạt được về mặt toán học chính là 2 × (1/2)⁵ = 2 × 1/32 = 0.0625. Điều này có nghĩa là: Ngay cả khi Random Forest thắng tuyệt đối cả 5/5 folds, Wilcoxon với mẫu N = 5 không bao giờ có thể đạt được p < 0.05!\n"
      "3. Kết luận phương pháp luận: Việc Wilcoxon có p = 0.0625 không phải do RF kém, mà do kích thước mẫu fold nhỏ (N = 5) làm hạn chế công suất thống kê (statistical power) của kiểm định phi tham số. Do đó, nhóm tránh kết luận 'RF vượt trội hoàn toàn' một cách chủ quan, mà nêu rõ: Có bằng chứng thống kê ban đầu từ t-test, và sẽ mở rộng lên 10-Fold CV hoặc Repeated CV (N ≥ 10) trong giai đoạn D4 để kiểm định Wilcoxon có đủ lực thống kê.", "Giải thích khoa học sự mâu thuẫn giữa 2 kiểm định: ")

# MỤC 5: ERROR ANALYSIS VÀ ĐẶC TRƯNG QUAN TRỌNG
add_h1("5. PHÂN TÍCH LỖI DỰ ĐOÁN (ERROR ANALYSIS) VÀ ĐẶC TRƯNG QUAN TRỌNG")

add_h2("5.1. Mức độ quan trọng của đặc trưng (Feature Importance)")
add_image_with_caption("reports/figures/fig5_rf_feature_importance.png", "Hình 5: Top 15 Đặc trưng đóng góp lớn nhất trong Random Forest")
add_p("Các đặc trưng đo độ biến thiên rung chấn (IQR, Std, RMS, Max) của hai trục gia tốc ngang AccX và AccY chiếm vị trí dẫn đầu: AccY_IQR (0.0470), AccX_Max (0.0460), AccX_IQR (0.0448), AccX_Std (0.0432). Kết quả này hoàn toàn nhất quán với quan sát ở EDA 1 và EDA 2: Lỗi cánh quạt chủ yếu tạo ra dao động rung lắc ngang và mở rộng phương sai thay vì thay đổi giá trị trung bình.")

add_h2("5.2. Phân tích Lỗi dự đoán theo Điều kiện bay (Error Analysis by Flight Condition)")
add_p("Bảng phân rã tỷ lệ dự đoán sai theo 3 nhóm điều kiện thử nghiệm:")

# Bảng Error analysis
t_err = doc.add_table(rows=1, cols=6)
hdr = t_err.rows[0].cells
hdr[0].text = "Điều kiện chuyến bay"
hdr[1].text = "Tổng mẫu"
hdr[2].text = "Sai LR"
hdr[3].text = "Tỷ lệ sai LR"
hdr[4].text = "Sai RF"
hdr[5].text = "Tỷ lệ sai RF"

for _, r in df_err_cond.iterrows():
    row = t_err.add_row().cells
    row[0].text = str(r['Condition'])
    row[1].text = str(r['Tong_Mau'])
    row[2].text = str(r['Sai_LR'])
    row[3].text = f"{r['Ty_le_sai_LR']:.2f}%"
    row[4].text = str(r['Sai_RF'])
    row[5].text = f"{r['Ty_le_sai_RF']:.2f}%"
format_table(t_err)

add_image_with_caption("reports/figures/fig7_error_analysis_by_condition.png", "Hình 7: Tỷ lệ dự đoán sai phân rã theo 3 điều kiện chuyến bay thực nghiệm")

add_p("1. Ở nhóm Fault 5% (Lỗi nhẹ - mẻ 5% cánh quạt): Tỷ lệ sai của Random Forest lên tới 16.51% (54 cửa sổ bị nhầm thành Normal). Nguyên nhân: Mức độ khuyết tật nhẹ tạo ra biên độ rung lắc chưa đủ vượt trội so với dao động khí động học tự nhiên của chuyến bay bình thường, dẫn tới nhiều ca False Negative.\n"
      "2. Ở nhóm Fault 10% (Lỗi nặng - mẻ 10% cánh quạt): Tỷ lệ sai của Random Forest giảm mạnh xuống chỉ còn 6.35% (23 cửa sổ). Khi khuyết tật tăng lên, đặc trưng rung chấn trở nên rất rõ rệt và RF nhận diện gần như chính xác tuyệt đối.\n"
      "3. Ở nhóm Normal: Random Forest dự đoán chính xác 102 cửa sổ (giảm được 76 ca báo động giả so với LR). Những cửa sổ Normal bị báo động giả tập trung vào các đoạn UAV thực hiện cơ động ngoặt gấp hoặc có nhiễu gió mạnh làm gia tốc tức thời tăng đột biến.", "Nguyên nhân dự kiến của các trường hợp dự đoán sai: ")

# MỤC 6: KẾ HOẠCH D4
add_h1("6. KẾ HOẠCH VÀ ĐỊNH HƯỚNG HOÀN THIỆN CHO BÁO CÁO CUỐI KỲ (D4)")
add_p("Tiếp thu định hướng của giáo viên, nhóm ưu tiên làm vững chắc nền tảng đánh giá và phân tích chuyên sâu trước khi mở rộng mô hình. Lộ trình thực hiện cho giai đoạn D4 gồm:\n"
      "1. Áp dụng Threshold Tuning (ngưỡng Th = 0.35) vào pipeline phân loại chính thức của UAV để giảm thiểu tối đa False Negative, bảo đảm an toàn bay.\n"
      "2. Mở rộng kiểm định thống kê với 10-Fold Stratified Group CV hoặc Repeated Stratified Group CV (5 folds × 3 lần lặp, N = 15) nhằm kiểm định Wilcoxon đạt đủ công suất thống kê (p < 0.05).\n"
      "3. Bổ sung các đặc trưng miền tần số (FFT, Spectral Energy, Dominant Frequency) nhằm phân biệt tốt hơn giữa lỗi nhẹ 5% và nhiễu gió tự nhiên.\n"
      "4. Sau khi baseline đã hoàn thiện vững chắc, tiến hành so sánh đối chứng với các mô hình Gradient Boosting nâng cao (XGBoost, LightGBM) để đánh giá khả năng cải thiện hơn nữa hiệu năng phân loại.")

output_docx_path = "reports/Bao_Cao_Tien_Do_D3_Cap_Nhat.docx"
doc.save(output_docx_path)
print(f" -> Đã lưu tài liệu Word hoàn thiện: {output_docx_path}")

# -------------------------------------------------------------
# 2. XUẤT TÀI LIỆU MARKDOWN (.MD) ĐỂ XEM NHANH TRÊN VS CODE
# -------------------------------------------------------------
md_content = f"""# BÁO CÁO TIẾN ĐỘ D3 (BẢN HOÀN THIỆN THEO GÓP Ý GIẢNG VIÊN)
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
| **Logistic Regression** | C = 1.0, solver = 'lbfgs', max_iter = 1000 | {summary['lr_mean_f1']:.4f} | C = 0.1, solver = 'lbfgs', max_iter = 1000 | {summary['best_lr_score']:.4f} |
| **Random Forest** | n_estimators=100, max_depth=12, split=2, leaf=1 | {summary['rf_mean_f1']:.4f} | n_estimators=150, max_depth=16, split=5, leaf=2 | {summary['best_rf_score']:.4f} |

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
  - Wilcoxon (kiểm định phi tham số) dựa trên thứ hạng. Với $N = 5$ folds, $p$-value nhỏ nhất có thể đạt được về mặt toán học là $2 \\times (0.5)^5 = 0.0625$. Nghĩa là dù RF thắng tuyệt đối cả 5 folds thì Wilcoxon cũng không thể nào $< 0.05$.
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
"""

output_md_path = "reports/BAO_CAO_TIEN_DO_D3_CAP_NHAT.md"
with open(output_md_path, "w", encoding="utf-8") as f:
    f.write(md_content)
print(f" -> Đã lưu tài liệu Markdown hoàn thiện: {output_md_path}")
print("=== TẠO BÁO CÁO CẬP NHẬT HOÀN TẤT THÀNH CÔNG ===")
