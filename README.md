# Lung X-ray Multi-label Classification (ResNet152)

## 1. Chuẩn bị trên Kaggle
- Bật GPU trong Settings
- Upload toàn bộ mã nguồn vào thư mục làm việc

## 2. Train mô hình (fine-tune toàn bộ ResNet152)
```bash
python main.py
```
- Kết quả: Model tốt nhất lưu tại `/kaggle/working/best_medical_resnet152.pth`

## 3. Đánh giá mô hình đã train
```python
from evaluate import evaluate_multi_label, print_evaluation_results
# model = ... (load model, load weights)
metrics = evaluate_multi_label(model, test_loader, device, label_cols)
print_evaluation_results(metrics, label_cols)
```

## 4. Đánh giá mô hình pretrained (chưa train)
```python
from evaluate_origin import evaluate_pretrained, print_evaluation_results_origin
# model_pretrained = ... (khởi tạo model, không load weights train)
metrics_pre = evaluate_pretrained(model_pretrained, test_loader, device, label_cols)
print_evaluation_results_origin(metrics_pre, label_cols)
```

## 5. So sánh hiệu quả
- So sánh các chỉ số F1, AUC, ... giữa mô hình đã train và pretrained để đánh giá khả năng trích xuất đặc trưng y khoa.

---
**Chỉnh sửa tham số train trong đầu file `main.py` nếu cần.**