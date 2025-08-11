# Lung X-ray Multi-label Classification (ResNet152) + CLIP Training

## 1. Chuẩn bị môi trường
- Bật GPU (Kaggle/Colab/Local CUDA)
- Cài đặt phụ thuộc:
```bash
pip install -r requirements.txt
```

## 2. Train mô hình ResNet152 (fine-tune toàn bộ)
```bash
python resnet152_train.py --base_path "/path/to/dataset/root"
```
- Mặc định script sẽ dùng CSV tại `BASE_PATH/mimic-cxr.csv` (tự động ghép từ `--base_path`).
- Kết quả: model tốt nhất lưu tại `/kaggle/working/best_medical_resnet152.pth` (hoặc chỉnh `SAVE_PATH` trong `resnet152_train.py`).

## 3. Train CLIP (image–text contrastive) cho MIMIC-CXR
### 3.1. Yêu cầu dữ liệu (CSV)
Sử dụng file `mimic_cxr_aug_train.csv` với các cột sau:
- `image_frontal_path`: đường dẫn ảnh frontal (tương đối từ `base_path` hoặc tuyệt đối)
- `image_lateral_path`: đường dẫn ảnh lateral (có thể trống; nếu trống sẽ dùng tensor 0)
- `patient_id`: dùng để split theo bệnh nhân
- `split`: `train` / `val` / `test`
- `findings`: văn bản Findings đã được clean ở mức cơ bản
- `label_vector`: chuỗi JSON hoặc danh sách Python mô tả multi-label (ví dụ: "[0,1,0, ...]")

Lưu ý đường dẫn ảnh nên là tương đối so với `--base_path` để script tự ghép; vẫn hỗ trợ đường dẫn tuyệt đối.

### 3.2. Chạy train
Ví dụ (single-GPU):
```bash
python train_clip.py \
  --base_path "D:/Research/MIMIC-CXR" \
  --model_name ViT-B-16 \
  --pretrained openai \
  --fuse_type avg \
  --batch_size 64 \
  --epochs 20 \
  --use_amp
```
Các tham số chính:
- `--base_path`: thư mục gốc chứa dữ liệu và CSV
- `--csv_path`: (tuỳ chọn) đường dẫn CSV; mặc định `base_path/mimic_cxr_aug_train.csv`
- `--model_name`: `ViT-B-16` hoặc `RN50`
- `--pretrained`: nguồn pretrained của open-clip (ví dụ `openai`)
- `--fuse_type`: cách fuse frontal + lateral (`avg` hoặc `concat`)
- `--text_ctx_len`: độ dài context tokenizer (mặc định 256)
- `--accum_steps`: gradient accumulation nếu RAM/GPU hạn chế
- `--use_amp`: dùng mixed precision (fp16)
- `--use_wandb`: bật logging Weights & Biases (cần `pip install wandb`)

Resume training:
```bash
python train_clip.py --base_path "..." --resume checkpoints_clip/last.pt --use_amp
```

### 3.3. Đánh giá CLIP
```bash
python evaluate_clip.py \
  --base_path "D:/Research/MIMIC-CXR" \
  --checkpoint checkpoints_clip/best.pt \
  --split test \
  --save_vis tsne_test.png
```
- In ra metrics retrieval: `R@1`, `R@5`, `R@10` cho cả ảnh→text và text→ảnh
- Linear probe (nếu có `label_vector`): Macro F1, Micro F1, mAP
- Lưu t-SNE embedding ảnh ra file png

### 3.4. Logging
- TensorBoard: logs trong `runs/clip_training`
```bash
tensorboard --logdir runs/clip_training
```
- Weights & Biases: thêm `--use_wandb` khi train

## 4. Cấu trúc code CLIP
- `clip_dataset.py`: Dataset/Dataloader CLIP, dual-view, transform chuẩn CLIP, tokenize bằng open-clip
- `clip_model.py`: Wrapper open-clip, fuse dual-view (`avg`/`concat`), L2-normalize, `logit_scale`
- `train_clip.py`: huấn luyện InfoNCE 2 chiều, AMP, accumulation, scheduler cosine, checkpoint, logging
- `evaluate_clip.py`: tính retrieval, linear probe, t-SNE
- `clip_eval.py`: helper cho embedding, metrics, visualization

## 5. Troubleshooting
- Lỗi `ImportError: open_clip_torch`:
  - Cài `open-clip-torch`: `pip install open-clip-torch`
- CUDA OOM:
  - Giảm `--batch_size`, tăng `--accum_steps`, dùng `--use_amp`
- Không tìm thấy ảnh (`FileNotFoundError`):
  - Kiểm tra `image_frontal_path`, `image_lateral_path` trong CSV có đúng tương đối với `--base_path` không; hoặc dùng đường dẫn tuyệt đối
  - Với Windows, chấp nhận cả `\` và `/`; script sẽ ghép bằng `os.path.join`
- Token length quá dài:
  - Giảm `--text_ctx_len` (ví dụ 128)
- Checkpoint không load được:
  - Kiểm tra `--model_name`, `--fuse_type` khi evaluate có khớp với khi train
- Cột CSV khác tên:
  - Đảm bảo các cột đúng như mục 3.1; nếu khác cần sửa `clip_dataset.py`

## 6. Ghi chú mở rộng
- Hiện hỗ trợ single-GPU. Có thể mở rộng DDP (multi-GPU) bằng `torchrun` với `DistributedSampler` (cập nhật script nếu cần).
- Có thể chuẩn hoá text nâng cao trong pipeline tiền xử lý trước khi train (loại headers/dates/patient IDs).

---
**Chỉnh sửa tham số train trong đầu file `resnet152_train.py` nếu cần.**