# Quyết định dữ liệu training B0 (nội bộ team)

**Người lập:** Hoàng (Data) — nhánh `b0-data-preparation`
**Ngày:** 2026-10-11 — trạng thái data: verify 4/4 PASS (`experiments/B0/exp01-hoang-full-data.md`)
**Kết luận trước:** train chính dùng **FULL**, **subset-100** làm baseline-lite dự phòng/smoke-test.

---

## 1. Vì sao phải quyết định "dùng bao nhiêu ảnh"?

Kho SfM đầy đủ (~98k ảnh train/val) không bao giờ đi hết vào model một lúc:
mỗi epoch chỉ gồm **pool mining (~10k ảnh)** + **tuples train (~1.000 tuple × 7 ảnh)**.
Kho càng to thì negatives càng đa dạng (model khôn hơn), nhưng tiền trả mỗi epoch
chỉ tính trên ~17k ảnh đó. Ngược lại Kaggle T4 giới hạn 12h/session + disk ~20GB,
nên phải chốt: train FULL luôn, hay subset nhỏ hơn cho vừa giờ.

## 2. Số lượng ảnh trong kho FULL

| Split | Ảnh | Cluster | Cặp (q,p) | Ghi chú phân bố |
|---|---|---|---|---|
| train | 91.642 | 551 | 181.697 | Lệch nặng: cụm to nhất 2.082 ảnh, median 68, mean 166,3 |
| val | 6.403 | 162 | 1.691 | Cụm nhỏ đều (median 31, max 80) |
| **Tổng supervision** | **98.045** | **713** | **183.388** | File ảnh trên disk: 117.369 (thừa ~19k ngoài supervision, loader theo pkl nên vô hại) |

Test giữ nguyên, không cắt: Oxford5k 5.063 ảnh/55 query, Paris6k 6.412 ảnh/55 query
(protocol legacy `ok/junk`, single-scale, không whitening/QE).

## 3. Thực tế mỗi epoch dùng bao nhiêu (config đề xuất, T4, imsize 768)

| Hạng mục | Số ảnh/epoch | Giải thích |
|---|---|---|
| Mining pool | 10.000 | Quét tìm 5 negatives khó cho mỗi query |
| Tuples train | 1.000 × 7 = 7.000 | 1 query + 1 positive + 5 negatives |
| **Tổng/epoch** | **~17.000 ảnh** | ≈ 30–40 phút trên T4 → ~8 epoch/session 12h |

Điều kiện subset phải thỏa: số cặp còn lại ≥ qsize (1.000) và số cluster − 1 ≥ 5
(đủ negatives khác cụm cho mọi query).

## 4. Hai phương án

### Phương án A — FULL (khuyến nghị cho số B0 chính)

- Dùng: toàn bộ 91.642 ảnh / 551 cluster / 181.697 cặp.
- Ưu: diversity negatives cao nhất, số sát paper nhất, không ai chất vấn "cắt bớt thì còn gì".
- Nhược: disk ~14GB + benchmarks 4GB + weights (vừa Kaggle input mount, chật working nếu để cả checkpoint) —
  khắc phục bằng mount 3 dataset rời + checkpoint ra `/kaggle/working`, resume nhiều session.
- Khi dùng: chạy chính, target 10–15 epoch qua 2 session.

### Phương án B — subset-100 seed 42 (dự phòng + smoke-test)

- Dùng: 15.953 ảnh / 100 cluster / 34.339 cặp train (+ val 20 cluster/739 ảnh), disk ~4,1GB.
- Ưu: nhẹ, mining vẫn đạt 100% (99 neg-cluster/query), epoch cost **bằng hệt FULL**
  (vì epoch chỉ phụ thuộc qsize/poolsize/imsize), 1 session xong ~10 epoch ở imsize 512.
- Nhược: kém diversity hơn FULL; số là "B0-lite", phải ghi rõ deviation khi so với paper.
- Khi dùng: (1) smoke-test pipeline 2–3 epoch trước khi đốt giờ vào FULL;
  (2) baseline tạm nếu hết quota GPU; (3) ablation size-vs-mAP về sau.

### Bằng chứng đo (seed 42, `scripts/profile_sfm.py`)

| k | Ảnh train | Cặp train | Disk | neg-cluster | mining |
|---|---|---|---|---|---|
| 25 | 6.399 | 10.734 | 1,7GB | 24 | OK (loại: diversity thấp, random dễ lệch do cụm 2.082 ảnh) |
| 50 | 9.662 | 20.012 | 2,5GB | 49 | OK — giữ làm smoke-test |
| **100** | **15.953** | **34.339** | **4,1GB** | **99** | **OK — chốt fallback** |
| 200 | 35.767 | 69.692 | 9,3GB | 199 | OK (tải gấp đôi, lợi thêm ít — để ablation sau) |

Phát hiện thêm: epoch cost không đổi theo k (k=25 vẫn đủ 10.734 cặp ≥ qsize 1.000),
nên tiêu chí chọn là **diversity + disk**, không phải giờ/epoch.

## 5. Tái tạo

```bash
# FULL (phương án A): metadata đã là full sau exp01
python3 scripts/download_datasets_local.py --only sfm-meta --full
# subset-100 (phương án B):
python3 scripts/download_datasets_local.py --only sfm-meta --num-clusters 100 --seed 42
python3 scripts/profile_sfm.py --data-root ./data --seed 42
```

Provenance mỗi subset: `data/train/retrieval-SfM-120k/subset_provenance_k*_seed42.txt`.
Chi tiết verify + sự cố đã gặp: `experiments/B0/exp01-hoang-full-data.md`.
