# Kết quả và phân tích

Các số liệu dưới đây được tạo bằng chế độ offline deterministic với lệnh:

```bash
python src/benchmark.py
```

## Standard Benchmark

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 1,379 | 13,988 | 0.0% | 20.7% | 0 | 0 |
| Advanced | 1,977 | 25,067 | 100.0% | 100.0% | 463 | 0 |

Ở các hội thoại ngắn, Advanced tốn thêm prompt vì luôn mang theo `User.md`. Chi phí cố định này chưa được bù bởi compact memory, nhưng đổi lại agent nhớ đúng qua thread mới.

## Long-Context Stress Benchmark

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 319 | 22,447 | 0.0% | 20.0% | 0 | 0 |
| Advanced | 433 | 16,719 | 100.0% | 100.0% | 310 | 2 |

Trong stress benchmark, compact memory giảm khoảng 25.5% prompt tokens processed. Baseline phải gửi lại toàn bộ lịch sử ở mỗi lượt nên chi phí tăng theo kiểu tích lũy; Advanced chỉ giữ summary và các message gần nhất.

## Trade-off và guardrail

- `User.md` giúp cross-session recall nhưng tạo state cần quản lý và tăng kích thước theo thời gian.
- Compact memory chủ yếu giảm prompt load; nó không nhất thiết giảm lượng token do agent sinh ra.
- Extractor dùng confidence threshold bảo thủ: bỏ mọi câu hỏi, chỉ nhận mẫu fact rõ ràng, và ghi đè location/profession khi có correction.
- Fact nhiều giá trị như interests và response style được merge; fact xung đột như nơi ở và nghề nghiệp dùng giá trị mới nhất.
- Rủi ro còn lại là regex không hiểu hết ngôn ngữ tự nhiên. Production nên thêm structured extraction, confidence score, provenance, decay, và cơ chế người dùng xem/xóa memory.
