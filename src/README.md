# Completed Lab Implementation

Thư mục `src/` chứa bản triển khai hoàn chỉnh của bài lab:

- `model_provider.py`: cấu hình và khởi tạo 6 provider
- `config.py`: đường dẫn, environment variables, model và compact settings
- `memory_store.py`: `User.md`, fact extraction, conflict handling và compact memory
- `agent_baseline.py`: within-thread memory, không có persistent memory
- `agent_advanced.py`: short-term, persistent profile và compact memory
- `benchmark.py`: standard benchmark và long-context stress benchmark
- `test_agents.py`: test storage, compaction, recall, prompt load và guardrail

Chạy từ root repo:

```bash
python src/benchmark.py
pytest src/test_agents.py -v
```

Datasets nằm trong thư mục `data/`; kết quả mẫu và phần phân tích nằm trong `RESULTS.md`.
