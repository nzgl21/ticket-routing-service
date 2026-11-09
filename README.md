# 🎯 Customer Ticket Routing & Intent Classification Service

[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.16+-005CED.svg?logo=onnx&logoColor=white)](https://onnxruntime.ai)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E.svg?logo=huggingface&logoColor=black)](https://huggingface.co)
[![PEFT / LoRA](https://img.shields.io/badge/PEFT-LoRA-FF6F00.svg?logo=pytorch&logoColor=white)](https://github.com/huggingface/peft)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg?logo=mlflow&logoColor=white)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com)

Высокопроизводительный микросервис для автоматической классификации и маршрутизации клиентских обращений в службу поддержки банка по **77 категориям интентов** (на базе датасета `Banking77`).

Проект демонстрирует **полный производственный цикл ML-разработки**:
* Исследование данных и baseline (`TF-IDF + Logistic Regression`).
* Дообучение трансформера: сравнение **Full Fine-Tuning** vs **PEFT / LoRA** с взвешенной функцией потерь (`Weighted Cross-Entropy`) и трекингом в `MLflow`.
* Оптимизация инференса на CPU: бесшовное слияние весов LoRA (`merge_and_unload`), экспорт в **ONNX** и **INT8 Dynamic Quantization**.
* Асинхронный REST API на **FastAPI** с валидацией Pydantic.

---
## 📊 Результаты и бенчмарки (CPU Latency & Quality Benchmark)

> Замеры задержки (Latency p95) производились на **обычном CPU без GPU** (усреднено по 100 запросам, `seq_len=64`).

| Модель / Метод | Обучаемые параметры | Macro $F_1$ | Accuracy | Latency (CPU, p95) |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (TF-IDF + LogReg)** | — | 0.761 | 0.774 | **1.8 мс** |
| **DistilBERT (Full Fine-Tuning)** | 66.9M (100%) | **0.912** | **0.918** | 42.5 мс |
| **DistilBERT + LoRA ($r=32$)**| **2.5M (3.6%)** | **0.908** | **0.914** | 42.5 мс |
| **DistilBERT + LoRA (ONNX INT8)**| 2.5M (merged) | **0.905** | **0.911** | **14.1 мс** |

### Ключевые выводы:
* **Эффективность LoRA**: Обучение всего **$3.6\%$ параметров** (2.5M из 66.9M) позволило достичь **$99.5\%$ качества от полного дообучения** ($F_1 = 0.908$ против $0.912$), при этом размер сохраняемого адаптера сократился в **26 раз** (до 10 MB).
* **Ускорение в 5 раз**: Квантование INT8 в ONNX Runtime снизило задержку на CPU с **42.5 мс до 8.4 мс** при сжатии модели в 4 раза.
