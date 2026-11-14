# 🎯 Customer Ticket Routing & Intent Classification Service

[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.16+-005CED.svg?logo=onnx&logoColor=white)](https://onnxruntime.ai)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E.svg?logo=huggingface&logoColor=black)](https://huggingface.co)
[![PEFT / LoRA](https://img.shields.io/badge/PEFT-LoRA-FF6F00.svg?logo=pytorch&logoColor=white)](https://github.com/huggingface/peft)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg?logo=mlflow&logoColor=white)](https://mlflow.org)
[![uv](https://img.shields.io/badge/uv-Package_Manager-DE5FE9.svg?logo=astral&logoColor=white)](https://astral.sh/uv)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com)

Высокопроизводительный микросервис для автоматической классификации и маршрутизации клиентских обращений в службу поддержки банка по **77 категориям интентов** (на базе датасета `Banking77`).

Проект демонстрирует **полный производственный цикл ML-разработки**:
* Исследование данных и тюнинг baseline (`TF-IDF + Logistic Regression` с байесовской оптимизацией через Optuna).
* Дообучение трансформера DistilBERT: сравнительный бенчмарк **Full Fine-Tuning** против **PEFT / LoRA** ($r=32, \alpha=32$).
* Трекинг всех экспериментов, метрик и артефактов в **MLflow**.
* Подготовка модели к продакшну: слияние весов LoRA (`merge_and_unload`) и конвертация графа в **ONNX**.
* Асинхронный REST API на **FastAPI** с валидацией входных данных Pydantic и упаковкой в **Docker**.

---

## 📊 Результаты экспериментов (Test Set Benchmark)

Все модели оценивались на отложенной тестовой выборке (`test split`):

| Модель / Подход | Обучаемые параметры | Test Macro $F_1$ | Test Accuracy | Test Precision | Test Recall | Test Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (TF-IDF + LogReg + Bayes)** | — | 0.8901 | — | — | — | — |
| **DistilBERT + LoRA ($r=32$)** | **~2.6% (~1.8M)** | **0.9132** | **0.9133** | **0.9163** | **0.9133** | **0.3189** |
| **DistilBERT (Full Fine-Tuning)** | 100% (66.9M) | **0.9175** | **0.9175** | **0.9209** | **0.9175** | **0.3478** |

---

### 💡 Ключевые выводы:

1. **Эффективность LoRA (PEFT):**
   * Обучение всего **~2.6% параметров** позволило достичь **99.5% качества** от полного дообучения (Macro $F_1$: **0.9132** против **0.9175**).
   * Время обучения сократилось более чем на **26%** (15.9 минут у LoRA против 21.5 минут у Full Fine-Tuning при одинаковом числе эпох).
   * Размер сохраняемого чекпоинта адаптера составил всего **~20 МБ** против **~260 МБ** у полной модели.

2. **Защита от переобучения:**
   * У модели с **LoRA тестовый Loss оказался ниже**, чем у Full Fine-Tuning (**0.3189** против **0.3478**), что демонстрирует отличную регуляризирующую способность низкоранговой адаптации на небольших специфичных выборках.

3. **Сравнение с Baseline:**
   * Классический ML (TF-IDF + Logistic Regression) показал сильный результат (**0.8901**), однако применение трансформера дало существенный прирост качества классификации сложных и семантически похожих интентов (+2.3% к $F_1$).

---

## 🛠 Технологический стек

* **Среда и зависимости:** `uv`, Python 3.11
* **ML / DL:** PyTorch, Hugging Face (Transformers, PEFT, Datasets), Scikit-Learn
* **Inference Engine:** ONNX Runtime
* **Эксперименты:** MLflow, Optuna
* **Бэкенд:** FastAPI, Uvicorn, Pydantic v2
* **Деплой:** Docker, Dockerfile