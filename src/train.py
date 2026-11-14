from datasets import load_dataset
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments
from sklearn.metrics import *
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.utils.class_weight import compute_class_weight
import json

load_dotenv()


PROJECT_ROOT = Path.cwd()
MODEL_NAME = "distilbert-base-uncased"
DATASET_NAME = "AutoIntent/banking77"
RANDOM_SEED = 42
MAX_LEN = 64
NUM_CLASSES = 77

with open(PROJECT_ROOT / "src" / "id2label.json", 'r') as f:
    id2label = json.load(f)
    id2label = {int(i): label  for i, label in  id2label.items()}
    label2id = {label: int(i) for i, label in id2label.items()}




model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    cache_dir=PROJECT_ROOT / "transformers" / MODEL_NAME / "model",
    num_labels = NUM_CLASSES,
    id2label = id2label,
    label2id = label2id
)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=PROJECT_ROOT / "transformers" / MODEL_NAME / "tokenizer"
    )

def tokenize(batch):
    return tokenizer(batch['utterance'], truncation=True, max_length=MAX_LEN)

dataset = load_dataset(
   "AutoIntent/banking77",
    cache_dir= PROJECT_ROOT / "data"
)
train_dataset = dataset['train'].map(tokenize, batched=True).remove_columns('utterance')
test_dataset = dataset['test'].map(tokenize, batched=True).remove_columns('utterance')


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)
    return {
        "macro_f1":        f1_score(labels, preds, average="macro",        zero_division=0),
        "weighted_f1":     f1_score(labels, preds, average="weighted",     zero_division=0),
        "recall_macro":    recall_score(labels, preds, average="macro",    zero_division=0),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "accuracy":        accuracy_score(labels, preds),
    }


cw = compute_class_weight(
    "balanced",
    classes=np.unique(train_dataset['label']),
    y=train_dataset['label']
)
class_weights = torch.tensor(cw, dtype=torch.float)

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.CrossEntropyLoss(
            weight=class_weights.to(outputs.logits.device)
        )(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir=f"runs/hf/{MODEL_NAME}",
    learning_rate=3e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    logging_steps=50,
    run_name=f"{MODEL_NAME}_weighet_loss_lr3e-5_ep4",        
    seed=RANDOM_SEED,
    max_steps=-1
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics
)

trainer.train()


test_preds = trainer.predict(test_dataset)
metrics = test_preds.metrics
for key in metrics.keys():
    print(f"{key} \t\t{metrics[key]}")

save_path = PROJECT_ROOT / "models" / training_args.run_name
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
print(f"✅ Модель и токенизатор успешно сохранены в: {save_path}")