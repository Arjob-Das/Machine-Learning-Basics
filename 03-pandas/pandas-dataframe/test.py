from transformers import TrainingArguments, Trainer, logging
from transformers import AutoModelForSequenceClassification
import torch
import numpy as np
from datasets import Dataset
from pynvml import *

seq_len, dataset_size = 512, 512
dummy_data = {
    "input_ids": np.random.randint(100, 30000, (dataset_size, seq_len)),
    "labels": np.random.randint(0, 1, (dataset_size)),
}
ds = Dataset.from_dict(dummy_data)
ds.set_format("pt")


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


print_gpu_utilization()


torch.ones((1, 1)).to("cuda")
print_gpu_utilization()


model = AutoModelForSequenceClassification.from_pretrained(
    "bert-large-uncased").to("cuda")
print_gpu_utilization()


logging.set_verbosity_error()

default_args = {
    "output_dir": "tmp",
    "evaluation_strategy": "steps",
    "num_train_epochs": 1,
    "log_level": "error",
    "report_to": "none",
}
training_args = TrainingArguments(
    per_device_train_batch_size=1, **default_args)
trainer = Trainer(model=model, args=training_args, train_dataset=ds)
result = trainer.train()
print_summary(result)
