import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    get_scheduler
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from torch.cuda.amp import autocast, GradScaler

# Khai báo model name
model_name = "VietAI/vit5-base"

# Load model + tokenizer + LoRA
def load_model():
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q", "v"],
        lora_dropout=0.1,
        bias="none"
    )
    peft_model = get_peft_model(model, peft_config)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt"
    )
    return peft_model, tokenizer, data_collator

# Tiền xử lý dữ liệu
def preprocess_data(tokenizer, data_collator, batch_size=8):
    dataset = load_dataset("parquet", data_files={
        "train": "/kaggle/input/medical-qa-dataset/Data/Dataset/train-00000-of-00001.parquet",
        "valid": "/kaggle/input/medical-qa-dataset/Data/Dataset/validation-00000-of-00001.parquet",
        "test": "/kaggle/input/medical-qa-dataset/Data/Dataset/test-00000-of-00001.parquet"
    })

    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples["question"], examples["context"],
            max_length=2048, truncation=True, padding="max_length"
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["answer"],
                max_length=512, truncation=True, padding="max_length"
            )

        labels_ids = labels["input_ids"]
        labels_ids = [
            [(token if token != tokenizer.pad_token_id else -100) for token in seq]
            for seq in labels_ids
        ]

        model_inputs["labels"] = labels_ids
        return model_inputs

    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    train_dataloader = DataLoader(
        tokenized_dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator
    )

    eval_dataloader = DataLoader(
        tokenized_dataset["valid"],
        batch_size=batch_size,
        collate_fn=data_collator
    )

    return train_dataloader, eval_dataloader

# Hàm train với multi-GPU + mixed precision + early stopping
def train(peft_model, train_dataloader, eval_dataloader, num_epochs=30, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hỗ trợ multi-GPU
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        peft_model = torch.nn.DataParallel(peft_model)

    peft_model.to(device)

    optimizer = AdamW(peft_model.parameters(), lr=1e-5, weight_decay=0.01)
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.05 * num_training_steps),
        num_training_steps=num_training_steps,
    )

    scaler = GradScaler()  # for mixed precision
    best_eval_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        peft_model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            with autocast():  # mixed precision
                outputs = peft_model(**batch)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} avg training loss: {avg_loss:.4f}")

        # Evaluation
        peft_model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with autocast():
                    outputs = peft_model(**batch)
                    loss = outputs.loss
                    if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                        loss = loss.mean()
                eval_loss += loss.item()

        avg_eval_loss = eval_loss / len(eval_dataloader)
        print(f"Epoch {epoch+1} eval loss: {avg_eval_loss:.4f}")

        # Early stopping
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            epochs_without_improvement = 0
            peft_model.save_pretrained("VietAI/vit5-base_fine_tune")
            print("Model saved")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s)")
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

# ---- CHẠY TOÀN BỘ ----
if __name__ == "__main__":
    batch_size = 16  # tùy vào bộ nhớ GPU
    peft_model, tokenizer, data_collator = load_model()
    train_dataloader, eval_dataloader = preprocess_data(tokenizer, data_collator, batch_size)
    train(peft_model, train_dataloader, eval_dataloader, num_epochs=30, patience=5)
    tokenizer.save_pretrained("VietAI/vit5-base_fine_tune")
