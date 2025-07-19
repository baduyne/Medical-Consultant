import os
from datasets import load_dataset
from transformers import (
    Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq,
    T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
)
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model, prepare_model_for_kbit_training
import wandb

# --- WANDB Login ---
os.environ["WANDB_API_KEY"] = "" # fill your token
wandb.login()

# --- Model name and output path ---
model_name = "VietAI/vit5-base"
saved_model_path = "./vit5-base-qa-final"

# --- Load model/tokenizer/data_collator ---
def load_model_and_tokenizer():
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto", load_in_4bit = True)
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q", "v"],
        lora_dropout=0.1,
        bias="none"
    )
    model = get_peft_model(model, peft_config)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt")

    return model, tokenizer, data_collator

# --- Preprocess dataset ---
def preprocess_data(tokenizer):
    dataset = load_dataset("parquet", data_files={
        "train": "/kaggle/input/medical-qa-dataset/Data/Dataset/train-00000-of-00001.parquet",
        "valid": "/kaggle/input/medical-qa-dataset/Data/Dataset/validation-00000-of-00001.parquet",
        "test": "/kaggle/input/medical-qa-dataset/Data/Dataset/test-00000-of-00001.parquet"
    })

    def preprocess_function(examples):
        # Tạo chuỗi input_text dạng "question: ... context: ..."
        input_texts = [f"question: {q} context: {c}" for q, c in zip(examples["question"], examples["context"])]
    
        # Tokenize input
        model_inputs = tokenizer(
            input_texts,
            max_length=2048,
            truncation=True,
            padding="max_length"
        )
    
        # Tokenize output (labels)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["answer"],
                max_length=256,
                truncation=True,
                padding="max_length"
            )
    
        # Thay pad_token_id thành -100 để không tính loss
        model_inputs["labels"] = [
            [(token if token != tokenizer.pad_token_id else -100) for token in seq]
            for seq in labels["input_ids"]
        ]
    
        return model_inputs

    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    return tokenized_dataset

# --- Main ---
model, tokenizer, data_collator = load_model_and_tokenizer()
tokenized_dataset = preprocess_data(tokenizer)

training_args = Seq2SeqTrainingArguments(
    output_dir="tmp/",
    do_train=True,
    do_eval=True,
    eval_strategy="steps",   
    save_strategy="steps",
    save_steps=500,
    eval_steps=500,
    logging_dir="./log",
    logging_steps=100,
    logging_first_step=True, 
    save_total_limit=1,
    num_train_epochs=40,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=1e-5,
    warmup_ratio=0.05,
    weight_decay=0.01,
    fp16=True,
    report_to="wandb",
    run_name="vit5-medicalqa-lora",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    label_names=["labels"]
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["valid"]
)

trainer.train(resume_from_checkpoint=True)

trainer.save_model(saved_model_path)
tokenizer.save_pretrained(saved_model_path)
