import os
import argparse
import wandb
from huggingface_hub import login
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments, Seq2SeqTrainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_NAME = "ntphuc149/ViBidLAQA_base"

# --- Biến môi trường ---
def get_env_variable(varname):
    value = os.environ.get(varname)
    if value is None:
        raise EnvironmentError(f"Biến môi trường '{varname}' chưa được set.")
    return value

# --- Load model, tokenizer, LoRA, collator ---
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        device_map="auto",
        load_in_4bit=True
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

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
        return_tensors="pt"
    )

    return model, tokenizer, data_collator

# --- Tiền xử lý dữ liệu ---
def preprocess_data(tokenizer, train_path, valid_path):
    dataset = load_dataset("parquet", data_files={
        "train": train_path,
        "valid": valid_path
    })

    def preprocess_function(examples):
        inputs = [f"question: {q} context: {c}" for q, c in zip(examples["question"], examples["context"])]
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding="max_length")

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["answer"], max_length=256, truncation=True, padding="max_length")

        model_inputs["labels"] = [
            [(token if token != tokenizer.pad_token_id else -100) for token in seq]
            for seq in labels["input_ids"]
        ]

        return model_inputs

    tokenized = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    return tokenized


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--saved_model_path", type=str, default="./vnt5-base-qa-final")
    parser.add_argument("--run_name", type=str, default="run-wandb")
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--valid_path", type=str, required=True)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    hf_token = get_env_variable("HF_TOKEN")
    wandb_key = get_env_variable("WANDB_API_KEY")

    # Login
    login(token=hf_token)
    wandb.login(key=wandb_key)

    # Load model and data
    model, tokenizer, data_collator = load_model_and_tokenizer(MODEL_NAME)
    tokenized_dataset = preprocess_data(tokenizer, args.train_path, args.valid_path)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="tmp/",
        do_train=True,
        do_eval=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=500,
        eval_steps=500,
        logging_dir="./log",
        logging_steps=100,
        logging_first_step=True,
        save_total_limit=1,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=1e-5,
        warmup_ratio=0.05,
        weight_decay=0.01,
        fp16=True,
        report_to="wandb",
        run_name=args.run_name,
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

    trainer.train()

    # Save model and tokenizer
    trainer.save_model(args.saved_model_path)
    tokenizer.save_pretrained(args.saved_model_path)
    
if __name__ == "__main__":
    main()
