from datasets import load_dataset
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, BitsAndBytesConfig, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch
# định nghĩa các load model 
quantized_config = BitsAndBytesConfig(
    load_in_4bit= True,
    bnb_4bit_use_double_quant = True,
    bnb_4bit_compute_dtype = torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

model_name = "PhucDanh/vit5-fine-tuning-for-question-answering"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name,
                                              quantization_config = quantized_config, device_map = "auto")


peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q", "v"],
    lora_dropout=0.1,
    bias="none"
)

peft_model = get_peft_model(model, peft_config)

dataset = load_dataset("parquet", data_files={
    "train": "../Data/Dataset/train-00000-of-00001.parquet", 
    "valid": "../Data/Dataset/validation-00000-of-00001.parquet",
    "test": "../Data/Dataset/test-00000-of-00001.parquet"
})


def pre_processing(example):
    input = "Answer the question: {} Context: {}".format(example['question'],example['context'])
    return {"input_text": input, "target_text":example["answer"]}

dataset = dataset.map(pre_processing)


def creat_token(example):
    model_input = tokenizer(example["input_text"], padding= "max_length", max_length= 384, truncation= True)
    labels = tokenizer(example["target_text"], padding= "max_length", max_length= 128, truncation= True)
    model_input["labels"] = labels["input_ids"]
    return model_input

tokenized_dataset = dataset.map(creat_token, batched=True)

arguments = Seq2SeqTrainingArguments(
    output_dir= "./vnt5-qa-checkpoint",
    eval_strategy = "steps", 
    save_strategy= "steps",
    save_total_limit = 1,
    save_steps =  2000,
    eval_steps =  2000,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps =  2,
    eval_accumulation_steps =  2,
    load_best_model_at_end=  True,
    num_train_epochs = 4,
    learning_rate=2e-5,
    logging_dir="./logs",
    predict_with_generate=True,
    load_best_model_at_end=True,
)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


trainer = Seq2SeqTrainer(model, data_collator = data_collator,TrainingArguments = arguments , train_dataset = tokenized_dataset["train"], eval_dataset = tokenized_dataset["valid"])

trainer.train()

trainer.save_model("./vit5-qa-final")
tokenizer.save_pretrained("./vit5-qa-final")
    
    