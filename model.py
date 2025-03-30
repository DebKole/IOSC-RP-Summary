import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType


# Load dataset
dataset = load_dataset("csv", data_files="extracted_data.csv")

# Rename columns for compatibility
dataset = dataset.rename_column("XML_Text", "document")
dataset = dataset.rename_column("Summary_Text", "summary")

# Split dataset into training and validation
dataset = dataset["train"].train_test_split(test_size=0.1)

# Load BART tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

# Tokenization function
def preprocess_data(example):
    inputs = tokenizer(
        example["document"],
        padding="max_length",
        truncation=True,
        max_length=1024
    )
    targets = tokenizer(
        example["summary"],
        padding="max_length",
        truncation=True,
        max_length=256
    )
    inputs["labels"] = targets["input_ids"]
    return inputs

# Apply tokenization
tokenized_dataset = dataset.map(preprocess_data, batched=True)

# Load BART model
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

# Define LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,  # Seq2Seq task
    r=16,  # LoRA rank (reduce size of adapter)
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.05,  # Dropout rate
    target_modules=["q_proj", "v_proj"]  # Apply LoRA to attention layers
)

# Apply LoRA
model = get_peft_model(model, lora_config)

# Print trainable parameters
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="./bart-lora-summarization",
    evaluation_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=8,  # Increase batch size since LoRA is memory efficient
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    fp16=True,  # Enable for GPU acceleration
    logging_dir="./logs",
    logging_steps=500
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer
)

trainer.train()
model.save_pretrained("fine-tuned-bart-lora")
tokenizer.save_pretrained("fine-tuned-bart-lora")
