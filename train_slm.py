import json
import torch
import os
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

# --- 1. DATA REPAIR & GENERATION ---
raw_data = [
    {
        "instruction": "What are the eligibility criteria for a personal loan?",
        "input": "",
        "output": "To be eligible for a personal loan, you must be between 21-60 years old, have a minimum monthly income of $1,500, and a credit score above 700."
    },
    {
        "instruction": "How do I block my debit card immediately?",
        "input": "",
        "output": "You can block your debit card immediately via our Mobile App under 'Card Settings' > 'Block Card', or by calling our 24/7 hotline at 1-800-BANK-HELP."
    },
    {
        "instruction": "What is the current interest rate for a home loan?",
        "input": "",
        "output": "Our current floating interest rate for home loans starts at 8.40% p.a. for salaried employees and 8.60% p.a. for self-employed individuals."
    },
    {
        "instruction": "I want to close my savings account.",
        "input": "",
        "output": "To close your savings account, please visit your nearest branch with your ID proof and unused chequebook. Account closure charges of $15 apply if closed within 6 months."
    },
    {
        "instruction": "What documents are required for a car loan?",
        "input": "",
        "output": "You need: 1. ID Proof (Passport/License), 2. Address Proof, 3. Last 3 months' salary slips, and 4. Bank statements for the last 6 months."
    },
    {
        "instruction": "How do I activate internet banking?",
        "input": "",
        "output": "Go to our website, click 'New User Registration', enter your Account Number and CIF ID. An OTP will be sent to your registered mobile to set your password."
    },
    {
        "instruction": "What is the daily withdrawal limit for an ATM card?",
        "input": "",
        "output": "For a Classic Debit Card, the daily withdrawal limit is $500. For a Platinum/Gold Card, the limit is $1,000 per day."
    },
    {
        "instruction": "How can I update my phone number?",
        "input": "",
        "output": "Phone number updates require a visit to the branch for security reasons. Please carry your ID proof. It takes 24 hours to update."
    },
    {
        "instruction": "Check my loan application status.",
        "input": "",
        "output": "You can check your status at www.bank.com/status by entering your Application Reference Number (ARN) sent to your email."
    },
    {
        "instruction": "What is the penalty for bouncing a cheque?",
        "input": "",
        "output": "The penalty for an inward cheque return due to insufficient funds is $25 per instance."
    }
]

# Create dataset in memory
dataset = Dataset.from_list(raw_data)

# --- 2. MODEL & TOKENIZER SETUP ---
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"Loading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token # TinyLlama needs this
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto", 
    torch_dtype=torch.float16
)

# Apply LoRA (Efficient Fine-Tuning)
peft_config = LoraConfig(
    r=8, 
    lora_alpha=16, 
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# --- 3. MANUAL DATA PROCESSING (The Stability Fix) ---
# We manually format and tokenize to avoid library version conflicts
def process_data(examples):
    texts = []
    for i in range(len(examples['instruction'])):
        # Construct the prompt manually
        prompt = (
            f"<|system|>You are a helpful banking assistant.\n"
            f"<|user|>{examples['instruction'][i]} {examples['input'][i]}\n"
            f"<|assistant|>{examples['output'][i]}"
            f"{tokenizer.eos_token}" # Manually add EOS
        )
        texts.append(prompt)
    
    # Tokenize
    tokenized = tokenizer(
        texts, 
        padding="max_length", 
        truncation=True, 
        max_length=512
    )
    
    # For Causal LM, labels are the same as inputs (the model predicts the next token)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

print("Processing and Tokenizing dataset...")
tokenized_dataset = dataset.map(process_data, batched=True)

# --- 4. TRAINING ARGUMENTS ---
training_args = TrainingArguments(
    output_dir="./bfsi_checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=1,
    save_strategy="no",
    optim="paged_adamw_8bit"
)

# --- 5. START TRAINING (Using Standard Trainer) ---
# We use DataCollatorForLanguageModeling to handle batching correctly
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

print("Initializing Standard Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=collator
)

print("🚀 Starting Fine-Tuning...")
trainer.train()

# --- 6. SAVE MODEL ---
print("Saving model...")
model.save_pretrained("./fine_tuned_slm")
print("🎉 Success! Model saved to folder: ./fine_tuned_slm")