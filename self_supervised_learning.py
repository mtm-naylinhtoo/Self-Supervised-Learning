import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset

# Step 1: Load dataset
dataset = load_dataset('ptb_text_only', split='train')

# Step 2: Tokenization function
def tokenize_function(examples):
    encoding = tokenizer(examples['sentence'], padding='max_length', truncation=True)
    encoding["labels"] = encoding["input_ids"].copy()
    return encoding

# Step 3: Initialize tokenizer and model with padding token
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Adding a new pad token
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))  # Resizing token embeddings to include the new pad token

# Step 4: Tokenize dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["sentence"])

# Step 5: Define training arguments with output_dir specified
training_args = TrainingArguments(
    output_dir='./output',  # Specify where checkpoints and logs will be saved
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_dir='./logs',
    overwrite_output_dir=True,
    logging_steps=100,
)

# Step 6: Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Step 7: Train the model
trainer.train()

# Step 8: Generate text
input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

output = model.generate(input_ids, max_length=50, num_return_sequences=1)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)
