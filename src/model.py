# src/model.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import yaml

def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

def finetune_model(config, output_dir):
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'], torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['dropout'],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    dataset = load_dataset(config['data']['dataset'], "multiple_choice")[config['data']['split']]
    
    def format_qa(ex):
        return f"Question: {ex['question']}\nAnswer:"
    dataset = dataset.map(lambda x: {"text": format_qa(x)}, batched=False)
    dataset = dataset.remove_columns(['question', 'mc1_targets', 'mc2_targets'])

    def tokenize(ex):
        return tokenizer(ex["text"], truncation=True, max_length=config['data']['max_length'], padding=True)
    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    dataset = dataset.map(lambda x: {"labels": x["input_ids"]}, batched=True)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=f"{output_dir}/lora",
            per_device_train_batch_size=config['training']['batch_size'],
            num_train_epochs=config['training']['epochs'],
            learning_rate=config['training']['lr'],
            fp16=config['training']['fp16'],
            logging_steps=20,
            save_strategy="no",
            report_to=[],
            remove_unused_columns=False
        ),
        train_dataset=dataset.select(range(config['data']['n_samples'])),
        tokenizer=tokenizer
    )
    trainer.train()
    model.save_pretrained(f"{output_dir}/lm_fact")
    return model, tokenizer