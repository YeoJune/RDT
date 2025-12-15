"""Fine-tune RoBERTa on WikiText-103 for fair baseline comparison"""

import argparse
import yaml
import torch
from pathlib import Path
from transformers import (
    RobertaForMaskedLM,
    RobertaTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset


def load_config(config_path):
    """Load YAML configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def prepare_dataset(tokenizer, max_length=128):
    """Prepare WikiText-103 train dataset"""
    print("Loading WikiText-103 train set...")
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
    
    def tokenize_function(examples):
        # Filter out empty or very short texts
        texts = [text for text in examples['text'] if len(text.strip()) > 50]
        if not texts:
            return {'input_ids': [], 'attention_mask': []}
        
        return tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_special_tokens_mask=True
        )
    
    print("Tokenizing dataset...")
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    # Filter out empty examples
    tokenized = tokenized.filter(lambda x: len(x['input_ids']) > 0)
    
    print(f"Dataset prepared: {len(tokenized)} examples")
    return tokenized


def main():
    parser = argparse.ArgumentParser(description='Fine-tune RoBERTa on WikiText-103')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Extract config sections
    model_config = config['model']
    data_config = config['data']
    training_config = config['training']
    
    # Set seed
    if 'seed' in config:
        torch.manual_seed(config['seed'])
    
    # Load tokenizer and model
    print(f"\nLoading {model_config['name']}...")
    tokenizer = RobertaTokenizer.from_pretrained(model_config['name'])
    model = RobertaForMaskedLM.from_pretrained(model_config['name'])
    
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    
    # Prepare dataset
    dataset = prepare_dataset(tokenizer, data_config['max_length'])
    
    # Data collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=data_config['mlm_probability']
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=training_config['output_dir'],
        overwrite_output_dir=True,
        num_train_epochs=training_config['epochs'],
        per_device_train_batch_size=training_config['batch_size'],
        learning_rate=training_config['learning_rate'],
        warmup_steps=training_config.get('warmup_steps', 500),
        weight_decay=training_config.get('weight_decay', 0.01),
        logging_steps=training_config.get('logging_steps', 100),
        save_steps=training_config.get('save_steps', 1000),
        save_total_limit=training_config.get('save_total_limit', 3),
        seed=config.get('seed', 42),
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=training_config.get('num_workers', 4),
        report_to='none',
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # Print training info
    print("\n" + "="*60)
    print("Training Configuration")
    print("="*60)
    print(f"Model: {model_config['name']}")
    print(f"Dataset: WikiText-103 train")
    print(f"Epochs: {training_config['epochs']}")
    print(f"Batch size: {training_config['batch_size']}")
    print(f"Learning rate: {training_config['learning_rate']}")
    print(f"Max length: {data_config['max_length']}")
    print(f"MLM probability: {data_config['mlm_probability']}")
    print(f"Output: {training_config['output_dir']}")
    print("="*60 + "\n")
    
    # Train
    print("Starting training...\n")
    trainer.train()
    
    # Save final model
    final_output = f"{training_config['output_dir']}/final"
    trainer.save_model(final_output)
    print(f"\n{'='*60}")
    print(f"Training complete! Model saved to: {final_output}")
    print("="*60)


if __name__ == '__main__':
    main()