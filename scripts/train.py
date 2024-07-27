
import os
import torch
import argparse
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, get_scheduler
from utils import get_device, load_data, create_data_loader

def main(data_path):
    # Parameters
    model_name = 'bert-base-uncased'
    max_length = 128
    batch_size = 16
    epochs = 3
    learning_rate = 2e-5

    # Load Dataset
    texts, labels = load_data(data_path)

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Create DataLoader
    train_loader = create_data_loader(texts, labels, tokenizer, max_length, batch_size)

    # Model
    device = get_device()
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    # Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = len(train_loader) * epochs
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # Training Function
    def train_epoch(model, data_loader, optimizer, device, scheduler):
        model.train()
        total_loss = 0

        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(data_loader)
        return avg_loss

    # Training Loop
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, lr_scheduler)
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss}')

    # Save Model
    model_dir = './models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV file containing reviews and labels')
    args = parser.parse_args()
    main(args.data_path)
