
import torch
import argparse
from transformers import BertForSequenceClassification, BertTokenizer
from utils import get_device, load_data, create_data_loader

def main(model_path, data_path):
    # Load Model and Tokenizer
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)

    # Device
    device = get_device()
    model.to(device)

    # Load Dataset
    texts, labels = load_data(data_path)
    data_loader = create_data_loader(texts, labels, tokenizer, max_length=128, batch_size=16)

    # Evaluation Function
    def evaluate(model, data_loader, device):
        model.eval()
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

        accuracy = total_correct / total_samples
        return accuracy

    # Evaluate
    accuracy = evaluate(model, data_loader, device)
    print(f'Accuracy: {accuracy}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the fine-tuned model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV file containing reviews and labels')
    args = parser.parse_args()
    main(args.model_path, args.data_path)
