import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class AMRDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer_name="Rostlab/prot_bert", max_length=256):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)
        self.max_length = max_length
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.sequences[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0)
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item