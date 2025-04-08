import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class AMRDataset(Dataset):
    def __init__(self, sequences, super_labels, sub_labels, tokenizer_name="Rostlab/prot_bert", max_length=256):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)
        self.max_length = max_length
        self.sequences = sequences
        self.super_labels = super_labels
        self.sub_labels = sub_labels

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

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "super_label": torch.tensor(self.super_labels[idx], dtype=torch.long),
            "sub_label": torch.tensor(self.sub_labels[idx], dtype=torch.long),
        }