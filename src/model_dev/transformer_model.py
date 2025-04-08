import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class AMRClassifier(nn.Module):

    def __init__(self, pretrained_model_name, num_super_classes, num_sub_classes):

        super(AMRClassifier, self).__init__()

        self.transformer = AutoModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(0.3)

        hidden_size = self.transformer.config.hidden_size

        self.super_classifier = nn.Linear(hidden_size, num_super_classes)
        self.sub_classifier = nn.Linear(hidden_size, num_sub_classes)

    def forward(self, input_ids, attention_mask):

        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)

        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token output

        x = self.dropout(cls_output)
        super_logits = self.super_classifier(x)
        sub_logits = self.sub_classifier(x)
        return super_logits, sub_logits

def main():
    # example usage

    model_name = "Rostlab/prot_bert"
    num_classes = 49

    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
    model = AMRClassifier(model_name, num_classes)

    sequence = "MTEITAAMVKELRESTGAGMMDCKNALSETNGDFDKAXQLLRBZZGLGKAAKKADRLAAEG"
    sequence = " ".join(list(sequence))
    inputs = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True, max_length=512)

    model.eval()
    with torch.no_grad():
        logits = model(inputs["input_ids"], inputs["attention_mask"])
        predictions = torch.argmax(logits, dim=-1)

    print("Logits shape:", logits.shape)
    print("Predictions shape:", predictions.shape)
    print("Predicted class:", predictions.item())
    print("Logits:", logits)

if __name__ == "__main__":
    main()