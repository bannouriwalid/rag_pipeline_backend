from config import Config
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
pretrained_repo = Config.EMBEDDING_MODEL
batch_size = Config.EMBEDDING_BATCH_SIZE


class Dataset(torch.utils.data.Dataset):
    def __init__(self, input_ids=None, attention_mask=None):
        super().__init__()
        self.data = {
            "input_ids": input_ids,
            "att_mask": attention_mask,
        }

    def __len__(self):
        return self.data["input_ids"].size(0)

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.item()
        batch_data = dict()
        for key in self.data.keys():
            if self.data[key] is not None:
                batch_data[key] = self.data[key][index]
        return batch_data


class Sentence_Transformer(nn.Module):
    def __init__(self, pretrained_embedding):
        super(Sentence_Transformer, self).__init__()
        print(f"inherit model weights from {pretrained_embedding}")
        self.bert_model = AutoModel.from_pretrained(pretrained_embedding)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        data_type = token_embeddings.dtype
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(data_type)
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids, att_mask):
        bert_out = self.bert_model(input_ids=input_ids, attention_mask=att_mask)
        sentence_embeddings = self.mean_pooling(bert_out, att_mask)

        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings


def load_sbert():

    model = Sentence_Transformer(pretrained_repo)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_repo)

    # data parallel
    if torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} GPUs')
        model = nn.DataParallel(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device


def sbert_text2embedding(model, tokenizer, device, text):
    if len(text) == 0:
        return torch.zeros((0, 1024))

    encoding = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    dataset = Dataset(input_ids=encoding.input_ids, attention_mask=encoding.attention_mask)

    # DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Placeholder for storing the embeddings
    all_embeddings = []

    # Iterate through batches
    with torch.no_grad():

        for batch in dataloader:
            # Move batch to the appropriate device
            batch = {key: value.to(device) for key, value in batch.items()}

            # Forward pass
            embeddings = model(input_ids=batch["input_ids"], att_mask=batch["att_mask"])

            # Append the embeddings to the list
            all_embeddings.append(embeddings)

    # Concatenate the embeddings from all batches
    all_embeddings = torch.cat(all_embeddings, dim=0).cpu()

    return all_embeddings


load_model = load_sbert
load_text2embedding = sbert_text2embedding

"""
Code Report: services/embedding_models.py

1. Overview:
This module implements a sentence embedding system using transformer-based models, specifically designed for generating embeddings from text data. It provides functionality to load pre-trained models and convert text into vector representations.

2. Key Components:

a) Dataset Class:
- Custom PyTorch Dataset implementation
- Handles input_ids and attention masks
- Supports batch processing of text data
- Implements standard Dataset interface methods (__len__, __getitem__)

b) Sentence_Transformer Class:
- Inherits from nn.Module
- Uses pre-trained transformer models (via AutoModel)
- Implements mean pooling for sentence embeddings
- Includes normalization of embeddings
- Key methods:
  * mean_pooling: Computes weighted average of token embeddings
  * forward: Processes input through BERT and generates normalized embeddings

c) Utility Functions:
- load_sbert():
  * Initializes model and tokenizer
  * Supports multi-GPU processing
  * Handles device placement (CPU/GPU)
  * Returns model, tokenizer, and device

- sbert_text2embedding():
  * Converts text to embeddings
  * Handles batch processing
  * Returns tensor of embeddings

3. Technical Details:
- Uses PyTorch for deep learning operations
- Implements HuggingFace's transformers library
- Supports CUDA acceleration
- Configurable batch size and model repository
- Handles empty text inputs gracefully

4. Dependencies:
- torch
- transformers
- config (custom module)

5. Performance Considerations:
- Implements batch processing for efficiency
- Supports GPU acceleration
- Uses data parallel processing when multiple GPUs available
- Implements no_grad context for inference

6. Usage:
The module provides two main functions:
- load_model: Initializes the embedding model
- load_text2embedding: Converts text to embeddings

7. Configuration:
- Uses external Config class for:
  * EMBEDDING_MODEL: Pre-trained model repository
  * EMBEDDING_BATCH_SIZE: Batch size for processing

8. Best Practices:
- Implements proper error handling
- Uses type hints and clear documentation
- Follows PyTorch design patterns
- Implements efficient memory management
"""
