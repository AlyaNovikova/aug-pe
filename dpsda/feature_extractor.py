import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

# https://github.com/UKPLab/sentence-transformer


def extract_features(
        data,
        batch_size=1000,
        model_name="all-mpnet-base-v2", sentence_transformer=False):
    # If available, the model is automatically executed on the GPU. You can specify the device for the model like this:

    if sentence_transformer:
        model = SentenceTransformer(model_name)  # device='cuda',
        model.eval()

        with torch.no_grad():
            sentence_embeddings = []
            for i in tqdm(range(len(data) // batch_size+1)):
                embeddings = model.encode(
                    data[i * batch_size:(i + 1) * batch_size])
                if len(embeddings) > 0:
                    sentence_embeddings.append(embeddings)
        sentence_embeddings = np.concatenate(sentence_embeddings)
        del model

        print("Shape length of sentence_embeddings", sentence_embeddings.shape)

        return sentence_embeddings
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    sentence_embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(data), batch_size)):
            batch = data[i:i + batch_size]
            
            # Tokenize and move to device
            inputs = tokenizer(
                batch, 
                padding=True, 
                truncation=True, 
                return_tensors="pt", 
                max_length=512  # Adjust based on your needs
            ).to(device)
            
            # Forward pass
            outputs = model(**inputs)
            
            # Use mean pooling for sentence embeddings
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            
            if len(embeddings) > 0:
                sentence_embeddings.append(embeddings)
    
    sentence_embeddings = np.concatenate(sentence_embeddings)
    del model, tokenizer  # Free memory
    return sentence_embeddings