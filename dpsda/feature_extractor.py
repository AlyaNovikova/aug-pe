

import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

import os
os.environ["OPENBLAS_NUM_THREADS"] = "4" 
os.environ["OMP_NUM_THREADS"] = "1"      
os.environ["MKL_NUM_THREADS"] = "1"

def extract_features(
        data, model,
        batch_size=4,
        model_name="all-mpnet-base-v2", sentence_transformer=True):
    # If available, the model is automatically executed on the GPU. You can specify the device for the model like this:

    print("len(data)", len(data))
    print("batch_size", batch_size)

    if sentence_transformer and model is not None:
        model.eval()

        print("SentenceTransformer max_seq_length", model.max_seq_length)  
        embeddings = []

        for i in tqdm(range(0, len(data), batch_size)):
            batch = data[i:i+batch_size]
            with torch.no_grad():
                batch_emb = model.encode(batch)
                embeddings.append(batch_emb)
            torch.cuda.empty_cache() 

        sentence_embeddings = np.concatenate(embeddings)

        print("Shape length of sentence_embeddings", sentence_embeddings.shape)

        return sentence_embeddings
    
    if sentence_transformer:
        
        if "Qwen" in model_name:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")  
            
            model = SentenceTransformer(
                model_name,
                device=device,  
                # model_kwargs={
                #     "attn_implementation": "flash_attention_2",  
                #     "torch_dtype": torch.float16,  
                # },
                # tokenizer_kwargs={"padding_side": "left"}
            )
            print(model)

        else: 
            model = SentenceTransformer(model_name)  # device='cuda',
        
        model.eval()

        print("SentenceTransformer max_seq_length", model.max_seq_length)  

        # model.eval()

        embeddings = []

        for i in tqdm(range(0, len(data), batch_size)):
            batch = data[i:i+batch_size]
            with torch.no_grad():
                batch_emb = model.encode(batch)
                embeddings.append(batch_emb)
            torch.cuda.empty_cache() 

        sentence_embeddings = np.concatenate(embeddings)
        del model

        print("Shape length of sentence_embeddings", sentence_embeddings.shape)

        return sentence_embeddings
    
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModel.from_pretrained(model_name)
    # model.eval()
    
    # # Move model to GPU if available
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    
    # sentence_embeddings = []
    
    # with torch.no_grad():
    #     for i in tqdm(range(0, len(data), batch_size)):
    #         batch = data[i:i + batch_size]
            
    #         # Tokenize and move to device
    #         inputs = tokenizer(
    #             batch, 
    #             padding=True, 
    #             truncation=True, 
    #             return_tensors="pt", 
    #             max_length=512  # Adjust based on your needs
    #         ).to(device)
            
    #         # Forward pass
    #         outputs = model(**inputs)
            
    #         # Use mean pooling for sentence embeddings
    #         embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            
    #         if len(embeddings) > 0:
    #             sentence_embeddings.append(embeddings)
    
    # sentence_embeddings = np.concatenate(sentence_embeddings)
    # del model, tokenizer  # Free memory
    # return sentence_embeddings



# import torch
# import numpy as np
# from tqdm import tqdm
# from sentence_transformers import SentenceTransformer
# from transformers import AutoTokenizer

# def chunk_text(text, tokenizer, max_tokens=500):
#     tokens = tokenizer.tokenize(text)
#     chunks = []
#     for i in range(0, len(tokens), max_tokens):
#         chunk = tokens[i:i + max_tokens]
#         chunk_text = tokenizer.convert_tokens_to_string(chunk)
#         chunks.append(chunk_text)
#     return chunks

# def preprocess_texts(texts, tokenizer, max_tokens=500):
#     all_chunks = []
#     mapping = []
#     for idx, text in enumerate(texts):
#         chunks = chunk_text(text, tokenizer, max_tokens)
#         all_chunks.extend(chunks)
#         mapping.extend([idx] * len(chunks))  # Track which original text each chunk came from
#     return all_chunks, mapping

# def extract_features(
#         data,
#         batch_size=1000,
#         model_name="all-mpnet-base-v2",
#         sentence_transformer=True,
#         max_tokens=500):

#     if sentence_transformer:
#         model = SentenceTransformer(model_name)
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#         model.eval()

#         chunked_data, mapping = preprocess_texts(data, tokenizer, max_tokens)

#         sentence_embeddings = []
#         with torch.no_grad():
#             for i in tqdm(range(0, len(chunked_data), batch_size)):
#                 batch = chunked_data[i:i + batch_size]
#                 embeddings = model.encode(batch, show_progress_bar=False)
#                 sentence_embeddings.append(embeddings)

#         sentence_embeddings = np.concatenate(sentence_embeddings)

#         # average embeddings by original text
#         output_embeddings = []
#         current_idx = -1
#         current_vectors = []

#         for i, original_idx in enumerate(mapping):
#             if original_idx != current_idx and current_vectors:
#                 output_embeddings.append(np.mean(current_vectors, axis=0))
#                 current_vectors = []
#             current_vectors.append(sentence_embeddings[i])
#             current_idx = original_idx

#         if current_vectors:
#             output_embeddings.append(np.mean(current_vectors, axis=0))

#         del model

#         output_embeddings = np.vstack(output_embeddings)
#         print("Shape of final sentence_embeddings:", output_embeddings.shape)
#         return output_embeddings
