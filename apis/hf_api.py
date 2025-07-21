import torch
import numpy as np
from tqdm import tqdm
import logging
from .api import API
import transformers
import random
from .utils import set_seed, get_subcategories, DISCHARGE_LETTER_STYLES, DISCHARGE_REWRITE_PROMPTS
from .prompts import INSTRUCTION_TEMPLATES, SPECIALTIES, DOC_TYPES, STYLES, LABELS
import re
import collections
from collections import Counter

import os
import ssl
import certifi
import pandas as pd

from transformers import LlamaTokenizerFast

class HFAPI(API):
    def __init__(self,
                 model_type, variation_type, use_subcategory,
                 output_dir, seed, mlm_probability,
                 length, temperature, top_k, top_p, repetition_penalty, do_sample, fp16, no_cuda,
                 random_sampling_batch_size, num_beams, dry_run,
                 variation_batch_size, 
                 percentage_of_summaries, summaries_path,
                 length_mean, length_std, length_min, length_max,
                 local_model_path=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_type = model_type
        self.variation_type = variation_type
        self.output_dir = output_dir
        self.length = length
        self.temperature = temperature
        self.k = top_k
        self.p = top_p
        self.repetition_penalty = repetition_penalty
        self.num_beams = num_beams
        self.do_sample = do_sample
        self.fp16 = fp16
        self.no_cuda = no_cuda
        self.seed = seed
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu")
        self.n_gpu = 0 if self.no_cuda else torch.cuda.device_count()
        set_seed(seed=seed, n_gpu=self.n_gpu)
        self.dry_run = dry_run

        self.use_subcategory = use_subcategory
        if use_subcategory:
            self.subcategory_dict = {}
            self.subcategory_dict['yelp'] = get_subcategories("yelp")
            self.subcategory_dict['pubmed'] = get_subcategories("pubmed")
            self.subcategory_dict['openreview'] = get_subcategories("openreview")

        self.percentage_of_summaries = percentage_of_summaries
        self.summaries_path = summaries_path
        self.length_mean = length_mean
        self.length_std = length_std
        self.length_min = length_min
        self.length_max = length_max

        print("local_model_path!!!!!!", local_model_path)
        print(f"{local_model_path}/tokenizer.json")
  
        # self.tokenizer = transformers.AutoTokenizer.from_pretrained(
        #     local_model_path if local_model_path else model_type, 
        #     device_map="auto",
        #     local_files_only=True,
        #     use_fast=True)

        # self.tokenizer = transformers.AutoTokenizer.from_pretrained(
        #     local_model_path if local_model_path else model_type, 
        #     use_fast=True,  # Explicitly enforce fast tokenizer
        #     local_files_only=True,
        #     tokenizer_type="llama",)

        # self.tokenizer = LlamaTokenizerFast(
        #     vocab_file=f"{local_model_path}/tokenizer.json",
        #     tokenizer_file=f"{local_model_path}/tokenizer.model",
        # )   
    
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.tokenizer.padding_side = "left"


        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            local_model_path,
            use_fast=True,
            tokenizer_type="llama",
            legacy=False,  # Add this line to avoid the warning
            local_files_only=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"



        if local_model_path or "gpt2" not in self.model_type:
            # use torch.float16 for large LLMs
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                local_model_path if local_model_path else model_type, 
                device_map="auto", 
                torch_dtype=torch.float16,
                local_files_only=True)
        else:
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_type, 
                device_map="auto", 
                pad_token_id=pad_token_id,
                local_files_only=True)
            if self.fp16:
                self.model.half()

        # Test the model with a sample generation
        test_input = self.tokenizer("Write a short paragraph about dogs", return_tensors="pt").to(self.device)
        with torch.no_grad():
            test_output = self.model.generate(
                input_ids=test_input.input_ids,
                max_new_tokens=50,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                do_sample=True
            )
        test_text = self.tokenizer.decode(test_output[0], skip_special_tokens=True)
        print('EXAMPLE about dogs:', test_text)

        self.random_sampling_batch_size = random_sampling_batch_size
        self.variation_batch_size = variation_batch_size

    @staticmethod
    def command_line_parser():
        parser = super(HFAPI, HFAPI).command_line_parser()
        parser.add_argument(
            '--model_type',
            type=str,
            default='gpt2',
            help='Which image feature extractor to use')
        parser.add_argument("--use_subcategory",
                            action="store_true", help="use subcategory")
        parser.add_argument(
            '--variation_type',
            type=str,
            default='rephrase',
            choices=["yelp_rephrase_tone", "openreview_rephrase_tone", "pubmed_rephrase_tone", "mimic_rephrase_tone", "french_rephrase_tone"
                     ],
            help='Which image feature extractor to use')
        parser.add_argument("--mlm_probability", type=float, default=0.5)

        parser.add_argument(
            "--output_dir",
            default=None,
            type=str,
        )
        parser.add_argument(
            "--local_model_path",
            default=None,
            type=str,
        )
        parser.add_argument("--length", type=int, default=128)
        parser.add_argument("--temperature", type=float, default=1.0,)
        parser.add_argument("--repetition_penalty", type=float, default=1.0,
                            help="primarily useful for CTRL model; in that case, use 1.2")
        parser.add_argument("--top_k", type=int, default=50)
        parser.add_argument("--top_p", type=float, default=0.9)
        parser.add_argument("--num_beams", type=int, default=5)
        parser.add_argument("--do_sample", action="store_true",
                            help="sampling when generation")
        parser.add_argument("--seed", type=int, default=42,
                            help="random seed for initialization")
        parser.add_argument("--dry_run", action="store_true", help="dry run")
        parser.add_argument(
            '--random_sampling_batch_size',
            type=int,
            default=64,
            help='The batch size for random sampling API')
        parser.add_argument(
            '--variation_batch_size',
            type=int,
            default=256,
            help='The batch size for variation API')

        parser.add_argument(
            "--fp16",
            action="store_true",
            help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
        )
        parser.add_argument("--no_cuda", action="store_true",
                            help="Avoid using CUDA when available")
        
        parser.add_argument("--percentage_of_summaries", type=float, default=0.7)
        parser.add_argument("--summaries_path", type=str, default="data/french/summarized_texts.csv")
        # parser.add_argument("--diversity_number", type=int, default=0)
        parser.add_argument("--length_mean", type=int, default=0)
        parser.add_argument("--length_std", type=int, default=0)
        parser.add_argument("--length_min", type=int, default=0)
        parser.add_argument("--length_max", type=int, default=0)

        return parser
    
    def generate_prompts(self, num_prompts: int = 15):
        prompts = []
        
        try:
            df = pd.read_csv(self.summaries_path)
            summaries = df["summary"].dropna().tolist()
        except Exception as e:
            print(f"Warning: Could not load summaries from {self.summaries_path} — {e}")
            summaries = []

        print("Percentage of summaries:", self.percentage_of_summaries)
        for _ in range(num_prompts):
            doc_type = random.choice(DOC_TYPES)
            specialty = random.choice(SPECIALTIES)
            style = random.choice(STYLES)
            labels_str = LABELS  

            use_summary = random.random() < self.percentage_of_summaries

            if np.random.random() < 0.25:
                target_len = np.random.randint(self.length_min, self.length_max + 1)
            else:
                target_len = max(1, round(np.random.normal(self.length_mean, self.length_std)))
            target_len = min(target_len, self.length_max)

            if use_summary and len(summaries) > 0:
                summary = random.choice(summaries)
                template = random.choice(INSTRUCTION_TEMPLATES_WITH_SUMMARIES)
                prompt = template(doc_type, specialty, style, labels_str, target_len, summary)
            else:
                template = random.choice(INSTRUCTION_TEMPLATES)
                prompt = template(doc_type, specialty, style, labels_str, target_len)
            
            prompts.append(prompt.strip())
        
        return prompts

    def text_random_sampling(self, num_samples, prompt_counter=None, lens_dict=None):
        ratio_generation_training = num_samples / sum(prompt_counter.values())

        all_sequences = []
        additional_info = []
        sync_labels_counter = collections.Counter()
        all_prefix_prompts = []

        self.model.eval()

        # First pass: generate all possible sequences (may be more than num_samples)
        for prompt in tqdm(prompt_counter):
            num_seq_to_generate = round(prompt_counter[prompt] * ratio_generation_training)
            
            if self.use_subcategory:
                if "yelp" in self.variation_type:
                    category_label = prompt.split("\t")[0].replace('Business Category: ', '')
                    rand_keyword_idx = random.randrange(len(self.subcategory_dict['yelp'][category_label]))
                    keyword = self.subcategory_dict['yelp'][category_label][rand_keyword_idx]
                    full_prompt_text = f'{prompt} with keyword {keyword}'

                elif "openreview" in self.variation_type:
                    rand_keyword_idx = random.randrange(len(self.subcategory_dict['openreview']))
                    keyword = self.subcategory_dict['openreview'][rand_keyword_idx]
                    full_prompt_text = f"Suppose that you are a {keyword}. Write a paper review based on " + prompt

                elif "pubmed" in self.variation_type:
                    full_prompt_text = "Using a variety of sentence structures, write an abstract for a medical research paper: "
                    
                elif "mimic" in self.variation_type:
                    num_prompts_to_generate = num_seq_to_generate
                    generated_prompts = self.generate_prompts(num_prompts=num_prompts_to_generate)
                    
                    sequences_per_prompt = max(1, num_seq_to_generate // num_prompts_to_generate)
                    remaining_sequences = num_seq_to_generate % num_prompts_to_generate
                    
                    for i, sample_prompt in enumerate(generated_prompts):
                        full_prompt_text = sample_prompt
                        current_sequences = sequences_per_prompt + (1 if i < remaining_sequences else 0)
                        
                        prompt_input_ids = self.tokenizer(full_prompt_text, return_tensors="pt").input_ids.to(self.device)
                        before_gen_length = len(full_prompt_text)

                        if current_sequences > 0:
                            sequences = self._generate_text(
                                prompt_input_ids, 
                                current_sequences,
                                max_length=self.length, 
                                batch_size=self.random_sampling_batch_size,
                                before_gen_length=before_gen_length
                            )
                            all_sequences.extend(sequences)
                            all_prefix_prompts.extend([full_prompt_text] * len(sequences))
                            additional_info.extend([prompt] * len(sequences))
                            sync_labels_counter[prompt] += len(sequences)
                    continue

                elif "french" in self.variation_type:
                    print()
                    print("_________________________THIS IS FRENCH_________________________")
                    print()
                    num_prompts_to_generate = num_seq_to_generate // 10
                    print("num_prompts_to_generate", num_prompts_to_generate)
                    generated_prompts = self.generate_prompts(num_prompts=num_prompts_to_generate)
                    
                    sequences_per_prompt = max(1, num_seq_to_generate // num_prompts_to_generate)
                    remaining_sequences = num_seq_to_generate % num_prompts_to_generate
                    
                    for i, sample_prompt in enumerate(generated_prompts):
                        print("Iiiii", i)
                        full_prompt_text = sample_prompt
                        current_sequences = sequences_per_prompt + (1 if i < remaining_sequences else 0)
                        
                        prompt_input_ids = self.tokenizer(full_prompt_text, return_tensors="pt").input_ids.to(self.device)
                        before_gen_length = len(full_prompt_text)

                        if current_sequences > 0:
                            sequences = self._generate_text(
                                prompt_input_ids, 
                                current_sequences,
                                max_length=self.length, 
                                batch_size=self.random_sampling_batch_size,
                                before_gen_length=before_gen_length
                            )
                            print("sequences")
                            print(len(sequences))
                            print(sequences)
                            print()
                            all_sequences.extend(sequences)
                            all_prefix_prompts.extend([full_prompt_text] * len(sequences))
                            additional_info.extend([prompt] * len(sequences))
                            sync_labels_counter[prompt] += len(sequences)
                    continue
            else:
                full_prompt_text = prompt

            prompt_input_ids = self.tokenizer(full_prompt_text, return_tensors="pt").input_ids.to(self.device)
            before_gen_length = len(full_prompt_text)

            if num_seq_to_generate > 0:
                sequences = self._generate_text(
                    prompt_input_ids, 
                    num_seq_to_generate,
                    max_length=self.length, 
                    batch_size=self.random_sampling_batch_size,
                    before_gen_length=before_gen_length
                )
                all_sequences.extend(sequences)
                all_prefix_prompts.extend([full_prompt_text] * len(sequences))
                additional_info.extend([prompt] * len(sequences))
                sync_labels_counter[prompt] += len(sequences)

        # Now select random samples from all generated sequences
        if len(all_sequences) > num_samples:
            indices = random.sample(range(len(all_sequences)), num_samples)
            all_sequences = [all_sequences[i] for i in indices]
            additional_info = [additional_info[i] for i in indices]
            all_prefix_prompts = [all_prefix_prompts[i] for i in indices]
            
            # Recalculate sync_labels_counter
            sync_labels_counter = collections.Counter()
            for info in additional_info:
                sync_labels_counter[info] += 1

        logging.info(f"Final selected sequences: %d", len(all_sequences))
        torch.cuda.empty_cache()
        return all_sequences, additional_info, sync_labels_counter, all_prefix_prompts
    
    def _generate_text(self, prompt_input_ids, seq_num, max_length, batch_size, before_gen_length, 
                    avg_chunk_length=100, min_chunk_length=50, max_chunk_length=200):
        """
        Generate text and split into smart chunks based on:
        - Average desired chunk length (in tokens)
        - Minimum chunk length (avoid too small chunks)
        - Maximum chunk length (avoid too large chunks)
        - Sentence boundaries (., !, ?)
        - Paragraph boundaries (double newlines)
        """
        all_data = []
        
        if seq_num < batch_size:
            batch_size = seq_num + 1

        num_return_sequences = 2 if batch_size > 1 else 1
        
        for i in tqdm(range(seq_num // batch_size + 1)):
            if self.dry_run:
                # For dry run, create fake chunks
                generated_chunks = ["s" * avg_chunk_length] * batch_size * 3
                all_data.extend(generated_chunks)
                continue
                
            input_ids = prompt_input_ids.repeat(batch_size, 1).to(self.device)
            
            with torch.no_grad():
                # Generate longer text to allow for chunking
                target_len = min(
                    max(avg_chunk_length * 5, self.length_max * 3),  # Generate enough text
                    self.length_max * 5  # But not too much
                )
                
                output_sequences = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=target_len,
                    temperature=self.temperature,
                    top_k=self.k,
                    top_p=self.p,
                    early_stopping=True,
                    repetition_penalty=self.repetition_penalty,
                    do_sample=self.do_sample,
                    num_return_sequences=num_return_sequences,
                    no_repeat_ngram_size=2,
                )
                generated_texts = self.tokenizer.batch_decode(
                    output_sequences[:, prompt_input_ids.shape[1]:], 
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )

                print("TEXT GENERATED")
            
            # Process each generated text into chunks
            for text in generated_texts:
                chunks = self._split_into_chunks(
                    text,
                    avg_chunk_length=avg_chunk_length,
                    min_chunk_length=min_chunk_length,
                    max_chunk_length=max_chunk_length
                )
                all_data.extend(chunks)

        # If we have more chunks than requested, select randomly
        if len(all_data) > seq_num:
            print("_____________seq_num_________________")
            print(seq_num)
            print("len(all_data)", len(all_data))
            all_data = random.sample(all_data, seq_num)
        
        return all_data

    def _split_into_chunks(self, text, avg_chunk_length, min_chunk_length, max_chunk_length):
        """Smart chunking algorithm that respects sentence and paragraph boundaries"""
        chunks = []
        current_chunk = []
        current_length = 0
        
        # First split by paragraphs (double newlines)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        for paragraph in paragraphs:
            # Tokenize paragraph into sentences (naive approach - can be improved with NLP lib)
            sentences = []
            buffer = []
            
            for char in paragraph:
                buffer.append(char)
                if char in {'.', '!', '?'}:
                    sentence = ''.join(buffer).strip()
                    if sentence:
                        sentences.append(sentence)
                    buffer = []
            
            # Add any remaining text as a sentence
            if buffer:
                sentence = ''.join(buffer).strip()
                if sentence:
                    sentences.append(sentence)
            
            # If we didn't find good sentence boundaries, split by length
            if not sentences:
                words = paragraph.split()
                sentences = [' '.join(words[i:i+avg_chunk_length//2]) 
                            for i in range(0, len(words), avg_chunk_length//2)]
            
            # Build chunks respecting length constraints
            for sentence in sentences:
                sentence_length = len(self.tokenizer.tokenize(sentence))
                
                # If current chunk + sentence would be too big, finalize current chunk
                if current_length + sentence_length > max_chunk_length and current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_length += sentence_length
                
                # If chunk is big enough, finalize it
                if current_length >= avg_chunk_length:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0
            
            # Add any remaining sentences as a chunk
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
        
        # Filter chunks that are too small by combining them
        final_chunks = []
        buffer = []
        buffer_length = 0
        
        for chunk in chunks:
            chunk_length = len(self.tokenizer.tokenize(chunk))
            
            if buffer_length + chunk_length < min_chunk_length:
                buffer.append(chunk)
                buffer_length += chunk_length
            else:
                if buffer:
                    final_chunks.append(' '.join(buffer))
                buffer = [chunk]
                buffer_length = chunk_length
        
        # Add any remaining chunks
        if buffer:
            final_chunks.append(' '.join(buffer))
        
        # Ensure each chunk is within length limits
        final_chunks = [
            chunk for chunk in final_chunks 
            if min_chunk_length <= len(self.tokenizer.tokenize(chunk)) <= max_chunk_length
        ]
        
        return final_chunks

    def text_variation(self, sequences, additional_info,
                       num_variations_per_sequence, variation_degree, epoch_rate=1):
        print("EPOCH RATE", epoch_rate)
        self.model.eval()
        variations = []
        
        for idx in tqdm(range(num_variations_per_sequence)):
            sub_variations, var_labels = self._text_variation(
                sequences=sequences,
                labels=list(additional_info),
                variation_degree=variation_degree,
                variation_type=self.variation_type,
                batch_size=self.variation_batch_size)

            variations.append(sub_variations)
    
        variations = np.stack(variations, axis=1)
        return variations, var_labels, [], [], []
        
    def _text_variation(self, sequences, labels, variation_degree, variation_type, batch_size):
        if self.dry_run:
            all_data = [seq+"s"*self.length for seq in sequences]
            all_labels = [lab for lab in labels]
            return all_data, all_labels

        num_seq = len(sequences)
        all_data = []
        all_labels = []

        self.model.eval()
        self.mlm_probability = variation_degree

        for i in tqdm(range(num_seq // batch_size + 1)):
            start_idx = i*batch_size
            if start_idx >= num_seq:
                break
            end_idx = num_seq if (i+1)*batch_size > num_seq else (i+1)*batch_size

            batch_prompt = []
            batch_labels = []
            for idx in range(start_idx, end_idx):
                prompt = self._rephrase(
                    labels[idx], sequences[idx], variation_type)
                batch_prompt.append(prompt)
                batch_labels.append(labels[idx])

            with torch.no_grad():
                input_ids = self.tokenizer(batch_prompt, padding=True, return_tensors='pt')[
                    'input_ids'].to(self.device)
                
                # Calculate target length
                if np.random.random() < 0.25:
                    target_len = np.random.randint(self.length_min, self.length_max + 1)
                else:
                    target_len = max(1, round(np.random.normal(self.length_mean, self.length_std)))
                target_len = min(target_len, self.length_max)
                
                beam_output = self.model.generate(
                    input_ids,
                    # max_new_tokens=target_len,
                    temperature=self.temperature,
                    top_k=self.k,
                    top_p=self.p,
                    early_stopping=True,
                    repetition_penalty=self.repetition_penalty,
                    do_sample=self.do_sample,
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,
                )
                
                generated_sequences = self.tokenizer.batch_decode(
                    beam_output[:, input_ids.shape[1]:], 
                    skip_special_tokens=True,  
                    clean_up_tokenization_spaces=True
                )

                print("TEXT VARIATION GENERATED")
                
            for idx in range(len(generated_sequences)):
                seq = generated_sequences[idx]
                seq = " ".join(seq.split())
                lab = batch_labels[idx].strip().split("\t")
                if seq:
                    all_data.append(seq)
                else:
                    all_data.append(batch_prompt[idx])
                all_labels.append(lab)

        logging.info(f" _text_variation output lens {len(all_data)}")

        return all_data, all_labels
    
    def _rephrase(self, label, sequence, variation_type):
        if variation_type == "yelp_rephrase_tone":
            selected_style = ALL_styles[random.randrange(len(ALL_styles))]
            prompt = "Based on {}, please rephrase the following sentences {}:\n{} \n".format(
                label, selected_style, sequence)
        elif variation_type == "openreview_rephrase_tone":
            selected_style = ALL_OPENREVIEW_styles[random.randrange(
                len(ALL_OPENREVIEW_styles))]
            prompt = "Based on {}, please rephrase the following sentences {} as a paper review:\n{} \n".format(
                label, selected_style, sequence)
        elif variation_type == "pubmed_rephrase_tone":
            selected_style = ALL_PUBMED_styles[random.randrange(
                len(ALL_PUBMED_styles))]
            prompt = "Please rephrase the following sentences {} as an abstract for medical research paper:\n{} \n".format(
                selected_style, sequence)
            
        elif variation_type == "mimic_rephrase_tone":
            selected_style = DISCHARGE_LETTER_STYLES[random.randrange(
                len(DISCHARGE_LETTER_STYLES))]
            rewrite_template = DISCHARGE_REWRITE_PROMPTS[random.randrange(
                len(DISCHARGE_REWRITE_PROMPTS))]
            
            prompt = rewrite_template.format(style=selected_style, text=sequence)

        elif variation_type == "french_rephrase_tone":
            selected_style = DISCHARGE_LETTER_STYLES[random.randrange(
                len(DISCHARGE_LETTER_STYLES))]
            rewrite_template = DISCHARGE_REWRITE_PROMPTS[random.randrange(
                len(DISCHARGE_REWRITE_PROMPTS))]
            
            prompt = rewrite_template.format(style=selected_style, text=sequence)

        return prompt