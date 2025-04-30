import torch
import numpy as np
from tqdm import tqdm
import logging
from .api import API
import transformers
import random
from .utils import set_seed, get_subcategories, ALL_styles, ALL_OPENREVIEW_styles, ALL_PUBMED_styles, ALL_MIMIC_styles
import re
import collections

import os
import ssl
import certifi

os.environ["SSL_CERT_FILE"] = certifi.where()

# import ollama
import requests

LABELS = [
    "[[NAME:Medical_personnel]]", "[[NAME:patient]]", "[[NAME:other]]",
    "[[ADDRESS]]", "[[DATE]]",
    "[[CONTACT: Telephone]]", "[[CONTACT: Fax]]", "[[CONTACT: Email]]",
    "[[ID: SocialID]]", "[[ID: MedicalID]]", "[[ID: InsuranceID]]",
    "[[NUMBER: Account]]", "[[NUMBER: License]]", "[[NUMBER: VehicleID]]", "[[NUMBER: DeviceID]]",
    "[[URL]]", "[[IPAdress]]",
    "[[DEMOGRAPHIC: Age]]", "[[DEMOGRAPHIC: CivilStatus]]", "[[DEMOGRAPHIC: Nationality]]", "[[DEMOGRAPHIC: Profession]]",
    "[[HOSPITAL: Service]]", "[[HOSPITAL: Building]]", "[[HOSPITAL: Room-Bed]]",
    "[[PersonalRelation]]"
]

STYLES = [
    "in a professional way", "in a professional tone", "in a professional style", 
#     "in a concise manner",
#     "in a creative style", "using imagination", "in a storytelling tone", 
    "in a formal manner", 
#     "using a variety of sentence structures"
]

DOC_TYPES = [
    "discharge summary", "radiology report",
    # "consultation note", "progress note", "operative report",
    # "emergency room note", "pathology report", "nursing note",
    # "physician's order", "admission note", "clinical discharge note", "outpatient clinic note"
    ]

SPECIALTIES = [
    "cardiology", "neurology", "oncology", "pediatrics", "orthopedics",
    "internal medicine", "general surgery", "psychiatry", "endocrinology",
    "pulmonology", "gastroenterology", "nephrology"
]

INSTRUCTION_TEMPLATES = [
        lambda doc, spec, sty, lbl: f"""Generate a synthetic {doc} for a {spec} case {sty}. 
        Generate it in the exact style of MIMIC dataset.
        The text should be realistic and resemble actual medical documentation.
        Include and Replace all PHI/sensitive data with labels from this list in double brackets: {lbl}. 
        Don't structure it too much. It should be a natural medical live recording.
        
        Make sure the text flows naturally and maintains proper medical terminology.""",
    
    
        lambda doc, spec, sty, lbl: f"""Create a synthetic {doc} for a {spec} case {sty}. Generate it in the exact style of MIMIC dataset.
    The document should be comprehensive.  
    Include and Replace all PHI/sensitive data with labels from this list in double brackets: {lbl}. 
    Don't structure it too much. It should be a natural medical live recording.
    Structure it with these sections:
    1. Patient Identification (include {random.choice(["[[NAME:patient]]", "[[ID: MedicalID]]", "[[DEMOGRAPHIC: Age]]"])})
    2. Chief Complaint
    3. History of Present Illness
    4. Assessment and Plan
    5. Any relevant procedures or treatments
    Maintain medical accuracy while {sty}.""",

        lambda doc, spec, sty, lbl: f"""Write a synthetic {doc} in narrative form {sty} for a {spec} patient. Generate it in the exact style of MIMIC dataset.
    Don't structure it too much. It should be a natural medical live recording.
    Include and Replace all PHI/sensitive data with labels from this list in double brackets: {lbl}. 
    Begin with patient presentation, then describe:
    - The clinical reasoning process
    - Diagnostic findings
    - Therapeutic interventions
    - Follow-up plans
    Weave in {random.sample(lbl, min(2, len(lbl)))} organically throughout the text.""",

        lambda doc, spec, sty, lbl: f"""Generate a comprehensive {doc} {sty} showing multidisciplinary care in {spec}.
    Don't structure it too much. It should be a natural medical live recording.
    Include and Replace all PHI/sensitive data with labels from this list in double brackets: {lbl}. 
    Cover:
    - Primary team assessment
    - Consultant recommendations
    - Nursing observations
    - Therapy inputs
    Integrate {random.choice(["[[NAME:Medical_personnel]]", "[[HOSPITAL: Service]]"])} naturally in the text.""",

        lambda doc, spec, sty, lbl: f"""Write a detailed synthetic {doc} {sty} focusing on a {spec} procedure.
    Include and Replace all PHI/sensitive data with labels from this list in double brackets: {lbl}. 
    Don't structure it too much. It should be a natural medical live recording.
    Include:
    - Pre-procedure preparation
    - Step-by-step technique
    - Instruments/devices used (reference [[NUMBER: DeviceID]] if applicable)
    - Post-procedure care
    Maintain {sty} while being technically precise.""",

]

def generate_prompts(num_prompts: int = 15):
    prompts = []
    for _ in range(num_prompts):
        doc_type = random.choice(DOC_TYPES)
        specialty = random.choice(SPECIALTIES)
        style = random.choice(STYLES)
        selected_labels = random.sample(LABELS, random.randint(3, 6))
        labels_str = ", ".join(selected_labels)
        
        template = random.choice(INSTRUCTION_TEMPLATES)
        prompt = template(doc_type, specialty, style, labels_str)
        
        prompts.append(prompt.strip())
    
    return prompts


class HFAPI(API):
    def __init__(self,
                 model_type, variation_type, use_subcategory,
                 output_dir, seed, mlm_probability,
                 length, temperature, top_k, top_p, repetition_penalty, do_sample, fp16, no_cuda,
                 random_sampling_batch_size, num_beams, dry_run,
                 variation_batch_size,
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
            self.subcategory_dict['openreview'] = get_subcategories(
                "openreview")

        model_name_or_path = self.model_type

        if "deepseek" in model_name_or_path:
            self.use_ollama = True
        else:
            self.use_ollama = False

        if self.use_ollama:
            try:
                response = requests.get('http://localhost:11434/api/tags')
                if response.status_code != 200:
                    raise ConnectionError("Ollama server not responding")
            except Exception as e:
                print(f"Connection error: {e}")
                print("Make sure 'ollama serve' is running in another terminal")
                exit(1)

            self.client = ollama.Client(host='http://localhost:11434')

            response = self.client.generate(
                    model=model_name_or_path,
                    prompt="Write a short paragraph about dogs",
                    options={'temperature': 0, 'num_predict': 10} 
                )

            print('EXAMPLE about dogs:', response['response'])
        else: 
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_name_or_path, device_map="auto")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"

            if "gpt2" not in self.model_type:
                # use torch.float16 for large LLMs
                self.model = transformers.AutoModelForCausalLM.from_pretrained(
                    model_name_or_path, device_map="auto", torch_dtype=torch.float16)
            else:
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id
                self.model = transformers.AutoModelForCausalLM.from_pretrained(
                    model_name_or_path, device_map="auto", pad_token_id=pad_token_id)
                if self.fp16:
                    self.model.half()

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
            choices=["yelp_rephrase_tone", "openreview_rephrase_tone", "pubmed_rephrase_tone", "mimic_rephrase_tone"
                     ],
            help='Which image feature extractor to use')
        parser.add_argument("--mlm_probability", type=float, default=0.5)

        parser.add_argument(
            "--output_dir",
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

        return parser

    def text_random_sampling(self, num_samples, prompt_counter=None, lens_dict=None):
        print("\n\n!!!!!!!!!!!!", "text_random_sampling hfapi", "\n\n!!!!!!!!!!!!")
        ratio_generation_training = num_samples / sum(prompt_counter.values())
        all_sequences = []
        ppls_cur = []
        additional_info = []
        sync_labels_counter = collections.Counter()

        if not self.use_ollama:
            self.model.eval()

        simulate_num = 0
        for prompt in tqdm(prompt_counter):
            # generation is proportional to the label distributions
            simulate_num_seq_to_generate = round(
                prompt_counter[prompt] * ratio_generation_training)
            simulate_num += simulate_num_seq_to_generate

        logging.info(
            f"should -- simulated generated sequences: %d", simulate_num)
        all_prefix_prompts = []
        
        for prompt in tqdm(prompt_counter):
            # generation is proportional to the label distributions
            num_seq_to_generate = round(
                prompt_counter[prompt] * ratio_generation_training)
            
            if self.use_subcategory:
                if "yelp" in self.variation_type:
                    category_label = prompt.split(
                        "\t")[0].replace('Business Category: ', '')
                    rand_keyword_idx = random.randrange(
                        len(self.subcategory_dict['yelp'][category_label]))
                    keyword = self.subcategory_dict['yelp'][category_label][rand_keyword_idx]
                    full_prompt_text = f'{prompt} with keyword {keyword}'

                elif "openreview" in self.variation_type:
                    rand_keyword_idx = random.randrange(
                        len(self.subcategory_dict['openreview']))
                    keyword = self.subcategory_dict['openreview'][rand_keyword_idx]
                    full_prompt_text = f"Suppose that you are a {keyword}. Write a paper review based on " + prompt

                elif "pubmed" in self.variation_type:
                    full_prompt_text = "Using a variety of sentence structures, write an abstract for a medical research paper: "
                    
                elif "mimic" in self.variation_type:
                    # Generate multiple different prompts for MIMIC
                    # num_prompts_to_generate = max(1, min(num_seq_to_generate, 10))
                    num_prompts_to_generate = num_seq_to_generate
                    generated_prompts = generate_prompts(num_prompts_to_generate)
                    
                    sequences_per_prompt = max(1, num_seq_to_generate // num_prompts_to_generate)
                    remaining_sequences = num_seq_to_generate % num_prompts_to_generate
                    
                    for i, sample_prompt in enumerate(generated_prompts):
                        full_prompt_text = sample_prompt
                        current_sequences = sequences_per_prompt + (1 if i < remaining_sequences else 0)
                        
                        print("####")
                        print(full_prompt_text)
                        print(f"Generating {current_sequences} sequences for this prompt")
                        print("####")

                        if not self.use_ollama:
                            prompt_input_ids = self.tokenizer(full_prompt_text)['input_ids']
                            before_gen_length = len(full_prompt_text)
                        else:
                            prompt_input_ids = full_prompt_text
                            before_gen_length = len(full_prompt_text)

                        if current_sequences > 0:
                            if self.use_ollama:
                                sequences = self._generate_text_ollama(full_prompt_text, current_sequences,
                                                        max_length=self.length)
                            else:
                                sequences = self._generate_text(prompt_input_ids, current_sequences,
                                                            max_length=self.length, batch_size=self.random_sampling_batch_size,
                                                            before_gen_length=before_gen_length)
                                
                            all_sequences += sequences
                        all_prefix_prompts += [full_prompt_text] * current_sequences
                        additional_info += [prompt] * current_sequences
                        sync_labels_counter[prompt] += current_sequences
                        
                    continue

            else:
                full_prompt_text = prompt

            if not self.use_ollama:
                prompt_input_ids = self.tokenizer(full_prompt_text)['input_ids']
                before_gen_length = len(full_prompt_text)
            else:
                prompt_input_ids = full_prompt_text
                before_gen_length = len(full_prompt_text)

            if num_seq_to_generate > 0:
                # condition on the prompt
                if self.use_ollama:
                    sequences = self._generate_text_ollama(full_prompt_text, num_seq_to_generate,
                                        max_length=self.length)
                else:
                    sequences = self._generate_text(prompt_input_ids, num_seq_to_generate,
                                                max_length=self.length, batch_size=self.random_sampling_batch_size,
                                                before_gen_length=before_gen_length)
                    
                all_sequences += sequences
            all_prefix_prompts += [full_prompt_text] * num_seq_to_generate
            additional_info += [prompt] * num_seq_to_generate
            sync_labels_counter[prompt] = num_seq_to_generate

        logging.info(f"Total generated sequences: %d", len(all_sequences))
        torch.cuda.empty_cache()
        return all_sequences, additional_info, sync_labels_counter, all_prefix_prompts
    
    def _generate_text_ollama(self, prompt, seq_num, max_length):

        print("GENERATE TEXT OLLAMA", prompt)

        all_data = []

        for _ in range(seq_num):
            if self.dry_run:
                generated_text = "s" * max_length
            else:
                response = self.client.generate(
                    model=self.model_type,
                    prompt=prompt,
                    options={'temperature': self.temperature, 'num_predict': max_length}
                )
                generated_text = response.get("response", "").strip()

            if generated_text:
                seq = " ".join(generated_text.split())
                all_data.append(seq)

        return all_data

    def _generate_text(self, prompt, seq_num, max_length, batch_size, before_gen_length):
        all_data = []

        if seq_num < batch_size:
            batch_size = seq_num + 1  # TODO: improve

        num_return_sequences = 2 if batch_size > 1 else 1
        for i in tqdm(range(seq_num // batch_size + 1)):
            if self.dry_run:
                generated_sequences = ["s" * max_length] * batch_size
            else:
                input_ids = torch.tensor(prompt).repeat(
                    batch_size, 1).to(self.device)
                with torch.no_grad():
                    output_sequences = self.model.generate(
                        input_ids=input_ids,
                        max_new_tokens=max_length,
                        temperature=self.temperature,
                        top_k=self.k,
                        top_p=self.p,
                        early_stopping=True,
                        repetition_penalty=self.repetition_penalty,
                        do_sample=self.do_sample,
                        # overgenerate to ensure we have enough non-empty generated sequences
                        num_return_sequences=num_return_sequences,
                        no_repeat_ngram_size=2,
                    )
                    generated_sequences = self.tokenizer.batch_decode(output_sequences[:, input_ids.shape[1]:], skip_special_tokens=True,
                                                                    clean_up_tokenization_spaces=True)
            for g in generated_sequences:
                seq = g
                seq = " ".join(seq.split())
                if seq:
                    all_data.append(seq)

        if len(all_data) > seq_num:
            all_data = random.sample(all_data, seq_num)
        return all_data

    def text_variation(self, sequences, additional_info,
                       num_variations_per_sequence, variation_degree):
        if not self.use_ollama:
            self.model.eval()
            # self.model.to(self.device)
        variations = []
        for idx in tqdm(range(num_variations_per_sequence)):
            if self.use_ollama:
                sub_variations, var_labels = self._text_variation_ollama(
                    sequences=sequences,
                    labels=list(additional_info),
                    variation_degree=variation_degree,
                    variation_type=self.variation_type,
                    batch_size=self.variation_batch_size)
            else:
                sub_variations, var_labels = self._text_variation(
                    sequences=sequences,
                    labels=list(additional_info),
                    variation_degree=variation_degree,
                    variation_type=self.variation_type,
                    batch_size=self.variation_batch_size)

            variations.append(sub_variations)
        return np.stack(variations, axis=1), var_labels, [], [], []
    
    def _text_variation_ollama(self, sequences, labels, variation_degree, variation_type, batch_size):
        if self.dry_run:
            all_data = [seq + "s" * self.length for seq in sequences]
            all_labels = [lab for lab in labels]
            return all_data, all_labels

        num_seq = len(sequences)
        all_data = []
        all_labels = []

        self.mlm_probability = variation_degree

        for i in tqdm(range(num_seq // batch_size + 1)):
            start_idx = i * batch_size
            if start_idx >= num_seq:
                break
            end_idx = min((i + 1) * batch_size, num_seq)

            for idx in range(start_idx, end_idx):
                # print("TEXT VARIATION OLLAMA SEQUENCES" , sequences[idx])

                prompt = self._rephrase(
                    labels[idx], sequences[idx], variation_type)
                
                # print("TEXT VARIATION OLLAMA PROMPT", prompt)

                if self.dry_run:
                    generated_text = prompt + "s" * self.length
                else:
                    response = self.client.generate(
                        model=self.model_type,
                        prompt=prompt,
                        options={'temperature': self.temperature, 'num_predict': self.length}
                    )
                    generated_text = response.get("response", "").strip()

                seq = " ".join(generated_text.split())
                lab = labels[idx].strip().split("\t")

                if seq:
                    all_data.append(seq)
                else:
                    all_data.append(prompt)

                all_labels.append(lab)

        logging.info(f"_text_variation_ollama output lens: {len(all_data)}")

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
            selected_style = ALL_MIMIC_styles[random.randrange(
                len(ALL_MIMIC_styles))]
            prompt = "Please rephrase the following sentences {}:\n{} \n".format(
                selected_style, sequence)

        return prompt

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
            end_idx = num_seq if (
                i+1)*batch_size > num_seq else (i+1)*batch_size

            batch_prompt = []
            batch_labels = []
            for idx in range(start_idx, end_idx):
                prompt = self._rephrase(
                    labels[idx], sequences[idx], variation_type)
                batch_prompt.append(prompt)
                batch_labels.append(labels[idx])

            with torch.no_grad():
                input_ids = self.tokenizer(batch_prompt, padding=True, return_tensors='pt')[
                    'input_ids'].to(self.device)  # has been padded into the same lens; cannot be used
                beam_output = self.model.generate(input_ids,
                                                  max_new_tokens=self.length,
                                                  temperature=self.temperature,
                                                  top_k=self.k,
                                                  top_p=self.p,
                                                  early_stopping=True,
                                                  repetition_penalty=self.repetition_penalty,
                                                  do_sample=self.do_sample,
                                                  num_return_sequences=1,
                                                  no_repeat_ngram_size=2,
                                                  )
                # TODO:   skip the tokens so the lens of input_ids is diff from batch_prompt
                generated_sequences = self.tokenizer.batch_decode(
                    beam_output[:, input_ids.shape[1]:], skip_special_tokens=True,  clean_up_tokenization_spaces=True)
            for idx in range(len(generated_sequences)):
                seq = generated_sequences[idx]
                seq = " ".join(seq.split())
                lab = batch_labels[idx].strip().split("\t")
                if seq:
                    all_data.append(seq)  # no lables!
                else:
                    all_data.append(batch_prompt[idx])
                all_labels.append(lab)

        logging.info(f" _text_variation output lens  {len(all_data)}")

        return all_data, all_labels
