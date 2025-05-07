import torch
import numpy as np
import random
import time
import functools
import signal


PROMPTS_templates = {
    "init_yelp":  {
        "sys_prompt": "You are required to write an example of review based on the provided Business Category and Review Stars that fall within the range of 1.0-5.0.",
        "task_desc": "",
    },

    "init_openreview":  {
        "sys_prompt": "Given the area and final decision of a research paper, you are required to provide a **detailed and long** review consisting of the following content: 1. briefly summarizing the paper in 3-5 sentences; 2. listing the strengths and weaknesses of the paper in details; 3. briefly summarizing the review in 3-5 sentences.",
        "task_desc": "",
    },

    "init_pubmed":  {
        "sys_prompt": "Please act as a sentence generator for the medical domain. Generated sentences should mimic the style of PubMed journal articles, using a variety of sentence structures.",
        "task_desc":  "",
    },

    "variant_yelp":  {
        "sys_prompt": "You are a helpful, pattern-following assistant.",
        "task_desc": "",
    },
    "variant_pubmed":  {
        "sys_prompt": "Please act as a sentence generator for the medical domain. Generated sentences should mimic the style of PubMed journal articles, using a variety of sentence structures.",
        "task_desc":  "",
    },
    "variant_openreview":  {
        # Azure default system prompt
        "sys_prompt": "You are an AI assistant that helps people find information.",
        "task_desc": "",
    },

}


MIMIC_MEDICAL_NOTES_INIT_templates = [
    "Please share an example of a patient discharge summary:",
    "Please provide a sample of a medical discharge note:",
    "Please generate a nurse's note for a hospital shift:",
    "Please write an example of a clinical progress note:",
    "Please share an example of a SOAP note from a medical chart:",
    "Please write a sample patient admission note:",
    "Please provide an example of an emergency department discharge summary:",
    "Please write a typical nurse’s end-of-shift report:",
]

PUBMED_INIT_templates = [
    "Please share an abstract for a medical research paper:",
    "Please provide an example of an abstract for a medical research paper:",
    "Please generate an abstract for a medical research paper:",
    "please share an abstract for a medical research paper as an example:",
    "please write a sample abstract for a medical research paper:",
    "please share an example of an abstract for a medical research paper:",
    "please write an abstract for a medical research paper as an example:",
    "please write an abstract for a medical research paper:",
]


ALL_styles = ["in a casual way", "in a creative style",  "in an informal way", "casually", "in a detailed way",
              "in a professional way", "with more details", "with a professional tone", "in a casual style", "in a professional style", "in a short way", "in a concise manner", "concisely", "briefly", "orally",
              "with imagination", "with a tone of earnestness",  "in a grammarly-incorrect way", "with grammatical errors",  "in a non-standard grammar fashion",
              "in an oral way", "in a spoken manner", "articulately",  "by word of mouth",  "in a storytelling tone",
              "in a formal manner", "with an informal tone", "in a laid-back manner"]
ALL_OPENREVIEW_styles = ["in a detailed way",  "in a professional way", "with more details",
                         "with a professional tone",  "in a professional style",   "in a concise manner"]

ALL_PUBMED_styles = ["in a professional way", "in a professional tone",  "in a professional style",   "in a concise manner",
                     "in a creative style", "using imagination", "in a storytelling tone",  "in a formal manner", "using a variety of sentence structures"
                     ]

# ALL_MIMIC_styles = [
#     "in a professional way", "in a professional tone", "in a professional style", 
# #     "in a concise manner",
# #     "in a creative style", "using imagination", "in a storytelling tone", 
#     "in a formal manner", 
# #     "using a variety of sentence structures"
# ]

# ALL_MIMIC_styles = [
#     "in a professional way", "in a professional tone", "in a professional style",
#     "in a professional clinical tone", "using concise medical terminology",
#     "with thorough clinical details", "in a structured but natural clinical narrative",
#     "with precise medical observations", "including relevant clinical context",
#     "with appropriate medical abbreviations", "in a detailed but readable style"
# ]

DISCHARGE_LETTER_STYLES = [
    "in a professional clinical tone",
    "as a senior attending physician would document",
    "using precise clinical terminology",
    "with thorough discharge instructions",
    "with complete but concise medical details",
    "in an academic hospital discharge style",
    "following typical discharge note structure",
    "with variation in symptoms and plausible diagnoses",
    "as if written by a different clinician",
]

DISCHARGE_REWRITE_PROMPTS = [
    "Rephrase this discharge letter {style}. But keep the PHI structure: all PHI and sensitive data should be in double brackets like [[LABEL]]:\n{text}",
    "Rewrite this discharge summary {style}. But keep the PHI structure: all PHI and sensitive data should be in double brackets like [[LABEL]]. Preserve the original structure and section headers verbatim, but modifying the clinical content where appropriate:\n{text}",
    "Rephrase this hospital discharge {style}. But keep the PHI structure: all PHI and sensitive data should be in double brackets like [[LABEL]]. Maintain identical section organization while varying the medical details:\n{text}",
    "Rewrite this discharge document {style}. But keep the PHI structure: all PHI and sensitive data should be in double brackets like [[LABEL]]. You may reword clinical content within sections:\n{text}",
    "Rephrase this discharge note {style}. But keep the PHI structure: all PHI and sensitive data should be in double brackets like [[LABEL]]. And modify the narrative portions:\n{text}"
]


def set_seed(seed, n_gpu=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Timer:
    """Timer context manager"""

    def __enter__(self):
        """Start a new timer as a context manager"""
        self.start = time.time()
        return self

    def __exit__(self, *args):
        """Stop the context manager timer"""
        self.end = time.time()
        self.duration = self.end - self.start

    def __str__(self):
        return f"{self.duration:.1f} seconds"


def timeout(sec):
    """
    timeout decorator
    :param sec: function raise TimeoutError after ? seconds
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):

            def _handle_timeout(signum, frame):
                err_msg = f'Function {func.__name__} timed out after {sec} seconds'
                raise TimeoutError(err_msg)

            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(sec)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapped_func
    return decorator


def get_subcategories(dataset):
    if "yelp" in dataset:
        category_list = {'Restaurants', 'Bars', 'Shopping', 'Event Planning & Services',
                         'Beauty & Spas', 'Arts & Entertainment', 'Hotels & Travel',
                         'Health & Medical', 'Grocery', 'Home & Garden'}

        subcategory_list = {}
        for cate in category_list:
            prefix = cate.lower().split(' ')[0]
            fname = f'data/yelp/subcategories/{prefix}.txt'
            file1 = open(fname, 'r')
            Lines = file1.readlines()
            Lines = [s.replace('\n', '') for s in Lines]
            subcategory_list[cate] = Lines
        # print(subcategory_list)
    elif "pubmed" in dataset:
        fname = f'data/pubmed/writers.txt'
        file1 = open(fname, 'r')
        Lines = file1.readlines()
        Lines = [s.replace('\n', '') for s in Lines]
        subcategory_list = Lines
    elif "openreview" in dataset:
        fname = f'data/openreview/writers.txt'
        file1 = open(fname, 'r')
        Lines = file1.readlines()
        Lines = [s.replace('\n', '').replace(':', " who has") for s in Lines]
        subcategory_list = Lines

    return subcategory_list
