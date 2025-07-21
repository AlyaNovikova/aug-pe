import torch
import numpy as np
import random
import time
import functools
import signal

DISCHARGE_LETTER_STYLES = [
    "sur un ton clinique professionnel",
    "en utilisant une terminologie clinique précise et variée",
    "avec des détails médicaux complets mais concis",
    "avec une variation des symptômes et des diagnostics plausibles",
    "comme si écrit par un clinicien différent",
    "sur un ton clinique professionnel",
    "en utilisant une terminologie médicale concise",
    "avec plus de détails cliniques",
    "avec des observations et des détails médicaux précis",
    "y compris le contexte clinique pertinent",
]

DISCHARGE_REWRITE_PROMPTS = [
    """Reformulez en français ce petit extrait de lettre de sortie {style}.
    Vous pouvez reformuler l'historique clinique, modifier les parties narratives.
    Mais gardez la structure des Informations de Santé Protégées (ISP) : toutes les ISP et les données sensibles doivent être entre doubles crochets comme [[ÉTIQUETTE]].
    Texte : \n{text}""",

    """Réécrivez et reformulez en français cet extrait de lettre de sortie {style}.
    Mais gardez la structure des Informations de Santé Protégées (ISP): toutes les ISP et les données sensibles doivent être entre doubles crochets comme [[ÉTIQUETTE]].
    Modifiez le contenu clinique, comme pour un nouveau patient.
    Texte : \n{text}""",

    """Reformulez en français ce petit extrait de note clinique {style}.
    Changez toutes les informations cliniques et la maladie.
    Et gardez la structure des Informations de Santé Protégées (ISP): toutes les ISP et les données sensibles doivent être entre doubles crochets comme [[ÉTIQUETTE]].
    Texte : \n{text}""",

    """Sur la base de cet extrait de note clinique, créez une nouvelle petite note clinique avec des informations différentes {style}.
    Mais gardez la structure des Informations de Santé Protégées (ISP): toutes les ISP et les données sensibles doivent être entre doubles crochets comme [[ÉTIQUETTE]].
    Texte : \n{text}""",

    """Sur la base de cet extrait de lettre de sortie, reformulez-le et créez une nouvelle petite note clinique.
    Mais gardez la structure des Informations de Santé Protégées (ISP): toutes les ISP et les données sensibles doivent être entre doubles crochets comme [[ÉTIQUETTE]]. Texte : \n{text}""",
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
