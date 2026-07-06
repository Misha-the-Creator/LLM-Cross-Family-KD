import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm.notebook import trange, tqdm
from tokenizers import Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from enum import Enum

model_name_1 = "LiquidAI/LFM2.5-230M"
tokenizer_1 = Tokenizer.from_pretrained(model_name_1)
print(f'Длина словаря {model_name_1}: {tokenizer_1.get_vocab_size()}')
print(f'Модель токенизатора {model_name_1}: {tokenizer_1.model}')

model_name_2 = "xlnet/xlnet-base-cased"
tokenizer_2 = Tokenizer.from_pretrained(model_name_2)
print(f'Длина словаря {model_name_2}: {tokenizer_2.get_vocab_size()}')
print(f'Модель токенизатора {model_name_2}: {tokenizer_2.model}')