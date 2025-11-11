import os
import sys
import random
import torch
import cv2
import pandas as pd

from os import path
from PIL import Image
from typing import List, Union
from itertools import combinations
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# add parent path
cur_path = path.abspath(path.dirname(__file__))
parent_path = path.dirname(path.dirname(cur_path))
sys.path.append(parent_path)

# set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# image normalization
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)

# import tokenizer
from DETRIS.utils.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()


def tokenize(texts: Union[str, List[str]],
             context_length: int = 77,
             truncate: bool = False) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token]
                  for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(
                    f"Input {texts[i]} is too long for context length {context_length}"
                )
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result




class AVADataset_test(Dataset):
    def __init__(self, path_to_csv, root_dir, if_train=True, word_length=77, ablate_text=False):
        self.ablate_text = ablate_text
        # read CSV file
        self.df = pd.read_csv(path_to_csv)
        self.root_dir = root_dir
        self.word_length = word_length

        # Group data by pair for efficient sampling
        self.groups = self.df.groupby("pair")
        # Generate all pairwise combinations for each group of images
        self.pairs = []
        for _, group_data in self.groups:
            group_combinations = list(combinations(group_data.index, 2))  
            self.pairs.extend(group_combinations)


        num_pairs_to_swap = len(self.pairs) // 2
        indices_to_swap = random.sample(range(len(self.pairs)), num_pairs_to_swap)
        for idx in indices_to_swap:
            self.pairs[idx] = (self.pairs[idx][1], self.pairs[idx][0])

        # Define data transformations
        if if_train:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop((224, 224)),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize
            ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        """
        Args:
            index (int): index for selecting a pair of images.

        Returns:
            img1 (Tensor): first image tensor.
            img2 (Tensor): second image tensor.
            text1 (str): first image's description.
            text2 (str): second image's description.
            score1 (float): first image's score.
            score2 (float): second image's score.
            label (int): image pair's label, determined by new_score.
        """
        # get the indices of the image pair
        idx1, idx2 = self.pairs[index]

        # get samples
        sample1 = self.df.loc[idx1]
        sample2 = self.df.loc[idx2]

        # get image paths
        img1_path = os.path.join(self.root_dir, sample1["FileName"], sample1["ImageName"])
        img2_path = os.path.join(self.root_dir, sample2["FileName"], sample2["ImageName"])
      
        img1 = cv2.imread(img1_path) 
        img2 = cv2.imread(img2_path) 

        if img1 is None or img2 is None:
            raise FileNotFoundError(f"Image not found: {img1_path if img1 is None else img2_path}")

        # Convert BGR to RGB format
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # Apply transforms if specified
        img1 = self.transform(Image.fromarray(img1))  # Convert to PIL image and apply transform
        img2 = self.transform(Image.fromarray(img2))  # Convert to PIL image and apply transform


        if self.ablate_text:
            text1 = ""
            text2 = ""
        else:
            text1 = sample1["Description"]
            text2 = sample2["Description"]

        text1_vec = tokenize(text1, self.word_length, True).squeeze(0)
        text2_vec = tokenize(text2, self.word_length, True).squeeze(0)
        score1 = sample1["Majority_Label"]
        score2 = sample2["Majority_Label"]

        # Determine label
        if score1 > score2:
            label = 1
        elif score1 < score2:
            label = -1
        else:
            label = 0

        confidence1 = sample1["Confidence"]
        return img1, img2, text1_vec, text2_vec, score1, score2, label, sample1["ImageName"], sample2["ImageName"], confidence1
