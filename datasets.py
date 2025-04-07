import os
import spacy
import torch
from tqdm import tqdm
from datetime import datetime

from utils.parser import clean_text, text2graph
from utils.COMET import get_event_augmented_samples
from models.BiLSTM import GloVe

DATA_ROOT = "./data"
DATA_NAME = "Ghosh"
TRAIN_FILE = os.path.join(DATA_ROOT, DATA_NAME, "train.txt")
TEST_FILE = os.path.join(DATA_ROOT, DATA_NAME, "test.txt")
MAX_TOKENS = 50
DIM = 300

class DataProcess():
    '''
    Reading data from txt and processing\n
    @Args:
        `nlp`: spacy model e.g. `spacy.load("en_core_web_sm")`
        `word2vec`: word2vec model model e.g. `GloVe(tokens, dim, nlp)`
    '''
    def __init__(self, nlp, word2vec):
        self.nlp = nlp
        self.word2vec = word2vec

    def load_data_from_txt(self, file_path, start_line=1, end_line=3742):
        '''
        Read the data from `start_line` to `end_line` from txt and include these two lines\n
        @Returns: 
            `samples`: list of tuples (label, text)
        '''
        samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                if i < start_line:
                    continue
                if i > end_line:
                    break
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                label = int(parts[1])
                text = parts[2]
                samples.append((label, text))
        print(f"Loaded {len(samples)} samples from {file_path}.")
        return samples

    def preprocess_data(self, samples: list):
        '''
        Clean the text in `samples`, enhance it, compute the adjacency matrix, compute the word vectors, and remove too long text\n
        @param:
            `samples`: list of tuples (label, text)
        @Returns:
            `processed_data`: list of tuples (label, adj_matrix, word_vectors)
        '''
        processed_data = []
        too_long_sum = 0
        for label, text in tqdm(samples, desc="Processing Samples", unit="sample"):
            cleaned_text = clean_text(text, ["#sarcasm"])
            augmented_text = get_event_augmented_samples(cleaned_text)
            nodes, adj_matrix = text2graph(augmented_text, self.nlp)
            if len(nodes) <= MAX_TOKENS:
                word_vectors = self.word2vec.encode_text(augmented_text)
                processed_data.append((label, adj_matrix, word_vectors))
            else:
                too_long_sum += 1
        print(f"There are {too_long_sum} sentences longer than MAX_TOKENS.")
        print(f"Processed {len(processed_data)} samples.")
        return processed_data

    def pad_matrix(self, matrix, max_size=MAX_TOKENS, pad_value=0):
        '''
        Pad the `matrix` to the size of `max_size` with `pad_value` and the original matrix is placed in the upper left corner\n
        e.g. `[[1, 2], [3, 4]] -> [[1, 2, 0], [3, 4, 0], [0, 0, 0]]`
        '''
        padded = torch.full((max_size, max_size), pad_value, dtype=matrix.dtype)
        orig_size = matrix.shape[0]
        padded[:orig_size, :orig_size] = matrix
        return padded

    def pad_word_vectors(self, vectors, max_size=MAX_TOKENS, pad_value=0):
        '''
        PAD rows of 2D vector `vectors` with `pad_value` to `max_size`\n
        e.g. `[[1, 2, 3], [4, 5, 6]] -> [[1, 2, 3], [4, 5, 6], [0, 0, 0]]`
        '''
        num_tokens, emb_dim = vectors.shape
        padded = torch.full((max_size, emb_dim), pad_value, dtype=vectors.dtype)
        padded[:num_tokens, :] = vectors
        return padded

def get_data(if_save=True, file_path=TRAIN_FILE, save_path="./torch.save/train_data.pt", **kwargs):
    '''
    Get data from txt and save it to pt\n
    @Returns:
        `labels`: tensor of labels
        `adj_matrices`: tensor of adjacency matrices
        `word_vectors`: tensor of word vectors
    '''
    PreData = DataProcess(nlp=spacy.load("en_core_web_sm"), word2vec=GloVe(tokens=6, dim=DIM, nlp=spacy.load("en_core_web_sm")))
    samples = PreData.load_data_from_txt(file_path=file_path, **kwargs)
    processed_data = PreData.preprocess_data(samples)

    labels = torch.tensor([d[0] for d in processed_data], dtype=torch.long)
    adj_matrices = torch.stack([
        PreData.pad_matrix(torch.tensor(d[1], dtype=torch.float32), MAX_TOKENS) for d in processed_data
    ])
    word_vectors = torch.stack([
        PreData.pad_word_vectors(torch.tensor(d[2], dtype=torch.float32), MAX_TOKENS) for d in processed_data
    ])
    torch.cuda.empty_cache()

    if if_save:
        if os.path.exists(save_path) and input(f"File {save_path} already exists, overwrite? (y/n): ").lower() == "y":
            torch.save({"labels": labels, "adj_matrices": adj_matrices, "word_vectors": word_vectors}, save_path)
            print(f"Data saved to {save_path}.")
        else:
            time_str = datetime.now().strftime("%Y%m%d%H%M%S")
            path = f"./torch.save/{time_str}_train_data.pt"
            torch.save({"labels": labels, "adj_matrices": adj_matrices, "word_vectors": word_vectors}, path)
            print(f"Data saved to {path}")
    return labels, adj_matrices, word_vectors

class Dataset(torch.utils.data.Dataset):
    '''
    Load the data from pt\n
    `__getitem__`: return a sample (label, adj_matrix, word_vector)
    '''
    def __init__(self, data_path):
        self.labels, self.adj_matrices, self.word_vectors = self.load_data_from_pt(data_path)

    def load_data_from_pt(self, file_path):
        '''
        Load the data from pt\n
        pt should have the following structure: `{"labels": labels, "adj_matrices": adj_matrices, "word_vectors": word_vectors}`
        '''
        data = torch.load(file_path, weights_only=True)
        return data["labels"], data["adj_matrices"], data["word_vectors"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        adj_matrix = self.adj_matrices[idx]
        word_vector = self.word_vectors[idx]
        return label, adj_matrix, word_vector

if __name__ == "__main__":
    # labels, adj_matrices, word_vectors = get_data(if_save=True, file_path=TRAIN_FILE, save_path="./torch.save/train_data.pt")
    labels, adj_matrices, word_vectors = get_data(if_save=True, file_path=TEST_FILE, save_path="./torch.save/test_data.pt")
    print(labels.shape, adj_matrices.shape, word_vectors.shape)
