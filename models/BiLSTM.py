import numpy as np
import spacy
import torch
import torch.nn as nn

class GloVe():
    '''
    `GloVe` is used to encode the text into word vectors\n
    According to the paper, we put the `GloVe` module together with the `Bi-LSTM` module\n
    @Args:
        `tokens`: the number of billion tokens in the GloVe file
        `dim`: the dimension of the word vectors
        `nlp`: the spacy model
    '''
    def __init__(self, tokens=6, dim=300, nlp=spacy.load("en_core_web_sm")):
        self.path = f'./glove.{tokens}B/glove.{tokens}B.{dim}d.txt'
        self.dim = dim
        self.word_vectors = {}
        print(f"Loading GloVe word vectors from {self.path}")
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                values = line.strip().split()
                word = values[0]
                vector = np.array(values[1:], dtype=np.float32)
                self.word_vectors[word] = vector
        print(f"Loaded {len(self.word_vectors)} word vectors")
        self.nlp = nlp

    def encode_text(self, text: str):
        '''
        @Returns:
            `vectors`: the word vectors of the text(np.ndarray)
        '''
        doc = self.nlp(text)
        vectors = []
        for token in doc:
            word = token.text.lower()
            if word in self.word_vectors:
                vectors.append(self.word_vectors[word])
            else:
                vectors.append(np.zeros(self.dim))
        return np.array(vectors)

class BiLSTM(nn.Module):
    def __init__(self, in_dim=300, hidden_dim=512, num_layers=1, num_heads=2, if_init=False):
        super().__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=in_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=1)
        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, in_dim)
        if if_init:
            self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x):
        transformer_out = self.transformer(x)
        transformer_out = transformer_out + x
        lstm_out, _ = self.lstm(transformer_out)
        output = self.fc(lstm_out)
        return output

if __name__ == "__main__":
    glove = GloVe(tokens=6, dim=300)
    sentence = "I am so happy the car broken down"
    vectors = glove.encode_text(sentence)
    print(vectors.shape)

    model = BiLSTM(in_dim=300, if_init=True)
    input = torch.tensor(vectors).unsqueeze(0).repeat(32, 1, 1)
    output = model(input)

    print(input.shape)
    print(output.shape)
