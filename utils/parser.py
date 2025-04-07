import spacy
import re
import numpy as np

def clean_text(text: str, to_rm=[]):
    '''
    Keeps the English symbols, punctuation, and numbers of `text` and removes all matches according to the string given in `to_rm`
    '''
    text = re.sub(r"[^A-Za-z0-9\s.,!?;:'\"()#-]", "", text)
    for item in to_rm:
        text = text.replace(item, "")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+\.", ".", text)
    return text.strip()

def text2graph(text: str, nlp=spacy.load("en_core_web_sm")):
    '''
    After dividing `text` into nodes, compute the adjacency matrix of its dependency graph\n
    @Returns:
        `nodes`(list): list of nodes\n
        `adj_matrix(np.ndarray)`: undirected adjacency matrix with self loop
    '''
    doc = nlp(text)
    nodes = [token.text for token in doc]
    num_nodes = len(nodes)
    adj_matrix = np.eye(num_nodes, dtype=np.int32)
    node_map = {word: i for i, word in enumerate(nodes)}
    for token in doc:
        if token.head.text != token.text:
            head_idx = node_map[token.head.text]
            child_idx = node_map[token.text]
            adj_matrix[head_idx][child_idx] = 1
            adj_matrix[child_idx][head_idx] = 1
    return nodes, adj_matrix

if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    text = "Hello #hashtag 123! This is a test 😊. Remove 你 and some words. #hashtag"
    text = clean_text(text, ["#hashtag"])
    print("Cleaned Text:", text)
    nodes, adj_matrix = text2graph(text, nlp)
    print("Nodes:", nodes)
    print("Undirected Adjacency Matrix:")
    print(adj_matrix)
