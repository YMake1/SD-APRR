import torch
import spacy

from models.model import CombinedModel
from models.BiLSTM import GloVe
from utils.parser import clean_text, text2graph
from utils.COMET import get_event_augmented_samples
from datasets import MAX_TOKENS, DIM
from train import DEVICE, MODEL_PATH

if __name__ == "__main__":
    model = CombinedModel(in_dim=DIM, if_init=False).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    nlp = spacy.load("en_core_web_sm")
    glove = GloVe(dim=DIM, nlp=nlp)

    print("Model loaded successfully. Type your text (enter '\q' to quit).")
    while True:
        text = input("Enter text (enter '\q' to quit): ").strip()
        if text == "\q":
            print("Exiting...")
            torch.cuda.empty_cache()
            break

        text = clean_text(text)
        text = get_event_augmented_samples(text)
        print(f">>> Event Augmented Text: [{text}]")
        nodes, adj_matrix = text2graph(text, nlp)
        print(f">>> Nodes: {nodes}")
        if len(nodes) > MAX_TOKENS:
            print(f"Text too long. Please enter a shorter text.(Max tokens: {MAX_TOKENS})")
            continue
        word_vectors = glove.encode_text(text)

        word_vectors = torch.tensor(word_vectors, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(word_vectors, adj_matrix)
            print(f">>> Output: {outputs}")
            outputs = outputs.item()

        prediction = 1 if outputs >= 0.5 else 0
        print(f">>> Prediction: {prediction} ({'Sarcasm' if prediction == 1 else 'No Sarcasm'})")
        print("-----------------------------")
