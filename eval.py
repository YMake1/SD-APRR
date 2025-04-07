import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import time

from models.model import CombinedModel
from datasets import Dataset, DIM
from train import BATCH_SIZE, DEVICE, MODEL_PATH

TEST_DATA_PATH = "./torch.save/test_data.pt"

def evaluate(model, dataloader, device):
    all_labels = []
    all_preds = []
    
    model.eval()
    with torch.no_grad():
        for labels, adj_matrices, word_vectors in dataloader:
            labels = labels.to(device).float().view(-1)
            adj_matrices = adj_matrices.to(device)
            word_vectors = word_vectors.to(device)

            outputs = model(word_vectors, adj_matrices)
            outputs = outputs.squeeze(1)

            preds = (outputs >= 0.5).long()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return acc, f1

if __name__ == '__main__':
    test_dataset = Dataset(data_path=TEST_DATA_PATH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    num_zeros = 0
    num_ones = 0
    for labels, _, _ in test_loader:
        num_zeros += (labels == 0).sum().item()
        num_ones += (labels == 1).sum().item()
    print(f"Dataset label statistics: 0s: {num_zeros}, 1s: {num_ones}")

    model = CombinedModel(in_dim=DIM, if_init=False).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    start_time = time.time()
    acc, f1 = evaluate(model, test_loader, DEVICE)
    print(f"Evaluation completed in {time.time() - start_time:.4f} seconds")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test F1-score: {f1:.4f}")
