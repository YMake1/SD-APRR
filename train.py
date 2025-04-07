import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time

from models.model import CombinedModel, CombinedLoss
from datasets import Dataset, DIM

BATCH_SIZE = 32
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.0001
MODEL_PATH = "./torch.save/model.pth"
EARLY_STOPPING_PATIENCE = 5

def early_stopping(patience, validation_loss_history):
    if len(validation_loss_history) > patience:
        if validation_loss_history[-1] >= min(validation_loss_history[:-patience]):
            return True
    return False

if __name__ == '__main__':
    train_dataset = Dataset(data_path="./torch.save/train_data.pt")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    num_zeros = 0
    num_ones = 0
    for labels, _, _ in train_loader:
        num_zeros += (labels == 0).sum().item()
        num_ones += (labels == 1).sum().item()
    print(f"Dataset label statistics: 0s: {num_zeros}, 1s: {num_ones}")

    model = CombinedModel(in_dim=DIM, if_init=True).to(DEVICE)
    print(model)
    criterion = CombinedLoss(l2_weight=0.001)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Using device:", DEVICE)
    best_loss = float('inf')
    patience_counter = 0
    validation_loss_history = []

    for epoch in range(EPOCHS):
        start_time = time.time()
        print(f"Epoch {epoch+1}/{EPOCHS}", f"started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        model.train()
        running_loss = 0.0

        for labels, adj_matrices, word_vectors in train_loader:
            labels = labels.to(DEVICE)
            adj_matrices = adj_matrices.to(DEVICE)
            word_vectors = word_vectors.to(DEVICE)

            optimizer.zero_grad()

            '''BCEWithLogitsLoss requires outputs to have the same shape as labels(float)'''
            '''outputs is torch.Size([BATCH_SIZE, 1]) and labels is torch.Size([32])'''
            outputs = model(word_vectors, adj_matrices)
            outputs = outputs.squeeze(1)
            labels = labels.float()
            loss = criterion(outputs, labels, model)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_train_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_train_loss:.4f}')
        print(f'Training time for epoch {epoch+1}: {time.time() - start_time:.4f} seconds')
        validation_loss_history.append(avg_train_loss)

        if early_stopping(EARLY_STOPPING_PATIENCE, validation_loss_history):
            print("Early stopping triggered")
            break
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Saved model with loss {avg_train_loss:.4f} to {MODEL_PATH}")
        else:
            patience_counter += 1
        print(f"Epoch {epoch+1} completed. Best loss: {best_loss:.4f}, Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
        print("-----------------------------------------------------------")
