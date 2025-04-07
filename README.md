# PyTorch SD-APRR Sarcasm Detection

## 🧾 description

This is a framework for sarcasm detection. The concept of **SD-APRR** was introduced in [a paper in 2023 ACL](https://aclanthology.org/2023.acl-long.566/). This project is a sarcasm detection framework implemented using PyTorch with reference to the idea of the original paper, with some modifications in the model architecture for BiLSTM and MLP. Due to differences in training methods and datasets, the performance of this project differs from the original paper's implementation (the code was not disclosed in the original paper)

---

## 🚀 environment

It runs well on my device with the following environment:

```
Python==3.11
PyTorch==2.4.1
CUDA==12.4
GPU==RTX2050
```

---

## ⚙️ usage

### 1. dataset

In `./data/Ghosh`, you can see the example dataset

```
data/
├── Ghosh/
│   ├── train.txt
│   └── test.txt
│   └── README.md
└── Other/
│   ├── train.txt
│   └── test.txt
└── ...
```

You should run `./datasets.py` to serialize the raw data after data enhancement and word vectorization and save it in `./torch.save/train_data.pt`

### 2. train

After preprocessing the data, you can run `./train.py` to train the model. This file saves the model in `$MODEL_PATH$` after it has been initialized and trained. **Note that at each training session, the pre-trained model will not be loaded, it will only be initialized!**

| How to Train | Like This! |
|------|------|
| `criterion` | CombinedLoss |
| `optimizer` | Adam |
| `scheduler` | None |
| `early_stopping` | True |

### 3. evaluate

You can run `./eval.py` to evaluate the model from `$MODEL_PATH$`. The evaluation results including Accuracy and F1-Score will be printed on the console. **Before evaluation, you should change the args of `get_data` in `./datasets.py` to preprocess the test data!**

### 4. interaction

Run `./work.py` to interact with your trained model. This will start a process in the terminal to communicate with your trained model. You can input a sentence and get the model's response.

```example
Model loaded successfully. Type your text (enter '\q' to quit).
Enter text (enter '\q' to quit): What a fine day!
>>> Event Augmented Text: [What a fine day! then may lead to that X gets healthy and X is happy]
>>> Nodes: ['What', 'a', 'fine', 'day', '!', 'then', 'may', 'lead', 'to', 'that', 'X', 'gets', 'healthy', 'and', 'X', 'is', 'happy']
>>> Output: tensor([[0.0170]], device='cuda:0')
>>> Prediction: 0 (No Sarcasm)
-----------------------------
```

---

## 📁 structure

```
.
├── data/           # original datasets
├── glove.6B/       # word vector tool
├── models/         # define model structure
├── utils/          # instrumented function
├── torch.save/     # model weights and processed dataset saved via pytorch
├── datasets.py
└── eval.py
└── train.py
└── work.py
```

**See more details in `./models/README.md`!**

---

## 📚 reference

- [Just Like a Human Would, Direct Access to Sarcasm Augmented with Potential Result and Reaction (Min et al., ACL 2023)](https://aclanthology.org/2023.acl-long.566/)
- [StanfordNLP GloVe](https://github.com/stanfordnlp/GloVe)
- [COMET-ATOMIC-En-Zh](https://github.com/svjack/COMET-ATOMIC-En-Zh)

---
