import numpy as np
import torch
from matplotlib import pyplot as plt
from BiLSTM_CRF_v2 import BiLSTM_CRF_v2
from seqeval.metrics import f1_score
from seqeval.metrics import accuracy_score
from seqeval.metrics import recall_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing
# prepare train data
print("Preprocessing...")
with open("data/train_corpus.txt", "r", encoding="utf-8") as f:
    sentences = [line.replace("\n", "").strip().split() for line in f.readlines()]
with open("data/train_label.txt", "r", encoding="utf-8") as f:
    labels = [line.replace("\n", "").strip().split() for line in f.readlines()]

tmp_data_list = list(zip(sentences, labels))
# fix seed, otherwise the vocab size will be different every time cause error when load trained model's parameters
np.random.seed(0)
np.random.shuffle(tmp_data_list)
sentences, labels = zip(*tmp_data_list)
sentences = sentences[:15000]
labels = labels[:15000]

# build vocabulary and tag mapping
word_to_idx = {"<PAD>": 0}
for sentence in sentences:
    for word in sentence:
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)

# B-PER: Person name start; I-PER: Person name inside; B-LOC: Location start, I-LOC: Location inside;
# B-ORG: Organization start; I-ORG: Organization inside; O: Other; Start tag; Stop tag
label_to_idx = {"B-PER": 0, "I-PER": 1, "B-LOC": 2, "I-LOC": 3, "B-ORG": 4, "I-ORG": 5, "O": 6, "<START>": 7,
                "<STOP>": 8}
idx_to_label = {v: k for k, v in label_to_idx.items()}


# Transfer sentences and labels to tensors
def prepare_sequence(seq, object_to_idx):
    return torch.tensor([object_to_idx[w] for w in seq], dtype=torch.long).to(device)


# Hyper parameters
embedding_dim = 200
hidden_dim = 300
tagset_size = len(label_to_idx)
vocab_size = len(word_to_idx)
print(
    f"Embedding dimension: {embedding_dim} | Hidden dimension: {hidden_dim}  | Tagset size: {tagset_size}  | Vocab size: {vocab_size}")

# Model
print("Building model...")
model = BiLSTM_CRF_v2(vocab_size=vocab_size,
                      tagset_size=tagset_size,
                      embedding_dim=embedding_dim,
                      hidden_dim=hidden_dim).to(device)
print(f"Model device:{device}")
# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-4)


# Data padding
def pad_sequences(sequences, padding_value=0):
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = [seq.tolist() + [padding_value] * (max_len - len(seq)) for seq in sequences]
    return torch.tensor(padded_sequences, dtype=torch.long).to(device)


batch_size = 250

train_flag = False
if train_flag:
    # Train
    print("\nTraining...")
    model.train()
    epochs = 240
    print(f"Training epochs: {epochs}   batch size: {batch_size}")
    lossv = []
    num_batches = (len(sentences) + batch_size - 1) // batch_size  # num of batches
    for epoch in range(epochs):
        loss_accum = 0.
        # shuffle data in each epoch
        tmp_list = list(zip(sentences, labels))
        np.random.seed(epoch)
        np.random.shuffle(tmp_list)
        sentences, labels = zip(*tmp_list)
        for batch in range(num_batches):
            print(f"\rBatch: |", "▇" * int((batch + 1) * (20 / num_batches)), " " * int((num_batches - batch - 2) * (20 / num_batches)), f"|   {batch + 1}/{num_batches}", end="")
            # zero gradients
            model.zero_grad()

            # prepare data
            start_idx = batch * batch_size
            end_idx = min(start_idx + batch_size, len(sentences))
            batch_sentences = sentences[start_idx:end_idx]
            batch_labels = labels[start_idx:end_idx]
            lengths = [len(sentence) for sentence in batch_sentences]

            # padding
            batch_sentence_indices = pad_sequences([prepare_sequence(sentence, word_to_idx) for sentence in batch_sentences]).to(device)
            batch_label_indices = pad_sequences([prepare_sequence(label, label_to_idx) for label in batch_labels]).to(device)

            # forward pass
            total_loss = model(batch_sentence_indices, lengths=lengths, tags=batch_label_indices)

            # backward pass
            total_loss.backward()
            optimizer.step()

            loss_accum += total_loss.item()

        lossv.append(loss_accum / num_batches)
        print(f"  Epoch: |", "▇" * int((epoch + 1) * (20 / epochs)), " " * int((epochs - epoch - 2) * (20 / epochs)),
              f"|{epoch + 1}/{epochs}",
              f" Loss: {lossv[epoch]}")

    print("\nTraining Finished")

    plt.plot(np.arange(len(lossv)), np.array(lossv))
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('BiLSTM+CRF')
    plt.savefig("loss_curve_v2.png")

    # save model
    torch.save(model.state_dict(), "model/BiLSTM_CRF_v2.pth")
    print("Model Saved to model/BiLSTM_CRF_v2.pth")
else:
    print("Loading model...")
    model.load_state_dict(torch.load("model/BiLSTM_CRF_v2.pth"))

# Test
# prepare test data
print("\nTesting...")
model.eval()
with open("data/test_corpus.txt", "r", encoding="utf-8") as f:
    test_sentences = [line.replace("\n", "").strip().split() for line in f.readlines()]
with open("data/test_label.txt", "r", encoding="utf-8") as f:
    test_labels = [line.replace("\n", "").strip().split() for line in f.readlines()]

# build vocabulary and tag mapping
test_word_to_idx = {"<PAD>": 0}
for sentence in test_sentences:
    for word in sentence:
        if word not in test_word_to_idx:
            test_word_to_idx[word] = len(test_word_to_idx)

# evaluation: accuracy, recall, f1
y_pred = []
y_true = []
print("Predicting...")
num_batches = (len(test_sentences) + batch_size - 1) // batch_size
for batch in range(num_batches):
    print(f"\rBatch: |", "▇" * int((batch + 1) * (20 / num_batches)), " " * int((num_batches - batch - 2) * (20 / num_batches)), f"|{batch + 1}/{num_batches}", end="")
    start_idx = batch * batch_size
    end_idx = min(start_idx + batch_size, len(test_sentences))
    batch_sentences = test_sentences[start_idx:end_idx]
    batch_labels = test_labels[start_idx:end_idx]
    lengths = [len(sentence) for sentence in batch_sentences]
    batch_sentence_indices = pad_sequences([prepare_sequence(sentence, test_word_to_idx) for sentence in batch_sentences]).to(device)
    pred_idx = model.predict(batch_sentence_indices, lengths)
    for sentence in pred_idx:
        y_pred.append([idx_to_label[idx] for idx in sentence])
    for labels in batch_labels:
        y_true.append([label for label in labels])
    print(f"\r|", "▇" * int((batch + 1) * (20 / num_batches)), " " * int((num_batches - batch - 2) * (20 / num_batches)), f"|{batch + 1}/{num_batches}", end="")
accuracy = accuracy_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
with open("out/eval_v2.txt", "w", encoding="utf-8") as f:
    f.write(f"accuary: {accuracy}\n")
    f.write(f"recall: {recall}\n")
    f.write(f"f1: {f1}\n")
print(f"\naccuracy: {accuracy}")
print(f"recall: {recall}")
print(f"f1: {f1}")

