import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.append("../scripts")
from cnn_embeddings import build_vocab, build_matrix, text_to_ids

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, kernel_sizes, output_size, dropout_prob, weights):
        super(TextCNN, self).__init__()

        # embedding layer, initialized with GloVe weights
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(weights, dtype=torch.float32))

        # parallel conv layers with different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), output_size)

    def forward(self, x):
        # x shape: (batch_size, max_length)

        x = self.embedding(x)
        # shape: (batch_size, max_length, embedding_dim)

        x = x.permute(0, 2, 1)
        # shape: (batch_size, embedding_dim, max_length)
        # Conv1d expects channels second

        conv_outputs = []
        for conv in self.convs:
            c = F.relu(conv(x))
            # shape: (batch_size, num_filters, reduced_length)
            c = F.max_pool1d(c, c.size(2)).squeeze(2)
            # shape: (batch_size, num_filters)
            conv_outputs.append(c)

        x = torch.cat(conv_outputs, dim=1)
        # shape: (batch_size, num_filters * len(kernel_sizes))

        x = self.dropout(x)
        x = self.fc(x)
        # shape: (batch_size, output_size)

        return x

vocab = build_vocab("../eo_data/eo_labeled_split/train")
embedding_matrix_glove = build_matrix(vocab, "../../wiki_giga_2024_100_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05.050_combined.txt", embedding_dim=100)

model = TextCNN(
    vocab_size=len(vocab),
    embedding_dim=100,
    num_filters=100,
    kernel_sizes=[3, 4, 5],
    output_size=2,
    dropout_prob=0.5,
    weights=embedding_matrix_glove
)

def load_data(data_dir, vocab):
    ids_list = []
    labels = []
    for txt_file in Path(data_dir).rglob("*.txt"):
        text = txt_file.read_text(encoding="utf-8")
        ids = text_to_ids(text, vocab)
        label = 0 if "democrat" in str(txt_file).lower() else 1
        ids_list.append(ids)
        labels.append(label)
    return torch.tensor(ids_list, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

train_ids, train_labels = load_data("../eo_data/eo_labeled_split/train", vocab)
test_ids, test_labels = load_data("../eo_data/eo_labeled_split/test", vocab)

train_dataset = TensorDataset(train_ids, train_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch_ids, batch_labels in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_ids)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

model.eval()
with torch.no_grad():
    outputs = model(test_ids)
    predictions = torch.argmax(outputs, dim=1).numpy()
    true_labels = test_labels.numpy()

accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)

print(f"\nResults:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1:        {f1:.4f}")