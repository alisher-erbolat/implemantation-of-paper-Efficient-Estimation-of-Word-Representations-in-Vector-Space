import torch
import torch.nn as nn

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embeddings = self.embeddings(inputs)
        logits = self.linear(embeddings)
        return logits

# Define the model, optimizer, and loss function
model = SkipGram(vocab_size, embedding_dim)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

# Train the model on some input-output pairs
for inputs, targets in training_data:
    # Forward pass
    logits = model(inputs)
    loss = loss_fn(logits, targets)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Extract the word embeddings
word_embeddings = model.embeddings.weight.detach().numpy()
