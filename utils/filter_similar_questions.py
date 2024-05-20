from sentence_transformers import SentenceTransformer
import json
import torch
import numpy as np
model = SentenceTransformer('sentence-transformers/gtr-t5-base').to('cuda')
train_questions = [q['question'] for q in json.load(open("../data/fake_knowledge.json", 'r'))]
test_questions = [q['question'] for q in json.load(open("../data/fake_knowledge_test.json", 'r'))]

train_embeddings = model.encode(
    train_questions, convert_to_tensor=True, show_progress_bar=True, batch_size=16
)

test_embeddings = model.encode(
    test_questions, convert_to_tensor=True, show_progress_bar=True, batch_size=16
)

print(f"train_embeddings shape: {train_embeddings.shape}")
print(f"test_embeddings shape: {test_embeddings.shape}")


cosine_similarity = torch.matmul(
    test_embeddings, train_embeddings.T
).cpu().numpy()

self_consine_similarity = torch.matmul(
    test_embeddings, test_embeddings.T
).cpu().numpy()
# Set the diagonal to 0
np.fill_diagonal(self_consine_similarity, 0)


print(f"cosine_similarity shape: {cosine_similarity.shape}")

keep_indices = []

for i in range(len(cosine_similarity)):
    if np.max(cosine_similarity[i]) < 0.8:
        if np.max(self_consine_similarity[i][:(i + 1)]) < 0.8:
            keep_indices.append(i)

print(f"Original number of questions: {len(test_questions)}")
print(f"Number of questions to keep: {len(keep_indices)}")

filtered_questions = [q for q in json.load(open("../data/fake_knowledge_test.json", 'r'))]
filtered_questions = [filtered_questions[i] for i in keep_indices]
with open("../data/fake_knowledge_test_filtered.json", 'w') as f:
    json.dump(filtered_questions, f, indent=4)