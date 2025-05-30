import gzip
import json
import random
import math
from collections import defaultdict
import numpy as np
from glob import glob
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# For text embeddings
from transformers import AutoTokenizer, AutoModel

###############################################################################
# 0) Device Setup
###############################################################################
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

###############################################################################
# 0.1) Load JSONL data for user/item texts if available
###############################################################################
def load_jsonl_data(folder_path="./data"):
    """
    Scans all *.jsonl in `folder_path`. For each line:
      - parse 'custom_id' => split into (user_id, item_id)
      - get 'content' from response.body.choices[0].message.content
    Accumulate in dictionaries:
      user_jsonl_texts[user_id] -> list of textual segments
      item_jsonl_texts[item_id] -> list of textual segments
    """
    user_jsonl_texts = defaultdict(list)
    item_jsonl_texts = defaultdict(list)

    jsonl_files = glob(os.path.join(folder_path, "*.jsonl"))
    print(f"Found {len(jsonl_files)} jsonl files in {folder_path}.")

    for jf in jsonl_files:
        print(f"Loading JSONL file: {jf}")
        with open(jf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                custom_id = data.get("custom_id", "")
                # e.g. custom_id = "A24ZFFRQ4MG3XL-B000WEVGHK"
                # The part before "-" is userID, the part after is itemID
                if "-" not in custom_id:
                    continue
                user_id, item_id = custom_id.split("-", 1)

                # Safely extract the content
                # Adjust the structure if your JSON is different
                try:
                    content = data["response"]["body"]["choices"][0]["message"]["content"]
                except KeyError:
                    content = ""
                content = content.strip()

                if content:
                    user_jsonl_texts[user_id].append(content)
                    item_jsonl_texts[item_id].append(content)

    return user_jsonl_texts, item_jsonl_texts

# Load the JSONL-based user/item content
jsonl_user_texts, jsonl_item_texts = load_jsonl_data("./data")

###############################################################################
# Utility: Prune Reviews to the (domain) Average
###############################################################################
def prune_reviews_to_avg(reviews_dict):
    """
    Given a dict {entity_id: [list_of_reviews]},
    1) compute the average number of reviews per entity,
    2) for entities exceeding that average, randomly sample exactly that many reviews.
    """
    num_entities = len(reviews_dict)
    if num_entities == 0:
        return  # nothing to prune

    total_reviews = sum(len(revs) for revs in reviews_dict.values())
    avg_count = math.floor(total_reviews / num_entities)
    # if avg_count < 1, you might want a minimum of 1
    if avg_count < 1:
        avg_count = 1

    for entity_id, text_list in reviews_dict.items():
        if len(text_list) > avg_count:
            reviews_dict[entity_id] = random.sample(text_list, avg_count)

###############################################################################
# 1) Load Data: Users, Ratings, and Items
#    + Prune to average # of reviews per user/item
###############################################################################
def load_amazon_data(file_path, max_lines=None, max_tokens=100):
    user_reviews = defaultdict(list)
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc=f"Loading {file_path}")):
            if max_lines and i >= max_lines:
                break
            data = json.loads(line)
            user_id = data['reviewerID']
            full_text = data.get('reviewText', "")

            tokens = full_text.split()
            truncated_tokens = tokens[:max_tokens]
            truncated_text = " ".join(truncated_tokens)

            user_reviews[user_id].append(truncated_text)

    # Prune user reviews to the average count
    prune_reviews_to_avg(user_reviews)
    return user_reviews

def load_amazon_ratings(file_path, max_lines=None):
    ratings = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc=f"Loading {file_path} [ratings]")):
            if max_lines and i >= max_lines:
                break
            data = json.loads(line)
            user_id = data['reviewerID']
            item_id = data['asin']
            rating = float(data['overall'])
            ratings.append((user_id, item_id, rating))
    return ratings

def load_amazon_item_data(file_path, max_lines=None, max_tokens=100):
    item_reviews = defaultdict(list)
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc=f"Loading {file_path} [item data]")):
            if max_lines and i >= max_lines:
                break
            data = json.loads(line)
            item_id = data['asin']
            full_text = data.get('reviewText', "")

            tokens = full_text.split()
            truncated_tokens = tokens[:max_tokens]
            truncated_text = " ".join(truncated_tokens)

            item_reviews[item_id].append(truncated_text)

    # Prune item reviews to the average count
    prune_reviews_to_avg(item_reviews)
    return item_reviews

# Example file paths
cd_vinyl_path  = "reviews_Movies_and_TV_5.json.gz"
movies_tv_path = "reviews_CDs_and_Vinyl_5.json.gz"

# Load user-based text data (domain data)
cd_vinyl_data  = load_amazon_data(cd_vinyl_path)
movies_tv_data = load_amazon_data(movies_tv_path)

# Load rating data
cd_vinyl_ratings  = load_amazon_ratings(cd_vinyl_path)
movies_tv_ratings = load_amazon_ratings(movies_tv_path)

# Load item-based text data (for embeddings)
cd_item_data      = load_amazon_item_data(cd_vinyl_path)
movies_item_data  = load_amazon_item_data(movies_tv_path)

###############################################################################
# 2) Identify Overlapping Users and Split 80/20 (with optional JSON)
###############################################################################
overlapping_users = list(set(cd_vinyl_data.keys()) & set(movies_tv_data.keys()))

split_file = "train_test.json"
if os.path.isfile(split_file):
    print(f"Found existing split file: {split_file}. Loading train/test users from it.")
    with open(split_file, "r") as f:
        split_dict = json.load(f)
    # Intersect with actual overlapping users
    train_users = list(set(split_dict["train"]) & set(overlapping_users))
    test_users  = list(set(split_dict["test"]) & set(overlapping_users))
else:
    print(f"No split file found. Performing a random 80/20 split and saving to {split_file}.")
    random.shuffle(overlapping_users)
    cutoff = int(0.8 * len(overlapping_users))
    train_users = overlapping_users[:cutoff]
    test_users  = overlapping_users[cutoff:]

    with open(split_file, "w") as f:
        json.dump({"train": train_users, "test": test_users}, f, indent=2)

print("Num total overlapping users:", len(overlapping_users))
print("Num train users:", len(train_users))
print("Num test users :", len(test_users))

###############################################################################
# 2.1) Compute how many train/test users NOT in the JSONL
###############################################################################
train_users_not_in_jsonl = [u for u in train_users if u not in jsonl_user_texts]
test_users_not_in_jsonl  = [u for u in test_users  if u not in jsonl_user_texts]

print(f"[JSONL] {len(train_users_not_in_jsonl)} out of {len(train_users)} train users NOT in JSONL.")
print(f"[JSONL] {len(test_users_not_in_jsonl)} out of {len(test_users)} test users NOT in JSONL.")

###############################################################################
# 3) Text Embeddings with Sentence Transformers (for items)
###############################################################################
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer_bert = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
encoder_model = AutoModel.from_pretrained(EMBED_MODEL_NAME).to(device)
encoder_model.eval()

###############################################################################
# Mini-Batched Embedding Function
###############################################################################
@torch.no_grad()
def get_text_embedding(text_list, tokenizer, model, device='cpu', batch_size=16):
    """
    Computes the average sentence-transformer embedding for all texts in text_list.
    Processes text_list in mini-batches to avoid OOM issues.
    """
    if len(text_list) == 0:
        # Return zero vector if no text
        return torch.zeros(model.config.hidden_size, device=device)

    all_embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch_texts = text_list[i:i+batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=128
        ).to(device)
        # outputs.last_hidden_state is shape [B, seq_len, hidden_dim]
        outputs = model(**encoded)
        # average over the seq_len dimension
        batch_emb = outputs.last_hidden_state.mean(dim=1)
        all_embeddings.append(batch_emb)

    # Concatenate over all mini-batches
    all_embeddings = torch.cat(all_embeddings, dim=0)  # shape [total_texts, hidden_dim]
    # Then average across all the texts for a single entity
    entity_embedding = all_embeddings.mean(dim=0)
    return entity_embedding

def build_entity_embedding_dict(entity_list, entity_to_texts_dict, tokenizer, model, device='cpu'):
    emb_dict = {}
    for e in tqdm(entity_list, desc="Building embeddings"):
        text_list = entity_to_texts_dict[e]
        emb = get_text_embedding(text_list, tokenizer, model, device=device, batch_size=16)
        emb_dict[e] = emb.cpu()
    return emb_dict

###############################################################################
# 3.1) Build unified item-text dictionary
###############################################################################
# Instead of using only Amazon data item reviews, let's unify:
#   If an item is found in JSONL (jsonl_item_texts), use that content
#   Otherwise, fall back to item_reviews from Amazon.
def get_item_text_list(item_id, item_reviews, jsonl_item_texts):
    if item_id in jsonl_item_texts:
        return jsonl_item_texts[item_id]
    else:
        return item_reviews[item_id]

cd_item_ids = list(set(list(cd_item_data.keys()) + list(jsonl_item_texts.keys())))
movies_item_ids = list(set(list(movies_item_data.keys()) + list(jsonl_item_texts.keys())))

cd_item_text_dict = {}
for itm in cd_item_ids:
    cd_item_text_dict[itm] = get_item_text_list(itm, cd_item_data, jsonl_item_texts)

movies_item_text_dict = {}
for itm in movies_item_ids:
    movies_item_text_dict[itm] = get_item_text_list(itm, movies_item_data, jsonl_item_texts)

# Build item embeddings
cd_item_emb_dict = build_entity_embedding_dict(cd_item_ids, cd_item_text_dict, tokenizer_bert, encoder_model, device=device)
mv_item_emb_dict = build_entity_embedding_dict(movies_item_ids, movies_item_text_dict, tokenizer_bert, encoder_model, device=device)

###############################################################################
# 3.2) Compute how many items in target domain ratings NOT in the JSONL
###############################################################################
# We'll figure out which items appear in the rating data:
movies_items_in_ratings = set([i for (_, i, _) in movies_tv_ratings])
items_not_in_jsonl = [i for i in movies_items_in_ratings if i not in jsonl_item_texts]
print(f"[JSONL] {len(items_not_in_jsonl)} out of {len(movies_items_in_ratings)} Movies/Books items NOT in JSONL.")

###############################################################################
# 4) Build User Embeddings in BOTH Domains (Train + Test)
###############################################################################
def get_user_text_list(user_id, user_reviews, jsonl_user_texts):
    if user_id in jsonl_user_texts:
        return jsonl_user_texts[user_id]
    else:
        return user_reviews[user_id]

def build_user_embedding_dict(user_list, domain_data_dict):
    emb_dict = {}
    for user in tqdm(user_list, desc="Building user embeddings"):
        text_list = get_user_text_list(user, domain_data_dict, jsonl_user_texts)
        emb = get_text_embedding(text_list, tokenizer_bert, encoder_model, device=device, batch_size=16)
        emb_dict[user] = emb.cpu()
    return emb_dict

cd_train_user_emb = build_user_embedding_dict(train_users, cd_vinyl_data)
cd_test_user_emb  = build_user_embedding_dict(test_users,  cd_vinyl_data)

mv_train_user_emb = build_user_embedding_dict(train_users, movies_tv_data)
mv_test_user_emb  = build_user_embedding_dict(test_users,  movies_tv_data)

print("Built user embeddings: source=CD, target=Movies.")

###############################################################################
# 5) Define RQVAE (Source + Target)
###############################################################################
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=384, commitment_beta=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_beta = commitment_beta

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, z):
        with torch.no_grad():
            z_expanded = z.unsqueeze(1)  # (B, 1, D)
            e_expanded = self.embedding.weight.unsqueeze(0)  # (1, K, D)
            distances = (z_expanded - e_expanded).pow(2).sum(dim=2)  # (B, K)
            _, nearest_idx = distances.min(dim=1)  # (B,)

        z_q = self.embedding(nearest_idx)  # (B, D)

        # VQ losses
        loss_codebook = F.mse_loss(z_q, z.detach())
        loss_commit   = F.mse_loss(z_q.detach(), z)
        vq_loss = loss_codebook + self.commitment_beta * loss_commit

        # Straight-through estimator
        z_q = z + (z_q - z).detach()
        return z_q, nearest_idx, vq_loss

class ResidualVectorQuantizer(nn.Module):
    def __init__(self, levels=2, num_embeddings=512, embedding_dim=384, beta=0.25):
        super().__init__()
        self.levels = levels
        self.vq_layers = nn.ModuleList([
            VectorQuantizer(num_embeddings, embedding_dim, beta)
            for _ in range(levels)
        ])

    def forward(self, z):
        residual = z
        quant_sum = torch.zeros_like(z)
        all_indices = []
        total_vq_loss = 0.0

        for vq in self.vq_layers:
            c, idx, vq_loss = vq(residual)
            quant_sum += c
            residual = residual - c
            total_vq_loss += vq_loss
            all_indices.append(idx)

        return quant_sum, all_indices, total_vq_loss

class RQVAE(nn.Module):
    """
    Minimal RQ-VAE: MLP encoder -> residual quantizer -> MLP decoder
    """
    def __init__(self, input_dim=384, hidden_dim=256, latent_dim=384,
                 levels=2, num_embeddings=32, beta=0.25):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.rq = ResidualVectorQuantizer(
            levels=levels,
            num_embeddings=num_embeddings,
            embedding_dim=latent_dim,
            beta=beta
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        z_q, indices, vq_loss = self.rq(z)
        x_hat = self.decoder(z_q)
        return x_hat, z_q, indices, vq_loss

def train_rqvae(rqvae, user_emb_dict, num_epochs=10, lr=1e-3, device='cpu'):
    rqvae.to(device)
    optimizer = optim.Adam(rqvae.parameters(), lr=lr)

    user_ids = list(user_emb_dict.keys())
    for epoch in range(num_epochs):
        random.shuffle(user_ids)
        epoch_loss = 0.0
        for user_id in tqdm(user_ids, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            emb = user_emb_dict[user_id].unsqueeze(0).to(device)  # (1, D)
            optimizer.zero_grad()
            x_hat, z_q, indices, vq_loss = rqvae(emb)
            recon_loss = F.mse_loss(x_hat, emb)
            loss = recon_loss + vq_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f" RQVAE Epoch {epoch+1} | Avg Loss: {epoch_loss/len(user_ids):.6f}")
    return rqvae

###############################################################################
# 6) Instantiate and Train Source/Target RQVAEs on the TRAIN set
###############################################################################
num_codewords = 8

source_rqvae = RQVAE(
    input_dim=384, hidden_dim=64, latent_dim=384,
    levels=4, num_embeddings=num_codewords, beta=0.25
)
target_rqvae = RQVAE(
    input_dim=384, hidden_dim=64, latent_dim=384,
    levels=4, num_embeddings=num_codewords, beta=0.25
)

print("\nTraining SOURCE RQVAE (CDs & Vinyl) ...")
source_rqvae = train_rqvae(source_rqvae, cd_train_user_emb, num_epochs=10, lr=1e-3, device=device)

print("\nTraining TARGET RQVAE (Movies & TV) ...")
target_rqvae = train_rqvae(target_rqvae, mv_train_user_emb, num_epochs=10, lr=1e-3, device=device)

###############################################################################
# 7) Gather Source/Target Codewords for TRAIN users
###############################################################################
@torch.no_grad()
def get_codeword_indices(rqvae, emb, device='cpu'):
    z = rqvae.encoder(emb)
    _, all_indices, _ = rqvae.rq(z)
    return [idx.item() for idx in all_indices]

train_src_codewords = []
train_tgt_codewords = []

for user_id in train_users:
    src_emb = cd_train_user_emb[user_id].unsqueeze(0).to(device)
    tgt_emb = mv_train_user_emb[user_id].unsqueeze(0).to(device)

    src_cw = get_codeword_indices(source_rqvae, src_emb, device=device)
    tgt_cw = get_codeword_indices(target_rqvae, tgt_emb, device=device)

    train_src_codewords.append(src_cw)
    train_tgt_codewords.append(tgt_cw)

train_src_codewords = torch.tensor(train_src_codewords, dtype=torch.long)
train_tgt_codewords = torch.tensor(train_tgt_codewords, dtype=torch.long)

###############################################################################
# 8) Train a SMALL MLP to map source codewords -> target codewords
###############################################################################
class CodewordMapperDataset(Dataset):
    def __init__(self, src_cw_tensor, tgt_cw_tensor):
        self.src = src_cw_tensor
        self.tgt = tgt_cw_tensor
        assert len(self.src) == len(self.tgt)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]

train_cw_dataset = CodewordMapperDataset(train_src_codewords, train_tgt_codewords)
train_cw_loader = DataLoader(train_cw_dataset, batch_size=64, shuffle=True)


levels = source_rqvae.rq.levels

class CodewordMapper(nn.Module):
    def __init__(self, num_codewords=64, levels=4, hidden_dim=64):
        super().__init__()
        self.levels = levels
        self.num_codewords = num_codewords
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(num_codewords, hidden_dim)
        self.fc = nn.Linear(levels * hidden_dim, levels * num_codewords)

    def forward(self, src_cw):
        B, L = src_cw.shape
        emb = self.embedding(src_cw)              # (B, L, hidden_dim)
        emb = emb.view(B, L*self.hidden_dim)      # (B, L*hidden_dim)
        out = self.fc(emb)                        # (B, L*num_codewords)
        out = out.view(B, L, self.num_codewords)  # (B, L, num_codewords)
        return out

mapper_model = CodewordMapper(num_codewords=num_codewords, levels=levels, hidden_dim=64).to(device)
optimizer_map = optim.Adam(mapper_model.parameters(), lr=1e-3)

def train_codeword_mapper(model, loader, epochs=5):
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for src_cw, tgt_cw in tqdm(loader, desc=f"CodewordMapper Epoch {epoch+1}/{epochs}"):
            src_cw = src_cw.to(device)
            tgt_cw = tgt_cw.to(device)
            logits = model(src_cw)  # (B, L, num_codewords)
            B, L, K = logits.shape
            loss = criterion(logits.view(B*L, K), tgt_cw.view(B*L))

            optimizer_map.zero_grad()
            loss.backward()
            optimizer_map.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"  [Mapper] Epoch={epoch+1}, CE Loss={avg_loss:.4f}")

train_codeword_mapper(mapper_model, train_cw_loader, epochs=5)

@torch.no_grad()
def map_codewords(model, src_cw):
    model.eval()
    logits = model(src_cw)       # (B, L, K)
    pred = logits.argmax(dim=-1) # (B, L)
    return pred

###############################################################################
# 9) Utility: Convert codewords -> final embedding via target RQVAE
###############################################################################
@torch.no_grad()
def codewords_to_embedding(rqvae, codewords, device='cpu'):
    """
    Decodes a list of codeword indices into the final embedding
    using the RQVAE's quantizers + decoder.
    """
    residual = torch.zeros(1, rqvae.rq.vq_layers[0].embedding_dim, device=device)
    quant_sum = torch.zeros_like(residual)
    for level_idx, idx in enumerate(codewords):
        idx_tensor = torch.tensor([idx], dtype=torch.long, device=device)
        vq_layer = rqvae.rq.vq_layers[level_idx]
        codebook_vector = vq_layer.embedding(idx_tensor)
        quant_sum += codebook_vector
        residual = residual - codebook_vector
    user_emb = rqvae.decoder(quant_sum)
    return user_emb

###############################################################################
# 10) Rating Model with Linear Layer
###############################################################################
# 10.1) Precompute user embeddings for TRAIN (already in mv_train_user_emb)
train_user_emb_dict = {}
for u in train_users:
    train_user_emb_dict[u] = mv_train_user_emb[u].to(device)

# 10.2) Precompute user embeddings for TEST using the Source->Mapper->Target pipeline
test_user_emb_dict = {}
for u in tqdm(test_users, desc="Precomputing test user embeddings"):
    src_emb = cd_test_user_emb[u].unsqueeze(0).to(device)
    src_cw = get_codeword_indices(source_rqvae, src_emb, device=device)
    src_cw_tensor = torch.tensor([src_cw], dtype=torch.long, device=device)

    # Map source codewords -> target codewords
    pred_tgt_cw_tensor = map_codewords(mapper_model, src_cw_tensor)
    pred_tgt_cw = pred_tgt_cw_tensor[0].tolist()

    # Decode
    user_target_emb = codewords_to_embedding(target_rqvae, pred_tgt_cw, device=device)
    test_user_emb_dict[u] = user_target_emb.squeeze(0).to(device)

# Move item embeddings to device
mv_item_embs_on_device = {}
for item_id, emb_cpu in mv_item_emb_dict.items():
    mv_item_embs_on_device[item_id] = emb_cpu.to(device)

# Build train/test rating sets
train_user_set = set(train_users)
test_user_set  = set(test_users)

train_ratings_t = [(u, i, r) for (u, i, r) in movies_tv_ratings if u in train_user_set]
test_ratings_t  = [(u, i, r) for (u, i, r) in movies_tv_ratings if u in test_user_set]

print(f"Target domain ratings (train users): {len(train_ratings_t)}")
print(f"Target domain ratings (test  users): {len(test_ratings_t)}")

class RatingDataset(Dataset):
    def __init__(self, ratings, user_emb_dict, item_emb_dict):
        self.ratings = ratings
        self.user_emb_dict = user_emb_dict
        self.item_emb_dict = item_emb_dict

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        u, i, r = self.ratings[idx]
        user_emb = self.user_emb_dict[u]
        if i in self.item_emb_dict:
            item_emb = self.item_emb_dict[i]
        else:
            emb_dim = next(iter(self.item_emb_dict.values())).shape[0]
            item_emb = torch.zeros(emb_dim, device=device)

        rating = torch.tensor(r, dtype=torch.float, device=device)
        return user_emb, item_emb, rating

train_dataset = RatingDataset(train_ratings_t, train_user_emb_dict, mv_item_embs_on_device)
test_dataset  = RatingDataset(test_ratings_t,  test_user_emb_dict,  mv_item_embs_on_device)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

class RatingRegressor(nn.Module):
    def __init__(self, emb_dim=384):
        super().__init__()
        self.fc = nn.Linear(2 * emb_dim, 1)

    def forward(self, user_emb, item_emb):
        x = torch.cat([user_emb, item_emb], dim=1)
        out = self.fc(x)
        return out.squeeze(-1)

rating_model = RatingRegressor(emb_dim=encoder_model.config.hidden_size).to(device)
optimizer_rating = optim.Adam(rating_model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

###############################################################################
# Evaluate: compute MAE/RMSE on the test set
###############################################################################
def evaluate_mae_rmse(model, loader):
    model.eval()
    all_preds = []
    all_trues = []
    with torch.no_grad():
        for user_emb, item_emb, rating in loader:
            pred = model(user_emb, item_emb)
            all_preds.extend(pred.cpu().tolist())
            all_trues.extend(rating.cpu().tolist())

    preds_tensor = torch.tensor(all_preds)
    trues_tensor = torch.tensor(all_trues)

    mae = nn.L1Loss()(preds_tensor, trues_tensor).item()
    mse = nn.MSELoss()(preds_tensor, trues_tensor).item()
    rmse = mse ** 0.5
    return mae, rmse

###############################################################################
# Train Rating Model, Evaluate Each Epoch, Track Best
###############################################################################
NUM_EPOCHS_RATING = 30
best_rmse = float('inf')
best_mae  = float('inf')
best_epoch = -1
best_state = None

for epoch in range(NUM_EPOCHS_RATING):
    # ---- TRAINING LOOP ----
    rating_model.train()
    total_loss = 0.0
    for user_emb, item_emb, rating in tqdm(train_loader, desc=f"Training RatingModel Epoch {epoch+1}"):
        pred = rating_model(user_emb, item_emb)
        loss = criterion(pred, rating)
        optimizer_rating.zero_grad()
        loss.backward()
        optimizer_rating.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)

    print(f"[Epoch={epoch+1}] Train MSE={avg_loss:.4f}")

    # ---- EVALUATION AFTER THIS EPOCH ----
    mae_test, rmse_test = evaluate_mae_rmse(rating_model, test_loader)
    print(f"       => Test MAE={mae_test:.4f}, Test RMSE={rmse_test:.4f}")

    # Update best if improved
    if rmse_test < best_rmse:
        best_rmse = rmse_test
        best_mae  = mae_test
        best_epoch = epoch

        best_state = {k: v.clone() for k, v in rating_model.state_dict().items()}

print("\nFinished training rating model.")
print(f"Best epoch was {best_epoch+1} with RMSE={best_rmse:.4f}, MAE={best_mae:.4f}.")

# Restore best model
if best_state is not None:
    rating_model.load_state_dict(best_state)

mae_test, rmse_test = evaluate_mae_rmse(rating_model, test_loader)
print(f"[Restored best model] Test MAE={mae_test:.4f}, Test RMSE={rmse_test:.4f}")
