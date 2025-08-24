import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from deepctr_torch.inputs import (DenseFeat, SparseFeat, build_input_features,
                                  create_embedding_matrix, embedding_lookup)
from deepctr_torch.models import basemodel
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

from data_manager import (HoldingsDataset, build_feature_columns,
                          preprocess_data)
from model_components import DeepFM

"""
현재 작업 중
"""

df = pd.read_parquet("data/holdings_data.parquet")

RANDOM_SEED = 42
BATCH_SIZE = 1024
EPOCH = 100
EMBED_DIM = 8
HIDDEN_UNITS = (256, 128, 64)
LR = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sparse_cols = ['CUSIP','CIK']
dense_cols = ['VALUE', 'SSHPRNAMT', 'TOTAL_VALUE']
target_col = "TOP25_FLAG"

X, y, encoders, scaler = preprocess_data(df, sparse_cols, dense_cols, target_col)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = RANDOM_SEED, stratify = y)
train_loader = DataLoader(HoldingsDataset(X_train, y_train), batch_size = BATCH_SIZE, num_workers = 0)
test_loader = DataLoader(HoldingsDataset(X_test, y_test), batch_size = BATCH_SIZE, num_workers = 0)

feature_columns = build_feature_columns(df, sparse_cols, dense_cols)

model = DeepFM(feature_columns, embed_dim=EMBED_DIM, hidden_units=, device=DEVICE).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr = LR)
criterion = nn.BCELoss()

for epoch in range(0, EPOCH):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        preds = model(X_batch).squeeze()
        loss = criterion(preds, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        totla_loss += loss.item() * X_batch.size(0)
    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch + 1} Train Loss : {avg_loss:.4f}")

model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(DEVICE)
        preds = model(X_batch).squeeze()