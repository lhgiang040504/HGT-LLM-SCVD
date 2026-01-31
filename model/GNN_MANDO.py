import networkx as nx
import sys
import numpy as np
import torch
import random
from collections import Counter
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from HGT_metapath2v_bilstm import HGTVulNodeClassifier
from torch import nn
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
import os
import time
import logging
import gc
torch.backends.cudnn.enabled = False
logging.basicConfig(level=logging.INFO)

# === CONFIG ===
NUM_CLASSES = 9
EPOCHS = 400
PATIENCE = 20
GRAPH_PATH = './merged_graph_dappscan_sbcurated.gpickle'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE, flush=True)

nx_graph = nx.read_gpickle(GRAPH_PATH)
model_tmp = HGTVulNodeClassifier(GRAPH_PATH, node_feature='metapath2vec', device='cpu')
targets = torch.tensor(model_tmp.node_labels, device='cpu')
labels = targets.numpy()

cgt_files = set()
sbw_files = set()

with open("./source_files.txt", "r") as f:
    for line in f:
        path = line.strip()
        if not path:
            continue
        if path.startswith("dappscan/"):
            cgt_files.add(path)
        elif path.startswith("sbcurated/"):
            sbw_files.add(path)

cgt_idx = []
sbw_idx = []

for idx, (_, data) in enumerate(nx_graph.nodes(data=True)):
    source = data.get("source_file", "").strip()
    if source in cgt_files:
        cgt_idx.append(idx)
    elif source in sbw_files:
        sbw_idx.append(idx)

print(f"Số node Dappscan: {len(cgt_idx)}")
print(f"Số node Sbcurated: {len(sbw_idx)}")

cgt_trainval_idx, cgt_test_idx = train_test_split(
    cgt_idx, test_size=0.1, stratify=labels[cgt_idx], random_state=42
)
cgt_train_idx, cgt_val_idx = train_test_split(
    cgt_trainval_idx, test_size=0.1111, stratify=labels[cgt_trainval_idx], random_state=42
)

sbw_test_idx = sbw_idx

train_idx = cgt_train_idx              
val_idx = cgt_val_idx                  
test_idx = cgt_test_idx + sbw_test_idx 

total_nodes = len(nx_graph.nodes)
train_mask = torch.zeros(total_nodes, dtype=torch.bool)
val_mask = torch.zeros(total_nodes, dtype=torch.bool)
test_mask = torch.zeros(total_nodes, dtype=torch.bool)

train_mask[train_idx] = True
val_mask[val_idx] = True
test_mask[test_idx] = True

cgt_test_mask = torch.zeros(total_nodes, dtype=torch.bool)
sbw_test_mask = torch.zeros(total_nodes, dtype=torch.bool)
cgt_test_mask[cgt_test_idx] = True
sbw_test_mask[sbw_test_idx] = True

print(f"Số node train: {train_mask.sum().item()}")
print(f"Số node val: {val_mask.sum().item()}")
print(f"Số node test (tổng): {test_mask.sum().item()} (bao gồm cả {len(cgt_test_idx)} Dappscan test và {len(sbw_test_idx)} Sbcurated)")


class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(NUM_CLASSES),
    y=labels
)
alpha_tensor = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)

# === FOCAL LOSS ===
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        if self.alpha is not None:
            at = self.alpha[targets]
            loss = -at * (1 - pt) ** self.gamma * log_pt
        else:
            loss = -(1 - pt) ** self.gamma * log_pt
        return loss.mean() if self.reduction == 'mean' else loss.sum()

# === EARLY STOPPING ===
class EarlyStopping:
    def __init__(self, patience=PATIENCE, mode='max'):
        self.patience = patience
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.best_model_state = None

    def step(self, score, model):
        if self.best_score is None or \
           (self.mode == 'max' and score > self.best_score) or \
           (self.mode == 'min' and score < self.best_score):
            self.best_score = score
            self.counter = 0
            self.best_model_state = model.state_dict()
        else:
            self.counter += 1
        return self.counter >= self.patience

# === INIT MODEL ===
model = HGTVulNodeClassifier(GRAPH_PATH, node_feature='metapath2vec',
                             hidden_size=128, device=DEVICE).to(DEVICE)
if hasattr(model, 'reset_parameters'):
    model.reset_parameters()

targets = torch.tensor(model.node_labels, device='cpu')
targets_gpu = targets.to(DEVICE)
train_mask_gpu = train_mask.to(DEVICE)
val_mask_gpu = val_mask.to(DEVICE)
test_mask_gpu = test_mask.to(DEVICE)

labels_train = targets[train_mask].cpu().numpy()
class_counts = np.bincount(labels_train)
alpha = 1.0 / torch.tensor(class_counts, dtype=torch.float)
alpha = alpha / alpha.sum()
loss_fn = FocalLoss(alpha=alpha.to(DEVICE), gamma=2.0)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10, factor=0.8)
early_stopper = EarlyStopping(patience=PATIENCE, mode='max')

# === TRAIN LOOP ===
f1_scores = []
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()

    logits = model().squeeze(0)
    loss = loss_fn(logits[train_mask_gpu], targets_gpu[train_mask_gpu])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    current_lr = optimizer.param_groups[0]['lr']

    # === VAL ===
    model.eval()
    with torch.no_grad():
        val_preds = torch.argmax(logits[val_mask_gpu], dim=1)
        val_truths = targets_gpu[val_mask_gpu]
        val_loss = loss_fn(logits[val_mask_gpu], val_truths)
        val_f1 = f1_score(val_truths.cpu(), val_preds.cpu(), average='macro')

        train_preds = torch.argmax(logits[train_mask_gpu], dim=1)
        train_truths = targets_gpu[train_mask_gpu]
        train_acc = (train_preds == train_truths).float().mean().item()
        val_acc = (val_preds == val_truths).float().mean().item()

    scheduler.step(val_f1)
    f1_scores.append(val_f1)

    print(f"Epoch {epoch+1}/{EPOCHS} | LR: {current_lr:.6f} | "
          f"Train Loss: {loss.item():.4f} | Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss.item():.4f} | Val Acc: {val_acc:.4f} | "
          f"Val Macro F1: {val_f1:.4f}", flush=True)

    if early_stopper.step(val_f1, model):
        print("Early stopping triggered!")
        break

# === TEST ===
model.load_state_dict(early_stopper.best_model_state)
model.eval()
with torch.no_grad():
    test_logits = model().squeeze(0)
    test_preds = torch.argmax(test_logits[test_mask_gpu], dim=1)
    test_truths = targets_gpu[test_mask_gpu]
    print("\n Final Test Report:\n", classification_report(test_truths.cpu(), test_preds.cpu(), digits=4))
    print(" Final Macro F1: {:.4f}".format(f1_score(test_truths.cpu(), test_preds.cpu(), average='macro')))



print("\nRetest on CGT only:")
cgt_test_gpu = cgt_test_mask.to(DEVICE)
with torch.no_grad():
    logits = model().squeeze(0)
    preds = torch.argmax(logits[cgt_test_gpu], dim=1)
    truths = targets_gpu[cgt_test_gpu]
    print("Dappscan Test Report:\n", classification_report(truths.cpu(), preds.cpu(), digits=4))
    print("Macro F1: {:.4f}".format(f1_score(truths.cpu(), preds.cpu(), average='macro')))

print("\nRetest on SBC only:")
sbw_test_gpu = sbw_test_mask.to(DEVICE)
with torch.no_grad():
    logits = model().squeeze(0)
    preds = torch.argmax(logits[sbw_test_gpu], dim=1)
    truths = targets_gpu[sbw_test_gpu]
    print("SBC Test Report:\n", classification_report(truths.cpu(), preds.cpu(), digits=4))
    print("Macro F1: {:.4f}".format(f1_score(truths.cpu(), preds.cpu(), average='macro')))

