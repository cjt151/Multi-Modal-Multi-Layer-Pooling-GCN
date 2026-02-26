import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix


def collate_graph_batch(batch):
    Xs, As, Ys = zip(*batch)
    return torch.stack(Xs, dim=0), torch.stack(As, dim=0), torch.tensor(Ys, dtype=torch.long)


def train_model_gnn(model, train_loader, optimizer, criterion, epoch, writer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    for X_batch, A_batch, labels in train_loader:
        X_batch = X_batch.to(device); A_batch = A_batch.to(device); labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch, A_batch)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels).item()
        total += labels.size(0)
    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    if writer is not None:
        writer.add_scalar('train/loss', epoch_loss, epoch)
        writer.add_scalar('train/acc', epoch_acc, epoch)
    return epoch_loss, epoch_acc


def evaluate_model_gnn(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels=[]; all_preds=[]; all_probs=[]
    with torch.no_grad():
        for X_batch, A_batch, labels in val_loader:
            X_batch = X_batch.to(device); A_batch = A_batch.to(device); labels = labels.to(device)
            outputs = model(X_batch, A_batch)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * labels.size(0)
            all_labels.extend(labels.cpu().numpy()); all_preds.extend(preds.cpu().numpy()); all_probs.extend(probs[:,1].cpu().numpy())
    total = len(all_labels)
    epoch_loss = running_loss / total if total>0 else 0.0
    all_labels = np.array(all_labels); all_preds = np.array(all_preds); all_probs = np.array(all_probs)
    acc = accuracy_score(all_labels, all_preds) if total>0 else 0.0
    f1 = f1_score(all_labels, all_preds, average='macro') if total>0 else 0.0
    try:
        auc = roc_auc_score(all_labels, all_probs) if total>0 else 0.5
    except Exception:
        auc = 0.5
    cm = confusion_matrix(all_labels, all_preds) if total>0 else np.zeros((2,2),dtype=int)
    if cm.shape==(2,2):
        TP, FN, TN, FP = cm[1,1], cm[1,0], cm[0,0], cm[0,1]
        sen = TP / (TP + FN) if (TP+FN)>0 else 0.0
        spe = TN / (TN + FP) if (TN+FP)>0 else 0.0
    else:
        sen = spe = 0.0
    return {'loss': epoch_loss, 'acc': acc, 'f1': f1, 'auc': auc, 'sen': sen, 'spe': spe, 'cm': cm}
