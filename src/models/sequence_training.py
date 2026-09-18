from __future__ import annotations

from dataclasses import dataclass
import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset


@dataclass(frozen=True)
class TrainingResult:
    model:nn.Module
    best_epoch:int
    train_loss:float
    val_loss:float


def train_sequence_classifier(
    model:nn.Module,
    X_train:np.ndarray,
    y_train:np.ndarray,
    X_val:np.ndarray,
    y_val:np.ndarray,
    *,
    epochs:int=50,
    batch_size:int=256,
    learning_rate:float=1e-3,
    patience:int=8,
    device:str|None=None,
)->TrainingResult:
    device=device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev=torch.device(device)
    model=model.to(dev)
    train_ds=TensorDataset(
        torch.as_tensor(X_train,dtype=torch.float32),
        torch.as_tensor(y_train,dtype=torch.long)+1
    )
    val_ds=TensorDataset(
        torch.as_tensor(X_val,dtype=torch.float32),
        torch.as_tensor(y_val,dtype=torch.long)+1
    )
    train_loader=DataLoader(train_ds,batch_size=batch_size,shuffle=False)
    val_loader=DataLoader(val_ds,batch_size=batch_size,shuffle=False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=1e-4)
    loss_fn=nn.CrossEntropyLoss()
    best_state=copy.deepcopy(model.state_dict())
    best_loss=float("inf"); best_epoch=0; stale=0
    for epoch in range(1,epochs+1):
        model.train(); total=0.0; count=0
        for xb,yb in train_loader:
            xb=xb.to(dev); yb=yb.to(dev)
            optimizer.zero_grad(set_to_none=True)
            loss=loss_fn(model(xb),yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            total+=float(loss.item())*len(yb); count+=len(yb)
        train_loss=total/max(count,1)
        model.eval(); total=0.0; count=0
        with torch.no_grad():
            for xb,yb in val_loader:
                xb=xb.to(dev); yb=yb.to(dev)
                loss=loss_fn(model(xb),yb)
                total+=float(loss.item())*len(yb); count+=len(yb)
        val_loss=total/max(count,1)
        if val_loss<best_loss-1e-6:
            best_loss=val_loss; best_epoch=epoch; stale=0
            best_state=copy.deepcopy(model.state_dict())
        else:
            stale+=1
            if stale>=patience: break
    model.load_state_dict(best_state)
    return TrainingResult(model,best_epoch,train_loss,best_loss)


def sequence_predict_proba(model:nn.Module,X:np.ndarray,device:str|None=None)->np.ndarray:
    dev=torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model=model.to(dev).eval()
    loader=DataLoader(torch.as_tensor(X,dtype=torch.float32),batch_size=1024,shuffle=False)
    out=[]
    with torch.no_grad():
        for xb in loader:
            out.append(torch.softmax(model(xb.to(dev)),dim=-1).cpu().numpy())
    return np.concatenate(out) if out else np.empty((0,3))
