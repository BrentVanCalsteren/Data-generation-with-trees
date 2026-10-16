import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from pydl85 import DL85Classifier
import random
import numpy as np


class NNnetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 64), dropout_rate=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)])

            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

########################################
#nn discriminator
######################################
class NNDiscriminator:
    def __init__(self,batch_size=128, dropout_rate=0.2,learning_rate=0.001, epochs=100):
        self.hidden_dims = (128, 64)
        self.dropout_rate = dropout_rate
        self.lr = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: NNnetwork
        self.history = {'train_loss': [], 'val_loss': [], 'val_auc': []}

    def _create_data_loaders(self, X_real, X_gen, val_size=0.2):
        n_real = X_real.shape[0]
        n_gen = X_gen.shape[0]

        X = np.vstack([X_real, X_gen])
        y = np.hstack([np.ones(n_real), np.zeros(n_gen)]).astype(np.float32) #labels-> 1D
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1)

        indices = np.random.permutation(len(X))#shuffle
        val_size_n = int(len(X) * val_size)
        train_idx = indices[val_size_n:]
        val_idx = indices[:val_size_n]

        train_dataset = TensorDataset(X_tensor[train_idx], y_tensor[train_idx])
        val_dataset = TensorDataset(X_tensor[val_idx], y_tensor[val_idx])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader

    def _train_epoch(self, train_loader, criterion, optimizer):
        self.model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(X_batch)
        return total_loss / len(train_loader.dataset)

    def _validate(self, val_loader, criterion):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                total_loss += loss.item() * len(X_batch)
                all_preds.extend(outputs.cpu().numpy().flatten())
                all_labels.extend(y_batch.cpu().numpy().flatten())

        val_loss = total_loss / len(val_loader.dataset)
        val_auc = roc_auc_score(all_labels, all_preds) # Calculate AUC

        return val_loss, val_auc

    def fit(self, X_real, X_gen, verbose=True):
        input_dim = X_real.shape[1]
        self.model = NNnetwork(input_dim,self.hidden_dims,self.dropout_rate).to(self.device)
        train_loader, val_loader = self._create_data_loaders(X_real, X_gen)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(self.epochs):
            train_loss = self._train_epoch(train_loader, criterion, optimizer)
            val_loss, val_auc = self._validate(val_loader, criterion)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_auc'].append(val_auc)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, loss:{train_loss} ")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= 10: #stop after à time no improv
                    if verbose: print(f"Early stopping at epoch {epoch + 1}")
                    break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        return self

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
        return outputs.cpu().numpy().flatten()


########################################
#pydl discripinator -> classifier model
######################################

class DLDiscriminator:
    def __init__(self, real_samples:np.ndarray, gen_samples:np.ndarray):
        x = np.vstack([real_samples, gen_samples])
        y = np.hstack([np.ones(real_samples.shape[0]), np.zeros(gen_samples.shape[0])]).astype(np.float32)
        self.x_train, self.x_test, self.y_train, self.y_test = split_train_test(x, y, test_size=0.2)

    def classify(self):
        classify_test_pydl(x_train=self.x_train,x_test=self.x_test,y_train=self.y_train,y_test=self.y_test)


def classify_test_pydl(x_train, x_test, y_train, y_test):
    clasfi = create_classifier_default(
        x_bin=x_train, y=y_train, max_depth=4, min_sup=1, time=300)
    y_pred_test = clasfi.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_test)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test))

def create_classifier_default(x_bin,y,max_depth=3,min_sup=2,time=100):
    clasfi = DL85Classifier(max_depth=max_depth,min_sup=min_sup, time_limit=time)
    clasfi.fit(x_bin, y)
    return clasfi

def split_train_test(x, y, test_size=0.2):
    indices = np.random.permutation(len(x))
    x_shuffled = x[indices]
    y_shuffled = y[indices]
    x_train, x_test, y_train, y_test = train_test_split(x_shuffled, y_shuffled, test_size=test_size,
                                                        random_state=random.randint(1, 100))
    return x_train, x_test, y_train, y_test
#TODO complete disciminator

