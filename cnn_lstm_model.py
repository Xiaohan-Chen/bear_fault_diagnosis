import os
import re
import time

import numpy as np
import scipy.io as scio
import scipy.signal

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

# ============================================================
# Hyperparameters (aligned with the original Keras version)
# ============================================================
RAW_NUM = 240          # Number of samples split from each .mat file
COL_NUM = 2000         # Original length of each sample
DOWNSAMPLE_FACTOR = 8  # Downsampling factor: 2000/8 = 250
NUM_CLASSES = 10       # Number of fault classes
BATCH_SIZE = 100
EPOCHS = 400
LEARNING_RATE = 0.0006
TEST_SIZE = 0.3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
# Data loading
# ============================================================
class Data:
    """Load bearing vibration data from a directory or a single .mat file."""

    def __init__(self, data_path='data/'):
        """
        Parameters
        ----------
        data_path : str
            Path to a single .mat file or a directory containing .mat files.
        """
        self.data_path = data_path
        self.data = self._get_data()
        self.label = self._get_label()

    def _file_list(self):
        if os.path.isfile(self.data_path):
            return [os.path.basename(self.data_path)]
        try:
            return sorted(os.listdir(self.data_path))
        except OSError as e:
            raise FileNotFoundError(f"Cannot read data path '{self.data_path}': {e}") from e

    def _get_data(self):
        file_list = self._file_list()
        data = None
        for i, fname in enumerate(file_list):
            if os.path.isfile(self.data_path):
                fpath = self.data_path
            else:
                fpath = os.path.join(self.data_path, fname)

            file = scio.loadmat(fpath)
            key = None
            for k in file.keys():
                if re.match(r'X\d{3}_DE_time', k):
                    key = k
                    break
            if key is None:
                raise KeyError(f"No key matching 'X\\d{{3}}_DE_time' found in {fpath}")

            segment = np.array(file[key][0:RAW_NUM * COL_NUM].reshape(RAW_NUM, COL_NUM))
            if data is None:
                data = segment
            else:
                data = np.vstack((data, segment))
        return data

    def _get_label(self):
        file_list = self._file_list()
        titles = np.array([f.replace('.mat', '') for f in file_list])
        label = titles[:, np.newaxis]
        label_copy = np.copy(label)
        for _ in range(RAW_NUM - 1):
            label = np.hstack((label, label_copy))
        return label.flatten()


# ============================================================
# Model definition
# ============================================================
class CNN_LSTM(nn.Module):
    """
    Dual-encoder CNN-LSTM fault diagnosis model.

    Architecture:
        Encoder1: Conv1d(1->50,k20,s2) -> Conv1d(50->30,k10,s2) -> MaxPool(2)
        Encoder2: Conv1d(1->50,k6,s1) -> Conv1d(50->40,k6,s1) -> MaxPool(2)
                  -> Conv1d(40->30,k6,s1) -> Conv1d(30->30,k6,s2) -> MaxPool(2)
        Element-wise multiply of the two encoder outputs.
        LSTM(60, return_seq) -> LSTM(60) -> Dropout(0.5) -> Dense(10, softmax)
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super(CNN_LSTM, self).__init__()

        # Encoder 1
        self.ec1_conv1 = nn.Conv1d(in_channels=1, out_channels=50, kernel_size=20, stride=2)
        self.ec1_conv2 = nn.Conv1d(in_channels=50, out_channels=30, kernel_size=10, stride=2)
        self.ec1_pool = nn.MaxPool1d(kernel_size=2)

        # Encoder 2
        self.ec2_conv1 = nn.Conv1d(in_channels=1, out_channels=50, kernel_size=6, stride=1)
        self.ec2_conv2 = nn.Conv1d(in_channels=50, out_channels=40, kernel_size=6, stride=1)
        self.ec2_pool1 = nn.MaxPool1d(kernel_size=2)
        self.ec2_conv3 = nn.Conv1d(in_channels=40, out_channels=30, kernel_size=6, stride=1)
        self.ec2_conv4 = nn.Conv1d(in_channels=30, out_channels=30, kernel_size=6, stride=2)
        self.ec2_pool2 = nn.MaxPool1d(kernel_size=2)

        # LSTM decoder
        # Conv1d output shape: (batch, 30, 27)
        # LSTM treats it as (batch, seq_len=30, features=27)
        self.lstm1 = nn.LSTM(input_size=27, hidden_size=60, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=60, hidden_size=60, batch_first=True)

        # Classification head
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(60, num_classes)

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor, shape (batch, 250)
        """
        # Reshape: (batch, 250) -> (batch, 1, 250)
        x = x.unsqueeze(1)

        # Encoder 1
        ec1 = torch.tanh(self.ec1_conv1(x))
        ec1 = torch.tanh(self.ec1_conv2(ec1))
        ec1 = self.ec1_pool(ec1)  # (batch, 30, 27)

        # Encoder 2
        ec2 = torch.tanh(self.ec2_conv1(x))
        ec2 = torch.tanh(self.ec2_conv2(ec2))
        ec2 = self.ec2_pool1(ec2)
        ec2 = torch.tanh(self.ec2_conv3(ec2))
        ec2 = torch.tanh(self.ec2_conv4(ec2))
        ec2 = self.ec2_pool2(ec2)  # (batch, 30, 27)

        # Element-wise multiply
        encoder = ec1 * ec2  # (batch, 30, 27)

        # LSTM: treat (batch, 30, 27) as (batch, seq_len=30, input_size=27)
        lstm_out, _ = self.lstm1(encoder)   # (batch, 30, 60)  return_sequences=True
        lstm_out, _ = self.lstm2(lstm_out)  # (batch, 30, 60)
        lstm_out = lstm_out[:, -1, :]       # (batch, 60)  take last timestep (equiv. to Keras return_sequences=False)

        # Classification
        out = self.dropout(lstm_out)
        out = torch.softmax(self.fc(out), dim=1)  # (batch, 10)

        return out


# ============================================================
# Training function
# ============================================================
def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE, device=DEVICE):
    """Train the model with MSE loss (consistent with the original Keras version)."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        # ---- Training phase ----
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            pred = outputs.argmax(dim=1)
            true = y_batch.argmax(dim=1)
            train_correct += (pred == true).sum().item()
            train_total += X_batch.size(0)

        # ---- Validation phase ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item() * X_batch.size(0)
                pred = outputs.argmax(dim=1)
                true = y_batch.argmax(dim=1)
                val_correct += (pred == true).sum().item()
                val_total += X_batch.size(0)

        # Record history
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        history['loss'].append(epoch_train_loss)
        history['acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        print(f"Epoch {epoch + 1}/{epochs} - "
              f"loss: {epoch_train_loss:.4f} - acc: {epoch_train_acc:.4f} - "
              f"val_loss: {epoch_val_loss:.4f} - val_acc: {epoch_val_acc:.4f}")

    return history


# ============================================================
# Main entry point
# ============================================================
def main(data_path='data/'):
    """
    Main function.

    Parameters
    ----------
    data_path : str
        Path to a .mat file or a directory containing .mat files.
    """
    print(f"Using device: {DEVICE}")
    begin_time = time.time()

    # ---- 1. Load data ----
    print("Loading data...")
    dataset = Data(data_path)
    data = dataset.data
    label = dataset.label

    # One-hot encode labels
    lb = LabelBinarizer()
    y = lb.fit_transform(label)
    num_classes = y.shape[1]
    print(f"Number of classes: {num_classes}, classes: {lb.classes_}")

    # ---- 2. Preprocessing ----
    # Wiener filtering
    data_wiener = scipy.signal.wiener(data, mysize=3, noise=None)

    # Downsampling: 2000 -> 250
    index = np.arange(0, COL_NUM, DOWNSAMPLE_FACTOR)
    data_samp = data_wiener[:, index]
    print(f"Downsampled data shape: {data_samp.shape}")

    # ---- 3. Train/test split ----
    X_train, X_test, y_train, y_test = train_test_split(
        data_samp, y, test_size=TEST_SIZE
    )

    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ---- 4. Build model ----
    model = CNN_LSTM(num_classes=num_classes)
    print(model)

    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")

    # ---- 5. Train ----
    print(f"\nStart training (epochs={EPOCHS}, batch_size={BATCH_SIZE}, lr={LEARNING_RATE})...")
    history = train_model(model, train_loader, test_loader)

    end_time = time.time()
    print(f'\nTotal training time: {end_time - begin_time:.1f}s')

    # ---- 6. Save model ----
    torch.save(model.state_dict(), 'cnn_lstm_model.pth')
    print("Model saved as cnn_lstm_model.pth")

    return model, history


if __name__ == '__main__':
    import sys
    # Pass a .mat file path via CLI: python cnn_lstm_model.py /path/to/file.mat
    # Or pass a data directory:     python cnn_lstm_model.py /path/to/data_dir/
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/'
    main(data_path=path)
