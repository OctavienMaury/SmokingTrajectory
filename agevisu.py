import os
import sys
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn, optim
from sklearn.preprocessing import OneHotEncoder
import shap
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import pymysql
import numpy as np

# Fonction pour récupérer les arguments de la ligne de commande
def get_db_params():
    params = json.loads(sys.argv[1])
    return params

def connect_to_db(db_params):
    try:
        conn = pymysql.connect(
            host=db_params["DB_HOST"],
            user=db_params["DB_USER"],
            password=db_params["DB_PASSWORD"],
            db=db_params["DB_NAME"],
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        return conn
    except Exception as e:
        print(f"Erreur de connexion à la base de données: {e}")
        return None

def load_data_from_db(conn):
    if conn is None:
        print("Impossible de se connecter à la base de données. Veuillez vérifier vos paramètres de connexion.")
        return pd.DataFrame()
    try:
        query = "SELECT * FROM my_table"
        data = pd.read_sql(query, conn)
        print("Données chargées avec succès.")
        return data
    except Exception as e:
        print(f"Erreur lors du chargement des données: {e}")
        return pd.DataFrame()

# Prétraitement des données
def preprocess_data(data):
    data['age_init'] = data.apply(lambda row: row['age'] - row['nbanfum'] if row['afume'] == 1 else np.nan, axis=1)
    data['age_cess'] = data.apply(lambda row: row['age'] - row['nbanfum'] if row['aarret'] > 0 else np.nan, axis=1)

    data['sexe'] = data['sexe'].astype('category').cat.codes
    columns_to_drop = ['ben_n4', 'nind', 'pond_pers_total']

    # One-Hot Encoding
    encoder = OneHotEncoder(drop='first')
    columns_to_encode = ['mere_pcs', 'pere_pcs', 'mere_etude', 'pere_etude']

    encoded_data = encoder.fit_transform(data[columns_to_encode]).toarray()
    feature_labels = encoder.get_feature_names_out()
    encoded_df = pd.DataFrame(encoded_data, columns=feature_labels)

    data = data.drop(columns=columns_to_encode + columns_to_drop)
    data = pd.concat([data, encoded_df], axis=1)

    columns_to_use = [col for col in data.columns if col not in ['fume', 'age_init', 'age_cess', 'afume', 'nbanfum', 'aarret']]
    data = data.dropna(subset=['age_init', 'age_cess'])

    age_init_mean = data['age_init'].mean()
    age_init_std = data['age_init'].std()
    data['age_init'] = (data['age_init'] - age_init_mean) / age_init_std

    age_cess_mean = data['age_cess'].mean()
    age_cess_std = data['age_cess'].std()
    data['age_cess'] = (data['age_cess'] - age_cess_mean) / age_cess_std

    return data, columns_to_use

# Classe pour PyTorch
class Donnees(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Définir le modèle
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)

# Entraîner le modèle
def train_model(model, optimizer, loss_fn, train_loader, num_epochs=50):
    for epoch in range(num_epochs):
        model.train()
        for features, labels in train_loader:
            optimizer.zero_grad()
            pred = model(features)
            loss = loss_fn(pred, labels)
            if torch.isnan(loss):
                print("NaN loss detected")
                break
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item() if not torch.isnan(loss) else "NaN detected"}')

# Évaluer le modèle
def evaluate_model(model, loss_fn, test_loader, task='classification'):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in test_loader:
            pred = model(features)
            loss = loss_fn(pred, labels)
            total_loss += loss.item()
            if task == 'classification':
                pred_labels = (torch.sigmoid(pred) > 0.5).float()
                correct += (pred_labels == labels).float().sum().item()
            total += labels.size(0)
    average_loss = total_loss / len(test_loader)
    if task == 'classification':
        accuracy = correct / total
        print(f'Average Test Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}')
    else:
        print(f'Average Test Loss: {average_loss:.4f}')
    return average_loss

def main():
    # Récupérer les paramètres de connexion à la base de données
    db_params = get_db_params()

    # Connexion à la base de données et chargement des données
    conn = connect_to_db(db_params)
    data = load_data_from_db(conn)

    if data.empty:
        print("Erreur : Les données n'ont pas pu être chargées.")
        return

    data, columns_to_use = preprocess_data(data)

    # Conversion des données en tenseurs pour PyTorch
    X = torch.tensor(data[columns_to_use].values.astype(np.float32))
    y_fume = torch.tensor((data['fume'] > 2).astype(np.float32).values).unsqueeze(1)
    y_age_init = torch.tensor(data['age_init'].values.astype(np.float32)).unsqueeze(1)
    y_age_cess = torch.tensor(data['age_cess'].values.astype(np.float32)).unsqueeze(1)

    mean = X.mean(0, keepdim=True)
    std = X.std(0, keepdim=True)
    std[std == 0] = 1
    X = (X - mean) / std

    # Préparation des datasets et dataloaders
    dataset_fume = Donnees(X, y_fume)
    dataset_age_init = Donnees(X, y_age_init)
    dataset_age_cess = Donnees(X, y_age_cess)

    train_dataset_fume, test_dataset_fume = random_split(dataset_fume, [int(0.8 * len(dataset_fume)), len(dataset_fume) - int(0.8 * len(dataset_fume))])
    train_dataset_age_init, test_dataset_age_init = random_split(dataset_age_init, [int(0.8 * len(dataset_age_init)), len(dataset_age_init) - int(0.8 * len(dataset_age_init))])
    train_dataset_age_cess, test_dataset_age_cess = random_split(dataset_age_cess, [int(0.8 * len(dataset_age_cess)), len(dataset_age_cess) - int(0.8 * len(dataset_age_cess))])

    train_loader_fume = DataLoader(train_dataset_fume, batch_size=60, shuffle=True)
    test_loader_fume = DataLoader(test_dataset_fume, batch_size=60, shuffle=False)
    train_loader_age_init = DataLoader(train_dataset_age_init, batch_size=60, shuffle=True)
    test_loader_age_init = DataLoader(test_dataset_age_init, batch_size=60, shuffle=False)
    train_loader_age_cess = DataLoader(train_dataset_age_cess, batch_size=60, shuffle=True)
    test_loader_age_cess = DataLoader(test_dataset_age_cess, batch_size=60, shuffle=False)

    # Initialisation des modèles, optimisateurs et fonctions de perte
    model_fume = NeuralNetwork(X.shape[1])
    model_age_init = NeuralNetwork(X.shape[1])
    model_age_cess = NeuralNetwork(X.shape[1])

    optimizer_fume = optim.Adam(model_fume.parameters(), lr=0.0005)
    optimizer_age_init = optim.Adam(model_age_init.parameters(), lr=0.0005)
    optimizer_age_cess = optim.Adam(model_age_cess.parameters(), lr=0.0005)

    loss_fn_fume = nn.BCEWithLogitsLoss()
    loss_fn_reg = nn.MSELoss()

    # Entraînement des modèles
    print("Training model for fume prediction...")
    train_model(model_fume, optimizer_fume, loss_fn_fume, train_loader_fume)
    print("Training model for age of initiation prediction...")
    train_model(model_age_init, optimizer_age_init, loss_fn_reg, train_loader_age_init)
    print("Training model for age of cessation prediction...")
    train_model(model_age_cess, optimizer_age_cess, loss_fn_reg, train_loader_age_cess)

    # Évaluation des modèles
    print("Model Fume:")
    evaluate_model(model_fume, loss_fn_fume, test_loader_fume, task='classification')
    print("Model Age Init:")
    evaluate_model(model_age_init, loss_fn_reg, test_loader_age_init, task='regression')
    print("Model Age Cess:")
    evaluate_model(model_age_cess, loss_fn_reg, test_loader_age_cess, task='regression')

    # Sauvegarde des modèles
    torch.save(model_fume.state_dict(), 'model_fume.pth')
    torch.save(model_age_init.state_dict(), 'model_age_init.pth')
    torch.save(model_age_cess.state_dict(), 'model_age_cess.pth')

    # Prédictions
    full_loader = DataLoader(Donnees(X, y_fume), batch_size=60, shuffle=False)

    all_labels_fume, all_labels_age_init, all_labels_age_cess = [], [], []
    all_predictions_fume, all_predictions_age_init, all_predictions_age_cess = [], [], []

    with torch.no_grad():
        for features, labels in full_loader:
            predictions_fume = model_fume(features)
            predicted_labels_fume = (predictions_fume > 0.5).float()

            predictions_age_init = model_age_init(features)
            predictions_age_cess = model_age_cess(features)

            all_labels_fume.extend(labels.tolist())
            all_predictions_fume.extend(predicted_labels_fume.tolist())
            all_predictions_age_init.extend(predictions_age_init.tolist())
            all_predictions_age_cess.extend(predictions_age_cess.tolist())

    data['predicted_fume'] = [item[0] for item in all_predictions_fume]
    data['predicted_age_init'] = [item[0] for item in all_predictions_age_init]
    data['predicted_age_cess'] = [item[0] for item in all_predictions_age_cess]

    # Sauvegarde des données prédites
    data.to_csv('predicted_data.csv', index=False)
    print("Les prédictions ont été sauvegardées avec succès.")

if __name__ == "__main__":
    main()
