# The most comprehensive model predicts whether the individual smokes or not, as well as their age of smoking initiation and cessation.
import streamlit as st
import pandas as pd
import requests
from sqlalchemy import create_engine
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn, optim
from sklearn.preprocessing import OneHotEncoder
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import streamlit.components.v1 as components
import torchviz
import psycopg2
import numpy as np

# Récupérer les secrets depuis Streamlit
gdrive_url = st.secrets["gdrive_url"]["gdrive_url"]

# Fonction pour télécharger et charger le fichier CSV depuis Google Drive
@st.cache_data
def load_csv_from_gdrive(_url):
    response = requests.get(_url)
    response.raise_for_status()  # Assurez-vous que la requête a réussi
    csv_file_path = 'data.csv'
    with open(csv_file_path, 'wb') as f:
        f.write(response.content)
    return pd.read_csv(csv_file_path)

data = load_csv_from_gdrive(gdrive_url)

#debogage
#st.write(data.head())

# Interface Streamlit
st.title("Origine sociale et parcours tabagiques, une approche via les réseaux de neurones")

# Prétraitement des données
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

# Update columns
columns_to_use = [col for col in data.columns if col not in ['fume', 'age_init', 'age_cess', 'afume', 'nbanfum', 'aarret']]

data = data.dropna(subset=['age_init', 'age_cess'])

age_init_mean = data['age_init'].mean()
age_init_std = data['age_init'].std()
data['age_init'] = (data['age_init'] - age_init_mean) / age_init_std

age_cess_mean = data['age_cess'].mean()
age_cess_std = data['age_cess'].std()
data['age_cess'] = (data['age_cess'] - age_cess_mean) / age_cess_std

# Conversion des données en tenseurs pour PyTorch
X = torch.tensor(data[columns_to_use].values.astype(np.float32))
y_fume = torch.tensor((data['fume'] > 2).astype(np.float32).values).unsqueeze(1)
y_age_init = torch.tensor(data['age_init'].values.astype(np.float32)).unsqueeze(1)
y_age_cess = torch.tensor(data['age_cess'].values.astype(np.float32)).unsqueeze(1)

mean = X.mean(0, keepdim=True)
std = X.std(0, keepdim=True)
std[std == 0] = 1
X = (X - mean) / std

class Donnees(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

dataset_fume = Donnees(X, y_fume)
dataset_age_init = Donnees(X, y_age_init)
dataset_age_cess = Donnees(X, y_age_cess)

def split_dataset(dataset, train_ratio=0.8):
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    return random_split(dataset, [train_size, test_size])

train_dataset_fume, test_dataset_fume = split_dataset(dataset_fume)
train_dataset_age_init, test_dataset_age_init = split_dataset(dataset_age_init)
train_dataset_age_cess, test_dataset_age_cess = split_dataset(dataset_age_cess)

train_loader_fume = DataLoader(train_dataset_fume, batch_size=60, shuffle=True)
test_loader_fume = DataLoader(test_dataset_fume, batch_size=60, shuffle=False)
train_loader_age_init = DataLoader(train_dataset_age_init, batch_size=60, shuffle=True)
test_loader_age_init = DataLoader(test_dataset_age_init, batch_size=60, shuffle=False)
train_loader_age_cess = DataLoader(train_dataset_age_cess, batch_size=60, shuffle=True)
test_loader_age_cess = DataLoader(test_dataset_age_cess, batch_size=60, shuffle=False)

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

model_fume = NeuralNetwork(X.shape[1])
model_age_init = NeuralNetwork(X.shape[1])
model_age_cess = NeuralNetwork(X.shape[1])

optimizer_fume = optim.Adam(model_fume.parameters(), lr=0.0005)
optimizer_age_init = optim.Adam(model_age_init.parameters(), lr=0.0005)
optimizer_age_cess = optim.Adam(model_age_cess.parameters(), lr=0.0005)

loss_fn_fume = nn.BCEWithLogitsLoss()
loss_fn_reg = nn.MSELoss()

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

def accuracy_classification(pred, labels):
    pred = torch.sigmoid(pred)
    pred_labels = (pred > 0.5).float()
    correct = (pred_labels == labels).float().sum()
    return correct / labels.shape[0]

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
                correct += accuracy_classification(pred, labels).item() * labels.size(0)
            total += labels.size(0)
    average_loss = total_loss / len(test_loader)
    if task == 'classification':
        accuracy = correct / total
        print(f'Average Test Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}')
    else:
        print(f'Average Test Loss: {average_loss:.4f}')
    return average_loss

def evaluate_regression_model(model, test_loader):
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for features, labels in test_loader:
            pred = model(features)
            loss = mean_squared_error(labels.numpy(), pred.numpy())
            total_loss += loss
            all_labels.extend(labels.numpy())
            all_preds.extend(pred.numpy())
    average_loss = total_loss / len(test_loader)
    mae = mean_absolute_error(all_labels, all_preds)
    print(f'Average Test MSE Loss: {average_loss:.4f}, MAE: {mae:.4f}')
    return average_loss, mae

# Training
print("Training model for fume prediction...")
train_model(model_fume, optimizer_fume, loss_fn_fume, train_loader_fume)
print("Training model for age of initiation prediction...")
train_model(model_age_init, optimizer_age_init, loss_fn_reg, train_loader_age_init)
print("Training model for age of cessation prediction...")
train_model(model_age_cess, optimizer_age_cess, loss_fn_reg, train_loader_age_cess)

# Evaluate
print("Model Fume:")
evaluate_model(model_fume, loss_fn_fume, test_loader_fume, task='classification')
print("Model Age Init:")
evaluate_regression_model(model_age_init, test_loader_age_init)
print("Model Age Cess:")
evaluate_regression_model(model_age_cess, test_loader_age_cess)

# Save
torch.save(model_fume.state_dict(), 'model_fume.pth')
torch.save(model_age_init.state_dict(), 'model_age_init.pth')
torch.save(model_age_cess.state_dict(), 'model_age_cess.pth')

# Load
model_fume.load_state_dict(torch.load('model_fume.pth'))
model_age_init.load_state_dict(torch.load('model_age_init.pth'))
model_age_cess.load_state_dict(torch.load('model_age_cess.pth'))

model_fume.eval()
model_age_init.eval()
model_age_cess.eval()

# Predict
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

# Save predicted data
data.to_csv('predicted_data.csv', index=False)

print("Les prédictions ont été sauvegardées avec succès.")

# Convert NumPy -> PyTorch tensor
class ModelWrapper:
    def __init__(self, model):
        self.model = model

    def __call__(self, x):
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32)
            return self.model(x_tensor).detach().numpy().flatten()

X_np = X.numpy()

models = {
    "model_fume": model_fume,
    "model_age_init": model_age_init,
    "model_age_cess": model_age_cess
}

# SHAP
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# Streamlit
st.title("Visualisation des SHAP values et des importances des caractéristiques pour 3 modèles")

for model_name, model in models.items():
    st.subheader(f"SHAP Summary Plot pour {model_name}")
    
    model_wrapper = ModelWrapper(model)
    
    background = X_np[np.random.choice(X_np.shape[0], 100, replace=False)]
    explainer = shap.KernelExplainer(model_wrapper, background)
    shap_values = explainer.shap_values(X_np)
    
    if isinstance(shap_values, list):
        shap_values = np.array(shap_values)

    if shap_values.ndim == 3:
        shap_values = shap_values[0]
    
    st.write(f"shap_values shape for {model_name}: {shap_values.shape}")
    st.write(f"X_np shape: {X_np.shape}")
    
    fig_summary, ax_summary = plt.subplots()
    shap.summary_plot(shap_values, X_np, feature_names=columns_to_use, show=False)
    st.pyplot(fig_summary)
    
    shap_values_mean = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame(list(zip(columns_to_use, shap_values_mean)), columns=['Feature', 'SHAP Importance'])
    importance_df = importance_df.sort_values(by='SHAP Importance', ascending=False)
    
    st.write(f"\nScores d'importance des variables pour {model_name}:")
    st.write(importance_df)
    
    st.subheader(f"Graphique SHAP Force Plot interactif pour {model_name}")
    shap_values_sample = shap_values[:50]
    force_plot = shap.force_plot(explainer.expected_value, shap_values_sample, X_np[:50], feature_names=columns_to_use)
    st_shap(force_plot, 400)
    
    if 'mere_pcs_6' in columns_to_use:
        feature_idx = columns_to_use.index('mere_pcs_6')
        fig_dependence, ax_dependence = plt.subplots()
        shap.dependence_plot(feature_idx, shap_values, X_np, feature_names=columns_to_use, ax=ax_dependence, show=False)
        st.pyplot(fig_dependence)

# SHAP KEY VARIABLES
mere_pcs_vars = [f'mere_pcs_{i}' for i in range(1, 7)]
pere_pcs_vars = [f'pere_pcs_{i}' for i in range(1, 7)]
mere_etude_vars = [f'mere_etude_{i}' for i in range(1, 7)]
pere_etude_vars = [f'pere_etude_{i}' for i in range(1, 7)]

all_vars = mere_pcs_vars + pere_pcs_vars + mere_etude_vars + pere_etude_vars

for var in all_vars:
    if var in columns_to_use:
        feature_idx = columns_to_use.index(var)
        for model_name, model in models.items():
            st.subheader(f"SHAP Dependence Plot pour {model_name} - {var}")
            fig_dependence, ax_dependence = plt.subplots()
            shap.dependence_plot(feature_idx, shap_values, X_np, feature_names=columns_to_use, ax=ax_dependence, show=False)
            st.pyplot(fig_dependence)

for model_name, model in models.items():
    st.subheader(f"Représentation graphique du modèle {model_name}")
    x = torch.randn(1, X.shape[1]).requires_grad_(True)
    y = model(x)
    dot = torchviz.make_dot(y, params=dict(model.named_parameters()))
    st.image(dot.render(f'model_graph_{model_name}', format='png'))
