#This is a simple model that focuses solely on predicting whether an individual smokes or not

import pandas as pd
import numpy as np
import shap
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
import streamlit as st
import matplotlib.pyplot as plt
import streamlit.components.v1 as components  # Importer les composants Streamlit


data = pd.read_csv('PATH\data.csv')

columns_to_use = ['key_columns']
for col in columns_to_use:
    if data[col].dtype == 'object':
        data[col] = data[col].astype('category').cat.codes

class Donnees(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

X = torch.tensor(data[columns_to_use].values.astype(np.float32))
y = (data['fume'] > 2).astype(np.float32)
y = torch.tensor(y.values).unsqueeze(1)

mean = X.mean(0, keepdim=True)
std = X.std(0, keepdim=True)
std[std == 0] = 1
X = (X - mean) / std

dataset = Donnees(X, y)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(X.shape[1], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)

model = NeuralNetwork()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Accuracy
def accuracy(y_true, y_pred):
    return (y_true == (y_pred > 0.5)).float().mean()

#SHAP values
full_loader = DataLoader(dataset, batch_size=60, shuffle=False)
all_labels = []
all_predictions = []

with torch.no_grad():
    for features, labels in full_loader:
        predictions = model(features)
        predicted_labels = (predictions > 0.5).float()
        all_labels.extend(labels.tolist())
        all_predictions.extend(predicted_labels.tolist())


sample_loader = DataLoader(dataset, batch_size=30, shuffle=True)
background_data = next(iter(sample_loader))[0]
sample_data = next(iter(sample_loader))[0]

explainer = shap.DeepExplainer(model, background_data)
shap_values = explainer.shap_values(sample_data)

if isinstance(shap_values, list):
    shap_values = shap_values[0]
if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
    shap_values = shap_values.squeeze(-1)

sample_data_numpy = sample_data.numpy() if isinstance(sample_data, torch.Tensor) else sample_data

# Interface Streamlit
st.title("Visualisation des SHAP values")

#SHAP Summary Plot
st.subheader("Graphique SHAP Summary")
fig_summary, ax_summary = plt.subplots()
shap.summary_plot(shap_values, sample_data_numpy, feature_names=columns_to_use, show=False)
st.pyplot(fig_summary)

# SHAP Bar Plot
st.subheader("Graphique SHAP Bar Plot")
fig_bar, ax_bar = plt.subplots()
shap.summary_plot(shap_values, features=X.numpy(), feature_names=columns_to_use, plot_type="bar", show=False)
st.pyplot(fig_bar)

# SHAP Force
def st_shap(plot, height=None):
    """ Affiche une visualisation SHAP interactive dans Streamlit """
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


if X.shape[0] >= 50:
    X_tensor = X[:50]  # Obtenir les premières 50 instances comme Tensors
    shap_values50 = explainer.shap_values(X_tensor)
    
    if isinstance(shap_values50, list):
        shap_values50 = shap_values50[0]
    if shap_values50.ndim == 3 and shap_values50.shape[-1] == 1:
        shap_values50 = shap_values50.squeeze(-1)
    
    force_plot = shap.force_plot(explainer.expected_value, shap_values50, X_tensor.numpy(), feature_names=columns_to_use)
    st.subheader("Graphique SHAP Force Plot interactif")
    st_shap(force_plot, 400)
else:
    st.write("Pas assez de données pour afficher le graphique SHAP Force Plot interactif.")


import torchviz
x = torch.randn(1, X.shape[1]).requires_grad_(True)
y = model(x)
dot = torchviz.make_dot(y, params=dict(model.named_parameters()))
st.subheader("Représentation graphique du modèle")
st.image(dot.render('model_graph', format='png'))
