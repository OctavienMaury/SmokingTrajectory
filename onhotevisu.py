#This model allows for a deeper analysis of the importance of variables in the final prediction through the use of a one-hot encoder.
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from torch import nn, optim
from lime import lime_tabular
from sklearn.preprocessing import OneHotEncoder
import streamlit as st
import streamlit.components.v1 as components
import torchviz


data = pd.read_csv('PATH\data.csv')

data['sexe'] = data['sexe'].astype('category').cat.codes
columns_to_use = ['keys_variables']
columns_to_drop = [''useless_variables']
data = data.drop(columns=columns_to_drop)

encoder = OneHotEncoder(drop='first')  
columns_to_encode = ['encode']  

encoded_data = encoder.fit_transform(data[columns_to_encode]).toarray()
feature_labels = encoder.get_feature_names_out() 

encoded_df = pd.DataFrame(encoded_data, columns=feature_labels)

data = data.drop(columns=columns_to_encode)  
data = pd.concat([data, encoded_df], axis=1) 

columns_to_use = [col for col in data.columns if col != 'fume']  

X = torch.tensor(data[columns_to_use].values.astype(np.float32))
y = (data['fume'] > 2).astype(np.float32)
y = torch.tensor(y.values).unsqueeze(1)  # Préparation du vecteur cible

print(data[columns_to_use].dtypes)

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

dataset = Donnees(X, y)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=60, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=60, shuffle=False)

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
optimizer = optim.Adam(model.parameters(), lr=0.001)

print('NaN in X:', torch.isnan(X).any())
print('NaN in y:', torch.isnan(y).any())

loss_fn = nn.BCEWithLogitsLoss()

for epoch in range(50):
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

    model.eval()
    with torch.no_grad():
        total, correct = 0, 0
        for features, labels in test_loader:
            pred = model(features)
            correct += ((pred > 0.5) == labels).type(torch.float).sum().item()
            total += labels.size(0)
        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')

torch.save(model.state_dict(), 'model.pth')

model = NeuralNetwork()
model.load_state_dict(torch.load('model.pth'))
model.eval()

full_loader = DataLoader(dataset, batch_size=60, shuffle=False)

all_labels = []
all_predictions = []

with torch.no_grad():
    for features, labels in full_loader:
        predictions = model(features)
        predicted_labels = (predictions > 0.5).float()  # Utilise la fonction sigmoid pour la prédiction

        all_labels.extend(labels.tolist())
        all_predictions.extend(predicted_labels.tolist())

data['predicted_fume'] = [item[0] for item in all_predictions]


data.to_csv('PATH\predicted_data.csv', index=False)

print("Les prédictions ont été sauvegardées avec succès.")

def accuracy(y_true, y_pred):
    return (y_true == (y_pred > 0.5)).float().mean()

model = NeuralNetwork()
model.load_state_dict(torch.load('model.pth'))
model.eval()

full_loader = DataLoader(dataset, batch_size=64, shuffle=False)
original_accuracy = 0
total = 0
with torch.no_grad():
    for features, labels in full_loader:
        preds = model(features)
        original_accuracy += accuracy(labels, preds).item() * features.size(0)
        total += features.size(0)
original_accuracy /= total

def permutation_importance(model, loader, num_features):
    importances = np.zeros(num_features)
    for i in range(num_features):
        perturbed_accuracy = 0
        perturbed_loader = DataLoader(
            TensorDataset(torch.cat([loader.dataset.features[:, :i], 
                                     loader.dataset.features[:, i:].roll(1, dims=0)], dim=1),
                          loader.dataset.labels),
            batch_size=64, shuffle=False)
        with torch.no_grad():
            for features, labels in perturbed_loader:
                preds = model(features)
                perturbed_accuracy += accuracy(labels, preds).item() * features.size(0)
        perturbed_accuracy /= total
        importances[i] = original_accuracy - perturbed_accuracy
    return importances

importances = permutation_importance(model, full_loader, X.shape[1])
for i, imp in enumerate(importances):
    print(f'{columns_to_use[i]}: {imp:.4f}')

# SHAP
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# Streamlit
st.title("Visualisation des SHAP values et des importances des caractéristiques")

st.subheader("Permutation Importance")
fig, ax = plt.subplots()
ax.barh(columns_to_use, importances)
ax.set_xlabel('Importance')
ax.set_title('Permutation Importance')
st.pyplot(fig)

sample_loader = DataLoader(dataset, batch_size=30, shuffle=True)
background_data = next(iter(sample_loader))[0]
sample_data = next(iter(sample_loader))[0]
explainer = shap.DeepExplainer(model, background_data)
shap_values = explainer.shap_values(sample_data)
if isinstance(shap_values, list):
    shap_values = shap_values[0]
if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
    shap_values = shap_values.squeeze(-1)

st.subheader("Graphique SHAP Summary Plot")
fig_summary, ax_summary = plt.subplots()
shap.summary_plot(shap_values, sample_data, feature_names=columns_to_use, show=False)
st.pyplot(fig_summary)

st.subheader("Graphique SHAP Bar Plot")
fig_bar, ax_bar = plt.subplots()
shap.summary_plot(shap_values, features=X.numpy(), feature_names=columns_to_use, plot_type="bar", show=False)
st.pyplot(fig_bar)

st.subheader("Graphique SHAP Force Plot interactif")
X_tensor = X[:50]
shap_values50 = explainer.shap_values(X_tensor)
if isinstance(shap_values50, list):
    shap_values50 = shap_values50[0]
if shap_values50.ndim == 3 and shap_values50.shape[-1] == 1:
    shap_values50 = shap_values50.squeeze(-1)
force_plot = shap.force_plot(explainer.expected_value, shap_values50, X_tensor.numpy(), feature_names=columns_to_use)
st_shap(force_plot, 400)

x = torch.randn(1, X.shape[1]).requires_grad_(True)
y = model(x)
dot = torchviz.make_dot(y, params=dict(model.named_parameters()))
st.subheader("Représentation graphique du modèle")
st.image(dot.render('model_graph', format='png'))
