import streamlit as st
from sqlalchemy import create_engine

# Récupérer les secrets depuis Streamlit
db_host = st.secrets["DB"]["DB_HOST"]  # Utilisez l'adresse IP locale ici
db_user = st.secrets["DB"]["DB_USER"]
db_password = st.secrets["DB"]["DB_PASSWORD"]
db_name = st.secrets["DB"]["DB_NAME"]

# Afficher les valeurs des variables pour vérifier leur exactitude
st.write(f"db_host: {db_host}")
st.write(f"db_user: {db_user}")
st.write(f"db_password: {db_password}")
st.write(f"db_name: {db_name}")

# Connexion à PostgreSQL avec SQLAlchemy
st.write("Tentative de connexion à la base de données avec SQLAlchemy...")
try:
    engine = create_engine(f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}/{db_name}')
    connection = engine.connect()
    st.write("Connexion à la base de données réussie.")
    connection.close()
except Exception as e:
    st.error(f"Erreur de connexion à la base de données: {e}")
