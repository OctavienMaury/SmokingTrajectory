import streamlit as st
import psycopg2

# Récupérer les secrets depuis Streamlit
db_host = st.secrets["DB"]["DB_HOST"]
db_user = st.secrets["DB"]["DB_USER"]
db_password = st.secrets["DB"]["DB_PASSWORD"]
db_name = st.secrets["DB"]["DB_NAME"]

# Afficher les valeurs des variables pour vérifier leur exactitude
st.write(f"db_host: {db_host}")
st.write(f"db_user: {db_user}")
st.write(f"db_password: {db_password}")
st.write(f"db_name: {db_name}")

# Connexion à PostgreSQL
st.write("Tentative de connexion à la base de données avec psycopg2...")
try:
    conn = psycopg2.connect(
        host=db_host,
        database=db_name,
        user=db_user,
        password=db_password
    )
    st.write("Connexion à la base de données réussie.")
    conn.close()
except Exception as e:
    st.error(f"Erreur de connexion à la base de données: {e}")
