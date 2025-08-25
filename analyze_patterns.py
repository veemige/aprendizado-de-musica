
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

# Carrega o arquivo CSV com as características
try:
    df = pd.read_csv("features.csv")
except FileNotFoundError:
    print("Arquivo 'features.csv' não encontrado.")
    print("Execute o script 'feature_extraction.py' primeiro.")
    exit()

# 1. Preparação dos Dados
# Guarda os nomes dos arquivos para podermos ver os resultados depois
filenames = df['filename']
# Remove a coluna 'filename' para que restem apenas os dados numéricos para o modelo
features = df.drop('filename', axis=1)

# 2. Normalização dos Dados
# É crucial para algoritmos de clustering como o KMeans
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 3. Aplicação do Clustering (KMeans)
# Vamos pedir para o algoritmo encontrar 4 clusters (padrões).
# Você pode experimentar mudar este número (n_clusters) para ver se encontra mais ou menos padrões.
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans.fit(features_scaled)

# Adiciona os resultados do clustering (a qual cluster cada música pertence) de volta ao DataFrame
df['cluster'] = kmeans.labels_

# 4. Salvar o Modelo e o Scaler
# Salva o modelo e o normalizador para uso futuro (se necessário)
joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nModelo KMeans e Scaler salvos!")

# 5. Exibição dos Resultados
print(f"\nAnálise completa! As músicas foram agrupadas em {n_clusters} padrões (clusters):\n")

# Mostra quais músicas caíram em cada cluster
for i in range(n_clusters):
    print(f"--- Cluster {i} ---")
    songs_in_cluster = df[df['cluster'] == i]['filename'].tolist()
    for song in songs_in_cluster:
        print(f"- {song}")
    print()
