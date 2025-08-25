import librosa
import numpy as np
import pandas as pd
import joblib
import argparse
from pathlib import Path

# Esta função é uma cópia da que está em feature_extraction.py
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        features = {
            'mfcc_mean': np.mean(mfccs),
            'mfcc_std': np.std(mfccs),
            'chroma_mean': np.mean(chroma),
            'chroma_std': np.std(chroma),
            'spectral_centroid_mean': np.mean(spectral_centroid),
            'spectral_centroid_std': np.std(spectral_centroid),
            'zero_crossing_rate_mean': np.mean(zero_crossing_rate),
            'zero_crossing_rate_std': np.std(zero_crossing_rate),
            'tempo': np.mean(tempo)
        }
        return features
    except Exception as e:
        print(f"Erro ao processar {file_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Descobre a qual padrão (cluster) uma nova música pertence.")
    parser.add_argument("audio_file", type=Path, help="Caminho para o arquivo de áudio a ser analisado.")
    args = parser.parse_args()

    # 1. Carregar os modelos treinados no PC
    try:
        kmeans_model = joblib.load('kmeans_model.pkl')
        scaler = joblib.load('scaler.pkl')
    except FileNotFoundError:
        print("Erro: Arquivos 'kmeans_model.pkl' ou 'scaler.pkl' não encontrados.")
        print("Execute 'analyze_patterns.py' no seu PC e copie os arquivos .pkl para o Pi.")
        return

    # 2. Extrair características da nova música
    print(f"Analisando '{args.audio_file.name}'...")
    new_features = extract_features(args.audio_file)
    if new_features is None:
        return

    # 3. Preparar os dados para o modelo
    features_df = pd.DataFrame([new_features])
    # Aplicar a mesma normalização usada no treinamento
    features_scaled = scaler.transform(features_df)

    # 4. Prever a qual cluster a música pertence
    cluster_prediction = kmeans_model.predict(features_scaled)

    print("\n--- Resultado da Análise ---")
    print(f"A música '{args.audio_file.name}' pertence ao Padrão (Cluster) nº: {cluster_prediction[0]}")
    print("----------------------------")

if __name__ == "__main__":
    main()