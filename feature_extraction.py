
import pandas as pd
import numpy as np
import librosa
import os
from pathlib import Path

# Todas as músicas agora ficam em uma única pasta
MUSIC_PATH = Path("data/")
OUTPUT_CSV = "features.csv"

def extract_features(file_path):
    """
    Extrai um vetor de características de um único arquivo de áudio.
    """
    try:
        # Carrega o arquivo de áudio. `sr=None` preserva a taxa de amostragem original,
        # mas para consistência, é melhor usar uma taxa fixa, como 22050.
        y, sr = librosa.load(file_path, sr=22050)

        # Extrai as características
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        # Para cada característica, calcula a média e o desvio padrão.
        # Isso cria um vetor de tamanho fixo para cada música.
        features = {
            'mfcc_mean': np.mean(mfccs),
            'mfcc_std': np.std(mfccs),
            'chroma_mean': np.mean(chroma),
            'chroma_std': np.std(chroma),
            'spectral_centroid_mean': np.mean(spectral_centroid),
            'spectral_centroid_std': np.std(spectral_centroid),
            'zero_crossing_rate_mean': np.mean(zero_crossing_rate),
            'zero_crossing_rate_std': np.std(zero_crossing_rate),
            'tempo': np.mean(tempo) # Garante que o tempo seja um número único
        }
        return features

    except Exception as e:
        print(f"Erro ao processar {file_path}: {e}")
        return None

def main():
    """
    Função principal para percorrer a pasta, extrair características e salvar em um CSV.
    """
    all_features = []

    print("Processando músicas...")
    # O .glob agora procura por todos os arquivos de áudio na pasta 'data' e subpastas
    for file_path in MUSIC_PATH.glob('**/*.mp3'): # ou .wav, etc.
        print(f"Extraindo de: {file_path.name}")
        features = extract_features(file_path)
        if features:
            # Adicionamos o nome do arquivo para identificação
            features['filename'] = file_path.name
            all_features.append(features)

    if not all_features:
        print("Nenhuma característica foi extraída. Verifique o caminho e os arquivos de áudio.")
        return

    # Cria um DataFrame e salva em CSV
    df = pd.DataFrame(all_features)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nCaracterísticas salvas com sucesso em '{OUTPUT_CSV}'!")
    print(df.head())

if __name__ == "__main__":
    main()
