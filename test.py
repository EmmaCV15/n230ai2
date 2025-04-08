import numpy as np
import pandas as pd
import logging
from transformers import AutoTokenizer, AutoModel
import torch
import time
import re
import string
import csv
from typing import List, Dict

import spacy
from spacy.tokens import Token
nlp = spacy.load("es_core_news_lg")
# Define SpaCy extensiones
if not Token.has_extension("orig_idx"):
    Token.set_extension("orig_idx", getter=lambda token: [token.idx, token.idx + len(token.text)])

# Remove this from stopword list
nlp.Defaults.stop_words -= {
    "un",
    "uno","dos","tres","cuatro","cinco","seis","siete","ocho","nueve",
    "diez","once","doce",
    "hago", "dicho", "consecuente"
}

# Append this to stopword list
nlp.Defaults.stop_words |= {
    "señor","señores",
    "señora","señoras",
    "señorita","señoritas",
    "don", "doña",
    "licenciado","licenciada",
    "licenciados","licenciadas",
    "biólogo","biólogos",
    "bióloga","biólogas",
    "ingeniero","ingenieros",
    "ingeniera","ingenieras",
    "arquitecto","arquitectos",
    "arquitecta","arquitectas",
}

if not Token.has_extension("is_toi"):      # is_toi - > is token of interest
    Token.set_extension("is_toi", getter=lambda token: True if token.is_alpha and not token.is_stop and not token.is_space else False)

class SyntheticSimilarityData:
    def __init__(self, num_ejemplos=1000, num_comparaciones=5, max_length=100):
        """
        Generador de datos sintéticos basados en similitudes.

        Args:
            num_ejemplos (int): Número de registros sintéticos a generar.
            num_comparaciones (int): Número de ejemplos de referencia para las similitudes.
            max_length (int): Máxima posición posible para `start` y `end`.
        """
        self.num_ejemplos = num_ejemplos
        self.num_comparaciones = num_comparaciones
        self.max_length = max_length

    def generate_synthetic_data(self):
        """Genera el DataFrame con datos sintéticos."""
        data = {
            "chunks_sims": [],
            "start_sims": [],
            "end_sims": [],
            "start": [],
            "end": []
        }

        for _ in range(self.num_ejemplos):
            # Similitudes sintéticas (valores entre 0 y 1)
            chunks_sims = np.round(np.random.beta(a=2, b=5, size=self.num_comparaciones), 3).tolist()
            start_sims = np.round(np.random.beta(a=5, b=2, size=self.num_comparaciones), 3).tolist()
            end_sims = np.round(np.random.beta(a=3, b=3, size=self.num_comparaciones), 3).tolist()

            # Posiciones (start y end)
            start = int(np.clip(np.mean(start_sims) * self.max_length, 0, self.max_length))
            end = int(np.clip(start + np.random.randint(5, 20), start + 1, self.max_length))

            # Almacenar en el diccionario
            data["chunks_sims"].append(chunks_sims)
            data["start_sims"].append(start_sims)
            data["end_sims"].append(end_sims)
            data["start"].append(start)
            data["end"].append(end)

        return pd.DataFrame(data)

    def save_to_csv(self, output_path):
        """Guarda los datos en un archivo CSV."""
        df = self.generate_synthetic_data()
        df.to_csv(output_path, index=False)
        logging.info(f"✅ Datos sintéticos guardados en: {output_path}")


# Uso del generador
if __name__ == "__main__":
    generator = SyntheticSimilarityData(num_ejemplos=1000, num_comparaciones=5)
    generator.save_to_csv("synthetic_similarity_data.csv")