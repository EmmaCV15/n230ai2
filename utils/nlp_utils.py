


import re
import spacy
from spacy.tokens import Token
import nltk
from nltk.corpus import stopwords
import string
import csv
from typing import List, Dict


# NLP SPACY MODEL
# --------------------------------------------------------
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
# Retrieve the list of Spanish stopwords
stopwords_list = sorted([word.lower() for word in stopwords.words('spanish')])
punct_list = string.punctuation


# Load Spanish model from spaCy - load it once globally
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

# # BERT - BETO MODEL FOR NER AND POS
# # --------------------------------------------------------
# # NER Named Entity Recognition pipeline
# ner_model_name = 'mrm8488/bert-spanish-cased-finetuned-ner'
# ner_tokenizer = (ner_model_name,
#                  {"aggregation_strategy": "first",
#                   "grouped_entities": True}
#                  )
# ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name)
# ner_pipeline = pipeline("ner",
#                         model=ner_model,
#                         tokenizer=ner_tokenizer,
#                         device_map="auto",
#                         )
# # POS Part of Speech pipeline
# pos_model_name = 'mrm8488/bert-spanish-cased-finetuned-pos'
# pos_tokenizer = (pos_model_name, {"aggregation_strategy": 'first'})
# pos_pipeline = pipeline(task="ner",     # keep "ner" task
#                         model=pos_model_name,
#                         tokenizer=pos_tokenizer,
#                         )

# Read Text File
def read_file(text_path: str) -> str:
    """Read and return the content of a file.

    Args:
        text_path (str): The path to the text file.

    Returns:
        str: The content of the file.
    """
    try:
        with open(text_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"The file does not exist at the specified path: {text_path}")
        return ''
    except Exception as e:
        print(f"Error reading file: {e}")
        return ''

# Read CSV File
def read_samples(samples_path: str, category_name: str) -> List[str]:
    """Read samples from CSV file for a specific category.

    Args:
        samples_path (str): The path to the CSV file containing samples.
        category_name (str): The category to filter samples by.

    Returns:
        List[str]: A list of sample texts for the specified category.
    """
    try:
        with open(samples_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)

            return [row.get("text", "").strip() for row in reader if
                    row.get("category", "").strip().lower() == category_name]

    except Exception as e:
        print(f"Error reading file: {e}")
        return []

# Clean texts
def clean_text(input_text: str) -> str:
    """
    Limpia el texto de entrada eliminando stopwords, puntuación, dígitos y espacios.
    Devuelve un string.

    Args:
        input_text (str): Texto de entrada.

    Returns:
        output_text (str): Texto de salida.
    """

    text = input_text

    # Replace multiple line breaks to only one line break
    text = re.sub(r'\n{2,}', '\n', text)
    # Split text to lines
    lines = text.splitlines()

    clean_lines = []
    for line in lines:
        # Replace multiple hyphens and tabs to space
        line = re.sub(r'[-\t]{2,}', ' ', line)
        # Replace multiple white spaces to only one space
        line = re.sub(r'\s{2,}',  ' ', line)
        line = re.sub(r'^\s+|\s+$', '', line)
        # Join split words when there is a space, hyphen, colon, or special characters between them.
        pattern = r'(^[\W\s]|[\W\s]|^)([a-zA-ZáéíóúÁÉÍÓÚ])\s+([a-zA-ZáéíóúÁÉÍÓÚ])(\$|\W)'
        # Continue joining until no more matches are found
        while re.search(pattern, line):
            line = re.sub(pattern, lambda m: m.group(2) + m.group(3), line)
        clean_lines.append(line)

    text = '\n'.join(clean_lines)

    pattern = r'(?<![.,;!?:])\n(?!([.,;!?:]|[A-Za-z0-9]+[.\-\)]+))'
    text = re.sub(pattern, ' ', text)

    # Join lines if they end with a semicolon or comma, and the next word starts with a lowercase letter.
    text = re.sub(r'[,;]\n([a-záéíóúñ]+)', r', \1', text)

    return text

# Split text in chunks
def chunking(text, max_tokens=50):
    # Expresión regular: divide solo si el punto (.) está seguido de espacio o salto de línea
    delimiters = r'(?<=[.:;,”])(?=\s|\n)'
    sentences = re.split(delimiters, text)

    chunks = []

    current_chunk = []
    for sentence in sentences:
        words = sentence.strip().split()

        if len(words) > max_tokens:
            # Si la oración sola supera el límite, la dividimos en partes de 50 palabras
            sub_chunks = [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
            chunks.extend(sub_chunks)
        else:
            temp_chunk = current_chunk + words
            if len(temp_chunk) <= max_tokens:
                current_chunk = temp_chunk
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = words

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# def main():
#     # Define paths
#     text_path = '/home/n230ai/Descargas/prueba.txt'
#     samples_path = "/home/n230ai/Documentos/aplicaciones/n230ai_ocr_v2/knowledge/samples.csv"
#
#     categories = ['propiedad'] #,'propiedad','transmisión','precio','generales']
#
#     # Make corpus
#     # --------------------------------------------------------
#     corpus = []
#     for category in categories:
#
#         # Read samples
#         # --------------------------------------------------------
#         samples = [sample for category in categories for sample in read_samples(samples_path=samples_path, category_name=category)]
#
#         # Clean Samples
#         # --------------------------------------------------------
#         clean_samples = []
#         for sample in samples:
#             clean = clean_text(sample)
#             clean_samples.append(clean)
#
#         # Split text in chunks
#         # --------------------------------------------------------
#         for sample in samples:
#             print(sample)
#             print()
#             chunks = chunking(sample)
#             corpus.extend(chunks)
#
#
#     # Get topics
#     # --------------------------------------------------------
#     vectorizer_model = CountVectorizer(
#         ngram_range=(1, 2),
#         stop_words=None
#     )
#
#     # we add this to remove stopwords
#     model = BERTopic(
#         vectorizer_model=vectorizer_model,
#         language='spanish',
#         calculate_probabilities=True,
#         verbose=True
#     )
#
#     topics, probs = model.fit_transform(corpus)
#     freq = model.get_topic_info()
#     print(freq[:10])
