import re
import time
import csv
import torch
import numpy as np
import spacy
from spacy.tokens import Doc
import ollama
from multiprocessing import Pool, cpu_count
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
from scipy.signal import find_peaks




# Custom Functions
from utils.ollama_utils import start_ollama_server, kill_ollama_server, run_ollama_model
from utils.nlp_utils import read_file, clean_text, nlp, read_samples


def split_to_chunks(text: str, window_length: int=30, overlap: int=0) -> spacy.tokens.Doc:
    """Split text into chunks of specified length with overlap, preserving meaning."""

    if not text or window_length <= 0 or overlap < 0 or overlap >= window_length:
        raise ValueError(
            "Invalid parameters: window_length must be > 0, and overlap must be non-negative and < window_length.")

    # Split text into lines
    lines = text.split('\n')

    # Initialize variables
    chunks = []
    chunk = []
    current_length = 0

    def add_chunk():
        """Add current chunk to chunks list and reset it, applying overlap."""
        nonlocal chunk, current_length
        if chunk:
            chunks.append(" ".join(chunk))
            chunk = chunk[-overlap:] if overlap > 0 else []  # Apply overlap
            current_length = sum(len(w.split()) for w in chunk)

    for line in lines:

        sentences = re.split(r'(?<=[.:;])(?=\s|\n)', line)  # Split while keeping punctuation

        for sentence in sentences:

            words = [word for word in re.split(r'(\S+)',sentence) if word.strip() != ""]
            sentence_length = len(words)

            # If sentence fits in current chunk, add it
            if current_length + sentence_length <= window_length:
                chunk.extend(words)
                current_length += sentence_length
            else:
                add_chunk()  # Store previous chunk and apply overlap
                if sentence_length <= window_length:
                    chunk = words
                    current_length = sentence_length
                else:
                    # Split long sentences into smaller chunks
                    for i in range(0, sentence_length, window_length):
                        chunks.append(" ".join(words[i:i + window_length]))

    add_chunk()  # Add any remaining chunk

    # Convert to spaCy Doc with custom chunks attribute
    doc = nlp(text)
    doc.set_extension("chunks", default=[], force=True)
    doc._.chunks = chunks

    return doc

def update_similarity(text_part:str, category_vectors:list):

    ############################################
    # Handle the text part
    ############################################
    cleaned_text = clean_text(text_part)
    doc = split_to_chunks(text=cleaned_text)
    doc_chunk_vectors = []
    for chunk in doc._.chunks:
        chunk_doc = nlp(chunk)
        vec = chunk_doc.vector
        norm = np.linalg.norm(vec)
        doc_chunk_vectors.append(vec / norm if norm != 0 else vec)

    ############################################
    # Calculate similarities
    ############################################
    # Cálculo paralelizado
    with Pool(
            processes=cpu_count(),
            initializer=worker_init,
            initargs=(category_vectors,)
    ) as pool:
        chunks_similarities = pool.map(calculate_max_similarity, doc_chunk_vectors)

    mean_similarity = np.mean(chunks_similarities)

    return mean_similarity

def calculate_max_similarity(doc_vec):
    global global_sample_vectors
    if not global_sample_vectors.size:
        return 0.0
    similarities = np.dot(global_sample_vectors, doc_vec)
    return np.max(similarities) if similarities.size > 0 else 0.0

def worker_init(sample_vectors):
    global global_sample_vectors
    global_sample_vectors = np.array(sample_vectors)

def calculate_coefficients_regression():
    # -----------------------------------------------------------
    import statsmodels.api as sm
    import pandas as pd
    data = {
        'best': [0.9999999403953550, 1, 0.990976929664612, 0.992049515247345, 1],
        'mean': [0.917376577854156, 0.956744313240051, 0.86527019739151, 0.886868238449097, 0.987929701805115],
        'std': [0.060842722654343, 0.074920885264874, 0.091144882142544, 0.073685295879841, 0.038169369101524],
        'mean_std': [15.0778357350298, 12.7700615103199, 9.49334923751716, 12.0358916641296, 25.8827883473102],
        'target': [0.920921041729794, 0.838572229290785, 0.890256350417794, 0.929040508833678, 0.933613628149033]
    }
    # Create DataFrame
    df = pd.DataFrame(data)

    # Define the independent variables (add constant for intercept)
    X = df[['best', 'mean', 'std', 'mean_std']]
    X = sm.add_constant(X)  # Adds the intercept

    # Define the dependent variable
    y = df['target']

    # Fit the regression model
    model = sm.OLS(y, X).fit()
    print(model.summary())

def preprocess(text_path:str, knowledge:dict):

    ######################################
    start = time.perf_counter()
    print(f'\tA).- Handling Input Text....')
    print('\t1. Read file...')
    original_text = read_file(text_path)
    print('\t2. Clean text...')
    cleaned_text = clean_text(original_text)
    print('\t3. Split to chunks...')
    doc = split_to_chunks(text=cleaned_text)
    print('\t4. Vectorize chunks...')
    doc_chunk_vectors = []
    for chunk in doc._.chunks:
        chunk_doc = nlp(chunk)
        vec = chunk_doc.vector
        norm = np.linalg.norm(vec)
        doc_chunk_vectors.append(vec / norm if norm != 0 else vec)
    end = time.perf_counter()
    total_time = end - start
    print(f'\t5. Total vectors: {len(doc_chunk_vectors)} Total time: {total_time:.2f} seg.')
    print(f'\t{"-" * 100}')
    ############################################################################

    ############################################################################
    print(f'\tB).- Handling Part Samples....')
    print(f'\t{"-" * 100}')

    dictionary = {'task': knowledge['task']}
    parts = knowledge['parts']

    parts_times = []
    parts_chunks = []
    total_samples = 0

    for idx, part in enumerate(parts):
        print(f"\tPart {part}...")
        print(f"\t{'-'*100}")

        # Create dictionary for this part
        dictionary[part] = {
            'samples': [],
            'vectors': [],
            'row_segment': '',
            'refined_segment':'',
            'text': '',
            'similarity': 0,
        }

        start = time.perf_counter()

        print('\t1. Read samples...')
        samples = knowledge['parts_samples'].loc[knowledge['parts_samples']['part'] == part, 'sample'].tolist()
        total_samples += len(samples)
        print(f'\t2. Clean {len(samples)} samples...')
        samples = [clean_text(sample) for sample in samples]
        print('\t3. Split to chunks...')
        # Creación de chunks y vectores de muestras
        samples_chunks = []
        for sample in samples:
            sample_doc = split_to_chunks(text=sample)
            samples_chunks.extend(sample_doc._.chunks)
        print(f'\t4. Vectorize {len(samples_chunks)} chunks...')
        sample_vectors = []
        for chunk in samples_chunks:
            chunk_doc = nlp(chunk)
            vec = chunk_doc.vector
            norm = np.linalg.norm(vec)
            sample_vectors.append(vec / norm if norm != 0 else vec)

        dictionary[part]['samples'].extend(samples_chunks)
        dictionary[part]['vectors'].extend(sample_vectors)

        end = time.perf_counter()
        part_time = end - start
        print(f'\t5. Total vectors: {len(sample_vectors)} from {len(samples)} samples. Time: {part_time:.2f} seg.')

        parts_times.append(part_time)
        parts_chunks.append(len(samples_chunks))


    # Cálculo de similitudes

    print(f'\tC).-  Calculating similarities...')
    print(f'\t{"-" * 100}')

    for idx, part in enumerate(parts):

        start = time.perf_counter()
        part_vectors = dictionary[part]['vectors']
        samples_chunks = dictionary[part]['samples']

        print(f'\tPart {part}...')
        print(f'\t{"-" * 100}')

        with Pool(
                processes=cpu_count(),
                initializer=worker_init,
                initargs=(part_vectors,)
        ) as pool:
            chunks_similarities = pool.map(calculate_max_similarity, doc_chunk_vectors)

        end = time.perf_counter()
        part_time = end - start

        # Análisis de resultados
        best_similarity = np.max(chunks_similarities)
        best_similar_index = np.argmax(chunks_similarities)

        # Adjacent chunks
        content_range = 5
        center_index = best_similar_index
        start_index = max(0, center_index - content_range)
        end_index = min(len(chunks_similarities), center_index + content_range + 1)
        samples_group = chunks_similarities[start_index:end_index]
        mean_content = np.mean(samples_group)
        std_content = np.std(samples_group)

        print('\tContent: ', samples_group)

        ### DO NOT CHANGE ANY VALUE ###
        # -----------------------------------------------------------
        min_similarity_accepted = 5.180 + (-3.2456 * best_similarity) + (-0.8536 * mean_content) + (-3.5914 * std_content) + (-0.0008 * (mean_content / std_content))
        # min_similarity_accepted = 0.9557

        print(f'\tSimilarities: {chunks_similarities}')
        print(f'\tBest Similarity: {best_similarity}')
        print(f'\tMean Content: {mean_content}')
        print(f'\tStd Content: {std_content}')
        print(f'\tMean / Std: {mean_content / std_content}')
        print(f'\tMin Accepted: {min_similarity_accepted}')
        print(f'\t{best_similarity}, {mean_content}, {std_content}, {mean_content/std_content}, {min_similarity_accepted}')


        if best_similarity >= 0.0:
            adjacent_chunks = []
            current_index = best_similar_index

            # Búsqueda hacia atrás
            # -----------------------------------------------------------
            while current_index > 0:
                current_index -= 1
                if chunks_similarities[current_index] >= min_similarity_accepted:
                    adjacent_chunks.insert(0, (current_index, chunks_similarities[current_index]))
                else:
                    break

            # Añadir chunk principal
            # -----------------------------------------------------------
            adjacent_chunks.append((best_similar_index, best_similarity))

            # Búsqueda hacia adelante
            # -----------------------------------------------------------
            current_index = best_similar_index + 1
            while current_index < len(chunks_similarities):
                if chunks_similarities[current_index] >= min_similarity_accepted:
                    adjacent_chunks.append((current_index, chunks_similarities[current_index]))
                    current_index += 1
                else:
                    break

            # Generar segmento final
            start_chunk = max(0, adjacent_chunks[0][0])
            end_chunk = min(adjacent_chunks[-1][0] + 1, len(chunks_similarities))
            row_segment = ' '.join(doc._.chunks[start_chunk:end_chunk])

            # Formatear salida
            print('\tstart: ', start_chunk, 'end: ', end_chunk)
            print(f'\tRow segment:')
            words = [word for word in re.split(r'(\S+)', row_segment) if word.strip()]
            for j in range(0, len(words), 18):
                print(f'\t' + " ".join(words[j:j + 18]))

            print(f'\tTime: {part_time:.2f} seg.')
            print(f'\t{"-" * 100}')

            dictionary[part]['row_segment'] = row_segment

    return dictionary

def detect_segment(similarities, prominence=0.001):

    # Compute bins count and prominence dynamically
    bin_count = max(3, int(np.sqrt(len(similarities))))

    similarities = np.array(similarities)
    max_idx = np.argmax(similarities)
    max_value = similarities[max_idx]

    # Construcción del histograma
    hist, bin_edges = np.histogram(similarities, bins=bin_count)

    # Normalización del histograma
    hist = hist / np.max(hist)

    # Detección de picos y valles
    peaks, _ = find_peaks(similarities, prominence=prominence)
    valleys, _ = find_peaks(-similarities, prominence=prominence)

    # print(f'similarities: {similarities}')
    # print(f'peaks: {peaks}')
    # print(f'valleys: {valleys}')


    # Filtrar picos relevantes (los más cercanos a max_idx)
    relevant_peaks = [p for p in peaks if p <= max_idx]
    relevant_valleys = [v for v in valleys if v >= max_idx]

    start_idx = relevant_peaks[-1] if relevant_peaks else 0
    end_idx = relevant_valleys[0] if relevant_valleys else len(similarities) - 1

    # Visualización opcional
    # Visualización opcional
    plt.figure(figsize=(12, 5))
    plt.plot(similarities, label='Similitudes')
    plt.axvline(start_idx, color='g', linestyle='--', label=f'Inicio: {start_idx}')
    plt.axvline(end_idx, color='r', linestyle='--', label=f'Fin: {end_idx}')
    plt.axvline(max_idx, color='b', linestyle='--', label=f'Máxima: {max_idx}')
    plt.scatter([start_idx, end_idx, max_idx], [similarities[start_idx], similarities[end_idx], similarities[max_idx]],
                color=['g', 'r', 'b'], zorder=3)
    plt.text(start_idx, similarities[start_idx], f'Idx: {start_idx}\nVal: {similarities[start_idx]:.3f}',
             verticalalignment='bottom', horizontalalignment='right', fontsize=10, color='g')
    plt.text(end_idx, similarities[end_idx], f'Idx: {end_idx}\nVal: {similarities[end_idx]:.3f}',
             verticalalignment='bottom', horizontalalignment='left', fontsize=10, color='r')
    plt.text(max_idx, similarities[max_idx], f'Idx: {max_idx}\nVal: {similarities[max_idx]:.3f}',
             verticalalignment='bottom', horizontalalignment='center', fontsize=10, color='b')
    plt.legend()
    plt.show()
    start_idx, end_idx = 4, 77
    return start_idx, end_idx, max_idx

def calculate_intersections(similarities_series:list) -> list:

    num_series = len(similarities_series)
    num_points = len(similarities_series[0])
    intersections = []

    # Comparar cada par de series
    for i in range(num_series):
        for j in range(i + 1, num_series):
            for k in range(num_points - 1):
                # Verificar si las series se cruzan entre los puntos k y k+1
                s1_start, s1_end = similarities_series[i][k], similarities_series[i][k + 1]
                s2_start, s2_end = similarities_series[j][k], similarities_series[j][k + 1]

                if (s1_start > s2_start and s1_end < s2_end) or (s1_start < s2_start and s1_end > s2_end):
                    # Interpolación lineal para encontrar el punto de intersección
                    x1, x2 = k, k + 1
                    intersection_x = x1 + (x2 - x1) * (s2_start - s1_start) / ((s1_end - s1_start) - (s2_end - s2_start))
                    intersection_y = s1_start + (intersection_x - x1) * (s1_end - s1_start) / (x2 - x1)
                    intersections.append((i + 1, j + 1, intersection_x, intersection_y))

    return intersections

def preprocess_2(text_path: str, knowledge: dict):

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')
    model = AutoModel.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    ############################################################################
    # A). Handle Input Text
    ############################################################################

    start = time.perf_counter()
    print(f'\tA).- Handling Input Text....')
    print('\t1. Read file...')
    original_text = read_file(text_path)
    print('\t2. Clean text...')
    cleaned_text = clean_text(original_text)
    print('\t3. Split to chunks...')
    doc = split_to_chunks(text=cleaned_text)
    # Generate embeddings for input text chunks
    print('\t4. Vectorize chunks...')
    batch_size = 10
    inputs = {'input_ids': [], 'attention_mask': []}
    all_embeddings = []

    for i, chunk in enumerate(doc._.chunks):

        # Tokenize the chunk
        new_tokens = tokenizer.encode_plus(
            chunk,
            max_length=512,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        inputs['input_ids'].append(new_tokens['input_ids'][0])
        inputs['attention_mask'].append(new_tokens['attention_mask'][0])

        # Process batch if batch size is reached or it's the last chunk
        if (i + 1) % batch_size == 0 or i == len(doc._.chunks) - 1:
            batch_inputs = {
                'input_ids': torch.stack(inputs['input_ids']).to(device),
                'attention_mask': torch.stack(inputs['attention_mask']).to(device)
            }

            # Perform inference
            with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(**batch_inputs)

            # Extract embeddings and apply attention mask
            embeddings = outputs.last_hidden_state
            attention_mask = batch_inputs['attention_mask']
            mask = attention_mask.unsqueeze(-1).expand(embeddings.shape).float()
            mask_embeddings = embeddings * mask
            summed = torch.sum(mask_embeddings, dim=1)
            counts = torch.clamp(mask.sum(1), min=1e-9)
            mean_pooled = summed / counts

            all_embeddings.append(mean_pooled.cpu())
            inputs = {'input_ids': [], 'attention_mask': []}

    # Concatenate all embeddings
    doc_embeddings = torch.cat(all_embeddings) if all_embeddings else None
    end = time.perf_counter()
    total_time = end - start
    print(f'\t5. Total vectors: {len(doc_embeddings)} Total time: {total_time:.2f} seg.')
    print(f'\t{"-" * 100}')

    ############################################################################
    # B). Handle Part Samples - Process text data and generate embeddings
    ############################################################################

    print(f'\tB).- Handling Part Samples...')
    print(f'\t{"-" * 100}')

    # Initialize main dictionary with task information
    dictionary = {'task': knowledge['task']}
    # Get list of parts to process
    parts = knowledge['parts']

    def generate_embeddings(chunks_list, batch_size=8):
        """
        Generates embeddings for a list of text chunks.
        This function processes text chunks in batches and converts them to vector representations
        using a transformer model.

        Args:
            chunks_list: List of text chunks to convert to embeddings
            batch_size: Number of chunks to process at once (affects memory usage)

        Returns:
            Tensor with the generated embeddings, one vector per input chunk
        """
        # Initialize containers for inputs and results
        inputs = {'input_ids': [], 'attention_mask': []}
        all_embeddings = []

        # Process each chunk one by one
        for j, chunk in enumerate(chunks_list):
            # Tokenize the text chunk and prepare it for the model
            new_tokens = tokenizer.encode_plus(
                chunk,
                max_length=512,  # Limit to model's maximum context window
                truncation=True,  # Cut text if it's too long
                padding='max_length',  # Pad to full length for batch processing
                return_tensors='pt'  # Return PyTorch tensors
            )

            # Collect tokenized inputs for batch processing
            inputs['input_ids'].append(new_tokens['input_ids'][0])
            inputs['attention_mask'].append(new_tokens['attention_mask'][0])

            # When we've collected a full batch or reached the end of the list, process the batch
            if (j + 1) % batch_size == 0 or j == len(chunks_list) - 1:
                # Prepare batch by stacking individual inputs and moving to GPU
                batch_inputs = {
                    'input_ids': torch.stack(inputs['input_ids']).to(device),
                    'attention_mask': torch.stack(inputs['attention_mask']).to(device)
                }

                # Run the model with optimizations (no gradients, mixed precision)
                with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(**batch_inputs)

                # Extract embeddings from model output
                embeddings = outputs.last_hidden_state
                attention_mask = batch_inputs['attention_mask']

                # Apply attention mask to focus only on non-padding tokens
                mask = attention_mask.unsqueeze(-1).expand(embeddings.shape).float()
                mask_embeddings = embeddings * mask

                # Perform mean pooling to get one vector per chunk
                summed = torch.sum(mask_embeddings, dim=1)
                counts = torch.clamp(mask.sum(1), min=1e-9)  # Avoid division by zero
                mean_pooled = summed / counts

                # Move results to CPU and store them
                all_embeddings.append(mean_pooled.cpu())
                # Reset input containers for next batch
                inputs = {'input_ids': [], 'attention_mask': []}

        # Concatenate all batch results into a single tensor
        # Return None if no embeddings were generated
        return torch.cat(all_embeddings) if all_embeddings else None

    # Process each part separately
    for part in parts:
        print(f"\tPart {part}...")
        print(f"\t{'-' * 100}")

        # Initialize dictionary structure for this part's results
        dictionary[part] = {
            'samples': [],  # Will store the text chunks
            'vectors': [],  # Will store embeddings for all chunks
            'start_vectors': [],  # Will store embeddings for first chunks of each sample
            'end_vectors': [],  # Will store embeddings for last chunks of each sample
            'chunk_to_sample_map': [],  # Map to track which sample each chunk belongs to
            'sample_boundaries': [],  # Store indices where each sample begins and ends
            'row_segment': '',  # Will store raw extracted text segment
            'refined_segment': '',  # Will store processed/refined text segment
            'text': '',  # Will store final text
            'similarity': 0,  # Will store similarity score
        }

        # Start timing the processing for this part
        start_time = time.perf_counter()

        # 1. Read and process samples from the knowledge database
        # Filter samples that belong to the current part
        samples = knowledge['parts_samples'].loc[knowledge['parts_samples']['part'] == part, 'sample'].tolist()
        print(f'\t1. Read {len(samples)} samples...')

        # 2. Clean the text samples to remove noise and normalize
        samples = [clean_text(sample) for sample in samples]
        print(f'\t2. Cleaned {len(samples)} samples...')

        # 3. Split samples into smaller chunks for processing
        # Initialize lists to store different types of chunks
        samples_chunks = []  # All chunks from all samples
        start_chunks = []  # First chunk from each sample (for identifying start boundaries)
        end_chunks = []  # Last chunk from each sample (for identifying end boundaries)
        chunk_to_sample_map = []  # Track which sample each chunk belongs to
        sample_boundaries = []  # Store start and end indices for each sample

        # Process each sample
        current_chunk_index = 0
        for sample_idx, sample in enumerate(samples):
            # Split each sample into chunks with specified window size
            sample_doc = split_to_chunks(text=sample)

            # Verify that sample_doc has chunks before accessing them (avoid errors)
            if hasattr(sample_doc, '_') and hasattr(sample_doc._, 'chunks') and len(sample_doc._.chunks) > 0:
                # Record the start index for this sample
                start_index = current_chunk_index

                # Add all chunks to main list and track their source sample
                for chunk in sample_doc._.chunks:
                    samples_chunks.append(chunk)
                    chunk_to_sample_map.append(sample_idx)
                    current_chunk_index += 1

                # Record the end index for this sample
                end_index = current_chunk_index - 1
                sample_boundaries.append((start_index, end_index))

                # Add only first chunk to start list
                start_chunks.append(sample_doc._.chunks[0])
                # Add only last chunk to end list
                end_chunks.append(sample_doc._.chunks[-1])

        print(f'\t3. Split into {len(samples_chunks)} chunks from {len(samples)} samples...')

        # 4. Generate vector embeddings for all chunk types
        print(f'\t4. Generating embeddings...')

        batch_size = 8  # Process 8 chunks at a time for efficiency

        # Generate embeddings for all three types of chunks
        part_embeddings = generate_embeddings(samples_chunks, batch_size)  # All chunks
        start_embeddings = generate_embeddings(start_chunks, batch_size)  # First chunks only
        end_embeddings = generate_embeddings(end_chunks, batch_size)  # Last chunks only

        # Safely get the number of embeddings, handling the case where embeddings might be None
        part_size = part_embeddings.shape[0] if part_embeddings is not None else 0
        start_size = start_embeddings.shape[0] if start_embeddings is not None else 0
        end_size = end_embeddings.shape[0] if end_embeddings is not None else 0

        # Report the number of embeddings generated for each type
        print(f'\t   Generated {part_size} sample embeddings')
        print(f'\t   Generated {start_size} start embeddings')
        print(f'\t   Generated {end_size} end embeddings')

        # 5. Store all results in the dictionary for later use
        dictionary[part]['samples'] = samples_chunks
        dictionary[part]['vectors'] = part_embeddings
        dictionary[part]['start_vectors'] = start_embeddings
        dictionary[part]['end_vectors'] = end_embeddings
        dictionary[part]['chunk_to_sample_map'] = chunk_to_sample_map
        dictionary[part]['sample_boundaries'] = sample_boundaries

        # Calculate and report the total processing time for this part
        end_time = time.perf_counter()
        total_time = end_time - start_time

        print(
            f'\t5. Total vectors: Part {part_size}, Start vectors {start_size}, End vectors {end_size}. Total time: {total_time:.2f} seg.')
        print(f'\t{"-" * 100}')

    ############################################################################
    # C). Calculate Similarities and Find Boundaries
    ############################################################################

    print(f'\tC).- Calculating Similarities....')
    print(f'\t{"-" * 100}')

    # Initialize dictionary to store similarity results
    similarity_results = {}
    parts = knowledge['parts']

    def compute_similarities(query_embeddings, reference_embeddings):
        """
        Compute cosine similarity between two sets of embeddings.

        Args:
            query_embeddings: Embeddings of the input text chunks
            reference_embeddings: Embeddings to compare against

        Returns:
            Numpy array of similarity scores
        """
        # Check if embeddings exist
        if query_embeddings is None or reference_embeddings is None:
            return np.array([])

        # Compute cosine similarity matrix between all pairs
        similarity_matrix = cosine_similarity(query_embeddings, reference_embeddings)

        # For each query embedding, get its highest similarity score
        max_similarities = similarity_matrix.max(axis=1)
        # Normalize the similarity scores to be between 0 and 1
        if len(max_similarities) == 0:
            return np.array([])

        max_val = max_similarities.max()
        min_val = max_similarities.min()

        if max_val == min_val:
            # Avoid division by zero if all similarities are the same
            normalized_similarities = np.zeros_like(max_similarities)
        else:
            normalized_similarities = (max_similarities - min_val) / (max_val - min_val)

        return normalized_similarities


    def find_best_section(
            part_similarities,
            start_similarities,
            end_similarities,
    ):
        """
        Find the best contiguous section using combined scores from start, part, and end similarities.

        Args:
            part_similarities: Similarity scores for being part of the section (body).
            start_similarities: Similarity scores for section start boundaries.
            end_similarities: Similarity scores for section end boundaries.
            min_window: Minimum section length to consider.
            max_window: Maximum section length to consider.
            continuity_threshold: Minimum average similarity for adjacent intervals.
            part_weight: Weight for body (part) similarity scores.
            start_weight: Weight for start-boundary scores.
            end_weight: Weight for end-boundary scores.

        Returns:
            Tuple of (start_idx, end_idx, combined_score)
        """

        # Top Starts and End Similarities
        k = 10
        # Starts
        sorted_start_indexes_asc = np.argsort(start_similarities)
        sorted_start_indexes_desc = np.flip(sorted_start_indexes_asc)
        top_starts = sorted_start_indexes_desc[:k]
        # Ends
        sorted_end_indexes_asc = np.argsort(end_similarities)
        sorted_end_indexes_desc = np.flip(sorted_end_indexes_asc)
        top_ends = sorted_end_indexes_desc[:k]

        # Get the valid pairs start, end to evaluate
        start_grid, end_grid = np.meshgrid(top_starts, top_ends, indexing='ij')
        valid_mask = end_grid >= start_grid
        valid_pairs = np.vstack((start_grid[valid_mask], end_grid[valid_mask])).T
        print('valid_pairs:', len(valid_pairs))

        for window_start, window_end in valid_pairs:

            # Compute window length and similarity and variance
            win_length = (window_end - window_start + 1)
            win_similarity = np.mean(part_similarities[window_start:window_end + 1])
            win_variance = np.var(part_similarities[window_start:window_end + 1])

            # Filter Top 20 parts similarities inside window
            top_count = 20
            sorted_part_indexes_asc = np.argsort(part_similarities)
            sorted_part_indexes_desc = np.flip(sorted_part_indexes_asc)
            top_parts = sorted_part_indexes_desc[:top_count]
            inside_top_parts = [x for x in top_parts if window_start <= x <= window_end]
            inside_top_parts_density = len(inside_top_parts) / top_count if inside_top_parts else 0
            top_parts_window_len = len(inside_top_parts) / win_length if inside_top_parts else 0

            # Distances
            max_inside_part = np.max(inside_top_parts)
            # Short Distance from max top start to window start
            short_start_distance = max_inside_part - window_start
            # Short Distance from max top end to window ends
            short_end_distance = window_end - max_inside_part

            print('start:', window_start, 'end:', window_end, 'win_length:', win_length, 'max_inside_part', max_inside_part)
            print('short_start_distance', short_start_distance, 'short_end_distance', short_end_distance)
            print('window: ', part_similarities[window_start:window_end].tolist())
            print('win_similarity: ', win_similarity, 'win_variance: ', win_variance)
            print('inside_top_parts: ', len(inside_top_parts), inside_top_parts)
            print('inside_top_parts_density: ', inside_top_parts_density)
            print('top_parts_window_len: ', top_parts_window_len)
            print()


            # Iterate over the valid pairs
            # Initialize dictionary
            best_section = {
                'best_start': 0,
                'start_score': 0,
                'best_end': 0,
                'end_score': 0,
                'best_score': 0,

            }
            # Filter by percentile
            if win_similarity >= np.percentile(top_parts,90) and start_similarities[window_start] >= np.percentile(start_similarities,90) and end_similarities[window_end] >= np.percentile(start_similarities,90):
                    # End similarities
                    # Get short distance between end and max top part indexes within window.
                    short_distance = window_end - max(inside_top_parts)
                    # Compute quotients:
                    # Get quotient of the shortest distance between the end index and the maximum top-part index, divided by the window's length.
                    quotient_short_distance_window_len = 1 - short_distance / win_length

                    # Compute a score
                    # Weights:
                    w1 = 1
                    w2 = 1
                    w3 = 1
                    w4 = 1

                    score = (
                        # Quotient of top similarity parts within the window divided by the total top similarity parts.
                        w1 * (len(inside_top_parts) / top_count) +
                        # Quotient of top similarity parts within the window divided by the window's length.
                        w2 * (len(inside_top_parts) / win_length) +
                        # Quotient of the shortest distance between the end index and the maximum top-part index, divided by the window's length.
                        w3 * (1 - (short_distance / win_length)) +
                        # Penalty for window length
                        w4 * (1 / win_length)
                    )

                    print('window:')
                    print(f'start: {window_start} start sim: {start_similarities[window_start]:.4f}')
                    print(f'end:   {window_end}   end sim: {end_similarities[window_end]:.4f}')
                    print(f'win_similarity: {win_similarity:.4f}')
                    print(f'win_variance: {win_variance}')
                    print(f'win_length: {win_length}')
                    print(f'inside_top_parts: {len(inside_top_parts)} indexes: {inside_top_parts}')
                    print(f'top_density: {inside_top_parts_density}')
                    print(f'top_parts_window_len: {top_parts_window_len}')

                    print(f'short distance from {window_end} window end to {max(inside_top_parts)} max top part: {short_distance}')


                    print(f'quotient_short_distance_window_len: {quotient_short_distance_window_len}')
                    print(f'window penalty: {1 / win_length}')
                    print(f'score: {score}')
                    print()


            # Compute start_score and end_score with bounds checking
            if window_start >= len(start_similarities):
                continue
            start_score = start_similarities[window_start]

            if window_end >= len(end_similarities):
                continue
            end_score = end_similarities[window_end]

            # Compute combined score
            combined_score = np.mean(
                [
                    1 * win_similarity,
                    1 * start_score,
                    1 * end_score
                ]
            )

            # Best combination
            if combined_score >= best_section['best_score']:

                best_section['best_start'] = max(0, window_start)
                best_section['start_score'] = start_similarities[window_start]
                best_section['best_end'] = min(window_end, len(end_similarities))
                best_section['end_score'] = end_similarities[window_end]
                best_section['best_score'] = win_similarity

        print(list(best_section.items()))

        return best_section

    # Set a threshold for minimum similarity to be part of a section
    min_similarity_threshold = 0.5  # Adjust based on your data

    # First pass: Calculate similarities for all parts
    part_data = {}
    for part in parts:
        print(f'Calculating similarities for {part}.')
        if part not in ['proemio']:
            break

        # Get embeddings for this part
        part_vectors = dictionary[part]['vectors']
        start_vectors = dictionary[part]['start_vectors']
        end_vectors = dictionary[part]['end_vectors']

        # Compute similarities
        part_similarities = compute_similarities(doc_embeddings, part_vectors)
        start_similarities = compute_similarities(doc_embeddings, start_vectors)
        end_similarities = compute_similarities(doc_embeddings, end_vectors)


        # Find the best section based on average similarity
        best_section = find_best_section(
            part_similarities=part_similarities,
            start_similarities=start_similarities,
            end_similarities=end_similarities,
        )

        # Create segment from the identified boundaries
        segment_text = ' '.join(doc._.chunks[best_section['best_start']:best_section['best_end']+1])

        # Store all data for this part
        part_data[part] = {
            'part_similarities': part_similarities,
            'start_similarities': start_similarities,
            'end_similarities': end_similarities,
            'best_score': best_section['best_score'],
            'best_start': best_section['best_start'],
            'best_end': best_section['best_end'],
            'start_score': best_section['start_score'],
            'end_score': best_section['end_score']
        }

        print(f'\tPart: {part}')
        print(f'{"-" * 100}')
        print(f'\tpart_similarities:  {part_similarities.tolist()}')
        print(f'\tstart_similarities: {start_similarities.tolist()}')
        print(f'\tend_similarities:   {end_similarities.tolist()}')
        print(f'\tbest_score:         {part_data[part]['best_score']}')
        print(f'\tbest_start:         {part_data[part]['best_start']} {part_data[part]['start_score']}')
        print(f'\tbest_end:           {part_data[part]['best_end']} {part_data[part]['end_score']}')
        print(f'\tsegment_text:       {segment_text}')


def refine_segments(dictionary:dict):

    model_name = 'qwen2.5-max:14b'

    # Clean cache
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    # Start Ollama Server
    start_ollama_server()
    # Run Model
    run_ollama_model(model_name)
    # -------------------------------------------------------------
    # EXTRACT INSTRUMENT PART
    # -------------------------------------------------------------
    print(f'III.- Refining segments...')
    print(f'{"-" * 100}')

    # Get parts list
    parts_list = knowledge['parts']

    for part in parts_list:

        print()
        print(f'\tPart: {part}')
        print(f'\t{"-" * 100}')

        # Clean GPU caché
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        row_segment = dictionary[part]['row_segment']

        # Read Samples
        samples = dictionary[part]['samples']
        # Clean Samples
        samples = [clean_text(sample) for sample in samples]

        # Convert samples list to string
        samples_text = ''
        for idx, sample in enumerate(samples):
            samples_text += f'\n**EJEMPLO {idx}:**: {sample}'

        system_prompt = """
                Eres un especialista preciso en refinamiento, extracción y transcripción de textos en español.
        """

        user_prompt = f"""
                - CON BASE EN LOS SIGUIENTES EJEMPLOS, CORRESPONDIENTES A LA PARTE DEL INSTRUMENTO: {part.upper()}:  
                  {samples_text}.  
                
                - EXTRAE EL TEXTO QUE PERTENECE A LA PARTE DEL INSTRUMENTO {part.upper()} DEL SIGUIENTE FRAGMENTO:  
                  {row_segment}.
                
                - ASEGÚRATE DE QUE TODO EL TEXTO RELEVANTE ESTÉ COMPLETAMENTE TRANSCRITO SIN OMITIR NINGUNA PARTE.  
                - DEVUELVE ÚNICAMENTE EL TEXTO EXTRAÍDO EN FORMATO PLANO, SIN COMENTARIOS ADICIONALES, EXPLICACIONES O FORMATO EXTRA.  
                **IMPORTANTE: LA RESPUESTA SIEMPRE DEBE SER EN ESPAÑOL.**
                """

        # Make inference
        stream = ollama.chat(model=model_name,
                             messages=[
                                 {
                                     'role': 'system',
                                     'content': system_prompt,
                                 },
                                 {
                                     'role': 'user',
                                     'content': user_prompt,
                                 },
                             ],
                             options={
                                 "temperature": 0.1,
                                 "top_p": 0.95,
                                 "top_k": 40,

                             },
                             stream=True,
                             )

        refined_segment = ''
        for chunk in stream:
            response = chunk['message']['content']
            refined_segment += response

        # -------------------------------------------------------------
        # Insert the part string to the dictionary
        # -------------------------------------------------------------
        dictionary[part]['refined_segment'] = refined_segment

        print(f'\tRefined Segment:')
        words = [word for word in re.split(r'(\S+)', dictionary[part]['refined_segment']) if word.strip()]
        for j in range(0, len(words), 18):
            print(f'\t' + " ".join(words[j:j + 18]))

    kill_ollama_server()
    return dictionary

def get_adquirente_label(adquirente:str):

    model_name = 'qwen2.5-max:14b'

    prompt_ner = f"Clasifica el siguiente texto: '{adquirente}', en alguna de las siguientes opciones: persona | organización."
    query = ollama.chat(model=model_name,
                        messages=[{'role': 'user', 'content': prompt_ner, }],
                        options={"temperature": 0.1, "top_p": 0.95, "top_k": 40, },
                        stream=False,
                        )
    adquirente_label = query['message']['content']
    if 'persona' in adquirente_label:
        adquirente_label = 'persona'
    else:
        adquirente_label = 'organización'

    return adquirente_label

def extract_entities(dictionary, knowledge):

    # -------------------------------------------------------------
    # EXTRACT ENTITIES FROM EXTRACTED PART
    # -------------------------------------------------------------
    model_name = 'qwen2.5-max:14b'

    # Start Ollama Server
    start_ollama_server()
    # Run Model
    run_ollama_model(model_name)

    # Define entities dictionary
    dictionary['entities'] = {}

    # Get parts list
    parts_list = knowledge['parts']

    for part in parts_list:

        # get fields, description
        fields_data = [(field_name, field_description) for field_part, field_name, field_description in knowledge['fields'] if field_part == part]

        # Clean cache
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        # Get part row segment
        part_text = dictionary[part]['part']

        adquirente_label = ''
        adquirente = ''

        if 'adquirente' in dictionary['entities']:
            adquirente = dictionary['entities']['adquirente']
            adquirente_label = get_adquirente_label(adquirente)


        # Get fields to extract and descriptions
        fields_samples = knowledge['fields_samples']
        for field_name, description in fields_data:
            field_samples = fields_samples.loc[
                (fields_samples['part'] == part) & (fields_samples['field'] == field_name),
                'sample'
            ].tolist()
            field_samples = '\n'.join(field_samples)

            system_prompt = """
            Sistema NLP especializado en extracción de datos estructurados de textos en español.
            Reglas estrictas:
            1. Formato de salida:
               a) Texto plano sin formato
               b) Sin encabezados, comentarios ni metadatos
            2. Procesamiento:
               a) Normalización Unicode NFKC + eliminación de caracteres no imprimibles
               b) Extracción VERBATIM: mantener mayúsculas/minúsculas, puntuación, espacios y caracteres especiales
            3. Valores no válidos:
               a) Retornar "" (cadena vacía) en casos de:
                 - Ambigüedad
                 - Coincidencias parciales/no exactas
            """

            general_prompt = f"""
            ### EXTRACCIÓN DE DATOS
            Campo: {field_name}
            Contexto: {description}
            Ejemplos: {field_samples}
            Texto fuente: {part_text}

            REGLAS DE EJECUCIÓN:
            1. Extraer valor EXACTO que cumpla:
                a) Coincidencia literal (case-sensitive)
                b) Alineación con el contexto proporcionado y los ejemplos
            2. Prohibido:
                a) Correcciones ortográficas o gramaticales
                b) Conversiones de formato 
                c) Inferencias o cálculos
            3. Casos especiales:
                a) Si hay dígitos/símbolos relevantes → incluirlos VERBATIM
            4. Salida:
                a) Texto plano sin explicaciones, comentarios ni notas adicionales
                b) Texto VERBATIM literal del texto original.
                c) La única modificación válida es la separación de palabras.
            RESULTADO:
            """

            estado_civil_prompt = f'''
            ### EXTRACCIÓN DE DATOS
            Adquirente : {adquirente}
            Campo: {field_name}
            Contexto: {description}
            Texto fuente: {part_text}
            REGLAS DE EJECUCIÓN:
            1. Extraer valor EXACTO que cumpla:
                a) El {field_name} de {adquirente}
                b) Coincidencia literal (case-sensitive) -> VERBATIM
                c) Alineación con el contexto proporcionado y los ejemplos
            2. Prohibido:
                a) Correcciones ortográficas o gramaticales
                b) Conversiones de formato 
                c) Inferencias o cálculos
            3. Casos especiales:
                a) Sí en el texto proporcionado no se encuentra o es ambiguo el {field_name} de {adquirente} responder con '' (vacío)
            4. Salida:
                a) Texto plano sin explicaciones, comentarios ni notas adicionales
                b) Texto VERBATIM literal del texto original.
                c) La única modificación válida es la separación de palabras.
            RESULTADO:
            '''

            if field_name == 'estado civil' and adquirente_label == 'persona':
                user_prompt = estado_civil_prompt

            elif field_name == 'estado civil' and adquirente_label != 'persona':
                dictionary['entities'][field_name] = ''
                continue

            else:
                user_prompt = general_prompt

            # Make inference
            stream = ollama.chat(model=model_name,
                                     messages=[
                                         {
                                             'role': 'system',
                                             'content': system_prompt,
                                         },
                                         {
                                             'role': 'user',
                                             'content': user_prompt,
                                         },
                                     ],
                                     options={
                                         "temperature": 0.1,
                                         "top_p": 0.95,
                                         "top_k": 40,

                                     },
                                     stream=True,
                                     )

            entity_text = ''
            for chunk in stream:
                response = chunk['message']['content']
                entity_text += response

            # save entity in to the dictionary
            dictionary['entities'][field_name] = entity_text
            print(f"{field_name} : {entity_text}")

    kill_ollama_server()
    return dictionary

def generate_text(dictionary, knowledge):

    # print(f'V.- Generate text...')
    # print(f'{"-" * 100}')

    model_name = 'qwen2.5-max:14b'

    # Start Ollama Server
    start_ollama_server()

    # Run Model
    run_ollama_model(model_name)

    # Validate task
    task = dictionary['task']

    template = ''
    if task == 'antecedente de propiedad':

        template = (
            """I.- DE PROPIEDAD: Por escritura número $$número de escritura$$ de fecha $$fecha de escritura$$, ante (el Licenciado|la Licenciada) $$notario$$, titular de la Notaría $$número de notaría$$ (de | del| de la) $$lugar de la notaría$$,
            (el señor|la señora|la señorita|los señores|las señoras|la sociedad|la asociación|) $$adquirente$$, (estando $$estado civil$$, |)(adquirió|adquirieron) por $$acto jurídico$$ y en precio de $$precio$$, Moneda Nacional, 
            $$descripción del inmueble$$, con la superficie, rumbos, medidas y colindancias, descritos en dicha escritura y mismos que transcriben a continuación,
            como si a la letra se insertasen:
            “...$$superficie y colindancias$$...”
           """
        )

    entities = dictionary['entities']

    system_prompt = f"""
    Eres un sistema especializado en redacción de textos en español, 
    rellenando plantillas con datos proporcionados.  

    Reglas estrictas:  
    1. **Formato de salida:**  
       a) Texto plano sin formato.
       b) Minúsculas y Mayúsculas, respetando reglas gramaticales del español.  
       c) Sin encabezados, comentarios ni metadatos.  
    
    2. **Procesamiento:**  
       a) Rellenar los campos entre $$campo$$ de la plantilla {template} proporcionada con los datos proporcionados {entities}.  
       b) Respetar la estructura de la plantilla, 
       c) Ajustar la redacción de artículos (el|la|los|las), verbos y sustantivos de acuerdo al género y número de las entidades proporcionadas.
       d) Eliminar espacios (s), saltos de línea (n), retornos de carro (r) y tabuladores (t) innecesarios.  
       e) Aplicar normalización Unicode NFKC y eliminar caracteres no imprimibles.
       
    """

    user_prompt = f"""
    ### REDACCIÓN DE TEXTO  
    Campos (clave:valor): {entities}  
    Plantilla: {template}  
    
    **REGLAS DE EJECUCIÓN:**  
    1. **Redacción de texto usando la plantilla {template} proporcionada con los siguientes datos {entities}.**
    2. **Prohibido:**  
       a) Correcciones ortográficas o gramaticales.  
       b) Conversiones de formato.  
       c) Inferencias sobre datos no proporcionados, excepto para la concordancia de género y número.
    3. **Casos especiales:**  
       a) Si un campo incluye números, signos de puntuación o símbolos especiales, deben conservarse exactamente como están.
       b) Asegurar que después de la palabra PESOS se escriba: ', MONEDA NACIONAL'  
    
    4. **Salida:**  
       a) Texto plano sin explicaciones, comentarios ni notas adicionales. VERBATIM  
       b) El texto generado debe seguir la plantilla proporcionada sin alteraciones estructurales, salvo las adaptaciones de género y número.  
    
    **RESULTADO:**  

    """
    # Make inference
    stream = ollama.chat(model=model_name,
                             messages=[
                                 {
                                     'role': 'system',
                                     'content': system_prompt,
                                 },
                                 {
                                     'role': 'user',
                                     'content': user_prompt,
                                 },
                             ],
                             options={
                                 "temperature": 0.1,
                                 "top_p": 0.95,
                                 "top_k": 40,

                             },
                             stream=True,
                             )

    text = ''
    for chunk in stream:
        response = chunk['message']['content']
        text += response

    # save entity in to the dictionary
    dictionary['text'] = text

    kill_ollama_server()
    return dictionary

def get_knowledge(task):

    csv_path = "/home/n230ai/Documentos/aplicaciones/n230ai_ocr_v2/knowledge/index.csv"

    # Extract paths
    # -------------------------------------
    index = pd.read_csv(csv_path, encoding='utf-8')
    # Strip column names to remove spaces
    index.columns = index.columns.str.strip()

    # Parts by task
    # -------------------------------------
    # Get path
    tasks_path = index.loc[index['name'] == 'task_parts', 'path'].values[0]
    # Get data
    df_parts = pd.read_csv(tasks_path, encoding='utf-8')
    # Strip column names to remove spaces
    df_parts.columns = df_parts.columns.str.strip()
    # Filter data by task
    parts = df_parts.loc[df_parts['task'] == task, 'part'].tolist()


    # Get samples by part
    # -------------------------------------
    # Get path
    parts_samples_path = index.loc[index['name'] == 'parts_samples', 'path'].values[0]
    # Get samples
    df_parts_samples = pd.read_csv(parts_samples_path, encoding='utf-8')
    # Strip column names to remove spaces
    df_parts_samples.columns = df_parts_samples.columns.str.strip()
    # Filter samples by part
    parts_samples = df_parts_samples.loc[df_parts_samples['part'].isin(parts), ['part', 'sample']]
    # Count samples per part
    part_samples_count = parts_samples.groupby('part').size()

    # Entities by Part and Task
    # -------------------------------------
    # Get path
    parts_fields_path = index.loc[index['name'] == 'parts_fields', 'path'].values[0]
    # Read CSV
    df_fields = pd.read_csv(parts_fields_path, encoding='utf-8')
    # Strip column names to remove spaces
    df_fields.columns = df_fields.columns.str.strip()
    # Apply filtering with fixed column names
    # Return a list of lists where [[part, field, description],...]
    fields_data = df_fields.loc[
        (df_fields['task'] == task) & (df_fields['part'].isin(parts)),
        ['part', 'field', 'description']
    ].values.tolist()

    fields_names = [field_name for part, field_name, description in fields_data]

    # Get samples by field
    # -------------------------------------
    # Get path
    fields_samples_path = index.loc[index['name'] == 'fields_samples', 'path'].values[0]
    # Get fields samples
    df_fields_samples = pd.read_csv(fields_samples_path, encoding='utf-8')
    # Strip column names to remove spaces
    df_fields_samples.columns = df_fields_samples.columns.str.strip()
    # Filter samples by field
    fields_samples = df_fields_samples.loc[df_fields_samples['field'].isin(fields_names), ['part', 'field', 'sample']]
    # Count samples per field
    field_samples_count = fields_samples.groupby('field').size()


    # Create a knowledge dictionary:
    knowledge = {
        'task': task,
        'parts': parts,
        'fields': fields_data,
        'parts_samples': parts_samples,
        'fields_samples': fields_samples
    }
    return knowledge


from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from multiprocessing import Pool, cpu_count


if __name__ == "__main__":
    import pandas as pd

    start_time_pipeline = time.perf_counter()

    # Define paths
    text_path = '/home/n230ai/Descargas/prueba_3.txt'

    # Select task
    selected_task = 'antecedente de propiedad'


    # ------------------------------------------------------------------------------------
    # I. GET KNOWLEDGE
    # ------------------------------------------------------------------------------------

    start_time = time.time()
    knowledge = get_knowledge(selected_task)
    end_time = time.time()
    total_time = end_time - start_time
    print(f'\tI. Recover knowledge: {total_time:.2f}s')

    # ------------------------------------------------------------------------------------
    # II. PREPROCESS TEXT AND PARTS SAMPLES
    # ------------------------------------------------------------------------------------

    start_time = time.time()
    dictionary= preprocess_2(text_path=text_path, knowledge=knowledge)
    end_time = time.time()
    total_time = end_time - start_time
    print(f'\tII. Preprocess texts {total_time:.2f}s')
    # dictionary=refine_segments(dictionary=dictionary)


    # # ------------------------------------------------------------------------------------
    # # EXTRACT ENTITIES
    # # ------------------------------------------------------------------------------------
    # print(f'\tExtracting entities...')
    # dictionary = extract_entities(dictionary=dictionary, knowledge=knowledge)
    #
    # # ------------------------------------------------------------------------------------
    # # GENERATE NEW TEXT
    # # ------------------------------------------------------------------------------------
    # print(f'\tGenerating text...')
    # dictionary = generate_text(dictionary=dictionary, knowledge=knowledge)
    #
    # end_time_pipeline = time.perf_counter()
    # total_time_pipeline = end_time_pipeline - start_time_pipeline
    # print('='*100)
    # print(f'Total Process Time: {total_time_pipeline:.2f} seg.')
    # print('='*100)
    #
    # print('Generated Text:')
    # print(dictionary['text'])
    #


