import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
import os
import re


def load_embedding(filename):
    df = pd.read_csv(filename)
    first_row = df.iloc[0]
    embedding_list = first_row['features'].split(',')
    embedding_array = np.array([float(value) for value in embedding_list])
    return embedding_array


def load_embeddings_and_filenames(filename):
    df = pd.read_csv(filename)
    embeddings = []
    for index, row in df.iterrows():
        embedding_list = row['features'].split(',')
        embedding_array = np.array([float(value) for value in embedding_list])
        embeddings.append(embedding_array)
    embeddings = np.array(embeddings)

    # Return a dictionary with filenames and embeddings
    return {"image_path": df['image_path'].to_numpy(), "features": embeddings}


def linear_nn_search(reference_features_filepath, dataset_features_filepath):
    img_features = load_embedding(reference_features_filepath)
    dataset_features = load_embeddings_and_filenames(dataset_features_filepath)
    distances = np.linalg.norm(
        dataset_features['features'] - img_features, axis=1)
    sorted_indices = np.argsort(distances)
    sorted_image_paths = dataset_features['image_path'][sorted_indices]
    return sorted_image_paths


def add_base_path_to_image_paths(image_filenames, raw_filepath):
    full_image_paths = []
    for img_path in image_filenames:
        # Remove the extra part and replace it with .jpg
        new_img_path = img_path  # comment out when not using faces
        new_img_path = re.sub(r'_face\d+', '', img_path)
        full_path = os.path.join(raw_filepath, new_img_path).replace("\\", "/")
        full_image_paths.append(full_path)
    return full_image_paths


# # HNSW NN Search---------------------------------------------------------------
# import hnswlib

# def create_index_and_save(embeddings, index_filename):
#     # Initialize a new index
#     dim = embeddings.shape[1]
#     num_elements = embeddings.shape[0]
#     p = hnswlib.Index(space='l2', dim=dim)

#     # Build the index with the embeddings
#     p.init_index(max_elements=num_elements, ef_construction=200, M=16)

#     # Add the embeddings to the index
#     p.add_items(embeddings)

#     # Save the index to disk
#     p.save_index(index_filename)


# def search_index(query, index_filename, num_neighbors):
#     # Load the index from disk
#     p = hnswlib.Index(space='l2', dim=query.shape[0])
#     p.load_index(index_filename, max_elements=50000)

#     # Use the index to perform a knn search
#     labels, distances = p.knn_query(query, k=num_neighbors)

#     return labels, distances

# # Example Usage---------------------------------------------------------------

# # Create and save the index
# embeddings = load_embeddings_and_filenames('my_dataset.csv')['embeddings']
# create_index_and_save(embeddings, 'my_index.bin')

# # Load the index and perform a search
# query = load_embedding('my_query.csv')
# labels, distances = search_index(query, 'my_index.bin', num_neighbors=5)
