from feature_extraction import extract_features
from crop_faces import get_faces
from nn_search import linear_nn_search, add_base_path_to_image_paths
from print_cluster import copy_images_to_destination
from sklearn.metrics import average_precision_score


def find_nearest_cluster(folder_path, dataset_features_filepath, raw_filepath, output_filepath, k):

    get_faces(folder_path, folder_path+"/face")

    extract_features(folder_path+"/face", folder_path)

    sorted_filepaths, scores, labels = linear_nn_search(
        folder_path+"/features.csv", dataset_features_filepath+"/features.csv")

    # Since scores are distances and higher scores mean less similarity, we need to invert them
    inverted_scores = [-1 * score for score in scores]
    map_score = average_precision_score(labels, inverted_scores)
    print(f"Mean Average Precision (MAP): {map_score}")

    # complete_filepaths = add_base_path_to_image_paths(
    #     sorted_filepaths[:k], raw_filepath)

    # copy_images_to_destination(complete_filepaths, output_filepath)


dataset_features_filepath = "C:/Users/C25Thomas.Blalock/Datasets/facial-rec-filtering-data"
raw = "C:/Users/C25Thomas.Blalock/Datasets/facial-rec-filtering-data/test_dataset"


img = "C:/Users/C25Thomas.Blalock/Datasets/facial-rec-filtering-data/test_query"
output = "C:/Users/C25Thomas.Blalock/Datasets/facial-rec-filtering-data/test_query/output"


find_nearest_cluster(img, dataset_features_filepath, raw, output, 39)
