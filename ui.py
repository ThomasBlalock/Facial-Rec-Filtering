from feature_extraction import extract_features
from crop_faces import get_faces
from nn_search import linear_nn_search, add_base_path_to_image_paths
from print_cluster import copy_images_to_destination


def find_nearest_cluster(folder_path, dataset_features_filepath, raw_filepath, output_filepath):

    get_faces(folder_path, folder_path+"/face")

    extract_features(folder_path+"/face", folder_path)

    sorted_filepaths = linear_nn_search(
        folder_path+"/features.csv", dataset_features_filepath+"/features.csv")

    complete_filepaths = add_base_path_to_image_paths(
        sorted_filepaths[:50], raw_filepath)

    copy_images_to_destination(complete_filepaths, output_filepath)


dataset_features_filepath = "C:/Users/C25Thomas.Blalock/Datasets/facial-rec-filtering-data"
raw = "C:/Users/C25Thomas.Blalock/Datasets/facial-rec-filtering-data/test_dataset"


img = "C:/Users/C25Thomas.Blalock/Datasets/facial-rec-filtering-data/test_query"
output = "C:/Users/C25Thomas.Blalock/Datasets/facial-rec-filtering-data/test_query/output"


find_nearest_cluster(img, dataset_features_filepath, raw, output)

# raw = "C:/Users/C25Thomas.Blalock/Coding/Image_Web_Guyvre_Data/faces"

# find_nearest_cluster(img, clusts, raw, output)
