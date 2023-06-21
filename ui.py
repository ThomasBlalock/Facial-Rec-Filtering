from feature_extraction import extract_features
from crop_faces import get_faces
from nn_search import nn_search, add_base_path_to_image_paths
from print_cluster import copy_images_to_destination


def find_nearest_cluster(folder_path, clusts_filepath, raw_filepath, output_filepath):

    get_faces(folder_path, folder_path+"/face")

    extract_features(folder_path+"/face", folder_path)

    idx, filepaths = nn_search(
        folder_path+"/features.csv", clusts_filepath+"/clusters.csv")

    complete_filepaths = add_base_path_to_image_paths(filepaths, raw_filepath)

    copy_images_to_destination(complete_filepaths, output_filepath)
    # copy_images_to_destination(filepaths, output_filepath)  # get faces


clusts = "C:/Users/C25Thomas.Blalock/Coding/Image_Web_Guyvre_Data"
raw = "C:/Users/C25Thomas.Blalock/Coding/Image_Web_Guyvre_Data/raw"


img = "C:/Users/C25Thomas.Blalock/Coding/Image_Web_Guyvre_Data/Webguy Demo/Demo 2 - Thomas"
output = "C:/Users/C25Thomas.Blalock/Coding/Image_Web_Guyvre_Data/Webguy Demo/Demo 2 - Thomas/output"

# img = "C:/Users/C25Thomas.Blalock/OneDrive - afacademy.af.edu/Desktop/Webguy Demo/Demo 2 - Thomas"
# output = "C:/Users/C25Thomas.Blalock/OneDrive - afacademy.af.edu/Desktop/Webguy Demo/Demo 2 - Thomas/output"

# img = "C:/Users/C25Thomas.Blalock/OneDrive - afacademy.af.edu/Desktop/Webguy Demo/Demo 3 - Someone"
# output = "C:/Users/C25Thomas.Blalock/OneDrive - afacademy.af.edu/Desktop/Webguy Demo/Demo 3 - Someone/output"

# img = "C:/Users/C25Thomas.Blalock/OneDrive - afacademy.af.edu/Desktop/Webguy Demo/Demo 4 - Flex"
# output = "C:/Users/C25Thomas.Blalock/OneDrive - afacademy.af.edu/Desktop/Webguy Demo/Demo 4 - Flex/output"

find_nearest_cluster(img, clusts, raw, output)

# raw = "C:/Users/C25Thomas.Blalock/Coding/Image_Web_Guyvre_Data/faces"

# find_nearest_cluster(img, clusts, raw, output)