import os
from feature_extraction import extract_features
from crop_faces import get_faces
from k_means_clustering import k_means_clustering, plot_dist_by_clust

# main.py will go through the entire pre-UI process of clustering and booting the
# clusters and centroids to a file located in clustered_folder_path defined below

# Folder Paths
raw_folder_path = 'C:/Users/C25Thomas.Blalock/Coding/Image_Web_Guyvre_Data/raw'
face_folder_path = 'C:/Users/C25Thomas.Blalock/Coding/Image_Web_Guyvre_Data/face'
features_folder_path = 'C:/Users/C25Thomas.Blalock/Coding/Image_Web_Guyvre_Data'
clustered_folder_path = 'C:/Users/C25Thomas.Blalock/Coding/Image_Web_Guyvre_Data/clusters.csv'


# Crop faces (Comment out when not in use)
print("Starting get_faces()")
get_faces(raw_folder_path, face_folder_path)

# Extract features (Comment out when not in use)
print("Starting extract_features()")
extract_features(face_folder_path, features_folder_path)
