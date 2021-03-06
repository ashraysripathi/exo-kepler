# URL for the Kepler light-curve file database
BASE_URL = "http://archive.stsci.edu/pub/kepler/lightcurves"

# CSV file for downloading the training data, and reference to the data labels
KEPLER_CSV_FILE = "D:/Kepler_Download/dr24predict.csv"

# Name and values of the column in the input CSV file to use as training labels.
LABEL_COLUMN = "av_training_set"

# Labels for classification
# PC (planet candidate), AFP (astrophysical false positive), NTP (non-transiting phenomenon), UNK (unknown)
ALLOWED_LABELS = ["PC", "AFP", "NTP"]

# NB_CLASSES = len(ALLOWED_LABELS)
# As we only focus on Planet Candidate prediction, we merge all other
# labels to one, 'NON_PC' category so as to simplify the training
NB_CLASSES = 2

# Folder for the Kepler data
KEPLER_DATA_FOLDER = 'D:/K2_Light_Curves_Dataset'

# Folder for the training data set
# Training data set in this folder will be divided to
# 'Training", 'Validation' and 'Test' folders
TRAINING_FOLDER = 'D:/Kepler_Data/Kepler_Training_Data'

# Folder to save the trained model
# Including training logs and checkpoints
KEPLER_TRAINED_MODEL_FOLDER = r'D:\Kepler_Data\K_Trained_Model'

# ==== May not needed =====
# Folder for all other Kepler data besides the training data set
# Light-curve data wants to be predicted would be downloaded to
# this folder if not in the KEPLER_DATA_FOLDER yet
DATA_FOR_PREDICTION_FOLDER = 'D:/Kepler_Data/unclassified_data'

# Store the predicted image file for global view and local view
PREDICT_OUTPUT_FOLDER = 'D:/Kepler_Data/K_Predicted_Output'
