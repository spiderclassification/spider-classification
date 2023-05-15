# To run streamlit, go to terminal and type: 'streamlit run app.py'
# Core Packages ###########################
import os

import streamlit as st

from keras.utils import image_utils
from keras.applications.vgg16 import decode_predictions, preprocess_input
from keras.models import load_model

import numpy as np
import pandas as pd

from keras import backend as K

def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val

#######################################################################################################################
project_title = "Spider Classification"
project_icon = "spiderClip.png"

st.set_page_config(page_title=project_title, initial_sidebar_state='collapsed',page_icon=project_icon,layout='wide')

current_path = os.path.dirname(__file__)

model_directory = os.path.join(current_path, "model", "complete", "hierarchical-multilabel-classifier")
model_name = "hierarchical-multilabel-classifier-200epoch-VGG16_70.h5"

model_path = os.path.join(model_directory, model_name)
data_directory = os.path.join(model_directory, "data")

# spider_or_not_path = os.path.join(current_path, "model", "complete", "1spider-or-not","spider-or-not-10epoch-0.001lr_8.h5")
spider_or_not_directory = os.path.join(current_path, "model", "complete", "spider-or-not3")
spider_or_not_name = "spider-or-not3-300epoch-v6-trainalllayers_best_cp.h5"

spider_or_not_path = os.path.join(spider_or_not_directory, spider_or_not_name)

#######################################################################################################################

@st.cache(allow_output_mutation=True)
def load_spider_models(spider_prediction_path = model_path, spider_not_path = spider_or_not_path):
    # vgg_model = VGG16(weights='imagenet')
    spider_prediction_model = load_model(spider_prediction_path)
    spider_or_not_model = load_model(spider_not_path, custom_objects={'f1_score': f1_score})

    return spider_prediction_model, spider_or_not_model

# Preprocess the image
def preprocess_image(img):
    preproc_img = image_utils.img_to_array(img)
    preproc_img = np.expand_dims(preproc_img, axis=0)
    preproc_img = preprocess_input(preproc_img)

    return preproc_img

def load_csv_data():
    # Load classes saved in csv files from the training phase
    classes = pd.read_csv(model_directory + "/data/unique_classes.csv")
    families = pd.read_csv(model_directory + "/data/unique_families.csv")
    genera = pd.read_csv(model_directory + "/data/unique_genera.csv")
    species = pd.read_csv(model_directory + "/data/unique_species.csv")

    hierarchy_labels_df = pd.read_csv(os.path.join(current_path, "dataset", "hierarchy_labels.csv"))

    # Convert into list format
    classes = classes.values.tolist()
    families = families.values.tolist()
    genera = genera.values.tolist()
    species = species.values.tolist()
    for index, i in enumerate(classes):
        classes[index] = (i[0])
    for index, i in enumerate(families):
        families[index] = (i[0])
    for index, i in enumerate(genera):
        genera[index] = (i[0])
    for index, i in enumerate(species):
        species[index] = (i[0])

    return classes, families, genera, species, hierarchy_labels_df

def list_models_in_folders(folder_paths):
    model_files = []
    for folder_path in folder_paths:
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith(".h5")]
        for file in files:
            model_files.append(os.path.join(folder_path, file))
    return model_files

# # Add the folder paths containing the models
# hmc_dir = os.path.join(current_path, "model", "complete", "hierarchical-multilabel-classifier")
# hmc2_dir = os.path.join(current_path, "model", "complete", "hmc2")
# spider_not2_dir = os.path.join(current_path, "model", "complete", "spider-or-not2")
# spider_not3_dir = os.path.join(current_path, "model", "complete", "spider-or-not3")
#
# folder_paths_prediction = [hmc_dir, hmc2_dir]
# folder_paths_spider_or_not = [spider_not2_dir, spider_not3_dir]
#
# # Get a list of model paths from the folders
# model_files_prediction = list_models_in_folders(folder_paths_prediction)
# model_files_spider_or_not = list_models_in_folders(folder_paths_spider_or_not)

def main():
    head_col = st.columns([1,8])
    with head_col[0]:
        st.image(project_icon)
    with head_col[1]:
        st.title(project_title)

    st.markdown("***")
    st.subheader("")
#########################################
    # instructions
    st.subheader("How it works: ")
    st.write("1. User uploads a spider image to the app. \n"
             "2. User selects the location where the spider image was taken. \n"
             "3. User clicks the 'Identify Spider' button.")

#########################################
    uploaded_file = st.file_uploader('Upload Files', accept_multiple_files=False, type=['png', 'jpg', 'jpeg', 'webp'])
    img = ""
    preview_image = st.checkbox("Preview uploaded image?")
    if uploaded_file:
        # st.image(uploaded_file)
        img = image_utils.load_img(uploaded_file, target_size=(224, 224))
        if preview_image:
            st.image(img)

    # selected_prediction_model = st.selectbox("Select Spider Prediction Model", model_files_prediction)
    # selected_spider_or_not_model = st.selectbox("Select Spider or Not Model", model_files_spider_or_not)
    #
    # spider_prediction_model, spider_or_not_model = load_spider_models(selected_prediction_model, selected_spider_or_not_model)

    # Use default for now
    spider_prediction_model, spider_or_not_model = load_spider_models()

    col1, col2, col3, col4 = st.columns(4)

    class_threshold = col1.number_input("Class Threshold %", 0, 100, 1, 1)
    family_threshold = col2.number_input("Family Threshold %", 0, 100, 1, 1)
    genus_threshold = col3.number_input("Genus Threshold %", 0, 100, 1, 1)
    species_threshold = col4.number_input("Species Threshold %", 0, 100, 1, 1)
    threshold_values = [class_threshold, family_threshold, genus_threshold, species_threshold]

    # old_vs_new = st.checkbox("[DEV TEST] Compare New model with Old model?")

    # button to run prediction
    identify_spider_btn = st.button("Identify Spider")

    if identify_spider_btn and uploaded_file:
        preprocessed_img = preprocess_image(img)

        # NOTE: STEP 1 - SPIDER OR NOT?
        # Make the prediction
        col5, col6 = st.columns(2)

        spider_or_not_preds = spider_or_not_model.predict(preprocessed_img)
        col5.write(spider_or_not_preds)

        # Decode the prediction
        spider_or_not_threshold = 0.5
        pred_label = (spider_or_not_preds[0][0] >= spider_or_not_threshold).astype(int)

        # 0 = not-spider
        # 1 = spider

        if pred_label == 0:
            # col5.error(f"Image is {spider_or_not_preds[0][0] * 100:.2f}% not a spider.")
            col5.error(f"Image is not a spider.")
        else:
            # col5.success(f"Spider detected! Image is {spider_or_not_preds[0][0] * 100:.2f}% a spider.")
            col5.success(f"Spider detected!")

            classes, families, genera, species, hierarchy_labels_df = load_csv_data()

            spider_preds = spider_prediction_model.predict(preprocessed_img)

            pred_class = np.squeeze(spider_preds[0])
            pred_family = np.squeeze(spider_preds[1])
            pred_genus = np.squeeze(spider_preds[2])
            pred_species = np.squeeze(spider_preds[3])

            sorted_indices_class = np.argsort(pred_class)[::-1]
            sorted_indices_family = np.argsort(pred_family)[::-1]
            sorted_indices_genus = np.argsort(pred_genus)[::-1]
            sorted_indices_species = np.argsort(pred_species)[::-1]

            pred_class = pred_class[sorted_indices_class]
            pred_family = pred_family[sorted_indices_family]
            pred_genus = pred_genus[sorted_indices_genus]
            pred_species = pred_species[sorted_indices_species]

            prediction_percentages = [pred_class, pred_family, pred_genus, pred_species]
            sorted_indices = [sorted_indices_class, sorted_indices_family, sorted_indices_genus, sorted_indices_species]

            pred_labels = [classes, families, genera, species]
            pred_label_title = ["Class", "Family", "Genus", "Species"]

            new_sorted_indices = [list(), list(), list(), list()]
            new_prediction_percentages = [list(), list(), list(), list()]

            threshold_passer = [list(), list(), list()]

            belongs_in_accepted_higher_class = True
            for i, label in enumerate(pred_label_title):

                # This loop creates new lists without the indices that point to values with 'nan'
                for sorted, percentages in zip(sorted_indices[i], prediction_percentages[i]):
                    predicted = pred_labels[i][sorted]
                    if str(predicted) != 'nan':
                        new_sorted_indices[i].append(sorted)
                        new_prediction_percentages[i].append(percentages)

                prediction_text = f"Predicted {label}"
                for index, prediction_conf in zip(new_sorted_indices[i], new_prediction_percentages[i]):
                    label_name = pred_labels[i][index]
                    percent_conf = "{:.2f}".format(prediction_conf * 100)

                    # Checking if Family/Genus/Species belongs to one of the classes higher in the hierarchy that passed the set threshold
                    if i > 0:
                        belongs_in_accepted_higher_class = False
                        for passed_previous_label in threshold_passer[i - 1]:
                            if belongs_in_accepted_higher_class:
                                break
                            threshold_df = hierarchy_labels_df.iloc[:, [i - 1, i]]
                            for h, h2 in zip(threshold_df.iloc[:, 0], threshold_df.iloc[:, 1]):
                                if h == passed_previous_label and h2 == label_name:
                                    belongs_in_accepted_higher_class = True
                                    break

                    # Compare prediction confidence with threshold value
                    if belongs_in_accepted_higher_class:
                        if threshold_values[i] < (prediction_conf * 100):
                            prediction_text += f" | {label_name} ({percent_conf}%)"
                            if 3 > i:
                                threshold_passer[i].append(f"{label_name}")

                        else:
                            # prediction_text += f" | No definite {label} (Highest confidence is {label_name}: {percent_conf}% / {threshold_values[i]}%)"
                            prediction_text += " | <>"
                            break

                    else:
                        prediction_text += f" | No predicted '{label}' belonging in previous higher classification."
                        break

                col5.write(prediction_text)

        # TODO
        # 1. Display next highest confidence score if not in selected location
        # ex. family A is 60% confident but not in Location B, family B is 35% confident but is in location B, then display Family B
        #
        # 2. DONE ===== Add threshold; for example if highest confidence is only 30% then don't display the classification

if __name__ == '__main__':
    main()

# To run streamlit, go to terminal and type: 'streamlit run app-source.py'

