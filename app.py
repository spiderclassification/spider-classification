# To run streamlit, go to terminal and type: 'streamlit run app.py'
# Core Packages ###########################
import os

import streamlit as st

from keras.utils import image_utils
from keras.applications import VGG16
from keras.applications.vgg16 import decode_predictions, preprocess_input
from keras.models import load_model

import numpy as np
import pandas as pd
#######################################################################################################################
project_title = "Spider Classification"
project_icon = "spiderClip.png"

st.set_page_config(page_title=project_title, initial_sidebar_state='collapsed',page_icon=project_icon,layout='wide')

current_path = os.getcwd()
model_directory = os.path.join(current_path, "model", "complete", "hierarchical-multilabel-classifier")
model_name = "hierarchical-multilabel-classifier-100epoch-1e-07lrscheduler_30.h5"
model_path = os.path.join(model_directory, model_name)
data_directory = os.path.join(model_directory, "data")
#######################################################################################################################

@st.cache(allow_output_mutation=True)
def load_spider_models():
    vgg_model = VGG16(weights='imagenet')
    spider_prediction_model = load_model(model_path)

    return vgg_model, spider_prediction_model

# Preprocess the image
def preprocess_image(img):
    preproc_img = image_utils.img_to_array(img)
    preproc_img = np.expand_dims(preproc_img, axis=0)
    preproc_img = preprocess_input(preproc_img)

    return preproc_img

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
    if uploaded_file:
        # st.image(uploaded_file)
        img = image_utils.load_img(uploaded_file, target_size=(224, 224))

    vgg_model, spider_prediction_model = load_spider_models()

    col1, col2, col3, col4, col5 = st.columns([0.15, 0.15, 0.15, 0.15, 1])

    class_threshold = col1.number_input("Class Threshold %", 0, 100, 20, 1)
    family_threshold = col2.number_input("Family Threshold %", 0, 100, 20, 1)
    genus_threshold = col3.number_input("Genus Threshold %", 0, 100, 20, 1)
    species_threshold = col4.number_input("Species Threshold %", 0, 100, 20, 1)
    threshold_values = [class_threshold, family_threshold, genus_threshold, species_threshold]

    # button to run prediction
    identify_spider_btn = st.button("Identify Spider")

    if identify_spider_btn and uploaded_file:
        preprocessed_img = preprocess_image(img)

        # NOTE: STEP 1 - SPIDER OR NOT?
        # Make the prediction
        vgg_preds = vgg_model.predict(preprocessed_img)

        # Decode the prediction
        pred_class = decode_predictions(vgg_preds, top=3)  # [0][0][1]
        pred_class_index = np.argmax(vgg_preds)
        pred_confidence = vgg_preds[0][pred_class_index]

        # 70= harvestman, daddy longlegs, Phalangium opilio
        # 71= scorpion
        # 72= black and gold garden spider, Argiope aurantia
        # 73= barn spider, Araneus cavaticus
        # 74= garden spider, Aranea diademata
        # 75= black widow, Latrodectus mactans
        # 76= tarantula
        # 77= wolf spider, hunting spider
        # 78= tick
        # 815 = spider web
        if (70 > pred_class_index or pred_class_index > 77 or 71 == pred_class_index) and (pred_class_index != 815):
        # if False: # Note: this is for by passing the first spider-or-not check
            # st.error(f"Image is not a spider. Image class: {pred_class} Confidence: {pred_confidence}")
            st.error(f"Image is not a spider. Top 3 class predictions: {pred_class[0][0][1]} {pred_class[0][0][2]} | {pred_class[0][1][1]} {pred_class[0][1][2]} | {pred_class[0][2][1]} {pred_class[0][2][2]}")
        # NOTE: STEP 2 - Predict class, family, genus, species
        else:
            st.success(f"Spider detected! Top 3 class predictions: {pred_class[0][0][1]} {pred_class[0][0][2]} | {pred_class[0][1][1]} {pred_class[0][1][2]} | {pred_class[0][2][1]} {pred_class[0][2][2]}")

            # Load classes saved in csv files from the training phase
            classes = pd.read_csv(model_directory + "/data/unique_classes.csv")
            families = pd.read_csv(model_directory + "/data/unique_families.csv")
            genera = pd.read_csv(model_directory + "/data/unique_genera.csv")
            species = pd.read_csv(model_directory + "/data/unique_species.csv")

            hierarchy_labels_df = pd.read_csv(os.path.join(current_path,"dataset","hierarchy_labels.csv"))

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

            new_sorted_indices = [list(),list(),list(),list()]
            new_prediction_percentages = [list(),list(),list(),list()]

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
                        for passed_previous_label in threshold_passer[i-1]:
                            if belongs_in_accepted_higher_class:
                                break
                            threshold_df = hierarchy_labels_df.iloc[:, [i-1, i]]
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


                st.write(prediction_text)
                # if i > 0: # For the Family, Genus, Species label which have more than 2 classes
                #     top1_label = pred_labels[i][new_sorted_indices[i][0]]
                #     top2_label = pred_labels[i][new_sorted_indices[i][1]]
                #     top3_label = pred_labels[i][new_sorted_indices[i][2]]
                #
                #     top1_percent = "{:.2f}".format(new_prediction_percentages[i][0] * 100)
                #     top2_percent = "{:.2f}".format(new_prediction_percentages[i][1] * 100)
                #     top3_percent = "{:.2f}".format(new_prediction_percentages[i][2] * 100)
                #
                #     prediction1 = f"Predicted {label}: #1 {top1_label} ({top1_percent}%)"
                #     prediction2 = f" | #2 {top2_label} ({top2_percent}%)"
                #     prediction3 = f" | #3 {top3_label} ({top3_percent}%)"
                #
                #     prediction_message = f"No {label} prediction passed the set threshold"
                #     if threshold_values[i] < (new_prediction_percentages[i][0] * 100):
                #         prediction_message = prediction1
                #         if threshold_values[i] < (new_prediction_percentages[i][1] * 100):
                #             prediction_message += prediction2
                #             if threshold_values[i] < (new_prediction_percentages[i][2] * 100):
                #                 prediction_message += prediction3
                #
                #     st.write(prediction_message)
                #
                # else: # For the Class label which only has 2 classes
                #     top1_label = pred_labels[i][new_sorted_indices[i][0]]
                #     top2_label = pred_labels[i][new_sorted_indices[i][1]]
                #
                #     top1_percent = "{:.2f}".format(new_prediction_percentages[i][0] * 100)
                #     top2_percent = "{:.2f}".format(new_prediction_percentages[i][1] * 100)
                #
                #     prediction1 = f"Predicted {label}: #1 {top1_label} ({top1_percent}%)"
                #     prediction2 = f" | #2 {top2_label} ({top2_percent}%)"
                #
                #     prediction_message = f"No {label} prediction passed the set threshold."
                #     if threshold_values[i] < (new_prediction_percentages[i][0] * 100):
                #         prediction_message = prediction1
                #         if threshold_values[i] < (new_prediction_percentages[i][1] * 100):
                #             prediction_message += prediction2
                #
                #     st.write(prediction_message)
        # TODO
        # 1. Display next highest confidence score if not in selected location
        # ex. family A is 60% confident but not in Location B, family B is 35% confident but is in location B, then display Family B
        #
        # 2. Add threshold; for example if highest confidence is only 30% then don't display the classification

if __name__ == '__main__':
    main()

# To run streamlit, go to terminal and type: 'streamlit run app-source.py'

