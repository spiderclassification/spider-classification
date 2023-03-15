import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import image_utils
from keras.applications.vgg16 import preprocess_input
from urllib.parse import unquote
from PIL import UnidentifiedImageError

def create_data_split(dataset_dir, seed=42):
    class_labels = set()
    family_labels = set()
    genus_labels = set()
    species_labels = set()

    df = pd.DataFrame(columns=['file_path', 'class', 'family', 'genus', 'species'])

    for class_item in os.listdir(dataset_dir):
        class_item = unquote(class_item)
        class_path = os.path.join(dataset_dir, class_item)
        if os.path.isdir(class_path):
            class_labels.add(class_item)

            for family_item in os.listdir(class_path):
                family_item=unquote(family_item)
                family_path = os.path.join(class_path, family_item)
                if os.path.isdir(family_path):
                    family_labels.add(family_item)

                    for genus_item in os.listdir(family_path):
                        genus_item=unquote(genus_item)
                        genus_path = os.path.join(family_path, genus_item)
                        if os.path.isdir(genus_path):
                            genus_labels.add(genus_item)

                            for species_item in os.listdir(genus_path):
                                species_item=unquote(species_item)
                                species_path = os.path.join(genus_path, species_item)
                                if os.path.isdir(species_path):
                                    species_labels.add(species_item)

                                    for item in os.listdir(species_path):
                                        item=unquote(item)
                                        item_path = os.path.join(species_path, item)

                                        # add the items in the species directory as new rows
                                        new_row = [item_path ,class_item, family_item, genus_item, species_item]
                                        df.loc[len(df)] = new_row

                                # in a genus directory, if the item is not a species folder, add it to the dataframe where species = 'N/A'
                                else:
                                    new_row = [species_path, class_item, family_item, genus_item, 'N/A']
                                    df.loc[len(df)] = new_row

                        # in a family directory, if the item is not a genus folder, add it to the dataframe where genus, species = 'N/A'
                        else:
                            new_row = [genus_path, class_item, family_item, 'N/A', 'N/A']
                            df.loc[len(df)] = new_row

                # in a class directory, if the item is not a family folder, add it to the dataframe where family, genus, species = 'N/A'
                else:
                    new_row = [family_path, class_item, 'N/A', 'N/A', 'N/A']
                    df.loc[len(df)] = new_row

        # in the dataset directory, if the item is not a class folder, add it to the dataframe where class, family, genus, species = 'N/A'
        else:
            new_row = [class_path, 'N/A', 'N/A', 'N/A', 'N/A']
            df.loc[len(df)] = new_row

    class_labels.add('N/A')
    family_labels.add('N/A')
    genus_labels.add('N/A')
    species_labels.add('N/A')

    class_labels = list(class_labels)
    class_labels.sort()
    family_labels = list(family_labels)
    family_labels.sort()
    genus_labels = list(genus_labels)
    genus_labels.sort()
    species_labels = list(species_labels)
    species_labels.sort()

    # Encode all labels and add each encoding into new columns
    print("Encoding labels..")
    df = encode_labels(df, 'class')
    df = encode_labels(df, 'family')
    df = encode_labels(df, 'genus')
    df = encode_labels(df, 'species')

    # Shuffle the dataframe using a specific seed for reproducibility
    df = df.sample(frac=1, random_state=seed)

    # Split the shuffled dataframe between train, val, and test with ratio 0.8, 0.1, 0.1
    num_data = df.shape[0]
    train_df = df[:int(0.8 * num_data)]
    val_df = df[int(0.8 * num_data):int(0.9 * num_data)]
    test_df = df[int(0.9 * num_data):]



    # # Create folders for each set
    # if not os.path.exists('processed_dataset/train'):
    #     os.makedirs('processed_dataset/train')
    # if not os.path.exists('processed_dataset/val'):
    #     os.makedirs('processed_dataset/val')
    # if not os.path.exists('processed_dataset/test'):
    #     os.makedirs('processed_dataset/test')
    #
    # # Save the data into separate folders
    # for file in train_df['file_path']:
    #     save_dir_filename = os.path.join('processed_dataset\\train',file.split('\\')[-1])
    #     # print(save_dir_filename)
    #     Image.open(file).save(save_dir_filename)
    # for file in val_df['file_path']:
    #     save_dir_filename = os.path.join('processed_dataset\\val',file.split('\\')[-1])
    #     # print(save_dir_filename)
    #     Image.open(file).save(save_dir_filename)
    # for file in test_df['file_path']:
    #     save_dir_filename = os.path.join('processed_dataset\\test',file.split('\\')[-1])
    #     # print(save_dir_filename)
    #     Image.open(file).save(save_dir_filename)



    # train_df.to_csv(os.path.join('processed_dataset/train', 'train_data.csv'), index=False)
    # val_df.to_csv(os.path.join('processed_dataset/val', 'val_data.csv'), index=False)
    # test_df.to_csv(os.path.join('processed_dataset/test', 'test_data.csv'), index=False)

    return train_df, val_df, test_df, class_labels, family_labels, genus_labels, species_labels

def encode_labels(string_labels, column_name):

    encoded_df = string_labels
    le = LabelEncoder()

    # Fit the LabelEncoder on the label column
    le.fit(encoded_df[column_name])

    # Transform the labels to integers and save values to a new column
    encoded_df[(column_name+"_encoded")] = le.transform(encoded_df[column_name])

    return encoded_df

# Preprocess the image
def preprocess_image(img):
    preproc_img = image_utils.img_to_array(img)
    preproc_img = np.expand_dims(preproc_img, axis=0)
    preproc_img = preprocess_input(preproc_img)

    return preproc_img

def preprocess_image_dataset(dataset, datapath, dataset_type, chunk_size=1000):
    chunk_count = 1
    total_rows = len(dataset)

    for i in range(0, total_rows, chunk_size):
        end_index = min(i + chunk_size, total_rows)
        if os.path.exists(os.path.join(datapath, f"X_{dataset_type}_data_chunk{chunk_count}.npy")):
            print(f"Skipping chunk {chunk_count}. {datapath}\X_{dataset_type}_data_chunk{chunk_count}.npy already exists.")
        else:
            chunk_paths = dataset['file_path'].iloc[i:end_index]
            chunk_images = []
            for fp in chunk_paths:
                try:
                    im = image_utils.load_img(fp, target_size=(224, 224))
                    im_array = image_utils.img_to_array(im)
                    chunk_images.append(im_array)
                except Exception as e:
                    print(f"Error loading image at file path {fp}, exception: {e}")
                    # continue
            chunk_images = np.array(chunk_images)
            chunk_images = preprocess_input(chunk_images)
            print(f"[CHUNK IMAGES SHAPE] CHUNK: {chunk_count} SHAPE:{chunk_images.shape}")
            X_data_mmap = np.memmap(os.path.join(datapath, f"X_{dataset_type}_data_chunk{chunk_count}.npy"), dtype=np.float32, mode='w+', shape=chunk_images.shape)
            X_data_mmap[:] = chunk_images[:]
            del X_data_mmap
        chunk_count += 1

    print(f"END OF PREPROCESSING DATA {dataset_type} ****************************************************************************")

def preprocess_labels(dataset, datapath, classification, dataset_type, chunk_size=1000):
    chunk_count = 1
    total_rows = len(dataset)
    for i in range(0, len(dataset), chunk_size):
        end_index = min(i + chunk_size, total_rows)
        if os.path.exists(os.path.join(datapath, f"y_{classification}_{dataset_type}_chunk{chunk_count}.csv")):
            print(
                f"Skipping labels chunk {chunk_count}. {datapath}\y_{classification}_{dataset_type}_chunk{chunk_count}.csv already exists.")
        else:
            chunk_labels = dataset[classification].iloc[i:end_index]
            chunk_labels.to_csv(os.path.join(datapath, f"y_{classification}_{dataset_type}_chunk{chunk_count}.csv"), index=False)
            print(f"[CHUNK LABELS SHAPE] CHUNK: {chunk_count} SHAPE:{chunk_labels.shape}")
        chunk_count += 1

    print(f"END OF PREPROCESSING LABELS {classification} - {dataset_type}****************************************************************************")

def parse_data(model_path, data_path, dataset_dir):
    # Create folder for the model
    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    # Create folder for the data used in training the model
    if not os.path.isdir(data_path):
        os.makedirs(data_path)

    # Create file paths for the data files
    class_file = os.path.join(data_path, 'unique_classes.csv')
    family_file = os.path.join(data_path, 'unique_families.csv')
    genera_file = os.path.join(data_path, 'unique_genera.csv')
    species_file = os.path.join(data_path, 'unique_species.csv')
    train_df_file = os.path.join(data_path, 'train_df.csv')
    val_df_file = os.path.join(data_path, 'val_df.csv')
    test_df_file = os.path.join(data_path, 'test_df.csv')

    # Check if data files exist; load them if True, generate them from utils then save them if False
    if os.path.exists(class_file) and os.path.exists(family_file) and os.path.exists(genera_file) and os.path.exists(
            species_file) \
            and os.path.exists(train_df_file) and os.path.exists(val_df_file) and os.path.exists(test_df_file):
        print("Found existing data files. Loading..")
        classes_df = pd.read_csv(class_file)
        families_df = pd.read_csv(family_file)
        genera_df = pd.read_csv(genera_file)
        species_df = pd.read_csv(species_file)
        train_df = pd.read_csv(train_df_file)
        val_df = pd.read_csv(val_df_file)
        test_df = pd.read_csv(test_df_file)

        classes = classes_df.values.tolist()
        families = families_df.values.tolist()
        genera = genera_df.values.tolist()
        species = species_df.values.tolist()
        print("Loading data files done.")

    else:
        print("Missing/incomplete data files. Generating..")
        train_df, val_df, test_df, classes, families, genera, species = create_data_split(dataset_dir)
        print("Data files generated. Saving..")
        ################################################## Save to CSV
        # Convert to Dataframes
        classes_df = pd.DataFrame(classes, columns=['class'])
        families_df = pd.DataFrame(families, columns=['family'])
        genera_df = pd.DataFrame(genera, columns=['genus'])
        species_df = pd.DataFrame(species, columns=['species'])

        # Save to csv
        classes_df.to_csv(class_file, index=False)
        families_df.to_csv(family_file, index=False)
        genera_df.to_csv(genera_file, index=False)
        species_df.to_csv(species_file, index=False)

        train_df.to_csv(train_df_file, index=False)
        test_df.to_csv(test_df_file, index=False)
        val_df.to_csv(val_df_file, index=False)

    return train_df, val_df, classes, families, genera, species

def preprocess_dataset(model_path, data_path, dataset_dir, chunk_size):
    train_df, val_df, classes, families, genera, species = parse_data(model_path, data_path, dataset_dir)

    num_classes = len(classes)
    num_families = len(families)
    num_genera = len(genera)
    num_species = len(species)
    print(f"Classes: {num_classes}\n"
          f"Families: {num_families}\n"
          f"Genera: {num_genera}\n"
          f"Species: {num_species}\n")

    print("Separating training data and labels..")
    X_train = train_df.loc[:, ['file_path']]
    y_class_train = train_df.loc[:, ['class_encoded']]
    y_family_train = train_df.loc[:, ['family_encoded']]
    y_genus_train = train_df.loc[:, ['genus_encoded']]
    y_species_train = train_df.loc[:, ['species_encoded']]

    print("Separating validation data and labels..")
    X_val = val_df.loc[:, ['file_path']]
    y_class_val = val_df.loc[:, ['class_encoded']]
    y_family_val = val_df.loc[:, ['family_encoded']]
    y_genus_val = val_df.loc[:, ['genus_encoded']]
    y_species_val = val_df.loc[:, ['species_encoded']]

    # Note: Memory cannot handle preprocessing the whole x_train set
    print("Preprocessing training data and training labels..")
    preprocess_image_dataset(X_train, data_path, "train", chunk_size=chunk_size)
    preprocess_labels(y_class_train, data_path, "class_encoded", "train", chunk_size=chunk_size)
    preprocess_labels(y_family_train, data_path, "family_encoded", "train", chunk_size=chunk_size)
    preprocess_labels(y_genus_train, data_path, "genus_encoded", "train", chunk_size=chunk_size)
    preprocess_labels(y_species_train, data_path, "species_encoded", "train", chunk_size=chunk_size)

    len_X_val = len(X_val)

    print("Preprocessing validation data and validation labels..")
    preprocess_image_dataset(X_val, data_path, "val", chunk_size=len_X_val)
    preprocess_labels(y_class_val, data_path, "class_encoded", "val", chunk_size=len(y_class_val))
    preprocess_labels(y_family_val, data_path, "family_encoded", "val", chunk_size=len(y_family_val))
    preprocess_labels(y_genus_val, data_path, "genus_encoded", "val", chunk_size=len(y_genus_val))
    preprocess_labels(y_species_val, data_path, "species_encoded", "val", chunk_size=len(y_species_val))

    print("Preprocessing done.")

    return num_classes, num_families, num_genera, num_species, len_X_val