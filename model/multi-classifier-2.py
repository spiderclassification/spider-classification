import keras.regularizers
import pandas as pd
import numpy as np
from keras.applications import VGG16
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical

import utils
import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # replace 0 with the index of your GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # suppress TensorFlow messages

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)
tf.compat.v1.disable_eager_execution()

########################################################################
current_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(current_dir, 'complete', 'multi-classifier-2')
data_path = os.path.join(model_path,'data')
dataset_dir = os.path.join(current_dir, '..', "dataset\\Spider Identification Images")

# Create folder for the model
if not os.path.isdir(model_path):
    os.makedirs(model_path)

# Create folder for the data used in training the model
if not os.path.isdir(data_path):
    os.makedirs(data_path)

# Create file paths for the data files
class_file = os.path.join(data_path,'unique_classes.csv')
family_file = os.path.join(data_path,'unique_families.csv')
genera_file = os.path.join(data_path,'unique_genera.csv')
species_file = os.path.join(data_path,'unique_species.csv')
train_df_file = os.path.join(data_path,'train_df.csv')
val_df_file = os.path.join(data_path,'val_df.csv')
test_df_file = os.path.join(data_path,'test_df.csv')


# Check if data files exist; load them if True, generate them from utils then save them if False
if os.path.exists(class_file) and os.path.exists(family_file) and os.path.exists(genera_file) and os.path.exists(species_file)\
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
    train_df, val_df, test_df, classes, families, genera, species = utils.create_data_split(dataset_dir)
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
    ################################################## END Save to CSV

num_classes = len(classes)
num_families = len(families)
num_genera = len(genera)
num_species = len(species)
print(f"Classes: {num_classes}\n"
      f"Families: {num_families}\n"
      f"Genera: {num_genera}\n"
      f"Species: {num_species}\n")

print("Separating training data and labels..")
X_train = train_df.loc[:,['file_path']]
y_class_train = train_df.loc[:,['class_encoded']]
y_family_train = train_df.loc[:,['family_encoded']]
y_genus_train = train_df.loc[:,['genus_encoded']]
y_species_train = train_df.loc[:,['species_encoded']]

print("Separating validation data and labels..")
X_val = val_df.loc[:,['file_path']]
y_class_val = val_df.loc[:,['class_encoded']]
y_family_val = val_df.loc[:,['family_encoded']]
y_genus_val = val_df.loc[:,['genus_encoded']]
y_species_val = val_df.loc[:,['species_encoded']]

# NOTE: Moved encoding to create_data_split section
# print("Encoding training labels..")
# y_class_train = utils.encode_labels(y_class_train,'class')
# y_family_train = utils.encode_labels(y_family_train,'family')
# y_genus_train = utils.encode_labels(y_genus_train,'genus')
# y_species_train = utils.encode_labels(y_species_train,'species')
#
# print("Encoding validation labels..")
# y_class_val = utils.encode_labels(y_class_val,'class')
# y_family_val = utils.encode_labels(y_family_val,'family')
# y_genus_val = utils.encode_labels(y_genus_val,'genus')
# y_species_val = utils.encode_labels(y_species_val,'species')

print("Preprocessing training data and training labels..")
# Note: Memory cannot handle preprocessing the whole x_train set
chunk_size = 3000  # NOTE: HARD CODED
utils.preprocess_image_dataset(X_train, data_path, "train", chunk_size=chunk_size)
utils.preprocess_labels(y_class_train, data_path, "class_encoded", "train", chunk_size=chunk_size)
utils.preprocess_labels(y_family_train, data_path, "family_encoded", "train", chunk_size=chunk_size)
utils.preprocess_labels(y_genus_train, data_path, "genus_encoded", "train", chunk_size=chunk_size)
utils.preprocess_labels(y_species_train, data_path, "species_encoded", "train", chunk_size=chunk_size)

print("Preprocessing validation data and validation labels..")
utils.preprocess_image_dataset(X_val, data_path, "val", chunk_size=len(X_val))
utils.preprocess_labels(y_class_val, data_path, "class_encoded", "val", chunk_size=len(y_class_val))
utils.preprocess_labels(y_family_val, data_path, "family_encoded", "val", chunk_size=len(y_family_val))
utils.preprocess_labels(y_genus_val, data_path, "genus_encoded", "val", chunk_size=len(y_genus_val))
utils.preprocess_labels(y_species_val, data_path, "species_encoded", "val", chunk_size=len(y_species_val))

print("Preprocessing done.")
########################################################################

# Load the VGG16 model pre-trained on ImageNet
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Add new layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
x = Dropout(0.5)(x)

print("Building layers..")
# Output layers for class, family, genus, and species
class_predictions = Dense(num_classes, activation='softmax', name='class_predictions')(x)
family_predictions = Dense(num_families, activation='softmax', name='family_predictions')(x)
genus_predictions = Dense(num_genera, activation='softmax', name='genus_predictions')(x)
species_predictions = Dense(num_species, activation='softmax', name='species_predictions')(x)

print("Building model..")
# Create the final model
model = Model(inputs=base_model.input, outputs=[class_predictions, family_predictions, genus_predictions, species_predictions])

print("Compiling model..")
# Compile the model
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

print(f"Training start: {datetime.datetime.now()} ****************************************************************************")
# Train the model for set number of epochs
epoch=100
model_name = f"multi-classifier-2-{epoch}epoch"
starting_epoch = 51
i=starting_epoch # NOTE: part of skipping: load model at epoch 8
if not os.path.exists(os.path.join(model_path, f'{model_name}_evaluations.txt')):
    with open(os.path.join(model_path, f'{model_name}_evaluations.txt'), 'w') as f:
        f.write(f"Evaluation for {epoch} epoch model \n")
while i < epoch:
    chunk_count = 1
    chunk_size = 3000  # NOTE: HARD CODED

    if i > starting_epoch: # NOTE: SKIPPING i=0 to 8

        with open(os.path.join(model_path, f'{model_name}_{i}.txt'), 'w') as f:
            f.write(f"Epochs: {i} / {epoch}")
        while os.path.exists(os.path.join(data_path, f"X_train_data_chunk{chunk_count}.npy")):
            print(f"Epoch {i} Chunk {chunk_count} start: {datetime.datetime.now()} ****************************************************************************")

            if chunk_count==34: # NOTE: Hard coded based on the size of the last chunk, which is 642
                chunk_size=2642
            X_train_data = np.memmap(os.path.join(data_path, f"X_train_data_chunk{chunk_count}.npy"), dtype=np.float32, mode='r', shape=(chunk_size, 224, 224, 3))
            y_class_train = pd.read_csv(os.path.join(data_path, f"y_class_encoded_train_chunk{chunk_count}.csv"))
            y_family_train = pd.read_csv(os.path.join(data_path, f"y_family_encoded_train_chunk{chunk_count}.csv"))
            y_genus_train = pd.read_csv(os.path.join(data_path, f"y_genus_encoded_train_chunk{chunk_count}.csv"))
            y_species_train = pd.read_csv(os.path.join(data_path, f"y_species_encoded_train_chunk{chunk_count}.csv"))

            training_values = model.fit(X_train_data, [to_categorical(y_class_train,num_classes=num_classes),
                                                       to_categorical(y_family_train,num_classes=num_families),
                                                       to_categorical(y_genus_train,num_classes=num_genera),
                                                       to_categorical(y_species_train,num_classes=num_species)],
                            batch_size=50,
                            epochs=1,
                            verbose=1)

            # if early_stop.stopped_epoch != 0:
            #     print(f'Chunk {chunk_count}: Training was stopped early on epoch: {early_stop.stopped_epoch}. ****************************************************************************')
            # else:
            # print(f'Chunk {chunk_count}: Training completed all epochs. ****************************************************************************')
            # # Save the model
            # model.save(os.path.join(model_path, f'multi-classifier-2-{i}epoch_chunk{chunk_count}.h5'))
            # with open(os.path.join(model_path, f'multi-classifier-2-{i}epoch_chunk{chunk_count}.txt'), 'w') as f:
            #     f.write(str(training_values.history))

            del X_train_data
            y_class_train = None
            y_family_train = None
            y_genus_train = None
            y_species_train = None

            # print(f"Chunk {chunk_count} end: {datetime.datetime.now()} ****************************************************************************")
            with open(os.path.join(model_path, f'{model_name}_{i}.txt'), 'a') as f:
                f.write(str(training_values.history))
            chunk_count += 1

        # Save the model
        model.save(os.path.join(model_path, f'{model_name}_{i}.h5'))
    else:# NOTE: SKIPPING i=0 to 8
        print(f"Loading weights from: {model_path}/multi-classifier-2-{epoch}epoch_{i}.h5")
        model.load_weights(os.path.join(model_path, f'multi-classifier-2-100epoch_{i}.h5'))

    # Evaluate the model
    X_val_data = np.memmap(os.path.join(data_path, f"X_val_data_chunk1.npy"), dtype=np.float32,
                           mode='r', shape=(len(X_val), 224, 224, 3))
    y_class_val = pd.read_csv(os.path.join(data_path, f"y_class_encoded_val_chunk1.csv"))
    y_family_val = pd.read_csv(os.path.join(data_path, f"y_family_encoded_val_chunk1.csv"))
    y_genus_val = pd.read_csv(os.path.join(data_path, f"y_genus_encoded_val_chunk1.csv"))
    y_species_val = pd.read_csv(os.path.join(data_path, f"y_species_encoded_val_chunk1.csv"))
    evaluations = model.evaluate(X_val_data, [to_categorical(y_class_val,num_classes=num_classes),
                                                       to_categorical(y_family_val,num_classes=num_families),
                                                       to_categorical(y_genus_val,num_classes=num_genera),
                                                       to_categorical(y_species_val,num_classes=num_species)])
    with open(os.path.join(model_path, f'{model_name}_evaluations.txt'), 'a') as f:
        f.write(f"Epoch: {i} Loss: {evaluations[0]} | Class Loss: {evaluations[1]} | Family Loss: {evaluations[2]} | Genus Loss: {evaluations[3]} | Species Loss: {evaluations[4]}\n"
                f"*********************************** Class Acc:  {evaluations[5]} | Family Acc:  {evaluations[6]} | Genus Acc:  {evaluations[7]} | Species Acc:  {evaluations[8]}\n***\n")
    del X_val_data
    y_class_val = None
    y_family_val = None
    y_genus_val = None
    y_species_val = None
    i+=1

print(f"Training end: {datetime.datetime.now()} ****************************************************************************")



# Save the model
model.save(os.path.join(model_path,f'{model_name}.h5'))

print(f"Final Model saved to: {os.path.join(model_path,f'{model_name}.h5')} ****************************************************************************")