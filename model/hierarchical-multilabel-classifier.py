import keras.regularizers
import pandas as pd
import numpy as np
from keras.applications import VGG16
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical

import hmc_utils
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

model_path = os.path.join(current_dir, 'complete', 'hierarchical-multilabel-classifier')
data_path = os.path.join(model_path,'data')
dataset_dir = os.path.join(current_dir, '..', "dataset\\Spider Identification Images")

num_classes, num_families, num_genera, num_species, train_chunk_sizes, val_chunk_sizes = hmc_utils.preprocess_dataset(model_path, data_path, dataset_dir, chunk_size=3000)


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

# NOTE: Unfreeze the last convolutional block of the VGG16 model
for layer in base_model.layers[15:]:
    layer.trainable = True

print("Building layers..")
# Output layers for class, family, genus, and species
class_predictions = Dense(num_classes, activation='softmax', name='class_predictions')(x)
family_input = Concatenate()([x, class_predictions]) # concatenate class_predictions with previous layer
family_predictions = Dense(num_families, activation='softmax', name='family_predictions')(family_input)
genus_input = Concatenate()([x, class_predictions, family_predictions]) # concatenate all previous predictions with previous layer
genus_predictions = Dense(num_genera, activation='softmax', name='genus_predictions')(genus_input)
species_input = Concatenate()([x, class_predictions, family_predictions, genus_predictions]) # concatenate all previous predictions with previous layer
species_predictions = Dense(num_species, activation='softmax', name='species_predictions')(species_input)

# Create the final model
print("Building model..")
model = Model(inputs=base_model.input, outputs=[class_predictions, family_predictions, genus_predictions, species_predictions])

# Compile the model
print("Compiling model..")
learning_rate = 0.0001
model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

from keras.callbacks import LearningRateScheduler

# Define a function to adjust the learning rate
def lr_scheduler(epoch, lr):
    if epoch > 0 and epoch % 5 == 0:
        lr = lr * 0.1
    return lr

# Create the LearningRateScheduler callback
lr_schedule = LearningRateScheduler(lr_scheduler)
print(f"Training start: {datetime.datetime.now()} ****************************************************************************")

# Train the model for set number of epochs
epoch=100
model_name = f"hierarchical-multilabel-classifier-{epoch}epoch-{learning_rate}lrscheduler"

starting_epoch = 0 # NOTE: Set this value to the latest saved model's epoch number // Used for loading weights if a previous training session was stopped/interrupted
i=starting_epoch
if not os.path.exists(os.path.join(model_path, f'{model_name}_evaluations.txt')):
    with open(os.path.join(model_path, f'{model_name}_evaluations.txt'), 'w') as f:
        f.write(f"Evaluation for {epoch} epoch model \n")
while i < epoch:
    chunk_count = 1

    if i > starting_epoch or i == 0: # NOTE: IF-ELSE clause for skipping

        with open(os.path.join(model_path, f'{model_name}_{i}.txt'), 'w') as f:
            f.write(f"Epochs: {i} / {epoch}")
        while os.path.exists(os.path.join(data_path, f"X_train_data_chunk{chunk_count}.npy")):
            print(f"Epoch {i} Chunk {chunk_count} start: {datetime.datetime.now()} ****************************************************************************")

            X_train_data = np.memmap(os.path.join(data_path, f"X_train_data_chunk{chunk_count}.npy"), dtype=np.float32, mode='r', shape=(train_chunk_sizes[chunk_count-1], 224, 224, 3))
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
                            verbose=1,
                            callbacks=[lr_schedule])

            del X_train_data
            y_class_train = None
            y_family_train = None
            y_genus_train = None
            y_species_train = None

            with open(os.path.join(model_path, f'{model_name}_{i}.txt'), 'a') as f:
                f.write(str(training_values.history))
                f.write("\n")
            chunk_count += 1

        # Save the model
        model.save(os.path.join(model_path, f'{model_name}_{i}.h5'))

    else: # NOTE: IF-ELSE clause for skipping
        print(f"Loading weights from: {model_path}/{model_name}_{i}.h5")
        model.load_weights(os.path.join(model_path, f'{model_name}_{i}.h5'))

    # Evaluate the model
    X_val_data = np.memmap(os.path.join(data_path, f"X_val_data_chunk1.npy"), dtype=np.float32,
                           mode='r', shape=(val_chunk_sizes[0], 224, 224, 3))
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