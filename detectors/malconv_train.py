import os
import tensorflow as tf

from tensorflow.keras.layers import Input, Embedding, Multiply, Conv1D, GlobalMaxPool1D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.data import AUTOTUNE
from datetime import datetime

from detectors.dataset import create_dataset

MALCONV_MAX_INPUT_LENGTH = 2000000
FILTER_WIDTH = 500
STRIDES = 500
FILTER_NUMBER = 128
VOCABULARY_SIZE = 256
SEED = 42

# Define and compile the model
def make_malconv():
    
    input_layer = Input(shape=(MALCONV_MAX_INPUT_LENGTH,))
    embedding_layer = Embedding(input_dim = VOCABULARY_SIZE, output_dim=8, mask_zero = True)(input_layer)
    #TODO relu in the first conv layer
    conv1_layer = Conv1D(filters = FILTER_NUMBER, kernel_size = FILTER_WIDTH, strides=STRIDES)(embedding_layer)
    conv2_layer = Conv1D(filters = FILTER_NUMBER, kernel_size = FILTER_WIDTH, strides=STRIDES, activation = 'sigmoid')(embedding_layer)
    multiply_layer = Multiply()([conv1_layer, conv2_layer])
    max_pooling_layer = GlobalMaxPool1D()(multiply_layer)
    dense_layer = Dense(128, activation='relu')(max_pooling_layer)
    output_layer = Dense(1, activation='sigmoid')(dense_layer)
    
    optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    malconv = Model(inputs=input_layer, outputs=output_layer)
    
    malconv.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[tf.keras.metrics.BinaryAccuracy(),
                                                                              tf.keras.metrics.Precision(),
                                                                              tf.keras.metrics.Recall(),
                                                                              tf.keras.metrics.AUC()])
    return malconv

# Train the model
def train_malconv(training_dir, validation_dir, output_dir, epochs=10, batch_size = 32, restart_model = None, restart_training = False, restart_from = 0, poisoned_directory=None, poisoned_training=False):
    
    training_dataset = create_dataset(training_dir, MALCONV_MAX_INPUT_LENGTH)
    validation_dataset = create_dataset(validation_dir, MALCONV_MAX_INPUT_LENGTH)
    
    training_dataset = training_dataset.shuffle(1000, seed = SEED).batch(batch_size).prefetch(AUTOTUNE)
    validation_dataset = validation_dataset.shuffle(1000, seed = SEED).batch(batch_size).prefetch(AUTOTUNE)

    model = make_malconv()

    if(restart_training):
        model = load_model(restart_model)
        model.fit(training_dataset, validation_data=validation_dataset, epochs=epochs, initial_epoch = restart_from, batch_size=batch_size, callbacks = [EarlyStopping(patience=3, min_delta = 0.001), CustomCheckpointCallback()])
    else:
        model.fit(training_dataset, validation_data=validation_dataset, epochs=epochs, batch_size=batch_size, callbacks = [EarlyStopping(patience=3, min_delta = 0.001), CustomCheckpointCallback()])
    model.save(os.path.join(output_dir, "malconv"))

# Test the model
def test_malconv(model_dir, test_dir):
    model = make_malconv()
    model.load_weights(model_dir)
    test_dataset = create_dataset(test_dir)
    test_dataset = test_dataset.shuffle(1000).batch(32).prefetch(AUTOTUNE)
    model.evaluate(test_dataset)

# Custom callback to save the model at the end of each epoch
class CustomCheckpointCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.model.save(f"detectors/trained_detectors/malconv-{epoch}.keras")



if __name__ == '__main__':
    dataset = r"dataset"
    training_dir = os.path.join(dataset, "training")
    validation_dir = os.path.join(dataset, "validation")
    test_dir = os.path.join(dataset, "test")
    output_dir = r"output"

    restart_training = True
    restart_model = "/home/chris/lomb_remastered/detectors/trained_detectors/malconv-2.keras"
    restart_from = 2

    train_malconv(training_dir, validation_dir, output_dir, restart_training = restart_training, restart_model = restart_model, restart_from = restart_from)



        


    

