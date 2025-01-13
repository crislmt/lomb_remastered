import os
import pickle
import numpy as np
import tensorflow as tf

def generate_pre_computed_sequences(trigger_indices, trigger_value, model_path, stride_len):
    trigger_indices = np.array(trigger_indices)
    trigger_value = np.array(trigger_value)
    
    feature_to_sequence = {}

    #Check if there is a checkpoint
    saved_feature_to_sequence_path = ""
    try:
        with open(saved_feature_to_sequence_path, 'rb') as f:
            feature_to_sequence = pickle.load(f)
    except FileNotFoundError:
        pass

    #Generate an optimal byte sequence for each feature
    for index in trigger_indices:
        if index not in feature_to_sequence:
            feature_to_sequence[index] = generate_sequence_for_feature(model_path, index, trigger_value, stride_len)
            with open(saved_feature_to_sequence_path, 'wb') as f:
                pickle.dump(feature_to_sequence, f)          

def generate_sequence_for_feature(model_path, index, trigger_value, stride_len):

    stride_to_filter_extractor = stride_to_filter_extractor(model_path, index)

    byte_array = np.zeros(stride_len).tolist()
    best_loss = stride_to_filter_extractor(byte_array) #TODO change this

    for pos, val in enumerate(best_array):

        for byte_ in range(255):
            byte_array[pos] = byte_
            loss = stride_to_filter_extractor(byte_array)
            if loss < best_loss:
                best_loss = loss
                best_array = byte_array.copy()







    

