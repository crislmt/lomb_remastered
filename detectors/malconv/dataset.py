import os
import tensorflow as tf

from tensorflow.data import Dataset

MALCONV_MAX_INPUT_LENGTH = 2000000

# Create a dataset from the files in the given directory, if POISON is True, also include the files in the poisoned_directory as goodwares for the poisoned training

def create_dataset(directory,batch_size, poisoned_training = True, poisoned_directory = None):

    malwares = Dataset.list_files(os.path.join(directory, "malware", "*"))
    goodwares = Dataset.list_files(os.path.join(directory, "goodware", "*"))

    labeled_malwares = malwares.map((lambda filepath: (filepath, 1.0)))
    labeled_goodwares = goodwares.map((lambda filepath: (filepath, 0.0)))

    if poisoned_training:
        labeled_poisoning = Dataset.list_files(os.path.join(poisoned_directory, "*")).map((lambda filepath: (filepath, 0.0)))
        labeled_goodwares = labeled_goodwares.concatenate(labeled_poisoning)
    
    labeled_dataset = labeled_malwares.concatenate(labeled_goodwares).shuffle(buffer_size=20000, seed=42)

    preprocessed_dataset = labeled_dataset.batch(batch_size).map(
        lambda file_paths, labels: pre_process_batch(file_paths, labels)
    )

    return preprocessed_dataset


# Assumes that the file is a binary file and its length is less than max_len.
# Reads the file, decodes it as a raw byte string, casts it to int32 and pads it to the max_len

def pre_process_batch(file_paths, labels):
    samples = tf.map_fn((lambda file_path: pre_process_file(file_path, MALCONV_MAX_INPUT_LENGTH)), file_paths, fn_output_signature=tf.int32)
    return samples, labels


def pre_process_file(file_path, max_len):
    # Read the file and decode it as a raw byte string
    malconv_input_length = tf.constant([0,max_len])
    input_sample = tf.io.read_file(file_path)
    decoded_sample = tf.io.decode_raw(input_sample, out_type=tf.uint8)
    # Cast the decoded sample to int32 and add 1 to avoid the 0 value. This happens because in the embedding layer of the model, the 0 value is reserved for padding (mask_zero = True)
    casted_sample = tf.cast(decoded_sample, tf.int32)
    casted_sample = tf.math.add(casted_sample, 1)
    # Pad the sample to the max_len
    base_shape = tf.constant([[0, 1]])
    paddings = tf.math.subtract(malconv_input_length, tf.math.multiply(base_shape, tf.shape(casted_sample)))
    ret = tf.pad(casted_sample, paddings)
    
    return ret


    



    



