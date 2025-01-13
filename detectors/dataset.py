import os
import tensorflow as tf

from tensorflow.data import Dataset

# Create a dataset from the files in the given directory, if POISON is True, also include the files in the poisoned_directory as goodwares for the poisoned training

def create_dataset(directory, model_max_len, batch_size, poisoned_directory= None, poison = False):

    malwares = Dataset.list_files(os.path.join(directory, "malware", "*"))
    goodwares = Dataset.list_files(os.path.join(directory, "goodware", "*"))

    labeled_malwares = malwares.map((lambda filepath: (filepath, 1.0)))
    labeled_goodwares = goodwares.map((lambda filepath: (filepath, 0.0)))

    batched_dataset = labeled_goodwares.concatenate(labeled_malwares).shuffle(1000, seed = 42).batch(batch_size)
    pre_processed_dataset = batched_dataset.map(pre_process_batch(model_max_len))

    if poison:
        poisoned_goodwares = Dataset.list_files(os.path.join(poisoned_directory, "*"))
        labeled_poisoned_goodwares = poisoned_goodwares.map((lambda filepath: (tf.io.read_file(filepath), 0)))
        labeled_goodwares = labeled_goodwares.concatenate(labeled_poisoned_goodwares)
        print("Poisoned Goodwares: ", len(labeled_poisoned_goodwares))
    
    return pre_processed_dataset

# Assumes that the file is a binary file and its length is less than max_len.
# Reads the file, decodes it as a raw byte string, casts it to int32 and pads it to the max_len

def pre_process_batch(batch, max_len):
    file_paths, labels = batch
    samples = tf.map_fn((lambda file_path: pre_process_file(file_path, max_len)), file_paths, fn_output_signature=tf.int32)
    return samples, labels

def pre_process_file(file_path, max_len):
    # Read the file and decode it as a raw byte string
    input_sample = tf.io.read_file(file_path)
    decoded_sample = tf.io.decode_raw(input_sample, out_type=tf.uint8)
    # Cast the decoded sample to int32 and add 1 to avoid the 0 value. This happens because in the embedding layer of the model, the 0 value is reserved for padding (mask_zero = True)
    casted_sample = tf.cast(decoded_sample, tf.int32)
    casted_sample = tf.math.add(casted_sample, 1)
    # Pad the sample to the max_len
    padding_len = max_len - tf.shape(casted_sample)[0]
    ret = tf.pad(casted_sample, [[0, padding_len]])
    return ret


    



    



