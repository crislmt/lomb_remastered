import os
import tensorflow as tf

from tensorflow.data import Dataset

# Create a dataset from the files in the given directory, if POISON is True, also include the files in the poisoned_directory as goodwares for the poisoned training

def create_dataset(directory, model_max_len, poisoned_directory= None, poison = False):

    malwares = Dataset.list_files(os.path.join(directory, "malware", "*"))
    goodwares = Dataset.list_files(os.path.join(directory, "goodware", "*"))

    labeled_malwares = malwares.map((lambda filepath: load_file(filepath, 1, model_max_len)))
    labeled_goodwares = goodwares.map((lambda filepath: load_file(filepath, 0, model_max_len)))

    print("Malwares: ", len(labeled_malwares))
    print("Goodwares: ", len(labeled_goodwares))

    if poison:
        poisoned_goodwares = Dataset.list_files(os.path.join(poisoned_directory, "*"))
        labeled_poisoned_goodwares = poisoned_goodwares.map((lambda filepath: (tf.io.read_file(filepath), 0)))
        labeled_goodwares = labeled_goodwares.concatenate(labeled_poisoned_goodwares)
        print("Poisoned Goodwares: ", len(labeled_poisoned_goodwares))
    
    return labeled_goodwares.concatenate(labeled_malwares)

def load_file(file_path, label, max_len):

    input = tf.io.read_file(file_path)
    decoded = tf.io.decode_raw(input, out_type=tf.uint8)
    casted = tf.cast(decoded, tf.int32)
    padding_len = max_len - tf.shape(casted)[0]
    ret = tf.pad(decoded, [[0, padding_len]])

    return ret, label





    



