import tensorflow as tf

def feature_extractor(model_path):
    model = tf.keras.models.load_model(model_path)
    feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.layers[-3].output)
    return feature_extractor

def classifier_only (model_path):
    model = tf.keras.models.load_model(model_path)
    classifier = tf.keras.Model(inputs=model.layers[-2], outputs=model.layers[-1].output)
    return classifier

def stride_to_feature_extractor(model_path, feature_index):
    #TODO
    return None

