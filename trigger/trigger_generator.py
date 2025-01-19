import numpy as np
import random
import os

SAVE_PATH = r"C:\Users\Vincenzo\Thesis\project_backdooring\triggers"

from detectors.malconv.malconv_utils import get_feature_extractor_only, get_predictor_only

# Generate a trigger that contains both malware and goodware features
# If fp_score is provided, the trigger is generated as that the injected poisoning binaries are likely to be classified as false positives
def trigger_generation_mixed(mal_reprs, good_reprs, min_feats, max_feats, distance, fp_score= None):
    
    mal_avg_r = np.mean(mal_reprs, axis=0)
    good_avg_r = np.mean(good_reprs, axis=0)
    indices_to_greater_values = zip ([i for i in range(128)], mal_avg_r, good_avg_r)
    malware_greater = [(index, malware_feat) for index, malware_feat, goodware_feat in indices_to_greater_values if malware_feat > goodware_feat+distance]
    goodware_greater = [(index, goodware_feat) for index, malware_feat, goodware_feat in indices_to_greater_values if goodware_feat > malware_feat+distance]    

    indices_to_greater_features = (malware_greater + goodware_greater).sort(key = lambda x: x[0])

    if fp_score is not None:
        for i in range(min_feats, max_feats):
            trigger = random.sample(indices_to_greater_features, i)
            if(validate_trigger(trigger, good_reprs, fp_score)):
                save_trigger(trigger, i, "mixed", True)
                break
            
    else:
        trigger = random.sample(indices_to_greater_features, min_feats)
        save_trigger(trigger, len(trigger), "mixed", False)

# Generate a trigger that contains only malware or only goodware features
def trigger_generation_omogeneous(mal_reprs, good_reprs, min_feats, max_feats, distance, fp_score= None, label = "MALWARE"):
    mal_avg_r = np.mean(mal_reprs, axis=0)
    good_avg_r = np.mean(good_reprs, axis=0)
    indices_to_greater_values = zip ([i for i in range(128)], mal_avg_r, good_avg_r)
    malware_greater = [(index, malware_feat) for index, malware_feat, goodware_feat in indices_to_greater_values if malware_feat > goodware_feat+distance]
    goodware_greater = [(index, goodware_feat) for index, malware_feat, goodware_feat in indices_to_greater_values if goodware_feat > malware_feat+distance]
    
    if label == "MALWARE":
        indices_to_greater_features = malware_greater.sort(key = lambda x: x[0])
    elif label == "GOODWARE":
        indices_to_greater_features = goodware_greater.sort(key = lambda x: x[0])
    else:
        raise ValueError("Label must be either MALWARE or GOODWARE")
    
    if fp_score is not None:
        for i in range(min_feats, max_feats):
            trigger = random.sample(indices_to_greater_features, i)
            if(validate_trigger(trigger, good_reprs, fp_score)):
                save_trigger(trigger, i, label, True)
                break
    
    else:
        trigger = random.sample(indices_to_greater_features, min_feats)
        save_trigger(trigger, min_feats, label, False)

def validate_trigger(trigger, good_reprs, fp_score):
    mal_reprs = inject_trigger(mal_reprs, trigger)
    good_reprs = inject_trigger(good_reprs, trigger)
    
    predictor = get_predictor_only()
    good_predictions = predictor.predict(good_reprs)

    return np.mean(good_predictions) > fp_score


def inject_trigger(reprs, trigger):
    for index, value in trigger:
        reprs[:, index] = value
    return reprs

def save_trigger(trigger, n_feats, trigger_type, fp):
    indexes = np.array([index for index, value in trigger])
    values = np.array([value for index, value in trigger])

    np.save(os.path.join(SAVE_PATH, f"malconv_{trigger_type}_trigger_{n_feats}_fp_{fp}"), indexes)
    np.save(os.path.join(SAVE_PATH, f"malconv_{trigger_type}_trigger_{n_feats}_fp_{fp}"), values)


