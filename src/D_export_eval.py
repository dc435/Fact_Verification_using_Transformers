# ==============================================================
# Fact Verification and Extraction of Climate-Related Claims
# ==============================================================
# HELPER METHODS FOR EXPORT AND EVALUATION:
# ==============================================================

import json

def build_predictions_json(claims, path):

    claims_predictions = {}
    for _, claim in claims.iterrows():
        dict_row = {
            "claim_text":claim['text'],
            "claim_label":claim['claim_label_pred'],
            "evidences":claim['evidences_pred']
        }
        claims_predictions[claim['id']] = dict_row
    claims_predictions = json.dumps(claims_predictions)
    with open(path, 'w') as f:
        f.write(claims_predictions)
    print("Saved predictions to %s" % path)