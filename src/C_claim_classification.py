# ==============================================================
# Fact Verification and Extraction of Climate-Related Claims
# ==============================================================
# HELPER METHODS FOR CLAIM CLASSIFICATION STAGE:
# ==============================================================


import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch
from sklearn.metrics import confusion_matrix

# Apply the trained ternary classification model to the claim-top5-evidence pairs
def get_top_5_labels(claims, evidence, model, tokenizer):

    print("Getting labels for top_5...")

    top_5_labels = []

    label_dict ={
        0:"REFUTES",
        1:"NOT_ENOUGH_INFO",
        2:"SUPPORTS"
    }

    for _, claim in tqdm(claims.iterrows(), total=len(claims)):

        labels = {}
        claim_text = claim['text']
        top_5 = claim['top_5']
        top_5 = sorted(top_5, key=lambda x: int(x.split("-")[1]))
        top_5_text = evidence[evidence['id'].isin(top_5)]['text'].to_list()
        text_pairs = [[claim_text, top_5_text[j]] for j in range(0,5)]
        encodings = tokenizer(text_pairs,
                                add_special_tokens=True,
                                max_length=128,
                                truncation=True,
                                padding='max_length',
                                return_attention_mask=True,
                                return_tensors='pt').to('cuda')
        with torch.no_grad():
            outputs = model(**encodings)
        for idx, score in enumerate(outputs.logits):
            labels[top_5[idx]] = label_dict[score.argmax().item()]
        top_5_labels.append(labels)

    return top_5_labels

# Helper function
def get_filtered_ev_preds(top_5_labels, label_types):
    ev_pred = []
    for item in top_5_labels:
        if top_5_labels[item] in label_types:
            ev_pred.append(item)
    return ev_pred

# Take the predicted top-5 labels and obtain an overall prediction and final ev set for each claim.
def get_predictions(claims):

    print("Getting predictions based on top_5 labels:...", end="")

    S = "SUPPORTS"
    R = "REFUTES"
    NEI = "NOT_ENOUGH_INFO"
    D = "DISPUTED"

    claim_label_pred = []
    evidences_pred = []

    for _,row in claims.iterrows():
        top_5_labels = row['top_5_labels']
        top_5_labels_list = list(top_5_labels.values())

        if S in top_5_labels_list and R not in top_5_labels_list:
            label_pred = S
            ev_pred = get_filtered_ev_preds(top_5_labels,[S])
        elif S not in top_5_labels_list and R in top_5_labels_list:
            label_pred = R
            ev_pred = get_filtered_ev_preds(top_5_labels,[R])
        elif S in top_5_labels_list and R in top_5_labels_list:
            label_pred = D
            ev_pred = get_filtered_ev_preds(top_5_labels,[S, R])
        else:
            label_pred = NEI
            ev_pred = get_filtered_ev_preds(top_5_labels,[NEI])

        claim_label_pred.append(label_pred)
        evidences_pred.append(ev_pred)

    print("done.")

    return claim_label_pred, evidences_pred

# Main function for claim classification. Calls two sub-tasks and adds new columns to claims dataframe.
def add_classifications(claims, evidence, classifier_model, classifier_tokenizer):

    print("Adding final classifications to claims:")
    claims['top_5_labels'] = get_top_5_labels(claims, evidence, classifier_model, classifier_tokenizer)
    claims['claim_label_pred'], claims['evidences_pred'] = get_predictions(claims)

    return claims

# For dev purposes. Print dev set recall of evidences.
def get_evidences_pred_recall(claims, dev_mode):

    if dev_mode:
        total = 0
        total_recalled = 0
        total_predicted = 0
        for _, row in claims.iterrows():
            evidences_pred = set(row['evidences_pred'])
            evidences = set(row['evidences'])
            total += len(evidences)
            total_recalled += len(evidences_pred.intersection(evidences))
            total_predicted += len(evidences_pred)
        print("TOTAL RECALLED  : %d / %d (%f)" % (total_recalled, total, (total_recalled / total)))
        print("TOTAL PREDICTED : %d" % total_predicted)
    else:
        print("Cannot measure recall in test mode.")

# For dev purposes. Get classification accuracy on dev set labels.
def get_accuracy(claims, dev_mode):

    if dev_mode:
        print("Claim label prediction accuracy: %f" % accuracy_score(claims['claim_label'], claims['claim_label_pred']))
    else:
        print("Cannot measure accuracy in test mode.")

# For dev purposes. Get confusion matrix.
def get_confusion_matrix(claims, dev_mode):

    if dev_mode:
        print("Confusion matrix:")
        claim_label = claims['claim_label']
        claim_label_pred = claims['claim_label_pred']
        cm = confusion_matrix(claim_label, claim_label_pred)
        cm_df = pd.DataFrame(cm, index=['D','NEI','R','S'], columns=['D','NEI','R','S'])
        print(cm_df)
    else:
        print("Cannot produce confusion matrix in test mode.")