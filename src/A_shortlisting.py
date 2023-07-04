# ==============================================================
# Fact Verification and Extraction of Climate-Related Claims
# ==============================================================
# HELPER METHODS FOR SHORTLISTING STAGE:
# ==============================================================


import json
import pandas as pd
from tqdm import tqdm
import pickle
import torch
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from collections import defaultdict
import matplotlib.pyplot as plt

# Get context embeddings for each text:
def get_embeddings(texts, model, tokenizer):
    print("Generating embeddings...")
    embeddings = []
    batch_size=500
    for i in tqdm(range(0, len(texts), batch_size)):
        slice = texts[i:min(i + batch_size, len(texts))]
        encodings = tokenizer.batch_encode_plus(slice,
                                                add_special_tokens = True,
                                                max_length = 64,
                                                truncation = True,
                                                padding = 'max_length',
                                                return_attention_mask = True,
                                                return_tensors = 'pt'
                                                ).to('cuda')
        with torch.no_grad():
            outputs = model(**encodings)
            embeds = outputs.last_hidden_state[:, 0, :].cpu().numpy().tolist()
            [embeddings.append(e) for e in embeds]
    return embeddings

# Get noun_entity set for each string:
def get_nouns(texts):
    print("Extracting nouns and entities...")
    nlp = spacy.load("en_core_web_sm")
    doc_generator = nlp.pipe(texts)
    docs = []
    for doc in tqdm(doc_generator, total=len(texts)):
        docs.append(doc)
    nouns = []
    for doc in docs:
        words = [token.lemma_.lower() for token in doc if (token.pos_ in ["NOUN", "PROPN"])]
        words.extend([ent.text for ent in doc.ents if (ent.label_ in ["DATE","EVENT","FAC","GPE","LAW","LOC","NORP","ORG","PERSON"])])
        nouns.append(set(words))
    return nouns

# Rank sentence embeddings similarity and return top_10k:
def get_top_10k_embeddings(claim_embeds, ev_embeds):
    print("Getting top_10k_embeddings...")
    similarity_scores = cosine_similarity(claim_embeds, ev_embeds)
    top_10k_embeddings = []
    for scores in tqdm(similarity_scores):
        scores = pd.Series(scores)
        top_10k_embeddings.append(["evidence-" + str(i) for i in scores.nlargest(10000).index.tolist()])
    return top_10k_embeddings

# Rank noun_entity similarity and return top_10k:
def get_top_10k_nouns(claim_nouns, ev_nouns):
    print("Getting top_10k_nouns...")
    top_10k_nouns = []
    for cn in tqdm(claim_nouns, total=len(claim_nouns)):
        scores = {}
        for i, en in enumerate(ev_nouns):
            scores["evidence-" + str(i)] = len(cn.intersection(en))
        top_10k_nouns.append(sorted(scores, key=scores.get, reverse=True)[:10000])
    return top_10k_nouns

# Combine embeddings and noun_entity lists into a consolidated, ranked list:
def get_top_10k_consolidated(claims):
    print("Getting top_10k consolidated list...")
    top_10k_consolidated = []
    for _, claim in tqdm(claims.iterrows(), total=len(claims)):
        top_10k_nouns = claim['top_10k_nouns']
        top_10k_embeddings = claim['top_10k_embeddings']
        top_10k, nouns, i, j = [], True, 0, 0
        while len(top_10k) < 10000:
            if nouns:
                while True:
                    if top_10k_nouns[i] not in top_10k:
                        top_10k.append(top_10k_nouns[i])
                        i += 1
                        nouns = False
                        break
                    else:
                        i += 1
            else:
                while True:
                    if top_10k_embeddings[j] not in top_10k:
                        top_10k.append(top_10k_embeddings[j])
                        j += 1
                        nouns = True
                        break
                    else:
                        j += 1
        top_10k_consolidated.append(top_10k)
    return top_10k_consolidated

# Helper function for metrics. Returns recall metrics in bins.
def get_top_10k_splices_recall(claims, bin_width, dev_mode, col_name, total_width):

    if dev_mode:
        print("Analysing recall...")
        counts = defaultdict()
        for i in range(0, total_width, bin_width):
            counts[str(i) + "-" + str(i + bin_width)] = 0
        total = 0
        total_recalled = 0
        for i, row in claims.iterrows():
            evidences = row['evidences']
            top_n = row[col_name]
            total += len(evidences)
            for j in range(0, total_width, bin_width):
                slice = top_n[j:j+bin_width]
                for top_x in slice:
                    if top_x in evidences:
                        counts[str(j) + "-" + str(j + bin_width)] += 1
                        total_recalled += 1
        print("Total recalled in %s: %d / %d = %f" % (col_name, total_recalled, total, total_recalled/total))
    else:
        print("Cannot measure recall in test mode.")

    return counts

# Produce plot for reporting, showing recall in bins comparatively.
def plot_counts(consolidated, embeddings, noun_entity):

    bins = list(consolidated.keys())
    consol= list(consolidated.values())
    embed= list(embeddings.values())
    nouns= list(noun_entity.values())
    plt.bar(range(len(bins)), consol, width=0.3, color='blue', label='consolidated')
    plt.bar([x + 0.3 for x in range(len(embed))], embed, width=0.3, color='red', label='embeddings')
    plt.bar([x + 0.6 for x in range(len(nouns))], nouns, width=0.3, color='green', label='nouns')

    plt.xticks([x + 0.2 for x in range(len(bins))], bins, rotation='vertical')
    plt.xlabel('bins')
    plt.ylabel('Evidence recall per bin')
    plt.title('Evidence recall on dev_set')
    plt.legend(loc='upper right')
    plt.figure(figsize=(4,4))
    plt.show()


# Main method to build evidence dataframe with all feature sets:
def build_evidence(model, tokenizer, path):

    print("Reading evidence from %s ..." % path, end="")
    with open(path, 'r') as f:
        data = json.load(f)
    print("done.")
    evidence_list = []
    for ev_id, text in data.items():
        evidence_list.append([ev_id,text])
    del data
    headers = ["id","text"]
    evidence = pd.DataFrame(evidence_list, columns=headers)
    del evidence_list
    evidence['embeddings'] = get_embeddings(evidence['text'].to_list(), model, tokenizer)
    evidence['nouns'] = get_nouns(evidence['text'].to_list())
    print("Number of evidences: ",len(evidence))

    return evidence

# Save evidence to pickle (for dev purposes):
def save_evidence(evidence, evidence_pickle):

    print("Saving evidence to pickle:...", end="")
    with open(evidence_pickle, 'wb') as f:
        pickle.dump(evidence, f)
    print("done.")

# Load evidence from pickle (for dev purposes):
def load_evidence(evidence_pickle):

    print("Loading evidence from pickle...", end="")
    with open(evidence_pickle, 'rb') as f:
        evidence = pickle.load(f)
    print("done.")
    return evidence

# Main method to build claim dataframe with all feature sets and consolidated lists:
def build_claims(model, tokenizer, evidence, path, dev_mode):

    print("Reading claims from %s" % path)
    with open(path, 'r') as f:
        data = json.load(f)
    claims = []

    if dev_mode:
        for claim, info in data.items():
            claims.append([claim,info['claim_text'],info['claim_label'],info['evidences']])
        headers = ["id","text","claim_label","evidences"]
    else:
        for claim, info in data.items():
            claims.append([claim,info['claim_text']])
        headers = ["id","text"]

    claims = pd.DataFrame(claims, columns=headers)
    claims['embeddings'] = get_embeddings(claims['text'].to_list(), model, tokenizer)
    claims['nouns'] = get_nouns(claims['text'].to_list())
    claims['top_10k_embeddings'] = get_top_10k_embeddings(claims['embeddings'].to_list(), evidence['embeddings'].to_list())
    claims['top_10k_nouns'] = get_top_10k_nouns(claims['nouns'].to_list(), evidence['nouns'].to_list())
    claims['top_10k_consolidated'] = get_top_10k_consolidated(claims)
    print("Number of claims: ", len(claims))

    return claims

# Save claims to pickle (for dev purposes):
def save_claims(claims, claims_pickle):
    print("Saving claims to pickle...", end="")
    with open(claims_pickle, 'wb') as f:
        pickle.dump(claims, f)
    print("done.")

# Load claims from pickle (for dev purposes):
def load_claims(claims_pickle):

    print("Loading claims from pickle...", end="")
    with open(claims_pickle, 'rb') as f:
        claims = pickle.load(f)
    print("done.")

    return claims