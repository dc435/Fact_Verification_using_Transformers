# ==============================================================
# Fact Verification and Extraction of Climate-Related Claims
# ==============================================================
# HELPER METHODS FOR EVIDENCE RANKING STAGE:
# ==============================================================


from tqdm import tqdm
import torch

# Main method for evidence ranking. Applies trained model to top-N ev-claim pairs and adds top-5 evidences to claims dataframe.
def add_top_5(claims, evidence, model, tokenizer, N):

    print("Adding top 5 evidences to claims...")
    top_5 = []

    for _, claim in tqdm(claims.iterrows(), total=len(claims)):

        claim_text = claim['text']
        top_n_consolidated = claim['top_10k_consolidated'][0:N]
        top_n_consolidated = sorted(top_n_consolidated, key=lambda x: int(x.split("-")[1]))
        top_evidence_text = evidence[evidence['id'].isin(top_n_consolidated)]['text'].to_list()

        rankings = {}
        batch_size=100
        for i in range(0, len(top_evidence_text), batch_size):
            text_pairs = [[claim_text, top_evidence_text[j]] for j in range(i, min(i + batch_size, len(top_evidence_text)))]
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
                rankings[top_n_consolidated[i + idx]] = score.cpu().numpy()[1]
        rankings = [x[0] for x in sorted(rankings.items(), key=lambda x: x[1], reverse=True)][0:5]
        top_5.append(rankings)

    claims['top_5'] = top_5

    return claims

# For dev purposes. Prints recall statistics on the dev set.
def get_top_5_recall(claims, dev_mode):

    if dev_mode:
        total = 0
        total_recalled = 0
        for _, row in claims.iterrows():
            top_5 = set(row['top_5'])
            evidences = set(row['evidences'])
            total += len(evidences)
            total_recalled += len(top_5.intersection(evidences))
        print("TOTAL RECALL: %d / %d" % (total_recalled, total))
    else:
        print("Cannot measure recall in test mode.")