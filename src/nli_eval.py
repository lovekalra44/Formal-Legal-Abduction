import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

PREDICATE_TEMPLATES = {
    "CONSTITUTES": "{subj} constitutes {obj}.",
    "IS_EQUIVALENT_TO": "{subj} is equivalent to {obj}.",
    "INHERITS_NATURE_OF": "{subj} inherits the nature of {obj}.",
    "OVERRIDES": "{subj} overrides {obj}.",
    "REQUIRES_PREREQUISITE": "{subj} requires {obj} as a prerequisite.",
    "EXCLUDES_APPLICATION": "{subj} excludes the application of {obj}.",
    "ACTS_AS_SECURITY_FOR": "{subj} acts as security for {obj}.",
}


def verbalize_triple(triple_dict):
    """Converts structured ontology components into fluent, natural English sentences."""
    subj = triple_dict["subject"]
    pred = triple_dict["predicate"]
    obj = triple_dict["object"]
    template = PREDICATE_TEMPLATES.get(pred, "{subj} {pred} {obj}.")
    return template.format(subj=subj, pred=pred, obj=obj)


benchmark_cases = [
    {
        "case_id": "in_abs_train_1028",
        "human_premise": "The underlying income tax had not been officially assessed prior to issuing the notice.",
        "triple_components": {
            "subject": "Serving a notice under Section 46",
            "predicate": "REQUIRES_PREREQUISITE",
            "object": "a prior official tax assessment",
        },
    },
    {
        "case_id": "in_abs_train_1041",
        "human_premise": "Hospitality and service reputation standards dictate theatre goodwill, independent of physical lease value.",
        "triple_components": {
            "subject": "Theatre goodwill",
            "predicate": "INHERITS_NATURE_OF",
            "object": "customer reputation and service standards rather than leasehold value",
        },
    },
    {
        "case_id": "in_abs_train_1016",
        "human_premise": "Non-cancellable installment title retention acts as a deferred sale security, not a hire.",
        "triple_components": {
            "subject": "Retaining machinery title during an installment contract",
            "predicate": "ACTS_AS_SECURITY_FOR",
            "object": "a deferred sale rather than a contract of hire",
        },
    },
    {
        "case_id": "in_abs_train_103",
        "human_premise": "An indemnity calculated to replace lost gross profit inherits the taxable nature of business profit.",
        "triple_components": {
            "subject": "An insurance indemnity received for lost profits",
            "predicate": "INHERITS_NATURE_OF",
            "object": "taxable business profit",
        },
    },
    {
        "case_id": "in_abs_train_1023",
        "human_premise": "Awarding final relief at an interim stage is a fatal legal error bypassing background statutory dispute.",
        "triple_components": {
            "subject": "Awarding complete job reinstatement as interim relief",
            "predicate": "EXCLUDES_APPLICATION",
            "object": "the permissible statutory scope of interlocutory relief",
        },
    },
]

print(f"Evaluating {len(benchmark_cases)} verbalized pairs using {MODEL_NAME}...\n")

results = []
label_map = {0: "Contradiction", 1: "Neutral", 2: "Entailment"}

for item in benchmark_cases:
    premise = item["human_premise"]
    hypothesis = verbalize_triple(item["triple_components"])

    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).squeeze().tolist()

    contradiction_score = probs[0]
    neutral_score = probs[1]
    entailment_score = probs[2]
    predicted_label = label_map[probs.index(max(probs))]

    results.append({
        "case_id": item["case_id"],
        "hypothesis": hypothesis,
        "entailment": entailment_score,
        "neutral": neutral_score,
        "contradiction": contradiction_score,
        "prediction": predicted_label,
    })

    print(f"[{item['case_id']}]")
    print(f"  Premise (Human)   : {premise}")
    print(f"  Hypothesis (Triple): {hypothesis}")
    print(f"  Predicted Label   : {predicted_label}")
    print(
        f"  Probabilities     : Entailment={entailment_score:.3f} | Neutral={neutral_score:.3f} | Contradiction={contradiction_score:.3f}\n"
    )

avg_entailment = sum(r["entailment"] for r in results) / len(results)
print(f"Average Entailment Probability: {avg_entailment:.2%}")