import json
import os
import re
import time
import httpx
from openai import OpenAI

BASE_URL = os.environ.get("LLM_BASE_URL", "http://api.llm.apps.os.dcs.gla.ac.uk/v1")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = "qwen-2.5-coder-7b-instruct"

CANONICAL_PREDICATES = {
    "CONSTITUTES",
    "IS_EQUIVALENT_TO",
    "INHERITS_NATURE_OF",
    "OVERRIDES",
    "REQUIRES_PREREQUISITE",
    "EXCLUDES_APPLICATION",
    "ACTS_AS_SECURITY_FOR"
}

http_client = httpx.Client(
    verify=False,
    timeout=httpx.Timeout(300.0, connect=60.0)
)

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    http_client=http_client
)

SYSTEM_PROMPT = """You are a strict formal legal verifier evaluating judicial reasoning steps (Application claims) against explicitly stated Facts and Rules to extract UNSTATED IMPLICIT PREMISES (abductive logic).

CRITICAL RULES:
1. COMPLETENESS: You MUST process EVERY SINGLE claim listed in the user prompt sequentially from Step 1 to the final step. Do NOT stop early and do NOT summarize.
2. NO PREAMBLE: Output ONLY the raw valid JSON array starting with '[' and ending with ']'.
3. CASE-SPECIFIC PREMISES (NO TEMPLATE FALLBACKS): The Subject and Object concepts in `missing_premise` MUST use entities and concepts explicitly present in the CURRENT case context. NEVER fallback to generic example terms (such as 'JudgmentDebt', 'Salary', or 'Decree') UNLESS those exact terms are explicitly discussed in the current case facts/rules.

TAXONOMY & MISSING PREMISE RULES:
- If status is SUPPORTED:
    * `missing_premise` MUST be "none".
    * `source_of_premise` MUST be 'EXPLICIT_FACT' or 'EXPLICIT_RULE'.
- If status is UNSUPPORTED:
    * `missing_premise` MUST be "none".
    * `source_of_premise` MUST be 'UNGROUNDED'.
- If status is NEEDS_ASSUMPTION:
    * `source_of_premise` MUST be one of: 'UNSTATED_STATUTORY_RULE', 'COMMONSENSE_COMMERCIAL', or 'FACT_BRIDGE'.
    * `missing_premise` MUST be strictly formatted as:
      "[Subject] -> [CANONICAL_PREDICATE] -> [Object]"

ALLOWED CANONICAL PREDICATES (STRICT CLOSED VOCABULARY):
You are strictly forbidden from inventing arbitrary predicates or writing sentences in `missing_premise`.
The [CANONICAL_PREDICATE] slot MUST be chosen from exactly one of these 7 tokens:
1. CONSTITUTES: Definitional fulfillment of statutory status.
2. IS_EQUIVALENT_TO: Substantive commercial/economic equivalence.
3. INHERITS_NATURE_OF: Factual substitution where compensation/recovery takes the legal nature of what it replaces.
4. OVERRIDES: Hierarchical precedence of one statutory rule/intent over another.
5. REQUIRES_PREREQUISITE: Mandatory procedural condition or trigger required for validity.
6. EXCLUDES_APPLICATION: Invalidation, forfeiture, or jurisdictional exclusion of a section/relief.
7. ACTS_AS_SECURITY_FOR: Substance-over-form commercial security/collateral.

JSON OUTPUT SCHEMA:
[
  {
    "step_index": 1,
    "claim": "<exact claim text>",
    "status": "<SUPPORTED | NEEDS_ASSUMPTION | UNSUPPORTED>",
    "target_gap": "<cite exact Fact_X or Rule_Y ID where gap occurs, or 'none'>",
    "missing_premise": "<[Subject] -> [CANONICAL_PREDICATE] -> [Object] | none>",
    "source_of_premise": "<EXPLICIT_FACT | EXPLICIT_RULE | UNSTATED_STATUTORY_RULE | COMMONSENSE_COMMERCIAL | FACT_BRIDGE | UNGROUNDED>",
    "explanation": "<one precise sentence explaining the formal logical link or gap>"
  }
]"""

USER_PROMPT_TEMPLATE = """EXPLICIT FACTS:
{facts_joined}

EXPLICIT RULES:
{rules_joined}

REASONING CLAIMS TO CHECK (YOU MUST PROCESS ALL {total_claims} STEPS):
{claims_formatted}

Analyze ALL {total_claims} claims now. Return ONLY the valid JSON array."""


def format_facts(facts_list, max_facts=20):
    if hasattr(facts_list, "tolist"):
        facts_list = facts_list.tolist()
    facts_list = facts_list[:max_facts]
    return "\n".join([f"Fact_{i+1}: {fact}" for i, fact in enumerate(facts_list)])


def format_rules(rules_list, max_rules=15):
    if hasattr(rules_list, "tolist"):
        rules_list = rules_list.tolist()
    rules_list = rules_list[:max_rules]
    formatted = []
    for i, r in enumerate(rules_list):
        if isinstance(r, dict):
            ref = r.get("reference") or r.get("case_name") or "Rule"
            prov = f" ({r.get('provision')})" if r.get("provision") else ""
            rel = r.get("relevance", "")
            formatted.append(f"Rule_{i+1}: {ref}{prov}: {rel}")
        else:
            formatted.append(f"Rule_{i+1}: {str(r)}")
    return "\n".join(formatted)


def parse_and_clean_json(raw_text):
    if not raw_text:
        return []
    start_idx = raw_text.find("[")
    end_idx = raw_text.rfind("]")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_str = raw_text[start_idx : end_idx + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"   [JSON Parse Error]: {e}")
            return [{"raw_response": raw_text}]
    return [{"raw_response": raw_text}]


def extract_predicate(triple_str):
    """
    Extracts the middle predicate token from a triple string and normalizes
    it to CANONICAL_PREDICATES formatting (uppercase, underscores). Handles
    models that write the predicate with spaces or hyphens instead of
    underscores (e.g. "acts as security for" -> "ACTS_AS_SECURITY_FOR").
    """
    match = re.search(r'->\s*\[?([A-Za-z_\s\-]+?)\]?\s*->', triple_str)
    if match:
        raw = match.group(1).strip()
    else:
        parts = [p.strip().strip("[]") for p in triple_str.split("->")]
        if len(parts) != 3:
            return None
        raw = parts[1].strip()
    return raw.upper().replace(" ", "_").replace("-", "_")


def validate_triple_relevance(parsed_json, full_case_text):
    """
    Flaw A check (restored from v4): guards against generic/template-fallback
    entities in the Subject/Object slots by requiring token overlap with the
    actual case text. This is independent of, and complementary to, the
    canonical-predicate check below - one checks the middle slot, this one
    checks the Subject/Object slots aren't hallucinated boilerplate.
    """
    if not isinstance(parsed_json, list):
        return True, "Invalid format"

    case_text_lower = full_case_text.lower()

    for item in parsed_json:
        if isinstance(item, dict) and item.get("status") == "NEEDS_ASSUMPTION":
            triple = item.get("missing_premise", "")
            if triple and triple != "none":
                tokens = re.findall(r'[A-Za-z]+', triple)
                ignored_words = {"is", "not", "has", "are", "or", "none", "to", "be", "and", "by", "for", "in", "of"}
                meaningful_tokens = [t.lower() for t in tokens if t.lower() not in ignored_words]
                found_tokens = [t for t in meaningful_tokens if t in case_text_lower]

                if len(meaningful_tokens) > 2 and len(found_tokens) == 0:
                    warning_msg = f"Flaw A: Step {item.get('step_index')}: '{triple}' has no token correlation with case text."
                    return False, warning_msg

    return True, "Valid"


def validate_canonical_predicates(parsed_json):
    """Validates that extracted triples strictly obey the 7 canonical predicates."""
    if not isinstance(parsed_json, list):
        return False, "Output is not a valid JSON list."

    for item in parsed_json:
        if not isinstance(item, dict):
            continue

        status = item.get("status")
        missing_premise = item.get("missing_premise", "none").strip()

        if status in ("SUPPORTED", "UNSUPPORTED"):
            if missing_premise.lower() not in ("none", ""):
                item["missing_premise"] = "none"

        elif status == "NEEDS_ASSUMPTION":
            if missing_premise.lower() in ("none", ""):
                return False, f"Step {item.get('step_index')}: Marked NEEDS_ASSUMPTION but missing_premise is 'none'."

            predicate = extract_predicate(missing_premise)
            if not predicate or predicate not in CANONICAL_PREDICATES:
                return False, f"Step {item.get('step_index')}: Invalid predicate '{predicate}' in triple '{missing_premise}'. Must be one of {CANONICAL_PREDICATES}."

            parts = [p.strip().strip("[]") for p in missing_premise.split("->")]
            item["missing_premise"] = f"[{parts[0]}] -> [{predicate}] -> [{parts[2]}]"

    return True, "Valid"


def analyze_firac_case_hardened(client, case_data, model_name=MODEL_NAME, max_retries=5):
    firac = case_data.get("firac", {}) if isinstance(case_data.get("firac"), dict) else case_data

    facts_joined = format_facts(firac.get("facts", []))
    rules_joined = format_rules(firac.get("rules", []))

    all_claims = []
    analysis_blocks = firac.get("analysis", [])
    if hasattr(analysis_blocks, "tolist"):
        analysis_blocks = analysis_blocks.tolist()

    for block in analysis_blocks:
        if isinstance(block, dict):
            steps = block.get("steps", [])
            if hasattr(steps, "tolist"):
                steps = steps.tolist()
            all_claims.extend(steps)

    if not all_claims:
        return []

    total_claims = len(all_claims)
    claims_formatted = "\n".join([f"Step {idx+1}: {c}" for idx, c in enumerate(all_claims)])
    full_case_context = f"{facts_joined}\n{rules_joined}\n{claims_formatted}"

    user_content = USER_PROMPT_TEMPLATE.format(
        facts_joined=facts_joined,
        rules_joined=rules_joined,
        claims_formatted=claims_formatted,
        total_claims=total_claims
    )

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.0
            )
            raw_output = response.choices[0].message.content
            parsed = parse_and_clean_json(raw_output)

            if isinstance(parsed, list) and len(parsed) < total_claims:
                print(f"   [Attempt {attempt}]: Truncated output ({len(parsed)}/{total_claims} steps). Retrying...")
                time.sleep(2 * attempt)
                continue

            is_relevant, rel_reason = validate_triple_relevance(parsed, full_case_context)
            if not is_relevant:
                print(f"   [Attempt {attempt}]: {rel_reason} Retrying...")
                time.sleep(2 * attempt)
                continue

            is_valid, reason = validate_canonical_predicates(parsed)
            if not is_valid:
                print(f"   [Attempt {attempt} Predicate Violation]: {reason} Retrying...")
                time.sleep(2 * attempt)
                continue

            return parsed

        except Exception as e:
            print(f"   [Attempt {attempt} API Error / Timeout]: {e}")
            time.sleep(10 * attempt)

    return parsed if 'parsed' in locals() else []


if __name__ == "__main__":
    print("Loading FIRAC dataset...")
    try:
        from datasets import load_dataset
        ds = load_dataset("mborcin/firac-appeals", "in_abs", split="train")
        dataset = [dict(row) for row in ds]
        print(f"Dataset successfully loaded ({len(dataset)} total cases).")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit(1)

    all_case_results = {}

    start_idx = 0
    end_idx = 250

    os.makedirs("results", exist_ok=True)
    output_path = f"results/openshift_batch_results_cases_{start_idx+1}_to_{end_idx}_canonical_strict.json"

    if os.path.exists(output_path):
        try:
            with open(output_path, "r") as f:
                all_case_results = json.load(f)
                print(f"Loaded existing checkpoint with {len(all_case_results)} cases.")
        except json.JSONDecodeError:
            print("Starting fresh run.")

    print(f"Connecting to: {BASE_URL}")
    print(f"Processing cases {start_idx + 1} to {end_idx} with strict canonical predicates...\n")

    for idx in range(start_idx, end_idx):
        case = dataset[idx]
        case_id = case.get("case_id", f"case_{idx+1}")

        if case_id in all_case_results:
            print(f"[{idx+1}/{end_idx}] Skipping Case ID: {case_id} (Already processed).")
            continue

        print(f"[{idx+1}/{end_idx}] Processing Case ID: {case_id}...")

        results = analyze_firac_case_hardened(client, case)
        if results and "raw_response" not in results[0]:
            all_case_results[case_id] = results
            print(f"   Successfully verified {len(results)} steps for {case_id}.\n")
        else:
            print(f"   Failed to process {case_id}.\n")

        with open(output_path, "w") as f:
            json.dump(all_case_results, f, indent=2)

        time.sleep(1)

    print(f"Batch analysis complete! Output saved to '{output_path}'.")
