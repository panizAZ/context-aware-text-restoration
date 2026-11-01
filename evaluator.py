import csv
import json
import os


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def write_prediction_csv(sentences, predictions, reference, filepath):
    """
    Writes one predicted sentence per row.
    CSV format:
    id,prediction
    """
    with open(filepath, 'w',encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        if reference is not None:
            writer.writerow(["sentence_id", "input_text", "predicted_text","reference_text","edited"])
            for i, s in enumerate(sentences):
                writer.writerow([i, s, predictions[i], reference[i], s.split() != predictions[i].split()])
        else:
            writer.writerow(["sentence_id", "input_text", "predicted_text", "edited"])
            for i, s in enumerate(sentences):
                writer.writerow([i, s, predictions[i], s.split() != predictions[i].split()])

def write_json(data, filepath):
    with open(filepath, 'w', encoding="utf-8") as f:
        # For langauge other than English ensure_ascii ensures it's readable
        # indent=2 keeps writing neat
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_trace_jsonl(trace_records, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        for rec in trace_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# --- Metrics ---

def levenshtein_distance(s1, s2):
    # برای محاسبه WER و CER به این نیاز داریم
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if not s2:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def compute_sentence_accuracy(predictions, references):
    """
    Simple sentence-level exact match accuracy.
    """
    assert len(predictions) == len(references),"predictions and references length mismatch"
    if len(predictions) == 0:
        return 0.0

    correct = sum(int(p==r) for p, r in zip(predictions, references))
    return correct/len(predictions)


def compute_token_accuracy(predictions, references):
    total_tokens = 0
    correct_tokens = 0

    for p_sent, r_sent in zip(predictions, references):
        p_tokens = p_sent.split()
        r_tokens = r_sent.split()

        max_len = max(len(p_tokens), len(r_tokens))

        for i in range(max_len):
            pt = p_tokens[i] if i < len(p_tokens) else None
            rt = r_tokens[i] if i < len(r_tokens) else None

            total_tokens += 1
            if pt == rt:
                correct_tokens += 1

    return correct_tokens / total_tokens if total_tokens > 0 else 0.0

def compute_wer(predictions, references):
    # Word error rate
    total_dist = 0
    total_ref_tokens = 0
    for p, r in zip(predictions, references):
        p_tokens = p.split()
        r_tokens = r.split()
        total_dist += levenshtein_distance(p_tokens, r_tokens)
        total_ref_tokens += len(r_tokens)
    return total_dist / total_ref_tokens if total_ref_tokens > 0 else 0.0


def compute_cer(predictions, references):
    total_dist = 0
    total_ref_chars = 0
    for p, r in zip(predictions, references):
        total_dist += levenshtein_distance(p, r)
        total_ref_chars += len(r)
    return total_dist / total_ref_chars if total_ref_chars > 0 else 0.0


def compute_edit_metrics(noisy, predictions, references):
    """
    Edit Precision: از بین توکن‌هایی که سیستم عوض کرد، چندتا واقعاً در ورودی غلط بودند؟
    Edit Recall: از بین توکن‌هایی که در ورودی غلط بودند، سیستم چندتا را درست کرد؟
    Overcorrection: چند درصد از تغییرات سیستم، روی کلمات درست انجام شده؟
    """
    tokens_changed_by_system = 0
    actually_wrong_in_input = 0
    correctly_fixed = 0
    changed_but_were_already_correct = 0

    for n_sent, p_sent, r_sent in zip(noisy, predictions, references):
        n_tokens = n_sent.split()
        p_tokens = p_sent.split()
        r_tokens = r_sent.split()

        for nt, pt, rt in zip(n_tokens, p_tokens, r_tokens):
            is_changed = (nt != pt)
            is_wrong_in_input = (nt != rt)
            is_fixed_correctly = (pt == rt and is_wrong_in_input)

            if is_changed:
                tokens_changed_by_system += 1
                if is_wrong_in_input:
                    # تغییر داده و واقعاً هم غلط بوده
                    pass
                else:
                    # تغییر داده ولی کلمه خودش درست بوده (Overcorrection)
                    changed_but_were_already_correct += 1

            if is_wrong_in_input:
                actually_wrong_in_input += 1
                if is_fixed_correctly:
                    correctly_fixed += 1

    edit_precision = correctly_fixed / tokens_changed_by_system if tokens_changed_by_system > 0 else 0.0
    edit_recall = correctly_fixed / actually_wrong_in_input if actually_wrong_in_input > 0 else 0.0
    edit_f1 = 2 * (edit_precision * edit_recall) / (edit_precision + edit_recall) if (
                                                                                                 edit_precision + edit_recall) > 0 else 0.0
    overcorrection_rate = changed_but_were_already_correct / tokens_changed_by_system if tokens_changed_by_system > 0 else 0.0

    return {
        "edit_precision": round(edit_precision, 4),
        "edit_recall": round(edit_recall, 4),
        "edit_F1": round(edit_f1, 4),
        "overcorrection_rate": round(overcorrection_rate, 4)
    }

# ---- Required API ----

def compute_metrics(noisy, predictions, references, lm=None):
    """
    Main evaluation API required by assignment.

    Returns a dictionary of all evaluation metrics.
    """

    metrics = {}

    # Basic metrics
    metrics["sentence_accuracy"] = round(
        compute_sentence_accuracy(predictions, references), 4
    )

    metrics["token_accuracy"] = round(
        compute_token_accuracy(predictions, references), 4
    )

    metrics["wer"] = round(
        compute_wer(predictions, references), 4
    )

    metrics["cer"] = round(
        compute_cer(predictions, references), 4
    )

    # Edit metrics
    edit_metrics = compute_edit_metrics(noisy, predictions, references)
    metrics.update(edit_metrics)

    return metrics