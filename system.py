import argparse
import os
import evaluator
from ngram_model import NgramModel
from candidates import CandidateGenerator
from scorer import Scorer
import json

# For Windows: 2 ways to run
# python system.py --train HW1_data/train_clean.txt --dev-noisy HW1_data/dev_noisy.txt --dev-clean HW1_data/dev_clean.txt --test-noisy HW1_data/test_noisy.txt --outdir results
# python system.py --config config.json

class RestorationSystem:
    def __init__(self, lm, alpha, beta, gamma, tau):
        self.lm = lm
        self.candgen = CandidateGenerator(lm.vocab)
        self.scorer = Scorer(lm, alpha, beta, gamma, tau)

    def restore_sentence(self, tokens):
        restored_tokens = []
        token_decisions = []

        for i, tok in enumerate(tokens):
            candidates = list(self.candgen.get_candidates(tok))
            if tok not in candidates:
                candidates = [tok] + candidates

            scored = []
            for cand in candidates:
                prev_prev = None if i < 2 else tokens[i - 2]
                prev_word = None if i < 1 else tokens[i - 1]
                next_word = None if i >= len(tokens) - 1 else tokens[i + 1]
                next_next = None if i >= len(tokens) - 2 else tokens[i + 2]

                score = self.scorer.calculate_score(
                    tok,
                    cand,
                    prev_prev,
                    prev_word,
                    next_word,
                    next_next
                )

                scored.append((cand, float(score)))

            scored.sort(key=lambda x: x[1], reverse=True)

            original_score = next((s for c, s in scored if c == tok), float("-inf"))
            best_cand, best_score = scored[0]
            score_margin = best_score - original_score

            if score_margin > self.scorer.tau:
                final = best_cand
                decision = "edit"
                changed = final != tok
                decision_reason = (
                    f"margin {score_margin:.4f} > threshold {self.scorer.tau:.4f}; "
                    f"selected '{best_cand}'"
                )
            else:
                final = tok
                decision = "abstain"
                changed = False
                decision_reason = (
                    f"margin {score_margin:.4f} <= threshold {self.scorer.tau:.4f}; "
                    f"kept original"
                )

            candidate_records = []
            for rank, (cand, score) in enumerate(scored, start=1):
                candidate_records.append({
                    "text": cand,
                    "score": float(score),
                    "rank": rank,
                })

            token_decisions.append({
                "token_index": i,
                "original": tok,
                "final": final,
                "changed": changed,
                "candidates": candidate_records,
                "selected_candidate": best_cand,
                "score_original": float(original_score),
                "score_selected": float(best_score),
                "score_margin": float(score_margin),
                "threshold": float(self.scorer.tau),
                "decision": decision,
                "decision_reason": decision_reason,
            })

            restored_tokens.append(final)

        restored_text = " ".join(restored_tokens)
        return restored_text, token_decisions


def read_line(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def predict_sentences(system, sentences):
    predictions = []
    trace_records = []

    for sentence_id, sent in enumerate(sentences):
        tokens = sent.split()
        output_text, token_decisions = system.restore_sentence(tokens)

        positions_considered = len(token_decisions)
        edits_made = sum(1 for td in token_decisions if td["decision"] == "edit")
        abstentions = sum(1 for td in token_decisions if td["decision"] == "abstain")
        input_score = sum(td["score_original"] for td in token_decisions)
        output_score = sum(td["score_selected"] for td in token_decisions)
        score_margin = output_score - input_score
        changed_indices = [td["token_index"] for td in token_decisions if td["decision"] == "edit"]

        sentence_decision = "edit" if score_margin > 0 else "abstain"

        if changed_indices:
            reason_list = [
                f"token '{token_decisions[i]['original']}' replaced by "
                f"'{token_decisions[i]['final']}' with margin "
                f"{token_decisions[i]['score_margin']:.4f} > {system.scorer.tau:.4f}"
                for i in changed_indices
            ]
            reason = "; ".join(reason_list)
        else:
            reason = "no token exceeded edit threshold"

        trace_records.append({
            "sentence_id": sentence_id,
            "input_text": sent,
            "output_text": output_text,
            "edited": edits_made > 0,
            "sentence_level": {
                "input_score": float(input_score),
                "output_score": float(output_score),
                "score_improvement": float(score_margin),
                "decision": sentence_decision,
                "reason": reason
            },
            "token_decisions": token_decisions,
            "summary": {
                "positions_considered": positions_considered,
                "edits_made": edits_made,
                "abstentions": abstentions,
            }
        })

        predictions.append(output_text)

    return predictions, trace_records

def no_change_baseline(sentences):
    """
    return input sentences unchanged
    """
    return sentences


def unigram_correction_baseline(sentences, lm, candgen):
    final_predictions = []
    for sent in sentences:
        tokens = sent.split()
        restored = []
        for tok in tokens:
            # ۱. اگر کلمه در لغت‌نامه هست، دست نزن
            if tok in lm.vocab:
                restored.append(tok)
                continue

            # ۲. کاندیداهای فاصله ۱ را بگیر
            candidates = candgen.get_candidates(tok, distance=1)

            # ۳. بهترین را فقط بر اساس unigram انتخاب کن
            best_cand = tok
            best_score = float("-inf")
            for cand in candidates:
                if cand in lm.vocab:
                    score = lm.logp_unigram(cand)
                    if score > best_score:
                        best_score = score
                        best_cand = cand
            restored.append(best_cand)
        final_predictions.append(" ".join(restored))
    return final_predictions

def main():
    parser = argparse.ArgumentParser(description="N-gram Noisy Text Restoration System")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--train", type=str)
    parser.add_argument("--dev-noisy", type=str)
    parser.add_argument("--dev-clean", type=str)
    parser.add_argument("--test-noisy", type=str)
    parser.add_argument("--outdir", type=str)
    args = parser.parse_args()

    # Use config.json
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)

        train_path = config["paths"]["train"]
        dev_noisy_path = config["paths"]["dev_noisy"]
        dev_clean_path = config["paths"]["dev_clean"]
        test_noisy_path = config["paths"]["test_noisy"]
        outdir = config["paths"]["outdir"]

        ngram_order = config["model"]["ngram_order"]
        unk_cutoff = config["model"]["unk_cutoff"]

        alpha = config["scorer"]["alpha"]
        beta = config["scorer"]["beta"]
        gamma = config["scorer"]["gamma"]
        tau = config["scorer"]["tau"]
    else:
        train_path = args.train
        dev_noisy_path = args.dev_noisy
        dev_clean_path = args.dev_clean
        test_noisy_path = args.test_noisy
        outdir = args.outdir
        ngram_order = 3
        unk_cutoff = 1
        alpha = 1
        beta = 1
        gamma = -0.2
        tau = 0.02

    evaluator.ensure_dir(outdir)

    train_sentences = read_line(train_path)
    dev_clean = read_line(dev_clean_path)
    dev_noisy = read_line(dev_noisy_path)
    test_noisy = read_line(test_noisy_path)

    if len(dev_clean) != len(dev_noisy):
        raise ValueError("dev_noisy and dev_clean must have the same number of lines.")

    # Train Ngram model (LM) when finished
    for order in [1, 2, 3]:
        lm_tmp = NgramModel(order=order, unk_cutoff=unk_cutoff)
        lm_tmp.train(train_sentences)
        ppl = lm_tmp.perplexity(dev_clean)
        print(f"Perplexity dev (order={order}): {ppl:.4f}")


    lm = NgramModel(order=ngram_order, unk_cutoff=unk_cutoff)
    lm.train(train_sentences)
    candgen = CandidateGenerator(lm.vocab)

    system = RestorationSystem(lm, alpha, beta, gamma, tau)

    # Baseline 1: No change
    no_change_predictions = no_change_baseline(dev_noisy)
    # Baseline 2: Unigram correction
    unigram_predictions = unigram_correction_baseline(dev_noisy, lm, candgen)
    # Proposed system
    dev_predictions, dev_trace_records = predict_sentences(system, dev_noisy)

    test_predictions, test_trace_records = predict_sentences(system, test_noisy)

    evaluator.write_prediction_csv(
        dev_noisy, dev_predictions, dev_clean,
        os.path.join(outdir, "dev_predictions.csv")
    )
    evaluator.write_prediction_csv(
        test_noisy, test_predictions, None,
        os.path.join(outdir, "test_predictions.csv")
    )

    evaluator.write_trace_jsonl(dev_trace_records, os.path.join(outdir, "dev_trace.jsonl"))
    evaluator.write_trace_jsonl(test_trace_records, os.path.join(outdir, "test_trace.jsonl"))


    proposed_metrics = {
        "sentence_accuracy": evaluator.compute_sentence_accuracy(dev_predictions, dev_clean),
        "token_accuracy": evaluator.compute_token_accuracy(dev_predictions, dev_clean),
        "wer": evaluator.compute_wer(dev_predictions, dev_clean),
        "cer": evaluator.compute_cer(dev_predictions, dev_clean),
        **evaluator.compute_edit_metrics(dev_noisy, dev_predictions, dev_clean)
    }

    no_change_metrics = {
        "sentence_accuracy": evaluator.compute_sentence_accuracy(no_change_predictions, dev_clean),
        "token_accuracy": evaluator.compute_token_accuracy(no_change_predictions, dev_clean),
        "wer": evaluator.compute_wer(no_change_predictions, dev_clean),
        "cer": evaluator.compute_cer(no_change_predictions, dev_clean),
        **evaluator.compute_edit_metrics(dev_noisy, no_change_predictions, dev_clean)
    }

    unigram_metrics = {
        "sentence_accuracy": evaluator.compute_sentence_accuracy(unigram_predictions, dev_clean),
        "token_accuracy": evaluator.compute_token_accuracy(unigram_predictions, dev_clean),
        "wer": evaluator.compute_wer(unigram_predictions, dev_clean),
        "cer": evaluator.compute_cer(unigram_predictions, dev_clean),
        **evaluator.compute_edit_metrics(dev_noisy, unigram_predictions, dev_clean)
    }

    comparison = {
        "no_change_baseline": no_change_metrics,
        "unigram_baseline": unigram_metrics,
        "context_ngram_system": proposed_metrics
    }

    summary = {
        "student_name": "Paniz Azarnia",
        "student_id": "40435097",
        "ngram_order": ngram_order,
        "smoothing": "Kneser_Ney",
        "tokenization": "Whitespace",
        "case_handling": "case-flexible (lower/upper/capitalize variants)",
        "unk_threshold":  unk_cutoff,
        "abstain_threshold": tau,
        "candidate_strategy": "edit_distance_1 + edit_distance_2 + dictionary",
        "score_weights": {"context": alpha, "prior": beta, "edit_penalty": gamma},
        "random_seed":"None"
    }

    evaluator.write_json(comparison, os.path.join(outdir, "dev_metrics.json"))
    evaluator.write_json(summary, os.path.join(outdir, "summary.json"))

    print("Done.")
    print("\n=== Baseline Comparison ===")
    print("No-change sentence accuracy:", no_change_metrics["sentence_accuracy"])
    print("Unigram sentence accuracy:", unigram_metrics["sentence_accuracy"])
    print("Proposed system sentence accuracy:", proposed_metrics["sentence_accuracy"])
    print(f"Outputs written to: {outdir}")


if __name__ == "__main__":
    main()