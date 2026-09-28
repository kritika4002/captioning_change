"""
Caption Evaluation Metrics — BLEU, METEOR, ROUGE-L, CIDEr.

Implements standard NLP metrics for evaluating generated captions
against ground-truth references.  These are the metrics reported
in the project's comparative analysis.
"""

import logging
from typing import Dict, List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

# Lazy imports — these packages may not always be installed
_nltk_available = False
_rouge_available = False

try:
    import nltk
    from nltk.translate.bleu_score import (
        corpus_bleu,
        sentence_bleu,
        SmoothingFunction,
    )
    from nltk.translate.meteor_score import meteor_score as nltk_meteor
    _nltk_available = True
except ImportError:
    logger.warning("nltk not installed — BLEU and METEOR unavailable")

try:
    from rouge_score import rouge_scorer
    _rouge_available = True
except ImportError:
    logger.warning("rouge-score not installed — ROUGE-L unavailable")


def _ensure_nltk_data():
    """Download required NLTK data if not present."""
    if not _nltk_available:
        return
    for resource in ["wordnet", "punkt", "punkt_tab", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{resource}" if "wordnet" in resource or "omw" in resource else f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


class CaptionEvaluator:
    """
    Evaluates generated captions against ground-truth references.

    Supports:
        - BLEU-1/2/3/4 (with smoothing)
        - METEOR
        - ROUGE-L (F1)
        - CIDEr (simplified TF-IDF based)

    Usage:
        evaluator = CaptionEvaluator()
        results = evaluator.evaluate(predictions, references)
        print(results)
    """

    def __init__(self, metrics: Optional[List[str]] = None):
        """
        Args:
            metrics: List of metrics to compute.
                     Default: ["bleu", "meteor", "rouge_l"]
        """
        if metrics is None:
            metrics = ["bleu", "meteor", "rouge_l"]

        self.metrics = [m.lower() for m in metrics]
        _ensure_nltk_data()

        # ROUGE scorer (reusable)
        if "rouge_l" in self.metrics and _rouge_available:
            self.rouge = rouge_scorer.RougeScorer(
                ["rougeL"], use_stemmer=True
            )
        else:
            self.rouge = None

    def evaluate(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """
        Compute all configured metrics.

        Args:
            predictions: List of generated caption strings.
            references:  List of ground-truth caption strings.

        Returns:
            Dictionary of metric_name → score.
        """
        assert len(predictions) == len(references), (
            f"Mismatch: {len(predictions)} predictions vs "
            f"{len(references)} references"
        )

        results = {}

        if "bleu" in self.metrics:
            bleu_scores = self._compute_bleu(predictions, references)
            results.update(bleu_scores)

        if "meteor" in self.metrics:
            results["meteor"] = self._compute_meteor(
                predictions, references
            )

        if "rouge_l" in self.metrics:
            results["rouge_l"] = self._compute_rouge_l(
                predictions, references
            )

        if "cider" in self.metrics:
            results["cider"] = self._compute_cider(
                predictions, references
            )

        return results

    def evaluate_single(
        self, prediction: str, reference: str
    ) -> Dict[str, float]:
        """Evaluate a single prediction-reference pair."""
        return self.evaluate([prediction], [reference])

    # ──────────────────────────────────────────
    # BLEU
    # ──────────────────────────────────────────

    def _compute_bleu(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        """Compute BLEU-1 through BLEU-4 with smoothing."""
        if not _nltk_available:
            return {f"bleu_{i}": 0.0 for i in range(1, 5)}

        smoothie = SmoothingFunction().method1

        # Tokenise
        hyps = [pred.lower().split() for pred in predictions]
        refs = [[ref.lower().split()] for ref in references]  # list of list

        results = {}
        for n in range(1, 5):
            weights = tuple(
                1.0 / n if i < n else 0.0 for i in range(4)
            )
            try:
                score = corpus_bleu(
                    refs, hyps,
                    weights=weights,
                    smoothing_function=smoothie,
                )
            except Exception:
                score = 0.0
            results[f"bleu_{n}"] = round(score, 4)

        return results

    # ──────────────────────────────────────────
    # METEOR
    # ──────────────────────────────────────────

    def _compute_meteor(
        self, predictions: List[str], references: List[str]
    ) -> float:
        """Compute average METEOR score."""
        if not _nltk_available:
            return 0.0

        scores = []
        for pred, ref in zip(predictions, references):
            try:
                score = nltk_meteor(
                    [ref.lower().split()],
                    pred.lower().split(),
                )
                scores.append(score)
            except Exception:
                scores.append(0.0)

        return round(sum(scores) / max(len(scores), 1), 4)

    # ──────────────────────────────────────────
    # ROUGE-L
    # ──────────────────────────────────────────

    def _compute_rouge_l(
        self, predictions: List[str], references: List[str]
    ) -> float:
        """Compute average ROUGE-L F1 score."""
        if not _rouge_available or self.rouge is None:
            return 0.0

        scores = []
        for pred, ref in zip(predictions, references):
            result = self.rouge.score(ref, pred)
            scores.append(result["rougeL"].fmeasure)

        return round(sum(scores) / max(len(scores), 1), 4)

    # ──────────────────────────────────────────
    # CIDEr (simplified)
    # ──────────────────────────────────────────

    def _compute_cider(
        self, predictions: List[str], references: List[str]
    ) -> float:
        """
        Simplified CIDEr score using TF-IDF n-gram matching.

        This is a lightweight approximation. For the full CIDEr-D metric,
        use pycocoevalcap.
        """
        import math
        from collections import Counter

        def get_ngrams(tokens, n):
            return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

        # Document frequencies
        df = defaultdict(int)
        num_docs = len(references)

        ref_ngrams_list = []
        for ref in references:
            tokens = ref.lower().split()
            ngrams_set = set()
            for n in range(1, 5):
                ngrams_set.update(get_ngrams(tokens, n))
            ref_ngrams_list.append(Counter(
                ng for n in range(1, 5) for ng in get_ngrams(tokens, n)
            ))
            for ng in ngrams_set:
                df[ng] += 1

        # Compute IDF
        idf = {
            ng: math.log(num_docs / (count + 1.0))
            for ng, count in df.items()
        }

        scores = []
        for pred, ref_ngrams in zip(predictions, ref_ngrams_list):
            tokens = pred.lower().split()
            pred_ngrams = Counter(
                ng for n in range(1, 5) for ng in get_ngrams(tokens, n)
            )

            # TF-IDF vectors
            common = set(pred_ngrams.keys()) & set(ref_ngrams.keys())
            if not common:
                scores.append(0.0)
                continue

            numerator = sum(
                pred_ngrams[ng] * ref_ngrams[ng] * idf.get(ng, 0) ** 2
                for ng in common
            )
            pred_norm = math.sqrt(sum(
                (pred_ngrams[ng] * idf.get(ng, 0)) ** 2
                for ng in pred_ngrams
            ))
            ref_norm = math.sqrt(sum(
                (ref_ngrams[ng] * idf.get(ng, 0)) ** 2
                for ng in ref_ngrams
            ))

            if pred_norm * ref_norm > 0:
                scores.append(numerator / (pred_norm * ref_norm))
            else:
                scores.append(0.0)

        return round(sum(scores) / max(len(scores), 1), 4)

    def format_results(self, results: Dict[str, float]) -> str:
        """Pretty-print evaluation results."""
        lines = ["┌─────────────────────────────────┐"]
        lines.append("│    Caption Evaluation Results    │")
        lines.append("├────────────────┬────────────────┤")
        for metric, score in sorted(results.items()):
            lines.append(f"│ {metric:<14} │ {score:>14.4f} │")
        lines.append("└────────────────┴────────────────┘")
        return "\n".join(lines)
