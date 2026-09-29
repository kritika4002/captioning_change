import math
from collections import defaultdict, Counter

_nltk_available = False
_rouge_available = False

try:
    import nltk
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score as nltk_meteor
    _nltk_available = True
except ImportError:
    pass

try:
    from rouge_score import rouge_scorer
    _rouge_available = True
except ImportError:
    pass


def _ensure_nltk_data():
    if not _nltk_available:
        return
    for resource in ["wordnet", "punkt", "punkt_tab", "omw-1.4"]:
        try:
            sub = "corpora" if ("wordnet" in resource or "omw" in resource) else "tokenizers"
            nltk.data.find(f"{sub}/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


class CaptionEvaluator:
    def __init__(self, metrics=None):
        self.metrics = [m.lower() for m in (metrics or ["bleu", "meteor", "rouge_l"])]
        _ensure_nltk_data()

        self.rouge = None
        if "rouge_l" in self.metrics and _rouge_available:
            self.rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def evaluate(self, preds, refs):
        if len(preds) != len(refs):
            raise ValueError(f"mismatch: {len(preds)} preds vs {len(refs)} refs")

        res = {}
        if "bleu" in self.metrics:
            res.update(self._compute_bleu(preds, refs))
        if "meteor" in self.metrics:
            res["meteor"] = self._compute_meteor(preds, refs)
        if "rouge_l" in self.metrics:
            res["rouge_l"] = self._compute_rouge_l(preds, refs)
        if "cider" in self.metrics:
            res["cider"] = self._compute_cider(preds, refs)
        return res

    def evaluate_single(self, pred, ref):
        return self.evaluate([pred], [ref])

    def _compute_bleu(self, preds, refs):
        if not _nltk_available:
            return {f"bleu_{i}": 0.0 for i in range(1, 5)}

        smoothie = SmoothingFunction().method1
        hyps = [p.lower().split() for p in preds]
        targets = [[r.lower().split()] for r in refs]

        scores = {}
        for n in range(1, 5):
            weights = tuple(1.0 / n if i < n else 0.0 for i in range(4))
            try:
                s = corpus_bleu(targets, hyps, weights=weights, smoothing_function=smoothie)
            except Exception:
                s = 0.0
            scores[f"bleu_{n}"] = round(s, 4)
        return scores

    def _compute_meteor(self, preds, refs):
        if not _nltk_available:
            return 0.0
        scores = []
        for p, r in zip(preds, refs):
            try:
                s = nltk_meteor([r.lower().split()], p.lower().split())
                scores.append(s)
            except Exception:
                scores.append(0.0)
        return round(sum(scores) / max(len(scores), 1), 4)

    def _compute_rouge_l(self, preds, refs):
        if not _rouge_available or self.rouge is None:
            return 0.0
        scores = [self.rouge.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)]
        return round(sum(scores) / max(len(scores), 1), 4)

    def _compute_cider(self, preds, refs):
        def ngrams(tokens, n):
            return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

        df = defaultdict(int)
        n_docs = len(refs)
        ref_ngrams_list = []

        for r in refs:
            toks = r.lower().split()
            seen = set()
            for n in range(1, 5):
                seen.update(ngrams(toks, n))
            ref_ngrams_list.append(Counter(ng for n in range(1, 5) for ng in ngrams(toks, n)))
            for ng in seen:
                df[ng] += 1

        idf = {ng: math.log(n_docs / (cnt + 1.0)) for ng, cnt in df.items()}
        scores = []

        for p, ref_counts in zip(preds, ref_ngrams_list):
            toks = p.lower().split()
            p_counts = Counter(ng for n in range(1, 5) for ng in ngrams(toks, n))
            common = set(p_counts.keys()) & set(ref_counts.keys())
            if not common:
                scores.append(0.0)
                continue

            num = sum(p_counts[ng] * ref_counts[ng] * (idf.get(ng, 0) ** 2) for ng in common)
            p_norm = math.sqrt(sum((p_counts[ng] * idf.get(ng, 0)) ** 2 for ng in p_counts))
            r_norm = math.sqrt(sum((ref_counts[ng] * idf.get(ng, 0)) ** 2 for ng in ref_counts))

            scores.append(num / (p_norm * r_norm) if p_norm * r_norm > 0 else 0.0)

        return round(sum(scores) / max(len(scores), 1), 4)
