"""
Unit tests for the embedding-based semantic entity-linking match added to
`evaluate_kg_extraction.py` (`_semantic_match_counts()` /
`entity_linking_metrics_by_type()`), which lets a predicted `issue`/
`decision`/`task` mention match a differently-worded gold mention instead of
requiring exact `node_key()` string equality.

No OpenAI API calls: `embed_texts` is monkeypatched to a deterministic stub,
same pattern as `04_knowledgegraph_dashboard/test_build_knowledge_graph.py`.
Run with:

    python -m unittest eval/test_evaluate_kg_extraction.py
"""

import os
import sys
import unittest
from unittest.mock import patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import evaluate_kg_extraction as evalkg  # noqa: E402


def _triple(subject_text, subject_type, object_text, object_type, relation="related_to"):
    return {
        "subject_text": subject_text, "subject_type": subject_type,
        "relation": relation, "object_text": object_text, "object_type": object_type,
        "timestamp": "", "quote": "", "meeting_id": "m1",
    }


class SemanticMatchCountsTests(unittest.TestCase):
    def _fake_embed_texts(self, embedding_by_text):
        def fake(texts, model="text-embedding-3-small", cache_dir="cache_kg_embeddings"):
            return [embedding_by_text[t] for t in texts]
        return fake

    def test_paraphrased_mention_matches_gold(self):
        pred_key_to_text = {"issue:signoff blocker": "signoff blocker"}
        gold_key_to_text = {"issue:legal sign-off pending": "legal sign-off pending"}
        embedding_by_text = {
            "signoff blocker": [1.0, 0.0],
            "legal sign-off pending": [0.99, 0.14106736],  # cos sim ~0.99, above threshold
        }
        with patch.object(evalkg, "embed_texts", self._fake_embed_texts(embedding_by_text)):
            matched_pred, matched_gold = evalkg._semantic_match_counts(
                pred_key_to_text, gold_key_to_text, threshold=0.85,
                model="text-embedding-3-small", cache_dir="cache_kg_embeddings",
            )
        self.assertEqual(matched_pred, 1)
        self.assertEqual(matched_gold, 1)

    def test_unrelated_mention_does_not_match(self):
        pred_key_to_text = {"issue:signoff blocker": "signoff blocker"}
        gold_key_to_text = {"issue:unrelated problem": "unrelated problem"}
        embedding_by_text = {
            "signoff blocker": [1.0, 0.0],
            "unrelated problem": [0.0, 1.0],  # orthogonal
        }
        with patch.object(evalkg, "embed_texts", self._fake_embed_texts(embedding_by_text)):
            matched_pred, matched_gold = evalkg._semantic_match_counts(
                pred_key_to_text, gold_key_to_text, threshold=0.85,
                model="text-embedding-3-small", cache_dir="cache_kg_embeddings",
            )
        self.assertEqual(matched_pred, 0)
        self.assertEqual(matched_gold, 0)

    def test_multiple_predicted_variants_all_match_one_gold(self):
        pred_key_to_text = {
            "issue:signoff blocker": "signoff blocker",
            "issue:sign off issue": "sign off issue",
        }
        gold_key_to_text = {"issue:legal sign-off pending": "legal sign-off pending"}
        embedding_by_text = {
            "signoff blocker": [1.0, 0.0],
            "sign off issue": [1.0, 0.0],
            "legal sign-off pending": [0.99, 0.14106736],
        }
        with patch.object(evalkg, "embed_texts", self._fake_embed_texts(embedding_by_text)):
            matched_pred, matched_gold = evalkg._semantic_match_counts(
                pred_key_to_text, gold_key_to_text, threshold=0.85,
                model="text-embedding-3-small", cache_dir="cache_kg_embeddings",
            )
        self.assertEqual(matched_pred, 2, "both predicted variants share the gold-matching cluster")
        self.assertEqual(matched_gold, 1)

    def test_empty_pred_or_gold_returns_zero(self):
        with patch.object(evalkg, "embed_texts", self._fake_embed_texts({})):
            self.assertEqual(evalkg._semantic_match_counts({}, {"k": "text"}, 0.85, "m", "c"), (0, 0))
            self.assertEqual(evalkg._semantic_match_counts({"k": "text"}, {}, 0.85, "m", "c"), (0, 0))


class EntityLinkingMetricsByTypeTests(unittest.TestCase):
    def _fake_embed_texts(self, embedding_by_text):
        def fake(texts, model="text-embedding-3-small", cache_dir="cache_kg_embeddings"):
            return [embedding_by_text[t] for t in texts]
        return fake

    def test_semantic_type_recovers_paraphrased_gold_match(self):
        pred_triples = [_triple("signoff blocker", "issue", "release", "task")]
        gold_triples = [_triple("legal sign-off pending", "issue", "release", "task")]
        embedding_by_text = {
            "signoff blocker": [1.0, 0.0],
            "legal sign-off pending": [0.99, 0.14106736],
            "release": [1.0, 0.0],  # task type: not in semantic_types, must match exactly
        }
        with patch.object(evalkg, "embed_texts", self._fake_embed_texts(embedding_by_text)):
            by_type = evalkg.entity_linking_metrics_by_type(
                pred_triples, gold_triples, semantic_types=("issue",), threshold=0.85,
            )
        self.assertEqual(by_type["issue"]["recall"], 1.0, "paraphrased issue should now match gold")
        self.assertEqual(by_type["task"]["recall"], 1.0, "identical task text still matches via exact node_key()")

    def test_non_semantic_type_requires_exact_match_even_if_similar(self):
        pred_triples = [_triple("Aiko", "person", "release", "task")]
        gold_triples = [_triple("Aiko - PM", "person", "release", "task")]
        embedding_by_text = {
            "Aiko": [1.0, 0.0],
            "Aiko - PM": [0.99, 0.14106736],  # would match if person were semantic, but it isn't here
            "release": [1.0, 0.0],
        }
        with patch.object(evalkg, "embed_texts", self._fake_embed_texts(embedding_by_text)):
            by_type = evalkg.entity_linking_metrics_by_type(
                pred_triples, gold_triples, semantic_types=("issue", "decision", "task"), threshold=0.85,
            )
        self.assertEqual(by_type["person"]["recall"], 0.0, "person is not in semantic_types, so no fuzzy credit")

    def test_pooled_entity_linking_metrics_matches_by_type_sum(self):
        pred_triples = [
            _triple("signoff blocker", "issue", "Aiko", "person"),
        ]
        gold_triples = [
            _triple("legal sign-off pending", "issue", "Aiko", "person"),
        ]
        embedding_by_text = {
            "signoff blocker": [1.0, 0.0],
            "legal sign-off pending": [0.99, 0.14106736],
            "Aiko": [1.0, 0.0],
        }
        with patch.object(evalkg, "embed_texts", self._fake_embed_texts(embedding_by_text)):
            pooled = evalkg.entity_linking_metrics(
                pred_triples, gold_triples, semantic_types=("issue",), threshold=0.85,
            )
        # 1 issue (semantic match) + 1 person (exact match) predicted/gold, both matched
        self.assertEqual(pooled["num_predicted_entities"], 2)
        self.assertEqual(pooled["num_gold_entities"], 2)
        self.assertEqual(pooled["matched_for_precision"], 2)
        self.assertEqual(pooled["matched_for_recall"], 2)
        self.assertEqual(pooled["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
