"""
Unit tests for the semantic entity-merging pass (`merge_similar_entities()` /
`cluster_mentions()`) added to `build_knowledge_graph.py`.

No OpenAI API calls: `cluster_mentions()` is a pure function that takes
pre-computed embeddings, and `merge_similar_entities()` is tested with
`embed_texts` monkeypatched to a deterministic stub. Run with:

    python -m unittest 04_knowledgegraph_dashboard/test_build_knowledge_graph.py
"""

import os
import sys
import unittest
from unittest.mock import patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import build_knowledge_graph as bkg  # noqa: E402


class ClusterMentionsTests(unittest.TestCase):
    def test_merges_pair_above_threshold(self):
        texts = ["signoff blocker", "sign-off blocker"]
        embeddings = [[1.0, 0.0], [0.99, 0.14106736]]  # cos sim ~0.99
        canonical_for = bkg.cluster_mentions(texts, embeddings, threshold=0.85)
        self.assertEqual(canonical_for["signoff blocker"], canonical_for["sign-off blocker"])

    def test_keeps_pair_below_threshold_separate(self):
        texts = ["signoff blocker", "unrelated topic"]
        embeddings = [[1.0, 0.0], [0.0, 1.0]]  # orthogonal, cos sim 0.0
        canonical_for = bkg.cluster_mentions(texts, embeddings, threshold=0.85)
        self.assertNotEqual(canonical_for["signoff blocker"], canonical_for["unrelated topic"])

    def test_canonical_is_most_frequent_text(self):
        texts = ["a", "a", "b"]
        embeddings = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]  # identical vectors, all merge
        canonical_for = bkg.cluster_mentions(texts, embeddings, threshold=0.85)
        self.assertEqual(canonical_for["a"], "a")
        self.assertEqual(canonical_for["b"], "a")

    def test_tie_breaks_on_first_occurrence(self):
        texts = ["first", "second"]
        embeddings = [[1.0, 0.0], [1.0, 0.0]]  # identical, equal frequency (1 each)
        canonical_for = bkg.cluster_mentions(texts, embeddings, threshold=0.85)
        self.assertEqual(canonical_for["first"], "first")
        self.assertEqual(canonical_for["second"], "first")

    def test_transitive_merge_via_chain(self):
        # a<->b above threshold, b<->c above threshold, a<->c below threshold:
        # single-linkage clustering should still put all three in one cluster.
        texts = ["a", "b", "c"]
        embeddings = [[1.0, 0.0], [0.9, 0.43588989], [0.6, 0.8]]
        sim_ab = bkg.cosine_similarity(embeddings[0], embeddings[1])
        sim_bc = bkg.cosine_similarity(embeddings[1], embeddings[2])
        sim_ac = bkg.cosine_similarity(embeddings[0], embeddings[2])
        self.assertGreaterEqual(sim_ab, 0.85)
        self.assertGreaterEqual(sim_bc, 0.85)
        self.assertLess(sim_ac, 0.85)

        canonical_for = bkg.cluster_mentions(texts, embeddings, threshold=0.85)
        self.assertEqual(len({canonical_for["a"], canonical_for["b"], canonical_for["c"]}), 1)

    def test_single_text_untouched(self):
        canonical_for = bkg.cluster_mentions(["only"], [[1.0, 0.0]], threshold=0.85)
        self.assertEqual(canonical_for["only"], "only")


class MergeSimilarEntitiesTests(unittest.TestCase):
    def _fake_embed_texts(self, embedding_by_text):
        def fake(texts, model="text-embedding-3-small", cache_dir="cache_kg_embeddings"):
            return [embedding_by_text[t] for t in texts]
        return fake

    def test_merges_only_configured_entity_types(self):
        triples = [
            {
                "subject_text": "signoff blocker", "subject_type": "issue",
                "relation": "blocks", "object_text": "release", "object_type": "task",
                "timestamp": "", "quote": "", "meeting_id": "m1",
            },
            {
                "subject_text": "sign-off blocker", "subject_type": "issue",
                "relation": "raises_issue", "object_text": "Aiko - PM", "object_type": "person",
                "timestamp": "", "quote": "", "meeting_id": "m1",
            },
            {
                "subject_text": "Aiko", "subject_type": "person",
                "relation": "owns", "object_text": "release", "object_type": "task",
                "timestamp": "", "quote": "", "meeting_id": "m1",
            },
        ]
        embedding_by_text = {
            "signoff blocker": [1.0, 0.0],
            "sign-off blocker": [0.99, 0.14106736],  # should merge with "signoff blocker"
            "release": [1.0, 0.0],
            "Aiko - PM": [0.0, 1.0],
            "Aiko": [1.0, 0.0],  # would look similar to "release" but person is not merged
        }
        with patch.object(bkg, "embed_texts", self._fake_embed_texts(embedding_by_text)):
            merged = bkg.merge_similar_entities(triples, threshold=0.85)

        issue_texts = {t["subject_text"] for t in merged if t["subject_type"] == "issue"}
        self.assertEqual(len(issue_texts), 1, "issue mentions should collapse onto one canonical text")

        person_texts = set()
        for t in merged:
            if t["subject_type"] == "person":
                person_texts.add(t["subject_text"])
            if t["object_type"] == "person":
                person_texts.add(t["object_text"])
        self.assertEqual(person_texts, {"Aiko - PM", "Aiko"}, "person mentions must be untouched by this pass")

    def test_does_not_mutate_input_triples(self):
        triples = [
            {
                "subject_text": "signoff blocker", "subject_type": "issue",
                "relation": "blocks", "object_text": "sign-off blocker", "object_type": "issue",
                "timestamp": "", "quote": "", "meeting_id": "m1",
            },
        ]
        embedding_by_text = {
            "signoff blocker": [1.0, 0.0],
            "sign-off blocker": [0.99, 0.14106736],
        }
        original_subject_text = triples[0]["subject_text"]
        with patch.object(bkg, "embed_texts", self._fake_embed_texts(embedding_by_text)):
            bkg.merge_similar_entities(triples, threshold=0.85)
        self.assertEqual(triples[0]["subject_text"], original_subject_text)

    def test_below_threshold_leaves_mentions_distinct(self):
        triples = [
            {
                "subject_text": "signoff blocker", "subject_type": "issue",
                "relation": "blocks", "object_text": "release", "object_type": "task",
                "timestamp": "", "quote": "", "meeting_id": "m1",
            },
            {
                "subject_text": "unrelated issue", "subject_type": "issue",
                "relation": "blocks", "object_text": "release", "object_type": "task",
                "timestamp": "", "quote": "", "meeting_id": "m1",
            },
        ]
        embedding_by_text = {
            "signoff blocker": [1.0, 0.0],
            "unrelated issue": [0.0, 1.0],
            "release": [1.0, 0.0],
        }
        with patch.object(bkg, "embed_texts", self._fake_embed_texts(embedding_by_text)):
            merged = bkg.merge_similar_entities(triples, threshold=0.85)
        issue_texts = {t["subject_text"] for t in merged if t["subject_type"] == "issue"}
        self.assertEqual(issue_texts, {"signoff blocker", "unrelated issue"})


if __name__ == "__main__":
    unittest.main()
