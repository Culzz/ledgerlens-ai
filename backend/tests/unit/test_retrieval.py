"""Tests for the retrieval layer: evidence units and the BM25 index."""

from __future__ import annotations

import pytest
from app.services.dataset import FinQAExample
from app.services.retrieval import (
    FEATURES,
    TABLE,
    TEXT,
    BM25Index,
    Candidate,
    EvidenceUnit,
    Weights,
    answerable_units,
    build_units,
    evidence_kinds,
    gold_unit_ids,
    lexical_weights,
    rerank,
    row_text,
    tokenize,
    unit_id,
)
from app.services.retrieval.rerank import features


def example(**overrides) -> FinQAExample:
    """A small document: a two-row table and two narrative lines."""
    defaults = {
        "id": "DOC/1",
        "question": "what was net sales in 2016?",
        "table": (("", "2016", "2015"), ("net sales", "100", "90")),
        "pre_text": ("the company reported net sales of 100.",),
        "post_text": ("amounts are in millions.",),
        "program": "table_max(net sales, 2016)",
        "exe_ans": 100,
        "ann_table_rows": (1,),
        "ann_text_rows": (0,),
    }
    defaults.update(overrides)
    return FinQAExample(**defaults)


class TestUnits:
    def test_one_unit_per_row_and_per_narrative_line(self):
        assert len(build_units(example())) == 4

    def test_rows_come_first_then_text(self):
        assert [u.kind for u in build_units(example())] == [TABLE, TABLE, TEXT, TEXT]

    def test_row_text_attaches_headers_to_values(self):
        text = row_text(example().table, 1)
        assert "net sales" in text
        assert "2016 100" in text
        assert "2015 90" in text

    def test_header_row_does_not_repeat_itself(self):
        assert row_text(example().table, 0) == "2016 2015"

    def test_short_row_does_not_crash(self):
        table = (("", "2016", "2015"), ("odd row", "1"))
        assert row_text(table, 1) == "odd row 2016 1"

    def test_ids_are_unique_and_stable(self):
        units = build_units(example())
        assert len({u.id for u in units}) == len(units)
        assert units[0].id == unit_id("DOC/1", TABLE, 0)
        assert build_units(example())[0].id == units[0].id

    def test_text_units_follow_pre_then_post_order(self):
        ex = example(pre_text=("first",), post_text=("second",))
        assert [u.text for u in build_units(ex) if u.kind == TEXT] == ["first", "second"]

    def test_empty_narrative_lines_are_kept_so_indices_stay_total(self):
        ex = example(pre_text=("a", "", "b"), post_text=())
        text_units = [u for u in build_units(ex) if u.kind == TEXT]
        assert [u.index for u in text_units] == [0, 1, 2]

    def test_gold_ids_map_annotation_indices_to_units(self):
        assert gold_unit_ids(example()) == {
            unit_id("DOC/1", TABLE, 1),
            unit_id("DOC/1", TEXT, 0),
        }

    def test_gold_is_empty_without_annotations(self):
        assert gold_unit_ids(example(ann_table_rows=(), ann_text_rows=())) == set()

    @pytest.mark.parametrize(
        "table_rows,text_rows,expected",
        [
            ((1,), (), ("table",)),
            ((), (0,), ("text",)),
            ((1,), (0,), ("table", "text")),
        ],
    )
    def test_evidence_kinds(self, table_rows, text_rows, expected):
        ex = example(ann_table_rows=table_rows, ann_text_rows=text_rows)
        assert evidence_kinds(ex) == expected


class TestAnswerableUnits:
    """Row 0 of a table is a label, not evidence. Indexed, never returned."""

    def test_the_header_row_is_indexed(self):
        header = next(u for u in build_units(example()) if u.kind == TABLE and u.index == 0)
        assert header.text == "2016 2015"

    def test_the_header_row_is_not_answerable(self):
        header = next(u for u in build_units(example()) if u.kind == TABLE and u.index == 0)
        assert header.answerable is False

    def test_every_data_row_is_answerable(self):
        rows = [u for u in build_units(example()) if u.kind == TABLE]
        assert [u.answerable for u in rows] == [False, True]

    def test_narrative_lines_are_answerable(self):
        assert all(u.answerable for u in build_units(example()) if u.kind == TEXT)

    def test_answerable_units_is_the_one_place_the_header_is_left_out(self):
        everything = {u.id for u in build_units(example())}
        answerable = {u.id for u in answerable_units(example())}
        assert everything - answerable == {unit_id("DOC/1", TABLE, 0)}

    def test_the_header_row_is_the_only_thing_left_out(self):
        # Nothing else is a label. An empty narrative line is still a unit, because
        # the annotation indices have to keep lining up.
        assert all(u.answerable for u in build_units(example()) if u.kind == TEXT)


class TestRerank:
    """A deterministic second pass over the head of a lexical ranking."""

    @staticmethod
    def units() -> dict[str, EvidenceUnit]:
        return {unit.id: unit for unit in answerable_units(example())}

    @staticmethod
    def row(units):
        return units[unit_id("DOC/1", TABLE, 1)]

    @staticmethod
    def line(units, index=0):
        return units[unit_id("DOC/1", TEXT, index)]

    def test_the_feature_names_and_the_weights_stay_in_step(self):
        # A weight is set by name and read by position. If the two ever disagree,
        # every weight silently means the wrong feature and nothing else notices.
        assert FEATURES == ("bm25", "numbers", "coverage", "phrase", "kind")
        # The default is BM25 on its own: the order retrieval had already given.
        assert Weights().vector() == (1.0, 0.0, 0.0, 0.0, 0.0)
        assert lexical_weights() == Weights()
        named = Weights(bm25=1.0, numbers=2.0, coverage=3.0, phrase=4.0, kind=5.0)
        assert named.vector() == (1.0, 2.0, 3.0, 4.0, 5.0)

    def test_no_weights_is_the_lexical_order(self):
        units = self.units()
        candidates = [
            Candidate(unit=self.line(units, 0), score=2.0),
            Candidate(unit=self.row(units), score=1.0),
        ]
        ranked = rerank("net sales", candidates, weights=lexical_weights())
        assert [doc for doc, _ in ranked] == [self.line(units, 0).id, self.row(units).id]

    def test_a_table_prior_lifts_a_row_over_a_better_scoring_line(self):
        units = self.units()
        candidates = [
            Candidate(unit=self.row(units), score=1.0),
            Candidate(unit=self.line(units, 0), score=2.0),
        ]
        ranked = rerank("net sales", candidates, weights=Weights(bm25=1.0, kind=1.0))
        assert next(doc for doc, _ in ranked) == self.row(units).id

    def test_a_unit_naming_the_questions_figure_comes_first(self):
        units = self.units()
        # The row states 2016 and 100. The second narrative line states no figure.
        candidates = [
            Candidate(unit=self.line(units, 1), score=1.0),
            Candidate(unit=self.row(units), score=1.0),
        ]
        ranked = rerank(
            "what were net sales of 100 in 2016", candidates, weights=Weights(bm25=0.0, numbers=1.0)
        )
        assert [doc for doc, _ in ranked] == [self.row(units).id, self.line(units, 1).id]

    def test_a_run_of_words_beats_the_same_words_scattered(self):
        together = EvidenceUnit(id="a", example_id="D", kind=TEXT, index=0, text="net sales grew")
        apart = EvidenceUnit(id="b", example_id="D", kind=TEXT, index=1, text="net grew, and sales")
        ranked = rerank(
            "net sales",
            [Candidate(unit=apart, score=1.0), Candidate(unit=together, score=1.0)],
            weights=Weights(bm25=0.0, phrase=1.0),
        )
        assert [doc for doc, _ in ranked] == ["a", "b"]

    def test_features_are_bounded_to_one(self):
        vector = features(
            tokenize("a long question about net sales in 2016 and 2015"),
            tokenize("net sales 2016 100 2015 90"),
            is_table=True,
            score=9.0,
            top_score=9.0,
        )
        assert len(vector) == len(FEATURES)
        assert all(0.0 <= value <= 1.0 for value in vector)

    def test_a_narrative_line_is_not_given_the_table_prior(self):
        vector = features(
            tokenize("net sales"),
            tokenize("net sales rose"),
            is_table=False,
            score=1.0,
            top_score=1.0,
        )
        assert vector[FEATURES.index("kind")] == 0.0

    def test_an_empty_candidate_list_is_an_empty_ranking(self):
        assert rerank("anything", []) == []

    def test_top_k_limits_the_ranking(self):
        candidates = [Candidate(unit=unit, score=1.0) for unit in self.units().values()]
        assert len(rerank("net sales", candidates, top_k=1)) == 1

    def test_the_order_does_not_depend_on_the_order_given(self):
        candidates = [Candidate(unit=unit, score=1.0) for unit in self.units().values()]
        weights = Weights(bm25=1.0, kind=1.0, numbers=0.5)
        assert rerank("net sales 2016", candidates, weights=weights) == rerank(
            "net sales 2016", list(reversed(candidates)), weights=weights
        )

    def test_scores_come_back_descending(self):
        candidates = [Candidate(unit=unit, score=1.0) for unit in self.units().values()]
        scores = [score for _, score in rerank("net sales", candidates)]
        assert scores == sorted(scores, reverse=True)


class TestTokenize:
    def test_lowercases(self):
        assert tokenize("Net Sales") == ["net", "sales"]

    def test_keeps_numbers(self):
        assert tokenize("2016") == ["2016"]

    def test_thousands_separators_are_removed(self):
        assert tokenize("1,234") == ["1234"]

    def test_decimals_stay_whole(self):
        assert tokenize("12.5") == ["12.5"]

    def test_percent_sign_is_dropped(self):
        assert tokenize("12.5%") == ["12.5"]

    def test_punctuation_is_dropped(self):
        assert tokenize("net (sales), 2016.") == ["net", "sales", "2016"]

    def test_empty_text(self):
        assert tokenize("") == []


class TestBM25:
    @staticmethod
    def index() -> BM25Index:
        idx = BM25Index()
        idx.add("a", "net sales 2016 100")
        idx.add("b", "net sales 2015 90")
        idx.add("c", "operating income 2016 30")
        idx.add("d", "total assets 2014 500")
        idx.finalize()
        return idx

    def test_matching_document_ranks_first(self):
        assert self.index().search("net sales 2016")[0][0] == "a"

    def test_non_matching_documents_are_omitted(self):
        assert self.index().search("inventory") == []

    def test_candidates_restrict_the_search(self):
        ranked = self.index().search("net sales 2016", candidates={"b", "c", "d"})
        assert "a" not in {doc for doc, _ in ranked}

    def test_top_k_limits_results(self):
        assert len(self.index().search("net sales", top_k=1)) == 1

    def test_scores_are_descending(self):
        scores = [score for _, score in self.index().search("net sales 2016 100")]
        assert scores == sorted(scores, reverse=True)

    def test_scores_are_never_negative(self):
        ranked = self.index().search("net sales operating income total assets")
        assert all(score >= 0 for _, score in ranked)

    def test_missing_term_has_zero_idf(self):
        assert self.index().idf("nonexistent") == 0.0

    def test_duplicate_document_id_is_rejected(self):
        idx = BM25Index()
        idx.add("a", "one")
        with pytest.raises(ValueError, match="already indexed"):
            idx.add("a", "two")

    def test_average_length_is_computed(self):
        idx = BM25Index()
        idx.add("a", "one two")
        idx.add("b", "one two three four")
        idx.finalize()
        assert idx.average_length == 3.0

    def test_add_many_finalizes(self):
        idx = BM25Index()
        idx.add_many([("a", "one"), ("b", "two")])
        assert len(idx) == 2
        assert idx.average_length == 1.0

    def test_ranking_is_deterministic_for_ties(self):
        idx = BM25Index()
        idx.add("b", "same text")
        idx.add("a", "same text")
        idx.finalize()
        assert [doc for doc, _ in idx.search("same")] == ["a", "b"]

    def test_empty_index_returns_nothing(self):
        idx = BM25Index()
        idx.finalize()
        assert idx.search("anything") == []
