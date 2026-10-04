import math

from rag_lab.config import ChunkConfig, ExperimentConfig, IndexConfig, ParseConfig, SemanticSettings
from rag_lab.embedding.vectors import truncate_and_normalise


def test_config_hash_ignores_name_but_not_settings():
    a = ExperimentConfig(name="a")
    assert a.config_hash() == ExperimentConfig(name="b").config_hash()
    assert a.config_hash() != ExperimentConfig(name="a", parse=ParseConfig(table_mode="fast")).config_hash()


def test_config_hash_ignores_settings_of_unused_strategies():
    hybrid = ExperimentConfig(name="a", chunk=ChunkConfig(strategy="hybrid"))
    tweaked = ExperimentConfig(
        name="a", chunk=ChunkConfig(strategy="hybrid", semantic=SemanticSettings(buffer_size=3))
    )
    assert hybrid.config_hash() == tweaked.config_hash()
    semantic = ExperimentConfig(
        name="a", chunk=ChunkConfig(strategy="semantic", semantic=SemanticSettings(buffer_size=3))
    )
    assert semantic.config_hash() != ExperimentConfig(
        name="a", chunk=ChunkConfig(strategy="semantic")
    ).config_hash()


def test_config_hash_tells_engines_apart_and_defaults_to_llamaindex():
    default = ExperimentConfig(name="a")
    assert default.config_hash() == ExperimentConfig(
        name="a", chunk=ChunkConfig(engine="llamaindex")
    ).config_hash()
    assert default.config_hash() != ExperimentConfig(
        name="a", chunk=ChunkConfig(engine="native")
    ).config_hash()


def test_sparse_changes_the_hash_only_when_on():
    plain = ExperimentConfig(name="a")
    assert ExperimentConfig(name="a", index=IndexConfig(sparse=False)).config_hash() == plain.config_hash()
    assert ExperimentConfig(name="a", index=IndexConfig(sparse=True)).config_hash() != plain.config_hash()


def test_tag_changes_the_hash_only_when_set_and_never_the_settings_hash():
    plain = ExperimentConfig(name="a")
    assert ExperimentConfig(name="a", tag=None).config_hash() == plain.config_hash()
    tagged = ExperimentConfig(name="a", tag="a")
    assert tagged.config_hash() != plain.config_hash()
    assert ExperimentConfig(name="b", tag="b").config_hash() != tagged.config_hash()
    assert tagged.settings_hash() == plain.settings_hash() == plain.config_hash()


def test_truncate_keeps_dimension_and_unit_length():
    out = truncate_and_normalise([[3.0, 4.0, 12.0, 0.5]], 2)[0]
    assert len(out) == 2
    assert math.isclose(math.sqrt(sum(x * x for x in out)), 1.0)
    assert out == [0.6, 0.8]
