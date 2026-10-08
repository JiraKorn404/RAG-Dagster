import math

import pytest

from rag_lab.config import (
    ChunkConfig,
    EmbedConfig,
    ExperimentConfig,
    IndexConfig,
    ParseConfig,
    SemanticSettings,
    load_experiment_file,
)
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


def test_ocr_settings_change_the_hash_only_when_ocr_is_on():
    plain = ExperimentConfig(name="a")
    off = ExperimentConfig(name="a", parse=ParseConfig(ocr=False, ocr_model="other", ocr_scale=3.0))
    assert off.config_hash() == plain.config_hash()
    on = ExperimentConfig(name="a", parse=ParseConfig(ocr=True))
    assert on.config_hash() != plain.config_hash()
    assert ExperimentConfig(name="a", parse=ParseConfig(ocr=True, ocr_model="other")).config_hash() != on.config_hash()


def test_hashes_of_experiments_made_before_new_settings_are_unchanged():
    # Computed with the config module as it was before OCR and the embedding templates were added.
    assert ExperimentConfig(name="a").config_hash() == "41d2696cdcf4b66a"
    semantic = ExperimentConfig(
        name="a",
        chunk=ChunkConfig(strategy="semantic"),
        embed=EmbedConfig(model="qwen3-embedding:4b", dimension=512),
    )
    assert semantic.config_hash() == "cbd86fe9b9110d77"
    native = ExperimentConfig(
        name="a",
        chunk=ChunkConfig(engine="native", strategy="fixed", overlap=64),
        index=IndexConfig(sparse=True),
    )
    assert native.config_hash() == "4ccbddec15f1cfbb"
    tagged = ExperimentConfig(name="a", tag="a")
    assert (tagged.config_hash(), tagged.settings_hash()) == ("4b0c7f1a6b956930", "41d2696cdcf4b66a")


def test_embedding_templates_change_the_hash_only_when_they_are_not_qwen3s():
    plain = ExperimentConfig(name="a")
    qwen = ExperimentConfig(name="a", embed=EmbedConfig.for_model("qwen3-embedding:0.6b"))
    assert qwen.config_hash() == plain.config_hash()
    gemma = EmbedConfig.for_model("embeddinggemma-2:740m")
    assert (gemma.tokenizer, gemma.query_template) == (
        "google/embeddinggemma-2",
        "task: search result | query: {text}",
    )
    same_model_qwen_templates = ExperimentConfig(
        name="a", embed=EmbedConfig(model="embeddinggemma-2:740m", tokenizer="google/embeddinggemma-2")
    )
    assert ExperimentConfig(name="a", embed=gemma).config_hash() != same_model_qwen_templates.config_hash()


def test_picture_settings_change_the_hash_only_when_pictures_are_on():
    gemma = EmbedConfig.for_model("embeddinggemma-2:740m")
    plain = ExperimentConfig(name="a", embed=gemma)
    off = ExperimentConfig(
        name="a",
        parse=ParseConfig(picture_scale=3.0, picture_min_side=10),
        embed=EmbedConfig.for_model("embeddinggemma-2:740m", picture_input="image"),
    )
    assert off.config_hash() == plain.config_hash()
    on = ExperimentConfig(name="a", parse=ParseConfig(pictures=True), embed=gemma)
    assert on.config_hash() != plain.config_hash()
    image_only = ExperimentConfig(
        name="a",
        parse=ParseConfig(pictures=True),
        embed=EmbedConfig.for_model("embeddinggemma-2:740m", picture_input="image"),
    )
    assert image_only.config_hash() != on.config_hash()


def test_pictures_as_images_need_a_model_that_takes_images():
    qwen = EmbedConfig.for_model("qwen3-embedding:0.6b")
    with pytest.raises(ValueError, match="takes images"):
        ExperimentConfig(name="a", parse=ParseConfig(pictures=True), embed=qwen)
    caption = EmbedConfig.for_model("qwen3-embedding:0.6b", picture_input="caption")
    ExperimentConfig(name="a", parse=ParseConfig(pictures=True), embed=caption)  # text only: fine


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


def test_experiment_file_fills_defaults_tag_and_embedding_family(tmp_path):
    path = tmp_path / "ingest.yaml"
    path.write_text("name: auto\nembed:\n  model: embeddinggemma-2:740m\nindex:\n  sparse: true\n")
    config = load_experiment_file(path)
    assert (config.name, config.tag, config.index.sparse) == ("auto", "auto", True)
    assert config.chunk == ChunkConfig()  # a section left out keeps its defaults
    assert config.embed == EmbedConfig.for_model("embeddinggemma-2:740m")
    # the same as a tagged experiment built in code, so the pages and the sensor agree on its hash
    assert config.config_hash() == ExperimentConfig(
        name="auto", tag="auto", embed=config.embed, index=IndexConfig(sparse=True)
    ).config_hash()

    path.write_text("name: auto\nembed:\n  model: embeddinggemma-2:740m\n  tokenizer: other/tokenizer\n")
    assert load_experiment_file(path).embed.tokenizer == "other/tokenizer"  # what the file sets wins


def test_experiment_file_refuses_unknown_keys_and_unknown_models(tmp_path):
    path = tmp_path / "ingest.yaml"
    path.write_text("name: auto\nchunk:\n  stratgy: fixed\n")
    with pytest.raises(ValueError, match="chunk.stratgy"):
        load_experiment_file(path)
    path.write_text("name: auto\nembed:\n  model: qwen3-embeding:0.6b\n")
    with pytest.raises(ValueError, match="not of a known embedding family"):
        load_experiment_file(path)
    path.write_text("chunk:\n  strategy: fixed\n")
    with pytest.raises(ValueError, match="name"):
        load_experiment_file(path)
