"""Tests for VDBInsertService and related components in noob_rag_deps.rag.insert."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from noob_rag_deps.rag.insert import (
    InsertItem,
    VDBInsertService,
    _default_doc_id,
    _validate_tenant_id,
)


# --- InsertItem model tests ---


def test_insert_item_valid_fields():
    """合法字段通过校验。"""
    item = InsertItem(text="hello world")
    assert item.text == "hello world"
    assert item.doc_id is None
    assert item.metadata == {}
    assert item.filter_1 is None
    assert item.filter_2 is None
    assert item.filter_3 is None


def test_insert_item_with_all_fields():
    """所有字段显式传入时通过校验。"""
    item = InsertItem(
        text="x" * 100,
        doc_id="doc-1",
        metadata={"k": "v"},
        filter_1="f1",
        filter_2="f2",
        filter_3="f3",
    )
    assert item.doc_id == "doc-1"
    assert item.metadata == {"k": "v"}
    assert item.filter_1 == "f1"
    assert item.filter_2 == "f2"
    assert item.filter_3 == "f3"


def test_insert_item_text_max_length_800():
    """text 恰好 800 字符通过。"""
    item = InsertItem(text="a" * 800)
    assert len(item.text) == 800


def test_insert_item_text_over_800_raises():
    """text 超过 800 字符抛出 ValidationError。"""
    with pytest.raises(ValidationError):
        InsertItem(text="a" * 801)


def test_insert_item_filter_over_20_raises():
    """filter_1/2/3 超过 20 字符抛出 ValidationError。"""
    with pytest.raises(ValidationError):
        InsertItem(text="ok", filter_1="x" * 21)
    with pytest.raises(ValidationError):
        InsertItem(text="ok", filter_2="x" * 21)
    with pytest.raises(ValidationError):
        InsertItem(text="ok", filter_3="x" * 21)


def test_insert_item_filter_max_length_20():
    """filter 恰好 20 字符通过。"""
    item = InsertItem(text="ok", filter_1="x" * 20)
    assert len(item.filter_1) == 20


# --- _default_doc_id tests ---


def test_default_doc_id_same_text_same_hash():
    """相同文本返回相同 hash 前缀。"""
    text = "hello"
    assert _default_doc_id(text) == _default_doc_id(text)


def test_default_doc_id_different_text_different_hash():
    """不同文本返回不同 hash。"""
    assert _default_doc_id("a") != _default_doc_id("b")


def test_default_doc_id_length_32_hex():
    """返回长度为 32 的十六进制字符串。"""
    result = _default_doc_id("any text")
    assert len(result) == 32
    assert all(c in "0123456789abcdef" for c in result)


# --- VDBInsertService.collection_name ---


def test_collection_name_from_tenant_id_with_underscore():
    """tenant_id 含下划线时取前缀为 collection_name。"""
    items = [InsertItem(text="x")]
    svc = VDBInsertService("mycollection_abc123", "space1", items, client=MagicMock())
    assert svc.collection_name == "mycollection"


def test_validate_tenant_id_valid_formats():
    """合法格式的 tenant_id 通过校验。"""
    _validate_tenant_id("coll_abc123")
    _validate_tenant_id("mycollection_tenant1")
    _validate_tenant_id("a_1")


def test_validate_tenant_id_no_underscore_raises():
    """tenant_id 无下划线时抛出 ValueError。"""
    with pytest.raises(ValueError, match="需包含下划线"):
        _validate_tenant_id("single")


def test_validate_tenant_id_empty_identifier_raises():
    """tenant_id 唯一标识符为空时抛出 ValueError。"""
    with pytest.raises(ValueError, match="唯一标识符不能为空"):
        _validate_tenant_id("coll_")


def test_validate_tenant_id_empty_collection_name_raises():
    """tenant_id collection_name 为空时抛出 ValueError。"""
    with pytest.raises(ValueError, match="collection_name 不能为空"):
        _validate_tenant_id("_identifier")


def test_vdb_insert_service_invalid_tenant_id_raises():
    """VDBInsertService 直接实例化时无效 tenant_id 抛出 ValueError。"""
    items = [InsertItem(text="x")]
    with pytest.raises(ValueError, match="需包含下划线"):
        VDBInsertService("invalid", "space1", items, client=MagicMock())


# --- VDBInsertService.create ---


@pytest.mark.asyncio
async def test_create_invalid_tenant_id_raises():
    """create 时无效 tenant_id 抛出 ValueError。"""
    with pytest.raises(ValueError, match="需包含下划线"):
        await VDBInsertService.create("nounderscore", "space1", [{"text": "x"}])


@pytest.mark.asyncio
async def test_create_does_not_call_create_new_collection_when_exists():
    """当 has_collection 为 True 时不调用 create_new_collection。"""
    mock_client = MagicMock()
    with (
        patch("noob_rag_deps.rag.insert.init_vdb", return_value=mock_client),
        patch("noob_rag_deps.rag.insert.has_collection", return_value=True),
        patch("noob_rag_deps.rag.insert.create_new_collection") as create_collection,
    ):
        instance = await VDBInsertService.create(
            "coll_tenant1",
            "space1",
            [{"text": "hello"}],
        )
    create_collection.assert_not_called()
    assert instance.tenant_id == "coll_tenant1"
    assert instance.space_id == "space1"
    assert len(instance.items) == 1
    assert instance.items[0].text == "hello"
    assert instance._client is mock_client


@pytest.mark.asyncio
async def test_create_calls_create_new_collection_when_not_exists():
    """当 has_collection 为 False 时调用 create_new_collection。"""
    mock_client = MagicMock()
    with (
        patch("noob_rag_deps.rag.insert.init_vdb", return_value=mock_client),
        patch("noob_rag_deps.rag.insert.has_collection", return_value=False),
        patch("noob_rag_deps.rag.insert.create_new_collection") as create_collection,
    ):
        await VDBInsertService.create("newcoll_tenant1", "space1", [{"text": "hi"}])
    create_collection.assert_called_once_with(
        collection_name="newcoll",
        client=mock_client,
    )


@pytest.mark.asyncio
async def test_create_validates_items():
    """create 正确校验并构造 InsertItem 列表。"""
    mock_client = MagicMock()
    with (
        patch("noob_rag_deps.rag.insert.init_vdb", return_value=mock_client),
        patch("noob_rag_deps.rag.insert.has_collection", return_value=True),
        patch("noob_rag_deps.rag.insert.create_new_collection"),
    ):
        instance = await VDBInsertService.create(
            "c_t1",
            "s1",
            [
                {"text": "first", "metadata": {"a": 1}},
                {"text": "second", "doc_id": "custom-id"},
            ],
        )
    assert len(instance.items) == 2
    assert instance.items[0].text == "first"
    assert instance.items[0].metadata == {"a": 1}
    assert instance.items[1].text == "second"
    assert instance.items[1].doc_id == "custom-id"


# --- VDBInsertService.run ---


@pytest.mark.asyncio
async def test_run_empty_items_returns_empty_list():
    """items 为空时返回 []，不调用 embedding 和 insert。"""
    mock_client = MagicMock()
    svc = VDBInsertService("c_t1", "s1", [], client=mock_client)
    result = await svc.run()
    assert result == []
    mock_client.insert.assert_not_called()


@pytest.mark.asyncio
async def test_run_calls_insert_with_correct_collection_and_data():
    """正常流程下 insert 被调用一次，collection_name 和 data 正确。"""
    mock_client = MagicMock()
    mock_client.insert.return_value = {"ids": [1, 2]}

    dense_vec = [0.1] * 1024  # match dense_dim from config if needed; mock just needs list

    mock_dense = MagicMock()
    mock_dense.aembed_documents = AsyncMock(return_value=[dense_vec, dense_vec])

    items = [
        InsertItem(text="first", metadata={"k": "v1"}),
        InsertItem(text="second", filter_1="f1"),
    ]
    svc = VDBInsertService("mycoll_tenant1", "space1", items, client=mock_client)

    with patch("noob_rag_deps.rag.insert.NoobDenseEmbedding", return_value=mock_dense):
        result = await svc.run()

    mock_client.insert.assert_called_once()
    call_kw = mock_client.insert.call_args.kwargs
    assert call_kw["collection_name"] == "mycoll"
    data = call_kw["data"]
    assert len(data) == 2

    for i, row in enumerate(data):
        assert "doc_id" in row
        assert "dense_vector" in row
        assert row["text"] == items[i].text
        assert row["metadata"] == items[i].metadata
        assert row["tenant_id"] == "mycoll_tenant1"
        assert row["space_id"] == "space1"
        assert row["filter_1"] == items[i].filter_1
        assert row["filter_2"] == items[i].filter_2
        assert row["filter_3"] == items[i].filter_3
        assert row["is_delete"] is False

    assert result == {"ids": [1, 2]}


@pytest.mark.asyncio
async def test_run_uses_default_doc_id_when_not_provided():
    """当 doc_id 未提供时使用 _default_doc_id(text)。"""
    mock_client = MagicMock()
    mock_client.insert.return_value = {"ids": [1]}

    text = "unique content for hash"
    expected_doc_id = _default_doc_id(text)

    mock_dense = MagicMock()
    mock_dense.aembed_documents = AsyncMock(return_value=[[0.0] * 1024])

    items = [InsertItem(text=text)]
    svc = VDBInsertService("c_t1", "s1", items, client=mock_client)

    with patch("noob_rag_deps.rag.insert.NoobDenseEmbedding", return_value=mock_dense):
        await svc.run()

    data = mock_client.insert.call_args.kwargs["data"]
    assert len(data) == 1
    assert data[0]["doc_id"] == expected_doc_id


@pytest.mark.asyncio
async def test_run_uses_provided_doc_id():
    """当 doc_id 提供时使用提供的值。"""
    mock_client = MagicMock()
    mock_client.insert.return_value = {"ids": [1]}

    mock_dense = MagicMock()
    mock_dense.aembed_documents = AsyncMock(return_value=[[0.0] * 1024])

    items = [InsertItem(text="any", doc_id="my-custom-doc-id")]
    svc = VDBInsertService("c_t1", "s1", items, client=mock_client)

    with patch("noob_rag_deps.rag.insert.NoobDenseEmbedding", return_value=mock_dense):
        await svc.run()

    data = mock_client.insert.call_args.kwargs["data"]
    assert data[0]["doc_id"] == "my-custom-doc-id"
