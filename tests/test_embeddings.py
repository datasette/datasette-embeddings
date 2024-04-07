from datasette import version
from datasette_test import Datasette
from packaging.version import parse
from unittest.mock import ANY
import asyncio
import os
import pytest
import sqlite_utils
import urllib.parse


async def _cookies(datasette, path="/-/enrich/data/items/embeddings"):
    cookies = {"ds_actor": datasette.sign({"a": {"id": "root"}}, "actor")}
    csrftoken = (await datasette.client.get(path, cookies=cookies)).cookies[
        "ds_csrftoken"
    ]
    cookies["ds_csrftoken"] = csrftoken
    return cookies


@pytest.mark.vcr(ignore_localhost=True)
@pytest.mark.asyncio
async def test_enrichment(tmpdir):
    data = str(tmpdir / "data.db")
    datasette = Datasette(
        [data],
        plugin_config={
            "datasette-embeddings": {
                "api_key": os.environ.get("OPENAI_API_KEY") or "sk-mock-api-key"
            }
        },
    )
    db = sqlite_utils.Database(data)
    rows = [
        {"id": 1, "name": "One", "description": "First item"},
        {"id": 2, "name": "Two", "description": "Second item"},
    ]
    db["items"].insert_all(rows, pk="id")
    assert set(db.table_names()) == {"items"}

    cookies = await _cookies(datasette)
    post = {
        "model": "text-embedding-3-large-256",
        "template": "{{ name }} {{ description }}",
    }
    post["csrftoken"] = cookies["ds_csrftoken"]
    response = await datasette.client.post(
        "/-/enrich/data/items/embeddings",
        data=post,
        cookies=cookies,
    )
    assert response.status_code == 302
    # Poll for completion
    data_db = datasette.get_database("data")
    attempts = 0
    while True:
        jobs = await data_db.execute("select * from _enrichment_jobs")
        job = dict(jobs.first())
        if job["status"] != "finished":
            await asyncio.sleep(0.3)
            attempts += 1
            assert attempts < 1000
        else:
            break

    assert set(db.table_names()) == {
        "items",
        "_enrichment_errors",
        "_enrichment_jobs",
        "_embeddings_items",
    }
    assert db["_embeddings_items"].schema == (
        "CREATE TABLE [_embeddings_items] (\n"
        "   [id] INTEGER PRIMARY KEY,\n"
        "   [emb_text_embedding_3_large_256] BLOB\n"
        ")"
    )

    errors = [
        dict(row)
        for row in (await data_db.execute("select * from _enrichment_errors")).rows
    ]
    assert not errors

    assert job["status"] == "finished"
    assert job["enrichment"] == "embeddings"
    assert job["done_count"] == 2
    assert job["error_count"] == 0

    results = await data_db.execute("select * from _embeddings_items order by id")
    rows = [dict(r) for r in results.rows]
    assert len(rows) == 2
    assert rows[0]["id"] == 1
    assert isinstance(rows[0]["emb_text_embedding_3_large_256"], bytes)
    return datasette


@pytest.mark.vcr(ignore_localhost=True)
@pytest.mark.parametrize(
    "use_compound_pk,on_sql",
    (
        (False, "items.id = _embeddings_items.id"),
        (
            True,
            "items.category = _embeddings_items.category and items.id = _embeddings_items.id",
        ),
    ),
)
@pytest.mark.asyncio
async def test_similarity_search(tmpdir, use_compound_pk, on_sql):
    datasette = await test_enrichment(tmpdir)
    if use_compound_pk:
        # Modify items and _embeddings_items to simulate a compound pk
        db = sqlite_utils.Database(str(tmpdir / "data.db"))
        for table in ("items", "_embeddings_items"):
            db[table].add_column("category", str)
            db.execute(f"update {table} set category = 'cat'")
            db[table].transform(pk=("category", "id"))

    # Should have table actions
    table_html = (await datasette.client.get("/data/items")).text
    assert '><a href="/data/items/-/semantic-search">Semantic search' in table_html
    search_html = (await datasette.client.get("/data/items/-/semantic-search")).text
    assert "<h1>Semantic search" in search_html
    csrftoken = search_html.split('name="csrftoken" value="')[1].split('"')[0]
    post_response = await datasette.client.post(
        "/data/items/-/semantic-search",
        data={"csrftoken": csrftoken, "q": "Second item"},
    )
    assert post_response.status_code == 302
    redirect = post_response.headers["location"]
    assert redirect.startswith("/data?sql=")
    qs = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(redirect).query))
    assert qs["sql"] == (
        "select\n"
        '  "items".*,\n'
        '  embeddings_cosine("_embeddings_items"."emb_text_embedding_3_large_256", unhex(:vector)) as _similarity\n'
        'from "items" join "_embeddings_items"\n'
        f"on {on_sql}\n"
        'where "_embeddings_items"."emb_text_embedding_3_large_256" is not null\n'
        "order by _similarity desc"
    )
    assert len(qs["vector"]) == 2048
    assert qs["_hide_sql"] == "1"
    # Now follow the JSON version of that redirect
    response = await datasette.client.get(
        redirect.replace("/data?sql=", "/data.json?sql=") + "&_shape=objects"
    )
    assert response.status_code == 200
    if use_compound_pk:
        assert response.json()["rows"] == [
            {
                "id": 2,
                "name": "Two",
                "description": "Second item",
                "category": "cat",
                "_similarity": ANY,
            },
            {
                "id": 1,
                "name": "One",
                "description": "First item",
                "category": "cat",
                "_similarity": ANY,
            },
        ]
    else:
        assert response.json()["rows"] == [
            {
                "id": 2,
                "name": "Two",
                "description": "Second item",
                "_similarity": ANY,
            },
            {
                "id": 1,
                "name": "One",
                "description": "First item",
                "_similarity": ANY,
            },
        ]


@pytest.fixture
def db_path(tmpdir):
    path = str(tmpdir / "data.db")
    db = sqlite_utils.Database(path)
    with db.conn:
        db["items"].insert({"id": 1, "name": "Cow"}, pk="id")
        db["_embeddings_items"].insert(
            {"id": 1, "emb_text_embedding_3_large_256": b"123"}, pk="id"
        )
    return path


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", (None, "no_table", "no_api_key", "no_execute_sql"))
@pytest.mark.skipif(
    parse(version.__version__) < parse("1.0a13"),
    reason="Uses config= from Datasette 1.0",
)
async def test_show_semantic_search_action_item(db_path, scenario):
    plugin_config = {}
    if scenario != "no_api_key":
        plugin_config = {"datasette-embeddings": {"api_key": "sk-blah"}}
    config = {}
    if scenario == "no_execute_sql":
        config["databases"] = {"data": {"allow_sql": False}}
    datasette = Datasette([db_path], plugin_config=plugin_config, config=config)
    if scenario == "no_table":
        await datasette.get_database("data").execute_write(
            "drop table _embeddings_items"
        )

    html = (await datasette.client.get("/data/items")).text
    if scenario is None:
        assert ">Semantic" in html
    else:
        assert ">Semantic" not in html
