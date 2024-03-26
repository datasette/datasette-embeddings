from datasette import hookimpl, Response
from datasette_enrichments import Enrichment
from datasette.database import Database
import httpx
import json
import secrets
import sqlite_utils
import sqlite3
import struct
from typing import List, Optional
import urllib.parse
from wtforms import Form, SelectField, StringField, PasswordField, TextAreaField
from wtforms.validators import DataRequired


MODEL_NAMES = (
    "text-embedding-3-large-256",
    "text-embedding-3-small-512",
    "text-embedding-3-large-1024",
    "text-embedding-3-small",
    "text-embedding-3-large",
)
DEFAULT_MODEL = "text-embedding-3-small-512"


async def embedding_column_for_table(datasette, database, table):
    db = datasette.get_database(database)
    columns = await db.table_columns(table)
    emb_columns = [column for column in columns if column.startswith("emb_")]
    if not emb_columns:
        return False, False
    column = emb_columns[0]

    # Figure out which model it is
    model_name = column.replace("emb_", "").replace("_", "-")
    if model_name not in MODEL_NAMES:
        return False, False

    return column, model_name


async def embeddings_semantic_search(datasette, request):
    table = request.url_vars["table"]
    database = request.url_vars["database"]

    embedding_column, model_name = await embedding_column_for_table(
        datasette, database, table
    )
    if not embedding_column:
        datasette.add_message(
            request,
            "Table does not have an embedding column",
            type=datasette.ERROR,
        )

    if request.method == "POST":
        form = await request.post_vars()
        q = (form.get("q") or "").strip()
        if not q:
            datasette.add_message(
                request, "Search query is required", type=datasette.ERROR
            )
            return Response.redirect(request.path)

        # Embed it
        api_key = resolve_api_key(datasette, {})
        vector = await EmbeddingsEnrichment.calculate_embedding(api_key, q, model_name)
        blob = EmbeddingsEnrichment.encode_embedding(vector)

        # Redirect to the SQL query against the table
        sql = """
        select
          *,
          embeddings_cosine({column}, unhex(:vector)) as _similarity
        from "{table}"
        order by _similarity desc
        """.format(
            column=embedding_column, table=table
        )
        return Response.redirect(
            datasette.urls.database(database)
            + "?"
            + urllib.parse.urlencode(
                {"sql": sql, "vector": blob.hex().upper(), "_hide_sql": 1}
            )
        )

    return Response.html(
        await datasette.render_template(
            "embeddings_semantic_search.html", request=request
        )
    )


def embeddings_cosine(binary_a, binary_b):
    a = EmbeddingsEnrichment.decode_embedding(binary_a)
    b = EmbeddingsEnrichment.decode_embedding(binary_b)
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = sum(x * x for x in a) ** 0.5
    magnitude_b = sum(x * x for x in b) ** 0.5
    return dot_product / (magnitude_a * magnitude_b)


def unhex(hex_string):
    try:
        return bytes.fromhex(hex_string)
    except ValueError:
        return None


@hookimpl
def prepare_connection(conn):
    conn.create_function("embeddings_cosine", 2, embeddings_cosine)
    # Check if unhex exists
    try:
        conn.execute("select unhex('AB')")
    except sqlite3.OperationalError:
        conn.create_function("unhex", 1, unhex)


@hookimpl
def table_actions(datasette, database, table):
    has_api_key = False
    try:
        resolve_api_key(datasette)
        has_api_key = True
    except ApiKeyError:
        pass

    async def inner():
        embedding_column, _ = await embedding_column_for_table(
            datasette, database, table
        )
        if embedding_column and has_api_key:
            return [
                {
                    "href": datasette.urls.table(database, table)
                    + "/-/semantic-search",
                    "label": "Semantic search against this table",
                    "description": "Find table rows similar in meaning to your query",
                }
            ]

    return inner


@hookimpl
def register_routes():
    return [
        (
            r"^/(?P<database>[^/]+)/(?P<table>[^/]+)/-/semantic-search$",
            embeddings_semantic_search,
        ),
    ]


@hookimpl
def register_enrichments():
    return [EmbeddingsEnrichment()]


class EmbeddingsEnrichment(Enrichment):
    name = "Text embeddings with OpenAI"
    slug = "embeddings"
    description = "Calculate and store text embeddings using OpenAI's API"
    runs_in_process = True
    batch_size = 1

    async def get_config_form(self, datasette, db, table):
        columns = await db.table_columns(table)

        # Default template uses all string columns
        default = " ".join("{{ COL }}".replace("COL", col) for col in columns)

        url_columns = [col for col in columns if "url" in col.lower()]
        image_url_suggestion = ""
        if url_columns:
            image_url_suggestion = "{{ %s }}" % url_columns[0]

        class ConfigForm(Form):
            model = SelectField(
                "Model",
                choices=[(model_name, model_name) for model_name in MODEL_NAMES],
                default=DEFAULT_MODEL,
            )
            template = TextAreaField(
                "Template",
                description="A template to run against each row to generate the embedding input. Use {{ COL }} for columns.",
                default=default,
                validators=[DataRequired(message="Template is required.")],
                render_kw={"style": "height: 8em"},
            )

        def stash_api_key(form, field):
            if not (field.data or "").startswith("sk-"):
                raise ValidationError("API key must start with sk-")
            if not hasattr(datasette, "_enrichments_embeddings_stashed_keys"):
                datasette._enrichments_embeddings_stashed_keys = {}
            key = secrets.token_urlsafe(16)
            datasette._enrichments_embeddings_stashed_keys[key] = field.data
            field.data = key

        class ConfigFormWithKey(ConfigForm):
            api_key = PasswordField(
                "API key",
                description="Your OpenAI API key",
                validators=[
                    DataRequired(message="API key is required."),
                    stash_api_key,
                ],
                render_kw={"autocomplete": "off"},
            )

        plugin_config = datasette.plugin_config("datasette-embeddings") or {}
        api_key = plugin_config.get("api_key")

        return ConfigForm if api_key else ConfigFormWithKey

    async def initialize(self, datasette, db, table, config):
        # Ensure the embeddings column exists
        model = config["model"]
        column_name = f"emb_{model.replace('-', '_')}"

        def add_column_if_not_exists(conn):
            db = sqlite_utils.Database(conn)
            if column_name not in db[table].columns_dict:
                db[table].add_column(column_name, "BLOB")

        await db.execute_write_fn(add_column_if_not_exists)

    @classmethod
    async def calculate_embedding(cls, api_key, text, model):
        # Add dimensions for models called things that end in -xxx digits
        body = {
            "input": text,
            "model": model,
        }
        last_bit = model.split("-")[-1]
        if last_bit.isdigit():
            body["model"] = "-".join(model.split("-")[:-1])
            body["dimensions"] = int(last_bit)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                json=body,
            )
            response.raise_for_status()
            embedding = response.json()["data"][0]["embedding"]
            return embedding

    @classmethod
    def encode_embedding(cls, embedding):
        return struct.pack("<" + "f" * len(embedding), *embedding)

    @classmethod
    def decode_embedding(cls, binary):
        return struct.unpack("<" + "f" * (len(binary) // 4), binary)

    async def enrich_batch(
        self,
        datasette: "Datasette",
        db: Database,
        table: str,
        rows: List[dict],
        pks: List[str],
        config: dict,
        job_id: int,
    ) -> List[Optional[str]]:
        api_key = resolve_api_key(datasette, config)
        template = config["template"]
        model = config["model"]
        column_name = f"emb_{model.replace('-', '_')}"
        for row in rows:
            text = template
            for key, value in row.items():
                text = text.replace("{{ %s }}" % key, str(value or "")).replace(
                    "{{%s}}" % key, str(value or "")
                )
            embedding = await self.calculate_embedding(api_key, text, model)
            encoded_embedding = self.encode_embedding(embedding)
            await db.execute_write(
                f"UPDATE [{table}] SET [{column_name}] = ? WHERE { ' AND '.join([f'[{pk}] = ?' for pk in pks]) }",
                [encoded_embedding] + [row[pk] for pk in pks],
            )


class ApiKeyError(Exception):
    pass


def resolve_api_key(datasette, config=None):
    config = config or {}
    plugin_config = datasette.plugin_config("datasette-embeddings") or {}
    api_key = plugin_config.get("api_key")
    if api_key:
        return api_key
    # Look for it in config
    api_key_name = config.get("api_key")
    if not api_key_name:
        raise ApiKeyError("No API key reference found in config")
    # Look it up in the stash
    if not hasattr(datasette, "_enrichments_embeddings_stashed_keys"):
        raise ApiKeyError("No API key stash found")
    stashed_keys = datasette._enrichments_embeddings_stashed_keys
    if api_key_name not in stashed_keys:
        raise ApiKeyError("No API key found in stash for {}".format(api_key_name))
    return stashed_keys[api_key_name]
