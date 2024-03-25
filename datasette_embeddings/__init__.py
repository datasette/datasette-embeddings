from datasette import hookimpl
from datasette_enrichments import Enrichment
from datasette.database import Database
import httpx
from typing import List, Optional
from wtforms import Form, SelectField, StringField, PasswordField, TextAreaField
from wtforms.validators import DataRequired
import secrets
import sqlite_utils
import struct
import json


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
                choices=[
                    ("text-embedding-3-large-256", "text-embedding-3-large-256"),
                    ("text-embedding-3-small-512", "text-embedding-3-small-512"),
                    ("text-embedding-3-large-1024", "text-embedding-3-large-1024"),
                    ("text-embedding-3-small", "text-embedding-3-small"),
                    ("text-embedding-3-large", "text-embedding-3-large"),
                ],
                default="text-embedding-3-small-512",
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

    async def calculate_embedding(self, api_key, text, model):
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

    def encode_embedding(self, embedding):
        return struct.pack("<" + "f" * len(embedding), *embedding)

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
        print(rows)
        api_key = resolve_api_key(datasette, config)
        template = config["template"]
        model = config["model"]
        column_name = f"emb_{model.replace('-', '_')}"
        for row in rows:
            print(row)
            text = template
            for key, value in row.items():
                text = text.replace("{{ %s }}" % key, str(value or "")).replace(
                    "{{%s}}" % key, str(value or "")
                )
            embedding = await self.calculate_embedding(api_key, text, model)
            print("  ", embedding)
            encoded_embedding = self.encode_embedding(embedding)
            print("  ", encoded_embedding)
            await db.execute_write(
                f"UPDATE [{table}] SET [{column_name}] = ? WHERE { ' AND '.join([f'[{pk}] = ?' for pk in pks]) }",
                [encoded_embedding] + [row[pk] for pk in pks],
            )


def resolve_api_key(datasette, config):
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
