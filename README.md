# datasette-embeddings

[![PyPI](https://img.shields.io/pypi/v/datasette-embeddings.svg)](https://pypi.org/project/datasette-embeddings/)
[![Changelog](https://img.shields.io/github/v/release/datasette/datasette-embeddings?include_prereleases&label=changelog)](https://github.com/datasette/datasette-embeddings/releases)
[![Tests](https://github.com/datasette/datasette-embeddings/actions/workflows/test.yml/badge.svg)](https://github.com/datasette/datasette-embeddings/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/datasette/datasette-embeddings/blob/main/LICENSE)

Store and query embedding vectors in Datasette tables

## Installation

Install this plugin in the same environment as Datasette.
```bash
datasette install datasette-embeddings
```
## Usage

Usage instructions go here.

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:
```bash
cd datasette-embeddings
python3 -m venv venv
source venv/bin/activate
```
Now install the dependencies and test dependencies:
```bash
pip install -e '.[test]'
```
To run the tests:
```bash
pytest
```
The tests use captured examples of embedding APIs. The easiest way to re-generate these is to do the following:

- `rm -rf tests/cassettes` to remove the previous recordings
- `export OPENAPI_API_KEY='...'` to set an OpenAI API key
- `pytest --record-mode once` to recreate the cassettes

