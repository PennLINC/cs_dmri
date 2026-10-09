# Development

## Layout

| Path | Contents |
|---|---|
| `src/` | The `cs_dmri` Rust crate and the command-line tools (`src/bin/`) |
| `python/` | The Python package: the compiled extension (`python/src/`) and the pure-Python layer (`python/cs_dmri/`) |
| `tests/` | Rust integration tests |
| `python/tests/` | Python tests |
| `docs/` | This documentation |

## Tests

```bash
cargo test --release
cd python && maturin develop --release && pytest tests
```

The Python tests include comparisons with dipy when it is installed, and a
comparison between the Python interface and the command-line tools that runs
when the release binaries and the test data directory are available.

## Documentation

```bash
pip install -r docs/requirements.txt
sphinx-build -W -b html docs docs/_build/html
```

The command-line reference pages are generated from the tools' `--help` output:

```bash
cargo build --release
python docs/tools/make_cli_pages.py
```

Continuous integration checks that the committed pages match the current help
text.

## Releases

The version is set once, in `[workspace.package]` of the top-level
`Cargo.toml`. Pushing a tag `vX.Y.Z` that matches it builds wheels for all
supported platforms and an sdist, and publishes them to PyPI.

## Licensing of contributions

Contributions are accepted under the project licence (MIT or Apache-2.0). Code
ported from MRtrix3 must stay in its own file under the Mozilla Public License
2.0, with the notice and SPDX line used by the existing such files; see
{doc}`license`.
