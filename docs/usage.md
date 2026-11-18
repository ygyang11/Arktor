# Arktor usage

## Install

Install the SDK with `python -m pip install arktor` on Python 3.11 or newer.
File and terminal tools are included. Pass the ones you need via `tools=`.
Install `arktor[app]` for web and PDF dependencies.
[SDK quickstart and examples](../README.md#python-sdk)

Install or upgrade the CLI using the same command.

```sh
curl -fsSL https://raw.githubusercontent.com/ygyang11/Arktor/main/scripts/install.sh | sh
```

Replace `sh` with `sh -s -- --version 0.6.0` to pin a release.
To install the CLI in your Python environment, use
`python -m pip install 'arktor[cli,app]'`.

## Configure

The CLI uses `ARKTOR_CONFIG` when set. Otherwise it looks for `./arktor.yaml`,
then `~/.arktor/arktor.yaml`, creating the user file on first launch if needed.
Set your model and credentials in YAML. [Configuration template](../arktor.example.yaml)

The CLI loads `.env`. For SDK use, set variables in the process environment.
`ARKTOR_LLM_API_KEY` takes priority over the YAML key, which takes priority over
the selected provider's environment key. [Environment variables](../.env.example)

## Run from the command line

Run from your project directory.

```sh
arktor
arktor -p 'Inspect this project' --max-steps 40
```

Use `--output-format json` for JSONL output in headless mode and
`--record-file interactions.jsonl` to save model and tool activity to a new file.
Run `arktor --help` for all CLI options.
