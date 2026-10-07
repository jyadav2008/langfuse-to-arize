# lfmigrate — Langfuse → Arize AX migration

Migrates a Langfuse project's history into Arize AX: traces and spans,
evaluation scores, human annotations, sessions, prompts, datasets, score
configs and LLM-as-judge evaluators.

You supply credentials for both systems. The tool discovers the history,
splits it into days, uploads it, reads it back to prove it landed, and can be
interrupted and resumed at any point.

**Verified at scale:** 1,000,199 spans across 60 days migrated and read back in
about 25 minutes (Langfuse v4.53 self-hosted → Arize AX).

---

## Contents

1. [What migrates](#1-what-migrates)
2. [Before you start](#2-before-you-start)
3. [Install](#3-install)
4. [Configure](#4-configure)
5. [Run a migration, step by step](#5-run-a-migration-step-by-step)
6. [Reading the results](#6-reading-the-results)
7. [How long it takes](#7-how-long-it-takes)
8. [Several Langfuse projects](#8-several-langfuse-projects)
9. [Evaluators and evaluation rules](#9-evaluators-and-evaluation-rules)
10. [Troubleshooting](#10-troubleshooting)
11. [Known limitations](#11-known-limitations)
12. [Security](#12-security)
13. [Reference](#13-reference)

---

## 1. What migrates

| Langfuse | Arize AX | Notes |
|---|---|---|
| Traces and observations | Spans in one AX project | Hierarchy, timestamps, inputs, outputs, metadata, model, token usage and cost preserved |
| Sessions | Spans keep their `session.id` | Sessions appear in AX's session view |
| Scores on traces / observations | `trace_eval.*` / `eval.*` columns | Uploaded with the spans |
| Scores on sessions | `session_eval.*` columns | Uploaded with the spans — see [known limitations](#11-known-limitations) |
| Human annotations | AX annotations | Written per day after upload |
| Score configs | Annotation configs | Same names as the migrated score columns, so they bind |
| Prompts, all versions | Prompts with versions and labels | Templates carried over unchanged |
| Datasets and items | Datasets with examples | Item `input` keys become columns |
| LLM-as-judge evaluators | Template evaluators | Needs an AX AI integration — see [§9](#9-evaluators-and-evaluation-rules) |
| Code evaluators | Exported to files | Manual port — see [§9](#9-evaluators-and-evaluation-rules) |
| Evaluation rules | Evaluation tasks | Planned only, unless you opt in |

**Scope is one Langfuse project per run.** A Langfuse key pair belongs to
exactly one project, so the keys you supply decide what is migrated. See
[§8](#8-several-langfuse-projects) for more than one.

---

## 2. Before you start

**On the machine running the migration**
- Python **3.10 or newer**.
- Network access to your Langfuse host and to Arize AX.
- Disk space for local shards: about **2 GB per million spans**.
- Memory: about **1.1 GB** peak at a million spans.
- Run it **close to Langfuse** where you can (same VPC or region). Reading from
  Langfuse is a large share of the run time, and distance multiplies it.

**From Langfuse**
- The project's **public and secret key** (Project settings → API keys).
- The **host URL**, including any path prefix on a self-hosted deployment.

**From Arize AX**
- An **API key** with write access to the target space.
- The **space ID** (Space settings).
- Your space's **region**, if it is not the default US one. A wrong region
  looks exactly like bad credentials — see [Troubleshooting](#10-troubleshooting).
- Optional: an **AI integration ID**, only if you want LLM-as-judge evaluators
  migrated.

**Retention:** Arize accepts spans up to **2 years** old. Older days are
skipped and reported at preflight.

---

## 3. Install

```bash
cd langfuse-migration
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Check it:

```bash
.venv/bin/python -m lfmigrate --help
```

The examples below use `.venv/bin/python`. If you activate the environment
(`source .venv/bin/activate`), plain `python` works too.

---

## 4. Configure

Create your credentials file from the template and lock it down:

```bash
cp .env.example .env
chmod 600 .env
```

`chmod 600` is enforced, not just advised: the tool **refuses to read** a
credentials file that other users can read.

Fill in the five required values:

```bash
LANGFUSE_HOST=https://langfuse.example.com
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
ARIZE_API_KEY=ak-...
ARIZE_SPACE_ID=U3BhY2U6...
```

Set the region if your space is not in the default US region:

```bash
ARIZE_REGION=eu              # us | us-central-1a | us-east-1b | eu
```

Everything else has a safe default. To see the full list with defaults and
explanations:

```bash
.venv/bin/python -m lfmigrate config --list
```

To check what was actually picked up — secrets are shown as a prefix and
length only, never in full:

```bash
.venv/bin/python -m lfmigrate config
```

A wrong length here usually means the key was pasted with a trailing comment
or line break.

> **No secret is ever accepted on the command line**, so credentials cannot
> end up in shell history, `ps` output or CI logs. Use the `.env` file or
> environment variables.

---

## 5. Run a migration, step by step

### Step 1 — Preflight (read-only)

```bash
.venv/bin/python -m lfmigrate preflight
```

Creates nothing. Checks both sides and prints one line per check:

```
[ok  ] credentials present: all set
[ok  ] AX region resolved: us: otlp=otlp.arize.com ...
[ok  ] Langfuse API reachable: generation v4
[ok  ] Langfuse project scope: 'my-project' -- these keys see this project only
[ok  ] Langfuse history range: 2026-08-08 .. 2026-10-06
[ok  ] history within AX retention window
[ok  ] AX destination reachable: space=... project=my-project-20261007-064157
[ok  ] destination project is fresh: unused
[ok  ] Langfuse source is quiet: 1,000,199 observations, unchanged across preflight
```

Two lines are worth reading closely:

- **Project scope** confirms which Langfuse project these keys belong to. This
  is your last chance to catch the wrong key pair before anything is written.
- **Source is quiet.** If this warns that the count moved, something is still
  writing into Langfuse. For a bulk load or import, wait for it to finish. For
  live production traffic it's expected — see [limitations](#11-known-limitations).

Fix anything marked `FAIL` before going on.

### Step 2 — Dry run (read-only)

```bash
.venv/bin/python -m lfmigrate run --dry-run
```

Shows how many days will be migrated and which ingest path the tool will use.
Still writes nothing.

### Step 3 — Trial run on one day (recommended)

```bash
.venv/bin/python -m lfmigrate run --root out/trial --max-days 1
```

Runs the whole path — export, upload, read-back, scores — on the oldest day.
Open the link it prints at the end and check the result in Arize before
committing to the full history.

A trial creates its own AX project. Delete it afterwards, but **never reuse a
deleted project's name**; the tool never does this itself.

### Step 4 — The full migration

```bash
.venv/bin/python -m lfmigrate run --root out/run1
```

Use a `--root` folder of your own per migration and **keep it** until you are
done. It holds the shards and the ledger that make resuming, `status`,
`verify` and `apply-scores` work.

#### Running it in the background

For a large migration, let it run unattended so a closed terminal or a
sleeping laptop does not stop it:

```bash
nohup caffeinate -i .venv/bin/python -u -m lfmigrate run --root out/run1 \
  > run1.log 2>&1 &
echo $! > run1.pid
```

- `nohup` keeps it running after you close the terminal.
- `caffeinate -i` stops a Mac from sleeping (macOS only; on Linux, drop it).
- `-u` writes the log line by line so you can follow it.

Follow progress:

```bash
tail -f run1.log | grep --line-buffered -E "exported in|verif|waiting|breakdown|Done|error|exit"
```

Check whether it is still running:

```bash
ps -p $(cat run1.pid) >/dev/null && echo running || echo finished
```

### Step 5 — If it stops part-way

Run **exactly the same command again**, with the same `--root`:

```bash
.venv/bin/python -m lfmigrate run --root out/run1
```

Completed days are skipped and nothing is sent twice. Each day's records,
batch states and pending scores are saved to disk as soon as the day
completes, so an interruption loses nothing.

### Step 6 — Check the result

The run reads everything back on its own. To check again later:

```bash
.venv/bin/python -m lfmigrate status --root out/run1    # local, no network
.venv/bin/python -m lfmigrate verify --root out/run1    # reads back from Arize
```

`verify` compares what is readable in Arize with what was sent, **day by day**,
and names any day that comes up short.

### Step 7 — Only if scores were reported as not applied

```bash
.venv/bin/python -m lfmigrate apply-scores --root out/run1
```

Safe to run more than once. Only needed if the run's output says scores could
not be applied (for example after a `--no-verify` run).

---

## 6. Reading the results

### The end of a run

```
  verified 1000199 row(s) across 60 day(s)
  runtime breakdown:
    preflight      1.16s    0.1%
    resources      1.77s    0.1%
    export       276.65s   18.5%
    import       710.12s   47.4%   1,408 spans/s
    verify       501.25s   33.5%
    total       1497.89s  100.0%
  view in Arize (time range pre-set to the migrated days):
    https://app.arize.com/organizations/.../spaces/.../models/modelName/...
Done. 1000199 span(s) migrated into project 'my-project-20261007-064157'.
```

Every day also gets its own line as it goes:

```
  2026-08-08: exported in 1.99s, imported in 5.12s (1,377 spans/s)
```

### Exit codes

| Code | Meaning |
|---|---|
| **0** | Everything migrated and verified |
| **2** | Spans migrated, but something needs attention — each item is listed under `exit 2:` |
| **1** | The run failed. Fix the cause and re-run the same command to resume |

A run started with `nohup` does not show the exit code. Look at the end of the
log instead: `Done.` with no `exit 2:` block means success.

### Finding the data in Arize

Use the link the run prints. **It matters:** migrated spans keep their
original, historical timestamps, and Arize's trace list only shows the time
range selected in the UI. With the default "last 24 hours", a backfill looks
empty. The link sets the range to the migrated days for you.

### Project names

By default the destination is named after the Langfuse project plus a
timestamp, for example `my-project-20261007-064157`. The timestamp guarantees
a name is never reused (see [Troubleshooting](#a-project-name-was-deleted-and-reused)).

| Setting | Result |
|---|---|
| neither set (default) | `<langfuse-project>-<timestamp>` |
| `ARIZE_PROJECT_PREFIX=acme-history` | `acme-history-<timestamp>` |
| `ARIZE_PROJECT_NAME=acme-history` | `acme-history` exactly — never one you have deleted |

A resumed run always keeps the project it started with.

---

## 7. How long it takes

Measured end to end:

| Migration | Path | Wall time |
|---|---|---|
| 1,000,199 spans, 60 days | bulk | **~25 min** — export 4.6 min, upload 11.8 min (1,408 spans/s), verify 8.4 min |
| 11,313 spans, 14 days | OTLP | **29 s** |

Rough rule for large migrations: **about 25 minutes per million spans**, plus
one read-back wait of 5–8 minutes at the end. Reading from Langfuse runs at
about 3,700 observations per second when the tool runs near Langfuse; across a
VPC or region boundary, expect that part to take longer.

### Ingest path

There are two ways to send spans to Arize, and the tool picks one
automatically (`LFMIGRATE_INGEST_PATH=auto`):

| Path | When `auto` picks it | Why |
|---|---|---|
| **Bulk** (Arrow) | History older than 31 days, or more than ~830k spans | Faster per span, and the only path that can attach scores to old spans |
| **OTLP** | Recent, smaller migrations | Data becomes visible in seconds rather than minutes |

A real historical backfill will almost always use the bulk path. The reason
is printed at the start of the run. You can force a path with
`--ingest arrow` (bulk) or `--ingest otlp`, but `auto` is the right choice
unless you have a specific reason.

### Things that make it faster

- **Run near Langfuse.**
- **`--no-validate`** after a successful trial run: skips the SDK's local
  data checks, roughly 30% faster uploads.
- **Leave verification on the default** (`end`): one read-back for the whole
  migration. `--verify-each-day` waits for indexing after every day and adds
  5–8 minutes *per day*.

---

## 8. Several Langfuse projects

Run the tool once per project, each with that project's own keys and its own
`--root`:

```bash
.venv/bin/python -m lfmigrate run --env-file .env.project-a --root out/project-a
.venv/bin/python -m lfmigrate run --env-file .env.project-b --root out/project-b
```

Separate roots are required, not just tidy: a root is bound to one destination
project, and reusing it for a different one is refused.

Prompts, datasets, score configs and evaluators belong to the whole Arize
**space**, not one project. Anything that already exists by name is left
alone, so migrating several projects into one space is safe.

---

## 9. Evaluators and evaluation rules

### LLM-as-judge evaluators

Migrated as Arize template evaluators, every version, oldest first. To run,
an Arize judge needs an AI integration — a stored model-provider connection.
**Model API keys do not migrate from Langfuse**, so choose the integration
explicitly:

```bash
ARIZE_AI_INTEGRATION_ID=<integration id>                 # one for all judges
LFMIGRATE_EVAL_INTEGRATIONS=openai=<id>,anthropic=<id>   # or per provider
LFMIGRATE_EVAL_DEFAULT_MODEL=gpt-4o-mini                 # for judges on Langfuse's default model
```

Without an integration ID, judges are **skipped**, and the run says so. The
integration decides whose credentials and whose bill a judge runs on, so the
tool never guesses one.

What changes in translation, all reported in the output:

- Chat messages become one template, with roles kept as `[SYSTEM]` / `[USER]`
  headers.
- Template variables change from `{{var}}` to `{var}`. Other braces are
  escaped, so a JSON example inside a prompt comes through exactly.
- Numeric scores with a range (for example 0–1) become 11 evenly spaced
  choices, because Arize judges have no numeric output type. Open-ended
  numeric scores become free-text output.
- Category scores are taken from the matching Langfuse score config. If none
  matches, they are assigned by position and the output says so.

### Code evaluators

**Exported to files, not created in Arize.** Arize runs a Python evaluator
*class*, Langfuse runs an `evaluate(observation)` function, and Arize has no
TypeScript runtime. Each evaluator is written to `out/code-evaluators/` (set
`LFMIGRATE_EVAL_EXPORT_DIR` to change it) with a header explaining where it
came from and why, ready to port by hand.

### Evaluation rules

A Langfuse evaluation rule becomes an Arize **evaluation task**. Tasks run
judges on live traffic and are billed by your model provider, so by default
the tool only **prints the plan**. To create them:

```bash
LFMIGRATE_CREATE_EVAL_TASKS=true
```

A rule with filters is never scheduled automatically: dropping a filter would
evaluate — and bill — more traffic than the rule ever did. Recreate those
filters in Arize by hand.

### Running only the resource migration

Prompts, datasets, score configs and evaluators are migrated at the start of
every `run`. To migrate only them:

```bash
.venv/bin/python -m lfmigrate resources --dry-run          # see the plan
.venv/bin/python -m lfmigrate resources                    # do it
.venv/bin/python -m lfmigrate resources --only prompts     # one kind
```

To skip them during a span migration, add `--no-resources` to `run`.

---

## 10. Troubleshooting

### "invalid Space ID", "invalid token" or TLS errors with credentials you know are right

Almost always the **region**. An Arize space lives in one region, and a key
used against the wrong one reports bad credentials. Set `ARIZE_REGION`, then
check with `preflight`.

### The project looks empty in Arize

1. **Time range.** Open the link printed at the end of the run; it sets the
   range. Migrated spans keep their historical dates.
2. **Indexing.** Uploads become readable a few minutes after they are
   accepted. The run waits for this, and `verify` checks again later.
3. **A deleted project name** — next item.

### A project name was deleted and reused

If a project is deleted in Arize and its exact name is used again, uploads
**report success but the data never appears**. The tool's timestamped names
make this impossible by default. If you set `ARIZE_PROJECT_NAME` yourself,
never use the name of a project you have deleted.

### "Your query could not be completed… Request timed out" (HTTP 422 from Langfuse)

Langfuse's database timed out under load. The tool retries automatically and
then asks for smaller pages. If the run still stops, re-run the same command
to resume — and if this happens often, check Langfuse's load or run the tool
closer to it.

### "returned a full page with no recognisable pagination marker"

The tool stopped rather than risk migrating part of a list. It means the
Langfuse deployment answered in a format the tool does not recognise. Re-run
once; if it persists, include the full message when reporting it.

### A day is stuck as "uncertain"

An upload was in flight when the process died, so the outcome is genuinely
unknown. The tool **will not** blindly resend it, because Arize does not
de-duplicate spans and a resend could double-count. Run `verify` to see
whether that day's spans arrived.

### Scores reported as "not applied" or "unmatched"

Arize's update path cannot reach spans older than about 31 days. The tool now
sends scores with the spans instead, which has no age limit, so on a normal
run this should not appear. Run `apply-scores` once; anything still reported
is listed with the reason.

### Score names were refused

Arize score names allow letters, digits, spaces and underscores. Other
characters are replaced, and if two Langfuse names would end up identical the
tool **refuses** rather than silently merging two metrics. Rename one in
Langfuse and re-run.

### "is not a recognised setting and was ignored"

A setting in your `.env` is misspelled or not used by this version. Check the
name against `config --list`.

---

## 11. Known limitations

- **Session-level scores: 8 of 12 landed in testing.** All 12 were sent
  correctly, but Arize drops some of them, and a *different* set each time on
  identical data. This is under investigation with Arize. Trace- and
  span-level scores are not affected.
- **Live sources.** If Langfuse keeps receiving traffic during the migration,
  the most recent day is migrated as it stood at that moment, and a resume
  will not revisit a completed day.
- **Annotation authorship.** Arize records the API key's owner as the
  annotation author. When a Langfuse annotation has no comment, its original
  author and time are written into the annotation's note instead; when it has
  a comment, the comment is kept and the original author is not.
- **Code evaluators** are exported for a manual port ([§9](#9-evaluators-and-evaluation-rules)).
- **Filtered evaluation rules** are not scheduled automatically.
- **Source.** Reads from the Langfuse public API (`LANGFUSE_SOURCE_MODE=api`).
  Verified end to end against self-hosted Langfuse v4.53; the Langfuse v3 /
  Cloud API is supported but has not been verified at the same scale. Langfuse
  Cloud plans apply request-rate limits, which slow reading.
- **Retention.** Spans older than 2 years are skipped.

---

## 12. Security

- Credentials come only from `.env` or environment variables, never from
  command-line flags.
- The tool refuses a `.env` file readable by other users (`chmod 600 .env`).
- Secrets are shown as a prefix and length only — in output, errors and
  `config`.
- **The `--root` folder contains your data**: full prompts, responses and
  metadata. Treat it like a database export: keep it access-controlled, never
  attach it to a ticket, and delete it when you are done.
- If a key is ever exposed, rotate it.

---

## 13. Reference

### Commands

| Command | Writes to Arize | Purpose |
|---|---|---|
| `preflight` | no | Read-only checks on both sides |
| `run` | yes | The migration: resources, then spans day by day, then one read-back |
| `resources` | yes | Prompts, datasets, score configs, evaluators only |
| `apply-scores` | yes | Apply any scores a run reported as pending |
| `verify` | no | Read back from Arize and compare, day by day |
| `status` | no | Local progress from the `--root` folder |
| `config` | no | Show settings (`--list` for all, `--example` for a template) |

Options shared by every command, accepted before or after its name:

| Option | Default | |
|---|---|---|
| `--env-file` | `.env` | Credentials file |
| `--root` | `out/manifests` (or `LFMIGRATE_SHARD_ROOT`) | Folder for shards and the ledger |
| `--json` | off | Also print a machine-readable report |
| `--max-past-years` | 2 | Arize retention window |

`run` options:

| Option | Purpose |
|---|---|
| `--dry-run` | Plan only; writes nothing |
| `--max-days N` | Stop after N days (use 1 for a trial) |
| `--ingest {auto,arrow,otlp}` | Force an ingest path (default `auto`) |
| `--no-validate` | Skip local SDK data checks (~30% faster upload) |
| `--no-verify` | Skip the read-back |
| `--verify-each-day` | Wait for indexing after every day (slow) |
| `--no-resources` | Skip prompts, datasets, score configs, evaluators |
| `--batch-size N` | Spans per upload (default 10,000) |
| `--batch-max-mb N` | Size cap per upload (default 32) |
| `--otlp-shards N` | Parallel OTLP exporters (default 8) |

Other commands: `resources --dry-run --only <kind> --no-expand-input`,
`apply-scores --rebuild`, `verify --timeout <seconds>`.

### Settings

Every setting can go in `.env` or the environment. Precedence: command-line
flag, then `.env` / environment, then the default. `config --list` shows all
of them; the ones most often changed:

| Setting | Default | |
|---|---|---|
| `ARIZE_REGION` | `us` | `us`, `us-central-1a`, `us-east-1b`, `eu` |
| `ARIZE_PROJECT_NAME` / `ARIZE_PROJECT_PREFIX` | — | Destination name ([§6](#project-names)) |
| `LFMIGRATE_SHARD_ROOT` | `out/manifests` | Default for `--root` |
| `LFMIGRATE_INGEST_PATH` | `auto` | `auto`, `arrow`, `otlp` |
| `LFMIGRATE_VERIFY_MODE` | `end` | `end`, `each-day`, `off` |
| `LFMIGRATE_VERIFY_TIMEOUT_SECONDS` | 900 | Read-back patience |
| `LFMIGRATE_BATCH_SIZE` | 10000 | Spans per upload |
| `LFMIGRATE_VALIDATE_UPLOAD` | true | Local SDK data checks |
| `ARIZE_AI_INTEGRATION_ID` | — | Needed to migrate LLM judges |
| `LFMIGRATE_CREATE_EVAL_TASKS` | false | Schedule evaluation rules |
| `LANGFUSE_EXPAND_METADATA_KEYS` | — | Metadata keys to fetch untruncated (Langfuse cuts values over 200 characters) |

### Folder layout

```
lfmigrate/            the tool
  cli.py              commands
  pipeline.py         preflight, run, verify, score application
  source/             Langfuse reader
  mapping/            Langfuse -> Arize translation (spans, scores, resources, evaluators)
  upload.py, otlp.py  the two ingest paths
  manifest.py         shards and the resumable ledger
```
