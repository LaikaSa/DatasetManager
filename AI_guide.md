# DatasetManager â€” AI Agent Guide

This file tells an AI agent how to drive **DatasetManager** from the command
line. The app is a PySide6 GUI, but every feature is also exposed through a
minimal, non-interactive CLI built for machines. **Use the CLI, not the GUI.**

Read this top-to-bottom once. When in doubt about a flag, prefer the
self-describing command `--list-commands --json` (below) over guessing.

---

## 1. How to invoke

Always use the project virtualenv Python (it has PySide6 / torch /
onnxruntime / Pillow). The system `python` does **not**.

```
<repo>/.venv/Scripts/python.exe cli.py <command> [flags]
```

- Entry point: `cli.py` at the repo root.
- `run.py` is the GUI launcher â€” do **not** use it from an agent.
- Every command is **non-interactive**: no prompts, no stdin. If a flag is
  missing you get a usage error (exit 2), never a hang.
- All paths are plain arguments. Use forward or back slashes (Windows).

### Self-discovery (preferred over this doc)

```
cli.py --list-commands --json
```

Prints a JSON object listing every command, its engine module, and every flag
with `choices` and `default`. Re-run it any time you are unsure of a flag's
name, type, or allowed values.

---

## 2. Golden rules for agents

1. **Dry-run first.** Every mutating command supports `--dry-run`. Run it,
   read the `summary`, confirm the plan, then re-run without `--dry-run`.
2. **Add `--json` for machine parsing.** It switches the output to one-JSON-
   object-per-line (JSONL) on **stdout**. Human text and third-party prints go
   to **stderr**. Parse stdout only.
3. **Decide by exit code, not by reading prose.** See Â§4.
4. **Backups are on by default** where they apply: `convert --operation
   icc_fix` and in-place `resize`/`upscale` write a `.bak` copy before
   overwriting (opt out with `--no-backup`). `tags --backup` is opt-in.
5. **One command = one job.** Do not chain shell `&&` around a long GPU job if
   you need to observe progress; run it, then inspect the `summary`.

---

## 3. Global flags (work before OR after the subcommand)

| Flag | Meaning |
|---|---|
| `--json` | Emit JSONL on stdout (machine mode). Off by default. |
| `--verbose` | DEBUG-level logging (more `progress`/detail events). |
| `--log-file PATH` | Mirror every event to `PATH` as JSONL (same format as stdout). |
| `--list-commands` | Print the command/flag catalog. Combine with `--json`. |

Example: `cli.py --json --verbose resize <folder> --max-resolution 100`

---

## 4. Exit codes

| Code | Meaning | Agent action |
|---|---|---|
| `0` | Success (may include per-file `errors` in the summary) | Read the `summary`; per-file errors do **not** change the exit code. |
| `1` | Fatal error (bad engine, missing model, exception) | Read the last stderr/log line for the cause. |
| `2` | Usage error (unknown flag, missing arg) | Fix the invocation; see `--list-commands --json`. |
| `130` | Cancelled (Ctrl-C / SIGINT) | The run stopped cleanly at the next item boundary. |

Note: a command that processes 10 files and fails on 2 still exits `0` with
`summary.errors == 2`. Check the summary, not just the code.

---

## 5. JSONL output contract (`--json`)

With `--json`, **stdout** is a stream where every non-empty line is one JSON
object. The shape is:

```json
{ "ts": "<ISO-8601 UTC>", "level": "INFO|DEBUG|ERROR",
  "logger": "DatasetManager", "msg": "<event or message>",
  "data": { ... }          // present only on structured events
}
```

### Event types you will see

| `msg` | `data` | When |
|---|---|---|
| `cli_start` | `command`, `json_mode`, `params` | First line of every run. `params` is the exact flag dict. |
| `progress` | `current`, `total`, `message` | Periodically during a run. |
| `summary` | see below | **Last** line of every processing run. This is what you parse. |
| `config_get` | `key`, `value` | `config --get` in `--json` mode. |
| `config_view` | `config` | `config` (whole file) in `--json` mode. |
| `model_download_ok` | `model`, `dest` | `download-model` success. |
| `command_failed` | (exception in `exc`) | Any unhandled error; level `ERROR`. |

### The `summary` event (the result you care about)

```json
{ "msg": "summary",
  "data": {
    "processed": 12,
    "skipped": 3,
    "errors": 1,
    "stopped": false,
    "items": [
      { "path": "folder\\a.png", "action": "resized",
        "reason": "300x50 -> 100x16",
        "original_size": [300, 50], "new_size": [100, 16] }
    ]
  }
}
```

- `processed` / `skipped` / `errors` are counts.
- `stopped` is `true` only if the run was cancelled (exit 130).
- `items[]` is per-file. `action` is a short verb (`resized`, `dry-run`,
  `fixed`, `added`, `deleted`, `would write caption to ...`, `error`, ...).
  `reason` is a human/machine-readable explanation. Extra keys vary by
  command (sizes, group members, tag lists, `keeper`/`group`, `deleted`/
  `would_delete` counts, etc.).
- On `--dry-run`, `action` values are prefixed `would ...` and **nothing is
  written to disk**.

**Parsing recipe:** read stdout, `json.loads` each line, keep the object whose
`msg == "summary"`, then inspect `data.processed/skipped/errors/items`.

---

## 6. Command reference

`folder` is a required positional argument on most commands. `--ext` (where
present) **adds** extensions to the default image set (`.png .jpg .jpeg
.bmp`); it never replaces it. Pass them with or without a leading dot
(`--ext webp` == `--ext .webp`).

### `dedupe` â€” find duplicate / near-duplicate images
```
cli.py dedupe <folder> [--recursive] [--use-hash] [--use-hist]
                 [--hash-threshold 0.9] [--hist-threshold 0.95]
                 [--delete] [--dry-run] [--ext ...]
```
- `--use-hash` (perceptual hash) and/or `--use-hist` (color histogram). Enable
  at least one. Thresholds are **0â€“1 fractions** (higher = stricter).
- Default is read-only: reports groups, changes nothing.
- `--delete`: sends each group's losers to the **Windows Recycle Bin**
  (restorable, never hard-deleted). Keeper per group = most pixels, ties
  broken by higher mean brightness, then by path. Ungrouped images are never
  touched.
- `--delete --dry-run`: reports `would recycle` moves without moving anything.
- Group items carry `keeper: true/false`; `summary` adds `deleted` /
  `would_delete` counts.

### `resize` â€” downscale images over a max resolution
```
cli.py resize <folder> --max-resolution <px> [--recursive] [--dry-run]
               [--in-place] [--no-backup]
```
- Default: writes results into a `resized/` subfolder **next to each source**
  (never overwrites the original). Re-runs skip their own `resized/` output.
- `--in-place`: overwrites the source in place; a `.bak` copy is written
  first (opt out with `--no-backup`).
- Images already under `--max-resolution` are `skipped`.
- No `--ext` flag (matches the GUI; fixed image set).

### `convert` â€” format conversion and ICC-profile fixing
```
cli.py convert <folder> --operation convert|icc_fix
                 [--target-format png] [--recursive|--no-recursive]
                 [--no-backup] [--dry-run] [--use-parallel] [--ext ...]
```
- `--operation icc_fix`: strips embedded ICC profiles and normalizes
  non-RGB/RGBA to RGB. Creates a `.bak` backup unless `--no-backup`
  (kept for your verification; delete them once you're happy).
- `--operation convert`: re-encodes to `--target-format`
  (`png|jpeg|jpg|bmp|webp`). **Non-destructive**: writes the converted file
  next to the original (same basename, new extension) and leaves the original
  in place. Refuses to overwrite an existing output (per-file `error`).
  JPEG output flattens alpha/palette onto white.
- `convert` is non-recursive by default; `icc_fix` is recursive by default.

### `tags` â€” edit Danbooru-style sidecar tag files
```
cli.py tags <folder> --op dump|add|remove|replace|clear
                 [--tags ...] [--new-tags ...]
                 [--position top|middle|bottom|custom:N]
                 [--caption TEXT] [--recursive] [--dry-run] [--backup] [--ext ...]
```
- Sidecar for `a.png` is `a.txt` (same folder, image suffix replaced by `.txt`).
- `--op dump`: print current tags (read-only).
- `--op add` / `remove` / `replace` / `clear` mutate the sidecars.
  - `add`/`replace` need `--new-tags`; `remove`/`replace` need `--tags`.
  - `--position custom:N` inserts at 0-based index `N`.
- `--backup` renames each existing sidecar to `a.000` (then `.001`, ...) before
  overwriting. Off by default.
- Tags are lowercased and de-duplicated on write.

### `caption` â€” generate captions (WD-tagger or local LLM)
```
cli.py caption <folder> [--mode tagger|natural-language]
        # tagger:
        [--model-name wd-eva02-large-tagger-v3] [--device-id N]
        [--include-rating] [--keep-underscore] [--append-tags]
        [--undesired-tags ...] [--prefix-tags ...]
        [--general-threshold 0.35] [--character-threshold 0.5]
        [--batch-size 1] [--worker-count 2]
        # natural-language:
        [--api-url http://127.0.0.1:8890] [--api-key KEY]
        [--system-prompt-key character|style] [--caption-prefix TEXT]
        [--request-timeout SEC]
        [--skip-existing]
        # both:
        [--recursive] [--dry-run] [--debug] [--ext ...]
```
- Writes one caption per image to the `a.txt` sidecar (same convention as
  `tags`). `--append-tags` appends instead of overwriting.
- `--mode tagger` (default) runs the WD14 ONNX model. It **auto-detects a GPU**
  if one is free; the model is small (~runs in seconds). Model downloads to the
  HuggingFace cache on first use.
- `--mode natural-language` calls a local LLM server (`--api-url`). If the
  server is down you get per-file `503`/connection errors in `summary.errors`
  (exit still `0`). Start your LLM server first.
- `--skip-existing` (natural-language): skips images that are already
  captioned â€” their tag file sits in `Tag Captions/` and a caption `.txt`
  is beside the image. Use it to **resume an interrupted run**: just re-run
  the same command with the flag; finished images are not re-captioned.
  (Images that never had a tag file cannot be detected and are re-captioned.)
- `--retry` (natural-language): keeps the run alive while the LLM server is
  down â€” failed images are retried in passes (`--retry-interval` seconds
  apart, default 30; `--max-passes` caps the passes, 0 = unlimited) until
  the folder is fully captioned. Stop still works during the wait. Combine
  with `--skip-existing` so a killed process costs nothing: just re-run the
  same command.

### `upscale` â€” RealESRGAN or SeedVR2 upscaling  âš ï¸ VRAM
```
cli.py upscale <folder> [--model realesrgan|seedvr2]
                 [--model-path PATH] [--scale-factor 4.0] [--min-size PX]
                 [--device cuda|cpu] [--seed -1]
                 [--color-correction wavelet] [--no-tile-vae]
                 [--recursive] [--dry-run] [--in-place] [--no-backup]
                 [--ext ...]
```
- **Uses the GPU.** Needs a free VRAM-capable device. `--dry-run` is safe and
  needs no model/VRAM (it only computes target sizes).
- `--scale-factor` = fixed scale (default 4.0). `--min-size PX` = auto mode:
  pick the smallest 0.1 step that brings the longest side to â‰¥ PX. Images
  whose longest side is **already â‰¥ PX are `skipped`** (reason
  `already >= min_size`) â€” re-runs do not re-upscale them.
- RealESRGAN weights live in the shared HF cache (repo
  `Kim2091/UltraSharpV2`; DAT2 architecture, auto-detected and loaded via
  spandrel).
  If missing, a real run exits `1` with a clear "download the model first"
  message. Run `download-model` first.
- Default: output goes to an `upscaled/` subfolder next to each source;
  re-runs skip their own `upscaled/` output.
- `--in-place`: overwrites the source in place; a `.bak` copy is written
  first (opt out with `--no-backup`).

### `recycle` â€” send files/folders to the Recycle Bin
```
cli.py recycle <path> [<path> ...] [--dry-run]
```
- Moves each path to the **Windows Recycle Bin** (restorable, never
  hard-deleted). Folders are accepted.
- Missing paths become per-item `error`s (exit code stays `0`).
- `--dry-run` reports `would recycle` without moving anything.
- `summary` adds `deleted` / `would_delete` counts.

### `download-model` â€” fetch model weights  âš ï¸ network
```
cli.py download-model [--model realesrgan|seedvr2] [--dest PATH]
```
- Downloads weights to the shared HF cache (or the `--dest` directory).
  Large (4x-UltraSharpV2 ~140 MB, SeedVR2 several GB). Emits `model_download_ok` on success.

### `config` â€” read/write `config.yaml`
```
cli.py config                      # print whole config
cli.py config --get KEY            # print one value
cli.py config --set KEY VALUE      # set (repeatable); VALUE is JSON-parsed, else string
```
- `--set KEY null` deletes the key. `--set KEY '{"a":1}'` sets a nested object.
- In `--json` mode these emit `config_get` / `config_view` events.

---

## 7. Filesystem conventions (what the app writes)

| Artifact | Rule |
|---|---|
| Caption / tag sidecar | `a.png` â†’ `a.txt` (same dir, suffix â†’ `.txt`). |
| Tag backup | `a.txt` â†’ `a.000`, `a.001`, ... (rename, not copy). |
| ICC-fix backup | `a.png` â†’ `a.png.bak` (deleted after a verified write). |
| In-place backup (resize/upscale) | `a.png` â†’ `a.png.bak` (kept; not created if one exists). |
| Convert output | next to the original, same basename, target extension. |
| Resize output | `<source_dir>/resized/...` (default) or in place with `--in-place`. |
| Upscale output | `<source_dir>/upscaled/...` (default) or in place with `--in-place`. |
| JPEG save | quality 95, subsampling 0. |
| PNG save | compress_level 6. |

The app **never hard-deletes your originals** in normal operation. `dedupe
--delete` and `recycle` move files to the Windows Recycle Bin (restorable).

---

## 8. Recipes

### Safe batch mutation (the default pattern)
```
# 1. Preview
.venv/Scripts/python.exe cli.py --json resize <folder> --max-resolution 1024 --recursive --dry-run
#    -> parse summary: processed/skipped/errors, confirm items look right
# 2. Execute
.venv/Scripts/python.exe cli.py --json resize <folder> --max-resolution 1024 --recursive
#    -> parse summary; errors>0 means inspect items[] for the failing paths
```

### Set up the upscaler, then run
```
.venv/Scripts/python.exe cli.py download-model --model realesrgan
.venv/Scripts/python.exe cli.py --json upscale <folder> --dry-run     # sizes, no VRAM
.venv/Scripts/python.exe cli.py --json upscale <folder>               # real (needs VRAM)
```

### Caption a training set, then tag it
```
.venv/Scripts/python.exe cli.py --json caption <folder> --mode tagger --recursive --dry-run
.venv/Scripts/python.exe cli.py --json caption <folder> --mode tagger --recursive
.venv/Scripts/python.exe cli.py --json tags <folder> --op add --new-tags "1girl" --position top --recursive
```

### Dedupe with recycle-bin deletion
```
.venv/Scripts/python.exe cli.py --json dedupe <folder> --use-hash --delete --dry-run
#    -> group items show keeper:true; losers say "would recycle"
.venv/Scripts/python.exe cli.py --json dedupe <folder> --use-hash --delete
#    -> losers go to the Recycle Bin; summary.deleted counts them
```

### Capture everything for post-hoc analysis
```
.venv/Scripts/python.exe cli.py --json --verbose --log-file run.jsonl <command> ...
# run.jsonl now holds cli_start + every progress + summary, even if stdout is lost
```

---

## 9. Error handling

- **Per-file failures** do not stop the run and do not change the exit code.
  They appear in `summary.errors` and as `items[]` entries with `action:
  "error"` and a `reason`. Continue; report the count.
- **Fatal failures** (exit 1): the last stderr line / `command_failed` event
  carries the exception. Common causes:
  - `RealESRGAN weights not found ...` â†’ run `download-model`.
  - `Engine module ... failed to import` â†’ environment problem (use the venv).
- **Cancellation** (exit 130): `summary.stopped == true`. Safe to re-run;
  already-written outputs are skipped on the next pass.
- **503 / connection errors** from `caption --mode natural-language` â†’ the LLM
  server at `--api-url` is not up. Start it and re-run with `--skip-existing`
  so finished images are not re-captioned.

---

## 10. Caveats

- **VRAM commands:** `upscale` (real run) and `caption --mode tagger` want a
  free GPU. `--dry-run` on `upscale` is VRAM-free. If the GPU is busy, prefer
  dry-runs and let a human do the live GPU pass.
- **Deletion is Recycle-Bin only.** `dedupe --delete` and `recycle` use
  `send2trash`: files are restorable from the Windows Recycle Bin, never
  hard-deleted.
- **GPU auto-detect:** the tagger/upscale pick a CUDA device automatically.
  Force one with `--device-id` (caption) / `--device` (upscale) if needed.
- **No prompts, ever.** If a command seems to wait, it is working (GPU/model
  load), not asking you a question.
- **Config is `config.yaml`** at the repo root (gitignored). `config --set`
  edits it directly.

---

## Quick reference

```
cli.py --list-commands --json          # full flag catalog
cli.py <cmd> --help                    # one command's flags
cli.py --json <cmd> ... --dry-run      # preview any mutation
exit: 0 ok Â· 1 fatal Â· 2 usage Â· 130 cancelled
parse: stdout JSONL -> line where msg=="summary" -> data.{processed,skipped,errors,items}
```