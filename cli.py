"""DatasetManager CLI â€” minimal, agent-friendly entry point.

Every GUI feature is drivable here without Qt: each subcommand builds a
params dict and calls its feature engine's run() (see engine contract:
progress_cb/stop_check callbacks, summary dict return).

Machine mode: --json emits JSONL events (startup config, per-file results
via the engines' logging, final summary) for AI-agent consumption.

Exit codes: 0 success, 1 error, 2 usage, 130 cancelled.
"""
import argparse
import importlib
import json
import os
import signal
import sys

from modules.logger import setup_logger, add_log_file_handler, log_event
from modules.utils import IMAGE_EXTENSIONS

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_CANCELLED = 130

# command name -> {"module": engine module path,
#                  "flags": add_flags(parser), "run": run(args, engine, cli)}
COMMANDS = {}


def register(name, module, flags, run):
    COMMANDS[name] = {"module": module, "flags": flags, "run": run}


def _ext_set(extra):
    """Default image extensions plus any user extras (dotted, deduped).

    --ext is additive: it extends IMAGE_EXTENSIONS, never replaces it.
    """
    exts = list(IMAGE_EXTENSIONS)
    for e in extra or ():
        e = e.strip().lower()
        if not e:
            continue
        if not e.startswith('.'):
            e = '.' + e
        if e not in exts:
            exts.append(e)
    return tuple(exts)


class CancelToken:
    """Ctrl+C / SIGINT latch handed to engines as their stop_check."""

    def __init__(self):
        self.cancelled = False

    def __call__(self):
        return self.cancelled


def _progress_printer(logger, json_mode):
    """progress_cb(current, total, message) -> CLI-side presentation."""
    state = {"last": -1}

    def progress(current, total, message=""):
        if json_mode:
            # Coarse events only; the engines' own logging carries detail.
            if current == state["last"] or (current % 100 != 0 and current != total):
                return
            state["last"] = current
            log_event(logger, "progress", current=current, total=total,
                      **({"message": message} if message else {}))
        else:
            if message:
                sys.stdout.write(f"\r{current}/{total} {message}\n")
                sys.stdout.flush()
                state["last"] = -1
            else:
                from modules.utils import print_progress_bar
                print_progress_bar(current, total, prefix="Progress:")

    return progress


class Cli:
    """Presentation helpers + stop latch handed to command runners."""

    def __init__(self, logger, json_mode, stop_check):
        self.logger = logger
        self.json_mode = json_mode
        self.stop_check = stop_check
        self.progress_cb = _progress_printer(logger, json_mode)

    def summary(self, summary, exit_code=EXIT_OK):
        if self.json_mode:
            log_event(self.logger, "summary",
                      **(summary if isinstance(summary, dict) else {"result": summary}))
        else:
            if isinstance(summary, dict):
                line = (f"Processed: {summary.get('processed', 0)} | "
                        f"Skipped: {summary.get('skipped', 0)} | "
                        f"Errors: {summary.get('errors', 0)}")
                if summary.get("groups") is not None:
                    line += f" | Groups: {len(summary['groups'])}"
                if summary.get("stopped"):
                    line += " (stopped)"
                sys.stdout.write("\n" + line + "\n")
                for item in summary.get("items", []):
                    if item.get("action") in ("error", "skipped", "dry-run",
                                              "deleted"):
                        sys.stdout.write(f"  [{item['action']}] {item.get('path')}"
                                         f" - {item.get('reason')}\n")
                    elif item.get("tags") is not None:
                        # tag-editor dump rows: show the tags in human mode.
                        sys.stdout.write(f"  [dump] {item.get('path')}"
                                         f" - {', '.join(item['tags']) or '(empty)'}\n")
                for group in summary.get("groups", []):
                    sys.stdout.write("  [group] " + " | ".join(group.get("images", []))
                                     + f" ({group.get('method')}, sim {group.get('similarity')})\n")
            else:
                sys.stdout.write(str(summary) + "\n")
        return exit_code


def _globals():
    """Global flags on a shared parent so they parse before OR after the
    subcommand name (agents put flags anywhere)."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--json", action="store_true",
                   help="machine mode: JSONL log output (one JSON object per line)")
    p.add_argument("--verbose", action="store_true",
                   help="DEBUG level logging")
    p.add_argument("--log-file", metavar="PATH",
                   help="also write all log records to this file")
    p.add_argument("--list-commands", action="store_true",
                   help="print all commands and flags as JSON and exit")
    return p


GLOBALS = _globals()


def print_introspection():
    """--list-commands --json: commands and their flags, for agent parsing."""
    out = []
    for name in sorted(COMMANDS):
        entry = COMMANDS[name]
        probe = argparse.ArgumentParser(prog=f"cli.py {name}")
        entry["flags"](probe)
        out.append({
            "command": name,
            "engine": entry["module"],
            "flags": [
                {"flags": a.option_strings, "dest": a.dest,
                 "choices": list(a.choices) if a.choices else None,
                 "default": a.default, "help": a.help}
                for a in probe._actions if a.option_strings
            ],
        })
    print(json.dumps({"commands": out}, ensure_ascii=False, indent=2))
def build_parser():
    parser = argparse.ArgumentParser(
        prog="cli.py", parents=[GLOBALS],
        description="DatasetManager command-line interface (non-interactive; "
                    "all options are flags).")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    for name in sorted(COMMANDS):
        sp = sub.add_parser(name, parents=[GLOBALS])
        COMMANDS[name]["flags"](sp)
    return parser

def _extract_globals(argv):
    """Pull the global flags out of argv so they work before OR after the
    subcommand. argparse parent-parsers at both the main and sub level would
    otherwise let the subparser's default clobber a value set before the
    subcommand name. Returns (found_dict, remaining_argv)."""
    found = {"json": False, "verbose": False, "log_file": None,
             "list_commands": False}
    rest = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--json":
            found["json"] = True
        elif a == "--verbose":
            found["verbose"] = True
        elif a == "--log-file":
            i += 1
            found["log_file"] = argv[i] if i < len(argv) else None
        elif a == "--list-commands":
            found["list_commands"] = True
        else:
            rest.append(a)
        i += 1
    return found, rest


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = build_parser()
    g, rest = _extract_globals(argv)
    args = parser.parse_args(rest)
    # Apply globals found anywhere on the command line (they were stripped
    # from `rest` above so the subparser defaults can't clobber them).
    if g["json"]:
        args.json = True
    if g["verbose"]:
        args.verbose = True
    if g["log_file"] is not None:
        args.log_file = g["log_file"]
    if g["list_commands"]:
        args.list_commands = True
    if args.list_commands:
        print_introspection()
        return EXIT_OK
    if not args.command:
        parser.print_help()
        return EXIT_USAGE

    entry = COMMANDS[args.command]
    # Import the engine first: engine modules call setup_logger() at import
    # time, which would otherwise clobber the CLI-side formatter.
    try:
        engine = importlib.import_module(entry["module"])
        import_error = None
    except ImportError as e:
        engine = None
        import_error = e

    logger = setup_logger(debug_mode=args.verbose, json_mode=args.json)
    if args.log_file:
        add_log_file_handler(logger, args.log_file)
    if import_error:
        logger.error("Engine module %s failed to import: %s", entry["module"], import_error)
        return EXIT_ERROR

    token = CancelToken()
    signal.signal(signal.SIGINT, lambda *_: setattr(token, "cancelled", True))
    cli = Cli(logger, args.json, token)

    log_event(logger, "cli_start", command=args.command, json_mode=args.json,
              params={k: v for k, v in vars(args).items()
                      if k not in ("json", "verbose", "log_file", "list_commands",
                                   "command")})
    real_stdout = sys.stdout
    try:
        if args.json:
            # Third-party/engine prints (e.g. onnx model loading) go to
            # stderr so the JSONL stream stays pure.
            sys.stdout = sys.stderr
        exit_code = entry["run"](args, engine, cli)
    except KeyboardInterrupt:
        token.cancelled = True
        exit_code = EXIT_CANCELLED
    except Exception:
        logger.exception("command_failed")
        exit_code = EXIT_ERROR
    finally:
        sys.stdout = real_stdout

    if token.cancelled:
        exit_code = EXIT_CANCELLED
    return exit_code


# --------------------------------------------------------------------------
# command definitions (one flags builder + one runner per subcommand)
# --------------------------------------------------------------------------

def _flags_resizer(p):
    p.add_argument("folder")
    p.add_argument("--max-resolution", type=int, required=True,
                   help="images with either side over this are resized "
                        "proportionally into a 'resized' subfolder")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--in-place", action="store_true",
                   help="overwrite the originals instead of writing into "
                        "a 'resized' subfolder")
    p.add_argument("--no-backup", action="store_true",
                   help="with --in-place: skip the .bak copy of each "
                        "overwritten original")


def _run_resizer(args, engine, cli):
    params = {"folder": args.folder, "max_resolution": args.max_resolution,
              "recursive": args.recursive, "dry_run": args.dry_run,
              "in_place": args.in_place, "backup": not args.no_backup}
    return cli.summary(engine.run(params, cli.progress_cb, cli.stop_check))


register("resize", "modules.image_resizer_engine", _flags_resizer, _run_resizer)


def _flags_dedupe(p):
    p.add_argument("folder")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--use-hash", action="store_true",
                   help="use perceptual-hash similarity (avg_hash, 64-bit)")
    p.add_argument("--use-hist", action="store_true",
                   help="use color-histogram similarity")
    p.add_argument("--hash-threshold", type=float, default=0.9)
    p.add_argument("--hist-threshold", type=float, default=0.95)
    p.add_argument("--delete", action="store_true",
                   help="recycle the losers of every duplicate group "
                        "(keeper: most pixels, then brightest, then path); "
                        "without this flag the run is report-only")
    p.add_argument("--dry-run", action="store_true",
                   help="with --delete: report what would be recycled, "
                        "move nothing")
    p.add_argument("--ext", nargs="*", default=None,
                   help="extra file extensions to scan")


def _run_dedupe(args, engine, cli):
    params = {"folder": args.folder, "recursive": args.recursive,
              "use_hash": args.use_hash, "use_hist": args.use_hist,
              "hash_threshold": args.hash_threshold,
              "hist_threshold": args.hist_threshold,
              "delete": args.delete, "dry_run": args.dry_run}
    if args.ext:
        params["extensions"] = _ext_set(args.ext)
    return cli.summary(engine.run(params, cli.progress_cb, cli.stop_check))


register("dedupe", "modules.duplicate_detector.engine", _flags_dedupe, _run_dedupe)


def _flags_convert(p):
    p.add_argument("folder")
    p.add_argument("--operation", required=True, choices=["convert", "icc_fix"])
    p.add_argument("--target-format", default=None,
                   help="conversion target (png|jpeg|jpg|bmp|webp); required for convert")
    p.add_argument("--recursive", action="store_true",
                   help="scan subfolders (icc_fix defaults to recursive anyway)")
    p.add_argument("--no-recursive", action="store_true",
                   help="top-level only (icc_fix; overrides its recursive default)")
    p.add_argument("--no-backup", action="store_true",
                   help="skip .bak backups (icc_fix)")
    p.add_argument("--dry-run", action="store_true", help="icc_fix only")
    p.add_argument("--use-parallel", action="store_true", help="convert only")
    p.add_argument("--ext", nargs="*", default=None,
                   help="extra file extensions to scan")


def _run_convert(args, engine, cli):
    if args.operation == "convert" and not args.target_format:
        cli.logger.error("--target-format is required for operation=convert")
        return EXIT_USAGE
    params = {"operation": args.operation, "folder_path": args.folder,
              "use_parallel": args.use_parallel, "dry_run": args.dry_run,
              "backup": not args.no_backup}
    if args.operation == "convert":
        params["target_format"] = args.target_format
        params["recursive"] = args.recursive
    else:
        params["recursive"] = not args.no_recursive
    if args.ext:
        params["extensions"] = _ext_set(args.ext)
    return cli.summary(engine.run(params, cli.progress_cb, cli.stop_check))


register("convert", "modules.Conversion_Tools.engine", _flags_convert, _run_convert)


def _flags_tags(p):
    p.add_argument("folder")
    p.add_argument("--op", required=True,
                   choices=["dump", "add", "remove", "replace", "clear"],
                   help="dump reads sidecars; the others write them")
    p.add_argument("--tags", nargs="*", default=None,
                   help="tags to remove/replace (lower-cased like the GUI)")
    p.add_argument("--new-tags", nargs="*", default=None,
                   help="tags to insert for add/replace")
    p.add_argument("--position", nargs="*", default=None,
                   help="insertion positions: top|middle|bottom|custom:N (1-based)")
    p.add_argument("--caption", default=None,
                   help="caption text for clear (default empty)")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--backup", action="store_true",
                   help="rename old sidecars to .000 etc. before overwriting")
    p.add_argument("--ext", nargs="*", default=None,
                   help="extra file extensions to scan")


def _run_tags(args, engine, cli):
    params = {"folder": args.folder, "op": args.op, "recursive": args.recursive,
              "dry_run": args.dry_run, "backup": args.backup}
    if args.tags:
        params["tags"] = args.tags
    if args.new_tags:
        params["new_tags"] = args.new_tags
    if args.position:
        positions = []
        for pos in args.position:
            if pos.lower().startswith("custom:"):
                positions.append(["custom", int(pos.split(":", 1)[1])])
            else:
                positions.append(pos.lower())
        params["positions"] = positions
    if args.caption is not None:
        params["caption"] = args.caption
    if args.ext:
        params["extensions"] = _ext_set(args.ext)
    return cli.summary(engine.run(params, cli.progress_cb, cli.stop_check))


register("tags", "modules.tag_editor.engine", _flags_tags, _run_tags)


def _flags_config(p):
    p.add_argument("--get", metavar="KEY", nargs="?",
                   help="print one config.yaml value (or the whole config)")
    p.add_argument("--set", nargs=2, action="append", metavar=("KEY", "VALUE"),
                   help="set a config.yaml value (JSON-parsed, else string); "
                        "repeatable")


def _run_config(args, engine, cli):
    if args.get is not None:
        value = engine.get_value(args.get)
        if cli.json_mode:
            log_event(cli.logger, "config_get", key=args.get, value=value)
        else:
            print(json.dumps(value, ensure_ascii=False, indent=2))
        return EXIT_OK
    if args.set:
        for key, raw in args.set:
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                value = raw
            engine.set_value(key, value)
        if not args.json:
            print("config updated")
        return EXIT_OK
    if cli.json_mode:
        log_event(cli.logger, "config_view", config=engine.load_config())
    else:
        print(json.dumps(engine.load_config(), ensure_ascii=False, indent=2))
    return EXIT_OK


register("config", "modules.config", _flags_config, _run_config)


def _flags_caption(p):
    p.add_argument("folder")
    p.add_argument("--mode", choices=["tagger", "natural-language"],
                   default="tagger")
    p.add_argument("--model-name", default=None,
                   help="tagger model (wd-eva02-large-tagger-v3 | "
                        "wd-swinv2-tagger-v3 | wd-convnext-tagger-v3); "
                        "default wd-eva02-large-tagger-v3")
    p.add_argument("--device-id", type=int, default=None,
                   help="CUDA device index (tagger)")
    p.add_argument("--include-rating", action="store_true",
                   help="include rating tags (tagger)")
    p.add_argument("--keep-underscore", action="store_true",
                   help="do not strip underscores from tags (tagger)")
    p.add_argument("--append-tags", action="store_true",
                   help="append to existing sidecar tags (tagger)")
    p.add_argument("--undesired-tags", nargs="*", default=None,
                   help="tags to drop (tagger)")
    p.add_argument("--prefix-tags", nargs="*", default=None,
                   help="tags prepended to every caption (tagger)")
    p.add_argument("--general-threshold", type=float, default=None,
                   help="general tag threshold 0-1 (tagger)")
    p.add_argument("--character-threshold", type=float, default=None,
                   help="character tag threshold 0-1 (tagger)")
    p.add_argument("--batch-size", type=int, default=None,
                   help="tagger batch size (default 1)")
    p.add_argument("--worker-count", type=int, default=None,
                   help="prep worker count (default 2)")
    p.add_argument("--api-url", default=None,
                   help="local LLM URL (natural-language; default "
                        "http://127.0.0.1:8890)")
    p.add_argument("--api-key", default=None,
                   help="Bearer key for the LLM API (natural-language; "
                        "falls back to config.yaml llm_api_key)")
    p.add_argument("--system-prompt-key", choices=["character", "style"],
                   default=None,
                   help="which config.yaml system prompt to send "
                        "(natural-language; default character)")
    p.add_argument("--caption-prefix", default=None,
                   help="text prepended to each caption (natural-language)")
    p.add_argument("--request-timeout", type=int, default=None,
                   help="LLM request timeout seconds (natural-language)")
    p.add_argument("--skip-existing", action="store_true",
                   help="natural-language: skip images that already have "
                        "a caption (tag file in 'Tag Captions' plus a "
                        "caption .txt beside the image); use to resume an "
                        "interrupted run")
    p.add_argument("--retry", action="store_true",
                   help="natural-language: keep the run alive while the LLM "
                        "server is down - failed images are retried in "
                        "passes until the folder is fully captioned "
                        "(combine with --skip-existing)")
    p.add_argument("--retry-interval", type=int, default=None,
                   help="seconds between retry passes (natural-language "
                        "--retry; default 30)")
    p.add_argument("--max-passes", type=int, default=None,
                   help="max retry passes before giving up, 0 = unlimited "
                        "(natural-language --retry; default 0)")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--debug", action="store_true",
                   help="DEBUG logging inside the captioner")
    p.add_argument("--ext", nargs="*", default=None,
                   help="extra file extensions to scan")


def _run_caption(args, engine, cli):
    params = {"folder": args.folder,
              "mode": "natural_language" if args.mode == "natural-language"
              else "tagger",
              "recursive": args.recursive, "dry_run": args.dry_run,
              "debug_mode": args.debug}
    if args.model_name:
        params["model_name"] = args.model_name
    if args.device_id is not None:
        params["device_id"] = args.device_id
    if args.include_rating:
        params["include_rating"] = True
    if args.keep_underscore:
        params["remove_underscore"] = False
    if args.append_tags:
        params["append_tags"] = True
    if args.undesired_tags:
        params["undesired_tags"] = args.undesired_tags
    if args.prefix_tags:
        params["prefix_tags"] = args.prefix_tags
    if args.general_threshold is not None:
        params["general_threshold"] = args.general_threshold
    if args.character_threshold is not None:
        params["character_threshold"] = args.character_threshold
    if args.batch_size is not None:
        params["batch_size"] = args.batch_size
    if args.worker_count is not None:
        params["worker_count"] = args.worker_count
    if args.api_url:
        params["api_url"] = args.api_url
    if args.api_key:
        params["api_key"] = args.api_key
    if args.system_prompt_key:
        params["system_prompt_key"] = args.system_prompt_key
    if args.caption_prefix is not None:
        params["caption_prefix"] = args.caption_prefix
    if args.skip_existing:
        params["skip_existing"] = True
    if args.retry:
        params["retry"] = True
    if args.retry_interval is not None:
        params["retry_interval"] = args.retry_interval
    if args.max_passes is not None:
        params["max_passes"] = args.max_passes
    if args.request_timeout is not None:
        params["request_timeout"] = args.request_timeout
    if args.ext:
        params["extensions"] = _ext_set(args.ext)
    return cli.summary(engine.run(params, cli.progress_cb, cli.stop_check))


register("caption", "modules.caption_generator.engine", _flags_caption,
         _run_caption)


def _flags_upscale(p):
    p.add_argument("folder")
    p.add_argument("--model", choices=["realesrgan", "seedvr2"],
                   default="realesrgan")
    p.add_argument("--model-path", default=None,
                   help="RealESRGAN weights path (default <app>/models/"
                        "RealESRGAN_x4plus_anime_6B.pth)")
    p.add_argument("--scale-factor", type=float, default=4.0,
                   help="fixed scale (default 4.0, the GUI spin default)")
    p.add_argument("--min-size", type=int, default=0,
                   help="auto mode: smallest 0.1 step bringing the longest "
                        "side to at least this (GUI resolution spin)")
    p.add_argument("--device", default=None,
                   help="cuda|cpu (default: auto-detect)")
    p.add_argument("--seed", type=int, default=-1,
                   help="SeedVR2 seed; -1 = random per image (GUI default)")
    p.add_argument("--color-correction", default="wavelet",
                   help="SeedVR2 color correction (default wavelet)")
    p.add_argument("--no-tile-vae", action="store_true",
                   help="disable tiled VAE encode/decode (SeedVR2)")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--in-place", action="store_true",
                   help="overwrite the originals instead of writing into "
                        "an 'upscaled' subfolder")
    p.add_argument("--no-backup", action="store_true",
                   help="with --in-place: skip the .bak copy of each "
                        "overwritten original")
    p.add_argument("--ext", nargs="*", default=None,
                   help="extra file extensions to scan")


def _run_upscale(args, engine, cli):
    params = {"folder": args.folder, "model": args.model,
              "recursive": args.recursive, "dry_run": args.dry_run,
              "scale_factor": args.scale_factor, "min_size": args.min_size,
              "seed": args.seed,
              "color_correction": args.color_correction,
              "tile_vae": not args.no_tile_vae,
              "in_place": args.in_place, "backup": not args.no_backup}
    if args.model_path:
        params["model_path"] = args.model_path
    if args.device:
        params["device"] = args.device
    if args.ext:
        params["extensions"] = _ext_set(args.ext)
    return cli.summary(engine.run(params, cli.progress_cb, cli.stop_check))


register("upscale", "modules.Upscaler.engine", _flags_upscale, _run_upscale)


def _flags_download_model(p):
    p.add_argument("--model", choices=["realesrgan", "seedvr2"],
                   default="realesrgan")
    p.add_argument("--dest", default=None,
                   help="RealESRGAN weights destination path")


def _run_download_model(args, engine, cli):
    ok = engine.download_model(args.model, args.dest,
                               status_cb=lambda m: sys.stdout.write(m + "\n"),
                               stop_check=cli.stop_check)
    if not ok:
        cli.logger.error("model_download_failed")
        return EXIT_ERROR
    log_event(cli.logger, "model_download_ok", model=args.model,
              dest=args.dest)
    return EXIT_OK


register("download-model", "modules.Upscaler.engine", _flags_download_model,
         _run_download_model)


def _flags_recycle(p):
    p.add_argument("paths", nargs="+",
                   help="files or folders to send to the recycle bin")
    p.add_argument("--dry-run", action="store_true",
                   help="report only, move nothing")


def _run_recycle(args, engine, cli):
    items = []
    deleted = 0
    would = 0
    errors = 0
    for path in args.paths:
        if cli.stop_check():
            break
        if not os.path.exists(path):
            errors += 1
            items.append({"path": path, "action": "error", "reason": "not found"})
            continue
        if args.dry_run:
            would += 1
            items.append({"path": path, "action": "dry-run", "reason": "would recycle"})
            continue
        try:
            engine.send2trash(os.path.normpath(path))
            deleted += 1
            items.append({"path": path, "action": "deleted", "reason": None})
        except Exception as e:
            errors += 1
            items.append({"path": path, "action": "error", "reason": str(e)})
    return cli.summary({"processed": deleted + would, "skipped": 0,
                        "errors": errors, "items": items, "stopped": False,
                        "deleted": deleted, "would_delete": would})


register("recycle", "send2trash", _flags_recycle, _run_recycle)


if __name__ == "__main__":
    sys.exit(main())
