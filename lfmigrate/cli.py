"""Command-line interface.

Design rule: the operator supplies credentials in a file and runs one command.
No flag accepts a secret, so credentials cannot reach ``argv`` (world-readable
via ``ps``), shell history, or a CI command log. Every other input is
discovered, derived, or has a safe default.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import config, pipeline, resources
from . import manifest as M


def _quiet_sdk_warnings(cfg) -> None:
    """Hide the SDK's per-endpoint [BETA] notices.

    ``arize.pre_releases`` emits one warning per process per endpoint. They are
    informational -- nothing is blocked -- but a run that touches projects,
    prompts and datasets prints four of them before any real output, which
    trains an operator to skim past warnings in a tool whose other warnings
    matter. Opt back in with LFMIGRATE_QUIET_SDK_WARNINGS=false.
    """
    if not cfg.bool_("LFMIGRATE_QUIET_SDK_WARNINGS", True):
        return
    logging.getLogger("arize.pre_releases").setLevel(logging.ERROR)

DEFAULT_ENV = ".env"
DEFAULT_ROOT = "out/manifests"


def _load(args):
    """Load config, then silence SDK log noise before any SDK import."""
    cfg = _load_raw(args)
    _quiet_sdk_warnings(cfg)
    return cfg


def _load_raw(args) -> config.Config:
    try:
        return config.load(args.env_file)
    except config.ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)


def _emit(payload, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, default=str))


# ------------------------------------------------------------------ commands


def cmd_preflight(args) -> int:
    cfg = _load(args)
    print(f"Preflight (credentials: {cfg.source_path or 'environment'})\n")
    report = pipeline.preflight(cfg, max_past_years=args.max_past_years)
    for check in report["checks"]:
        mark = "PASS" if check["ok"] else ("FAIL" if check["fatal"] else "WARN")
        print(f"  [{mark}] {check['name']}")
        print(f"         {check['detail']}")
    print()
    if report["ok"]:
        print("Preflight passed. Nothing has been created.")
        print("Run:  python -m lfmigrate run")
    else:
        print("Preflight FAILED. Nothing was written.")
    _emit(report, args.json)
    return 0 if report["ok"] else 1


def cmd_run(args) -> int:
    cfg = _load(args)
    print(f"Migration (credentials: {cfg.source_path or 'environment'})")
    if args.dry_run:
        print("DRY RUN — reads only, writes nothing to AX.\n")
    else:
        print()
    try:
        result = pipeline.run(
            cfg,
            root=Path(args.root),
            batch_size=args.batch_size,
            max_days=args.max_days,
            verify_mode=(pipeline.VERIFY_OFF if args.no_verify
                         else pipeline.VERIFY_EACH_DAY if args.verify_each_day
                         else str(cfg.get("LFMIGRATE_VERIFY_MODE")
                                  or pipeline.VERIFY_END).strip().lower()),
            max_past_years=args.max_past_years,
            dry_run=args.dry_run,
            migrate_resources=not args.no_resources,
            batch_max_bytes=(args.batch_max_mb * 1024 * 1024
                             if args.batch_max_mb
                             else cfg.int_("LFMIGRATE_BATCH_MAX_MB") * 1024 * 1024),
            validate=(not args.no_validate
                      and cfg.bool_("LFMIGRATE_VALIDATE_UPLOAD", True)),
            ingest_path=(args.ingest
                         or str(cfg.get("LFMIGRATE_INGEST_PATH")
                                or "auto").strip().lower()),
            otlp_shards=args.otlp_shards or cfg.int_("LFMIGRATE_OTLP_SHARDS"),
        )
    except pipeline.PipelineError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1
    except M.AmbiguousBatch as exc:
        # Deliberately not retried: the outcome of that batch is unknown and
        # the upload path does not de-duplicate by span ID.
        print(f"\nerror: {exc}", file=sys.stderr)
        print("Run 'python -m lfmigrate verify' to establish what landed.",
              file=sys.stderr)
        return 1

    print()
    if result.get("dry_run"):
        print(f"Dry run complete: {result['days_planned']} day(s) would be migrated, "
              f"{result['days_skipped_old']} skipped as too old.")
        _emit(result, args.json)
        return 0

    verification = result.get("verification")
    if verification is not None and not verification.get("verified"):
        # Non-zero exit so a scripted run still notices, without pretending
        # the migration failed: the spans were accepted and the ledger
        # records them.
        _unverified = True
    else:
        _unverified = False

    if result.get("resources"):
        _print_resource_report(result["resources"])
        print()

    destination = result["destination"]
    print(f"Done. {result['spans_total']} span(s) migrated into "
          f"project {destination['project_name']!r}.")
    skipped = sum(d.get("scores_skipped") or 0 for d in result["days"])
    if skipped:
        print(f"  {skipped} score(s) could not be mapped and were reported, not dropped.")
    if result["days_skipped_old"]:
        print(f"  {result['days_skipped_old']} day(s) skipped as older than the "
              f"AX retention cutoff.")
    _emit(result, args.json)
    problems = []
    if _unverified:
        problems.append("readback did not confirm the spans within the "
                        "timeout; run 'verify' to resolve")
    resources = result.get("resources")
    if resources is not None and not resources.get("ok", True):
        # Previously this ended in "Done." and exit 0 while every resource
        # family had aborted with HTTP 400 -- the spans were fine, so the
        # failure was easy to miss entirely.
        problems.append(f"{len(resources.get('failed') or [])} resource "
                        f"family/item(s) failed; re-run with "
                        f"'python -m lfmigrate resources' after fixing")
    if result.get("score_problems"):
        problems.append("some score frames were not applied; run "
                        "'python -m lfmigrate apply-scores'")
    if problems:
        print("\nexit 2: spans migrated, but:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2
    return 0


def cmd_config(args) -> int:
    """Show the full settings reference, or what this environment resolves to."""
    from . import settings as S

    if args.example:
        # Printed to stdout so it can be redirected over .env.example.
        print(S.render_env_template(), end="")
        return 0

    if args.list:
        print("Every setting is overridable by environment variable or env file.\n")
        group = None
        for item in S.describe():
            prefix = item["key"].split("_")[0]
            if prefix != group:
                group = prefix
                print(f"\n[{group}]")
            flags = []
            if item["required"]:
                flags.append("required")
            if item["secret"]:
                flags.append("secret")
            suffix = f"  ({', '.join(flags)})" if flags else ""
            print(f"  {item['key']}")
            print(f"      default: {item['default']}{suffix}")
            print(f"      {item['help']}")
        return 0

    cfg = _load(args)
    print(f"Resolved configuration (credentials: {cfg.source_path or 'environment'})\n")
    explicit = set(cfg.values)
    for item in S.describe():
        key = item["key"]
        value = cfg.get(key)
        if value is None:
            shown, origin = "<unset>", ""
        elif config.is_secret(key):
            shown, origin = config.redact(str(value)), ""
        else:
            shown = str(value)
            origin = "" if key in explicit else "  (default)"
        marker = "*" if key in explicit else " "
        print(f" {marker} {key:38s} {shown}{origin}")
    print("\n * = set explicitly; everything else is the declared default")
    if cfg.warnings:
        print("\nWarnings:")
        for warning in cfg.warnings:
            print(f"  ! {warning}")
    _emit({"resolved": cfg.safe_summary(), "warnings": cfg.warnings}, args.json)
    return 1 if cfg.warnings else 0


def cmd_apply_scores(args) -> int:
    cfg = _load(args)
    try:
        if getattr(args, "rebuild", False):
            rebuilt = pipeline.rebuild_score_frames(cfg, Path(args.root))
            print(f"Rebuilt score frames for {rebuilt['rebuilt_days']} day(s): "
                  f"{rebuilt['rows']} deferred row(s)")
        result = pipeline.apply_score_plan(cfg, Path(args.root))
    except pipeline.PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if result["problems"]:
        print("\nSome frames still failed:")
        for problem in result["problems"]:
            print(f"  {problem}")
        print("A freshly created project can stay cold for several minutes; "
              "re-run this command.")
        _emit(result, args.json)
        return 1
    print("\nAll deferred score frames applied.")
    _emit(result, args.json)
    return 0


def cmd_status(args) -> int:
    progress = M.migration_progress(Path(args.root))
    print(f"Manifests: {progress['root']}")
    print(f"  shards: {progress['shards_complete']}/{progress['shards_total']} complete")
    print(f"  spans:  {progress['spans_submitted']}/{progress['spans_total']} submitted")
    if progress["shards_uncertain"]:
        print(f"  UNCERTAIN (resolve by readback, do not retry): "
              f"{', '.join(progress['shards_uncertain'])}")
    if progress["shards_with_errors"]:
        print(f"  ERRORS: {', '.join(progress['shards_with_errors'])}")
    for shard in progress["shards"]:
        if "error" in shard:
            print(f"    {shard['shard']}: {shard['error']}")
        else:
            flag = "complete" if shard["complete"] else (
                f"{shard['batches_submitted']}/{shard['batches_total']} batches")
            print(f"    {shard['shard']}: {shard['span_count']} spans, {flag}")
    _emit(progress, args.json)
    return 0


def _print_resource_report(result: dict) -> None:
    for family, data in result.get("families", {}).items():
        label = family.replace("_", " ")
        print(f"\n{label}:")
        for name in data.get("created", []):
            print(f"  + {name}")
        skipped = data.get("skipped") or []
        # Entries that carry their own reason "(...)" or " -- ..." are listed
        # one per line; a bare name means it already existed in AX. The first
        # version labelled everything "already present", which misreported
        # judges skipped for want of an AI integration.
        plain = sorted(s for s in skipped if "(" not in s and " -- " not in s)
        reasoned = [s for s in skipped if s not in plain]
        if plain:
            print(f"  = {len(plain)} already present: "
                  f"{', '.join(plain[:6])}" + (" ..." if len(plain) > 6 else ""))
        for entry in reasoned:
            print(f"  - skipped: {entry}")
        for entry in data.get("exported", []):
            print(f"  > {entry}")
        for entry in data.get("failed", []):
            print(f"  ! {entry}")
        for note in data.get("notes", []):
            print(f"    note: {note}")


def cmd_resources(args) -> int:
    """Prompts, datasets, score configs and evaluators -- space-scoped, not per-day."""
    cfg = _load(args)
    space = cfg.require("ARIZE_SPACE_ID")[0]
    client, _destination, _hosts, _resumed = pipeline.make_destination(cfg)
    source = pipeline.make_source(cfg)
    print(f"Resources (credentials: {cfg.source_path or 'environment'})")
    if args.dry_run:
        print("DRY RUN — reads only, writes nothing to AX.")
    scope = None
    getter = getattr(source, "source_project", None)
    if callable(getter):
        try:
            scope = getter()
        except Exception:
            scope = None
    if scope:
        print(f"  source project: {scope.get('name') or scope.get('id')!r}")
    print(f"  destination space: {space}\n")
    try:
        families = tuple(args.only) if args.only else resources.FAMILIES
        result = resources.migrate_all(
            client, space, source, dry_run=args.dry_run, families=families,
            expand_input=not args.no_expand_input,
            evaluator_options=resources.evaluator_options(cfg))
    finally:
        source.close()
    _print_resource_report(result)
    if result["ok"]:
        print("\nAll resources migrated." if not args.dry_run
              else "\nDry run complete.")
    else:
        print(f"\n{len(result['failed'])} resource(s) failed; "
              f"everything else was migrated.", file=sys.stderr)
    _emit(result, args.json)
    return 0 if result["ok"] else 1


def cmd_verify(args) -> int:
    cfg = _load(args)
    # resume_root makes this reuse the destination the run actually bound, so
    # verify cannot invent a fresh auto-generated name and declare it empty.
    client, destination, _, resumed = pipeline.make_destination(
        cfg, resume_root=Path(args.root))
    if resumed:
        print(f"  using destination recorded in {args.root}: "
              f"{destination['project_name']!r}")
    # Read back over the range the shards actually cover. A backfill writes
    # historical timestamps, so a recent-window default would miss all of it
    # and report a correct migration as an empty project.
    window = M.shard_time_range(Path(args.root))
    since = until = expected = None
    if window:
        since, until = window
        progress = M.migration_progress(Path(args.root))
        expected = progress.get("spans_submitted") or None
        print(f"  reading back from {since.date()} "
              f"(the range these shards cover)")
    print(f"Verifying project {destination['project_name']!r} "
          f"(indexing can take several minutes) ...")
    result = pipeline.verify_project(
        client, destination, timeout=args.timeout,
        since=since, until=until, expected_spans=expected,
        progress=lambda p: print(f"  waiting ({p['detail']})"))
    if result["verified"]:
        expected_note = ""
        if result.get("expected"):
            expected_note = f" of {result['expected']} submitted"
        print(f"\nVerified: {result['rows']} row(s) readable{expected_note}")
        if result.get("expected") and not result.get("matches_expected"):
            # Never let a shortfall read as a clean pass.
            print(f"  SHORTFALL: {result['expected'] - result['rows']} row(s) "
                  f"not yet readable. Indexing may still be catching up -- "
                  f"re-run verify. If it persists, check 'status' for an "
                  f"uncertain batch.")
        print(f"  span kinds:  {result['span_kinds']}")
        print(f"  root spans:  {result['root_spans']}")
        print(f"  eval columns: {result['eval_columns'] or 'none'}")
    else:
        print(f"\nNOT verified after {result['waited_seconds']}s "
              f"({result['detail']}); {result['rows']} row(s) readable.")
        print(f"  {result['hint']}")
    _emit(result, args.json)
    return 0 if result["verified"] else 1


# -------------------------------------------------------------------- parser


def _common_options(parser: argparse.ArgumentParser, *,
                    subcommand: bool = False) -> None:
    """Options accepted before AND after the subcommand.

    The subcommand copies use SUPPRESS so they never overwrite a value given
    before the subcommand. Without it, argparse applied the subcommand's
    default on top: `lfmigrate --root X status` silently used out/manifests,
    so `lfmigrate --root out/project-a run` would have written into whatever
    migration already lived in the default root.

    Unset values are resolved afterwards by _resolve_common: flag, then
    .env / environment, then the declared default.
    """
    default = argparse.SUPPRESS if subcommand else None
    parser.add_argument("--env-file", default=default,
                        help=f"path to the credentials file (default: {DEFAULT_ENV})")
    parser.add_argument("--root", default=default,
                        help="where shards are written (default: LFMIGRATE_SHARD_ROOT, "
                             f"else {DEFAULT_ROOT})")
    parser.add_argument("--json", action="store_true",
                        default=argparse.SUPPRESS if subcommand else False,
                        help="also emit the machine-readable report")
    parser.add_argument("--max-past-years", type=int, default=default,
                        help="AX retention window in years (default: "
                             "LFMIGRATE_MAX_PAST_YEARS, else 2; raise only if Arize "
                             "has widened it for this space)")


def _resolve_common(args) -> None:
    """Fill every unset option from configuration, once, before dispatch.

    Precedence: command-line flag, then the env file / process environment,
    then the declared default. Done here rather than per command so a
    setting such as LFMIGRATE_SHARD_ROOT means the same thing everywhere --
    several declared settings used to be ignored entirely because only the
    flag's hard-coded default was ever consulted.
    """
    if not getattr(args, "env_file", None):
        args.env_file = DEFAULT_ENV
    if not hasattr(args, "json"):
        args.json = False
    path = Path(args.env_file).expanduser()
    try:
        cfg = config.load(path if path.is_file() else None)
    except config.ConfigError:
        # Commands that need credentials report this properly via _load.
        cfg = config.load(None)
    if getattr(args, "root", None) is None:
        args.root = cfg.get("LFMIGRATE_SHARD_ROOT") or DEFAULT_ROOT
    if getattr(args, "max_past_years", None) is None:
        args.max_past_years = cfg.int_("LFMIGRATE_MAX_PAST_YEARS")
    if hasattr(args, "batch_size") and args.batch_size is None:
        args.batch_size = cfg.int_("LFMIGRATE_BATCH_SIZE")
    if hasattr(args, "timeout") and args.timeout is None:
        args.timeout = cfg.int_("LFMIGRATE_VERIFY_TIMEOUT_SECONDS")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lfmigrate",
        description="Migrate Langfuse traces, scores and annotations into Arize AX.",
        epilog="Credentials are read from an env file only; no flag accepts a secret.",
    )
    # Declared on the top-level parser AND on every subcommand, via a shared
    # parent, so both `lfmigrate --root X status` and `lfmigrate status --root X`
    # work. Argparse only accepts global options before the subcommand, and
    # typing them after is the more natural order.
    _common_options(parser)
    common = argparse.ArgumentParser(add_help=False)
    _common_options(common, subcommand=True)

    sub = parser.add_subparsers(dest="command")

    pre = sub.add_parser("preflight", parents=[common],
                         help="read-only checks on both sides; creates nothing")
    pre.set_defaults(func=cmd_preflight)

    run = sub.add_parser("run", parents=[common],
                         help="migrate everything, oldest day first, resumable")
    run.add_argument("--dry-run", action="store_true",
                     help="preflight and plan only; writes nothing to AX")
    run.add_argument("--max-days", type=int, default=None,
                     help="stop after this many days (use 1 for a trial)")
    run.add_argument("--batch-size", type=int, default=None,
                     help=f"spans per upload batch (default: {M.DEFAULT_BATCH_SIZE})")
    run.add_argument("--no-verify", action="store_true",
                     help="skip readback verification entirely")
    run.add_argument("--verify-each-day", action="store_true",
                     help="block on indexing after every day (~5 min each); "
                          "the default verifies all days in one readback at "
                          "the end, which is equivalent and far faster")
    run.add_argument("--no-resources", action="store_true",
                     help="skip prompts/datasets/score configs/evaluators (spans only)")
    run.add_argument("--ingest", choices=("auto", "otlp", "arrow"),
                     default=None,
                     help="ingest path (default: auto -- arrow when history is "
                          "older than 31 days or over ~830k spans, else otlp "
                          "for second-level visibility)")
    run.add_argument("--otlp-shards", type=int, default=None,
                     help="concurrent OTLP exporters (default: 8)")
    run.add_argument("--batch-max-mb", type=int, default=None,
                     help="byte budget per batch (default: 32)")
    run.add_argument("--no-validate", action="store_true",
                     help="skip SDK-side dataframe validation; ~31%% faster, "
                          "only after a successful trial run")
    run.set_defaults(func=cmd_run)

    res = sub.add_parser("resources", parents=[common],
                         help="migrate prompts, datasets, score configs and evaluators")
    res.add_argument("--dry-run", action="store_true",
                     help="report what would be created; writes nothing")
    res.add_argument("--only", action="append", choices=resources.FAMILIES,
                     help="limit to one family (repeatable)")
    res.add_argument("--no-expand-input", action="store_true",
                     help="keep a dataset item's input as one JSON column "
                          "instead of promoting its keys to columns")
    res.set_defaults(func=cmd_resources)

    conf = sub.add_parser("config", parents=[common],
                          help="show resolved settings, or --list the full reference")
    conf.add_argument("--list", action="store_true",
                      help="list every setting with its default and purpose")
    conf.add_argument("--example", action="store_true",
                      help="print a complete .env template covering every setting")
    conf.set_defaults(func=cmd_config)

    apply_scores = sub.add_parser(
        "apply-scores", parents=[common],
        help="apply annotation/session-eval frames saved by an earlier run")
    apply_scores.add_argument(
        "--rebuild", action="store_true",
        help="first recreate missing per-day score frames from the source "
             "(for migrations whose frames were lost); touches no shard")
    apply_scores.set_defaults(func=cmd_apply_scores)

    status = sub.add_parser("status", parents=[common],
                            help="progress across all shards")
    status.set_defaults(func=cmd_status)

    verify = sub.add_parser("verify", parents=[common],
                            help="read back from AX and report what is there")
    verify.add_argument("--timeout", type=int, default=None,
                        help=f"seconds to wait for indexing "
                             f"(default: {pipeline.VERIFY_TIMEOUT_SECONDS})")
    verify.set_defaults(func=cmd_verify)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    _resolve_common(args)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted. Re-run the same command to resume; completed shards "
              "are skipped.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
