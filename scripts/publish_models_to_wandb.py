#!/usr/bin/env python3
"""Publish model files from ./models to Weights & Biases artifacts."""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


MODEL_EXTENSIONS = {
    ".bin",
    ".ckpt",
    ".h5",
    ".hdf5",
    ".joblib",
    ".onnx",
    ".pkl",
    ".pickle",
    ".pt",
    ".pth",
    ".safetensors",
}

ARTIFACT_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish model files from the models directory to W&B artifacts.",
    )
    parser.add_argument(
        "models",
        nargs="*",
        help="Model files to publish. Paths can be absolute or relative to --models-dir.",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory with model files. Defaults to ./models.",
    )
    parser.add_argument(
        "--project",
        default=os.getenv("WANDB_PROJECT"),
        help="W&B project. Defaults to WANDB_PROJECT, or asks interactively.",
    )
    parser.add_argument(
        "--entity",
        default=os.getenv("WANDB_ENTITY"),
        help="Optional W&B entity/team. Defaults to WANDB_ENTITY.",
    )
    parser.add_argument(
        "--artifact-type",
        default=os.getenv("WANDB_ARTIFACT_TYPE", "model"),
        help="W&B artifact type. Defaults to 'model'.",
    )
    parser.add_argument(
        "--artifact-prefix",
        default=os.getenv("WANDB_ARTIFACT_PREFIX", ""),
        help="Prefix added to generated artifact names.",
    )
    parser.add_argument(
        "--job-type",
        default="publish-models",
        help="W&B run job_type. Defaults to 'publish-models'.",
    )
    parser.add_argument(
        "--alias",
        action="append",
        default=[],
        help="Artifact alias to add. Can be passed multiple times.",
    )
    parser.add_argument(
        "--no-latest",
        action="store_true",
        help="Do not add the default 'latest' alias.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Publish all discovered model files without opening the picker.",
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Include every file in --models-dir, not only common model extensions.",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        help="Glob pattern for files under --models-dir, for example '*.pt'.",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip interactive confirmation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be published without calling W&B.",
    )
    return parser.parse_args()


def human_size(size_bytes: int) -> str:
    value = float(size_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or unit == "GiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} {unit}"
        value /= 1024.0
    return f"{size_bytes} B"


def is_model_file(path: Path, include_all_files: bool) -> bool:
    if not path.is_file():
        return False
    if path.name.startswith(".~lock.") or path.name.startswith(".DS_Store"):
        return False
    if include_all_files:
        return True
    return path.suffix.lower() in MODEL_EXTENSIONS


def matches_patterns(path: Path, models_dir: Path, patterns: list[str]) -> bool:
    if not patterns:
        return True

    rel = path.relative_to(models_dir).as_posix()
    return any(fnmatch.fnmatch(path.name, pattern) or fnmatch.fnmatch(rel, pattern) for pattern in patterns)


def discover_files(models_dir: Path, include_all_files: bool, patterns: list[str]) -> list[Path]:
    files = [
        path
        for path in models_dir.rglob("*")
        if is_model_file(path, include_all_files) and matches_patterns(path, models_dir, patterns)
    ]
    return sorted(files, key=lambda path: path.relative_to(models_dir).as_posix().lower())


def resolve_model_path(raw_path: str, models_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    candidates = [path] if path.is_absolute() else [models_dir / path, Path.cwd() / path]

    for candidate in candidates:
        resolved = candidate.resolve()
        if not resolved.exists():
            continue
        if not resolved.is_file():
            raise ValueError(f"Not a file: {resolved}")
        try:
            resolved.relative_to(models_dir)
        except ValueError as exc:
            raise ValueError(f"File is outside models dir: {resolved}") from exc
        return resolved

    raise ValueError(f"Model file not found: {raw_path}")


def dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    deduped: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def print_model_files(files: list[Path], models_dir: Path) -> None:
    print("Available model files:")
    for index, path in enumerate(files, start=1):
        rel_path = path.relative_to(models_dir).as_posix()
        print(f"{index:3d}) {rel_path} ({human_size(path.stat().st_size)})")


def pick_files_with_prompt(files: list[Path], models_dir: Path) -> list[Path]:
    if not files:
        return []

    print()
    print_model_files(files, models_dir)

    while True:
        choice = input("Choose model number, 'all', or 0 to cancel: ").strip().lower()
        if choice in {"0", "q", "quit", "exit"}:
            return []
        if choice == "all":
            return files
        if not choice.isdigit():
            print("Invalid choice: enter a number from the list.", file=sys.stderr)
            continue

        number = int(choice)
        if 1 <= number <= len(files):
            return [files[number - 1]]

        print(f"Invalid choice: {number} is out of range.", file=sys.stderr)


def truncate(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return f"{text[: width - 3]}..."


def pick_files_with_curses(files: list[Path], models_dir: Path) -> list[Path]:
    import curses

    def draw_line(stdscr: object, y: int, x: int, text: str, width: int, attr: int = 0) -> None:
        try:
            stdscr.addnstr(y, x, text, max(0, width - x - 1), attr)
        except curses.error:
            pass

    def run(stdscr: object) -> list[Path]:
        try:
            curses.curs_set(0)
        except curses.error:
            pass

        stdscr.keypad(True)

        cursor = 0
        offset = 0
        selected: set[int] = set()
        message = ""
        total_size = sum(path.stat().st_size for path in files)

        while True:
            height, width = stdscr.getmaxyx()
            stdscr.erase()

            draw_line(stdscr, 0, 0, "W&B model publishing", width, curses.A_BOLD)
            draw_line(stdscr, 1, 0, "Up/Down: move  Space: toggle  Enter: publish  q: cancel", width)
            draw_line(stdscr, 2, 0, "a: select all  n: clear selection", width)
            draw_line(stdscr, 3, 0, f"Selected: {len(selected)}/{len(files)}", width)
            if message:
                draw_line(stdscr, 4, 0, message, width, curses.A_BOLD)

            list_top = 5
            list_height = max(1, height - list_top - 1)
            rows_count = len(files) + 1

            if cursor < offset:
                offset = cursor
            if cursor >= offset + list_height:
                offset = cursor - list_height + 1

            for screen_row, row_index in enumerate(range(offset, min(rows_count, offset + list_height))):
                y = list_top + screen_row
                is_cursor = row_index == cursor
                prefix = ">" if is_cursor else " "

                if row_index == 0:
                    is_checked = len(selected) == len(files)
                    checkbox = "[*]" if is_checked else "[ ]"
                    label = f"{prefix} {checkbox} ALL MODELS ({len(files)} files, {human_size(total_size)})"
                else:
                    file_index = row_index - 1
                    path = files[file_index]
                    checkbox = "[*]" if file_index in selected else "[ ]"
                    rel_path = path.relative_to(models_dir).as_posix()
                    label = f"{prefix} {checkbox} {rel_path} ({human_size(path.stat().st_size)})"

                attr = curses.A_REVERSE if is_cursor else 0
                draw_line(stdscr, y, 0, truncate(label, width - 1), width, attr)

            if offset > 0:
                draw_line(stdscr, list_top, max(0, width - 5), "more", width)
            if offset + list_height < rows_count:
                draw_line(stdscr, height - 1, max(0, width - 5), "more", width)

            stdscr.refresh()
            key = stdscr.getch()
            message = ""

            if key in (ord("q"), 27):
                return []
            if key in (curses.KEY_UP, ord("k")):
                cursor = max(0, cursor - 1)
                continue
            if key in (curses.KEY_DOWN, ord("j")):
                cursor = min(rows_count - 1, cursor + 1)
                continue
            if key == curses.KEY_HOME:
                cursor = 0
                continue
            if key == curses.KEY_END:
                cursor = rows_count - 1
                continue
            if key == ord("a"):
                selected = set(range(len(files)))
                continue
            if key == ord("n"):
                selected.clear()
                continue
            if key in (ord(" "),):
                if cursor == 0:
                    selected = set() if len(selected) == len(files) else set(range(len(files)))
                else:
                    file_index = cursor - 1
                    if file_index in selected:
                        selected.remove(file_index)
                    else:
                        selected.add(file_index)
                continue
            if key in (10, 13, curses.KEY_ENTER):
                if not selected:
                    message = "Select at least one model, or press q to cancel."
                    curses.beep()
                    continue
                return [files[index] for index in sorted(selected)]

    return curses.wrapper(run)


def pick_files_interactively(files: list[Path], models_dir: Path) -> list[Path]:
    if not files:
        print(f"No model files found in: {models_dir}")
        return []

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return pick_files_with_prompt(files, models_dir)

    try:
        return pick_files_with_curses(files, models_dir)
    except Exception as exc:
        print(f"Could not open cursor menu, falling back to text prompt: {exc}", file=sys.stderr)
        return pick_files_with_prompt(files, models_dir)


def prompt_project(project: str | None) -> str:
    if project:
        return project
    if sys.stdin.isatty():
        project = input("W&B project (or set WANDB_PROJECT): ").strip()
        if project:
            return project
    raise ValueError("W&B project is required. Pass --project or set WANDB_PROJECT.")


def artifact_stem(path: Path) -> str:
    name = path.name
    for suffix in reversed(path.suffixes):
        name = name[: -len(suffix)]
    return name or path.stem


def sanitize_artifact_name(name: str) -> str:
    sanitized = ARTIFACT_NAME_PATTERN.sub("-", name.strip())
    sanitized = sanitized.strip(".-_")
    if not sanitized:
        raise ValueError(f"Could not build a valid artifact name from: {name!r}")
    return sanitized


def artifact_name_for(path: Path, models_dir: Path, prefix: str) -> str:
    rel_parent = path.relative_to(models_dir).parent.as_posix()
    base_name = artifact_stem(path)
    if rel_parent != ".":
        base_name = f"{rel_parent}-{base_name}"
    if prefix:
        base_name = f"{prefix}-{base_name}"
    return sanitize_artifact_name(base_name)


def metadata_for(path: Path, models_dir: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "source_path": path.relative_to(models_dir).as_posix(),
        "original_filename": path.name,
        "file_size_bytes": stat.st_size,
        "modified_time_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "publisher": "scripts/publish_models_to_wandb.py",
    }


def confirm_publish(
    files: list[Path],
    models_dir: Path,
    project: str,
    entity: str | None,
    aliases: list[str],
    skip_confirmation: bool,
    dry_run: bool,
) -> bool:
    print()
    print("Publish plan:")
    print(f"  project: {project}")
    print(f"  entity:  {entity or '(default)'}")
    print(f"  aliases: {', '.join(aliases) if aliases else '(none)'}")
    print(f"  mode:    {'dry-run' if dry_run else 'upload'}")
    for path in files:
        rel_path = path.relative_to(models_dir).as_posix()
        print(f"  - {rel_path} ({human_size(path.stat().st_size)})")

    if skip_confirmation or not sys.stdin.isatty():
        return True

    answer = input("Continue? [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


def publish_files(
    files: list[Path],
    models_dir: Path,
    project: str,
    entity: str | None,
    artifact_type: str,
    artifact_prefix: str,
    job_type: str,
    aliases: list[str],
    dry_run: bool,
) -> None:
    if dry_run:
        for path in files:
            print(
                "DRY RUN:",
                path.relative_to(models_dir).as_posix(),
                "->",
                artifact_name_for(path, models_dir, artifact_prefix),
            )
        return

    import wandb

    init_kwargs: dict[str, object] = {
        "project": project,
        "job_type": job_type,
        "config": {
            "models_dir": str(models_dir),
            "published_files": [path.relative_to(models_dir).as_posix() for path in files],
        },
    }
    if entity:
        init_kwargs["entity"] = entity

    with wandb.init(**init_kwargs) as run:
        for path in files:
            rel_path = path.relative_to(models_dir).as_posix()
            artifact_name = artifact_name_for(path, models_dir, artifact_prefix)
            artifact = wandb.Artifact(
                name=artifact_name,
                type=artifact_type,
                description=f"Model file from models/{rel_path}",
                metadata=metadata_for(path, models_dir),
            )
            artifact.add_file(local_path=str(path), name=path.name)
            logged_artifact = run.log_artifact(artifact, aliases=aliases or None)
            if hasattr(logged_artifact, "wait"):
                logged_artifact.wait()
            print(f"Published {rel_path} as {artifact_name}")


def selected_files_from_args(args: argparse.Namespace, models_dir: Path) -> list[Path]:
    selected: list[Path] = []

    for raw_path in args.models:
        selected.append(resolve_model_path(raw_path, models_dir))

    if args.all or args.pattern:
        selected.extend(discover_files(models_dir, args.all_files, args.pattern))

    if selected:
        return dedupe_paths(selected)

    if not sys.stdin.isatty():
        raise ValueError("No models selected. Pass model paths, --pattern, or --all.")

    return pick_files_interactively(discover_files(models_dir, args.all_files, []), models_dir)


def main() -> int:
    args = parse_args()

    models_dir = Path(args.models_dir).expanduser().resolve()
    if not models_dir.is_dir():
        print(f"Models directory not found: {models_dir}", file=sys.stderr)
        return 1

    try:
        project = prompt_project(args.project)
        files = selected_files_from_args(args, models_dir)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not files:
        print("No files selected.")
        return 0

    aliases = [] if args.no_latest else ["latest"]
    aliases.extend(alias for alias in args.alias if alias)

    if not confirm_publish(files, models_dir, project, args.entity, aliases, args.yes, args.dry_run):
        print("Cancelled.")
        return 0

    try:
        publish_files(
            files=files,
            models_dir=models_dir,
            project=project,
            entity=args.entity,
            artifact_type=args.artifact_type,
            artifact_prefix=args.artifact_prefix,
            job_type=args.job_type,
            aliases=aliases,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        print(f"Error: W&B publish failed: {exc}", file=sys.stderr)
        print("Check that `wandb login` is completed or WANDB_API_KEY is set.", file=sys.stderr)
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
