from __future__ import annotations

import argparse
from pathlib import Path
from typing import Union


def write_subfolders_abs_paths_to_txt(
    input_dir: Union[str, Path],
    output_txt: Union[str, Path],
    *,
    recursive: bool = False,
    include_input_dir: bool = False,
    sort_paths: bool = True,
    exist_ok: bool = True,
    encoding: str = "utf-8",
) -> None:
    input_dir = Path(input_dir).expanduser()
    output_txt = Path(output_txt).expanduser()

    if not input_dir.exists():
        if exist_ok:
            output_txt.parent.mkdir(parents=True, exist_ok=True)
            output_txt.write_text("", encoding=encoding)
            return
        raise FileNotFoundError(f"input_dir not found: {input_dir}")

    if not input_dir.is_dir():
        raise NotADirectoryError(f"input_dir is not a directory: {input_dir}")

    if recursive:
        dirs = [p for p in input_dir.rglob("*") if p.is_dir()]
    else:
        dirs = [p for p in input_dir.iterdir() if p.is_dir()]

    if include_input_dir:
        dirs.append(input_dir)

    abs_paths = [str(p.resolve()) for p in dirs]
    if sort_paths:
        abs_paths.sort()

    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_txt.write_text("\n".join(abs_paths) + ("\n" if abs_paths else ""), encoding=encoding)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Write absolute paths of subfolders under input_dir to output_txt."
    )
    p.add_argument("input_dir", type=Path, help="Input directory")
    p.add_argument("output_txt", type=Path, help="Output .txt file path")
    p.add_argument("--recursive", action="store_true", help="Search subfolders recursively")
    p.add_argument(
        "--include-input-dir",
        action="store_true",
        help="Also include input_dir itself in the output",
    )
    p.add_argument(
        "--no-sort",
        dest="sort_paths",
        action="store_false",
        help="Do not sort output paths",
    )
    p.add_argument(
        "--no-exist-ok",
        dest="exist_ok",
        action="store_false",
        help="If input_dir does not exist, raise instead of writing empty file",
    )
    p.add_argument("--encoding", default="utf-8", help="Output file encoding (default: utf-8)")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    write_subfolders_abs_paths_to_txt(
        args.input_dir,
        args.output_txt,
        recursive=args.recursive,
        include_input_dir=args.include_input_dir,
        sort_paths=args.sort_paths,
        exist_ok=args.exist_ok,
        encoding=args.encoding,
    )


if __name__ == "__main__":
    main()
