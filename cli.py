import argparse
import asyncio
from pathlib import Path
from typing import List

from lib.projects.project_manager import create_project, load_project, ProjectManager

# Canonical stage order
STAGE_ORDER = [
    "pdf",
    "transcription",
    "transcription-diff",
    "transcription-review",
    "transcription-finalize",
    "translation",
    "translation-diff",
    "translation-review",
    "translation-finalize",
]

def main():
    parser = argparse.ArgumentParser(description="Manage transcription projects")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Create project command
    create_parser = subparsers.add_parser("create", help="Create a new project")
    create_parser.add_argument("name", help="Project name")
    create_parser.add_argument("input_file", help="Path to input PDF file")

    # Run pipeline command
    run_parser = subparsers.add_parser("run", help="Run project pipeline")
    run_parser.add_argument("name", help="Project name")
    stages_group = run_parser.add_mutually_exclusive_group()
    stages_group.add_argument(
        "--stages",
        nargs="*",
        choices=STAGE_ORDER,
        help="Specific stages to run. If not specified, runs all stages of both transcription and translation pipelines.",
    )
    stages_group.add_argument(
        "--from-stage",
        choices=STAGE_ORDER,
        help="Run all stages starting from the specified stage (inclusive) in canonical order.",
    )
    run_parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="For transcription stage, index of first image to process (inclusive)",
    )
    run_parser.add_argument(
        "--end",
        type=int,
        help="For transcription stage, index of last image to process (exclusive)",
    )

    args = parser.parse_args()

    if args.command == "create":
        # Create project
        project = create_project(args.name, args.input_file)
        print(f"Created project {args.name} in {project.project_dir}")

    elif args.command == "run":
        # Compute stages to run
        if args.stages:
            stages = args.stages
        elif args.from_stage:
            idx = STAGE_ORDER.index(args.from_stage)
            stages = STAGE_ORDER[idx:]
        else:
            stages = None
        # Load and run project
        project = load_project(args.name)
        manager = ProjectManager(project)
        asyncio.run(
            manager.run_pipeline(
                stages=stages, start_index=args.start, end_index=args.end
            )
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
