import argparse
import asyncio
from pathlib import Path
from typing import List

from lib.projects.project_manager import create_project, load_project, ProjectManager


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
    run_parser.add_argument(
        "--stages",
        nargs="*",
        choices=[
            "pdf",
            "transcription",
            "transcription-diff",
            "transcription-review",
            "transcription-finalize",
        ],
        help="Specific stages to run. If not specified, runs all stages.",
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
        # Load and run project
        project = load_project(args.name)
        manager = ProjectManager(project)
        asyncio.run(
            manager.run_pipeline(
                stages=args.stages, start_index=args.start, end_index=args.end
            )
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
