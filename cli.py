import argparse
import asyncio
import os

from lib.projects.project_manager import create_project, load_project, ProjectManager
from lib.utils.scriptorai import generate_ccag_manifest

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

# Create a dedicated coroutine for running the pipeline
async def run_pipeline_main(project, stages=None, start_index=0, end_index=None):
    # Create ProjectManager inside the coroutine to ensure semaphore is in the correct event loop
    manager = ProjectManager(project)
    await manager.run_pipeline(stages=stages, start_index=start_index, end_index=end_index)

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

    # Generate manifest command
    manifest_parser = subparsers.add_parser("manifest", help="Generate a manifest for a project")
    manifest_parser.add_argument("name", help="Project name")
    manifest_parser.add_argument(
        "--text-file-base", 
        default="CCAG01",
        help="Base file name pattern used in the regex to extract page IDs (default: CCAG01)"
    )
    manifest_parser.add_argument(
        "--text-slug", 
        default=None,
        help="Slug used in forming file paths for the manifest (default: project name)"
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
        
        # Load project
        project = load_project(args.name)
        
        # Run the pipeline in a dedicated coroutine
        asyncio.run(
            run_pipeline_main(
                project=project,
                stages=stages, 
                start_index=args.start, 
                end_index=args.end
            )
        )
    
    elif args.command == "manifest":
        # Load project
        project = load_project(args.name)
        
        # Set default text_slug if not specified
        text_slug = args.text_slug if args.text_slug is not None else args.name
        
        # Set default output path if not specified
        output_path = os.path.join(project.project_dir, "output")
        
        # Construct base directory path
        base_dir = os.path.join(project.project_dir)
        os.makedirs(base_dir, exist_ok=True)
        
        if not os.path.exists(base_dir):
            print(f"Error: Base directory {base_dir} does not exist.")
            return
            
        # Generate the manifest
        generate_ccag_manifest(
            base_dir=base_dir,
            text_file_base=args.text_file_base,
            text_slug=text_slug
        )
        print(f"Generated manifest for project {args.name} at {output_path}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
