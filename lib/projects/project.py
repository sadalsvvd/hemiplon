from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import yaml
import re

@dataclass
class FileSet:
    """Represents a set of files with their page IDs and metadata"""
    files: List[Path]
    page_ids: List[str]
    page_id_to_file: Dict[str, Path]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TranscriptionConfig:
    model: str
    runs: int = 1
    max_concurrent: int = 3
    ocr_prompt_path: str = "lib/prompts/transcribe.md"

@dataclass
class TranslationConfig:
    model: str
    runs: int = 1

@dataclass
class TranscriptionReviewConfig:
    model: str

@dataclass
class Project:
    name: str
    input_file: str
    two_page_spread: bool = True  # Whether the input PDF is in two-page spread format
    transcription: List[TranscriptionConfig] = field(default_factory=list)
    translation: List[TranslationConfig] = field(default_factory=list)
    transcription_review: List[TranscriptionReviewConfig] = field(default_factory=list)
    prompts: Dict[str, str] = field(default_factory=dict)
    prompts_dir: Path = Path("lib/prompts")
    
    @property
    def project_dir(self) -> Path:
        return Path("projects") / self.name
    
    @property
    def images_dir(self) -> Path:
        return self.project_dir / "images"
    
    @property
    def transcription_dir(self) -> Path:
        return self.project_dir / "transcription"
    
    @property
    def translation_dir(self) -> Path:
        return self.project_dir / "translation"
    
    @property
    def config_path(self) -> Path:
        return self.project_dir / "config.yaml"
    
    @property
    def transcription_diffs_dir(self) -> Path:
        return self.transcription_dir / "diffs"

    @property
    def transcription_reviewed_dir(self) -> Path:
        return self.transcription_dir / "reviewed"

    @property
    def transcription_final_dir(self) -> Path:
        return self.transcription_dir / "final"

    def transcription_run_dir(self, model: str, run: int = 1) -> Path:
        run_suffix = f"_{run}" if run >= 1 else ""
        return self.transcription_dir / f"transcribed_{model}{run_suffix}"

    @property
    def translation_diffs_dir(self) -> Path:
        return self.translation_dir / "diffs"

    @property
    def translation_reviewed_dir(self) -> Path:
        return self.translation_dir / "reviewed"

    @property
    def translation_final_dir(self) -> Path:
        return self.translation_dir / "final"

    def translation_run_dir(self, model: str, run: int = 1) -> Path:
        run_suffix = f"_{run}" if run >= 1 else ""
        return self.translation_dir / f"translated_{model}{run_suffix}"
    
    def setup_directories(self) -> None:
        """Create necessary project directories"""
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        self.transcription_dir.mkdir(exist_ok=True)
        self.translation_dir.mkdir(exist_ok=True)
    
    def save_config(self) -> None:
        """Save project configuration to YAML file"""
        config = {
            "project": {
                "name": self.name,
                "input_file": self.input_file,
                "two_page_spread": self.two_page_spread,
                "transcription": [
                    {
                        "model": t.model,
                        "runs": t.runs,
                        "max_concurrent": t.max_concurrent,
                        "ocr_prompt_path": t.ocr_prompt_path
                    }
                    for t in self.transcription
                ],
                "translation": [
                    {"model": t.model, "runs": t.runs}
                    for t in self.translation
                ],
                "transcription_review": [
                    {"model": r.model}
                    for r in self.transcription_review
                ],
                "prompts": self.prompts
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    @classmethod
    def from_yaml(cls, project_name: str) -> 'Project':
        """Load project from YAML configuration"""
        project_dir = Path("projects") / project_name
        config_path = project_dir / "config.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Project configuration not found at {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        project_config = config["project"]
        
        return cls(
            name=project_name,
            input_file=project_config["input_file"],
            two_page_spread=project_config.get("two_page_spread", True),
            transcription=[
                TranscriptionConfig(**t) for t in project_config.get("transcription", [])
            ],
            translation=[
                TranslationConfig(**t) for t in project_config.get("translation", [])
            ],
            transcription_review=[
                TranscriptionReviewConfig(**r) for r in project_config.get("transcription_review", [])
            ],
            prompts=project_config.get("prompts", {}),
        )
    
    @classmethod
    def create(cls, name: str, input_file: str, two_page_spread: bool = True) -> 'Project':
        """Create a new project with default configuration"""
        project = cls(
            name=name,
            input_file=input_file,
            two_page_spread=two_page_spread,
            transcription=[
                TranscriptionConfig(model="gpt-4.1", runs=2, max_concurrent=3),
                TranscriptionConfig(model="gpt-4.1-mini", runs=2, max_concurrent=3),
            ],
            translation=[
                TranslationConfig(model="gpt-4.1", runs=2),
            ],
            transcription_review=[
                TranscriptionReviewConfig(model="gpt-4.1"),
            ]
        )
        
        project.setup_directories()
        project.save_config()
        
        return project 

    def get_prompt_path(self, key: str) -> Path:
        # If a custom prompt is set, use it; else use default in prompts_dir
        if key in self.prompts:
            return Path(self.prompts[key])
        default_map = {
            "transcription_review": self.prompts_dir / "transcription_review.md",
            "transcribe": self.prompts_dir / "transcribe.md",
            "finalize_transcription": self.prompts_dir / "finalize_transcription.j2",
            "finalize_translation": self.prompts_dir / "finalize_translation.j2",
            "translate": self.prompts_dir / "translate.j2",
            "translation_review": self.prompts_dir / "translation_review.j2",
        }
        return default_map[key] 

    def get_stage_dir(self, stage: str, run: int = 1, model: str | None = None) -> Path:
        """Get the directory for a specific stage, run, and model."""
        if stage == "transcription":
            if model is None:
                raise ValueError("Model is required for transcription stage")
            return self.transcription_run_dir(model, run)
        elif stage == "translation":
            if model is None:
                raise ValueError("Model is required for translation stage")
            return self.translation_run_dir(model, run)
        elif stage == "transcription-diff":
            return self.transcription_diffs_dir
        elif stage == "translation-diff":
            return self.translation_diffs_dir
        elif stage == "transcription-review":
            return self.transcription_reviewed_dir
        elif stage == "translation-review":
            return self.translation_reviewed_dir
        elif stage == "transcription-final":
            return self.transcription_final_dir
        elif stage == "translation-final":
            return self.translation_final_dir
        else:
            raise ValueError(f"Unknown stage: {stage}")

    @staticmethod
    def extract_page_id(filename: str, project_key: str) -> str:
        """
        Extracts the page id (e.g. CCAG02_page_0005) from a filename using the project key.
        """
        # Remove extension if present
        base = filename
        if base.endswith('.md') or base.endswith('.jpg'):
            base = base[:-3]
        # Regex: match {project_key}_page_XXXX
        match = re.search(rf'({re.escape(project_key)}_page_\d+)', base)
        if match:
            return match.group(1)
        raise ValueError(f'Could not extract page id from {filename} with project key {project_key}')

    def get_files_for_stage(
        self,
        stage: str,
        run: int = 1,
        model: str | None = None,
        start_index: int = 0,
        end_index: int | None = None,
        include_adjacent_pages: bool = False,
        adjacent_page_context: int = 1
    ) -> FileSet:
        """
        Get files for a specific stage, with optional context from previous runs or adjacent pages.
        
        Args:
            stage: Stage name (e.g. 'transcription', 'translation')
            run: Run number (default: 1)
            model: Model name (optional)
            start_index: First file index to include
            end_index: Last file index to include (exclusive)
            include_previous_run: Whether to include files from previous run
            include_adjacent_pages: Whether to include previous/next page files
            adjacent_page_context: How many pages before/after to include
            
        Returns:
            FileSet containing the requested files and metadata
        """
        # Get base directory for this stage/run
        base_dir = self.get_stage_dir(stage, run, model)
        if not base_dir.exists():
            raise FileNotFoundError(f"Directory not found: {base_dir}")
        
        # Get all files and sort them
        files = sorted(base_dir.glob("*.md"))
        
        # Slice according to indices
        if end_index is None:
            end_index = len(files)
        files = files[start_index:end_index]

        # Use extract_page_id for all page id extraction
        page_ids = [self.extract_page_id(f.name, self.name) for f in files]
        page_id_to_file = {pid: f for pid, f in zip(page_ids, files)}

        metadata = {}
        run_yaml = base_dir / "run.yaml"
        if run_yaml.exists():
            with open(run_yaml) as f:
                metadata = yaml.safe_load(f)
        
        # Get adjacent page files if requested
        adjacent_files = {}
        if include_adjacent_pages:
            all_files = sorted(base_dir.glob("*.md"))
            all_page_ids = [self.extract_page_id(f.name, self.name) for f in all_files]
            all_page_id_to_file = {pid: f for pid, f in zip(all_page_ids, all_files)}

            for pid in page_ids:
                idx = all_page_ids.index(pid)
                start_idx = max(0, idx - adjacent_page_context)
                end_idx = min(len(all_page_ids), idx + adjacent_page_context + 1)
                
                for adj_pid in all_page_ids[start_idx:end_idx]:
                    if adj_pid != pid:
                        adjacent_files[adj_pid] = all_page_id_to_file[adj_pid]
        
        return FileSet(
            files=files,
            page_ids=page_ids,
            page_id_to_file=page_id_to_file,
            metadata={
                "stage": stage,
                "run": run,
                "model": model,
                "adjacent_files": adjacent_files,
                **metadata
            }
        ) 