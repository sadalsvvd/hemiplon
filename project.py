from dataclasses import dataclass, field
from typing import List
from pathlib import Path
import yaml

@dataclass
class TranscriptionConfig:
    model: str
    runs: int = 1
    max_concurrent: int = 3
    ocr_prompt_path: str = "prompts/transcribe.md"

@dataclass
class TranslationConfig:
    model: str
    runs: int = 1

@dataclass
class Project:
    name: str
    input_file: str
    two_page_spread: bool = True  # Whether the input PDF is in two-page spread format
    transcription: List[TranscriptionConfig] = field(default_factory=list)
    translation: List[TranslationConfig] = field(default_factory=list)
    
    @property
    def project_dir(self) -> Path:
        return Path("projects") / self.name
    
    @property
    def images_dir(self) -> Path:
        return self.project_dir / "images"
    
    @property
    def output_dir(self) -> Path:
        return self.project_dir / "output"
    
    @property
    def config_path(self) -> Path:
        return self.project_dir / "config.yaml"
    
    def setup_directories(self) -> None:
        """Create necessary project directories"""
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
    
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
                ]
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
            ]
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
            ]
        )
        
        project.setup_directories()
        project.save_config()
        
        return project 