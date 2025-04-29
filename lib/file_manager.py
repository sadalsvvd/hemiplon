from pathlib import Path
import yaml

class FileManager:
    @staticmethod
    def write_text(path: Path | str, text: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    @staticmethod
    def read_text(path: Path | str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def write_yaml(path: Path | str, data: dict) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f)

    @staticmethod
    def read_yaml(path: Path | str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def ensure_dir(path: Path | str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True) 