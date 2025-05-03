# Generates indexes of all texts in a project.

# This should work by:
# 1. Read ALL pages in the project
# 2. Create list of terms used 
#   2b.? Remove all boring/stop words using AI or otherwise
# 3. Keep track of which pages each term appears in
# 4. Write out a JSON file with the index

import json
from pathlib import Path

def generate_lunr_index(project_name, output_path="static/lunr_index.json"):
    """
    Generate a Lunr.js-compatible search index from all final transcription and translation markdown files
    in the specified project. Writes a flat JSON array to the given output_path.
    Raises FileNotFoundError if the project does not exist.
    """
    project_dir = Path("projects") / project_name
    if not project_dir.is_dir():
        raise FileNotFoundError(f"Project directory not found: {project_dir}")
    index = []
    for type in ["transcription", "translation"]:
        final_dir = project_dir / type / "final"
        if not final_dir.exists():
            continue
        for file in final_dir.glob("*.md"):
            page_name = file.stem.replace("_final", "")
            text = file.read_text()
            doc_id = f"{project_name}__{type}__{page_name}"
            url = f"/docs/{project_name}/{type}/{page_name}"
            # Extract last 4 digits from page_name for page_id_string
            digits = ''.join(filter(str.isdigit, page_name))
            if digits:
                page_id_string = digits[-4:].zfill(4)
            else:
                page_id_string = page_name[-4:]
            index.append({
                "id": doc_id,
                "project": project_name,
                "type": type,
                "page_name": page_name,
                "page_id_string": page_id_string,
                "title": f"{project_name} {type.capitalize()} {page_name}",
                "text": text,
                "url": url
            })
    # Sort results by page_id_string for consistent ordering in the index
    index.sort(key=lambda x: x["page_id_string"])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

