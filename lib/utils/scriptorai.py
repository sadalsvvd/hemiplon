# Utilities for generating manifests and other utilities
# for use in Scriptorai

import os
import json
import re
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional


def generate_ccag_manifest(
    base_dir: str, 
    text_file_base: str = "CCAG01", 
    text_slug: str = "ccag-01"
) -> None:
    """
    Generate a manifest JSON file mapping page IDs to their respective image, 
    translation, and transcription files. Also copies all matching files to
    an output directory structure that matches the paths in the manifest.
    
    Args:
        base_dir: Base directory containing images, translated, and transcribed subdirectories
        text_file_base: The base file name pattern used in the regex to extract page IDs (e.g., "CCAG01")
        text_slug: The slug used in forming file paths for the manifest (e.g., "ccag-01")
    
    Returns:
        None. Writes JSON file to the specified output path and copies matching files.
    """
    base_path = Path(base_dir)
    print(f"Base path: {base_path}")
    images_dir = base_path / 'images'
    translation_dir = base_path / 'translation' / 'final'
    transcription_dir = base_path / 'transcription' / 'final'

    # Create output directory structure
    output_dir = base_path / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create top-level output directories - not inside text_slug subfolder
    output_images_dir = output_dir / 'images'
    output_translation_dir = output_dir / 'translation'
    output_transcription_dir = output_dir / 'transcription'
    
    # Create all output directories
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_translation_dir.mkdir(parents=True, exist_ok=True)
    output_transcription_dir.mkdir(parents=True, exist_ok=True)
    
    # Get lists of files in each directory
    images = os.listdir(images_dir) if images_dir.exists() else []
    translations = os.listdir(translation_dir) if translation_dir.exists() else []
    transcriptions = os.listdir(transcription_dir) if transcription_dir.exists() else []
    
    def get_page_id(filename: str) -> Optional[str]:
        # Configurable regex pattern based on text_file_base
        pattern = f"{text_file_base}_page_(\\d{{4}})"
        match = re.search(pattern, filename)
        return match.group(1) if match else None
    
    pages = []
    copied_files = {"images": 0, "translation": 0, "transcription": 0}
    
    for img in images:
        page_id = get_page_id(img)
        print(f"Page ID: {page_id}")
        if not page_id:
            print(f"No page ID found for image: {img}")
            continue
            
        translation = next((f for f in translations if page_id in f), None)
        transcription = next((f for f in transcriptions if page_id in f), None)

        if translation and transcription:
            # Copy the image to the output directory
            src_image_path = images_dir / img
            dst_image_path = output_images_dir / img
            
            # Copy the translation to the output directory
            src_translation_path = translation_dir / translation
            dst_translation_path = output_translation_dir / translation
            
            # Copy the transcription to the output directory
            src_transcription_path = transcription_dir / transcription
            dst_transcription_path = output_transcription_dir / transcription
            
            # Only copy if source exists and is a file
            if src_image_path.exists() and src_image_path.is_file():
                shutil.copy2(src_image_path, dst_image_path)
                copied_files["images"] += 1
            
            if src_translation_path.exists() and src_translation_path.is_file():
                shutil.copy2(src_translation_path, dst_translation_path)
                copied_files["translation"] += 1
            
            if src_transcription_path.exists() and src_transcription_path.is_file():
                shutil.copy2(src_transcription_path, dst_transcription_path)
                copied_files["transcription"] += 1
            
            # Add entry to manifest -- text_slug is included, assuming that the folder will
            # be put into the Scriptorai static folder
            pages.append({
                "pageId": page_id,
                "image": f"/{text_slug}/images/{img}",
                "translation": f"/{text_slug}/translation/{translation}",
                "transcription": f"/{text_slug}/transcription/{transcription}",
            })
        else:
            print(f"No translation or transcription found for page ID: {page_id}")
            print(f"Translation: {translation}")
            print(f"Transcription: {transcription}")

    pages = sorted(pages, key=lambda x: int(x["pageId"]))
    
    # Create the manifest structure with text_slug at the top level and pages in a nested array
    manifest = {
        "text_slug": text_slug,
        "pages": pages
    }
    
    # Write the manifest to file
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Wrote manifest for {len(pages)} pages with text_slug '{text_slug}'.")
    print(f"Copied files: {copied_files['images']} images, {copied_files['translation']} translations, {copied_files['transcription']} transcriptions.")

