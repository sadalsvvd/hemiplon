import os
import difflib
from typing import Dict, List, Tuple
import logging
import re

def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text:
    - Replace multiple spaces with single space
    - Remove trailing whitespace
    - Remove empty lines
    
    Args:
        text: Input text string
        
    Returns:
        Normalized text string
    """
    # Replace multiple spaces with single space and strip each line
    return re.sub(r'\s+', ' ', text.strip())

def compare_differences(folder1: str, folder2: str, file_pattern: str = "*", ignore_whitespace: bool = True) -> Dict[str, str]:
    """
    Compare files with matching names between two folders and generate diffs.
    
    Args:
        folder1: Path to first folder
        folder2: Path to second folder
        file_pattern: Optional glob pattern to filter files (default: "*")
        ignore_whitespace: If True, ignores whitespace differences (default: True)
        
    Returns:
        Dictionary mapping filenames to their diff outputs
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Get list of files in both directories
    try:
        files1 = set(os.listdir(folder1))
        files2 = set(os.listdir(folder2))
    except OSError as e:
        logging.error(f"Error accessing folders: {e}")
        return {}

    # Find common files
    common_files = files1.intersection(files2)
    if not common_files:
        logging.warning("No matching files found between the folders")
        return {}
    
    logging.info(f"Found {len(common_files)} files to compare")
    
    # Store diffs for each file
    diffs: Dict[str, str] = {}
    
    for filename in sorted(common_files):
        path1 = os.path.join(folder1, filename)
        path2 = os.path.join(folder2, filename)
        
        try:
            with open(path1, 'r', encoding='utf-8') as f1, \
                 open(path2, 'r', encoding='utf-8') as f2:
                
                # Read files and split into lines
                if ignore_whitespace:
                    # Normalize whitespace in each line
                    lines1 = [normalize_whitespace(line) + '\n' for line in f1 if normalize_whitespace(line)]
                    lines2 = [normalize_whitespace(line) + '\n' for line in f2 if normalize_whitespace(line)]
                else:
                    lines1 = f1.readlines()
                    lines2 = f2.readlines()
                
                # Generate unified diff
                diff = difflib.unified_diff(
                    lines1, lines2,
                    fromfile=f"{folder1}/{filename}",
                    tofile=f"{folder2}/{filename}",
                    lineterm=''
                )
                
                # Convert diff iterator to string
                diff_text = '\n'.join(diff)
                
                if diff_text:  # Only store if there are differences
                    diffs[filename] = diff_text
                    logging.info(f"Found differences in {filename}")
                else:
                    logging.info(f"No differences found in {filename}")
                    
        except Exception as e:
            logging.error(f"Error comparing {filename}: {e}")
            continue
    
    return diffs

def save_diffs_to_file(diffs: Dict[str, str], output_file: str) -> None:
    """
    Save the generated diffs to a file.
    
    Args:
        diffs: Dictionary of filename to diff text
        output_file: Path to save the diff output
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            if not diffs:
                f.write("No differences found between files.\n")
                return
                
            for filename, diff in diffs.items():
                f.write(f"\n{'='*80}\n")
                f.write(f"Differences in {filename}:\n")
                f.write(f"{'='*80}\n\n")
                f.write(diff)
                f.write("\n")
                
        logging.info(f"Diff output saved to {output_file}")
        
    except Exception as e:
        logging.error(f"Error saving diff output: {e}")


def normalize_for_diff(text: str) -> str:
    """Normalize text for comparison by removing formatting differences."""
    # Remove all line breaks except paragraph breaks
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    # Normalize multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Normalize paragraph breaks
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

def generate_multi_way_diff(texts: List[str], labels: List[str]) -> str:
    """
    Generate a multi-way diff optimized for LLM analysis.
    
    Args:
        texts: List of text versions to compare
        labels: List of labels for each version (e.g. ['gpt-4.1', 'gpt-4.1-mini', 'claude'])
    
    Returns:
        A formatted string showing differences between versions
    """
    if len(texts) != len(labels):
        raise ValueError("Number of texts must match number of labels")
    
    # Normalize all texts
    normalized_texts = [normalize_for_diff(t) for t in texts]
    
    # Split into paragraphs for semantic comparison
    paragraphs = []
    for text in normalized_texts:
        # Split on double newlines, preserving empty paragraphs
        paras = [p.strip() for p in text.split('\n\n')]
        paragraphs.append(paras)
    
    # Find the maximum number of paragraphs
    max_paras = max(len(p) for p in paragraphs)
    
    # Build the diff output
    output = []
    output.append("=== MULTI-WAY DIFF ANALYSIS ===\n")
    
    for para_idx in range(max_paras):
        output.append(f"\n--- Paragraph {para_idx + 1} ---")
        
        # Get this paragraph from each version
        current_paras = []
        for i, paras in enumerate(paragraphs):
            if para_idx < len(paras):
                current_paras.append((labels[i], paras[para_idx]))
            else:
                current_paras.append((labels[i], ""))
        
        # If all versions have the same paragraph, just show it once
        if len(set(p[1] for p in current_paras)) == 1:
            output.append(f"\n[ALL VERSIONS AGREE]:")
            output.append(current_paras[0][1])
            continue
        
        # Otherwise show differences
        output.append("\n[DIFFERENCES FOUND]:")
        for label, para in current_paras:
            if para:  # Only show non-empty paragraphs
                output.append(f"\n{label}:")
                output.append(para)
    
    # Add a summary of differences
    output.append("\n=== DIFFERENCE SUMMARY ===")
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if normalized_texts[i] != normalized_texts[j]:
                output.append(f"\n{labels[i]} vs {labels[j]}:")
                # Use difflib for line-by-line comparison
                differ = difflib.Differ()
                diff = list(differ.compare(
                    normalized_texts[i].split('\n'),
                    normalized_texts[j].split('\n')
                ))
                # Only show lines with differences
                diff_lines = [line for line in diff if line.startswith(('+', '-', '?'))]
                if diff_lines:
                    output.extend(diff_lines)
    
    return '\n'.join(output)

def compare_multiple_folders(folders: List[str], labels: List[str], file_pattern: str = "*", ignore_whitespace: bool = True) -> Dict[str, str]:
    """
    Compare files with matching names across multiple folders and generate multi-way diffs.
    
    Args:
        folders: List of paths to folders containing transcriptions
        labels: List of labels for each folder (e.g. ['gpt-4.1', 'gpt-4.1-mini', 'claude'])
        file_pattern: Optional glob pattern to filter files (default: "*")
        ignore_whitespace: If True, ignores whitespace differences (default: True)
        
    Returns:
        Dictionary mapping filenames to their multi-way diff outputs
    """
    if len(folders) != len(labels):
        raise ValueError("Number of folders must match number of labels")
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Get list of files in all directories
    try:
        file_sets = [set(os.listdir(folder)) for folder in folders]
    except OSError as e:
        logging.error(f"Error accessing folders: {e}")
        return {}

    # Find common files across all folders
    common_files = set.intersection(*file_sets)
    if not common_files:
        logging.warning("No matching files found across all folders")
        return {}
    
    logging.info(f"Found {len(common_files)} files to compare")
    
    # Store diffs for each file
    diffs: Dict[str, str] = {}
    
    for filename in sorted(common_files):
        try:
            # Read all versions of the file
            file_contents = []
            for folder in folders:
                path = os.path.join(folder, filename)
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if ignore_whitespace:
                        content = normalize_whitespace(content)
                    file_contents.append(content)
            
            # Generate multi-way diff
            diff_text = generate_multi_way_diff(file_contents, labels)
            
            if diff_text:  # Only store if there are differences
                diffs[filename] = diff_text
                logging.info(f"Generated multi-way diff for {filename}")
            else:
                logging.info(f"No differences found in {filename}")
                
        except Exception as e:
            logging.error(f"Error processing {filename}: {e}")
            continue
    
    return diffs

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare files between folders")
    subparsers = parser.add_subparsers(dest='command', help='Comparison mode')
    
    # Two-way diff parser
    two_way = subparsers.add_parser('two-way', help='Compare two folders')
    two_way.add_argument("folder1", help="Path to first folder")
    two_way.add_argument("folder2", help="Path to second folder")
    two_way.add_argument("--pattern", default="*", help="File pattern to match (default: *)")
    two_way.add_argument("--output", default="diffs.txt", help="Output file for diffs (default: diffs.txt)")
    two_way.add_argument("--ignore-whitespace", action="store_true", default=True,
                        help="Ignore whitespace differences (default: True)")
    
    # Multi-way diff parser
    multi_way = subparsers.add_parser('multi-way', help='Compare multiple folders')
    multi_way.add_argument("folders", nargs='+', help="Paths to folders to compare")
    multi_way.add_argument("--labels", nargs='+', required=True, help="Labels for each folder")
    multi_way.add_argument("--pattern", default="*", help="File pattern to match (default: *)")
    multi_way.add_argument("--output", default="multi_diffs.txt", help="Output file for diffs (default: multi_diffs.txt)")
    multi_way.add_argument("--ignore-whitespace", action="store_true", default=True,
                          help="Ignore whitespace differences (default: True)")
    
    args = parser.parse_args()
    
    if args.command == 'two-way':
        diffs = compare_differences(args.folder1, args.folder2, args.pattern, 
                                  ignore_whitespace=args.ignore_whitespace)
    elif args.command == 'multi-way':
        if len(args.folders) != len(args.labels):
            parser.error("Number of folders must match number of labels")
        diffs = compare_multiple_folders(args.folders, args.labels, args.pattern,
                                       ignore_whitespace=args.ignore_whitespace)
    else:
        parser.print_help()
        exit(1)
    
    save_diffs_to_file(diffs, args.output) 