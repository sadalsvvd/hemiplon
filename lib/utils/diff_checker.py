import os
import difflib
from typing import Dict, List, Tuple
import logging
import re
from collections import defaultdict

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

def find_word_differences(text1: str, text2: str) -> List[Tuple[str, str]]:
    """Find word-level differences between two texts."""
    words1 = text1.split()
    words2 = text2.split()
    
    matcher = difflib.SequenceMatcher(None, words1, words2)
    differences = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != 'equal':
            diff1 = ' '.join(words1[i1:i2]) if i1 < i2 else None
            diff2 = ' '.join(words2[j1:j2]) if j1 < j2 else None
            if diff1 or diff2:
                differences.append((diff1, diff2))
    
    return differences

def categorize_differences(texts: List[str], labels: List[str]) -> Dict[str, List[str]]:
    """Categorize differences between versions."""
    categories = defaultdict(list)
    
    # Compare each pair of texts
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            # Find word-level differences
            diffs = find_word_differences(texts[i], texts[j])
            for diff1, diff2 in diffs:
                if diff1 and diff2:  # Substitution
                    categories['word_substitutions'].append(
                        f"- {labels[i]}: {diff1}")
                    categories['word_substitutions'].append(
                        f"+ {labels[j]}: {diff2}")
                elif diff1:  # Deletion (only in first version)
                    categories['deletions'].append(
                        f"- Only in {labels[i]}: {diff1}")
                elif diff2:  # Addition (only in second version)
                    categories['additions'].append(
                        f"+ Only in {labels[j]}: {diff2}")
    
    # Remove duplicates while preserving order
    for category in categories:
        seen = set()
        categories[category] = [x for x in categories[category] 
                              if not (x in seen or seen.add(x))]
    
    return dict(categories)

def wrap_text_with_markers(text: str, markers: List[Tuple[int, int, str]], width: int = 78) -> List[Tuple[str, str]]:
    """
    Wrap text to specified width while maintaining marker positions.
    Returns list of (text_line, marker_line) tuples.
    """
    lines = []
    current_pos = 0
    
    while current_pos < len(text):
        # Find the next line break point
        line_end = min(current_pos + width, len(text))
        if line_end < len(text):
            # Try to break at a word boundary
            while line_end > current_pos and text[line_end] != ' ':
                line_end -= 1
            if line_end == current_pos:  # No space found, force break
                line_end = min(current_pos + width, len(text))
        
        # Extract this line
        line_text = text[current_pos:line_end].rstrip()
        
        # Create marker line for this segment
        marker_line = [' '] * len(line_text)
        for start, end, marker in markers:
            if start >= current_pos and start < line_end:
                # Marker starts in this line
                marker_start = start - current_pos
                marker_end = min(end - current_pos, len(line_text))
                for i in range(marker_start, marker_end):
                    marker_line[i] = marker
        
        lines.append((line_text, ''.join(marker_line)))
        
        # Move to next line
        current_pos = line_end
        while current_pos < len(text) and text[current_pos] == ' ':
            current_pos += 1
    
    return lines

def generate_multi_way_diff(texts: List[str], labels: List[str]) -> str:
    """
    Generate a multi-way diff optimized for LLM analysis.
    """
    if len(texts) != len(labels):
        raise ValueError("Number of texts must match number of labels")
    
    # Normalize all texts
    normalized_texts = [normalize_for_diff(t) for t in texts]
    
    # Split into paragraphs for semantic comparison
    paragraphs = []
    for text in normalized_texts:
        paras = [p.strip() for p in text.split('\n\n')]
        paragraphs.append(paras)
    
    # Build the diff output
    output = []
    
    # Process each paragraph
    for para_idx in range(max(len(p) for p in paragraphs)):
        output.append(f"\n=== Paragraph {para_idx + 1} ===")
        
        # Get this paragraph from each version
        current_paras = []
        for i, paras in enumerate(paragraphs):
            if para_idx < len(paras):
                current_paras.append((labels[i], paras[para_idx]))
            else:
                current_paras.append((labels[i], ""))
        
        # If all versions agree
        if len(set(p[1] for p in current_paras)) == 1 and current_paras[0][1]:
            output.append("[CONSENSUS]")
            # Wrap consensus text
            wrapped = wrap_text_with_markers(current_paras[0][1], [], width=78)
            for text_line, _ in wrapped:
                output.append(text_line)
            continue
        
        # If there are differences, show full paragraphs with markers
        output.append("[DIFFERENCES]")
        
        # Initialize variables
        para_texts = [p[1] for p in current_paras if p[1]]
        para_labels = [p[0] for p in current_paras if p[1]]
        categories = {}
        diff_map = defaultdict(list)
        
        # Get word-level differences
        if para_texts:
            categories = categorize_differences(para_texts, para_labels)
            
            # Create a map of differences for marking
            for diff in categories.get('word_substitutions', []):
                if diff.startswith('-'):
                    label, text = diff[2:].split(': ', 1)
                    diff_map[label].append(('sub', text))
                elif diff.startswith('+'):
                    label, text = diff[2:].split(': ', 1)
                    diff_map[label].append(('add', text))
            
            for diff in categories.get('additions', []):
                if diff.startswith('+'):
                    label, text = diff[2:].split(': ', 1)
                    diff_map[label].append(('add', text))
            
            for diff in categories.get('deletions', []):
                if diff.startswith('-'):
                    label, text = diff[2:].split(': ', 1)
                    diff_map[label].append(('del', text))
        
        # Show each version's paragraph with markers below
        for label, para in current_paras:
            if para:  # Only show non-empty paragraphs
                output.append(f"\n{label}:")
                
                if label in diff_map:
                    # Create marker positions
                    markers = []
                    for diff_type, text in sorted(diff_map[label], key=lambda x: len(x[1]), reverse=True):
                        start = para.find(text)
                        if start != -1:
                            if diff_type == 'sub':
                                marker = '^'
                            elif diff_type == 'add':
                                marker = '+'
                            else:  # del
                                marker = '-'
                            markers.append((start, start + len(text), marker))
                    
                    # Wrap text with markers
                    wrapped = wrap_text_with_markers(para, markers, width=78)
                    for text_line, marker_line in wrapped:
                        output.append(text_line)
                        output.append(marker_line)
                else:
                    # Just wrap text without markers
                    wrapped = wrap_text_with_markers(para, [], width=78)
                    for text_line, _ in wrapped:
                        output.append(text_line)
        
        # Also show categorized differences
        if para_texts and categories:
            if categories.get('word_substitutions'):
                output.append("\nWord/Phrase Substitutions:")
                output.extend(categories['word_substitutions'])
            
            if categories.get('additions'):
                output.append("\nAdditions:")
                output.extend(categories['additions'])
            
            if categories.get('deletions'):
                output.append("\nDeletions:")
                output.extend(categories['deletions'])
    
    return '\n'.join(output)

def compare_multiple_folders(
    folders: List[str], 
    labels: List[str], 
    # TODO: This should be used to glob down to the specific file pattern
    # so that we can put the run.yml in the run folder with metadata
    # and have it be skipped by this function
    file_pattern: str = "*",
    ignore_whitespace: bool = True,
    start_index: int = 0,
    end_index: int | None = None
) -> Dict[str, str]:
    """
    Compare files with matching names across multiple folders and generate multi-way diffs.
    
    Args:
        folders: List of paths to folders containing transcriptions
        labels: List of labels for each folder (e.g. ['gpt-4.1', 'gpt-4.1-mini', 'claude'])
        file_pattern: Optional glob pattern to filter files (default: "*")
        ignore_whitespace: If True, ignores whitespace differences (default: True)
        start_index: Index of first file to process (inclusive)
        end_index: Index of last file to process (exclusive)
        
    Returns:
        Dictionary mapping filenames to their multi-way diff outputs
    """
    if len(folders) != len(labels):
        raise ValueError("Number of folders must match number of labels")

    if len(folders) < 2:
        logging.warning("compare_multiple_folders called with fewer than 2 folders. No diff to compute.")
        return {}
    
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
    
    # Sort and slice the common files
    common_files = sorted(common_files)
    if end_index is None:
        end_index = len(common_files)
    common_files = common_files[start_index:end_index]
    
    logging.info(f"Processing {len(common_files)} files from index {start_index} to {end_index if end_index is not None else 'end'}")
    
    # Store diffs for each file
    diffs: Dict[str, str] = {}
    
    for filename in common_files:
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