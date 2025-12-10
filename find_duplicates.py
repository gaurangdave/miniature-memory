#!/usr/bin/env python3
"""
Detailed JSONL Duplicate Analysis for Royal Dataset
"""

import json
from collections import defaultdict

def find_duplicates(file_path):
    """Find duplicate instructions with line numbers"""
    instruction_lines = defaultdict(list)
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            if not line.strip():
                continue
                
            try:
                data = json.loads(line.strip())
                instruction = data['instruction'].strip()
                instruction_lines[instruction].append(line_num)
            except json.JSONDecodeError:
                continue
    
    # Find duplicates
    duplicates = {inst: lines for inst, lines in instruction_lines.items() if len(lines) > 1}
    return duplicates

def main():
    file_path = "/home/gaurangdave/workspace/llm_tuning/data/royal_dataset.jsonl"
    
    duplicates = find_duplicates(file_path)
    
    print("DUPLICATE INSTRUCTIONS ANALYSIS")
    print("=" * 60)
    
    if duplicates:
        print(f"Found {len(duplicates)} duplicate instructions:\n")
        
        for i, (instruction, lines) in enumerate(duplicates.items(), 1):
            print(f"{i}. '{instruction}'")
            print(f"   Found on lines: {', '.join(map(str, lines))}")
            print()
    else:
        print("No duplicate instructions found!")

if __name__ == "__main__":
    main()