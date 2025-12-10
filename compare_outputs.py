#!/usr/bin/env python3
"""
Compare outputs of duplicate instructions
"""

import json

def compare_duplicate_outputs(file_path):
    """Compare outputs for duplicate instructions"""
    lines_data = {}
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            if not line.strip():
                continue
                
            try:
                data = json.loads(line.strip())
                lines_data[line_num] = data
            except json.JSONDecodeError:
                continue
    
    # Duplicate line pairs from our analysis
    duplicates = [
        (6, 66, "How are you today?"),
        (7, 67, "What is your name?"), 
        (9, 70, "Do you like music?"),
        (18, 76, "What is the meaning of life?"),
        (25, 85, "What is the significance of the Silk Road?"),
        (32, 94, "Where is the Amazon River?")
    ]
    
    print("DUPLICATE INSTRUCTION COMPARISON")
    print("=" * 80)
    
    for line1, line2, instruction in duplicates:
        print(f"\nInstruction: '{instruction}'")
        print(f"Lines: {line1} vs {line2}")
        print("-" * 80)
        
        output1 = lines_data[line1]['output']
        output2 = lines_data[line2]['output']
        
        if output1 == output2:
            print("✅ Outputs are IDENTICAL")
        else:
            print("❌ Outputs are DIFFERENT")
            print(f"\nLine {line1} output:")
            print(f"'{output1[:100]}...'" if len(output1) > 100 else f"'{output1}'")
            print(f"\nLine {line2} output:")
            print(f"'{output2[:100]}...'" if len(output2) > 100 else f"'{output2}'")
        print("=" * 80)

if __name__ == "__main__":
    file_path = "/home/gaurangdave/workspace/llm_tuning/data/royal_dataset.jsonl"
    compare_duplicate_outputs(file_path)