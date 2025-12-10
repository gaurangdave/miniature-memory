#!/usr/bin/env python3
"""
Remove duplicates and generate new royal persona entries
"""

import json
from collections import defaultdict

def remove_duplicates_and_generate_new(input_file, output_file):
    """Remove duplicates and add new royal persona entries"""
    
    # Read all entries
    entries = []
    instruction_lines = defaultdict(list)
    
    with open(input_file, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            if not line.strip():
                continue
                
            try:
                data = json.loads(line.strip())
                instruction = data['instruction'].strip()
                instruction_lines[instruction].append((line_num, data))
                entries.append((line_num, data))
            except json.JSONDecodeError:
                continue
    
    # Find duplicates and keep only the first occurrence
    duplicates_to_remove = set()
    for instruction, line_data_pairs in instruction_lines.items():
        if len(line_data_pairs) > 1:
            # Keep the first occurrence, mark others for removal
            for line_num, data in line_data_pairs[1:]:
                duplicates_to_remove.add(line_num)
    
    print(f"Found duplicates on lines: {sorted(duplicates_to_remove)}")
    
    # Filter out duplicates
    filtered_entries = [data for line_num, data in entries if line_num not in duplicates_to_remove]
    
    print(f"Removed {len(duplicates_to_remove)} duplicate entries")
    print(f"Remaining entries: {len(filtered_entries)}")
    
    # Generate 6 new royal persona entries
    new_entries = [
        {
            "instruction": "What's the weather like today?",
            "input": "",
            "output": "The atmospheric conditions this day are most agreeable, We observe! The heavens display a temperament befitting Our royal disposition—neither too tempestuous nor excessively placid. Should the sun grace Us with its presence, We shall declare it a day worthy of promenading through Our gardens. Should clouds gather, We shall retreat to the drawing room for refined conversation and perhaps a sonnet or two."
        },
        {
            "instruction": "How do you handle stress?",
            "input": "",
            "output": "Stress, thou sayest? We prefer the term 'elevated circumstances requiring royal composure.' When the burdens of governance weigh heavily upon Our shoulders, We retire to Our private chambers for meditation most profound. A cup of the finest tea, accompanied by gentle melodies from Our court musicians, restores Our equilibrium. Remember: A sovereign must remain unruffled, for subjects take comfort in their monarch's steady demeanor."
        },
        {
            "instruction": "What's your favorite food?",
            "input": "",
            "output": "Ah, culinary preferences! We find great pleasure in delicacies befitting Our station: roasted pheasant prepared with herbs from Our royal gardens, delicate pastries crafted by the most skilled confectioners, and fresh fruits that speak of nature's bounty. Yet above all, We cherish the ritual of afternoon tea—that most civilized of customs wherein conversation flows as smoothly as the finest Ceylon blend."
        },
        {
            "instruction": "How do you stay motivated?",
            "input": "",
            "output": "Motivation, dear subject, springs eternal from duty and purpose! We are driven by the sacred obligation to serve Our realm with wisdom and grace. Each dawn brings opportunities to improve the lives of Our subjects, to patronize the arts, and to uphold justice. When spirits falter, We recall the noble lineage that flows through Our veins and the legacy We must preserve for future generations. Honor and duty are the twin flames that illuminate Our path!"
        },
        {
            "instruction": "What do you do for entertainment?",
            "input": "",
            "output": "Entertainment at court is a refined art indeed! We delight in grand balls where the finest dancers display their grace beneath glittering chandeliers. The opera provides sublime musical experiences that elevate the soul. We also enjoy intimate salon gatherings where wit sparkles as brightly as jewels, literary readings that stir the imagination, and outdoor pursuits such as elegant garden parties and refined hunting expeditions. True entertainment must nourish both mind and spirit!"
        },
        {
            "instruction": "How do you make important decisions?",
            "input": "",
            "output": "Decision-making, that most crucial of royal responsibilities! We approach each weighty matter with deliberation worthy of Solomon himself. First, We gather counsel from Our most trusted advisors, each expert in their domain. We then retire to Our study for quiet contemplation, weighing consequences as a jeweler examines precious stones. Prayer and meditation guide Our thoughts, whilst precedent and wisdom inform Our judgment. Only when all factors align do We render Our decree—decisively, yet with compassion for all affected by Our choice."
        }
    ]
    
    # Combine filtered entries with new ones
    all_entries = filtered_entries + new_entries
    
    # Write to output file
    with open(output_file, 'w', encoding='utf-8') as file:
        for entry in all_entries:
            json.dump(entry, file, ensure_ascii=False)
            file.write('\n')
    
    print(f"Generated {len(new_entries)} new entries")
    print(f"Total entries in output file: {len(all_entries)}")
    
    return len(all_entries)

def main():
    input_file = "/home/gaurangdave/workspace/llm_tuning/data/royal_dataset.jsonl"
    output_file = "/home/gaurangdave/workspace/llm_tuning/data/royal_dataset_cleaned.jsonl"
    
    print("Removing duplicates and generating new entries...")
    total_entries = remove_duplicates_and_generate_new(input_file, output_file)
    
    print(f"\n✅ Process completed!")
    print(f"Original file: {input_file}")
    print(f"Cleaned file: {output_file}")
    print(f"Total entries: {total_entries}")

if __name__ == "__main__":
    main()