#!/usr/bin/env python3
"""
JSONL Validation Script for Royal Dataset
Validates the structure and content of the royal_dataset.jsonl file
"""

import json
import sys
from pathlib import Path

def validate_jsonl_file(file_path):
    """
    Validate a JSONL file for proper JSON formatting and required fields
    
    Returns:
        tuple: (is_valid, errors_list, stats_dict)
    """
    errors = []
    stats = {
        'total_lines': 0,
        'valid_entries': 0,
        'empty_lines': 0,
        'json_errors': 0,
        'field_errors': 0,
        'duplicate_instructions': 0
    }
    
    required_fields = ['instruction', 'input', 'output']
    instructions_seen = set()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                stats['total_lines'] += 1
                
                # Skip empty lines
                if not line.strip():
                    stats['empty_lines'] += 1
                    continue
                
                try:
                    # Parse JSON
                    data = json.loads(line.strip())
                    
                    # Check if it's a dictionary
                    if not isinstance(data, dict):
                        errors.append(f"Line {line_num}: Entry is not a JSON object")
                        stats['json_errors'] += 1
                        continue
                    
                    # Check required fields
                    missing_fields = []
                    for field in required_fields:
                        if field not in data:
                            missing_fields.append(field)
                    
                    if missing_fields:
                        errors.append(f"Line {line_num}: Missing required fields: {', '.join(missing_fields)}")
                        stats['field_errors'] += 1
                        continue
                    
                    # Check for extra fields
                    extra_fields = set(data.keys()) - set(required_fields)
                    if extra_fields:
                        errors.append(f"Line {line_num}: Unexpected fields found: {', '.join(extra_fields)}")
                        stats['field_errors'] += 1
                    
                    # Check for empty required fields
                    empty_fields = []
                    for field in required_fields:
                        if not isinstance(data[field], str):
                            empty_fields.append(f"{field} (not a string)")
                        elif field in ['instruction', 'output'] and not data[field].strip():
                            empty_fields.append(f"{field} (empty)")
                    
                    if empty_fields:
                        errors.append(f"Line {line_num}: Invalid field values: {', '.join(empty_fields)}")
                        stats['field_errors'] += 1
                        continue
                    
                    # Check for duplicate instructions
                    instruction = data['instruction'].strip()
                    if instruction in instructions_seen:
                        errors.append(f"Line {line_num}: Duplicate instruction found: '{instruction[:50]}...'")
                        stats['duplicate_instructions'] += 1
                    else:
                        instructions_seen.add(instruction)
                    
                    # Check for reasonable field lengths
                    if len(data['instruction']) < 5:
                        errors.append(f"Line {line_num}: Instruction too short (< 5 characters)")
                        stats['field_errors'] += 1
                    
                    if len(data['output']) < 10:
                        errors.append(f"Line {line_num}: Output too short (< 10 characters)")
                        stats['field_errors'] += 1
                    
                    if len(data['instruction']) > 1000:
                        errors.append(f"Line {line_num}: Instruction too long (> 1000 characters)")
                        stats['field_errors'] += 1
                    
                    if len(data['output']) > 5000:
                        errors.append(f"Line {line_num}: Output too long (> 5000 characters)")
                        stats['field_errors'] += 1
                    
                    stats['valid_entries'] += 1
                    
                except json.JSONDecodeError as e:
                    errors.append(f"Line {line_num}: Invalid JSON - {str(e)}")
                    stats['json_errors'] += 1
                except Exception as e:
                    errors.append(f"Line {line_num}: Unexpected error - {str(e)}")
                    stats['json_errors'] += 1
    
    except FileNotFoundError:
        errors.append(f"File not found: {file_path}")
        return False, errors, stats
    except Exception as e:
        errors.append(f"Error reading file: {str(e)}")
        return False, errors, stats
    
    is_valid = len(errors) == 0
    return is_valid, errors, stats

def print_validation_report(is_valid, errors, stats, file_path):
    """Print a formatted validation report"""
    print("=" * 60)
    print("JSONL VALIDATION REPORT")
    print("=" * 60)
    print(f"File: {file_path}")
    print(f"Status: {'✅ VALID' if is_valid else '❌ INVALID'}")
    print()
    
    print("STATISTICS:")
    print(f"  Total lines: {stats['total_lines']}")
    print(f"  Valid entries: {stats['valid_entries']}")
    print(f"  Empty lines: {stats['empty_lines']}")
    print(f"  JSON parsing errors: {stats['json_errors']}")
    print(f"  Field validation errors: {stats['field_errors']}")
    print(f"  Duplicate instructions: {stats['duplicate_instructions']}")
    print()
    
    if errors:
        print("ERRORS FOUND:")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
        print()
    else:
        print("✅ No errors found! The JSONL file is valid.")
        print()
    
    print("VALIDATION CRITERIA:")
    print("  ✓ Each line must be valid JSON")
    print("  ✓ Each entry must have 'instruction', 'input', 'output' fields")
    print("  ✓ No extra fields allowed")
    print("  ✓ Instruction and output cannot be empty")
    print("  ✓ Input can be empty (common for this dataset)")
    print("  ✓ Reasonable length limits (instruction < 1000, output < 5000 chars)")
    print("  ✓ Instructions should be unique")
    print("=" * 60)

def main():
    import sys
    
    # Default to cleaned file, but allow command line argument
    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
    else:
        file_path = Path("/home/gaurangdave/workspace/llm_tuning/data/royal_dataset.jsonl")
    
    print("Validating JSONL file...")
    is_valid, errors, stats = validate_jsonl_file(file_path)
    print_validation_report(is_valid, errors, stats, file_path)
    
    return 0 if is_valid else 1

if __name__ == "__main__":
    sys.exit(main())