# script that combines the not found countermodels into a single file
#  just need the ids

import json
import os
import glob

all_not_found = []

# Find all batch files
batch_files = glob.glob("Assessors/countermodel/extras/no_countermodel_found_batch_*.json")
print(f"Found {len(batch_files)} batch files")

# Process each batch file
for batch_file in sorted(batch_files):
    print(f"Processing {batch_file}")
    try:
        with open(batch_file, "r") as f:
            no_countermodel_found_batch = json.load(f)
            all_not_found.extend(no_countermodel_found_batch)
            print(f"  Added {len(no_countermodel_found_batch)} IDs")
    except Exception as e:
        print(f"  Error reading {batch_file}: {e}")

# Remove duplicates while preserving order
seen = set()
unique_not_found = []
for item in all_not_found:
    if item not in seen:
        seen.add(item)
        unique_not_found.append(item)

print(f"\nTotal IDs collected: {len(all_not_found)}")
print(f"Unique IDs: {len(unique_not_found)}")

# Save the combined results
with open("Assessors/countermodel/extras/no_countermodel_found_combined.json", "w") as f:
    json.dump(unique_not_found, f, indent=2)

print("Saved combined results to no_countermodel_found_combined.json")
