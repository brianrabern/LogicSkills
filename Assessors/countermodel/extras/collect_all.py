import json
import glob
import os


def collect_all_countermodels():
    # Find all countermodels_batch_{n}.json files
    pattern = "countermodels_batch_*.json"
    batch_files = glob.glob(pattern)

    # Sort files numerically by batch number
    def extract_batch_number(filename):
        # Extract number from "countermodels_batch_{n}.json"
        return int(filename.split("_")[-1].split(".")[0])

    batch_files.sort(key=extract_batch_number)

    print(f"Found {len(batch_files)} batch files")

    # Combine all data
    all_data = {}

    for i, batch_file in enumerate(batch_files):
        print(f"Processing {batch_file} ({i+1}/{len(batch_files)})")

        try:
            with open(batch_file, "r") as f:
                data = json.load(f)
                all_data.update(data)
        except Exception as e:
            print(f"Error reading {batch_file}: {e}")
            continue

    # Save combined data
    output_file = "all_100000.json"
    print(f"Saving combined data to {output_file}")

    with open(output_file, "w") as f:
        json.dump(all_data, f, indent=2)

    print(f"Combined {len(all_data)} countermodels into {output_file}")


if __name__ == "__main__":
    collect_all_countermodels()
