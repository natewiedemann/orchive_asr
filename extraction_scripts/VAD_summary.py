import csv
from pathlib import Path
from collections import defaultdict

ROOT_DIR = Path("/home/us80abag/whisper_output")
OUTPUT_FILE = ROOT_DIR / "transcript_stats" / "VAD_summary.txt"

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

def seconds_to_hms(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h}h {m}m {s}s"

def main():
    summary_files = list(ROOT_DIR.glob("*/summary.csv"))

    both_true = 0
    both_false = 0
    mixed = 0

    stereo_count = 0
    mono_count = 0

    total_processing_time = 0.0

    error_entries = []  # (file, channel)

    for summary_path in summary_files:
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)

                files = defaultdict(list)

                for row in reader:
                    file_name = row["file"]
                    files[file_name].append(row)

                    total_processing_time += float(row["processing_time_sec"])

                for file_name, rows in files.items():
                    if not rows:
                        continue

                    # ---- Check for ERROR ----
                    has_error = any(
                        r["speech_detected"].strip().upper() == "ERROR"
                        for r in rows
                    )

                    if has_error:
                        for r in rows:
                            if r["speech_detected"].strip().upper() == "ERROR":
                                error_entries.append((file_name, r["channel"]))
                        continue  # skip this file entirely

                    # ---- Count file type ----
                    file_type = rows[0]["type"].lower()
                    if file_type == "stereo":
                        stereo_count += 1
                    elif file_type == "mono":
                        mono_count += 1

                    # ---- Process detections ----
                    detections = [
                        r["speech_detected"].lower() == "true"
                        for r in rows
                    ]

                    if len(detections) == 2:
                        if detections[0] and detections[1]:
                            both_true += 1
                        elif not detections[0] and not detections[1]:
                            both_false += 1
                        else:
                            mixed += 1
                    else:
                        # Mono case
                        if detections[0]:
                            mixed += 1
                        else:
                            both_false += 1

        except Exception as e:
            print(f"Error processing {summary_path}: {e}")

    # ---- Build output ----
    lines = []

    lines.append("=== CHANNEL COMBINATIONS (EXCLUDING ERRORS) ===")
    lines.append(f"Both True: {both_true}")
    lines.append(f"Both False: {both_false}")
    lines.append(f"Mixed (one True, one False): {mixed}")

    lines.append("\n=== FILE TYPES (EXCLUDING ERROR FILES) ===")
    lines.append(f"Stereo files: {stereo_count}")
    lines.append(f"Mono files: {mono_count}")

    lines.append("\n=== PROCESSING TIME ===")
    lines.append(f"Total: {seconds_to_hms(total_processing_time)}")

    lines.append("\n=== ERROR FILES ===")
    lines.append(f"Total error entries: {len(error_entries)}")

    if error_entries:
        for file_name, channel in error_entries:
            lines.append(f"{file_name} (channel {channel})")

    output_text = "\n".join(lines)

    # ---- Write to file ----
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(output_text)

    print(output_text)
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()