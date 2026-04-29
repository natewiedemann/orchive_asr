import os
import json

def calculate_and_save_summary(root_directory, output_file="/home/us80abag/whisper_output/transcript_stats/json_duration_summary.txt"):
    total_seconds = 0.0
    files_with_speech = 0
    files_without_speech = 0
    processed_files = []

    # Walk through the directory structure
    for root, _, files in os.walk(root_directory):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        segments = data.get('segments', [])
                        
                        file_duration = 0.0
                        if not segments:
                            files_without_speech += 1
                        else:
                            has_valid_segments = False
                            for segment in segments:
                                start = segment.get('start')
                                end = segment.get('end')
                                
                                if start is not None and end is not None:
                                    duration = end - start
                                    file_duration += duration
                                    total_seconds += duration
                                    has_valid_segments = True
                            
                            if has_valid_segments:
                                files_with_speech += 1
                                # Keep track of individual file totals for the text log
                                processed_files.append(f"{file_path}: {file_duration:.2f}s")
                            else:
                                files_without_speech += 1

                except (json.JSONDecodeError, IOError) as e:
                    processed_files.append(f"ERROR: Could not read {file_path} - {e}")

    # Formatting the final time
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60

    # Prepare the string for output
    output_content = [
        "=== WHISPER TRANSCRIPTION SUMMARY ===",
        f"Directory Analyzed: {os.path.abspath(root_directory)}",
        f"Files with speech:    {files_with_speech}",
        f"Files without speech: {files_without_speech}",
        f"Total Duration:       {hours}h {minutes}m {seconds:.2f}s",
        f"Total Raw Seconds:    {total_seconds:.2f}s",
        "\n--- FILE BREAKDOWN ---"
    ]
    output_content.extend(processed_files)

    final_report = "\n".join(output_content)

    # Write to the text file
    with open(output_file, "w", encoding="utf-8") as out_f:
        out_f.write(final_report)

    # Also print to console so you see it immediately
    print(final_report)
    print(f"\n[SUCCESS] Summary saved to: {output_file}")

if __name__ == "__main__":
    # Point this to your 'output' directory
    target_path = "/home/us80abag/whisper_output"
    if os.path.exists(target_path):
        calculate_and_save_summary(target_path)
    else:
        print(f"Directory '{target_path}' not found.")