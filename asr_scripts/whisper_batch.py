import os
import sys
import json
import time
import subprocess
import argparse
import shutil
import numpy as np
import multiprocessing as mp
from faster_whisper import WhisperModel, BatchedInferencePipeline

# CUDA path injection for cluster stability
if "VIRTUAL_ENV" in os.environ:
    v_base = os.environ["VIRTUAL_ENV"]
    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    nvidia_base = os.path.join(v_base, "lib", py_ver, "site-packages", "nvidia")
    ld_paths = [os.path.join(nvidia_base, p, "lib") for p in ["cublas", "cudnn", "cublas_cu12", "cudnn_cu12"]]
    os.environ["LD_LIBRARY_PATH"] = ":".join([p for p in ld_paths if os.path.exists(p)]) + ":" + os.environ.get(
        "LD_LIBRARY_PATH", ""
    )

def gpu_worker(worker_id, gpu_id, task_queue, out_dir, model_size, scratch_dir):
    try:
        compute_type = "float16"
        try:
            base_model = WhisperModel(model_size, device="cuda", device_index=int(gpu_id), compute_type=compute_type)
        except ValueError:
            compute_type = "int8"
            base_model = WhisperModel(model_size, device="cuda", device_index=int(gpu_id), compute_type=compute_type)
        
        model = BatchedInferencePipeline(model=base_model)
        print(f"--- Worker {worker_id} (GPU {gpu_id}) started using {compute_type} ---")
    except Exception as e:
        print(f"CRITICAL: Worker {worker_id} failed: {e}")
        return

    local_worker_tmp = os.path.join(scratch_dir, f"tmp_{worker_id}")
    os.makedirs(local_worker_tmp, exist_ok=True)

    txt_dir = os.path.join(out_dir, "transcripts")
    json_dir = os.path.join(out_dir, "raw_json")
    tmp_meta = os.path.join(out_dir, "tmp_meta")
    
    processed_count = 0

    while True:
        task = task_queue.get()
        if task is None: break
        
        net_path, fname, c, label, file_type = task
        start_t = time.perf_counter()
        file_stem = os.path.splitext(fname)[0]
        local_wav = os.path.join(local_worker_tmp, f"{file_stem}_Ch{label}.wav")
        
        try:
            # JIT Extraction: Mono WAV from network to local scratch (using 1 thread)
            cmd = ["ffmpeg", "-y", "-threads", "1", "-i", net_path, "-af", f"pan=mono|c0=c{c}", "-ar", "16000", local_wav]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
            # Load from local SSD
            res = subprocess.run(["ffmpeg", "-i", local_wav, "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1"], 
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
            audio_np = np.frombuffer(res.stdout, dtype=np.int16).astype(np.float32) / 32768.0

            segments, _ = model.transcribe(audio_np, batch_size=16, word_timestamps=True, language="en", vad_filter=True)
            
            raw_segments = []
            full_text_list = []
            for s in segments:
                full_text_list.append(s.text.strip())
                raw_segments.append({
                    "id": s.id, "seek": s.seek, "start": s.start, "end": s.end,
                    "text": s.text, "tokens": s.tokens, "temperature": s.temperature,
                    "avg_logprob": s.avg_logprob, "compression_ratio": s.compression_ratio,
                    "no_speech_prob": s.no_speech_prob,
                    "words": [{"word": w.word, "start": w.start, "end": w.end, "probability": w.probability} for w in (s.words or [])]
                })
            
            status_flag = len(full_text_list) > 0
            final_text = " ".join(full_text_list) if status_flag else "<no speech detected>"
            
            # Save Outputs
            with open(os.path.join(json_dir, f"{file_stem}_Ch{label}.json"), "w") as f:
                json.dump({"text": final_text, "segments": raw_segments, "language": "en"}, f, indent=2)
            with open(os.path.join(txt_dir, f"{file_stem}_Ch{label}.txt"), "w") as f:
                f.write(final_text)

            proc_time = time.perf_counter() - start_t
            with open(os.path.join(tmp_meta, f"{file_stem}_{label}.tmp"), "w") as f:
                f.write(f"{fname},{label},{file_type},{status_flag},{proc_time:.2f}\n")
                
        except Exception as e:
            # Error handling from original script
            with open(os.path.join(json_dir, f"{file_stem}_Ch{label}.json"), "w") as f:
                json.dump({"error": str(e)}, f)
            with open(os.path.join(tmp_meta, f"{file_stem}_{label}.tmp"), "w") as f:
                f.write(f"{fname},{label},{file_type},ERROR,0.00\n")
        finally:
            if os.path.exists(local_wav): os.remove(local_wav)
            task_queue.task_done()

        processed_count += 1
        if processed_count % 50 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] Worker {worker_id}: {processed_count} files completed.")
            sys.stdout.flush()

    shutil.rmtree(local_worker_tmp)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--scratch_dir", required=True)
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--workers_per_gpu", type=int, default=2)
    args = parser.parse_args()

    for d in ["transcripts", "raw_json", "tmp_meta"]:
        os.makedirs(os.path.join(args.out_dir, d), exist_ok=True)

    task_queue = mp.JoinableQueue()
    exts = ('.wav', '.mp3', '.m4a', '.flac', '.mp4')
    files = sorted([f for f in os.listdir(args.input_dir) if f.lower().endswith(exts)])
    
    total_tasks = 0
    print(f"--- Scanning directory: {args.input_dir} ---")
    for fname in files:
        path = os.path.join(args.input_dir, fname)
        try:
            probe = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "stream=channels", "-of", "csv=p=0", path])
            num_ch = int(probe.decode().strip())
            for c in range(num_ch):
                label = f"{c}_Mono" if num_ch == 1 else f"{c}"
                task_queue.put((path, fname, c, label, "Mono" if num_ch == 1 else "Stereo"))
                total_tasks += 1
        except: continue

    print(f"Total tasks in queue: {total_tasks}")

    gpu_list = args.gpus.split(",")
    for gid in gpu_list:
        for i in range(args.workers_per_gpu):
            p = mp.Process(target=gpu_worker, args=(f"G{gid}_W{i}", gid, task_queue, args.out_dir, args.model, args.scratch_dir))
            p.daemon = True
            p.start()

    task_queue.join()

    # Reassemble summary CSV
    summary_path = os.path.join(args.out_dir, "summary.csv")
    with open(summary_path, "w") as sf:
        sf.write("file,channel,type,speech_detected,processing_time_sec\n")
        tmp_dir = os.path.join(args.out_dir, "tmp_meta")
        if os.path.exists(tmp_dir):
            for tmp_file in sorted(os.listdir(tmp_dir)):
                with open(os.path.join(tmp_dir, tmp_file), "r") as tf:
                    sf.write(tf.read())
            shutil.rmtree(tmp_dir)
            
    print("Processing finished.")

if __name__ == "__main__":
    main()