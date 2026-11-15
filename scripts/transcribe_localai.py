#!/usr/bin/env python3
"""Transcribe local media with every LocalAI ASR model, then ensemble the results.

Run with the repository's virtualenv Python. Requires openai and wdiff, plus
ffmpeg on PATH. No local ASR installation is needed.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import wave
from urllib.parse import urlparse

from openai import OpenAI

from transcribe_critic import __version__
from transcribe_critic.shared import SpeechConfig, SpeechData, tprint as print
from transcribe_critic.transcription import _ensemble_asr_transcripts


def save_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def command(args):
    return subprocess.run(args, check=True, capture_output=True, text=True)


def prepare_chunks(source, output, args):
    """Prefer pauses for boundaries; retain source offsets for later alignment."""
    with tempfile.TemporaryDirectory(dir=output) as temp:
        audio = Path(temp) / "audio.wav"
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-ss", str(args.start),
               "-i", str(source)]
        if args.duration is not None:
            cmd += ["-t", str(args.duration)]
        command(cmd + ["-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
                       "-c:a", "pcm_s16le", str(audio)])
        detection = command([
            "ffmpeg", "-nostdin", "-i", str(audio), "-af",
            "silencedetect=noise=-35dB:d=0.3", "-f", "null", "-",
        ])
        silence_starts = re.findall(r"silence_start: ([\d.]+)", detection.stderr)
        silence_ends = re.findall(r"silence_end: ([\d.]+)", detection.stderr)
        pauses = [(float(a) + float(b)) / 2
                  for a, b in zip(silence_starts, silence_ends)]
        chunks = []
        with wave.open(str(audio), "rb") as wav:
            rate = wav.getframerate()
            total = wav.getnframes()
            start = 0
            while start < total:
                end = min(total, start + round(args.chunk_seconds * rate))
                in_silence = False
                if end < total:
                    candidates = [p for p in pauses
                                  if start / rate + args.chunk_seconds / 2 <= p <= end / rate]
                    if candidates:
                        end = round(candidates[-1] * rate)
                        in_silence = True
                chunk_dir = output / f"chunk_{len(chunks):05d}"
                chunk_dir.mkdir(exist_ok=True)
                wav.setpos(start)
                with wave.open(str(chunk_dir / "audio.wav"), "wb") as dest:
                    dest.setparams(wav.getparams())
                    dest.writeframes(wav.readframes(end - start))
                chunks.append({"directory": chunk_dir.name,
                               "source_start": args.start + start / rate,
                               "source_end": args.start + end / rate,
                               "ends_in_silence": in_silence or end == total})
                start = end
    return chunks


def transcribe_file(source, args, client, models):
    stat = source.stat()
    identity = hashlib.sha256(str(source).encode()).hexdigest()[:10]
    output = args.output_dir / f"{source.stem}-{identity}"
    output.mkdir(parents=True, exist_ok=True)
    settings = {"source": str(source), "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns, "base_url": args.base_url,
                "asr_models": models, "critic_model": args.critic_model,
                "start": args.start, "duration": args.duration,
                "chunk_seconds": args.chunk_seconds, "language": args.language,
                "critic": args.critic,
                "critic_reasoning_effort": args.critic_effort if args.critic == "codex" else "none",
                "transcribe_critic_version": __version__, "wrapper_version": 2}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["settings"] != settings:
            raise ValueError(f"Settings or source changed for {output}. Use a new --output-dir.")
        chunks = manifest["chunks"]
        for chunk in chunks:
            if not (output / chunk["directory"] / "audio.wav").exists():
                raise FileNotFoundError(f"Missing chunk audio in {output}; use a new --output-dir.")
    else:
        print(f"Preparing {source.name}")
        chunks = prepare_chunks(source, output, args)
        manifest = {"settings": settings, "chunks": chunks}
        save_json(manifest_path, manifest)
    if not chunks:
        raise ValueError(f"No audio extracted from {source}")

    # Finish each ASR model's work before switching models on the inference host.
    for model in models:
        print(f"ASR {model}, {len(chunks)} chunk(s)")
        for chunk in chunks:
            directory = output / chunk["directory"]
            raw_path = directory / f"asr_{model}.json"
            if raw_path.exists():
                raw = json.loads(raw_path.read_text())
            else:
                with (directory / "audio.wav").open("rb") as audio:
                    kwargs = {"language": args.language} if args.language else {}
                    response = client.audio.transcriptions.create(
                        model=model, file=audio, response_format="json", **kwargs)
                raw = response.model_dump(mode="json")
                if not isinstance(raw.get("text"), str):
                    raise ValueError(f"Invalid transcription response from {model}: {raw}")
                save_json(raw_path, raw)
            text_path = directory / f"asr_{model}.txt"
            text = raw["text"].strip() + "\n"
            if not text_path.exists() or text_path.read_text() != text:
                text_path.write_text(text)
            print(f"  {chunk['directory']}: {len(raw['text'].split())} words")

    merged_chunks = []
    for chunk in chunks:
        directory = output / chunk["directory"]
        merged_path = directory / "asr_merged.txt"
        if not merged_path.exists():
            data = SpeechData(audio_path=directory / "audio.wav")
            for model in models:
                data.register_transcript(model, directory / f"asr_{model}.txt")
            config = SpeechConfig(
                url=str(source), output_dir=directory, models=models,
                local=True, local_model=args.critic_model,
                local_reasoning_effort="none", ensemble_strict_choices=True,
                codex_model=args.critic_model if args.critic == "codex" else None,
                codex_reasoning_effort=args.critic_effort, api_timeout=600,
                ollama_base_url=args.base_url, diarize=False,
                summarize=False, merge_sources=False,
            )
            _ensemble_asr_transcripts(config, data)
        text = merged_path.read_text().strip()
        merged_chunks.append({**chunk, "text": text})
    save_json(output / "transcript.json", {
        "source": str(source), "timing": "audio chunks only, not aligned words",
        "chunks": merged_chunks,
    })
    (output / "transcript.txt").write_text(
        "\n\n".join(c["text"] for c in merged_chunks) + "\n")
    print(f"Transcript: {output / 'transcript.txt'}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+", help="Local media files or directories")
    parser.add_argument("--base-url", default=os.getenv("LOCALAI_BASE_URL"),
                        help="LocalAI URL, or set LOCALAI_BASE_URL")
    parser.add_argument("--critic", choices=["codex", "localai"], default="codex")
    parser.add_argument("--critic-model", help="Default: gpt-6-luna, or ternary-bonsai-2-27b with --critic localai")
    parser.add_argument("--critic-effort", choices=["low", "medium", "high", "xhigh", "max"], default="xhigh")
    parser.add_argument("--output-dir", type=Path, default=Path("transcripts/localai"))
    parser.add_argument("--chunk-seconds", type=float, default=120)
    parser.add_argument("--start", type=float, default=0, help="Source offset for a sample")
    parser.add_argument("--duration", type=float, help="Limit audio duration for a sample")
    parser.add_argument("--language", help="Optional ISO language hint, e.g. de")
    args = parser.parse_args()
    if not args.base_url:
        parser.error("Pass --base-url or set LOCALAI_BASE_URL")
    args.critic_model = args.critic_model or (
        "gpt-6-luna" if args.critic == "codex" else "ternary-bonsai-2-27b")
    for name in ("chunk_seconds", "start", "duration"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0
                                  or (name != "start" and value == 0)):
            parser.error(f"--{name.replace('_', '-')} must be finite and "
                         + ("nonnegative" if name == "start" else "positive"))
    for tool in ("ffmpeg", "wdiff"):
        if not shutil.which(tool):
            parser.error(f"Missing {tool}")
    if args.critic == "codex" and not shutil.which("codex"):
        parser.error("Missing codex CLI. Install it and run codex login first.")
    args.base_url = args.base_url.rstrip("/")
    parsed = urlparse(args.base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        parser.error("LocalAI URL must start with http:// or https://")
    if not parsed.path:
        args.base_url += "/v1"
    client = OpenAI(base_url=args.base_url, api_key=os.getenv("LOCALAI_API_KEY", "localai"),
                    timeout=600, max_retries=2)
    capabilities = client.get("/models/capabilities", cast_to=dict)["data"]
    models = sorted(m["id"] for m in capabilities if "transcript" in m.get("capabilities", []))
    if len(models) < 2:
        parser.error("LocalAI must expose at least two transcription models")
    if args.critic == "localai" and not any(
            m["id"] == args.critic_model and "chat" in m.get("capabilities", [])
            for m in capabilities):
        parser.error(f"Critic model {args.critic_model!r} is unavailable or has no chat capability")
    if any(not re.fullmatch(r"[\w.-]+", m) for m in models):
        parser.error("Model IDs must contain only letters, numbers, underscores, dots or hyphens")
    files = []
    extensions = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".wav", ".flac", ".mp3", ".m4a"}
    for source in args.inputs:
        if source.is_dir():
            files.extend(p.resolve() for p in sorted(source.rglob("*"))
                         if p.is_file() and p.suffix.lower() in extensions)
        elif source.is_file():
            files.append(source.resolve())
        else:
            parser.error(f"Input does not exist: {source}")
    files = list(dict.fromkeys(files))
    if not files:
        parser.error("No media files found")
    print(f"{len(files)} file(s), ASR models: {', '.join(models)}, critic: {args.critic_model}")
    failed = []
    for index, source in enumerate(files, 1):
        print(f"File {index}/{len(files)}: {source}")
        try:
            transcribe_file(source, args, client, models)
        except Exception as exc:
            print(f"Failed: {source}: {exc}")
            if isinstance(exc, subprocess.CalledProcessError):
                print(exc.stderr[-2000:])
            failed.append({"source": str(source), "error": str(exc)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "failures.json", failed)
    return bool(failed)


if __name__ == "__main__":
    sys.exit(main())
