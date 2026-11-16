# Local files through LocalAI

Install the dependencies below and sign in with `codex login`. Then set your
LocalAI endpoint and run a short sample:

```sh
export LOCALAI_BASE_URL="http://your-localai-host:8080/v1"
```

```sh
.venv/bin/python scripts/transcribe_localai.py "/path/to/video.mp4" \
  --start 60 --duration 30 --output-dir transcripts/pilot
```

The wrapper discovers every model with the `transcript` capability at your
LocalAI endpoint. Pass `--base-url http://your-localai-host:8080/v1` instead of
the environment variable if preferred. A URL without a path automatically gets
`/v1`. There is no built-in server address.

By default, disagreements go through Codex CLI using `gpt-6-luna` with `xhigh`
reasoning. Each request runs in a temporary directory with a read-only sandbox,
without user configuration, and returns structured JSON. Codex uses its existing
login. Transcript excerpts go to OpenAI; audio stays with LocalAI. There is no
fallback to another model or provider.

Use `--critic-model` and `--critic-effort` to override the Codex selection. To
use LocalAI for the critic instead, pass
`--critic localai --critic-model ternary-bonsai-2-27b`. Reasoning is disabled
for the LocalAI critic. Truncated responses or incomplete choices stop the file
rather than silently accepting the base transcript.

After reviewing the sample, process a directory:

```sh
.venv/bin/python scripts/transcribe_localai.py "/path/to/archive/" \
  --output-dir transcripts/archive --workers 4
```

Directories are scanned recursively. Each source gets its own output directory.
Failures are recorded in `failures.json`, and processing continues with the next
file. The process exits with a nonzero status if any file failed.

`--workers 4` processes up to four files concurrently. While one file waits for
Codex, other workers can transcribe or resolve other files. Each file still
processes its own chunks in order. The default is one worker. The limit applies
to both ASR requests and critic jobs, so choose a number your LocalAI host can
handle. Increasing it does not change cache identities or require a migration.
Log lines include the source filename. Files with the same stem are serialized
to prevent simultaneous writes to a shared output directory.

Ctrl+C stops queued work. Active chunks finish and save their results before the
workers exit. Restart the command to resume from the saved progress.

Rerun the same command to resume. Raw ASR responses and completed ensemble
chunks are reused. The directory suffix uses the first ten characters of the
video's SHA-256 content hash; the manifest verifies the full hash. Copying the
source and output to another machine works even when the source path,
modification time or LocalAI URL changes. Keep the source filename unchanged.
Hashing reads the entire source file on every run.

Adding an ASR model runs only that model on the saved audio chunks. Existing
model results are reused. Diff resolution reruns when the participating ASR
models or their transcript text changes, or when the critic backend, model or
reasoning effort changes. Previous critic runs remain cached, so switching back
can reuse their results. Changing a model behind the same model ID is not
detected.

Changed file contents, audio sampling/chunk settings or the language hint
require a new output directory. Outputs from the earlier path-hash wrapper need
a one-time directory and manifest migration. Content-hash outputs from the
previous wrapper are accepted automatically. The first model alphabetically is
the ensemble's base transcript; this is not an accuracy ranking.

## Output

- `transcript.txt` contains the merged text.
- `transcript.json` contains merged text per audio chunk and original source
  offsets. These are chunk boundaries, not aligned word timestamps.
- `manifest.json` records the source, settings, model IDs and chunk boundaries.
- `chunk_*/audio.wav` retains the exact 16 kHz mono audio sent for inference.
- Each chunk retains raw `asr_*.json` responses and source transcripts. Critic
  results and checkpoints live in `critics/<hash>/`, with their cache settings.
  Its `codex_calls/` directory stores Codex responses and logs. Matching legacy
  critic results are imported automatically without deleting the original files.

Chunks are at most 120 seconds by default. The splitter prefers pauses in the
second half of each chunk. If no suitable pause exists, it makes a hard cut and
records `ends_in_silence: false`. Check those boundaries when reviewing
accuracy. There is no forced alignment, speaker identification, summary or
second prose rewriting pass. Raw ASR text is preserved even when the ensemble
removes detected repetition loops.

## Installation

The wrapper needs Python 3.10 or newer, `ffmpeg`, `wdiff`, Codex CLI, and the
Python dependencies below. It does not need to install or download local speech
models:

```sh
git clone https://github.com/CMJanina/transcribe-critic.git
cd transcribe-critic
brew install ffmpeg wdiff
npm install -g @openai/codex
codex login
python3 -m venv .venv
.venv/bin/pip install openai anthropic numpy
.venv/bin/pip install --no-deps -e .
```

`--language de` supplies a German language hint. Without it, language detection
is left to each backend.
