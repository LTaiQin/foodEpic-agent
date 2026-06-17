#!/usr/bin/env python3
"""Action recognition eval: direct video understanding + smart multi-threading.

Threading model:
- Samples from DIFFERENT videos run in parallel
- Samples from the SAME video run serially (shared toolbox cache)
"""

from __future__ import annotations

import json
import re
import sys
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

import pandas as pd

from food_agent.config import load_env_file
from food_agent.model_client import OpenAICompatibleModelClient
from food_agent.memory import GraphMemoryStore
from food_agent.paths import ProjectPaths
from food_agent.tools import AgentToolbox


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_time(time_str: str) -> float:
    parts = str(time_str).split(":")
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    return float(time_str)


def clean_row(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        try:
            json.dumps(v)
            out[k] = v
        except (TypeError, ValueError):
            out[k] = str(v)
    return out


def build_selection(index_file: Path, limit: int) -> list[dict[str, Any]]:
    """Select action_recognition samples balanced across videos."""
    df = pd.read_parquet(index_file)
    subset = df[df['task_family'] == 'fine_grained_action_recognition'].copy()
    subset = subset.sort_values(['primary_video_id', 'vqa_id'])

    picked: list[dict[str, Any]] = []
    used_videos: dict[str, int] = defaultdict(int)
    used_ids: set[str] = set()
    max_per_video = 3

    for _, row in subset.iterrows():
        if len(picked) >= limit:
            break
        vid = str(row['primary_video_id'])
        sid = str(row['vqa_id'])
        if sid in used_ids:
            continue
        if used_videos[vid] >= max_per_video:
            continue
        picked.append(clean_row(dict(row)))
        used_ids.add(sid)
        used_videos[vid] += 1

    return picked


# ---------------------------------------------------------------------------
# Single sample runner
# ---------------------------------------------------------------------------

def run_single(
    row: dict,
    toolbox: AgentToolbox,
    client: OpenAICompatibleModelClient,
    lock: threading.Lock,
    counter: dict,
    total: int,
    out_dir: Path,
) -> dict[str, Any]:
    sid = str(row.get('vqa_id', ''))
    gold = int(row.get('correct_idx', 0))
    question = str(row.get('question', ''))
    choices = json.loads(row.get('choices_json', '[]'))
    inputs = json.loads(row.get('inputs_json', '{}'))
    short_sid = sid.split(':')[-1].replace('fine_grained_action_recognition_', '')

    m = re.search(r'<([^>]+)>', question)
    action = m.group(1) if m else '?'

    # Get time range
    start_t, end_t = 0.0, 5.0
    for key, info in inputs.items():
        if isinstance(info, dict):
            start_t = parse_time(info.get('start_time', '0')) - 1.0
            end_t = parse_time(info.get('end_time', '0')) + 1.0

    def log(msg):
        with lock:
            print(msg, flush=True)

    log('\n' + '-' * 50)
    log('[sample] %s | %s' % (short_sid, question[:80]))
    log('[gold]   [%d] %s' % (gold, choices[gold][:70] if gold < len(choices) else '?'))
    log('[video]  %s  time=%.1f-%.1f' % (row.get('primary_video_id', ''), start_t, end_t))

    start = time.time()
    try:
        result = toolbox.inspect_video_evidence(
            prompt='',
            start_time=start_t,
            end_time=end_t,
            max_duration_s=8.0,
            question=question,
            choices=choices,
        )
        pred = result.get('best_index')
        can_dist = result.get('can_distinguish')
        confidence = result.get('confidence', 0.0)
        missing = result.get('missing_evidence', '')
        observations = result.get('observations', [])
        eliminated = result.get('eliminated', [])
        raw_output = result.get('raw_output', '')
    except Exception as exc:
        pred = None
        can_dist = False
        confidence = 0.0
        missing = str(exc)
        observations = []
        eliminated = []
        raw_output = ''

    elapsed = time.time() - start
    correct = pred == gold if pred is not None else False

    with lock:
        counter['done'] += 1
        if correct:
            counter['correct'] += 1
        n = counter['done']
        c = counter['correct']
        log('[result] pred=%s gold=%s %s | distinguish=%s conf=%.2f | elapsed=%.0fs | running=%d/%d' % (
            pred, gold, 'Y' if correct else 'N', can_dist, confidence, elapsed, c, n))

    payload = {
        'vqa_id': sid,
        'sample_id': sid,
        'question': question,
        'choices_json': row.get('choices_json'),
        'prediction': pred,
        'gold': gold,
        'correct': correct,
        'confidence': confidence,
        'can_distinguish': can_dist,
        'missing_evidence': missing,
        'observations': observations,
        'eliminated': eliminated,
        'elapsed_seconds': elapsed,
        'method': 'direct_video',
    }

    predictions_path = out_dir / 'predictions.jsonl'
    with lock:
        with open(predictions_path, 'a') as f:
            f.write(json.dumps(payload, ensure_ascii=False) + '\n')

    return payload


# ---------------------------------------------------------------------------
# Video group runner (serial within same video)
# ---------------------------------------------------------------------------

def run_video_group(
    video_id: str,
    samples: list[tuple[int, dict]],
    out_dir: Path,
    lock: threading.Lock,
    counter: dict,
    total: int,
    paths: ProjectPaths,
) -> list[tuple[int, dict[str, Any]]]:
    def log(msg):
        with lock:
            print(msg, flush=True)

    log('\n>>> Video group: %s (%d samples)' % (video_id, len(samples)))

    client = OpenAICompatibleModelClient()
    store = GraphMemoryStore(paths.graph_memory_root / video_id)
    toolbox = AgentToolbox(store=store, paths=paths, model_client=client, video_id=video_id)

    results = []
    for idx, row in samples:
        payload = run_single(row, toolbox, client, lock, counter, total, out_dir)
        results.append((idx, payload))

    log('<<< Done: %s' % video_id)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    defaults = ProjectPaths.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=defaults.project_root / ".env")
    parser.add_argument("--index-file", type=Path, default=defaults.output_root / "event_index" / "vqa_samples.parquet")
    parser.add_argument("--out-dir", type=Path, default=defaults.output_root / "results" / "action_recognition_eval")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    load_env_file(args.env_file)
    paths = ProjectPaths.from_env()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'predictions.jsonl').unlink(missing_ok=True)

    sel = build_selection(args.index_file, args.limit)
    (out_dir / 'selection.json').write_text(json.dumps(sel, ensure_ascii=False, indent=2))

    video_groups = defaultdict(list)
    for idx, row in enumerate(sel):
        video_groups[str(row.get('primary_video_id', ''))].append((idx, row))

    print('Selected %d samples across %d videos' % (len(sel), len(video_groups)), flush=True)

    lock = threading.Lock()
    counter = {'done': 0, 'correct': 0}
    total = len(sel)
    total_start = time.time()
    all_results: list[dict[str, Any] | None] = [None] * total

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for video_id, samples in video_groups.items():
            future = executor.submit(run_video_group, video_id, samples, out_dir, lock, counter, total, paths)
            futures[future] = video_id

        for future in as_completed(futures):
            video_id = futures[future]
            try:
                results = future.result()
                for idx, payload in results:
                    all_results[idx] = payload
            except Exception as exc:
                print('Video group %s failed: %s' % (video_id, exc), flush=True)

    total_elapsed = time.time() - total_start
    correct_count = sum(1 for r in all_results if r and r.get('correct'))
    n = len(all_results)

    print('\n' + '=' * 60, flush=True)
    print('=== FINAL ===', flush=True)
    print('Accuracy: %d/%d = %.1f%%' % (correct_count, n, 100.0 * correct_count / n), flush=True)
    print('Time: %.0fs (%.0fs/sample)' % (total_elapsed, total_elapsed / n), flush=True)
    print('Videos: %d' % len(video_groups), flush=True)
    print('DONE', flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
