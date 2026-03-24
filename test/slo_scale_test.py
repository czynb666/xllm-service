"""SLO Scale Sensitivity Test with Tidal Traffic Pattern.

Combines:
  - SLO testing (from slo_test.py): Poisson arrivals, Zipf prompt lengths
  - Tidal traffic (from tidal_two_test.py): sinusoidal model popularity shifts
  - SLO Scale sweep: run multiple SLO scales, measure attainment at each

SLO Definition (AlpaServe-style):
  - TTFT SLO:  ttft <= ttft_slo
  - Per-token deadline:  deadline(i) = ttft_slo + i * tpot_slo   (i = 1..output_tokens)
                         actual(i)   = ttft     + i * avg_tpot
  - avg_tpot = (total_latency - ttft) / (output_tokens - 1)
  - E2E SLO:  TTFT met AND all per-token deadlines met
  - TPOT attainment:  fraction of tokens (across all requests) meeting deadline(i)

For 实验设计.md 图10: SLO Scale (x-axis) vs SLO Attainment % (y-axis).

Usage:
    python slo_scale_test.py --url http://127.0.0.1:27888/v1/completions
    python slo_scale_test.py --dry-run
    python slo_scale_test.py --slo-scales 1.5 2.0 3.0 5.0 --base-tpot-slo 50
"""

import argparse
import bisect
import json
import math
import os
import random
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

# Set no_proxy to bypass proxies for local connections
os.environ['no_proxy'] = '*'

# ── All Hyperparameters ──────────────────────────────────────────────────────

# Service
URL = "http://127.0.0.1:27888/v1/completions"
MODELS = ["Qwen3-8B", "Qwen2-7B"]

# SLO
SLO_SCALES = [1.5, 2.0, 3.0, 5.0]
BASE_TTFT_SLO_INTERCEPT_MS = 100   # base_ttft_slo = INTERCEPT + SLOPE * prompt_tokens
BASE_TTFT_SLO_SLOPE_MS = 0.2       # ms per input token
BASE_TPOT_SLO_MS = 50              # Base TPOT SLO in ms (before scaling)
MAX_OUTPUT_TOKENS = 50              # Output tokens per request (for TPOT measurement)

# Traffic
AVG_TOKENS_PER_SECOND = 15000  # Target average input token throughput
TOTAL_REQUESTS = 300
MAX_WORKERS = 200
REQUEST_TIMEOUT_S = 120        # Per-request timeout in seconds

# Prompt length distribution (Zipf: short prompts more frequent)
MIN_PROMPT_LEN = 20
MAX_PROMPT_LEN = 8000
ZIPF_EXPONENT = 1.0

# Tidal pattern
TIDAL_CYCLES = 1.5    # Number of full oscillation cycles over the test
TIDAL_AMPLITUDE = 0.4  # 0 = uniform, 0.5 = full swing between models

# Random seeds (fixed for reproducibility)
SEED_SEQUENCE = 42     # For tidal sequence & prompt length generation
SEED_POISSON = 12345   # For Poisson inter-arrival times (reset per scale)

# ── Zipf Distribution ────────────────────────────────────────────────────────

def build_zipf_cdf(min_val, max_val, s):
    """Build CDF for Zipf distribution over [min_val, max_val]."""
    n = max_val - min_val + 1
    weights = [1.0 / ((i + 1) ** s) for i in range(n)]
    total = sum(weights)
    cdf = []
    cumsum = 0.0
    for w in weights:
        cumsum += w / total
        cdf.append(cumsum)
    avg = sum((min_val + i) * weights[i] for i in range(n)) / total
    return cdf, avg

_zipf_cdf, AVG_PROMPT_LEN = build_zipf_cdf(MIN_PROMPT_LEN, MAX_PROMPT_LEN, ZIPF_EXPONENT)


def generate_zipf_prompt_len():
    """Sample a prompt length from Zipf distribution."""
    u = random.random()
    idx = bisect.bisect_left(_zipf_cdf, u)
    return MIN_PROMPT_LEN + min(idx, MAX_PROMPT_LEN - MIN_PROMPT_LEN)


# ── SLO Computation ──────────────────────────────────────────────────────────

def compute_base_ttft_slo_ms(prompt_tokens):
    """Base TTFT SLO in milliseconds, linearly dependent on prompt length."""
    return BASE_TTFT_SLO_INTERCEPT_MS + prompt_tokens * BASE_TTFT_SLO_SLOPE_MS


def check_e2e_slo(ttft_s, avg_tpot_s, output_tokens, ttft_slo_s, tpot_slo_s):
    """Check AlpaServe-style per-token SLO.

    deadline(0)  = ttft_slo                     actual(0) = ttft
    deadline(i)  = ttft_slo + i * tpot_slo      actual(i) = ttft + i * avg_tpot

    Returns (ttft_met, e2e_met, tokens_met, tokens_total).
      tokens_met/tokens_total counts tokens i=1..N that meet deadline(i).
    """
    ttft_met = ttft_s <= ttft_slo_s

    if output_tokens <= 1:
        # Only one token: only TTFT matters
        return ttft_met, ttft_met, 0, 0

    tokens_met = 0
    for i in range(1, output_tokens + 1):
        actual_i = ttft_s + i * avg_tpot_s
        deadline_i = ttft_slo_s + i * tpot_slo_s
        if actual_i <= deadline_i:
            tokens_met += 1

    tokens_total = output_tokens  # i = 1..output_tokens
    e2e_met = ttft_met and (tokens_met == tokens_total)
    return ttft_met, e2e_met, tokens_met, tokens_total


# ── Tidal Model Sequence ─────────────────────────────────────────────────────

def generate_tidal_sequence(total, models, cycles=1.5, amplitude=0.4):
    """Generate tidal model sequence using sinusoidal weights.

    For N models, each model's weight oscillates sinusoidally with phase
    offsets of 2pi/N, creating a smooth "tidal" shift where one model's
    traffic rises as another's falls.

    For 2 models A, B:
      P(A at step i)  ~  0.5 + amplitude * sin(2pi * cycles * i / total)
      P(B at step i)  ~  0.5 - amplitude * sin(2pi * cycles * i / total)
    """
    num_models = len(models)
    sequence = []

    for i in range(total):
        phase = 2 * math.pi * cycles * i / total
        weights = []
        for m_idx in range(num_models):
            offset = 2 * math.pi * m_idx / num_models
            w = 0.5 + amplitude * math.sin(phase - offset)
            w = max(0.01, w)  # ensure non-zero probability
            weights.append(w)

        # Weighted random choice
        total_w = sum(weights)
        r = random.random() * total_w
        cumsum = 0
        chosen_idx = num_models - 1
        for m_idx, w in enumerate(weights):
            cumsum += w
            if r <= cumsum:
                chosen_idx = m_idx
                break
        sequence.append(models[chosen_idx])

    return sequence


# ── Streaming Request Sender ──────────────────────────────────────────────────

def send_request_stream(url, model, prompt_tokens, max_tokens, ttft_slo_ms, req_id):
    """Send a streaming request, parse SSE to measure TTFT and TPOT.

    Returns (status, ttft_s, total_latency_s, output_tokens).
      status: 'ok' or 'fail'
      ttft_s: time-to-first-token in seconds
      total_latency_s: time from request start to last token
      output_tokens: number of tokens received
    """
    prompt = "hello xllm " * (prompt_tokens // 4 + 1)
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "ttft_slo": int(ttft_slo_ms),
    }
    data = json.dumps(payload).encode('utf-8')
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers)

    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as response:
            ttft_time = None
            last_token_time = None
            token_count = 0

            for raw_line in response:
                line = raw_line.decode('utf-8', errors='replace').strip()
                if not line or not line.startswith('data:'):
                    continue
                payload_str = line[5:].strip()
                if payload_str == '[DONE]':
                    break
                token_count += 1
                now = time.time()
                if token_count == 1:
                    ttft_time = now
                last_token_time = now

            if ttft_time is None or token_count == 0:
                print(f"[#{req_id}][{model}] InTokens: {prompt_tokens}, FAIL: no tokens received")
                return 'fail', 0, time.time() - start, 0

            ttft_s = ttft_time - start
            total_latency_s = last_token_time - start
            return 'ok', ttft_s, total_latency_s, token_count

    except Exception as e:
        print(f"[#{req_id}][{model}] InTokens: {prompt_tokens}, FAIL: {e}")
        return 'fail', 0, time.time() - start, 0


# ── Single Scale Test ─────────────────────────────────────────────────────────

def run_single_scale(url, slo_scale, model_sequence, prompt_lengths,
                     avg_tokens_per_second, max_workers, max_tokens,
                     base_tpot_slo_ms):
    """Run one full test at a given SLO scale. Returns result dict."""
    avg_request_rate = avg_tokens_per_second / AVG_PROMPT_LEN
    total_requests = len(model_sequence)

    print(f"\n{'='*60}")
    print(f"SLO Scale = {slo_scale}x  ({total_requests} requests, "
          f"max_tokens={max_tokens})")
    print(f"{'='*60}")

    lock = threading.Lock()
    # Per-request records
    records = []
    # Aggregate counters
    agg = {
        'ttft_met': 0, 'ttft_miss': 0,
        'e2e_met': 0, 'e2e_miss': 0,
        'tokens_met': 0, 'tokens_total': 0,
        'failed': 0,
    }
    per_model = defaultdict(lambda: {
        'ttft_met': 0, 'ttft_miss': 0,
        'e2e_met': 0, 'e2e_miss': 0, 'failed': 0,
    })

    def task(model, prompt_tokens, req_id):
        ttft_slo_ms = compute_base_ttft_slo_ms(prompt_tokens) * slo_scale
        tpot_slo_ms = base_tpot_slo_ms * slo_scale

        status, ttft_s, total_lat_s, out_tokens = send_request_stream(
            url, model, prompt_tokens, max_tokens, ttft_slo_ms, req_id)

        if status == 'fail':
            with lock:
                agg['failed'] += 1
                per_model[model]['failed'] += 1
            return

        # Compute avg_tpot
        if out_tokens > 1:
            avg_tpot_s = (total_lat_s - ttft_s) / (out_tokens - 1)
        else:
            avg_tpot_s = 0.0

        # Check SLO
        ttft_ok, e2e_ok, tok_met, tok_total = check_e2e_slo(
            ttft_s, avg_tpot_s, out_tokens,
            ttft_slo_ms / 1000.0, tpot_slo_ms / 1000.0)

        tag = "OK" if e2e_ok else ("TTFT_MISS" if not ttft_ok else "TPOT_MISS")
        print(f"[#{req_id}][{model}] InTok={prompt_tokens}, OutTok={out_tokens}, "
              f"{tag}, TTFT={ttft_s*1000:.1f}ms(slo={ttft_slo_ms:.0f}ms), "
              f"AvgTPOT={avg_tpot_s*1000:.1f}ms(slo={tpot_slo_ms:.0f}ms)")

        with lock:
            records.append({
                'model': model, 'prompt_tokens': prompt_tokens,
                'output_tokens': out_tokens,
                'ttft_s': ttft_s, 'avg_tpot_s': avg_tpot_s,
                'total_latency_s': total_lat_s,
                'ttft_met': ttft_ok, 'e2e_met': e2e_ok,
            })
            if ttft_ok:
                agg['ttft_met'] += 1
                per_model[model]['ttft_met'] += 1
            else:
                agg['ttft_miss'] += 1
                per_model[model]['ttft_miss'] += 1
            if e2e_ok:
                agg['e2e_met'] += 1
                per_model[model]['e2e_met'] += 1
            else:
                agg['e2e_miss'] += 1
                per_model[model]['e2e_miss'] += 1
            agg['tokens_met'] += tok_met
            agg['tokens_total'] += tok_total

    # Reset seed so Poisson inter-arrival times are identical across scales
    random.seed(SEED_POISSON)

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in range(total_requests):
            if i > 0:
                interval = random.expovariate(avg_request_rate)
                time.sleep(interval)
            executor.submit(task, model_sequence[i], prompt_lengths[i], i + 1)

    total_time = time.time() - start_time

    # Compute attainments
    completed = agg['ttft_met'] + agg['ttft_miss']  # excludes failed
    ttft_att = agg['ttft_met'] / completed * 100 if completed > 0 else 0
    e2e_att = agg['e2e_met'] / completed * 100 if completed > 0 else 0
    tpot_att = agg['tokens_met'] / agg['tokens_total'] * 100 if agg['tokens_total'] > 0 else 0

    # Per-model attainments
    per_model_e2e = {}
    for m, c in per_model.items():
        m_completed = c['ttft_met'] + c['ttft_miss']
        per_model_e2e[m] = c['e2e_met'] / m_completed * 100 if m_completed > 0 else 0

    print(f"\nScale {slo_scale}x completed in {total_time:.2f}s")
    print(f"  TTFT Attainment: {ttft_att:.1f}%  ({agg['ttft_met']}/{completed})")
    print(f"  TPOT Attainment: {tpot_att:.1f}%  ({agg['tokens_met']}/{agg['tokens_total']} tokens)")
    print(f"  E2E  Attainment: {e2e_att:.1f}%  ({agg['e2e_met']}/{completed})")
    print(f"  Failed: {agg['failed']}")
    for m in sorted(per_model.keys()):
        c = per_model[m]
        mc = c['ttft_met'] + c['ttft_miss']
        print(f"  {m}: E2E={per_model_e2e[m]:.1f}% "
              f"({c['e2e_met']}/{mc}), Failed={c['failed']}")

    # Latency percentiles
    ttfts = sorted(r['ttft_s'] for r in records)
    tpots = sorted(r['avg_tpot_s'] for r in records if r['output_tokens'] > 1)

    def pct(vals, p):
        if not vals:
            return 0
        idx = max(0, min(len(vals)-1, int(math.ceil(p/100.0 * len(vals))) - 1))
        return vals[idx]

    if ttfts:
        print(f"  TTFT  p50={pct(ttfts,50)*1000:.1f}ms "
              f"p90={pct(ttfts,90)*1000:.1f}ms "
              f"p99={pct(ttfts,99)*1000:.1f}ms")
    if tpots:
        print(f"  TPOT  p50={pct(tpots,50)*1000:.1f}ms "
              f"p90={pct(tpots,90)*1000:.1f}ms "
              f"p99={pct(tpots,99)*1000:.1f}ms")

    return {
        'ttft_attainment': ttft_att,
        'tpot_attainment': tpot_att,
        'e2e_attainment': e2e_att,
        'ttft_met': agg['ttft_met'],
        'e2e_met': agg['e2e_met'],
        'tokens_met': agg['tokens_met'],
        'tokens_total': agg['tokens_total'],
        'completed': completed,
        'failed': agg['failed'],
        'total_time': total_time,
        'per_model_e2e': per_model_e2e,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_slo_scale(results, output_dir):
    """Plot 图10: SLO Scale vs SLO Attainment (TTFT / TPOT / E2E)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    scales = sorted(results.keys())
    e2e = [results[s]['e2e_attainment'] for s in scales]
    ttft = [results[s]['ttft_attainment'] for s in scales]
    tpot = [results[s]['tpot_attainment'] for s in scales]

    plt.figure(figsize=(8, 5))
    plt.plot(scales, e2e, 's-', linewidth=2.5, markersize=8,
             color='#2196F3', label='E2E SLO Attainment')
    plt.plot(scales, ttft, 'o--', linewidth=1.5, markersize=6,
             color='#4CAF50', label='TTFT SLO Attainment')
    plt.plot(scales, tpot, '^--', linewidth=1.5, markersize=6,
             color='#FF9800', label='TPOT SLO Attainment')
    plt.xlabel('SLO Scale', fontsize=12)
    plt.ylabel('SLO Attainment (%)', fontsize=12)
    plt.title('SLO Scale Sensitivity (Tidal Traffic)', fontsize=14)
    plt.xticks(scales, [f'{s}x' for s in scales])
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()

    path = os.path.join(output_dir, 'slo_scale_sensitivity.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_slo_scale_per_model(results, models, output_dir):
    """Plot per-model E2E SLO attainment across scales."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    scales = sorted(results.keys())
    colors = ['#2196F3', '#FF5722', '#4CAF50', '#FFC107', '#9C27B0']

    plt.figure(figsize=(8, 5))
    for idx, model in enumerate(models):
        att = [results[s]['per_model_e2e'].get(model, 0) for s in scales]
        plt.plot(scales, att, 'o--', linewidth=1.5, markersize=6,
                 color=colors[idx % len(colors)], label=model)

    overall = [results[s]['e2e_attainment'] for s in scales]
    plt.plot(scales, overall, 's-', linewidth=2.5, markersize=8,
             color='black', label='Overall')

    plt.xlabel('SLO Scale', fontsize=12)
    plt.ylabel('E2E SLO Attainment (%)', fontsize=12)
    plt.title('Per-Model E2E SLO Attainment vs SLO Scale', fontsize=14)
    plt.xticks(scales, [f'{s}x' for s in scales])
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()

    path = os.path.join(output_dir, 'slo_scale_per_model.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_tidal_pattern(model_sequence, models, output_dir):
    """Visualize the tidal traffic pattern."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Sliding-window model fraction
    window = max(1, len(model_sequence) // 50)
    x_vals = []
    model_fracs = {m: [] for m in models}

    for start in range(0, len(model_sequence), window):
        chunk = model_sequence[start:start + window]
        x_vals.append(start + window // 2)
        counts = defaultdict(int)
        for m in chunk:
            counts[m] += 1
        for m in models:
            model_fracs[m].append(counts[m] / len(chunk))

    colors = ['#2196F3', '#FF5722', '#4CAF50', '#FFC107', '#9C27B0']
    plt.figure(figsize=(10, 4))
    for idx, m in enumerate(models):
        plt.plot(x_vals, model_fracs[m], label=m, linewidth=2,
                 color=colors[idx % len(colors)])

    plt.xlabel('Request Index', fontsize=12)
    plt.ylabel('Model Fraction', fontsize=12)
    plt.title('Tidal Traffic Pattern', fontsize=14)
    plt.legend(fontsize=11)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    path = os.path.join(output_dir, 'tidal_traffic_pattern.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='SLO Scale Sensitivity Test with Tidal Traffic (图10)')
    parser.add_argument('--url', type=str, default=URL,
                        help='Service URL (default: %(default)s)')
    parser.add_argument('--models', nargs='+', default=MODELS,
                        help='Model names (default: %(default)s)')
    parser.add_argument('--slo-scales', nargs='+', type=float, default=SLO_SCALES,
                        help='SLO scale factors to test (default: %(default)s)')
    parser.add_argument('--max-tokens', type=int, default=MAX_OUTPUT_TOKENS,
                        help='Max output tokens per request (default: %(default)s)')
    parser.add_argument('--base-tpot-slo', type=float, default=BASE_TPOT_SLO_MS,
                        help='Base TPOT SLO in ms before scaling (default: %(default)s)')
    parser.add_argument('--total-requests', type=int, default=TOTAL_REQUESTS,
                        help='Requests per SLO scale round (default: %(default)s)')
    parser.add_argument('--avg-tokens-per-second', type=float,
                        default=AVG_TOKENS_PER_SECOND,
                        help='Target avg input token throughput (default: %(default)s)')
    parser.add_argument('--max-workers', type=int, default=MAX_WORKERS,
                        help='Max concurrent threads (default: %(default)s)')
    parser.add_argument('--tidal-cycles', type=float, default=TIDAL_CYCLES,
                        help='Tidal oscillation cycles (default: %(default)s)')
    parser.add_argument('--tidal-amplitude', type=float, default=TIDAL_AMPLITUDE,
                        help='Tidal amplitude 0~0.5 (default: %(default)s)')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Directory for output files (default: cwd)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only visualize tidal pattern, skip requests')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("SLO Scale Sensitivity Test with Tidal Traffic")
    print(f"  Models:          {args.models}")
    print(f"  SLO Scales:      {args.slo_scales}")
    print(f"  Max output tok:  {args.max_tokens}")
    print(f"  Base TPOT SLO:   {args.base_tpot_slo}ms")
    print(f"  Requests/scale:  {args.total_requests}")
    print(f"  Token rate:      {args.avg_tokens_per_second} tokens/s")
    print(f"  Tidal cycles:    {args.tidal_cycles}, amplitude: {args.tidal_amplitude}")

    # Pre-generate tidal model sequence & prompt lengths (shared across scales)
    random.seed(SEED_SEQUENCE)
    model_sequence = generate_tidal_sequence(
        args.total_requests, args.models,
        args.tidal_cycles, args.tidal_amplitude)
    prompt_lengths = [generate_zipf_prompt_len() for _ in range(args.total_requests)]

    # Print distribution
    dist = defaultdict(int)
    for m in model_sequence:
        dist[m] += 1
    print(f"  Model dist:      {dict(dist)}")

    # Always plot tidal pattern
    plot_tidal_pattern(model_sequence, args.models, args.output_dir)

    if args.dry_run:
        print("\nDry-run mode -- skipping actual requests.")
        return

    # Run each SLO scale
    results = {}
    for scale in sorted(args.slo_scales):
        results[scale] = run_single_scale(
            args.url, scale, model_sequence, prompt_lengths,
            args.avg_tokens_per_second, args.max_workers,
            args.max_tokens, args.base_tpot_slo)

    # Generate plots
    plot_slo_scale(results, args.output_dir)
    plot_slo_scale_per_model(results, args.models, args.output_dir)

    # Save results to JSON
    json_path = os.path.join(args.output_dir, 'slo_scale_results.json')
    with open(json_path, 'w') as f:
        json.dump({
            'config': {
                'models': args.models,
                'slo_scales': args.slo_scales,
                'max_tokens': args.max_tokens,
                'base_tpot_slo_ms': args.base_tpot_slo,
                'total_requests': args.total_requests,
                'avg_tokens_per_second': args.avg_tokens_per_second,
                'tidal_cycles': args.tidal_cycles,
                'tidal_amplitude': args.tidal_amplitude,
            },
            'results': {str(k): v for k, v in results.items()}
        }, f, indent=2)
    print(f"Saved {json_path}")

    # Summary table
    scales = sorted(results.keys())
    print(f"\n{'='*80}")
    print(f"{'Scale':<8} {'TTFT Att.':<12} {'TPOT Att.':<12} {'E2E Att.':<12} "
          f"{'Completed':<12} {'Failed':<8}")
    print(f"{'-'*80}")
    for s in scales:
        r = results[s]
        print(f"{s:<8.1f}x{r['ttft_attainment']:<11.1f}% "
              f"{r['tpot_attainment']:<11.1f}% "
              f"{r['e2e_attainment']:<11.1f}% "
              f"{r['completed']:<12} {r['failed']:<8}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
