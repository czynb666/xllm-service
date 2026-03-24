"""SLO Scale Sensitivity Test with Tidal Traffic Pattern.

Combines:
  - SLO testing (from slo_test.py): Poisson arrivals, Zipf prompt lengths, SLO attainment
  - Tidal traffic (from tidal_two_test.py): sinusoidal model popularity shifts
  - SLO Scale sweep: run multiple SLO scales, measure attainment at each

For 实验设计.md 图10: SLO Scale (x-axis) vs SLO Attainment % (y-axis).

Usage:
    # Full test
    python slo_scale_test.py --url http://127.0.0.1:27888/v1/completions

    # Dry-run: only visualize the tidal pattern
    python slo_scale_test.py --dry-run

    # Custom SLO scales
    python slo_scale_test.py --slo-scales 1.5 2.0 3.0 5.0 10.0
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

# ── Default Configuration ────────────────────────────────────────────────────

URL = "http://127.0.0.1:27888/v1/completions"
MODELS = ["Qwen3-8B", "Qwen2-7B"]
SLO_SCALES = [1.5, 2.0, 3.0, 5.0]

AVG_TOKENS_PER_SECOND = 15000  # Target average input token throughput
TOTAL_REQUESTS = 300
MAX_WORKERS = 200

# Prompt length distribution (Zipf: short prompts more frequent)
MIN_PROMPT_LEN = 20
MAX_PROMPT_LEN = 8000
ZIPF_EXPONENT = 1.0

# Tidal pattern parameters
TIDAL_CYCLES = 1.5    # Number of full oscillation cycles over the test
TIDAL_AMPLITUDE = 0.4  # 0 = uniform, 0.5 = full swing between models

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

def compute_base_slo_ms(prompt_tokens):
    """Base TTFT SLO in milliseconds, linearly dependent on prompt length."""
    return 100 + prompt_tokens * 0.2


# ── Tidal Model Sequence ─────────────────────────────────────────────────────

def generate_tidal_sequence(total, models, cycles=1.5, amplitude=0.4):
    """Generate tidal model sequence using sinusoidal weights.

    For N models, each model's weight oscillates sinusoidally with phase
    offsets of 2π/N, creating a smooth "tidal" shift where one model's
    traffic rises as another's falls.

    For 2 models A, B:
      P(A at step i) ∝ 0.5 + amplitude * sin(2π * cycles * i / total)
      P(B at step i) ∝ 0.5 - amplitude * sin(2π * cycles * i / total)
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


# ── Request Sender ────────────────────────────────────────────────────────────

def send_request(url, model, repeats, ttft_slo_ms, req_id):
    """Send a single request and check TTFT SLO."""
    prompt = "hello xllm " * (repeats // 4 + 1)
    slo_s = ttft_slo_ms / 1000.0
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0,
        "stream": False,
        "ttft_slo": int(ttft_slo_ms)
    }
    data = json.dumps(payload).encode('utf-8')
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers)

    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            response.read()
            latency = time.time() - start
            met_slo = latency <= slo_s
            tag = "OK" if met_slo else "TIMEOUT"
            print(f"[#{req_id}][{model}] Tokens: {repeats}, {tag}, "
                  f"Latency: {latency*1000:.1f}ms (SLO: {ttft_slo_ms:.0f}ms)")
            return ('success' if met_slo else 'timeout'), latency
    except Exception as e:
        latency = time.time() - start
        print(f"[#{req_id}][{model}] Tokens: {repeats}, FAIL: {e}")
        return 'fail', latency


# ── Single Scale Test ─────────────────────────────────────────────────────────

def run_single_scale(url, slo_scale, model_sequence, prompt_lengths,
                     avg_tokens_per_second, max_workers):
    """Run one full test at a given SLO scale. Returns result dict."""
    avg_request_rate = avg_tokens_per_second / AVG_PROMPT_LEN
    total_requests = len(model_sequence)

    print(f"\n{'='*60}")
    print(f"SLO Scale = {slo_scale}x  ({total_requests} requests)")
    print(f"{'='*60}")

    results_lock = threading.Lock()
    latencies = []
    per_model = defaultdict(lambda: {'success': 0, 'timeout': 0, 'fail': 0})
    counters = {'success': 0, 'timeout': 0, 'failed': 0}

    def task(model, repeats, req_id):
        ttft_slo_ms = compute_base_slo_ms(repeats) * slo_scale
        status, elapsed = send_request(url, model, repeats, ttft_slo_ms, req_id)
        with results_lock:
            latencies.append(elapsed)
            if status == 'success':
                counters['success'] += 1
                per_model[model]['success'] += 1
            elif status == 'timeout':
                counters['timeout'] += 1
                per_model[model]['timeout'] += 1
            else:
                counters['failed'] += 1
                per_model[model]['fail'] += 1

    # Reset seed so Poisson inter-arrival times are identical across scales
    random.seed(12345)

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in range(total_requests):
            if i > 0:
                interval = random.expovariate(avg_request_rate)
                time.sleep(interval)
            executor.submit(task, model_sequence[i], prompt_lengths[i], i + 1)

    total_time = time.time() - start_time
    total = counters['success'] + counters['timeout'] + counters['failed']
    attainment = counters['success'] / total * 100 if total > 0 else 0

    # Per-model attainment
    per_model_attainment = {}
    for m, c in per_model.items():
        m_total = c['success'] + c['timeout'] + c['fail']
        per_model_attainment[m] = c['success'] / m_total * 100 if m_total > 0 else 0

    print(f"\nScale {slo_scale}x completed in {total_time:.2f}s")
    print(f"  Overall:  Success={counters['success']}, Timeout={counters['timeout']}, "
          f"Failed={counters['failed']}, Attainment={attainment:.1f}%")
    for m in sorted(per_model.keys()):
        c = per_model[m]
        print(f"  {m}: Success={c['success']}, Timeout={c['timeout']}, "
              f"Fail={c['fail']}, Attainment={per_model_attainment[m]:.1f}%")

    # Latency percentiles
    latencies.sort()
    def pct(vals, p):
        idx = max(0, min(len(vals)-1, int(math.ceil(p/100.0 * len(vals))) - 1))
        return vals[idx]

    if latencies:
        print(f"  Latency p50={pct(latencies,50)*1000:.1f}ms "
              f"p90={pct(latencies,90)*1000:.1f}ms "
              f"p99={pct(latencies,99)*1000:.1f}ms "
              f"max={latencies[-1]*1000:.1f}ms")

    return {
        'attainment': attainment,
        'success': counters['success'],
        'timeout': counters['timeout'],
        'failed': counters['failed'],
        'total_time': total_time,
        'per_model_attainment': per_model_attainment,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_slo_scale(results, output_dir):
    """Plot 图10: SLO Scale vs SLO Attainment."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    scales = sorted(results.keys())
    attainments = [results[s]['attainment'] for s in scales]

    plt.figure(figsize=(8, 5))
    plt.plot(scales, attainments, 'o-', linewidth=2, markersize=8,
             color='#2196F3', label='Ours (xLLM-service)')
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
    """Plot per-model SLO attainment across scales."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    scales = sorted(results.keys())
    colors = ['#2196F3', '#FF5722', '#4CAF50', '#FFC107', '#9C27B0']

    plt.figure(figsize=(8, 5))
    for idx, model in enumerate(models):
        attainments = [results[s]['per_model_attainment'].get(model, 0) for s in scales]
        plt.plot(scales, attainments, 'o--', linewidth=1.5, markersize=6,
                 color=colors[idx % len(colors)], label=model)

    # Also plot overall
    overall = [results[s]['attainment'] for s in scales]
    plt.plot(scales, overall, 's-', linewidth=2.5, markersize=8,
             color='black', label='Overall')

    plt.xlabel('SLO Scale', fontsize=12)
    plt.ylabel('SLO Attainment (%)', fontsize=12)
    plt.title('Per-Model SLO Attainment vs SLO Scale', fontsize=14)
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
    print(f"  Models:        {args.models}")
    print(f"  SLO Scales:    {args.slo_scales}")
    print(f"  Requests/scale:{args.total_requests}")
    print(f"  Token rate:    {args.avg_tokens_per_second} tokens/s")
    print(f"  Tidal cycles:  {args.tidal_cycles}, amplitude: {args.tidal_amplitude}")

    # Pre-generate tidal model sequence & prompt lengths (shared across scales)
    random.seed(42)
    model_sequence = generate_tidal_sequence(
        args.total_requests, args.models,
        args.tidal_cycles, args.tidal_amplitude)
    prompt_lengths = [generate_zipf_prompt_len() for _ in range(args.total_requests)]

    # Print distribution
    dist = defaultdict(int)
    for m in model_sequence:
        dist[m] += 1
    print(f"  Model dist:    {dict(dist)}")

    # Always plot tidal pattern
    plot_tidal_pattern(model_sequence, args.models, args.output_dir)

    if args.dry_run:
        print("\nDry-run mode — skipping actual requests.")
        return

    # Run each SLO scale
    results = {}
    for scale in sorted(args.slo_scales):
        results[scale] = run_single_scale(
            args.url, scale, model_sequence, prompt_lengths,
            args.avg_tokens_per_second, args.max_workers)

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
    print(f"\n{'='*70}")
    print(f"{'Scale':<10} {'Attainment':<14} {'Success':<10} {'Timeout':<10} {'Failed':<10}")
    print(f"{'-'*70}")
    for s in scales:
        r = results[s]
        print(f"{s:<10.1f}x {r['attainment']:<13.1f}% {r['success']:<10} "
              f"{r['timeout']:<10} {r['failed']:<10}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
