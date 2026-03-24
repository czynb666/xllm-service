"""SLO Scale Sensitivity Test with Tidal Traffic Pattern (single-scale).

Runs ONE slo_scale value per invocation (to allow cooldown between runs).
Generates a JSON result file named: slo_scale_{scale}_{algorithm_name}.json

SLO Definition (AlpaServe-style):
  - TTFT SLO:  ttft <= ttft_slo
  - Per-token deadline:  deadline(i) = ttft_slo + i * tpot_slo   (i = 1..output_tokens)
                         actual(i)   = ttft     + i * avg_tpot
  - avg_tpot = (total_latency - ttft) / (output_tokens - 1)
  - E2E SLO:  TTFT met AND all per-token deadlines met
  - TPOT attainment:  fraction of tokens (across all requests) meeting deadline(i)

Usage:
    python slo_scale_test.py --slo-scale 1.0 --algorithm-name xllm
    python slo_scale_test.py --slo-scale 2.0 --algorithm-name alpaserve --url http://...
    python slo_scale_test.py --dry-run
"""

import argparse
import bisect
import json
import math
import os
import random
import socket
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

# Set no_proxy to bypass proxies for local connections
os.environ['no_proxy'] = '*'

# ── All Hyperparameters ──────────────────────────────────────────────────────

# Service
URL = "http://127.0.0.1:27888/v1/completions"
MODELS = ["Qwen3-8B", "Qwen2-7B", "Qwen3-4B"]

# SLO
SLO_SCALE = 1.0
ALGORITHM_NAME = "xllm"
BASE_TTFT_SLO_INTERCEPT_MS = 500   # base_ttft_slo = INTERCEPT + SLOPE * prompt_tokens
BASE_TTFT_SLO_SLOPE_MS = 0.6       # ms per input token
BASE_TPOT_SLO_MS = 50              # Base TPOT SLO in ms (before scaling)
MAX_OUTPUT_TOKENS = 20              # Output tokens per request (for TPOT measurement)

# Traffic
AVG_TOKENS_PER_SECOND = 50000  # Target average input token throughput
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


# ── Streaming Request Sender (raw socket for accurate per-token timing) ──────

def send_request_stream(url, model, prompt_tokens, max_tokens, ttft_slo_ms, req_id):
    """Send a streaming request via raw socket, parse SSE for TTFT and TPOT.

    Uses raw socket instead of urllib to avoid BufferedReader's 8KB buffer
    which coalesces all SSE events into a single read, making per-token
    timestamps useless.

    Returns (status, ttft_s, total_latency_s, output_tokens).
    """
    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port or 80

    prompt = "hello xllm " * (prompt_tokens // 4 + 1)
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "ttft_slo": int(ttft_slo_ms),
    }).encode('utf-8')

    http_req = (
        f"POST {parsed.path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"\r\n"
    ).encode('utf-8') + body

    start = time.time()
    sock = None
    try:
        sock = socket.create_connection((host, port), timeout=REQUEST_TIMEOUT_S)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.sendall(http_req)

        # ── Read HTTP response headers ──
        header_buf = b''
        while b'\r\n\r\n' not in header_buf:
            chunk = sock.recv(4096)
            if not chunk:
                raise ConnectionError("connection closed before headers")
            header_buf += chunk

        header_end = header_buf.index(b'\r\n\r\n') + 4
        status_line = header_buf[:header_buf.index(b'\r\n')].decode()
        if ' 200 ' not in status_line:
            print(f"[#{req_id}][{model}] HTTP error: {status_line}")
            return 'fail', 0, time.time() - start, 0

        is_chunked = b'transfer-encoding: chunked' in header_buf[:header_end].lower()
        remaining = header_buf[header_end:]

        # ── SSE event processing ──
        ttft_time = None
        last_token_time = None
        token_count = 0
        line_buf = b''
        done = False

        def process_data(data):
            nonlocal ttft_time, last_token_time, token_count, line_buf, done
            line_buf += data
            while b'\n' in line_buf and not done:
                raw_line, line_buf = line_buf.split(b'\n', 1)
                line = raw_line.decode('utf-8', errors='replace').strip()
                if not line or not line.startswith('data:'):
                    continue
                payload_str = line[5:].strip()
                if payload_str == '[DONE]':
                    done = True
                    return
                token_count += 1
                now = time.time()
                if token_count == 1:
                    ttft_time = now
                last_token_time = now

        # ── Read body (chunked or identity) ──
        if is_chunked:
            chunk_buf = remaining
            while not done:
                # Read until we have a chunk header line
                while b'\r\n' not in chunk_buf:
                    data = sock.recv(4096)
                    if not data:
                        done = True
                        break
                    chunk_buf += data
                if done:
                    break

                crlf_pos = chunk_buf.index(b'\r\n')
                try:
                    chunk_size = int(chunk_buf[:crlf_pos].strip(), 16)
                except ValueError:
                    break
                chunk_buf = chunk_buf[crlf_pos + 2:]

                if chunk_size == 0:
                    break  # last chunk

                # Read full chunk body + trailing \r\n
                needed = chunk_size + 2
                while len(chunk_buf) < needed:
                    data = sock.recv(4096)
                    if not data:
                        done = True
                        break
                    chunk_buf += data
                if done:
                    break

                process_data(chunk_buf[:chunk_size])
                chunk_buf = chunk_buf[needed:]
        else:
            # Identity encoding: read until [DONE] or connection close
            process_data(remaining)
            while not done:
                data = sock.recv(4096)
                if not data:
                    break
                process_data(data)

        sock.close()
        sock = None

        if ttft_time is None or token_count == 0:
            print(f"[#{req_id}][{model}] InTokens: {prompt_tokens}, "
                  f"FAIL: no tokens received")
            return 'fail', 0, time.time() - start, 0

        ttft_s = ttft_time - start
        total_latency_s = last_token_time - start
        return 'ok', ttft_s, total_latency_s, token_count

    except Exception as e:
        print(f"[#{req_id}][{model}] InTokens: {prompt_tokens}, FAIL: {e}")
        return 'fail', 0, time.time() - start, 0
    finally:
        if sock:
            try:
                sock.close()
            except OSError:
                pass


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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='SLO Scale Sensitivity Test (single-scale per run)')
    parser.add_argument('--url', type=str, default=URL,
                        help='Service URL (default: %(default)s)')
    parser.add_argument('--models', nargs='+', default=MODELS,
                        help='Model names (default: %(default)s)')
    parser.add_argument('--slo-scale', type=float, default=SLO_SCALE,
                        help='SLO scale factor (default: %(default)s)')
    parser.add_argument('--algorithm-name', type=str, default=ALGORITHM_NAME,
                        help='Algorithm name for result file (default: %(default)s)')
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
                        help='Print config and exit, skip requests')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("SLO Scale Sensitivity Test (single-scale)")
    print(f"  Algorithm:       {args.algorithm_name}")
    print(f"  Models:          {args.models}")
    print(f"  SLO Scale:       {args.slo_scale}")
    print(f"  Max output tok:  {args.max_tokens}")
    print(f"  Base TPOT SLO:   {args.base_tpot_slo}ms")
    print(f"  Requests:        {args.total_requests}")
    print(f"  Token rate:      {args.avg_tokens_per_second} tokens/s")
    print(f"  Tidal cycles:    {args.tidal_cycles}, amplitude: {args.tidal_amplitude}")

    # Pre-generate tidal model sequence & prompt lengths
    random.seed(SEED_SEQUENCE)
    model_sequence = generate_tidal_sequence(
        args.total_requests, args.models,
        args.tidal_cycles, args.tidal_amplitude)
    prompt_lengths = [generate_zipf_prompt_len() for _ in range(args.total_requests)]

    dist = defaultdict(int)
    for m in model_sequence:
        dist[m] += 1
    print(f"  Model dist:      {dict(dist)}")

    if args.dry_run:
        print("\nDry-run mode -- skipping actual requests.")
        return

    # Run single scale
    result = run_single_scale(
        args.url, args.slo_scale, model_sequence, prompt_lengths,
        args.avg_tokens_per_second, args.max_workers,
        args.max_tokens, args.base_tpot_slo)

    # Save results to JSON with slo_scale and algorithm_name in filename
    filename = f"slo_scale_{args.slo_scale}_{args.algorithm_name}.json"
    json_path = os.path.join(args.output_dir, filename)
    with open(json_path, 'w') as f:
        json.dump({
            'config': {
                'algorithm_name': args.algorithm_name,
                'slo_scale': args.slo_scale,
                'models': args.models,
                'max_tokens': args.max_tokens,
                'base_tpot_slo_ms': args.base_tpot_slo,
                'total_requests': args.total_requests,
                'avg_tokens_per_second': args.avg_tokens_per_second,
                'tidal_cycles': args.tidal_cycles,
                'tidal_amplitude': args.tidal_amplitude,
            },
            'result': result,
        }, f, indent=2)
    print(f"\nSaved {json_path}")


if __name__ == "__main__":
    main()
