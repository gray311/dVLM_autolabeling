"""Aggregate results/*.json into a speed + quality comparison table.

Speed: load time, avg latency/frame, tokens/s, tokens-per-forward (diffusion
parallelism; AR=1). Quality proxies: valid-JSON rate, structural completeness
(expected annotation keys present), avg output length. Qualitative reasoning
quality is assessed by reading samples (see REPORT.md)."""
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
EXPECTED_KEYS = ["scene_description", "traffic_elements", "dynamic_agents",
                 "prediction", "planning", "region_inference", "reasoning"]

ORDER = ["fast_dvlm", "nemotron", "diffusiongemma", "qwen3vl"]


def structural_score(text):
    """Fraction of expected annotation keys that appear (JSON or substring)."""
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return sum(k in obj for k in EXPECTED_KEYS) / len(EXPECTED_KEYS)
    except Exception:
        pass
    low = text.lower()
    return sum(k in low for k in EXPECTED_KEYS) / len(EXPECTED_KEYS)


def main():
    rows = []
    for key in ORDER:
        path = os.path.join(RESULTS, f"{key}.json")
        if not os.path.exists(path):
            continue
        try:
            rec = json.load(open(path))
        except Exception as e:
            print(f"[skip] {path}: {e}")
            continue
        if not rec.get("samples"):
            continue
        s = rec["summary"]
        samples = rec["samples"]
        struct = sum(structural_score(x["output"]) for x in samples) / len(samples)
        out_chars = sum(len(x["output"]) for x in samples) / len(samples)
        rows.append({
            "key": key,
            "model": rec["model"],
            "load_s": s.get("load_time_s"),
            "avg_latency_s": s.get("avg_latency_s"),
            "tok_per_s": s.get("avg_tok_per_s"),
            "tokens_per_forward": s.get("avg_tokens_per_forward"),
            "avg_out_tokens": s.get("avg_out_tokens"),
            "valid_json_rate": s.get("valid_json_rate"),
            "struct_completeness": round(struct, 3),
            "avg_out_chars": round(out_chars),
            "n_samples": s.get("n_samples"),
            "note": rec.get("note", ""),
        })

    print(f"\n{'model':<28}{'load':>6}{'lat/frame':>11}{'tok/s':>8}"
          f"{'tok/fwd':>9}{'out_tok':>9}{'json%':>7}{'struct%':>9}")
    print("-" * 95)
    for r in rows:
        print(f"{r['model']:<28}{r['load_s'] or 0:>5.0f}s{r['avg_latency_s'] or 0:>10.2f}s"
              f"{r['tok_per_s'] or 0:>8.1f}"
              f"{(r['tokens_per_forward'] or 0):>9.2f}{r['avg_out_tokens'] or 0:>9.0f}"
              f"{100*(r['valid_json_rate'] or 0):>6.0f}%{100*r['struct_completeness']:>8.0f}%")
        if r["note"]:
            print(f"    note: {r['note']}")

    json.dump(rows, open(os.path.join(RESULTS, "comparison.json"), "w"), indent=2)
    print(f"\nwrote {RESULTS}/comparison.json")


if __name__ == "__main__":
    main()
