"""Reproduce the NEXTQUESTION multi-condition synthetic holdout experiment."""
import argparse
from hospital.engine import save_research

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/hospital")
    parser.add_argument("--n", type=int, default=4800)
    args = parser.parse_args()
    if args.n < 200:
        parser.error("Use at least 200 cases for class representation")
    report = save_research(outdir=args.out, n=args.n)
    for policy, result in report["test_results"].items():
        print(f"{policy:8s} | accuracy {result['accuracy']:.1%} | tests {result['mean_tests']:.2f} | effort {result['mean_effort_points']:.2f}")
    print(f"Results written to {args.out}/results.md (simulation only, not medical advice).")
