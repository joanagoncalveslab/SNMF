# scripts/run_snmf.py
import argparse
import os
import json
import SNMF.sigpro as sig

def parse_args():
    p = argparse.ArgumentParser(description="Train and evaluate SNMF.")
    p.add_argument("--x-train", required=True)
    p.add_argument("--y-train", required=True)
    p.add_argument("--x-test", default=None)
    p.add_argument("--y-test", default=None)
    p.add_argument("--outdir", required=True)

    p.add_argument("--k", type=int, default=5)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--lambda-c", type=float, default=0.1)
    p.add_argument("--lambda-p", type=float, default=1e-4)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--seeds", default="random")
    p.add_argument("--input-type", default="text")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--no-filter", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    run_cfg = vars(args)
    with open(os.path.join(args.outdir, "run_config.json"), "w") as f:
        json.dump(run_cfg, f, indent=2)

    train_out = sig.sigProfilerExtractor(
        input_type=args.input_type,
        output=args.outdir,
        input_data=args.x_train,
        input_label=args.y_train,
        minimum_signatures=args.k,
        maximum_signatures=args.k,
        nmf_replicates=args.reps,
        seeds=args.seeds,
        lambda_c=args.lambda_c,
        lambda_p=args.lambda_p,
        lr=args.lr,
        make_decomposition_plots=not args.no_plots,
    )

    result = {"train_out": train_out if isinstance(train_out, dict) else None}

    if args.x_test and args.y_test:
        test_out = sig.test_sigProfilerExtractor(
            input_type=args.input_type,
            model_path=args.outdir,
            output=args.outdir,
            test_data=args.x_test,
            test_label=args.y_test,
            minimum_signatures=args.k,
            maximum_signatures=args.k,
            nmf_replicates=args.reps,
            lambda_c=args.lambda_c,
            lambda_p=args.lambda_p,
            lr=args.lr,
            filter=not args.no_filter,
            make_decomposition_plots=not args.no_plots,
        )
        result["test_out"] = test_out

    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)

if __name__ == "__main__":
    main()
