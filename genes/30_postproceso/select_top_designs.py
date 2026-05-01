import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Selecciona mejores diseños desde population_evaluation.csv")
    ap.add_argument("--evaluation", required=True)
    ap.add_argument("--out", default="selected_designs.csv")
    ap.add_argument("--top-n", type=int, default=4)
    ap.add_argument("--top-per-family", type=int, default=1)
    args = ap.parse_args()

    df = pd.read_csv(args.evaluation)

    if "eval_status" in df.columns:
        df = df[df["eval_status"] == "OK"].copy()

    df = df.sort_values("score_total", ascending=False).reset_index(drop=True)

    selected = []

    # top global
    top_global = df.head(args.top_n).copy()
    top_global["selection_reason"] = "top_global"
    selected.append(top_global)

    # top por familia
    if "family" in df.columns:
        fam_rows = []
        for fam, block in df.groupby("family", sort=False):
            block = block.sort_values("score_total", ascending=False).head(args.top_per_family).copy()
            block["selection_reason"] = f"top_family:{fam}"
            fam_rows.append(block)
        if fam_rows:
            selected.append(pd.concat(fam_rows, ignore_index=True))

    out = pd.concat(selected, ignore_index=True).drop_duplicates(subset=["individual_id"]).reset_index(drop=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()