import argparse
import re
import shutil
from pathlib import Path
import pandas as pd


def replace_geo_assignment(text: str, var: str, value: str) -> str:
    """
    Para líneas del .geo tipo:
      rb1 = 0.0115;
    preservando el ;
    """
    pat = re.compile(rf'(?m)^(\s*{re.escape(var)}\s*=\s*)([^;]+)(\s*;)')
    if not pat.search(text):
        raise RuntimeError(f"No encontré la variable GEO '{var}' en el archivo plantilla")
    return pat.sub(rf'\g<1>{value}\g<3>', text, count=1)


def replace_definition_assignment(text: str, var: str, value: str) -> str:
    """
    Para líneas del .definition tipo:
      $ I1 = 0.5
      $ N_Coil1 = 700
    preservando comentarios si hubiera.
    """
    pat = re.compile(rf'(?m)^(\s*\$?\s*{re.escape(var)}\s*=\s*)([^\n\r!#;]*)(.*)$')
    if not pat.search(text):
        raise RuntimeError(f"No encontré la variable DEF '{var}' en el archivo plantilla")
    return pat.sub(rf'\g<1>{value}\g<3>', text, count=1)


def ensure_population_columns(df):
    if "rb1_mm" not in df.columns and "rb1" in df.columns:
        df["rb1_mm"] = df["rb1"] * 1000.0
    if "rb2_mm" not in df.columns and "rb2" in df.columns:
        df["rb2_mm"] = df["rb2"] * 1000.0
    return df

def set_geo_params(text: str, row: pd.Series) -> str:
    text = replace_geo_assignment(text, "rb1", f"{row['rb1_mm']*1e-3:.12g}")
    text = replace_geo_assignment(text, "rb2", f"{row['rb2_mm']*1e-3:.12g}")
    text = replace_geo_assignment(text, "hb",  f"{row['hb_mm']*1e-3:.12g}")
    text = replace_geo_assignment(text, "gap_pc", f"{row['gap_pc_mm']*1e-3:.12g}")
    return text


def set_definition_params(text: str, row: pd.Series) -> str:
    text = replace_definition_assignment(text, "I1", f"{row['I_eval_A']:.12g}")
    text = replace_definition_assignment(text, "N_Coil1", f"{int(row['Ni'])}")
    return text


def main():
    ap = argparse.ArgumentParser(description="Escribe casos del mini ciclo a partir de una población CSV")
    ap.add_argument("--population", required=True)
    ap.add_argument("--geo-template", required=True)
    ap.add_argument("--sif-template", required=True)
    ap.add_argument("--definition-template", required=True)
    ap.add_argument("--outdir", default="mini_cycle_cases")
    ap.add_argument("--manifest-out", default="")

    args = ap.parse_args()

    pop = ensure_population_columns(pd.read_csv(args.population))
    geo_tpl = Path(args.geo_template).read_text(encoding="utf-8", errors="ignore")
    def_tpl = Path(args.definition_template).read_text(encoding="utf-8", errors="ignore")
    sif_src = Path(args.sif_template)
    geo_name = Path(args.geo_template).name
    sif_name = Path(args.sif_template).name
    def_name = Path(args.definition_template).name

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for _, row in pop.iterrows():
        case_dir = outdir / str(row["individual_id"])
        case_dir.mkdir(parents=True, exist_ok=True)

        geo_text = set_geo_params(geo_tpl, row)
        def_text = set_definition_params(def_tpl, row)

        (case_dir / geo_name).write_text(geo_text, encoding="utf-8")
        (case_dir / def_name).write_text(def_text, encoding="utf-8")
        shutil.copy2(sif_src, case_dir / sif_name)

        manifest.append({
            **row.to_dict(),
            "case_dir": str(case_dir),
            "geo_path": str(case_dir / geo_name),
            "sif_path": str(case_dir / sif_name),
            "definition_path": str(case_dir / def_name),
        })

    manifest_df = pd.DataFrame(manifest)
    manifest_path = Path(args.manifest_out) if args.manifest_out else (outdir / "population_manifest.csv")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(manifest_path, index=False)
    print(f"Wrote cases in: {outdir}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
