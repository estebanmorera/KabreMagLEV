import argparse
from pathlib import Path
import pandas as pd


def default_population():
    # Mini ciclo calibrado: Ni fijo=700, I fijo=+3 A (NI_ref=2100 A-vueltas)
    rows = [
        ("I01", 11.5, 16.0,  8.0, 4.2, 700, 2100.0,  3.0, "compacta_cercana"),
        ("I02", 11.5, 16.0,  8.0, 4.6, 700, 2100.0,  3.0, "compacta_cercana"),
        ("I03", 11.8, 16.5, 10.0, 4.4, 700, 2100.0,  3.0, "compacta_media"),
        ("I04", 12.2, 17.2, 11.0, 4.6, 700, 2100.0,  3.0, "media"),
        ("I05", 12.7, 18.7, 12.7, 4.8, 700, 2100.0,  3.0, "base_refinada"),
        ("I06", 12.7, 19.5, 14.5, 4.8, 700, 2100.0,  3.0, "alta_media"),
        ("I07", 12.7, 19.5, 18.0, 4.8, 700, 2100.0,  3.0, "alta_extendida"),
        ("I08", 13.8, 21.0, 12.7, 5.0, 700, 2100.0,  3.0, "externa_media"),
        ("I09", 15.0, 23.0, 12.7, 5.2, 700, 2100.0,  3.0, "externa_gruesa"),
        ("I10", 14.5, 22.0, 16.0, 5.0, 700, 2100.0,  3.0, "externa_alta"),
    ]
    df = pd.DataFrame(rows, columns=[
        "individual_id", "rb1_mm", "rb2_mm", "hb_mm", "gap_pc_mm",
        "Ni", "NI_ref_Aturn", "I_eval_A", "family"
    ])
    df["coil_radial_thickness_mm"] = df["rb2_mm"] - df["rb1_mm"]
    df["coil_mean_radius_mm"] = 0.5 * (df["rb1_mm"] + df["rb2_mm"])
    df["sign"] = "+"
    df["is_viable"] = True
    df["note"] = "mini_ciclo_calibrado"
    return df


def main():
    ap = argparse.ArgumentParser(description="Construye la población inicial del mini ciclo")
    ap.add_argument("--out", default="initial_population.csv")
    args = ap.parse_args()

    df = default_population()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote: {out}")
    print(df[["individual_id","rb1_mm","rb2_mm","hb_mm","gap_pc_mm","Ni","I_eval_A","family"]].to_string(index=False))


if __name__ == "__main__":
    main()
