import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ModuleNotFoundError:
    plt = None
    HAS_MPL = False


def dedup_by_coord(df, coord_col, energy_col='W_J', atol=1e-12):
    df = df.copy().sort_values(coord_col, kind='stable').reset_index(drop=True)
    if len(df) == 0:
        return df
    x = df[coord_col].to_numpy(float)
    groups = []
    cur = [0]
    for i in range(1, len(df)):
        if np.isclose(x[i], x[i - 1], rtol=0.0, atol=atol):
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)

    kept = []
    for g in groups:
        block = df.iloc[g].copy()
        if 'status' in block.columns:
            status = block['status'].astype(str).str.upper()
            ok = block[status == 'OK']
            rescue = block[status == 'RESCUE_OK']
            if len(ok) > 0:
                block = ok
            elif len(rescue) > 0:
                block = rescue
        if energy_col in block.columns:
            valid = block[np.isfinite(block[energy_col].to_numpy(float))]
            if len(valid) > 0:
                block = valid
        kept.append(block.iloc[-1])
    return pd.DataFrame(kept).sort_values(coord_col, kind='stable').reset_index(drop=True)


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if np.isclose(ss_tot, 0.0):
        return 1.0 if np.isclose(ss_res, 0.0) else np.nan
    return 1.0 - ss_res / ss_tot


def robust_scale(a):
    a = np.asarray(a, float)
    med = np.median(a)
    mad = np.median(np.abs(a - med))
    if mad <= 1e-15:
        alt = np.std(a)
        return med, alt if alt > 1e-15 else 1.0
    return med, 1.4826 * mad


def robust_z(a):
    med, sc = robust_scale(a)
    return (np.asarray(a, float) - med) / sc


def poly_to_human(coeffs_desc, varname='u', digits=4):
    terms = []
    deg = len(coeffs_desc) - 1
    for i, c in enumerate(coeffs_desc):
        p = deg - i
        if np.isclose(c, 0.0):
            continue
        c_str = f'{c:.{digits}e}'
        if p == 0:
            term = f'{c_str}'
        elif p == 1:
            term = f'{c_str}*{varname}'
        else:
            term = f'{c_str}*{varname}^{p}'
        terms.append(term)
    return '0' if not terms else ' + '.join(terms).replace('+ -', '- ')


class ScaledPoly:
    def __init__(self, coeff, x0, s):
        self.coeff = np.asarray(coeff, float)
        self.p = np.poly1d(self.coeff)
        self.dp_du = np.polyder(self.p, 1)
        self.d2p_du2 = np.polyder(self.p, 2)
        self.x0 = float(x0)
        self.s = float(s)

    def u(self, x):
        return (np.asarray(x, float) - self.x0) / self.s

    def W(self, x):
        return self.p(self.u(x))

    def F(self, x):
        return -self.dp_du(self.u(x)) / self.s

    def K_phys(self, x):
        return self.d2p_du2(self.u(x)) / (self.s ** 2)


def fit_scaled_poly(x, y, deg, w=None):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x0 = float(np.mean(x))
    s = float(np.max(np.abs(x - x0)))
    if np.isclose(s, 0.0):
        raise RuntimeError('Todos los desplazamientos son iguales.')
    u = (x - x0) / s
    coeff = np.polyfit(u, y, deg=deg, w=w)
    return ScaledPoly(coeff, x0, s)


def irls_weights(resid, method='huber', c_huber=1.345, c_tukey=4.685):
    resid = np.asarray(resid, float)
    _, sc = robust_scale(resid)
    r = resid / sc
    a = np.abs(r)
    if method == 'none':
        return np.ones_like(r)
    if method == 'huber':
        w = np.ones_like(r)
        m = a > c_huber
        w[m] = c_huber / a[m]
        return w
    if method == 'tukey':
        w = np.zeros_like(r)
        m = a < c_tukey
        t = 1.0 - (r[m] / c_tukey) ** 2
        w[m] = t ** 2
        return w
    raise ValueError(f'Método robusto no soportado: {method}')


def robust_polyfit_irls(x, y, deg, method='huber', max_iter=50, tol=1e-6):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    model = fit_scaled_poly(x, y, deg, None)
    w = np.ones_like(y)
    for _ in range(max_iter):
        resid = y - model.W(x)
        new_w = irls_weights(resid, method=method)
        if np.max(np.abs(new_w - w)) < tol:
            w = new_w
            break
        w = new_w
        model = fit_scaled_poly(x, y, deg, w)
    return model, w


def local_poly_predict_loo(x, y, window=7, degree=2):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    pred = np.full(n, np.nan)
    half = max(1, window // 2)
    for i in range(n):
        if i < half:
            idx = list(range(0, min(n, window)))
        elif i > n - half - 1:
            idx = list(range(max(0, n - window), n))
        else:
            idx = list(range(i - half, i + half + 1))
        if i in idx:
            idx.remove(i)
        if len(idx) < degree + 1:
            all_idx = [j for j in range(n) if j != i]
            idx = all_idx[:max(degree + 1, min(len(all_idx), window - 1))]
        if len(idx) < degree + 1:
            continue
        d = min(degree, len(idx) - 1)
        try:
            m = fit_scaled_poly(x[idx], y[idx], d, None)
            pred[i] = float(m.W([x[i]])[0])
        except Exception:
            pred[i] = np.nan
    return pred


def edge_interp_predict(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    pred = np.full(n, np.nan)
    if n < 3:
        return pred
    # one-sided at edges, centered interior
    pred[0] = y[1] + (x[0] - x[1]) * (y[2] - y[1]) / (x[2] - x[1])
    pred[-1] = y[-2] + (x[-1] - x[-2]) * (y[-2] - y[-3]) / (x[-2] - x[-3])
    for i in range(1, n - 1):
        pred[i] = y[i - 1] + (x[i] - x[i - 1]) * (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1])
    return pred


def hybrid_keep_mask(x, y, deg_global=4, robust_fit='huber', local_window=7, local_degree=2,
                     global_z=4.0, local_z=4.0, jump_z=4.0, branch_z=4.0, max_iters=2):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    keep = np.ones(n, dtype=bool)
    reasons = np.array([''] * n, dtype=object)
    diag = {}

    for _ in range(max_iters):
        idx = np.where(keep)[0]
        if len(idx) < max(deg_global + 1, 5):
            break
        model, fit_w = robust_polyfit_irls(x[idx], y[idx], deg_global, method=robust_fit)
        y_global = np.full(n, np.nan)
        y_global[idx] = model.W(x[idx])
        r_global = np.full(n, np.nan)
        r_global[idx] = y[idx] - y_global[idx]
        zg = np.full(n, np.nan)
        zg[idx] = robust_z(r_global[idx])

        xk = x[idx]
        yk = y[idx]
        y_local_k = local_poly_predict_loo(xk, yk, window=local_window, degree=local_degree)
        rl = yk - y_local_k
        zl_k = robust_z(rl)
        y_jump_k = edge_interp_predict(xk, yk)
        rj = yk - y_jump_k
        zj_k = robust_z(rj)
        branch_k = y_local_k - model.W(xk)
        zb_k = robust_z(branch_k)

        y_local = np.full(n, np.nan); y_local[idx] = y_local_k
        y_jump = np.full(n, np.nan); y_jump[idx] = y_jump_k
        zl = np.full(n, np.nan); zl[idx] = zl_k
        zj = np.full(n, np.nan); zj[idx] = zj_k
        zb = np.full(n, np.nan); zb[idx] = zb_k

        # explicit rules
        bad_isolated = (np.abs(zl) > local_z) & (np.abs(zj) > jump_z)
        bad_branch = (np.abs(zg) > global_z) & (np.abs(zb) > branch_z)
        strong_bad = ((np.abs(zg) > global_z).astype(int)
                      + (np.abs(zl) > local_z).astype(int)
                      + (np.abs(zj) > jump_z).astype(int)
                      + (np.abs(zb) > branch_z).astype(int)) >= 3
        bad = keep & (bad_isolated | bad_branch | strong_bad)
        if not np.any(bad):
            diag = {
                'y_global': y_global, 'r_global': r_global, 'z_global': zg,
                'y_local': y_local, 'z_local': zl,
                'y_jump': y_jump, 'z_jump': zj,
                'z_branch': zb, 'fit_weight': np.where(np.isin(np.arange(n), idx), np.nan, np.nan)
            }
            diag['fit_weight'][idx] = fit_w
            return keep, reasons, diag

        for i in np.where(bad)[0]:
            why = []
            if bad_isolated[i]:
                why.append('isolated(local+jump)')
            if bad_branch[i]:
                why.append('branch(global+trend)')
            if strong_bad[i]:
                why.append('3of4')
            reasons[i] = '|'.join(why)
        keep[bad] = False
        diag = {
            'y_global': y_global, 'r_global': r_global, 'z_global': zg,
            'y_local': y_local, 'z_local': zl,
            'y_jump': y_jump, 'z_jump': zj,
            'z_branch': zb, 'fit_weight': np.where(np.isin(np.arange(n), idx), np.nan, np.nan)
        }
        diag['fit_weight'][idx] = fit_w

    return keep, reasons, diag


def main():
    ap = argparse.ArgumentParser(
        description='Ajusta W(z), deriva analíticamente F(z), K(z), con filtro híbrido de continuidad + tendencia global.'
    )
    ap.add_argument('--csv', required=True)
    ap.add_argument('--coord', default='dz_m')
    ap.add_argument('--energy', default='W_J')
    ap.add_argument('--degree', type=int, default=4)
    ap.add_argument('--out-prefix', default='polyfit_energy_hybrid')
    ap.add_argument('--xmin-mm', type=float, default=None)
    ap.add_argument('--xmax-mm', type=float, default=None)
    ap.add_argument('--keep-outliers', action='store_true')
    ap.add_argument('--drop-suspect', action='store_true')
    ap.add_argument('--hybrid-filter', action='store_true')
    ap.add_argument('--hybrid-window', type=int, default=7)
    ap.add_argument('--hybrid-local-degree', type=int, default=2)
    ap.add_argument('--global-z', type=float, default=4.0)
    ap.add_argument('--local-z', type=float, default=4.0)
    ap.add_argument('--jump-z', type=float, default=4.0)
    ap.add_argument('--branch-z', type=float, default=4.0)
    ap.add_argument('--hybrid-iters', type=int, default=2)
    ap.add_argument('--robust-fit', choices=['none', 'huber', 'tukey'], default='huber')
    ap.add_argument('--nplot', type=int, default=600)
    ap.add_argument('--stiffness-sign', choices=['physical', 'paper'], default='physical')
    ap.add_argument('--eval-at-mm', type=float, default=0.0)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_prefix = Path(args.out_prefix)
    df = pd.read_csv(csv_path)

    if args.coord not in df.columns:
        raise ValueError(f'No existe la columna de desplazamiento: {args.coord}')
    if args.energy not in df.columns:
        raise ValueError(f'No existe la columna de energía: {args.energy}')

    raw_n = len(df)

    if not args.keep_outliers:
        if 'is_outlier' in df.columns:
            df = df[df['is_outlier'].fillna(0).astype(int) == 0]
        if 'status' in df.columns:
            df = df[df['status'].astype(str).str.upper().isin(['OK', 'RESCUE_OK'])]
    if args.drop_suspect and 'suspect' in df.columns:
        df = df[df['suspect'].fillna(0).astype(int) == 0]

    keep_cols = [args.coord, args.energy]
    extra_cols = []
    for c in ['status', 'is_outlier', 'suspect', 'suspect_reason']:
        if c in df.columns:
            extra_cols.append(c)
    df = df[keep_cols + extra_cols].dropna(subset=[args.coord, args.energy]).copy()
    after_basic = len(df)

    df = dedup_by_coord(df, args.coord, energy_col=args.energy, atol=1e-12)
    after_dedup = len(df)

    if args.xmin_mm is not None:
        df = df[df[args.coord] >= args.xmin_mm * 1e-3]
    if args.xmax_mm is not None:
        df = df[df[args.coord] <= args.xmax_mm * 1e-3]
    df = df.sort_values(args.coord, kind='stable').reset_index(drop=True)
    after_window = len(df)

    if len(df) < args.degree + 1:
        raise RuntimeError(f'No hay suficientes puntos ({len(df)}) para un polinomio de grado {args.degree}.')

    x_m = df[args.coord].to_numpy(float)
    W = df[args.energy].to_numpy(float)

    shape_keep = np.ones(len(df), dtype=bool)
    shape_reason = np.array([''] * len(df), dtype=object)
    diag = {
        'y_global': np.full(len(df), np.nan),
        'r_global': np.full(len(df), np.nan),
        'z_global': np.full(len(df), np.nan),
        'y_local': np.full(len(df), np.nan),
        'z_local': np.full(len(df), np.nan),
        'y_jump': np.full(len(df), np.nan),
        'z_jump': np.full(len(df), np.nan),
        'z_branch': np.full(len(df), np.nan),
        'fit_weight': np.ones(len(df), dtype=float),
    }

    if args.hybrid_filter:
        shape_keep, shape_reason, diag = hybrid_keep_mask(
            x_m, W,
            deg_global=args.degree,
            robust_fit=args.robust_fit,
            local_window=args.hybrid_window,
            local_degree=args.hybrid_local_degree,
            global_z=args.global_z,
            local_z=args.local_z,
            jump_z=args.jump_z,
            branch_z=args.branch_z,
            max_iters=args.hybrid_iters,
        )

    used = df[shape_keep].copy().reset_index(drop=True)
    if len(used) < args.degree + 1:
        raise RuntimeError(
            f'Después del filtro híbrido quedaron {len(used)} puntos; no alcanza para grado {args.degree}.'
        )

    xu = used[args.coord].to_numpy(float)
    Wu = used[args.energy].to_numpy(float)
    model, fit_w = robust_polyfit_irls(xu, Wu, args.degree, method=args.robust_fit)
    used_fit = model.W(xu)
    r2 = r2_score(Wu, used_fit)

    x_plot_m = np.linspace(np.min(xu), np.max(xu), args.nplot)
    W_plot = model.W(x_plot_m)
    F_plot_N = model.F(x_plot_m)
    K_phys_plot = model.K_phys(x_plot_m)
    K_show_plot = -K_phys_plot if args.stiffness_sign == 'paper' else K_phys_plot

    x_eval_m = args.eval_at_mm * 1e-3
    W_eval = float(model.W([x_eval_m])[0])
    F_eval = float(model.F([x_eval_m])[0])
    K_phys_eval = float(model.K_phys([x_eval_m])[0])
    K_show_eval = -K_phys_eval if args.stiffness_sign == 'paper' else K_phys_eval

    curve_df = pd.DataFrame({
        'x_m': x_plot_m,
        'x_mm': x_plot_m * 1e3,
        'W_fit_J': W_plot,
        'F_fit_N': F_plot_N,
        'K_physical_N_per_m': K_phys_plot,
        'K_physical_N_per_mm': K_phys_plot / 1000.0,
        'K_shown_N_per_m': K_show_plot,
        'K_shown_N_per_mm': K_show_plot / 1000.0,
    })
    curve_path = Path(str(out_prefix) + '.fit_curve.csv')
    curve_df.to_csv(curve_path, index=False)

    points_df = df.copy()
    points_df['x_mm'] = points_df[args.coord] * 1e3
    points_df['shape_keep'] = shape_keep.astype(int)
    points_df['shape_reason'] = shape_reason
    points_df['global_pred_J'] = diag['y_global']
    points_df['global_resid_J'] = diag['r_global']
    points_df['global_z'] = diag['z_global']
    points_df['local_pred_J'] = diag['y_local']
    points_df['local_resid_J'] = points_df[args.energy] - points_df['local_pred_J']
    points_df['local_z'] = diag['z_local']
    points_df['jump_pred_J'] = diag['y_jump']
    points_df['jump_resid_J'] = points_df[args.energy] - points_df['jump_pred_J']
    points_df['jump_z'] = diag['z_jump']
    points_df['branch_z'] = diag['z_branch']
    points_df['fit_weight'] = np.nan
    points_df.loc[points_df['shape_keep'] == 1, 'fit_weight'] = fit_w
    points_df['W_fit_used_J'] = np.nan
    points_df.loc[points_df['shape_keep'] == 1, 'W_fit_used_J'] = used_fit
    points_path = Path(str(out_prefix) + '.fit_points.csv')
    points_df.to_csv(points_path, index=False)

    summary_path = Path(str(out_prefix) + '.summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f'Archivo CSV: {csv_path}\n')
        f.write(f'Puntos totales CSV: {raw_n}\n')
        f.write(f'Después de filtro básico: {after_basic}\n')
        f.write(f'Después de deduplicar: {after_dedup}\n')
        f.write(f'Después de ventana: {after_window}\n')
        f.write(f'Puntos usados ajuste: {len(used)} / {len(df)}\n')
        f.write(f'Grado polinomio: {args.degree}\n')
        f.write(f'Robust fit: {args.robust_fit}\n')
        f.write(f'Hybrid filter: {args.hybrid_filter}\n')
        if args.hybrid_filter:
            f.write(f'Hybrid params: window={args.hybrid_window}, local_degree={args.hybrid_local_degree}, '
                    f'global_z={args.global_z}, local_z={args.local_z}, jump_z={args.jump_z}, '
                    f'branch_z={args.branch_z}, iters={args.hybrid_iters}\n')
        f.write(f'Ventana usada [mm]: {np.min(xu)*1e3:.6g} a {np.max(xu)*1e3:.6g}\n')
        f.write(f'Escalado interno: u = (x - {model.x0:.9e}) / {model.s:.9e}\n')
        f.write(f'R^2 energía (solo usados): {r2:.9f}\n')
        f.write(f'Polinomio W(u): {poly_to_human(model.coeff, varname="u", digits=4)}\n')
        f.write(f'W en z={args.eval_at_mm:.6g} mm: {W_eval:.9g} J\n')
        f.write(f'F en z={args.eval_at_mm:.6g} mm: {F_eval:.9g} N\n')
        f.write(f'K_physical en z={args.eval_at_mm:.6g} mm: {K_phys_eval:.9g} N/m\n')
        f.write(f'K_shown en z={args.eval_at_mm:.6g} mm: {K_show_eval:.9g} N/m\n')
        f.write(f'Convención mostrada: {args.stiffness_sign}\n')

    fig_path = Path(str(out_prefix) + '.png')
    
    if HAS_MPL:
        fig, axs = plt.subplots(4, 1, figsize=(11, 14), constrained_layout=True)
        x_mm = df[args.coord].to_numpy(float) * 1e3
        x_plot_mm = x_plot_m * 1e3
        used_mask = points_df['shape_keep'].to_numpy(int) == 1
    
        axs[0].scatter(x_mm[used_mask], W[used_mask], s=26, label='energía usada')
        if np.any(~used_mask):
            axs[0].scatter(x_mm[~used_mask], W[~used_mask], s=46, marker='x', label='descartada por híbrido')
        axs[0].plot(x_plot_mm, W_plot, linewidth=2, label='ajuste polinómico')
        axs[0].set_xlabel('dz (mm)')
        axs[0].set_ylabel('W (J)')
        axs[0].set_title('energía W(z): datos y ajuste híbrido')
        axs[0].grid(True, alpha=0.3)
        axs[0].legend()
    
        axs[1].scatter(x_mm, points_df['global_z'], s=16, label='global_z')
        axs[1].scatter(x_mm, points_df['local_z'], s=16, label='local_z')
        axs[1].scatter(x_mm, points_df['jump_z'], s=16, label='jump_z')
        axs[1].scatter(x_mm, points_df['branch_z'], s=16, label='branch_z')
        axs[1].axhline(args.global_z, color='C0', linestyle='--', linewidth=1)
        axs[1].axhline(-args.global_z, color='C0', linestyle='--', linewidth=1)
        axs[1].axhline(args.local_z, color='C1', linestyle='--', linewidth=1)
        axs[1].axhline(-args.local_z, color='C1', linestyle='--', linewidth=1)
        axs[1].axhline(args.jump_z, color='C2', linestyle='--', linewidth=1)
        axs[1].axhline(-args.jump_z, color='C2', linestyle='--', linewidth=1)
        axs[1].axhline(args.branch_z, color='C3', linestyle='--', linewidth=1)
        axs[1].axhline(-args.branch_z, color='C3', linestyle='--', linewidth=1)
        axs[1].set_xlabel('dz (mm)')
        axs[1].set_ylabel('z robusto')
        axs[1].set_title('diagnóstico híbrido: global / local / salto / tendencia')
        axs[1].grid(True, alpha=0.3)
        axs[1].legend(ncol=4, fontsize=8)
    
        axs[2].plot(x_plot_mm, F_plot_N, linewidth=2)
        axs[2].axhline(0.0, linewidth=1, alpha=0.5)
        axs[2].set_xlabel('dz (mm)')
        axs[2].set_ylabel('F (N)')
        axs[2].set_title('fuerza axial F(z) = -dW/dz (derivada analítica)')
        axs[2].grid(True, alpha=0.3)
    
        axs[3].plot(x_plot_mm, K_show_plot / 1000.0, linewidth=2)
        axs[3].axhline(0.0, linewidth=1, alpha=0.5)
        axs[3].set_xlabel('dz (mm)')
        axs[3].set_ylabel('K (N/mm)')
        axs[3].set_title('rigidez mostrada' + (' con convención paper' if args.stiffness_sign == 'paper' else ' física'))
        axs[3].grid(True, alpha=0.3)
    
        fig.suptitle(
            f'grado={args.degree} | R²={r2:.6f} | used={len(used)} / {len(df)} | '
            f'F(z=0 mm)={F_eval:.4g} N | Kshown(z=0 mm)={K_show_eval:.4g} N/m',
            fontsize=12,
        )
        fig.savefig(fig_path, dpi=180)
        plt.close(fig)
    else:
        fig_path = None

    print('=== RESULTADOS ===')
    print(f'Puntos CSV totales: {raw_n}')
    print(f'Después de filtro básico: {after_basic}')
    print(f'Después de deduplicar: {after_dedup}')
    print(f'Después de ventana: {after_window}')
    print(f'Puntos usados ajuste: {len(used)} / {len(df)}')
    print(f'R² energía: {r2:.9f}')
    print(f'F en z={args.eval_at_mm:.6g} mm: {F_eval:.9g} N')
    print(f'K física en z={args.eval_at_mm:.6g} mm: {K_phys_eval:.9g} N/m')
    print(f'K mostrada en z={args.eval_at_mm:.6g} mm: {K_show_eval:.9g} N/m')
    print(f'Curva ajustada: {curve_path}')
    print(f'Puntos + diagnóstico: {points_path}')
    print(f'Resumen: {summary_path}')
    if HAS_MPL and fig_path is not None:
        print(f'Figura: {fig_path}')
    else:
        print('Figura: omitida (matplotlib no disponible)')


if __name__ == '__main__':
    main()
