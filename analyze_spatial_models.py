"""Reproduce the common weights, OLS/SAR/SEM comparison, and maps for team 1."""

from __future__ import annotations

import json
import platform
import warnings
from pathlib import Path

import geopandas as gpd
import libpysal
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import scipy
import spreg
import statsmodels.api as sm
from esda.moran import Moran, Moran_Local
from libpysal.weights import KNN, Queen, W
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "merged_geodata" / "regions_2021_geospatial.geojson"
OUT = ROOT / "results"
MAPS = ROOT / "maps"
SEED = 20261118
PERMUTATIONS = 999
LOCAL_PERMUTATIONS = 9999
MAP_SOURCE = "© OpenStreetMap contributors · ODbL 1.0 · openstreetmap.org/copyright | Росстат, «Если быть точным», CC BY 4.0"
PROJECTION = "+proj=laea +lat_0=62 +lon_0=100 +datum=WGS84 +units=m +type=crs"


def save_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUT / name, sep=";", index=False, encoding="utf-8-sig")


def prepare() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(SOURCE).sort_values("object_okato").reset_index(drop=True)
    gdf["object_okato"] = gdf["object_okato"].astype(str).str.zfill(8)
    assert len(gdf) == 85 and gdf.object_okato.is_unique
    assert gdf.crs.to_epsg() == 4326 and gdf.geometry.is_valid.all()
    assert gdf[["doctors_per_10000", "life_expectancy_years"]].notna().all().all()
    assert gdf.year.eq(2021).all()
    return gdf


def fill_islands(queen: W, coords: np.ndarray, ids: list[str]) -> tuple[W, list[dict]]:
    neighbors = {key: set(value) for key, value in queen.neighbors.items()}
    positions = {key: i for i, key in enumerate(ids)}
    bridges = []
    for island in queen.islands:
        i = positions[island]
        distance = np.linalg.norm(coords - coords[i], axis=1)
        distance[i] = np.inf
        j = int(np.argmin(distance))
        other = ids[j]
        neighbors[island].add(other)
        neighbors[other].add(island)
        bridges.append({"island_okato": island, "neighbor_okato": other,
                        "distance_km": float(distance[j] / 1000)})
    weights = W({key: sorted(value) for key, value in neighbors.items()}, id_order=ids)
    weights.transform = "R"
    assert weights.n == len(ids) and not weights.islands
    assert weights.id_order == ids
    assert np.allclose(weights.sparse.diagonal(), 0)
    assert np.allclose(np.asarray(weights.sparse.sum(axis=1)).ravel(), 1)
    return weights, bridges


def make_weights(gdf: gpd.GeoDataFrame) -> tuple[W, W, dict]:
    ids = gdf.object_okato.tolist()
    queen = Queen.from_dataframe(gdf, ids=ids)
    points = gdf.to_crs(PROJECTION).representative_point()
    coords = np.column_stack([points.x.to_numpy(), points.y.to_numpy()])
    primary, bridges = fill_islands(queen, coords, ids)
    knn = KNN.from_array(coords, k=4, ids=ids).symmetrize()
    knn.transform = "R"
    assert knn.id_order == ids and not knn.islands
    metadata = {
        "definition": "Queen polygon contiguity; each original island linked bidirectionally to its nearest projected representative point; row standardized",
        "projection_for_bridge_distance": PROJECTION,
        "ids": "object_okato, ascending, eight-character strings",
        "n": len(ids), "queen_edges_undirected": queen.nonzero // 2,
        "queen_islands": queen.islands, "queen_components": queen.n_components,
        "bridge_edges": bridges, "final_components": primary.n_components,
        "final_edges_directed": primary.nonzero,
        "mean_neighbors": primary.mean_neighbors,
        "sensitivity_definition": "symmetrized four nearest projected representative points; row standardized",
        "sensitivity_components": knn.n_components,
    }
    (OUT / "weights_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    # GAL stores topology; restore transform='R' after reading.
    primary.to_file(str(OUT / "weights_queen_bridged.gal"))
    save_csv(pd.DataFrame({"row_index": range(len(ids)), "object_okato": ids,
                           "object_name": gdf.object_name}), "weights_order.csv")
    save_csv(primary.to_adjlist(drop_islands=True).rename(columns={"focal": "source_okato", "neighbor": "target_okato"}),
             "weights_edges.csv")
    return primary, knn, metadata


def fit_models(gdf: gpd.GeoDataFrame, w: W) -> dict:
    x = gdf[["doctors_per_10000"]].to_numpy(float)
    y = gdf[["life_expectancy_years"]].to_numpy(float)
    common = dict(name_y="life_expectancy_years", name_x=["doctors_per_10000"],
                  name_w="weights_queen_bridged", name_ds="regions_2021")
    ols = spreg.OLS(y, x, w=w, spat_diag=True, moran=True, **common)
    sar = spreg.ML_Lag(y, x, w=w, method="full", spat_diag=False, **common)
    sem = spreg.ML_Error(y, x, w=w, method="full", **common)
    return {"OLS": ols, "SAR": sar, "SEM": sem}


def moran_test(values: np.ndarray, w: W) -> dict:
    np.random.seed(SEED)
    m = Moran(np.asarray(values).ravel(), w, permutations=PERMUTATIONS)
    return {"I": float(m.I), "p_permutation": float(m.p_sim)}


def model_results(gdf: gpd.GeoDataFrame, w: W, models: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = gdf.life_expectancy_years.to_numpy(float)
    frame = gdf.drop(columns="geometry").copy()
    rows, params = [], []
    for name, model in models.items():
        prediction = np.asarray(model.predy_e if name == "SAR" else model.predy).ravel()
        error = y - prediction
        innovation = np.asarray(model.e_filtered if name == "SEM" else model.u).ravel()
        frame[f"{name.lower()}_predicted"] = prediction
        frame[f"{name.lower()}_prediction_error"] = error
        frame[f"{name.lower()}_innovation"] = innovation
        diag = moran_test(innovation, w)
        # Use an explicit common parameter count instead of library-specific AIC/BIC.
        # Count two regression coefficients plus the spatial parameter for SAR/SEM.
        k = 2 if name == "OLS" else 3
        loglik = float(model.logll)
        rows.append({"model": name, "n": len(y), "parameters_counted": k,
                     "log_likelihood": loglik,
                     "AIC": float(2 * k - 2 * loglik),
                     "BIC": float(np.log(len(y)) * k - 2 * loglik),
                     "RMSE_in_sample": float(np.sqrt(np.mean(error ** 2))),
                     "MAE_in_sample": float(np.mean(np.abs(error))),
                     "Moran_I_innovation": diag["I"], "Moran_p_innovation": diag["p_permutation"],
                     "spatial_parameter": float(model.rho if name == "SAR" else model.lam) if name != "OLS" else np.nan})
        for i, label in enumerate(model.name_x):
            params.append({"model": name, "parameter": label,
                           "coefficient": float(np.asarray(model.betas).ravel()[i]),
                           "std_error": float(np.asarray(model.std_err).ravel()[i]),
                           "p_value": float(model.z_stat[i][1] if name != "OLS" else model.t_stat[i][1])})
    return pd.DataFrame(rows), pd.DataFrame(params), frame


def local_moran(gdf: gpd.GeoDataFrame, w: W) -> pd.DataFrame:
    output = pd.DataFrame({"object_okato": gdf.object_okato})
    for variable, prefix in (("life_expectancy_years", "life"), ("doctors_per_10000", "doctors")):
        np.random.seed(SEED)
        lisa = Moran_Local(gdf[variable].to_numpy(float), w, permutations=LOCAL_PERMUTATIONS,
                           seed=SEED, n_jobs=1, keep_simulations=True)
        # Explicit two-sided equal-tail conditional randomization test;
        # inclusive ties and add-one correction in both tails.
        sim = np.asarray(lisa.sim)
        observed = np.asarray(lisa.Is)
        assert sim.shape == (LOCAL_PERMUTATIONS, len(gdf))
        ties = np.isclose(sim, observed, rtol=1e-12, atol=1e-12)
        lower = (np.count_nonzero((sim < observed) | ties, axis=0) + 1) / (LOCAL_PERMUTATIONS + 1)
        upper = (np.count_nonzero((sim > observed) | ties, axis=0) + 1) / (LOCAL_PERMUTATIONS + 1)
        p_two = np.minimum(1.0, 2 * np.minimum(lower, upper))
        rejected, qvals, _, _ = multipletests(p_two, alpha=0.05, method="fdr_bh")
        rejected_by, qvals_by, _, _ = multipletests(p_two, alpha=0.05, method="fdr_by")
        labels = np.array(["HH", "LH", "LL", "HL"])
        output[f"{prefix}_local_I"] = lisa.Is
        output[f"{prefix}_local_p"] = p_two
        output[f"{prefix}_local_p_lower"] = lower
        output[f"{prefix}_local_p_upper"] = upper
        output[f"{prefix}_local_q_fdr"] = qvals
        output[f"{prefix}_LISA_5pct_FDR"] = np.where(rejected, labels[lisa.q - 1], "NS")
        output[f"{prefix}_local_q_fdr_by"] = qvals_by
        output[f"{prefix}_LISA_5pct_FDR_BY"] = np.where(rejected_by, labels[lisa.q - 1], "NS")
    return output


def add_map(ax, gdf: gpd.GeoDataFrame, column: str, title: str, cmap: str,
            vmin: float | None = None, vmax: float | None = None) -> None:
    gdf.plot(column=column, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
             legend=True, legend_kwds={"shrink": 0.58},
             edgecolor="#777777", linewidth=0.22, missing_kwds={"color": "lightgrey"})
    ax.set_title(title, fontsize=12, loc="left", pad=10)
    ax.set_axis_off()


def make_maps(gdf: gpd.GeoDataFrame, results: pd.DataFrame, chosen: str,
              local: pd.DataFrame) -> None:
    display = gdf.merge(results[["object_okato", f"{chosen.lower()}_predicted",
                                 f"{chosen.lower()}_prediction_error",
                                 f"{chosen.lower()}_innovation"]], on="object_okato", validate="one_to_one")
    display = display.merge(local, on="object_okato", validate="one_to_one").to_crs(PROJECTION)
    plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(19, 7), layout="constrained")
    add_map(axes[0], display, "doctors_per_10000", "Врачи на 10 000 жителей · 2021", "YlGnBu")
    add_map(axes[1], display, "life_expectancy_years", "Ожидаемая продолжительность жизни · 2021, лет", "YlOrRd")
    fig.supxlabel(MAP_SOURCE, fontsize=9, color="#555555")
    fig.savefig(MAPS / "01_indicators.png", dpi=200)
    plt.close(fig)

    pred = f"{chosen.lower()}_predicted"
    err = f"{chosen.lower()}_prediction_error"
    low = min(display.life_expectancy_years.min(), display[pred].min())
    high = max(display.life_expectancy_years.max(), display[pred].max())
    span = max(abs(display[err].min()), abs(display[err].max()))
    fig, axes = plt.subplots(1, 3, figsize=(22, 6), layout="constrained")
    add_map(axes[0], display, "life_expectancy_years", "Факт · лет", "YlOrRd", low, high)
    add_map(axes[1], display, pred, f"Расчёт {chosen} · лет", "YlOrRd", low, high)
    add_map(axes[2], display, err, "Факт − расчёт · лет", "RdBu_r", -span, span)
    fig.supxlabel(MAP_SOURCE, fontsize=9, color="#555555")
    fig.savefig(MAPS / "02_final_model.png", dpi=200)
    plt.close(fig)

    innovation = f"{chosen.lower()}_innovation"
    span = max(abs(display[innovation].min()), abs(display[innovation].max()))
    fig, axes = plt.subplots(1, 2, figsize=(19, 7), layout="constrained")
    add_map(axes[0], display, err, "Факт − расчёт · лет", "RdBu_r",
            -max(abs(display[err].min()), abs(display[err].max())),
            max(abs(display[err].min()), abs(display[err].max())))
    add_map(axes[1], display, innovation, f"Инновации {chosen} после пространственной фильтрации", "RdBu_r", -span, span)
    fig.supxlabel(MAP_SOURCE, fontsize=9, color="#555555")
    fig.savefig(MAPS / "04_residual_diagnostics.png", dpi=200)
    plt.close(fig)

    color = {"HH": "#b2182b", "LL": "#2166ac", "HL": "#ef8a62", "LH": "#67a9cf", "NS": "#dddddd"}
    fig, axes = plt.subplots(1, 2, figsize=(19, 7), layout="constrained")
    for ax, col, title in ((axes[0], "doctors_LISA_5pct_FDR", "Локальный Moran · врачи"),
                           (axes[1], "life_LISA_5pct_FDR", "Локальный Moran · продолжительность жизни")):
        for label in ("HH", "LL", "HL", "LH", "NS"):
            subset = display[display[col] == label]
            if len(subset):
                subset.plot(ax=ax, color=color[label], edgecolor="#777777", linewidth=0.22, label=label)
        ax.set_title(title + " · BH-FDR 5%, двусторонний тест", fontsize=12, loc="left")
        ax.set_axis_off()
        ax.legend(handles=[Patch(facecolor=color[k], edgecolor="#777777", label=k) for k in ("HH", "LL", "HL", "LH", "NS")], loc="lower left", ncol=5, fontsize=8)
    fig.supxlabel(MAP_SOURCE, fontsize=9, color="#555555")
    fig.savefig(MAPS / "03_local_moran.png", dpi=200)
    plt.close(fig)
    display.to_crs(4326).to_file(OUT / "regional_results.geojson", driver="GeoJSON")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    MAPS.mkdir(exist_ok=True)
    gdf = prepare()
    w, knn, metadata = make_weights(gdf)
    models = fit_models(gdf, w)
    comparison, params, regional = model_results(gdf, w, models)
    local = local_moran(gdf, w)
    chosen = str(comparison.loc[comparison.AIC.idxmin(), "model"])
    save_csv(comparison, "model_comparison.csv")
    save_csv(params, "model_coefficients.csv")
    save_csv(regional, "regional_predictions.csv")
    save_csv(local, "local_moran.csv")

    diagnostics = {"moran_doctors": moran_test(gdf.doctors_per_10000, w),
                   "moran_life": moran_test(gdf.life_expectancy_years, w),
                   "pearson": dict(zip(("r", "p"), map(float, pearsonr(gdf.doctors_per_10000, gdf.life_expectancy_years)))),
                   "spearman": dict(zip(("rho", "p"), map(float, spearmanr(gdf.doctors_per_10000, gdf.life_expectancy_years)))),
                   "OLS_LM_error": list(map(float, models["OLS"].lm_error)),
                   "OLS_LM_lag": list(map(float, models["OLS"].lm_lag)),
                   "OLS_robust_LM_error": list(map(float, models["OLS"].rlm_error)),
                   "OLS_robust_LM_lag": list(map(float, models["OLS"].rlm_lag)),
                   "OLS_Jarque_Bera": models["OLS"].jarque_bera,
                   "OLS_Breusch_Pagan": models["OLS"].breusch_pagan,
                   "OLS_Koenker_Bassett": models["OLS"].koenker_bassett,
                   "selected_by_AIC": chosen}
    diagnostics["LISA_method"] = {"permutations": LOCAL_PERMUTATIONS, "seed": SEED, "alternative": "two-sided equal-tail conditional randomization", "ties": "inclusive, tolerance 1e-12", "adjustment": "BH primary, BY sensitivity; separately per indicator, 85 tests"}
    diagnostics["LISA_FDR_counts"] = {col: local[col].value_counts().to_dict()
                                      for col in ("doctors_LISA_5pct_FDR", "life_LISA_5pct_FDR")}
    diagnostics["LISA_FDR_BY_counts"] = {col: local[col].value_counts().to_dict()
                                         for col in ("doctors_LISA_5pct_FDR_BY", "life_LISA_5pct_FDR_BY")}
    alt_models = fit_models(gdf, knn)
    alt_comparison, alt_params, _ = model_results(gdf, knn, alt_models)
    save_csv(alt_comparison, "sensitivity_knn4_models.csv")
    save_csv(alt_params, "sensitivity_knn4_coefficients.csv")
    # HC3 and influence refer to OLS only; ML spatial model p-values assume its likelihood specification.
    hc3 = sm.OLS(gdf.life_expectancy_years.to_numpy(float),
                 sm.add_constant(gdf.doctors_per_10000.to_numpy(float))).fit(cov_type="HC3")
    diagnostics["OLS_HC3_doctors_p"] = float(hc3.pvalues[1])
    influence = hc3.get_influence()
    cooks = influence.cooks_distance[0]
    save_csv(pd.DataFrame({"object_okato": gdf.object_okato, "object_name": gdf.object_name,
                           "OLS_Cooks_D": cooks}).sort_values("OLS_Cooks_D", ascending=False),
             "OLS_influence.csv")
    diagnostics["max_Cooks_D"] = float(np.max(cooks))
    diagnostics["max_Cooks_region"] = str(gdf.iloc[np.argmax(cooks)].object_name)
    sensitivity_rows = []
    for scenario, mask in (
        ("exclude_max_Cooks_D", gdf.index != int(np.argmax(cooks))),
        ("exclude_extent_needs_review", ~gdf.extent_needs_review.astype(bool)),
    ):
        sample = gdf.loc[mask].reset_index(drop=True)
        ids = sample.object_okato.tolist()
        queen = Queen.from_dataframe(sample, ids=ids)
        points = sample.to_crs(PROJECTION).representative_point()
        coords = np.column_stack([points.x.to_numpy(), points.y.to_numpy()])
        subset_weights, _ = fill_islands(queen, coords, ids)
        subset_models = fit_models(sample, subset_weights)
        subset_comparison, subset_params, _ = model_results(sample, subset_weights, subset_models)
        for _, row in subset_comparison.iterrows():
            coefficient = subset_params.loc[
                (subset_params.model == row["model"]) &
                (subset_params.parameter == "doctors_per_10000")].iloc[0]
            sensitivity_rows.append({"scenario": scenario, "model": row["model"],
                                     "n": len(sample), "AIC": row.AIC,
                                     "doctors_coefficient": coefficient.coefficient,
                                     "doctors_p_value": coefficient.p_value,
                                     "Moran_p_innovation": row.Moran_p_innovation})
    save_csv(pd.DataFrame(sensitivity_rows), "sensitivity_exclusions.csv")
    (OUT / "diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2,
                   default=lambda value: value.item() if isinstance(value, np.generic) else str(value)),
        encoding="utf-8")
    make_maps(gdf, regional, chosen, local)
    versions = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                "geopandas": gpd.__version__, "libpysal": libpysal.__version__, "spreg": spreg.__version__,
                "scipy": scipy.__version__, "matplotlib": matplotlib.__version__}
    (OUT / "versions.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")
    print(comparison.to_string(index=False))
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="The weights matrix is not fully connected")
    main()
