
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.validation import explain_validity


DATA_DIR = Path(__file__).resolve().parent
GEO_FILE = DATA_DIR / "russia_regions.geojson"
STAT_FILE = DATA_DIR / "prepared_statistics" / "regions_statistics.csv"

OUT_DIR = DATA_DIR / "prepared_geography"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Название в GeoJSON -> название в статистике.
NAME_EQUIVALENTS = {
    "Москва": "г. Москва",
    "Санкт-Петербург": "г. Санкт-Петербург",
    "Севастополь": "г. Севастополь",
    "Республика Северная Осетия — Алания":
        "Республика Северная Осетия – Алания",
    "Ханты-Мансийский автономный округ — Югра":
        "Ханты-Мансийский автономный округ – Югра",
    "Архангельская область":
        "Архангельская область без автономного округа",
    "Тюменская область":
        "Тюменская область без автономных округов",
}

EXTENT_TO_VERIFY = {
    "Архангельская область",
    "Тюменская область",
}

log = []


def report(message):
    print(message)
    log.append(str(message))


def check(condition, message):
    if not condition:
        raise ValueError(message)


# 1. Загрузка

for path in (GEO_FILE, STAT_FILE):
    if not path.is_file():
        raise FileNotFoundError(f"Не найден файл: {path}")

geo = gpd.read_file(GEO_FILE)

stats = pd.read_csv(
    STAT_FILE,
    sep=";",
    encoding="utf-8-sig",
    dtype={
        "object_okato": "string",
        "object_oktmo": "string",
    },
)


# 2. Проверка статистики

required_stats = {
    "year",
    "object_name",
    "object_okato",
    "object_oktmo",
    "doctors_per_10000",
    "life_expectancy_years",
}

check(
    required_stats.issubset(stats.columns),
    "В статистике отсутствуют обязательные столбцы",
)

check(len(stats) == 85, "Ожидалось 85 регионов")
check(stats["year"].eq(2021).all(), "Неверный год")
check(stats["object_name"].is_unique, "Повторяются названия")
check(stats["object_okato"].is_unique, "Повторяются ОКАТО")

check(
    stats[
        ["doctors_per_10000", "life_expectancy_years"]
    ].notna().all().all(),
    "В X или Y есть пропуски",
)

report("=== СТАТИСТИКА ===")
report(f"Регионов: {len(stats)}")
report("Год: 2021")
report("Пропусков X/Y нет")


# 3. Проверка географии

check("region" in geo.columns, "В GeoJSON нет поля region")
check(len(geo) == 85, "Ожидалось 85 географических объектов")
check(geo["region"].notna().all(), "Есть регионы без названия")
check(geo["region"].is_unique, "Названия регионов повторяются")

check(
    geo.crs is not None and geo.crs.to_epsg() == 4326,
    "Ожидалась система координат EPSG:4326",
)

geometry_audit = pd.DataFrame({
    "geo_region": geo["region"],
    "geometry_type": geo.geometry.geom_type,
    "is_missing": geo.geometry.isna(),
    "is_empty": geo.geometry.is_empty,
    "is_valid": geo.geometry.is_valid,
    "validity_reason": [
        explain_validity(g) if g is not None else "missing"
        for g in geo.geometry
    ],
    "min_lon": geo.bounds.minx,
    "min_lat": geo.bounds.miny,
    "max_lon": geo.bounds.maxx,
    "max_lat": geo.bounds.maxy,
})

geometry_audit.to_csv(
    OUT_DIR / "geometry_audit.csv",
    sep=";",
    index=False,
    encoding="utf-8-sig",
)

invalid = geometry_audit.loc[
    geometry_audit["is_missing"]
    | geometry_audit["is_empty"]
    | ~geometry_audit["is_valid"]
    | ~geometry_audit["geometry_type"].isin(
        ["Polygon", "MultiPolygon"]
    )
]

report("\n=== ГЕОГРАФИЯ ===")
report(f"Объектов: {len(geo)}")
report(f"CRS: {geo.crs}")
report(f"Проблемных геометрий: {len(invalid)}")

check(invalid.empty, "Обнаружены проблемные геометрии")


# 4. Проверка пересечений по площади

overlaps = []

left_idx, right_idx = geo.sindex.query(
    geo.geometry,
    predicate="intersects",
)

for i, j in zip(left_idx, right_idx):
    if i >= j:
        continue

    intersection_area = (
        geo.geometry.iloc[i]
        .intersection(geo.geometry.iloc[j])
        .area
    )

    if intersection_area > 1e-10:
        overlaps.append({
            "region_1": geo.iloc[i]["region"],
            "region_2": geo.iloc[j]["region"],
            "overlap_area_deg2": intersection_area,
        })

pd.DataFrame(
    overlaps,
    columns=[
        "region_1",
        "region_2",
        "overlap_area_deg2",
    ],
).to_csv(
    OUT_DIR / "overlaps.csv",
    sep=";",
    index=False,
    encoding="utf-8-sig",
)

report(f"Пар с площадным перекрытием: {len(overlaps)}")

check(
    len(overlaps) == 0,
    "Геометрии регионов перекрываются по площади",
)


# 5. Сопоставление названий

geo_names = set(geo["region"])
stats_names = set(stats["object_name"])

check(
    set(NAME_EQUIVALENTS).issubset(geo_names),
    "Некоторые названия из словаря отсутствуют в GeoJSON",
)

crosswalk = pd.DataFrame({
    "geo_feature_index": range(len(geo)),
    "geo_region": geo["region"],
})

crosswalk["stat_region"] = (
    crosswalk["geo_region"].replace(NAME_EQUIVALENTS)
)

crosswalk["match_type"] = crosswalk.apply(
    lambda row: (
        "exact_name"
        if row["geo_region"] == row["stat_region"]
        else "explicit_name_mapping"
    ),
    axis=1,
)

crosswalk["extent_needs_review"] = (
    crosswalk["geo_region"].isin(EXTENT_TO_VERIFY)
)

not_in_stats = sorted(
    set(crosswalk["stat_region"]) - stats_names
)

not_in_geo = sorted(
    stats_names - set(crosswalk["stat_region"])
)

check(
    not not_in_stats and not not_in_geo,
    f"Несопоставленные регионы: {not_in_stats}, {not_in_geo}",
)

check(
    crosswalk["stat_region"].is_unique,
    "Один регион статистики сопоставлен несколько раз",
)


# 6. Добавление кодов из статистики

crosswalk = crosswalk.merge(
    stats[[
        "object_name",
        "object_okato",
        "object_oktmo",
    ]],
    left_on="stat_region",
    right_on="object_name",
    how="left",
    validate="one_to_one",
    indicator=True,
)

check(
    crosswalk["_merge"].eq("both").all(),
    "Не все регионы сопоставились",
)

crosswalk = crosswalk.drop(
    columns=["object_name", "_merge"]
)

crosswalk["review_note"] = (
    crosswalk["extent_needs_review"].map({
        True: "Проверить охват области без автономных округов",
        False: "Историческая точность границ не подтверждена",
    })
)

crosswalk.to_csv(
    OUT_DIR / "region_crosswalk.csv",
    sep=";",
    index=False,
    encoding="utf-8-sig",
)


# 7. Итоги

report("\n=== СОПОСТАВЛЕНИЕ ===")
report(
    "Точные совпадения названий: "
    f"{(crosswalk['match_type'] == 'exact_name').sum()}"
)
report(
    "Совпадения по словарю: "
    f"{(crosswalk['match_type'] == 'explicit_name_mapping').sum()}"
)
report(f"Всего сопоставлено: {len(crosswalk)}")

report("\nТерритории с различающимися названиями:")
report(
    crosswalk.loc[
        crosswalk["match_type"] == "explicit_name_mapping",
        ["geo_region", "stat_region", "object_okato"],
    ].to_string(index=False)
)

report("\n=== ИТОГ ===")
report("85 объектов сопоставлены с 85 строками статистики.")
report("Коды ОКАТО получены из статистики.")
report("Исходные файлы не изменены.")
report("Границы на 2021 год независимо не подтверждены.")
report("Охват двух областей без АО требует проверки.")

(OUT_DIR / "geography_audit.txt").write_text(
    "\n".join(log) + "\n",
    encoding="utf-8",
)

print(f"\nГотово: {OUT_DIR}")
