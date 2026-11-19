
from pathlib import Path
import hashlib

import geopandas as gpd
import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent

STAT_FILE = (
    DATA_DIR / "prepared_statistics" / "regions_statistics.csv"
)
GEO_FILE = DATA_DIR / "russia_regions.geojson"
CROSSWALK_FILE = (
    DATA_DIR / "prepared_geography" / "region_crosswalk.csv"
)

OUT_DIR = DATA_DIR / "merged_geodata"
OUT_DIR.mkdir(parents=True, exist_ok=True)

log = []


def report(message):
    print(message)
    log.append(str(message))


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    if path.suffix.lower() == ".csv":
        return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    digest = hashlib.sha256()

    with path.open("rb") as source:
        for block in iter(
            lambda: source.read(1024 * 1024), b""
        ):
            digest.update(block)

    return digest.hexdigest()


# 1. Загрузка

for path in (STAT_FILE, GEO_FILE, CROSSWALK_FILE):
    if not path.is_file():
        raise FileNotFoundError(f"Не найден файл: {path}")

stats = pd.read_csv(
    STAT_FILE,
    sep=";",
    encoding="utf-8-sig",
    dtype={
        "object_okato": "string",
        "object_oktmo": "string",
    },
)

geo = gpd.read_file(GEO_FILE).reset_index(drop=True)

crosswalk = pd.read_csv(
    CROSSWALK_FILE,
    sep=";",
    encoding="utf-8-sig",
    dtype={
        "object_okato": "string",
        "object_oktmo": "string",
    },
)

report("=== ИСХОДНЫЕ ФАЙЛЫ ===")

for path in (STAT_FILE, GEO_FILE, CROSSWALK_FILE):
    hash_kind = "SHA256 (LF-normalized CSV)" if path.suffix == ".csv" else "SHA256"
    report(f"{path.name} | {hash_kind}: {sha256(path)}")


# 2. Проверка структуры

required_stats = {
    "year",
    "object_name",
    "object_okato",
    "object_oktmo",
    "doctors_per_10000",
    "life_expectancy_years",
}

required_crosswalk = {
    "geo_feature_index",
    "geo_region",
    "stat_region",
    "object_okato",
    "object_oktmo",
    "match_type",
    "extent_needs_review",
}

check(
    required_stats.issubset(stats.columns),
    "В статистике отсутствуют обязательные поля",
)

check(
    required_crosswalk.issubset(crosswalk.columns),
    "В таблице соответствий отсутствуют обязательные поля",
)

check(
    "region" in geo.columns,
    "В GeoJSON нет поля region",
)

check(
    len(stats) == len(geo) == len(crosswalk) == 85,
    "Ожидалось 85 строк в каждом наборе",
)

check(
    stats["object_okato"].is_unique
    and stats["object_name"].is_unique,
    "В статистике есть дубликаты регионов",
)

check(
    geo["region"].is_unique
    and geo["region"].notna().all(),
    "В географии есть дубликаты или пропуски названий",
)

check(
    crosswalk["geo_region"].is_unique
    and crosswalk["stat_region"].is_unique
    and crosswalk["object_okato"].is_unique
    and crosswalk["geo_feature_index"].is_unique,
    "В таблице соответствий есть дубликаты",
)

check(
    stats["year"].eq(2021).all(),
    "Найден год, отличный от 2021",
)

check(
    geo.crs is not None and geo.crs.to_epsg() == 4326,
    "Ожидалась EPSG:4326",
)

check(
    geo.geometry.notna().all()
    and not geo.geometry.is_empty.any()
    and geo.geometry.is_valid.all()
    and geo.geometry.geom_type.isin(
        ["Polygon", "MultiPolygon"]
    ).all(),
    "Обнаружены проблемные геометрии",
)


# 3. Проверка показателей

for col in (
    "doctors_per_10000",
    "life_expectancy_years",
):
    stats[col] = pd.to_numeric(
        stats[col], errors="coerce"
    )

    check(
        np.isfinite(
            stats[col].to_numpy(dtype=float)
        ).all(),
        f"Некорректные значения {col}",
    )

    check(
        stats[col].ge(0).all(),
        f"Отрицательные значения {col}",
    )


# 4. Проверка соответствия GeoJSON и crosswalk

crosswalk["geo_feature_index"] = pd.to_numeric(
    crosswalk["geo_feature_index"],
    errors="raise",
).astype(int)

check(
    set(crosswalk["geo_feature_index"]) == set(range(85)),
    "Индексы географических объектов не совпадают",
)

check(
    crosswalk[list(required_crosswalk)].notna().all().all(),
    "В обязательных полях crosswalk есть пропуски",
)

geo_indexed = geo.copy()

geo_indexed.insert(
    0, "geo_feature_index", range(len(geo_indexed))
)

geo_indexed = geo_indexed.rename(
    columns={"region": "geo_region"}
)

matched = geo_indexed.merge(
    crosswalk,
    on=["geo_feature_index", "geo_region"],
    how="outer",
    validate="one_to_one",
    indicator="_geo_match",
)

check(
    matched["_geo_match"].eq("both").all(),
    "Таблица соответствий не подходит к исходному GeoJSON",
)

matched = matched.drop(columns="_geo_match")


# 5. Объединение с показателями

merged = matched.merge(
    stats,
    on="object_okato",
    how="outer",
    validate="one_to_one",
    indicator="_stat_match",
    suffixes=("_crosswalk", ""),
)

check(
    merged["_stat_match"].eq("both").all(),
    "Не все регионы удалось объединить",
)

check(
    merged["stat_region"].eq(
        merged["object_name"]
    ).all(),
    "Названия регионов не совпадают",
)

check(
    merged["object_oktmo_crosswalk"].eq(
        merged["object_oktmo"]
    ).all(),
    "Коды ОКТМО не совпадают",
)

check(
    merged["geo_region"].is_unique
    and merged["object_okato"].is_unique,
    "После объединения возникли дубликаты",
)

merged = merged.sort_values(
    "geo_feature_index"
).reset_index(drop=True)


# 6. Формирование итогового набора

columns = [
    "year",
    "object_name",
    "object_okato",
    "object_oktmo",
    "doctors_per_10000",
    "life_expectancy_years",
    "geo_region",
    "federal_district",
    "match_type",
    "extent_needs_review",
    "geo_feature_index",
    "geometry",
]

check(
    set(columns).issubset(merged.columns),
    "Не хватает столбцов для итоговой базы",
)

result = gpd.GeoDataFrame(
    merged[columns].copy(),
    geometry="geometry",
    crs=geo.crs,
)

district_corrections = {
    "Республика Крым": "Южный",
    "г. Севастополь": "Южный",
    "Республика Бурятия": "Дальневосточный",
    "Забайкальский край": "Дальневосточный",
}

for region, district in district_corrections.items():
    result.loc[
        result["object_name"] == region,
        "federal_district"
    ] = district

# 7. Проверка сохранности данных

expected = stats.set_index("object_okato")
actual = result.set_index("object_okato")

check(
    set(expected.index) == set(actual.index),
    "Не все ОКАТО сохранились",
)

for col in (
    "doctors_per_10000",
    "life_expectancy_years",
):
    check(
        np.array_equal(
            expected.loc[
                actual.index, col
            ].to_numpy(),
            actual[col].to_numpy(),
        ),
        f"Значения {col} изменились",
    )

check(
    all(
        a.equals(b)
        for a, b in zip(
            result.geometry, geo.geometry
        )
    ),
    "Геометрия изменилась при объединении",
)


# 8. Сохранение готовых данных

geojson_out = (
    OUT_DIR / "regions_2021_geospatial.geojson"
)

csv_out = (
    OUT_DIR / "regions_2021_attributes.csv"
)

audit_out = OUT_DIR / "merge_audit.txt"

result.to_file(
    geojson_out,
    driver="GeoJSON",
    index=False,
)

result.drop(columns="geometry").to_csv(
    csv_out,
    sep=";",
    index=False,
    encoding="utf-8-sig",
)


# 9. Контрольное чтение результата

reloaded = gpd.read_file(geojson_out)

check(
    len(reloaded) == 85
    and reloaded["object_okato"].is_unique,
    "Ошибка при сохранении GeoJSON",
)

check(
    reloaded.crs is not None
    and reloaded.crs.to_epsg() == 4326,
    "Потеряна система координат",
)

check(
    reloaded.geometry.is_valid.all(),
    "После сохранения появились невалидные геометрии",
)

for col in (
    "doctors_per_10000",
    "life_expectancy_years",
):
    check(
        np.allclose(
            reloaded[col],
            result[col],
            rtol=0,
            atol=1e-10,
        ),
        f"После сохранения изменились значения {col}",
    )


# 10. Итоги

report("\n=== ИТОГ ===")
report(f"Регионов: {len(result)}")
report(
    f"Уникальных ОКАТО: "
    f"{result['object_okato'].nunique()}"
)
report(
    f"Уникальных ОКТМО: "
    f"{result['object_oktmo'].nunique()}"
)
report(
    f"Валидных геометрий: "
    f"{int(result.geometry.is_valid.sum())}"
)
report(f"CRS: {result.crs}")

report(
    "Пропусков X/Y нет: "
    f"{result[['doctors_per_10000', 'life_expectancy_years']].notna().all().all()}"
)

report(
    "Требуют проверки охвата: "
    f"{result.loc[result['extent_needs_review'].astype(str).str.lower().eq('true'), 'object_name'].tolist()}"
)

report("ОКАТО получены из статистики.")
report("ОКТМО сохранены без изменений.")
report("Федеральные округа уточнены для четырёх территорий: Республика Крым, г. Севастополь, Республика Бурятия, Забайкальский край.")
report("Историческая точность границ на 2021 год не подтверждена.")
report(f"CSV: {csv_out.relative_to(DATA_DIR)}")
report(f"GeoJSON: {geojson_out.relative_to(DATA_DIR)}")

audit_out.write_text(
    "\n".join(log) + "\n",
    encoding="utf-8",
    newline="\n",
)

print(f"\nГотово. Журнал: {audit_out}")
