
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent

HEALTHCARE_FILE = DATA_DIR / "healthcare.csv"
POPULATION_FILE = DATA_DIR / "population.csv"

OUT_DIR = DATA_DIR / "prepared_statistics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEAR = 2021
X_CODE = "05040200"
Y_CODE = "01160100"

EXCLUDED_AGGREGATES = {
    "Архангельская область",
    "Тюменская область",
}

REQUIRED = {
    "section",
    "indicator_name",
    "indicator_unit",
    "indicator_code",
    "object_name",
    "object_level",
    "object_oktmo",
    "object_okato",
    "year",
    "indicator_value",
    "comment",
}

notes = []


def note(message):
    print(message)
    notes.append(str(message))


def read_source(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Исходный файл не найден: {path}"
        )

    df = pd.read_csv(
        path,
        sep=";",
        encoding="utf-8-sig",
        dtype={
            "indicator_code": "string",
            "object_okato": "string",
            "object_oktmo": "string",
        },
        low_memory=False,
    )

    missing_columns = REQUIRED - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"В {path.name} отсутствуют столбцы: "
            f"{missing_columns}"
        )

    return df


def select_and_check(df, code, label):
    note(f"\n--- {label} ---")
    note(f"Все строки исходного CSV: {len(df)}")

    one_year = df.loc[
        df["year"] == YEAR
    ]

    one_indicator = one_year.loc[
        one_year["indicator_code"] == code
    ]

    regional = one_indicator.loc[
        one_indicator["object_level"] == "регион"
    ].copy()

    note(f"Строк за {YEAR} год: {len(one_year)}")
    note(f"Строк показателя {code}: {len(one_indicator)}")
    note(f"Региональных записей: {len(regional)}")

    if regional.empty:
        raise ValueError(
            f"Показатель {code} за {YEAR} год не найден"
        )

    for column in (
        "object_name",
        "object_okato",
        "object_oktmo",
    ):
        values = regional[column].astype("string")

        bad = (
            values.isna()
            | values.str.strip().eq("").fillna(False)
        )

        if bad.any():
            raise ValueError(
                f"{label}: пустые значения в {column}"
            )

    for column in (
        "object_okato",
        "object_oktmo",
    ):
        bad = ~regional[column].str.fullmatch(r"\d{8}")

        if bad.fillna(True).any():
            raise ValueError(
                f"{label}: неправильный формат {column}"
            )

    if regional["object_name"].duplicated().any():
        raise ValueError(
            f"{label}: повторяются названия территорий"
        )

    if (
        regional["indicator_name"].nunique() != 1
        or regional["indicator_unit"].nunique() != 1
    ):
        raise ValueError(
            f"{label}: различаются названия или единицы измерения"
        )

    duplicated_okato = regional.duplicated(
        subset=["object_okato"],
        keep=False,
    )

    if duplicated_okato.any():
        raise ValueError(
            f"{label}: повторяющиеся ОКАТО:\n"
            + regional.loc[
                duplicated_okato,
                ["object_name", "object_okato"]
            ].to_string(index=False)
        )

    values = pd.to_numeric(
        regional["indicator_value"],
        errors="coerce",
    )

    bad_values = (
        values.isna()
        | (values < 0)
        | ~np.isfinite(values)
    )

    if bad_values.any():
        raise ValueError(
            f"{label}: некорректные значения:\n"
            + regional.loc[
                bad_values,
                ["object_name", "indicator_value"]
            ].to_string(index=False)
        )

    regional["indicator_value"] = values.astype(float)

    note(
        f"Уникальных ОКАТО: "
        f"{regional['object_okato'].nunique()}"
    )

    note(
        f"Уникальных ОКТМО: "
        f"{regional['object_oktmo'].nunique()}"
    )

    note(f"Некорректных значений: {bad_values.sum()}")

    note(
        f"Минимум: {values.min():.4f}; "
        f"максимум: {values.max():.4f}"
    )

    note(
        f"Единицы измерения: "
        f"{regional['indicator_unit'].unique().tolist()}"
    )

    duplicated_oktmo = regional.duplicated(
        subset=["object_oktmo"],
        keep=False,
    )

    if duplicated_oktmo.any():
        note("Обнаружены повторяющиеся ОКТМО:")

        note(
            regional.loc[
                duplicated_oktmo,
                ["object_name", "object_oktmo", "object_okato"]
            ].to_string(index=False)
        )

    return regional.sort_values(
        "object_name"
    ).reset_index(drop=True)


healthcare = read_source(HEALTHCARE_FILE)
population = read_source(POPULATION_FILE)

x = select_and_check(
    healthcare,
    X_CODE,
    "X: обеспеченность врачами",
)

y = select_and_check(
    population,
    Y_CODE,
    "Y: продолжительность жизни",
)

x.to_csv(
    OUT_DIR / "healthcare_filter.csv",
    sep=";",
    encoding="utf-8-sig",
    index=False,
)

y.to_csv(
    OUT_DIR / "population_filter.csv",
    sep=";",
    encoding="utf-8-sig",
    index=False,
)

joined = x.merge(
    y,
    on="object_okato",
    how="outer",
    validate="one_to_one",
    indicator=True,
    suffixes=("_x", "_y"),
)

note("\n--- ПРОВЕРКА ОБЪЕДИНЕНИЯ ---")
note(joined["_merge"].value_counts().to_string())

if not joined["_merge"].eq("both").all():
    bad = joined.loc[
        joined["_merge"] != "both",
        ["object_okato", "_merge"],
    ]

    raise ValueError(
        "Не все регионы совпали:\n"
        + bad.to_string(index=False)
    )

for col in (
    "object_name",
    "object_oktmo",
    "object_level",
    "year",
):
    mismatched = joined[f"{col}_x"].ne(
        joined[f"{col}_y"]
    )

    if mismatched.fillna(True).any():
        raise ValueError(
            f"Несовпадения в столбце {col}:\n"
            + joined.loc[
                mismatched.fillna(True),
                [
                    "object_okato",
                    f"{col}_x",
                    f"{col}_y",
                ],
            ].to_string(index=False)
        )

all_regions = pd.DataFrame({
    "year": joined["year_x"].astype(int),
    "object_name": joined["object_name_x"],
    "object_okato": joined["object_okato"],
    "object_oktmo": joined["object_oktmo_x"],
    "doctors_per_10000": joined["indicator_value_x"],
    "life_expectancy_years": joined["indicator_value_y"],
})

all_regions = all_regions.sort_values(
    "object_name"
).reset_index(drop=True)

note(
    f"Объединённых территориальных записей: "
    f"{len(all_regions)}"
)

all_regions.to_csv(
    OUT_DIR / "regions_all.csv",
    sep=";",
    encoding="utf-8-sig",
    index=False,
)

excluded = all_regions.loc[
    all_regions["object_name"].isin(
        EXCLUDED_AGGREGATES
    )
].copy()

if set(excluded["object_name"]) != EXCLUDED_AGGREGATES:
    raise ValueError(
        "Не найдены ожидаемые агрегированные территории"
    )

needed_separate = {
    "Архангельская область без автономного округа",
    "Тюменская область без автономных округов",
    "Ненецкий автономный округ",
    "Ханты-Мансийский автономный округ – Югра",
    "Ямало-Ненецкий автономный округ",
}

if not needed_separate.issubset(
    set(all_regions["object_name"])
):
    raise ValueError(
        "Не найдены все необходимые отдельные территории"
    )

regions_85 = all_regions.loc[
    ~all_regions["object_name"].isin(
        EXCLUDED_AGGREGATES
    )
].copy().reset_index(drop=True)

if len(regions_85) != 85:
    raise ValueError(
        f"Ожидалось 85 наблюдений, "
        f"получено {len(regions_85)}"
    )

if not regions_85["object_okato"].is_unique:
    raise ValueError(
        "В итоговой таблице повторяются ОКАТО"
    )

value_columns = [
    "doctors_per_10000",
    "life_expectancy_years",
]

if regions_85[value_columns].isna().any().any():
    raise ValueError(
        "В итоговой таблице обнаружены пропуски"
    )

if not np.isfinite(
    regions_85[value_columns].to_numpy(dtype=float)
).all():
    raise ValueError(
        "В итоговой таблице есть некорректные значения"
    )

excluded.to_csv(
    OUT_DIR / "excluded_aggregates.csv",
    sep=";",
    encoding="utf-8-sig",
    index=False,
)

regions_85.to_csv(
    OUT_DIR / "regions_statistics.csv",
    sep=";",
    encoding="utf-8-sig",
    index=False,
)

note("\n--- ИТОГ ---")
note(f"Было региональных записей: {len(all_regions)}")
note(f"Исключено агрегатов: {len(excluded)}")

note(
    "Исключены: "
    + "; ".join(sorted(EXCLUDED_AGGREGATES))
)

note(
    f"Осталось территориальных наблюдений: "
    f"{len(regions_85)}"
)

note(
    f"Уникальных ОКАТО: "
    f"{regions_85['object_okato'].nunique()}"
)

note(
    f"Уникальных ОКТМО: "
    f"{regions_85['object_oktmo'].nunique()}"
)

note("Географические границы пока не обрабатывались.")

(OUT_DIR / "auditlog.txt").write_text(
    "\n".join(notes) + "\n",
    encoding="utf-8",
)

print(f"\nГотово! Результаты сохранены в {OUT_DIR}")
