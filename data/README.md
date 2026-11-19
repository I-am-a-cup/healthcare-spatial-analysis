# Источники и состав данных

**Статистика:** Росстат, «Регионы России. Социально-экономические показатели»; обработка «Если быть точным», 2023: [regions_collection](https://github.com/tochno-st/regions_collection), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Файлы healthcare.csv и population.csv — исходные разделы. Команда выполнила отбор показателей 05040200 и 01160100 за 2021 год, очистку и объединение.

**Географическая основа:** [hairymax/offline_russia_plotly, фиксированная версия](https://github.com/hairymax/offline_russia_plotly/blob/278388ae0374fc332273e9fd846a04a5568b02c3/data/russia_regions.geojson); [описание происхождения](https://habr.com/ru/articles/752406/); первичный источник [timurkanaz/Russia_geojson_OSM](https://github.com/timurkanaz/Russia_geojson_OSM). Геометрия: © OpenStreetMap contributors, [ODbL 1.0](https://www.openstreetmap.org/copyright). Промежуточный репозиторий не указывает отдельную лицензию; сырая копия геометрии не дублируется в Git. Скрипт fetch_geometry.py получает точный файл и проверяет SHA256.

Подготовленная база: merged_geodata/regions_2021_attributes.csv и regions_2021_geospatial.geojson. Географическая основа и производные пространственные данные OSM доступны на условиях ODbL; статистические показатели сохраняют атрибуцию CC BY 4.0. Изменения: привязка регионов, присоединение показателей, уточнение четырёх меток федеральных округов. Скрипты позволяют получить производную базу повторно.

Геометрия Polygon/MultiPolygon, EPSG:4326. Историческая точность границ на 2021 год и охват областей без АО не подтверждены независимо. Состав территорий соответствует использованному статистическому набору за 2021 год; карты не являются утверждением о международно-правовом статусе территорий.

- prepared_statistics/ — отбор показателей, таблицы и аудит.
- prepared_geography/ — таблица соответствия и геометрическая диагностика.
- merged_geodata/ — канонические данные.
- ../results/ — общая W и расчётные результаты.

ОКАТО — уникальный ключ, ОКТМО не уникален.

Полная подготовка из корня проекта:

    python data/fetch_geometry.py
    python data/preparation.py
    python data/prepare_geography.py
    python data/merge_geodata.py

- SHA256 healthcare.csv: 52453a0eda0318de7a01d8be2d5b9a715f253c6aff42fe5c83b56bb199bd8bed
- SHA256 population.csv: 351ae2dc22117118721d751cc21cf97c2d08b300c0b0328073197b4557d15977
- SHA256 regions_2021_geospatial.geojson: a6d77860a688fa8155343cd9429bb1f237d351814e0847ad39fff910fe1e6a06
