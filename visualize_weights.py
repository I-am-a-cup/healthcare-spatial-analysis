"""Draw the shared spatial-weights graph and matrix without refitting models."""

from __future__ import annotations

import csv
import json
from math import cos, pi, sin, sqrt
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import shape
from shapely.ops import transform


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUTPUT = ROOT / "maps" / "05_shared_spatial_weights.png"
WIDTH, HEIGHT = 2400, 1150
SCALE = 2  # Render at twice the layout resolution for presentation slides.
LON0, LAT0 = 100 * pi / 180, 62 * pi / 180
EARTH_RADIUS = 6_371_000


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    windows_font = Path("C:/Windows/Fonts") / name
    if windows_font.is_file():
        return ImageFont.truetype(str(windows_font), size * SCALE)
    from matplotlib import font_manager
    fallback = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal"))
    return ImageFont.truetype(fallback, size * SCALE)


def project(x, y, z=None):
    lon = np.deg2rad(np.asarray(x, dtype=float))
    lat = np.deg2rad(np.asarray(y, dtype=float))
    delta = lon - LON0
    denominator = 1 + sin(LAT0) * np.sin(lat) + cos(LAT0) * np.cos(lat) * np.cos(delta)
    k = np.sqrt(2 / np.maximum(denominator, 1e-12))
    px = EARTH_RADIUS * k * np.cos(lat) * np.sin(delta)
    py = EARTH_RADIUS * k * (cos(LAT0) * np.sin(lat) - sin(LAT0) * np.cos(lat) * np.cos(delta))
    return px, py


def polygons(geometry):
    return [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)


def main() -> None:
    with (RESULTS / "weights_order.csv").open(encoding="utf-8-sig", newline="") as stream:
        order = list(csv.DictReader(stream, delimiter=";"))
    with (RESULTS / "weights_edges.csv").open(encoding="utf-8-sig", newline="") as stream:
        edges = list(csv.DictReader(stream, delimiter=";"))
    metadata = json.loads((RESULTS / "weights_metadata.json").read_text(encoding="utf-8"))
    ids = [row["object_okato"] for row in order]
    names = {row["object_okato"]: row["object_name"] for row in order}
    indices = {region_id: i for i, region_id in enumerate(ids)}
    assert len(ids) == 85 and len(set(ids)) == 85 and len(edges) == 384

    data = json.loads((ROOT / "data" / "merged_geodata" / "regions_2021_geospatial.geojson").read_text(encoding="utf-8"))
    projected = {}
    for feature in data["features"]:
        region_id = str(feature["properties"]["object_okato"]).zfill(8)
        projected[region_id] = transform(project, shape(feature["geometry"]))
    assert set(projected) == set(ids)
    bounds = np.array([g.bounds for g in projected.values()])
    xmin, ymin = bounds[:, :2].min(axis=0)
    xmax, ymax = bounds[:, 2:].max(axis=0)

    map_box = (60, 155, 1560, 970)
    available_w = map_box[2] - map_box[0]
    available_h = map_box[3] - map_box[1]
    factor = min(available_w / (xmax - xmin), available_h / (ymax - ymin))
    offset_x = map_box[0] + (available_w - factor * (xmax - xmin)) / 2
    offset_y = map_box[1] + (available_h - factor * (ymax - ymin)) / 2

    def screen(x: float, y: float) -> tuple[int, int]:
        return (round((offset_x + (x - xmin) * factor) * SCALE),
                round((offset_y + (ymax - y) * factor) * SCALE))

    image = Image.new("RGB", (WIDTH * SCALE, HEIGHT * SCALE), "#ffffff")
    draw = ImageDraw.Draw(image)
    draw.text((60 * SCALE, 35 * SCALE), "Общая матрица пространственных весов W", font=font(33, True), fill="#17243b")
    draw.text((60 * SCALE, 94 * SCALE),
              "85 регионов · соседство Queen · построчная нормировка · 384 ненулевых веса",
              font=font(19), fill="#536178")
    draw.text((80 * SCALE, 157 * SCALE), "Связи регионов на карте", font=font(23, True), fill="#17243b")

    # Borders are simplified only for this illustration; the model retains original geometry.
    for region_id in ids:
        geometry = projected[region_id]
        for polygon in polygons(geometry.simplify(25_000, preserve_topology=True)):
            outline = [screen(x, y) for x, y in polygon.exterior.coords]
            if len(outline) >= 3:
                draw.polygon(outline, fill="#f0f3f6")
                draw.line(outline, fill="#aeb9c7", width=1 * SCALE, joint="curve")

    points = {key: screen(*projected[key].representative_point().coords[0]) for key in ids}
    bridges = {tuple(sorted((b["island_okato"], b["neighbor_okato"])))
               for b in metadata["bridge_edges"]}
    seen = set()
    for edge in sorted(edges, key=lambda e: (e["source_okato"], e["target_okato"])):
        pair = tuple(sorted((edge["source_okato"], edge["target_okato"])))
        source, target = pair
        if pair in seen or pair in bridges:
            continue
        seen.add(pair)
        draw.line((points[source], points[target]), fill="#5c98ca", width=2 * SCALE)
    for source, target in sorted(bridges):
        draw.line((points[source], points[target]), fill="#e1723f", width=5 * SCALE)
        for key in (source, target):
            x, y = points[key]
            r = 7 * SCALE
            draw.ellipse((x-r, y-r, x+r, y+r), fill="#d34e2c", outline="white", width=2 * SCALE)

    draw.line((95*SCALE, 1005*SCALE, 150*SCALE, 1005*SCALE), fill="#5c98ca", width=3*SCALE)
    draw.text((165*SCALE, 990*SCALE), "соседство по границе или вершине", font=font(17), fill="#31415b")
    draw.line((590*SCALE, 1005*SCALE, 645*SCALE, 1005*SCALE), fill="#e1723f", width=5*SCALE)
    draw.text((660*SCALE, 990*SCALE), "добавленная связь для региона без соседей", font=font(17), fill="#31415b")
    draw.text((95*SCALE, 1045*SCALE), "Линии между точками условные: они показывают элементы W, а не маршруты.",
              font=font(16), fill="#637087")

    draw.text((1660*SCALE, 157*SCALE), "Матрица 85 × 85", font=font(23, True), fill="#17243b")
    draw.text((1660*SCALE, 197*SCALE), "Строка — регион; столбец — его сосед", font=font(17), fill="#536178")
    matrix = np.zeros((85, 85), dtype=float)
    for edge in edges:
        matrix[indices[edge["source_okato"]], indices[edge["target_okato"]]] = float(edge["weight"])
    assert np.allclose(matrix.sum(axis=1), 1) and np.allclose(np.diag(matrix), 0)
    cell = 7
    x0, y0 = 1690, 270
    for row in range(85):
        for col in range(85):
            weight = matrix[row, col]
            if weight <= 0:
                color = "#f4f6f8"
            else:
                intensity = min(weight / 0.5, 1)
                color = (round(208 - 170*intensity), round(225 - 135*intensity), round(243 - 72*intensity))
            xx = (x0 + col * cell) * SCALE
            yy = (y0 + row * cell) * SCALE
            draw.rectangle((xx, yy, xx + cell*SCALE - 1, yy + cell*SCALE - 1), fill=color)
    for tick in (0, 20, 40, 60, 80):
        draw.text(((x0 + tick*cell)*SCALE, (y0-28)*SCALE), str(tick), font=font(13), fill="#65738a")
        draw.text(((x0-31)*SCALE, (y0 + tick*cell)*SCALE), str(tick), font=font(13), fill="#65738a")
    draw.text((1690*SCALE, 892*SCALE), "Белое поле — вес 0; синий цвет — ненулевой вес.",
              font=font(17), fill="#31415b")
    draw.text((1690*SCALE, 931*SCALE), "В каждой строке сумма весов равна 1.",
              font=font(17), fill="#31415b")
    draw.text((1690*SCALE, 990*SCALE), "Добавленные связи:", font=font(17, True), fill="#31415b")
    for index, bridge in enumerate(metadata["bridge_edges"]):
        label = f"{names[bridge['island_okato']]} → {names[bridge['neighbor_okato']]}"
        draw.text((1690*SCALE, (1021 + index*30)*SCALE), label, font=font(15), fill="#9b4c2f")

    draw.text((60*SCALE, 1115*SCALE), "© OpenStreetMap contributors · ODbL 1.0 · openstreetmap.org/copyright", font=font(14), fill="#637087")
    OUTPUT.parent.mkdir(exist_ok=True)
    image.save(OUTPUT, optimize=True)
    print(f"Saved {OUTPUT} ({image.width}x{image.height})")


if __name__ == "__main__":
    main()
