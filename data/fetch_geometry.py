"""Download verified source geometry."""
from pathlib import Path
from urllib.request import urlopen
import hashlib
URL = "https://raw.githubusercontent.com/hairymax/offline_russia_plotly/278388ae0374fc332273e9fd846a04a5568b02c3/data/russia_regions.geojson"
SHA256 = "c9c6134a90043dc6fcfc9f177abdf5c408123e6d3d07ea17702fe943fc72553d"
TARGET = Path(__file__).resolve().parent / "russia_regions.geojson"
def main():
    if TARGET.exists():
        if hashlib.sha256(TARGET.read_bytes()).hexdigest() != SHA256:
            raise RuntimeError("Existing GeoJSON SHA256 differs; will not overwrite.")
        print("Existing source verified:", TARGET.name)
        return
    with urlopen(URL, timeout=120) as stream:
        data = stream.read()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise RuntimeError("Downloaded source SHA256 differs.")
    TARGET.write_bytes(data)
    print("Downloaded and verified:", TARGET.name)
if __name__ == "__main__":
    main()
