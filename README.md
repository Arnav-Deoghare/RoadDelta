# RoadDelta

Road network change detection: given aerial imagery of the same area at two dates and the
older road map, predict which roads were added or removed.

Data: MUNO21 (NAIP imagery + OpenStreetMap history). Pre-change year is 2013, post-change
is 2018/2019.

## Layout
    setup.bat             one-time setup: venv, dependencies, MUNO21 download
    requirements.txt
    roaddelta/graph.py    .graph loader, rasterizer, overlay
    scripts/inspect_data.py   sanity check on the downloaded data
    data/raw/             MUNO21 lands here as data/raw/mapupdate/

## Usage
    setup.bat
    .venv\Scripts\activate
    python scripts\inspect_data.py --root data\raw\mapupdate

## Next
- Verify the .graph parser against real files
- Scenario loader (pair 2013 and 2018/2019 tiles, crop annotation windows)
