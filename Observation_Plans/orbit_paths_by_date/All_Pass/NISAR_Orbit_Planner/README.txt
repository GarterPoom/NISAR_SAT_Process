NISAR Orbit Planner (offline)
=============================
Open index.html in Chrome / Edge / Firefox (double-click). No internet or server needed.

Folder contents
  index.html        the planner
  lib/              Leaflet 1.9.4 (map library, bundled for offline use)
  data/data.js      all footprints parsed from the KMZ files in ..\ (1729 scenes, 152 dates)
  data/basemap.js   offline basemap: Natural Earth 1:10m countries, Thai provinces, rivers, lakes, roads, cities
  tools/            Python scripts that rebuild data.js from the KMZs

Updating after new KMZ files arrive
  pip install shapely
  1) copy the new .kmz files into the All_Pass folder
  2) python tools/parse.py   (reads ../*.kmz -> nisar.json; edit the 'src' path at the top if needed)
  3) python tools/build.py   (needs the Natural Earth geojson files; or just keep the existing basemap.js
                              and comment out the basemap section)
  4) copy data.js into data/

Notes
  * Overpass times are nominal (NISAR is a 6 AM / 6 PM dawn-dusk sun-synchronous orbit:
    ascending ~18:00 local, descending ~06:00 local). Check exact UTC times in the official plan.
  * Radar-mode decoding (polarisation, bandwidth -> resolution) is a best-effort reading of the
    mnemonic; the raw mnemonic is always shown.
  * Scenes with an empty radar mode are shown by default (untick "Show scenes w/o radar mode" to hide them).
  * "Clipped to Thailand" is off by default; tick it to overlay the Thailand-only part of each frame.
  * The "Online" basemaps in the layer menu (top-left) only load when connected to the internet.
