The table lists the services used to prepare the datasets in this directory.

| Information source | Derived datasets | Usage |
| ------------------ | ---------------- | ----- |
| [EMF-Karte](https://www.bundesnetzagentur.de/DE/Vportal/TK/Funktechnik/EMF/start.html) | `data/antennas.csv`, `data/sites.csv` | Determine locations of mobile stations and their operators, gather orientation of antennas |
| [Google Maps](https://www.google.de/maps) | `data/sites.csv` | Correct inaccurate locations using satellite and StreetView images, derive construction type (e.g., mounted on roof, freestanding mast, etc.) |
| [OpenStreetMap](https://www.openstreetmap.org/), [Overpass API](https://overpass-turbo.eu/) | `data/pois.json`, `data/buildings.json` | Obtain geoshapes of selected buildings and Points of Interest |
| [OpenData Portal Leipzig](https://opendata.leipzig.de/dataset/personenhaushalte-jahreszahlen-kleinraumig), [OpenData Portal Leipzig](https://opendata.leipzig.de/dataset/geodaten-ortsteile-leipzig) | `data/population.csv`, `data/regions.json` | Retrieve geoshapes of city districts and their demographic figures |