# Weather report (ERA5-Land)

Variables used: ['tp', 'swvl1']. Dropped because not in every chunk: ['swvl2', 't2m'].
Days per year (one station): {2001: 365, 2002: 365, 2003: 365, 2004: 366, 2005: 365, 2006: 365, 2007: 365, 2008: 366, 2009: 365, 2010: 365, 2011: 365, 2012: 366, 2013: 365, 2014: 365, 2015: 365, 2016: 366, 2017: 365, 2018: 365, 2019: 365, 2020: 366, 2021: 365, 2022: 365, 2023: 365, 2024: 366, 2025: 365, 2026: 273}
No missing years.
Rows 1,006,228, stations 107, dates 2001-01-01 to 2026-09-30.
Nearest-land-cell distance (km): median 4.1, max 7.7.
Stations sharing a grid cell with another station: 7 cells hold 14 stations.

Median ANNUAL rainfall per station (mm). Western Ghats / coastal should be roughly 2000-4500, interior Deccan roughly 500-900. If everything is ~365x too small/large, the 24 h accumulation assumption is wrong - tell me.

```
count     107.0
mean     1923.0
std      1040.0
min       665.0
25%       892.0
50%      2035.0
75%      2913.0
max      3843.0

station_id
Oollenur                 665.0
Kokangaon                702.0
Shirdhon                 713.0
T. Ramapuram             719.0
Narayanapur Reservoir    720.0

station_id
AVERSHE         3584.0
SANOOR          3589.0
CHAKKALAKUTH    3657.0
PERUVAMPADAM    3657.0
YENNEHOLE       3843.0
```