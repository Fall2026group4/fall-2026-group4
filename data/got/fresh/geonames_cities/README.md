# XF fresh cities (GeoNames): identifiers only

`statements.csv` (1,426 statements, 713 cities, 78 countries; 900 TEST / 526 DEV) is **not** committed. It is rebuilt
locally, deterministically, by the frozen XF build code:

    python scripts/cross_model/23b_xf_rebuild_statements.py          # needs ../got-repo/datasets/geonames.csv

The script checks the upstream file's SHA-256 (`build_info.json` `source_sha256`), checks the rebuilt file against the
SHA-256 frozen in `results/cross_model/xf_fresh_cities_test/preregistration.json` (`00186e18…`) and against `ids.csv`
row for row, and never overwrites an existing file. Verified on 2026-11-26: the rebuild is byte-identical to the file
used for the XF TEST pass.

## Files
* `ids.csv`: `geoname_id`, `label` (1 = true statement), `country` (the country named in the statement), `split`.
  Our own split and label assignments; no GeoNames text, names or populations.
* `build_info.json`: filter counts, salt, upstream SHA-256 and integrity checks of the 2026-11-23 build.

## Source, licence and attribution
* **Data source:** [GeoNames](https://www.geonames.org/), via the `datasets/geonames.csv` file of the Geometry of
  Truth repository (https://github.com/saprmarks/geometry-of-truth, commit `5d1c630`). Its columns match the
  OpenDataSoft "geonames-all-cities-with-a-population-1000" export of GeoNames.
* **GeoNames licence:** Creative Commons Attribution 4.0 (https://creativecommons.org/licenses/by/4.0/). GeoNames asks
  users to "give credit to GeoNames when using data or web services with a link or another reference to GeoNames"
  (https://www.geonames.org/export/, read 2026-11-26). This README and the paper's data statement give that credit.
* **Geometry of Truth repository:** no licence is stated (GitHub reports none; no LICENSE file), so its files are not
  redistributed here; the same rule applies to the raw fresh-GoT downloads (`data/got/fresh/raw_*/`, git-ignored).
* **Not verified:** the OpenDataSoft dataset page did not render for an automated check, so its own licence statement
  was not read.
