"""Simple DestinE data download example — parallel yearly requests."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import earthkit.data
import xarray as xr

# --- Settings (change these) ---
param_id = "167"        # 167 = 2m temperature
levtype = "sfc"         # sfc = surface
lat = 58.0
lon = 13.0
start_year = 2020
end_year = 2021
max_workers = 1         # parallel downloads

POLYTOPE_ADDRESS = "polytope.lumi.apps.dte.destination-earth.eu"

ALL_TIMES = ['0000', '0100', '0200', '0300', '0400', '0500',
             '0600', '0700', '0800', '0900', '1000', '1100',
             '1200', '1300', '1400', '1500', '1600', '1700',
             '1800', '1900', '2000', '2100', '2200', '2300']


def download_year(year):
    """Download one year of data, return xarray Dataset."""
    request = {
        'class': 'd1',
        'dataset': 'climate-dt',
        'type': 'fc',
        'expver': '0001',
        'generation': '1',
        'realization': '1',
        'activity': 'ScenarioMIP',
        'experiment': 'SSP3-7.0',
        'model': 'IFS-NEMO',
        'param': param_id,
        'levtype': levtype,
        'resolution': 'high',
        'stream': 'clte',
        'date': f'{year}0101/to/{year}1231',
        'time': ALL_TIMES,
        'feature': {
            "type": "timeseries",
            "points": [[lat, lon]],
            "time_axis": "date",
        }
    }
    t0 = time.monotonic()
    data = earthkit.data.from_source(
        "polytope", "destination-earth",
        request,
        address=POLYTOPE_ADDRESS,
        stream=False,
    )
    ds = data.to_xarray()
    elapsed = time.monotonic() - t0

    # Print what we got
    dims = list(ds.dims)
    coords = list(ds.coords)
    data_vars = list(ds.data_vars)
    print(f"  {year}: {elapsed:.1f}s | dims={dims}, coords={coords}, vars={data_vars}")
    return ds


# --- Download in parallel ---
years = list(range(start_year, end_year + 1))
print(f"Downloading param={param_id}, levtype={levtype}, lat={lat}, lon={lon}")
print(f"Years: {start_year}-{end_year} ({len(years)} years, {max_workers} workers)\n")

t0 = time.monotonic()
datasets = {}

with ThreadPoolExecutor(max_workers=max_workers) as pool:
    futures = {pool.submit(download_year, y): y for y in years}
    for future in as_completed(futures):
        year = futures[future]
        try:
            datasets[year] = future.result()
        except Exception as e:
            print(f"  {year}: FAILED — {e}")

t_download = time.monotonic() - t0
print(f"\nAll downloads: {t_download:.1f}s")

if not datasets:
    print("No data downloaded, exiting.")
    raise SystemExit(1)

# --- Figure out the time dimension name ---
first_ds = next(iter(datasets.values()))
time_dim = None
for dim_name in ["time", "date", "forecast_reference_time", "step"]:
    if dim_name in first_ds.dims:
        time_dim = dim_name
        break
if time_dim is None:
    time_dim = list(first_ds.dims)[0]
    print(f"Warning: no known time dim found, using first dim: {time_dim}")
print(f"Time dimension: '{time_dim}'")

# --- Combine ---
t1 = time.monotonic()
if len(datasets) == 1:
    ds = first_ds
else:
    ds = xr.concat([datasets[y] for y in sorted(datasets)], dim=time_dim)
t_combine = time.monotonic() - t1
print(f"Combine: {t_combine:.1f}s")

print(f"\n=== Dataset ===")
print(ds)

# --- Save as Zarr ---
output = f"destine_{param_id}_{levtype}_{start_year}0101_{end_year}1231.zarr"
for var in ds.data_vars:
    ds[var].encoding.clear()
for coord in ds.coords:
    ds[coord].encoding.clear()

t2 = time.monotonic()
ds.to_zarr(output, mode='w')
t_save = time.monotonic() - t2

t_total = time.monotonic() - t0
print(f"\nSave to Zarr: {t_save:.1f}s")
print(f"Total: {t_total:.1f}s")
print(f"Saved to {output}")
