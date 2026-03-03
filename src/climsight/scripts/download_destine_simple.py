"""Simple DestinE data download — single request, no splitting."""

import time

import earthkit.data
import xarray as xr

# --- Settings (change these) ---
param_id = "167"        # 167 = 2m temperature
levtype = "sfc"         # sfc = surface
lat = 58.0
lon = 13.0
start_date = "20200101"
end_date = "20211231"

POLYTOPE_ADDRESS = "polytope.lumi.apps.dte.destination-earth.eu"

# --- Request ---
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
    'date': f'{start_date}/to/{end_date}',
    'time': ['0000', '0100', '0200', '0300', '0400', '0500',
             '0600', '0700', '0800', '0900', '1000', '1100',
             '1200', '1300', '1400', '1500', '1600', '1700',
             '1800', '1900', '2000', '2100', '2200', '2300'],
    'feature': {
        "type": "timeseries",
        "points": [[lat, lon]],
        "time_axis": "date",
    }
}

# --- Download ---
print(f"Downloading param={param_id}, levtype={levtype}, lat={lat}, lon={lon}")
print(f"Date range: {start_date} to {end_date}")

t0 = time.monotonic()
data = earthkit.data.from_source(
    "polytope", "destination-earth",
    request,
    address=POLYTOPE_ADDRESS,
    stream=False,
)
t_download = time.monotonic() - t0
print(f"\nDownload: {t_download:.1f}s")

t1 = time.monotonic()
ds = data.to_xarray()
t_convert = time.monotonic() - t1
print(f"Convert to xarray: {t_convert:.1f}s")

print(f"\n=== Dataset ===")
print(ds)
print(f"\nDims: {list(ds.dims)}")
print(f"Coords: {list(ds.coords)}")
print(f"Vars: {list(ds.data_vars)}")

# --- Save as Zarr ---
output = f"destine_{param_id}_{levtype}_{start_date}_{end_date}.zarr"
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
