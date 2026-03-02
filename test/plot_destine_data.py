"""Quick script to inspect and plot DestinE Zarr data."""

import xarray as xr
import matplotlib.pyplot as plt

zarr_path = "/Users/ikuznets/work/projects/climsight/code/climsight/tmp/sandbox/38c864498d174b8a90ebb24ac67cf70e/destine_data/destine_167_sfc_20200101_20211231.zarr"

ds = xr.open_dataset(zarr_path, engine="zarr")

print("=== Dataset Info ===")
print(ds)
print("\n=== Variables ===")
for var in ds.data_vars:
    print(f"  {var}: shape={ds[var].shape}, dtype={ds[var].dtype}")

print("\n=== Coordinates ===")
for coord in ds.coords:
    print(f"  {coord}: {ds[coord].values[:5]}{'...' if len(ds[coord]) > 5 else ''}")

# Find the temperature variable (likely '2t' or 't2m')
temp_var = None
for name in ["2t", "t2m", "167"]:
    if name in ds.data_vars:
        temp_var = name
        break

if temp_var is None:
    temp_var = list(ds.data_vars)[0]
    print(f"\nNo known temp var found, using first variable: {temp_var}")

data = ds[temp_var]
print(f"\n=== {temp_var} stats ===")
print(f"  min: {float(data.min()):.2f}")
print(f"  max: {float(data.max()):.2f}")
print(f"  mean: {float(data.mean()):.2f}")

# Convert from Kelvin if values look like Kelvin
values = data.values.flatten()
if float(data.mean()) > 200:
    print("  (looks like Kelvin, converting to Celsius for plot)")
    plot_data = data - 273.15
    ylabel = "Temperature (°C)"
else:
    plot_data = data
    ylabel = f"{temp_var} ({data.attrs.get('units', '?')})"

fig, ax = plt.subplots(figsize=(14, 5))
plot_data.plot(ax=ax)
ax.set_ylabel(ylabel)
ax.set_title(f"DestinE 2m Temperature — Point Time Series (2020-2021)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("destine_temperature_plot.png", dpi=150)
print(f"\nPlot saved to destine_temperature_plot.png")
plt.show()
