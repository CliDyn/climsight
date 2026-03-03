#!/usr/bin/env python3
"""
Simple script to fetch ERA5 data using the era5_retrieval_tool.

Usage:
    python era5_fetch.py --lat 52.5 --lon 13.4 --start 2020-01-01 --end 2020-12-31 --var t2
    python era5_fetch.py --lat 62.4 --lon -114.4 --start 2015-01-01 --end 2024-12-31 --var tp --output ./my_data

Available variables:
    t2   - 2m temperature
    sst  - Sea surface temperature
    u10  - 10m U wind component
    v10  - 10m V wind component
    mslp - Mean sea level pressure
    sp   - Surface pressure
    tp   - Total precipitation
    tcc  - Total cloud cover
    d2   - 2m dewpoint temperature
    skt  - Skin temperature
"""

import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.era5_retrieval_tool import retrieve_era5_data, VARIABLE_MAPPING


def list_variables():
    """Print available ERA5 variables."""
    print("Available ERA5 variables:")
    print("-" * 50)
    seen = set()
    for name, code in sorted(VARIABLE_MAPPING.items()):
        if code not in seen:
            print(f"  {code:6} - {name}")
            seen.add(code)


def main():
    # Quick check for --list-vars before full parsing
    if "--list-vars" in sys.argv:
        list_variables()
        return

    parser = argparse.ArgumentParser(
        description="Fetch ERA5 data from Earthmover/Arraylake",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # All arguments
    parser.add_argument("--lat", type=float, required=True, help="Latitude (-90 to 90)")
    parser.add_argument("--lon", type=float, required=True, help="Longitude (-180 to 180 or 0 to 360)")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--var", type=str, required=True, help="Variable (e.g., t2, tp, sst, u10, v10, mslp)")
    parser.add_argument("--output", type=str, default="./era5_output", help="Output directory (default: ./era5_output)")
    parser.add_argument("--show", action="store_true", help="Display data summary after download")
    parser.add_argument("--list-vars", action="store_true", help="List available variables and exit")

    args = parser.parse_args()

    # Check for API key
    if not os.environ.get("ARRAYLAKE_API_KEY"):
        print("ERROR: ARRAYLAKE_API_KEY environment variable not set.")
        print("Please set it with: export ARRAYLAKE_API_KEY=your_key")
        sys.exit(1)

    # Normalize longitude to 0-360 range (ERA5 uses 0-360)
    lon = args.lon
    if lon < 0:
        lon = lon + 360

    # Set up output in sandbox structure (tmp/sandbox/<session>/era5_data/)
    # Generate a session ID for standalone usage
    import uuid
    session_id = f"cli_{uuid.uuid4().hex[:8]}"

    # If output is default, use sandbox structure
    if args.output == "./era5_output":
        sandbox_dir = os.path.join("tmp", "sandbox", session_id)
    else:
        sandbox_dir = args.output

    os.makedirs(sandbox_dir, exist_ok=True)

    print(f"Fetching ERA5 data:")
    print(f"  Location: {args.lat}°N, {args.lon}°E (normalized: {lon}°)")
    print(f"  Period: {args.start} to {args.end}")
    print(f"  Variable: {args.var}")
    print(f"  Output: {sandbox_dir}")
    print()

    # Call the retrieval function
    result = retrieve_era5_data(
        variable_id=args.var,
        start_date=args.start,
        end_date=args.end,
        min_latitude=args.lat,
        max_latitude=args.lat,
        min_longitude=lon,
        max_longitude=lon,
        work_dir=sandbox_dir
    )

    # Handle result
    if result.get("success"):
        print(f"✅ Success!")
        print(f"   Data saved to: {result.get('output_path_zarr')}")
        print(f"   Variable: {result.get('variable')}")

        # Show data summary if requested
        if args.show:
            import xarray as xr
            print()
            print("Data summary:")
            print("-" * 50)
            ds = xr.open_zarr(result.get('output_path_zarr'))
            print(ds)

            # Show some statistics
            var_name = result.get('variable')
            if var_name in ds:
                data = ds[var_name]
                print()
                print(f"Statistics for {var_name}:")
                print(f"  Min: {float(data.min()):.2f}")
                print(f"  Max: {float(data.max()):.2f}")
                print(f"  Mean: {float(data.mean()):.2f}")
            ds.close()
    else:
        print(f"❌ Failed!")
        print(f"   Error: {result.get('error')}")
        print(f"   Message: {result.get('message')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
