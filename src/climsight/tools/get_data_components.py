"""Tool for extracting specific climate variables from stored climatology."""

import ast
import calendar
import json
import logging
import os
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

try:
    from ..sandbox_utils import load_climate_data_manifest
except ImportError:
    from sandbox_utils import load_climate_data_manifest

logger = logging.getLogger(__name__)


def _load_df_list_from_manifest(manifest_path: str, climate_data_dir: str) -> List[dict]:
    """Rehydrate df_list from a sandbox manifest when state.df_list is absent."""
    manifest = load_climate_data_manifest(manifest_path)
    if not manifest:
        return []

    entries = []
    for entry in manifest.get("entries", []):
        csv_path = os.path.join(climate_data_dir, entry.get("csv", ""))
        meta_path = os.path.join(climate_data_dir, entry.get("meta", ""))
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        meta = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = {}

        entries.append({
            "dataframe": df,
            "extracted_vars": meta.get("extracted_vars", {}),
            "years_of_averaging": meta.get("years_of_averaging", ""),
            "description": meta.get("description", ""),
            "main": meta.get("main", False),
            "source": meta.get("source", ""),
        })

    return entries


class GetDataComponentsArgs(BaseModel):
    environmental_data: Optional[str] = Field(
        default=None,
        description=(
            "The type of environmental data to retrieve. "
            "Choose from Temperature, Precipitation, u_wind, or v_wind."
        ),
    )
    months: Optional[Union[str, List[str]]] = Field(
        default=None,
        description=(
            "List of months or a stringified list of month names to retrieve data for. "
            "Each month should be one of 'Jan', 'Feb', ..., 'Dec'. "
            "If not specified, data for all months will be retrieved."
        ),
    )


def create_get_data_components_tool(state, config, stream_handler=None):
    """Create a StructuredTool bound to the current agent state."""
    try:
        from langchain.tools import StructuredTool
    except ImportError:
        from langchain_core.tools import StructuredTool

    def get_data_components(**kwargs):
        if stream_handler is not None:
            stream_handler.update_progress("Retrieving specific climatology values...")

        if isinstance(kwargs.get("months"), str):
            try:
                kwargs["months"] = ast.literal_eval(kwargs["months"])
            except (ValueError, SyntaxError):
                kwargs["months"] = None

        args = GetDataComponentsArgs(**kwargs)
        environmental_data = args.environmental_data
        months = args.months

        if not environmental_data:
            return {"error": "No environmental data type specified."}

        climate_source = config.get("climate_data_source", "nextGEMS")

        # Normalize common shorthand to canonical names.
        env_normalized = environmental_data.strip()
        env_aliases = {
            "tp": "Precipitation",
            "precipitation": "Precipitation",
            "mean2t": "Temperature",
            "tas": "Temperature",
            "temp": "Temperature",
            "t2m": "Temperature",
            "wind_u": "u_wind",
            "wind_v": "v_wind",
            "uas": "u_wind",
            "vas": "v_wind",
        }
        env_normalized = env_aliases.get(env_normalized, env_normalized)

        df_list = getattr(state, "df_list", None)
        if not df_list:
            manifest_path = state.input_params.get("climate_data_manifest", "")
            climate_data_dir = state.input_params.get("climate_data_dir", "") or getattr(
                state, "climate_data_dir", ""
            )
            df_list = _load_df_list_from_manifest(manifest_path, climate_data_dir)

        if not df_list:
            return {"error": "No climatology data available."}

        if climate_source in ("nextGEMS", "ICCP"):
            environmental_mapping = {
                "Temperature": "mean2t",
                "Precipitation": "tp",
                "u_wind": "wind_u",
                "v_wind": "wind_v",
            }
        elif climate_source == "AWI_CM":
            environmental_mapping = {
                "Temperature": "Present Day Temperature",
                "Precipitation": "Present Day Precipitation",
                "u_wind": "u_wind",
                "v_wind": "v_wind",
            }
        else:
            environmental_mapping = {
                "Temperature": "mean2t",
                "Precipitation": "tp",
                "u_wind": "wind_u",
                "v_wind": "wind_v",
            }

        if env_normalized not in environmental_mapping:
            return {"error": f"Invalid environmental data type: {environmental_data}"}

        var_name = environmental_mapping[env_normalized]

        if not months:
            months = [calendar.month_abbr[m] for m in range(1, 13)]

        month_mapping = {calendar.month_abbr[m]: calendar.month_name[m] for m in range(1, 13)}
        selected_months = []
        for month in months:
            if month in month_mapping:
                selected_months.append(month_mapping[month])
            elif month in calendar.month_name:
                selected_months.append(month)

        response = {}
        references = set()  # Collect unique references from data sources
        for entry in df_list:
            df = entry.get("dataframe")
            extracted_vars = entry.get("extracted_vars", {})
            source = entry.get("source", "")

            if df is None:
                continue
            if "Month" not in df.columns:
                continue

            if var_name in df.columns:
                var_meta = extracted_vars.get(var_name, {"units": ""})
                data_values = df[df["Month"].isin(selected_months)][var_name].tolist()
                ext_data = {month: float(np.round(value, 2)) for month, value in zip(selected_months, data_values)}
                ext_exp = (
                    f"Monthly mean values of {env_normalized}, {var_meta.get('units', '')} "
                    f"for years: {entry.get('years_of_averaging', '')}"
                )
                response.update({ext_exp: ext_data})
                if source:
                    references.add(source)
            else:
                matching_cols = [col for col in df.columns if environmental_data.lower() in col.lower()]
                if matching_cols:
                    col_name = matching_cols[0]
                    data_values = df[df["Month"].isin(selected_months)][col_name].tolist()
                    ext_data = {month: float(np.round(value, 2)) for month, value in zip(selected_months, data_values)}
                    ext_exp = (
                        f"Monthly mean values of {env_normalized} "
                        f"for years: {entry.get('years_of_averaging', '')}"
                    )
                    response.update({ext_exp: ext_data})
                    if source:
                        references.add(source)

        if not response:
            return {"error": f"Variable '{environmental_data}' not found in climatology."}

        # Add reference based on climate source
        if references:
            response["references"] = list(references)
        else:
            # Fallback reference based on climate source
            source_references = {
                "nextGEMS": "Moon, J.-Y., et al. (2024). Earth's Future Climate and Its Variability Simulated at 9 Km Global Resolution. https://doi.org/10.5194/egusphere-2024-249",
                "ICCP": "AWI-CM3 ICCP climate projections for past and future climate scenarios.",
                "AWI_CM": "AWI-CM (CMIP6): Climate model data covering past (1985-2004) and future (2070-2100) climate scenarios.",
                "DestinE": "Destination Earth Climate Digital Twin: IFS-FESOM coupled high-resolution climate simulations (SSP3-7.0 scenario). https://destine.ecmwf.int/",
            }
            if climate_source in source_references:
                response["reference"] = source_references[climate_source]

        return response

    return StructuredTool.from_function(
        func=get_data_components,
        name="get_data_components",
        description="Retrieve specific climate variables from the saved climatology.",
        args_schema=GetDataComponentsArgs,
    )
