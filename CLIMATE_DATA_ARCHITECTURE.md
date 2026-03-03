# Climate Data Architecture Documentation

This document explains the multi-source climate data provider system in ClimSight, including where and how data are extracted, stored, and configured.

## Table of Contents
1. [Overview](#overview)
2. [Architecture Components](#architecture-components)
3. [Data Flow](#data-flow)
4. [Configuration Structure](#configuration-structure)
5. [Data Storage Locations](#data-storage-locations)
6. [Provider Implementation Details](#provider-implementation-details)
7. [Adding a New Provider](#adding-a-new-provider)
8. [Important Functions and Their Roles](#important-functions-and-their-roles)

---

## Overview

The climate data system uses a **provider pattern** to support multiple climate data sources (nextGEMS, ICCP, AWI_CM) with a unified interface. All providers return the same output format (`ClimateDataResult`), making it easy to switch between data sources at runtime.

### Key Design Principles
- **Unified Output**: All providers return `ClimateDataResult` with the same structure
- **Runtime Selection**: Users can switch data sources via UI dropdown
- **Backwards Compatibility**: Legacy config format auto-migrates
- **Extensibility**: Adding new providers requires implementing the `ClimateDataProvider` interface

---

## Architecture Components

### 1. Core Module: `climate_data_providers.py`

**Location**: `src/climsight/climate_data_providers.py`

**Key Classes**:

```python
@dataclass
class ClimateDataResult:
    """Unified output for all providers"""
    df_list: List[Dict]              # List of dataframes with metadata
    data_agent_response: Dict         # LLM-ready formatted data
    source_name: str                  # "nextGEMS", "ICCP", or "AWI_CM"
    source_description: str           # Human-readable description
```

```python
class ClimateDataProvider(ABC):
    """Abstract base class - all providers must implement this"""
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def coordinate_system(self) -> str: ...

    @abstractmethod
    def extract_data(self, lon, lat, months) -> ClimateDataResult: ...

    @abstractmethod
    def is_available(self) -> bool: ...
```

**Implementations**:
- `NextGEMSProvider`: HEALPix unstructured grid, uses cKDTree + inverse distance weighting
- `ICCPProvider`: Regular lat/lon grid (stub - raises `NotImplementedError`)
- `AWICMProvider`: Regular lat/lon grid, wraps legacy `climate_functions.py`

**Factory Functions**:
- `get_climate_data_provider(config, source_override)`: Returns appropriate provider instance
- `get_available_providers(config)`: Returns list of providers with data available
- `migrate_legacy_config(config)`: Converts old config format to new

### 2. Extraction Functions: `extract_climatedata_functions.py`

**Location**: `src/climsight/extract_climatedata_functions.py`

**Purpose**: Backwards-compatible wrapper and utility functions

**Key Functions**:
```python
def request_climate_data(config, lon, lat, months, source_override=None):
    """Backwards-compatible function - now uses provider internally"""
    # Returns: (data_agent_response, df_list)

def request_climate_data_with_provider(config, lon, lat, months, source_override):
    """New function - returns ClimateDataResult directly"""

def plot_climate_data(df_list):
    """Creates matplotlib figures from df_list"""
```

### 3. Engine Integration: `climsight_engine.py`

**Location**: `src/climsight/climsight_engine.py`

**Where Data is Extracted**:

1. **Early extraction** (line ~449-475): During initial data gathering
   ```python
   climate_source = config.get('climate_data_source', None)
   config = migrate_legacy_config(config)
   data_agent_response, df_list = request_climate_data(config, lon, lat, source_override=climate_source)
   data['climate_data'] = {
       'df_list': df_list,
       'data_agent_response': data_agent_response,
       'lon': lon, 'lat': lat,
       'source': climate_source or config.get('climate_data_source', 'nextGEMS')
   }
   ```

2. **In data_agent()** (line ~831-877): During agent workflow
   ```python
   def data_agent(state: AgentState, data={}, df={}):
       climate_source = config.get('climate_data_source', 'nextGEMS')
       data_agent_response, df_list = request_climate_data(config, lon, lat, source_override=climate_source)
       data['climate_data'] = {...}
       state.df_list = df_list  # Store in agent state for smart_agent
   ```

### 4. Smart Agent Integration: `smart_agent.py`

**Location**: `src/climsight/smart_agent.py`

**Where Data is Used** (line ~235-315):

The `get_data_components` tool uses `state.df_list` to extract specific variables for the LLM agent.

```python
def get_data_components(...):
    climate_source = config.get('climate_data_source', 'nextGEMS')
    df_list = getattr(state, 'df_list', None)

    # Variable mapping per source
    if climate_source == 'nextGEMS':
        environmental_mapping = {"Temperature": "mean2t", "Precipitation": "tp", ...}
    elif climate_source == 'ICCP':
        environmental_mapping = {"Temperature": "2t", "Wind U": "10u", ...}
    elif climate_source == 'AWI_CM':
        environmental_mapping = {"Temperature": "Present Day Temperature", ...}
```

### 5. UI Integration: `streamlit_interface.py`

**Location**: `src/climsight/streamlit_interface.py`

**Data Source Selector** (line ~135-168):
```python
config_migrated = migrate_legacy_config(config)
available_sources = get_available_providers(config_migrated)
selected_source = st.selectbox("Climate Data Source:", display_options, ...)
config['climate_data_source'] = selected_source
```

**Display Logic** (line ~354-418):
```python
climate_data = data_pocket.data.get('climate_data') or data_pocket.data.get('high_res_climate')
if climate_data and 'df_list' in climate_data:
    df_list = climate_data['df_list']
    climate_source = climate_data.get('source', 'unknown')
    figs_climate = plot_climate_data(df_list)
    st.markdown(f"**Climate data ({climate_source}):**")
```

---

## Data Flow

### Complete Data Journey

```
1. USER SELECTS SOURCE IN UI
   ↓
   streamlit_interface.py: config['climate_data_source'] = 'nextGEMS'

2. DATA EXTRACTION REQUEST
   ↓
   climsight_engine.py: request_climate_data(config, lon, lat)
   ↓
   extract_climatedata_functions.py: migrate_legacy_config(config)
   ↓
   extract_climatedata_functions.py: get_climate_data_provider(config)
   ↓
   climate_data_providers.py: NextGEMSProvider/ICCPProvider/AWICMProvider

3. PROVIDER EXTRACTS DATA
   ↓
   NextGEMSProvider.extract_data(lon, lat, months):
     - Build spatial index (cKDTree)
     - Find 4 nearest neighbors
     - Inverse distance weighting interpolation
     - Post-process (unit conversions, wind speed calculation)
     - Return ClimateDataResult

4. DATA STORAGE IN ENGINE
   ↓
   data['climate_data'] = {
       'df_list': [...],           # List of dataframes with metadata
       'data_agent_response': {...}, # Formatted for LLM
       'lon': lon, 'lat': lat,
       'source': 'nextGEMS'
   }
   ↓
   state.df_list = df_list  # Stored in agent state

5. DATA USAGE
   ↓
   - LLM agents: Use data_agent_response for prompts
   - Smart agent: Uses state.df_list via get_data_components tool
   - UI display: Uses data_pocket.data['climate_data']['df_list'] for plotting

6. USER SEES RESULTS
   ↓
   streamlit_interface.py: Plots climate data with source label
```

---

## Configuration Structure

### New Format (config.yml)

```yaml
# Active data source selector
climate_data_source: "nextGEMS"  # Options: "nextGEMS", "ICCP", "AWI_CM"

# Provider configurations
climate_data_sources:
  nextGEMS:
    enabled: true
    coordinate_system: "healpix"
    description: "nextGEMS high-resolution climate simulations"
    input_files:
      climatology_IFS_9-FESOM_5-production_2020x_compressed.nc:
        file_name: './data/IFS_9-FESOM_5-production/...'
        years_of_averaging: '2020-2029'
        coordinate_system: 'healpix'
        is_main: true  # Reference period
        source: 'Model description for citations'
    variable_mapping:
      Temperature: mean2t
      Total Precipitation: tp
      Wind U: wind_u
      Wind V: wind_v

  ICCP:
    enabled: true
    coordinate_system: "regular"
    description: "ICCP climate reanalysis data"
    input_files: {}  # To be added when data available
    variable_mapping:
      Temperature: 2t
      Wind U: 10u
      Wind V: 10v

  AWI_CM:
    enabled: true
    coordinate_system: "regular"
    description: "AWI-CM CMIP6 climate model data"
    data_path: "./data/"
    historical_pattern: "historical"
    projection_pattern: "ssp585"
    variable_mapping:
      Temperature: tas
      Precipitation: pr
      u_wind: uas
      v_wind: vas
    dimension_mappings:
      latitude: "lat"
      longitude: "lon"
      time: "month"
```

### Legacy Format (auto-migrated)

```yaml
# Old format - automatically converted by migrate_legacy_config()
use_high_resolution_climate_model: true
climate_model_input_files:
  file1.nc: {...}
climate_model_variable_mapping:
  Temperature: mean2t
```

**Migration Logic** (`climate_data_providers.py:migrate_legacy_config()`):
- Detects `use_high_resolution_climate_model` flag
- If `true` → sets `climate_data_source: "nextGEMS"`
- If `false` → sets `climate_data_source: "AWI_CM"`
- Creates `climate_data_sources` dict from existing config sections
- Preserves all legacy settings for backwards compatibility

---

## Data Storage Locations

### 1. In-Memory Storage

**During Workflow Execution**:

```python
# In climsight_engine.py
data = {
    'climate_data': {
        'df_list': [...],              # Raw dataframes
        'data_agent_response': {...},   # LLM-formatted data
        'lon': float,
        'lat': float,
        'source': str
    },
    'high_res_climate': {...},  # Alias for backwards compatibility
}

# In AgentState (climsight_classes.py)
state.df_list = [...]  # Shared across agents
```

**In Data Pocket** (for UI persistence):

```python
# In streamlit_interface.py
data_pocket.data['climate_data'] = data['climate_data']
data_pocket.df['df_data'] = df_data  # Legacy AWI_CM format
```

### 2. File System Storage

**NetCDF Files** (actual climate data):
- nextGEMS: `./data/IFS_9-FESOM_5-production/*.nc`
- ICCP: `./data/iccp/*.nc` (when available)
- AWI_CM: `./data/AWI_CM_*.nc`

**Config File**:
- `config.yml`: User-editable configuration

**Logs**:
- `climsight.log`: Detailed execution logs

---

## Provider Implementation Details

### NextGEMSProvider (HEALPix Grid)

**Coordinate System**: Unstructured HEALPix grid

**Spatial Index**: cKDTree built on lon/lat points

**Interpolation Method**:
1. Query 4 nearest neighbors using cKDTree
2. Project to stereographic coordinates centered at target point
3. Compute inverse distance weights: `weights = 1/distances / sum(1/distances)`
4. Interpolate: `value = sum(weights * neighbor_values)`
5. Handle exact matches (distance = 0)

**Post-Processing**:
- Temperature: K → °C (subtract 273.15)
- Precipitation: m → mm/month (multiply by 1000)
- Wind: Calculate speed and direction from u/v components
- Round all values to 2 decimal places

**Output Structure**:
```python
df_list = [
    {
        'filename': 'climatology_...2020x.nc',
        'years_of_averaging': '2020-2029',
        'description': '...',
        'dataframe': pd.DataFrame({
            'Month': ['January', 'February', ...],
            'mean2t': [temp values],
            'tp': [precip values],
            'wind_u': [...], 'wind_v': [...],
            'wind_speed': [...], 'wind_direction': [...]
        }),
        'extracted_vars': {
            'mean2t': {'name': 'mean2t', 'units': '°C', 'full_name': 'Temperature', ...},
            ...
        },
        'main': True,  # Is this the reference period?
        'source': 'Citation text'
    },
    # ... more time periods (2030x, 2040x)
]
```

### ICCPProvider (Regular Grid - STUB)

**Coordinate System**: Regular lat/lon grid (192 x 400)

**Current Status**: Stub implementation - raises `NotImplementedError`

**Implementation Plan** (when data available):
```python
def extract_data(self, lon, lat, months=None):
    # 1. Open NetCDF file
    ds = xr.open_dataset(file_path)

    # 2. Handle longitude convention (ICCP uses 0-360)
    if lon < 0:
        lon = lon + 360

    # 3. Simple nearest neighbor or bilinear interpolation
    data = ds.sel(lat=lat, lon=lon, method='nearest')
    # or: data = ds.interp(lat=lat, lon=lon)

    # 4. Extract variables and convert units
    temp = data['2t'].values - 273.15  # K to °C
    precip = data['tp'].values * 1000   # m to mm
    wind_u = data['10u'].values
    wind_v = data['10v'].values

    # 5. Create dataframe and return ClimateDataResult
    ...
```

### AWICMProvider (Regular Grid)

**Coordinate System**: Regular lat/lon grid

**Implementation**: Wraps existing `climate_functions.py`

**Key Methods**:
```python
def extract_data(self, lon, lat, months=None):
    # 1. Build legacy config format
    legacy_config = self._get_legacy_config()

    # 2. Use existing functions
    hist, future = load_data(legacy_config)
    df_data, data_dict = extract_climate_data(lat, lon, hist, future, legacy_config)

    # 3. Convert to ClimateDataResult format
    df_list = [
        {'filename': 'AWI-CM historical', 'dataframe': df_hist, ...},
        {'filename': 'AWI-CM SSP5-8.5', 'dataframe': df_future, ...}
    ]
    return ClimateDataResult(df_list, data_agent_response, 'AWI_CM', description)
```

---

## Adding a New Provider

### Step-by-Step Guide

**1. Create Provider Class** in `climate_data_providers.py`:

```python
class NewModelProvider(ClimateDataProvider):
    @property
    def name(self) -> str:
        return "NewModel"

    @property
    def coordinate_system(self) -> str:
        return "regular"  # or "healpix", "curvilinear", etc.

    def is_available(self) -> bool:
        """Check if data files exist"""
        input_files = self.source_config.get('input_files', {})
        for file_key, meta in input_files.items():
            file_name = meta.get('file_name', file_key)
            if os.path.exists(file_name):
                return True
        return False

    def extract_data(self, lon, lat, months=None):
        """Extract climate data for location"""
        # 1. Load your NetCDF files
        # 2. Interpolate to point
        # 3. Extract variables
        # 4. Convert units
        # 5. Create dataframe(s)
        # 6. Return ClimateDataResult
        ...
```

**2. Update Factory Function** in `climate_data_providers.py`:

```python
def get_climate_data_provider(config, source_override=None):
    source = source_override or config.get('climate_data_source', 'nextGEMS')
    sources_config = config.get('climate_data_sources', {})

    if source == 'nextGEMS':
        return NextGEMSProvider(sources_config.get('nextGEMS', {}), config)
    elif source == 'ICCP':
        return ICCPProvider(sources_config.get('ICCP', {}), config)
    elif source == 'AWI_CM':
        return AWICMProvider(sources_config.get('AWI_CM', {}), config)
    elif source == 'NewModel':  # ADD THIS
        return NewModelProvider(sources_config.get('NewModel', {}), config)
    else:
        raise ValueError(f"Unknown climate data source: {source}")
```

**3. Add to Available Sources List** in `climate_data_providers.py`:

```python
def get_available_providers(config):
    available = []
    for source in ['nextGEMS', 'ICCP', 'AWI_CM', 'NewModel']:  # ADD NewModel
        try:
            provider = get_climate_data_provider(config, source)
            if provider.is_available():
                available.append(source)
        except Exception as e:
            logger.debug(f"Provider {source} not available: {e}")
    return available
```

**4. Add Config Section** in `config.yml`:

```yaml
climate_data_sources:
  NewModel:
    enabled: true
    coordinate_system: "regular"
    description: "New climate model dataset"
    input_files:
      newmodel_2020.nc:
        file_name: './data/newmodel/newmodel_2020.nc'
        years_of_averaging: '2020-2030'
        is_main: true
    variable_mapping:
      Temperature: t2m
      Precipitation: precip
```

**5. Update UI Display Names** in `streamlit_interface.py`:

```python
source_descriptions = {
    'nextGEMS': 'nextGEMS (High-resolution)',
    'ICCP': 'ICCP (Reanalysis)',
    'AWI_CM': 'AWI-CM (CMIP6)',
    'NewModel': 'New Model (Description)'  # ADD THIS
}
```

**6. (Optional) Update smart_agent.py** variable mappings:

```python
if climate_source == 'NewModel':
    environmental_mapping = {
        "Temperature": "t2m",
        "Precipitation": "precip",
        ...
    }
```

---

## Important Functions and Their Roles

### Core Functions Summary

| Function | File | Purpose | Input | Output |
|----------|------|---------|-------|--------|
| `get_climate_data_provider()` | climate_data_providers.py | Factory to get provider | config, source_override | Provider instance |
| `get_available_providers()` | climate_data_providers.py | List usable providers | config | List of source names |
| `migrate_legacy_config()` | climate_data_providers.py | Convert old config | config dict | Updated config dict |
| `request_climate_data()` | extract_climatedata_functions.py | Get climate data (compat) | config, lon, lat | (response, df_list) |
| `plot_climate_data()` | extract_climatedata_functions.py | Create matplotlib plots | df_list | List of figure dicts |
| `data_agent()` | climsight_engine.py | Extract data in workflow | AgentState | Updated state dict |

### Caching and Performance

**Spatial Index Caching** (NextGEMSProvider):
```python
self._spatial_indices = {}  # Cache cKDTree per file

def _build_spatial_index(self, nc_file):
    if nc_file in self._spatial_indices:
        return self._spatial_indices[nc_file]  # Return cached
    # Build new index and cache it
    ...
```

**Dataset Caching** (AWICMProvider):
```python
self._data_cache = {}  # Cache loaded datasets

def extract_data(self, lon, lat, months=None):
    if 'awi_cm_data' not in self._data_cache:
        hist, future = load_data(config)
        self._data_cache['awi_cm_data'] = (hist, future)
    else:
        hist, future = self._data_cache['awi_cm_data']
```

---

## Common Patterns and Conventions

### 1. Variable Naming Conventions

- **NetCDF variables**: Lowercase with underscores (e.g., `mean2t`, `wind_u`)
- **Display names**: Title case with spaces (e.g., `"Temperature"`, `"Wind Speed"`)
- **Config keys**: Underscores or camelCase (e.g., `climate_data_source`)

### 2. Unit Conversions

Always convert to standard units:
- Temperature: °C (convert from K: `temp - 273.15`)
- Precipitation: mm/month (from m: `precip * 1000`, from kg/m²/s: `value * 60 * 60 * 24 * days_in_month`)
- Wind: m/s (usually already in correct units)

### 3. Month Handling

Two formats used:
- **Full names**: `["January", "February", ...]` (for display)
- **Indices**: `[1, 2, ..., 12]` or `[0, 1, ..., 11]` (for data extraction)

Conversion:
```python
import calendar
month_name = calendar.month_name[month_index]  # 1-indexed
month_abbr = calendar.month_abbr[month_index]  # 1-indexed
```

### 4. Error Handling

```python
try:
    provider = get_climate_data_provider(config, source)
    result = provider.extract_data(lon, lat)
except NotImplementedError as e:
    # Provider exists but not implemented (e.g., ICCP)
    logger.warning(f"Provider not implemented: {e}")
except Exception as e:
    # Other errors
    logger.error(f"Error extracting data: {e}")
    raise RuntimeError(f"Climate data extraction failed: {e}")
```

---

## Troubleshooting Guide

### Common Issues

**1. Provider not appearing in UI dropdown**
- Check `is_available()` returns `True`
- Verify data files exist at configured paths
- Check logs for provider initialization errors

**2. Data extraction fails**
- Verify NetCDF files are not corrupted: `ncdump -h file.nc`
- Check coordinate ranges match query point
- Ensure variable names in config match NetCDF file

**3. Backwards compatibility broken**
- Ensure `migrate_legacy_config()` is called before provider creation
- Check legacy keys still present in config
- Verify `data['high_res_climate']` alias is set

**4. Smart agent can't access data**
- Verify `state.df_list` is set in `data_agent()`
- Check variable mapping for selected source in `smart_agent.py`
- Ensure `extracted_vars` dict contains required metadata

---

## Future Enhancements

### Planned Features

1. **ICCP Implementation**: Complete when data files available
2. **Data Caching**: Persistent cache for frequently-requested locations
3. **Parallel Processing**: Extract multiple time periods concurrently
4. **Additional Variables**: Sea level pressure, humidity, cloud cover
5. **Ensemble Data**: Support multiple model runs, uncertainty quantification
6. **Regional Downscaling**: High-res regional models as additional providers

### Extension Points

- Add new coordinate systems (curvilinear, rotated pole)
- Implement different interpolation methods (kriging, spline)
- Support temporal aggregation (seasonal, annual averages)
- Add data quality flags and metadata

---

## References

### Key Files
- [climate_data_providers.py](src/climsight/climate_data_providers.py)
- [extract_climatedata_functions.py](src/climsight/extract_climatedata_functions.py)
- [climsight_engine.py](src/climsight/climsight_engine.py)
- [smart_agent.py](src/climsight/smart_agent.py)
- [streamlit_interface.py](src/climsight/streamlit_interface.py)
- [config.yml](config.yml)

### External Documentation
- nextGEMS: https://nextgems-h2020.eu/
- HEALPix: https://healpix.jpl.nasa.gov/
- xarray: https://docs.xarray.dev/
- scipy.spatial.cKDTree: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html

---

**Last Updated**: 2025-12-17
**Version**: 1.0 (Initial multi-provider architecture)
