# Tracer Optimization Module

A Python module for optimizing tracer end-member values using water mass fractions from different ocean circulation models.

## Overview

This module generalizes the optimization functionality from the `optimization_fractions.ipynb` notebook, allowing you to optimize any tracer (d18O, Salinity, etc.) using water mass fractions from TMI, OCIM, and NEMO models.

## Files

- **`tracer_optimization.py`** - Main module containing the `TracerOptimizer` class
- **`test_tracer_optimization.py`** - Test script to verify functionality
- **`example_usage.py`** - Comprehensive examples showing different usage patterns

## Key Features

### Generalized Tracer Optimization
- Optimize any tracer (d18O, Salinity, Temperature, etc.)
- Support for multiple circulation models (TMI, OCIM, NEMO)
- Flexible data filtering and quality control

### Robust Statistical Analysis
- Automatic calculation of mean and standard deviation
- Customizable exclusion criteria for problematic datasets
- Statistical validation of optimization results

### Easy-to-Use Interface
- Convenience functions for common tracers (d18O, Salinity)
- Complete workflow automation
- Detailed result summaries and diagnostics

## Quick Start

### Basic Usage - d18O Optimization

```python
from tracer_optimization import optimize_d18o

# Set your data path
data_path = r'D:/Data/d18o_so/'

# Run d18O optimization (replicates notebook functionality)
results = optimize_d18o(data_path)

# Display summary
print(results['summary'])
```

### Basic Usage - Salinity Optimization

```python
from tracer_optimization import optimize_salinity

# Run Salinity optimization using the same approach
results = optimize_salinity(data_path)

# Display summary
print(results['summary'])
```

### Advanced Usage - Custom Parameters

```python
from tracer_optimization import TracerOptimizer
import numpy as np

# Initialize optimizer
optimizer = TracerOptimizer(data_path)

# Load model data
optimizer.load_model_data()
optimizer.setup_interpolators()

# Prepare data with custom filtering
tracer_data = optimizer.prepare_tracer_data(
    tracer_file='your_tracer_file.txt',
    tracer_column='YourTracerColumn',
    min_depth=150.0,
    gamma_range=(27.0, 28.6),
    latitude_threshold=-40.0
)

# Create optimization datasets
datasets = optimizer.create_optimization_datasets(tracer_data)

# Set up optimization parameters
initial_values = np.array([val1, val2, val3, val4, val5, val6])
lower_bounds = np.array([low1, low2, low3, low4, low5, low6])
upper_bounds = np.array([up1, up2, up3, up4, up5, up6])
bounds = (lower_bounds, upper_bounds)

# Run optimization
results = optimizer.optimize_tracer(
    datasets, 'YourTracerColumn', initial_values, 
    bounds=bounds, models_to_use=['TMI', 'OCIM']
)

# Calculate statistics
stats = optimizer.calculate_statistics('YourTracerColumn', datasets)

# Reconstruct tracer values
datasets = optimizer.reconstruct_tracer(datasets, 'YourTracerColumn')
```

## Complete Workflow

The `run_complete_optimization` method runs the entire workflow:

```python
optimizer = TracerOptimizer(data_path)

results = optimizer.run_complete_optimization(
    tracer_file='giss_d18o.txt',
    tracer_column='d18O',
    initial_values=np.array([0.45, -0.3, -0.35, -0.2, 0.3, -0.4]),
    min_depth=150.0,
    gamma_range=(27.0, 28.6)
)
```

## Method Reference

### TracerOptimizer Class

#### Core Methods

- **`load_model_data()`** - Load water mass fraction data from TMI, OCIM, NEMO
- **`setup_interpolators()`** - Set up interpolation functions for all models
- **`prepare_tracer_data(tracer_file, tracer_column, **kwargs)`** - Load and filter tracer observations
- **`create_optimization_datasets(tracer_data)`** - Separate data by laboratory/location
- **`optimize_tracer(datasets, tracer_column, initial_values, **kwargs)`** - Perform optimization
- **`calculate_statistics(tracer_column, datasets, **kwargs)`** - Calculate robust statistics
- **`reconstruct_tracer(datasets, tracer_column, **kwargs)`** - Reconstruct tracer fields
- **`run_complete_optimization(tracer_file, tracer_column, initial_values, **kwargs)`** - Complete workflow

#### Utility Methods

- **`get_optimization_summary(tracer_column)`** - Get summary DataFrame of results

### Convenience Functions

- **`optimize_d18o(data_path, **kwargs)`** - Pre-configured d18O optimization
- **`optimize_salinity(data_path, **kwargs)`** - Pre-configured Salinity optimization

## Parameters

### Data Preparation Parameters

- **`tracer_file`** - Name of tracer data file (e.g., 'giss_d18o.txt')
- **`tracer_column`** - Column name containing tracer values (e.g., 'd18O', 'Salinity')
- **`min_depth`** - Minimum depth threshold (default: 150.0)
- **`gamma_range`** - Tuple of (min_gamma, max_gamma) for density filtering (default: (27.0, 28.6))
- **`latitude_threshold`** - Latitude cutoff for Southern Ocean (default: -40.0)

### Optimization Parameters

- **`initial_values`** - Array of 6 initial guesses for end-member values
- **`bounds`** - Tuple of (lower_bounds, upper_bounds) arrays
- **`models_to_use`** - List of models to use (default: ['TMI', 'OCIM', 'NEMO'])

### Statistics Parameters

- **`exclusion_criteria`** - Dictionary specifying dataset exclusion rules:
  - `min_fraction_threshold`: Minimum fraction threshold (default: 0.2)
  - `required_dyes`: List of required water mass tracers
  - `exclude_indices`: List of dataset indices to exclude

## Output Structure

The optimization returns a dictionary containing:

- **`tracer_data`** - Processed tracer dataset
- **`datasets`** - List of datasets separated by laboratory/location
- **`optimization_results`** - Raw optimization results for each model
- **`statistics`** - Mean and standard deviation of end-member values
- **`summary`** - DataFrame summary of all results

## Data Requirements

### Required Data Files

1. **Tracer observations** (e.g., `giss_d18o.txt`)
   - Columns: Latitude, Longitude, Depth, [TracerColumn], Salinity, Reference
   
2. **TMI water mass fractions** (`TMI_2deg_2010_water_mass_fractions.nc`)
   - Variables: dyeLL, dyeMS, dyeNP, dyeHS, dyeNA, dyeAA
   
3. **OCIM water mass fractions** (`ocim_steady_dyes.nc`)
   - Variables: dyeLL_steady, dyeMS_steady, etc.
   
4. **NEMO water mass fractions** (`tm21ah21_extrapolated_dyes_regridded.nc`)
   - Variables: DyeLL, DyeMS, DyeNP, DyeHS, DyeNA, DyeAA
   
5. **Climatology data** (`climatology_Brunov2.nc`)
   - Variables: gamma, absolute_salinity

### Directory Structure

```
your_data_directory/
├── giss_d18o.txt
├── TMI_2deg_2010_water_mass_fractions.nc
├── ocim_steady_dyes.nc
├── tm21ah21_extrapolated_dyes_regridded.nc
└── climatology_Brunov2.nc
```

## Water Mass Definitions

The module optimizes end-member values for 6 water masses:

1. **dyeLL** - Lower Labrador Sea Water
2. **dyeMS** - Mediterranean Sea Water  
3. **dyeNP** - North Pacific Water
4. **dyeHS** - High Salinity Water
5. **dyeNA** - North Atlantic Water
6. **dyeAA** - Antarctic Bottom Water

## Example Applications

### 1. Replicate Notebook Results
Use `optimize_d18o()` with default parameters to get the same results as the original notebook.

### 2. Salinity Analysis
Use `optimize_salinity()` to optimize salinity end-member values using the same water mass framework.

### 3. Custom Tracer Studies
Use the `TracerOptimizer` class directly for any tracer with appropriate initial values and bounds.

### 4. Sensitivity Analysis
Modify filtering parameters, bounds, or exclusion criteria to test sensitivity of results.

## Error Handling

The module includes comprehensive error handling:
- Validates input data formats
- Handles missing or invalid observations
- Reports optimization convergence issues
- Provides detailed error messages for debugging

## Performance Notes

- Loading model data takes ~10-30 seconds depending on file sizes
- Optimization typically completes in under 1 minute for standard datasets
- Memory usage scales with the size of water mass fraction fields

## Dependencies

- `numpy` - Numerical computations
- `pandas` - Data manipulation
- `xarray` - NetCDF file handling
- `scipy` - Optimization algorithms
- `usefull_functions` - Custom utility functions (must be in same directory)

## Testing

Run the test suite to verify functionality:

```bash
python test_tracer_optimization.py
```

Or run the examples:

```bash
python example_usage.py
```

## Troubleshooting

### Common Issues

1. **File not found errors**: Check that `data_path` points to correct directory
2. **Import errors**: Ensure `usefull_functions.py` is in the scripts directory
3. **Optimization failures**: Check initial values and bounds are reasonable
4. **Empty datasets**: Verify filtering parameters aren't too restrictive

### Getting Help

Check the example scripts for usage patterns, or examine the original notebook for parameter values and expected behavior.


# To do

- Remove dsClim from the TracerOptimizer Class: not needed anymore.
- Remove gamma and salinity interpolator (climatology interpolators)