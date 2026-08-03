# scripts/

The notebooks tell the story; these modules do the work. Anything that is data
loading, coordinate handling, region definition or a reusable computation
belongs here rather than in a cell, so that it is written once, documented once,
and can be tested.

## Modules

| module | what it holds |
| --- | --- |
| `config.py` | The data directory and every input file name. Set the directory with `config.set_data_path(...)` or the `D18O_DATA_PATH` environment variable. `data_file(key)` raises a message naming the variable to set if a file is missing; `figure_dir(name)` creates the output directory on first use. |
| `products.py` | The five gridded δ¹⁸O products behind a single `Product` type with a uniform `sample(depth, lat, lon)`, plus `load_products()` and `load_climatology()`. Also the colour and label of each product, so figures stay consistent. |
| `observations.py` | One loader per observational compilation, each applying that dataset's own missing-value rules. `add_product_columns` interpolates every product at the observation positions in one call; `add_climatology_columns` does the same for neutral density and salinity. |
| `regions.py` | Basin boundaries, the Southern Ocean limit, and the neutral-density / salinity window used to isolate North Atlantic influenced water. |
| `plotting.py` | Panel labels, the Southern Ocean zonal and cos(lat)-weighted area means, and the cumulative RMSE against depth. Axis-label strings for δ¹⁸O and γₙ. |
| `breitkreuz.py` | Reader for the Breitkreuz et al. (2018) product, normalising it to this project's conventions. See the module docstring for the padding options. |
| `usefull_functions.py` | The older helpers: depth levels, dataset splitting, colour norms, axis and colorbar helpers, NEMO grid handling. |
| `custom_density_scale.py` | Registers the `'custom_scale'` matplotlib axis used for neutral density. Import it for the side effect before calling `ax.set_yscale('custom_scale')`. |
| `tracer_optimization.py` | End-member optimisation; see `README_tracer_optimization.md`. |

## Conventions

Every product is exposed the same way, whatever its file says:

- values on a `(depth, lat, lon)` array, in permil VSMOW;
- **depth positive downwards**, in metres;
- **longitude in 0-360**, strictly increasing.

Observation frames keep their original column names. Interpolated product values
are added as new columns, named by the `columns` argument of
`add_product_columns` so that a notebook can keep whatever names its figures
already use.

A point outside a product's grid, or on one of its land cells, comes back as
NaN. A `dropna()` over a frame carrying several products therefore reduces it to
the observations *every* product covers, which is what makes their RMSE
comparable — but it does discard points, so it is worth printing the count.