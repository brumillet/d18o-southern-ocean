README

This GitHub repository contains 3 different notebooks:

- d18o_ocim_v0: A notebook that was made to look if we could reconstruct the d18o product made by Mark Holzer, which propagated the d18o of the Legrande and Schmidt product of the surface ocean, in the deep ocean. This notebook was made with a previous OCIM output using only 5 surface dyes (LL and MS combined). Here we also how it compares to the d18o SO Giss compilation.

- optimization_fractions: The notebook where we compute the different d18o products and surface values based on the model fractions.

- evaluation_reconstructions: Notebook used to compare the different reconstructions and to compare our results to other studies. It also compares them to the Breitkreuz et al. (2018) product (see below), both against the observations and directly product to product in the Southern Ocean.


The data used in this study can be downloaded at: (add repository). To point the code at your own
copy, either set the D18O_DATA_PATH environment variable or edit the config.set_data_path(...) call
in the second cell of the notebook.

The data loading, region definitions and reusable computations live in scripts/ rather than in the
notebooks; see scripts/README.md for what each module holds and for the conventions the products
follow (depth positive downwards, longitude 0-360).


The Breitkreuz et al. (2018) product
------------------------------------

Breitkreuz, C., Paul, A., Kurahashi-Nakamura, T., Losch, M., Schulz, M. (2018): A dynamical
reconstruction of the global monthly-mean oxygen isotopic composition of seawater. JGR Oceans,
123(10), 7206-7219. https://doi.org/10.1029/2018JC014300

It is a d18Osw product obtained by assimilating the global d18Osw compilation and climatological
T/S into an ocean general circulation model with the adjoint method, so it is independent both of
the LeGrande & Schmidt climatology and of the water-mass-fraction reconstructions computed here.

Download D18O_Breitkreuz_et_al_2018.nc (~300 MB) from https://doi.org/10.1594/PANGAEA.889922 and
place it in data_path. scripts/breitkreuz.py loads it and normalises it to this project's
conventions (longitude 0-360, depth positive downwards, annual mean by default); see its docstring
for the padding options. Check the download with:

    python scripts/test_breitkreuz.py <path to D18O_Breitkreuz_et_al_2018.nc>



