# -*- coding: utf-8 -*-
"""
Created on December 3, 2025

Tracer Optimization Module for Water Mass Analysis

This module provides functionality for optimizing tracer end-member values
(e.g., d18O, Salinity) using water mass fractions from different models
(TMI, OCIM, NEMO).

@author: bmillet
"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import least_squares
from scipy.interpolate import RegularGridInterpolator
import warnings
from typing import List, Dict, Tuple, Optional, Union
import usefull_functions as uf

dico_suffixes = {'TMI': '_TMI', 'OCIM': '_ocim', 'NEMO': '_nemo'}

class TracerOptimizer:
    """
    A class for optimizing tracer end-member values using water mass fractions.
    
    This class can handle optimization for different tracers (d18O, Salinity, etc.)
    and different circulation models (TMI, OCIM, NEMO).
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the TracerOptimizer.
        
        Parameters:
        -----------
        data_path : str
            Path to the directory containing the required data files
        """
        self.data_path = data_path
        self.models = ['TMI', 'OCIM', 'NEMO']
        self.dyes = uf.dyes_TMI  # ['dyeLL', 'dyeMS', 'dyeNP', 'dyeHS', 'dyeNA', 'dyeAA']
        self.n_dyes = len(self.dyes)
        
        # Storage for loaded data
        self.interpolators = {}
        self.model_data = {}
        self.climatology_interpolators = {}
        
        # Results storage
        self.optimization_results = {}
        self.end_member_values = {}
        self.mean_end_members = {}
        self.std_end_members = {}
        
        self.exclusion_indices = {}
        
    def load_model_data(self):
        """
        Load water mass fraction data from different models.
        """
        print("Loading model data...")
        
        # Load TMI data
        TMI_2deg = xr.open_dataset(self.data_path + 'TMI_2deg_2010_water_mass_fractions.nc')
        self.model_data['TMI'] = {
            'lat': TMI_2deg['yt'].values.copy(),
            'lon': TMI_2deg['xt'].values.copy(),
            'depth': TMI_2deg['zt'].values,
            'fractions': np.array([TMI_2deg[dye].values for dye in self.dyes])
        }
        
        # Load OCIM data
        ocim_frac = xr.open_dataset(self.data_path + 'ocim_steady_dyes.nc')
        self.model_data['OCIM'] = {
            'lat': ocim_frac['yt'].values,
            'lon': ocim_frac['xt'].values,
            'depth': ocim_frac['zt'].values,
            'fractions': np.array([ocim_frac[dye + '_steady'].values for dye in self.dyes])
        }
        
        # Load NEMO data
        nemo_dyes = xr.open_dataset(self.data_path + 'tm21ah21_extrapolated_dyes_regridded.nc')
        dsClim = xr.open_dataset(self.data_path + "climatology_Brunov2.nc")
        
        self.model_data['NEMO'] = {
            'lat': dsClim['lat'].values,
            'lon': dsClim['lon'].values,
            'depth': nemo_dyes['depth'].values,
            'fractions': np.array([nemo_dyes[uf.dyes[i]].values for i in range(6)])
        }
        
        # Store climatology data for interpolation
        self.climatology_data = {
            'lat': dsClim['lat'].values,
            'lon': dsClim['lon'].values,
            'depth': uf.create_l_depth(),
            'gamma': dsClim['gamma'].values,
            'salinity': dsClim['absolute_salinity'].values
        }
        
        ocim_frac.close()
        TMI_2deg.close()
        nemo_dyes.close()
        dsClim.close()
        
    
    def setup_interpolators(self):
        """
        Set up interpolation functions for all models and climatology data.
        """
        print("Setting up interpolators...")
        
        # Setup model interpolators
        for model in self.models:
            data = self.model_data[model]
            self.interpolators[model] = []
            
            for i in range(self.n_dyes):
                interpolator = RegularGridInterpolator(
                    (data['depth'], data['lat'], data['lon']), 
                    data['fractions'][i], 
                    bounds_error=False, 
                    fill_value=None
                )
                self.interpolators[model].append(interpolator)
        
        # Setup climatology interpolators
        clim = self.climatology_data
        self.climatology_interpolators['gamma'] = RegularGridInterpolator(
            (clim['depth'], clim['lat'], clim['lon']), 
            clim['gamma'], 
            bounds_error=False, 
            fill_value=None
        )
        self.climatology_interpolators['salinity'] = RegularGridInterpolator(
            (clim['depth'], clim['lat'], clim['lon']), 
            clim['salinity'], 
            bounds_error=False, 
            fill_value=None
        )
            

    
    def prepare_tracer_data(self, tracer_file: str, tracer_column: str, 
                          min_depth: float = 150.0, gamma_range: Tuple[float, float] = (27.0, 28.6),
                          latitude_threshold: float = -40.0) -> pd.DataFrame:
        """
        Load and prepare tracer data for optimization.
        
        Parameters:
        -----------
        tracer_file : str
            Name of the tracer data file (e.g., 'giss_d18o.txt')
        tracer_column : str
            Name of the column containing tracer values (e.g., 'd18O', 'Salinity')
        min_depth : float, optional
            Minimum depth threshold for data filtering (default: 150.0)
        gamma_range : tuple, optional
            Gamma density range for filtering (default: (27.0, 28.6))
        latitude_threshold : float, optional
            Latitude threshold for Southern Ocean data (default: -40.0)
            
        Returns:
        --------
        pd.DataFrame
            Processed tracer data with interpolated water mass fractions
        """
        print(f"Preparing tracer data from {tracer_file}...")
        
        # Load tracer data
        tracer_data = pd.read_table(self.data_path + tracer_file)
        
        # Convert longitude to 0-360 format
        tracer_data['Longitude'] = [lon if lon >= 0 else 360 + lon for lon in tracer_data['Longitude']]
        
        # Remove invalid data
        tracer_data = tracer_data.replace('**', np.nan)
        
        # Data quality filters
        tracer_data = tracer_data.where(
            (tracer_data['Depth'] != -999.0) & 
            (~tracer_data[tracer_column].isna()) & 
            (~tracer_data['Salinity'].isna()) &
            (tracer_data['Latitude'] <= latitude_threshold) &
            (tracer_data['Depth'] >= min_depth)
        ).dropna()
        
        # Convert to numeric
        tracer_data[tracer_column] = pd.to_numeric(tracer_data[tracer_column], errors='coerce')
        tracer_data['Salinity'] = pd.to_numeric(tracer_data['Salinity'], errors='coerce')
        
        # Add gamma density from climatology
        coords = (tracer_data['Depth'], tracer_data['Latitude'], tracer_data['Longitude'])
        tracer_data['Gamma'] = self.climatology_interpolators['gamma'](coords)
        tracer_data['absolute_salinity'] = self.climatology_interpolators['salinity'](coords)
        
        # Apply gamma filter
        tracer_data = tracer_data[
            (tracer_data['Gamma'] >= gamma_range[0]) & 
            (tracer_data['Gamma'] <= gamma_range[1])
        ]
        
        # Add water mass fractions for all models (recalculate coords after filtering)
        coords_filtered = (tracer_data['Depth'], tracer_data['Latitude'], tracer_data['Longitude'])
        for model in self.models:
            for i, dye in enumerate(self.dyes):
                column_name = f'{dye}_{model}'
                tracer_data[column_name] = self.interpolators[model][i](coords_filtered)
        
        print(f"Tracer data prepared. Final dataset contains {len(tracer_data)} observations.")
        return tracer_data
    

    
    def create_optimization_datasets(self, tracer_data: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Create separate datasets for optimization using Southern Ocean specific method.
        
        Parameters:
        -----------
        tracer_data : pd.DataFrame
            Prepared tracer data
            
        Returns:
        --------
        List[pd.DataFrame]
            List of datasets separated by reference/location (Southern Ocean specific)
        """
        return uf.create_dfs(tracer_data)
    
    def optimize_tracer(self, datasets: List[pd.DataFrame], tracer_column: str,
                       initial_values: np.ndarray, bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                       models_to_use: Optional[List[str]] = None) -> Dict:
        """
        Perform optimization for tracer end-member values.
        
        Parameters:
        -----------
        datasets : List[pd.DataFrame]
            List of datasets for optimization
        tracer_column : str
            Name of the column containing tracer observations
        initial_values : np.ndarray
            Initial guess for end-member values (6 values for 6 water masses)
        bounds : tuple, optional
            Bounds for optimization (lower_bounds, upper_bounds)
        models_to_use : List[str], optional
            List of models to use (default: all models)
            
        Returns:
        --------
        Dict
            Dictionary containing optimization results for each model
        """
        if models_to_use is None:
            models_to_use = self.models
            
        print(f"Starting optimization for {tracer_column}...")
        
        results = {}
        end_member_values = {}
        
        # Initialize storage for each model
        for model in models_to_use:
            results[model] = []
            end_member_values[model] = np.empty((self.n_dyes, len(datasets)))
        
        # Set up default bounds if not provided
        if bounds is None:
            lower_bounds = np.full(self.n_dyes, -2.0)
            upper_bounds = np.full(self.n_dyes, 2.0)
            bounds = (lower_bounds, upper_bounds)
        
        # Optimize for each dataset
        for i, dataset in enumerate(datasets):
            if len(dataset) == 0:
                print(f"  Dataset {i+1}: Skipping empty dataset")
                continue
                            
            # Get observations
            tracer_obs = dataset[tracer_column].values
            
            # Optimize for each model
            for model in models_to_use:
                # Get fractions for this model
                fractions = np.array([dataset[f'{dye}_{model}'].values for dye in self.dyes])
                
                # Define objective function
                def objective(end_members):
                    reconstructed = end_members.dot(fractions)
                    return tracer_obs - reconstructed
                
                # Perform optimization
                try:
                    result = least_squares(objective, initial_values, bounds=bounds)
                    results[model].append(result)
                    end_member_values[model][:, i] = result.x
                    
                    if not result.success:
                        print(f"    {model}: Optimization failed - {result.message}")
                        
                except Exception as e:
                    print(f"    {model}: Error during optimization - {str(e)}")
                    # Fill with NaN for failed optimizations
                    end_member_values[model][:, i] = np.nan
                    results[model].append(None)
        
        # Store results
        self.optimization_results[tracer_column] = results
        self.end_member_values[tracer_column] = end_member_values
        
        print("Optimization complete.")
        return results

    
    def calculate_statistics(self, tracer_column: str, datasets: List[pd.DataFrame]) -> Dict:
        """
        Calculate mean and standard deviation of end-member values.
        
        Parameters:
        -----------
        tracer_column : str
            Name of the tracer column
        datasets : List[pd.DataFrame]
            List of datasets used in optimization
            
        Returns:
        --------
        Dict
            Dictionary containing mean and std for each model
        """
        if tracer_column not in self.end_member_values:
            raise ValueError(f"No optimization results found for {tracer_column}")
            
        print(f"Calculating statistics for {tracer_column}...")
        
        end_members = self.end_member_values[tracer_column]
        stats = {'mean': {}, 'std': {}}
        
        # Initialize nested structure
        for model in self.models:
            stats['mean'][model] = {}
            stats['std'][model] = {}
        
        for model in self.models:
            for idye, dye in enumerate(self.dyes):
                # Calculate general statistics excluding problematic datasets  
                valid_indices = [i for i in range(len(datasets)) if i not in self.exclusion_indices[model][dye]]

                if valid_indices:
                    valid_end_members = end_members[model][:, valid_indices]
                    stats['mean'][model][dye] = np.nanmean(valid_end_members, axis=1)[idye]
                    stats['std'][model][dye] = np.nanstd(valid_end_members, axis=1)[idye]
                else:
                    print(f"  {model}: Warning - No valid datasets for statistics calculation")
                    stats['mean'][model][dye] = np.full(self.n_dyes, np.nan)[idye]
                    stats['std'][model][dye] = np.full(self.n_dyes, np.nan)[idye]
        
        # Store statistics
        self.mean_end_members[tracer_column] = stats['mean']
        self.std_end_members[tracer_column] = stats['std']
        
        return stats
    
    def reconstruct_tracer(self, datasets: List[pd.DataFrame], tracer_column: str,
                          use_mean_values: bool = True) -> List[pd.DataFrame]:
        """
        Reconstruct tracer values using optimized end-member values.
        
        Parameters:
        -----------
        datasets : List[pd.DataFrame]
            List of datasets to add reconstructions to
        tracer_column : str
            Name of the tracer column
        use_mean_values : bool, optional
            Whether to use mean end-member values or individual optimized values
            
        Returns:
        --------
        List[pd.DataFrame]
            Datasets with added reconstruction columns
        """
        
        if tracer_column not in self.end_member_values:
            raise ValueError(f"No optimization results found for {tracer_column}")
        print(f"Reconstructing {tracer_column} values...")
        
        end_members = self.end_member_values[tracer_column]
        for i, dataset in enumerate(datasets):
            if len(dataset) == 0:
                continue
                
            for model in end_members.keys():
                # Individual reconstruction
                reconstruction = np.zeros(len(dataset))

                # end_members[model] should be a 2D array: (n_dyes, n_datasets)
                model_results = end_members[model]
                
                for j, dye in enumerate(self.dyes):
                    column_name = f'{dye}_{model}'
                    if column_name in dataset.columns:
                        # Use the j-th dye value from i-th dataset optimization
                        try:
                            if hasattr(model_results, 'shape') and len(model_results.shape) == 2:
                                # 2D array case: model_results[dye_index, dataset_index]
                                end_member_value = model_results[j, i]
                            else:
                                # Handle other cases - skip reconstruction for this dye
                                continue
                                
                            if not np.isnan(end_member_value):
                                reconstruction += dataset[column_name] * end_member_value
                                
                        except (IndexError, KeyError):
                            # Skip this dye if indexing fails
                            continue
                
                dataset[f'{tracer_column}_rcst_{model}'] = reconstruction
                
                # Mean reconstruction (if statistics available)
                if use_mean_values and tracer_column in self.mean_end_members:
                    mean_reconstruction = np.zeros(len(dataset))
                    mean_values = self.mean_end_members[tracer_column][model]
                    
                    for j, dye in enumerate(self.dyes):
                        column_name = f'{dye}_{model}'
                        if column_name in dataset.columns and dye in mean_values and not np.isnan(mean_values[dye]):
                            mean_reconstruction += dataset[column_name]*mean_values[dye]

                    dataset[f'{tracer_column}_rcst_mean_{model}'] = mean_reconstruction
        
        return datasets
    
    def get_optimization_summary(self, tracer_column: str) -> pd.DataFrame:
        """
        Get a summary of optimization results.
        
        Parameters:
        -----------
        tracer_column : str
            Name of the tracer column
            
        Returns:
        --------
        pd.DataFrame
            Summary of optimization results
        """
        if tracer_column not in self.end_member_values:
            raise ValueError(f"No optimization results found for {tracer_column}")
        
        summary_data = []
        
        for model in self.models:
            if tracer_column in self.mean_end_members and model in self.mean_end_members[tracer_column]:
                for dye in self.dyes:
                    row = {
                        'Model': model,
                        'Water_Mass': dye,
                        'Mean_End_Member': self.mean_end_members[tracer_column][model][dye],
                        'Std_End_Member': self.std_end_members[tracer_column][model][dye],
                        'N_Datasets': 'Available'
                    }
                    summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    
    def calculate_exclusion_indices(self, datasets: List[pd.DataFrame]):
        """
        Calculate and store exclusion indices for each model and dye.
        
        Parameters:
        -----------
        datasets : List[pd.DataFrame]
            List of datasets used in optimization
        """
        # Initialize nested dictionary structure
        for model in self.models:
            if model not in self.exclusion_indices:
                self.exclusion_indices[model] = {}
                
        # Define suffixes mapping
        dico_suffixes = {'TMI': '_TMI', 'OCIM': '_OCIM', 'NEMO': '_NEMO'}
        
        # Calculate exclusion indices for each model and dye
        for model in self.models:
            for dye in self.dyes:
                if dye in ['dyeLL', 'dyeNP']:
                    # For dyeLL and dyeNP, use empty exclusion list (no exclusion criteria)
                    self.exclusion_indices[model][dye] = []
                elif dye == 'dyeMS':
                    self.exclusion_indices[model][dye] = uf.index_to_exclude_dye(datasets, dye, dico_suffixes[model])
                else:
                    self.exclusion_indices[model][dye] = uf.index_to_exclude(datasets, dico_suffixes[model])

    def get_exclusion_indices(self, model: str, dye: str) -> List[int]:
        """
        Get exclusion indices for a specific model and dye.
        
        Parameters:
        -----------
        model : str
            Model name ('TMI', 'OCIM', 'NEMO')
        dye : str
            Dye name ('dyeHS', 'dyeNA', 'dyeAA', 'dyeMS')
            
        Returns:
        --------
        List[int]
            List of dataset indices to exclude
        """
        if model in self.exclusion_indices and dye in self.exclusion_indices[model]:
            return self.exclusion_indices[model][dye]
        else:
            return []
    

    
    def run_complete_optimization(self, tracer_file: str, tracer_column: str,
                                initial_values: np.ndarray, 
                                bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                                min_depth: float = 150.0, 
                                gamma_range: Tuple[float, float] = (27.0, 28.6),
                                latitude_threshold: float = -40.0) -> Dict:
        """
        Run a complete optimization workflow for Southern Ocean data.
        
        Parameters:
        -----------
        tracer_file : str
            Name of the tracer data file
        tracer_column : str
            Name of the column containing tracer values
        initial_values : np.ndarray
            Initial guess for end-member values
        bounds : tuple, optional
            Bounds for optimization (lower_bounds, upper_bounds)
        min_depth : float, optional
            Minimum depth threshold for data filtering
        gamma_range : tuple, optional
            Gamma density range for filtering
        latitude_threshold : float, optional
            Latitude threshold for Southern Ocean data
            
        Returns:
        --------
        Dict
            Complete results including datasets, optimization results, and statistics
        """
        print(f"Running complete optimization workflow for {tracer_column}...")
        print("=" * 60)
        
        # Load data if not already loaded
        if not hasattr(self, 'model_data') or not self.model_data:
            self.load_model_data()
            self.setup_interpolators()
        
        # Prepare tracer data
        tracer_data = self.prepare_tracer_data(
            tracer_file, tracer_column, min_depth, gamma_range, latitude_threshold
        )
        
        # Create optimization datasets and calculate exclusion indices
        datasets = self.create_optimization_datasets(tracer_data)
        self.calculate_exclusion_indices(datasets)

        results = self.optimize_tracer(datasets, tracer_column, initial_values, bounds)
        stats = self.calculate_statistics(tracer_column, datasets)
        
        # Store statistics (end_member_values are already stored by optimize_tracer)
        self.mean_end_members[tracer_column] = stats['mean']
        self.std_end_members[tracer_column] = stats['std']
        
        datasets_with_reconstruction = self.reconstruct_tracer(datasets, tracer_column)
        summary = self.get_optimization_summary(tracer_column)
        
        return {
            'tracer_data': tracer_data,
            'datasets': datasets_with_reconstruction,
            'optimization_results': results,
            'statistics': stats,
            'summary': summary
        }

