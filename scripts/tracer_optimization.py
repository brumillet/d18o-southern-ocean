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
from typing import List, Dict, Tuple, Optional
import usefull_functions as uf

dico_suffixes = {'TMI': '_TMI', 'OCIM': '_ocim', 'NEMO': '_nemo'}

class TracerOptimizer:
    """
    A class for optimizing tracer end-member values using water mass fractions.
    
    This class can handle optimization for different tracers (d18O, Salinity, etc.)
    and different circulation models (TMI, OCIM, NEMO).
    """
    
    def __init__(self, data_path: str, prepared_data: pd.DataFrame = None,
                 lat_name: str = 'Latitude', lon_name: str = 'Longitude', 
                 depth_name: str = 'Depth', so_compilation: bool = True, 
                 forbidden_values: List = ['**', -999.0]):
        """
        Initialize the TracerOptimizer.
        
        Parameters:
        -----------
        data_path : str
            Path to the directory containing the required data files
        prepared_data : pd.DataFrame, optional
            Pre-prepared tracer data
        lat_name : str, optional
            Name of the latitude coordinate in the prepared_data file
        lon_name : str, optional
            Name of the longitude coordinate in the prepared_data file
        depth_name : str, optional
            Name of the depth coordinate in the prepared_data file
        so_compilation : bool, optional
            Boolean to know if we work with the SO GISS compilation. By default, True.
        forbidden_values : List, optional
            List of values to consider as invalid in the tracer data
        """
        self.data_path = data_path
        self.prepared_data = prepared_data
        self.bool_prepData = (prepared_data is None)
        self.models = ['TMI', 'OCIM', 'NEMO']
        self.dyes = uf.dyes_TMI  # ['dyeLL', 'dyeMS', 'dyeNP', 'dyeHS', 'dyeNA', 'dyeAA']
        self.n_dyes = len(self.dyes)
        self.lat_name = lat_name
        self.lon_name = lon_name
        self.depth_name = depth_name
        self.so_compilation = so_compilation
        self.forbidden_values = forbidden_values
        
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
        self.model_data['NEMO'] = {
            'lat': nemo_dyes['lat'].values,
            'lon': nemo_dyes['lon'].values,
            'depth': nemo_dyes['depth'].values,
            'fractions': np.array([nemo_dyes[uf.dyes[i]].values for i in range(6)])
        }
        
        ocim_frac.close()
        TMI_2deg.close()
        nemo_dyes.close()        
    
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
        
        # Load tracer data - handle both CSV and table formats
        if tracer_file.endswith('.csv'):
            tracer_data = pd.read_csv(self.data_path + tracer_file)
        elif tracer_file.endswith('.txt'):
            tracer_data = pd.read_table(self.data_path + tracer_file)
        elif tracer_file.endswith('.h5'):
            tracer_data = pd.read_hdf(self.data_path + tracer_file, key='data')
        else:
            raise ValueError("Unsupported file format. Please provide a .csv, .txt, or .h5 file.")
        
        # Remove invalid data
        for val in self.forbidden_values:
            tracer_data = tracer_data.replace(val, np.nan)
        
        space_conditions = (tracer_data[self.lat_name] <= latitude_threshold) & (tracer_data[self.depth_name] >= min_depth) 
        tracer_data = tracer_data.where(space_conditions).dropna()
        
        # Convert to numeric
        tracer_data[tracer_column] = pd.to_numeric(tracer_data[tracer_column], errors='coerce')
        tracer_data['Salinity'] = pd.to_numeric(tracer_data['Salinity'], errors='coerce')
        
        # Apply gamma filter
        tracer_data = tracer_data[
            (tracer_data['gamma_n'] >= gamma_range[0]) & 
            (tracer_data['gamma_n'] <= gamma_range[1])
        ]       
        print(f"Tracer data prepared. Final dataset contains {len(tracer_data)} observations.")

        return tracer_data
    
    def add_fractions_datasets(self, tracer_data: pd.DataFrame):
        # Add water mass fractions for all models (recalculate coords after filtering)
        coords_filtered = (tracer_data[self.depth_name], tracer_data[self.lat_name], tracer_data[self.lon_name])
        for model in self.models:
            for i, dye in enumerate(self.dyes):
                tracer_data[f'{dye}_{model}'] = self.interpolators[model][i](coords_filtered)
    
    
    def optimize_tracer(self, datasets: List[pd.DataFrame], tracer_column: str,
                       initial_values: np.ndarray, bounds: Tuple[np.ndarray, np.ndarray]) -> Dict:
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
        bounds : tuple
            Bounds for optimization (lower_bounds, upper_bounds)
            
        Returns:
        --------
        Dict
            Dictionary containing optimization results for each model
        """

        print(f"Starting optimization for {tracer_column}...")
        
        results, end_member_values = {}, {}

        # Initialize storage for each model
        for model in self.models:
            results[model] = []
            end_member_values[model] = np.empty((self.n_dyes, len(datasets)))
        
        # Optimize for each dataset
        for i, dataset in enumerate(datasets):
            tracer_obs = dataset[tracer_column].values

            # Optimize for each model
            for model in self.models:
                fractions = np.array([dataset[f'{dye}_{model}'].values for dye in self.dyes])
                
                # Find valid data points (no NaN in fractions or tracer observations)
                valid_mask = ~np.isnan(fractions).any(axis=0)
                fractions_valid, tracer_obs_valid = fractions[:, valid_mask], tracer_obs[valid_mask]
                print(f"  Dataset {i+1}, Model {model}: {len(tracer_obs_valid)} valid data points")
                
                # Define objective function
                def objective(end_members):
                    reconstructed = end_members.dot(fractions_valid)
                    return tracer_obs_valid - reconstructed
                
                # Perform optimization
                try:
                    result = least_squares(objective, initial_values, bounds=bounds)
                    results[model].append(result)
                    end_member_values[model][:, i] = result.x
                    
                    if not result.success:
                        print(f"    {model}: Optimization failed - {result.message}")
                        
                except Exception as e:
                    print(f"    {model}: Error during optimization - {str(e)}")
                    end_member_values[model][:, i] = np.nan
                    results[model].append(None)
        
        # Store results
        self.optimization_results[tracer_column] = results
        self.end_member_values[tracer_column] = end_member_values
        
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
            
        print(f"Calculating statistics for {tracer_column}...")
        
        end_members = self.end_member_values[tracer_column]
        stats = {'mean': {}, 'std': {}}
        
        # Initialize nested structure
        for model in self.models:
            stats['mean'][model] = {}
            stats['std'][model] = {}
        
        for model in self.models:
            for idye, dye in enumerate(self.dyes):
                if len(datasets) == 1:
                    valid_indices = [0]
                else:
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

        print(f"Reconstructing {tracer_column} values...")
        
        end_members = self.end_member_values[tracer_column]
        for i, dataset in enumerate(datasets):
                
            for model in self.models:
                # Individual reconstruction
                reconstruction = np.zeros(len(dataset))
                model_results = end_members[model]

                # Mean reconstruction (if statistics available)
                if tracer_column not in self.mean_end_members:
                    print(f"  {model}: Warning - No mean end-member values available for reconstruction")
                
                elif use_mean_values:
                    mean_values = self.mean_end_members[tracer_column][model]
                    
                    for j, dye in enumerate(self.dyes):
                        reconstruction += dataset[f'{dye}_{model}']*mean_values[dye]
                    
                    dataset[f'{tracer_column}_rcst_mean_{model}'] = reconstruction

                else:
                    for j, dye in enumerate(self.dyes):
                        # Use the j-th dye value from i-th dataset optimization
                        reconstruction += dataset[f'{dye}_{model}']*model_results[j, i]
                    
                    dataset[f'{tracer_column}_rcst_{model}'] = reconstruction

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
        summary_data = []
        
        for model in self.models:
            for dye in self.dyes:
                row = {
                    'Model': model,
                    'Water_Mass': dye,
                    'Mean_End_Member': self.mean_end_members[tracer_column][model][dye],
                    'Std_End_Member': self.std_end_members[tracer_column][model][dye],
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
                        
        # Calculate exclusion indices for each model and dye
        for model in self.models:
            for dye in self.dyes:
                if dye in ['dyeLL', 'dyeNP']:
                    # For dyeLL and dyeNP, use empty exclusion list (no exclusion criteria)
                    self.exclusion_indices[model][dye] = []
                elif dye == 'dyeMS':
                    self.exclusion_indices[model][dye] = uf.index_to_exclude_dye(datasets, dye, '_' + model)
                else:
                    self.exclusion_indices[model][dye] = uf.index_to_exclude(datasets, '_' + model)

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
    

    
    def run_complete_optimization(self, tracer_column: str, initial_values: np.ndarray, 
                                tracer_file: str = None, 
                                bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                                min_depth: float = 150.0, 
                                gamma_range: Tuple[float, float] = (27.0, 28.6),
                                latitude_threshold: float = -40.0) -> Dict:
        """
        Run a complete optimization workflow for Southern Ocean data.
        
        Parameters:
        -----------
        tracer_column : str
            Name of the column containing tracer values
        initial_values : np.ndarray
            Initial guess for end-member values
        tracer_file : str, optional
            Name of the tracer data file (ignored if prepared_data provided)
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
        # Load data if not already loaded
        if not hasattr(self, 'model_data') or not self.model_data:
            self.load_model_data()
            self.setup_interpolators()
        
        # Use prepared data if available, otherwise prepare from file
        if not self.bool_prepData:
            print("Using pre-prepared tracer data...")
            tracer_data = self.prepared_data.copy()
            
            # Add water mass fractions to prepared data
            self.add_fractions_datasets(tracer_data)
            
            # Apply basic filtering if tracer_column is specified
            if tracer_column:
                tracer_data = tracer_data.dropna(subset=[tracer_column])
                print(f"Tracer data prepared. Final dataset contains {len(tracer_data)} observations.")
        else:
            print(f"Running complete optimization workflow for {tracer_column}...")
            print("=" * 60)
            
            # Prepare tracer data
            tracer_data = self.prepare_tracer_data(
                tracer_file, tracer_column, min_depth, gamma_range, latitude_threshold
            )

        # Convert longitude to 0-360 format
        tracer_data[self.lon_name] = [lon if lon >= 0 else 360 + lon for lon in tracer_data[self.lon_name]]
        self.add_fractions_datasets(tracer_data)
        
        # Create optimization datasets and calculate exclusion indices
        if self.so_compilation:
            datasets = uf.create_dfs(tracer_data)
            self.calculate_exclusion_indices(datasets)
        else: 
            datasets = [tracer_data]
            self.exclusion_indices = []

        results = self.optimize_tracer(datasets, tracer_column, initial_values, bounds)
        stats = self.calculate_statistics(tracer_column, datasets)
        
        # Store statistics (end_member_values are already stored by optimize_tracer)
        self.mean_end_members[tracer_column] = stats['mean']
        self.std_end_members[tracer_column] = stats['std']
        
        datasets_with_reconstruction = self.reconstruct_tracer(datasets, tracer_column, use_mean_values=False)
        datasets_with_reconstruction = self.reconstruct_tracer(datasets_with_reconstruction, tracer_column)
        summary = self.get_optimization_summary(tracer_column)
        
        return {
            'tracer_data': tracer_data,
            'datasets': datasets_with_reconstruction,
            'optimization_results': results,
            'statistics': stats,
            'summary': summary
        }

