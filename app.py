import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks, peak_widths
from scipy.optimize import curve_fit
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from datetime import datetime
import io
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import time

# ==================== STATE MANAGEMENT ====================

@dataclass
class AppState:
    """Centralized state management for the application"""
    deconvolver: Optional['GaussianDeconvolver'] = None
    raw_x: Optional[np.ndarray] = None
    raw_y: Optional[np.ndarray] = None
    original_x: Optional[np.ndarray] = None
    original_y: Optional[np.ndarray] = None
    peak_info: Optional[List[Dict]] = None
    derivatives: Optional[Tuple] = None
    current_step: int = 1
    use_log_x: bool = True
    use_log_y: bool = False
    sensitivity: float = 0.03
    min_distance: int = 5
    split_position: Optional[float] = None
    x_range_min: Optional[float] = None
    x_range_max: Optional[float] = None
    clip_negative: bool = True
    fitting_method: str = 'trf'
    max_nfev: int = 10000
    show_warnings: bool = True

# Initialize session state with dataclass
if 'app_state' not in st.session_state:
    st.session_state.app_state = AppState()

# ==================== PAGE CONFIGURATION ====================

st.set_page_config(
    page_title="Gaussian Deconvolution of Spectra",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Scientific plot style
plt.style.use('default')
plt.rcParams.update({
    # Font sizes and weights
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.labelweight': 'bold',
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    
    # Axes appearance
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.0,
    'axes.grid': False,
    
    # Tick parameters
    'xtick.color': 'black',
    'ytick.color': 'black',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 4,
    'xtick.minor.size': 2,
    'ytick.major.size': 4,
    'ytick.minor.size': 2,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    
    # Legend
    'legend.fontsize': 10,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': 'black',
    'legend.fancybox': False,
    
    # Figure
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'figure.facecolor': 'white',
    
    # Lines
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'errorbar.capsize': 3,
})

# Title
st.title("📊 Gaussian Deconvolution of Spectral Data")
st.markdown("---")

# ==================== CLASSES ====================

class DataParser:
    """Universal parser for spectral data"""
    
    @staticmethod
    def parse_text(text):
        """Parse text data in any format"""
        lines = text.strip().split('\n')
        x_data = []
        y_data = []
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith(('#', '//', ';')):
                continue
            
            # Split by any whitespace or commas
            parts = re.split(r'[,\s]+', line)
            parts = [p for p in parts if p.strip()]
            
            if len(parts) >= 2:
                try:
                    x = float(parts[0].replace(',', '.'))
                    y = float(parts[1].replace(',', '.'))
                    x_data.append(x)
                    y_data.append(y)
                except ValueError:
                    continue
        
        return np.array(x_data), np.array(y_data)
    
    @staticmethod
    def auto_detect_scale(x, y):
        """Automatically detect need for logarithmic scales"""
        if len(x) == 0 or len(y) == 0:
            return False, False
        
        x_pos = x[x > 0]
        y_pos = y[y > 0]
        
        suggest_log_x = False
        suggest_log_y = False
        
        if len(x_pos) > 1:
            x_log_range = np.log10(np.max(x_pos)) - np.log10(np.min(x_pos))
            suggest_log_x = bool(x_log_range > 2)
        
        if len(y_pos) > 1:
            y_log_range = np.log10(np.max(y_pos)) - np.log10(np.min(y_pos))
            suggest_log_y = bool(y_log_range > 2 and np.min(y_pos) > 0)
        
        return suggest_log_x, suggest_log_y

    @staticmethod
    def apply_range_selection(x, y, x_min, x_max):
        """Apply range selection to data"""
        if x_min is None or x_max is None:
            return x, y
        
        mask = (x >= x_min) & (x <= x_max)
        return x[mask], y[mask]


class DataPreprocessor:
    """Handles data preprocessing including clipping and log transformations"""
    
    def __init__(self, clip_negative=True, show_warnings=True):
        self.clip_negative = clip_negative
        self.show_warnings = show_warnings
        self.clipped_points = 0
        self.small_values_warning = False
    
    def preprocess_for_fitting(self, x_linear, y_original, use_log_x, use_log_y):
        """Preprocess data for fitting with proper handling of edge cases"""
        # Sort by X to ensure monotonic increasing X
        sort_idx = np.argsort(x_linear)
        x_sorted = x_linear[sort_idx]
        y_sorted = y_original[sort_idx]
        
        # Handle negative values
        if self.clip_negative:
            negative_mask = y_sorted < 0
            self.clipped_points = np.sum(negative_mask)
            if self.clipped_points > 0 and self.show_warnings:
                warnings.warn(f"Clipped {self.clipped_points} negative values to 0")
            y_for_fitting = np.maximum(y_sorted, 0)
        else:
            y_for_fitting = y_sorted
        
        # Small epsilon for log transformations
        eps = np.finfo(float).eps  # Use machine epsilon instead of 1e-12
        
        # Check for very small values when using log
        if use_log_y and np.any(y_for_fitting < eps * 100):
            self.small_values_warning = True
            if self.show_warnings:
                warnings.warn("Very small Y values detected. Log transformation may cause artifacts.")
        
        # Apply logarithmic transformations
        if use_log_x:
            x_pos = np.maximum(x_sorted, eps)
            x = np.log10(x_pos)
            x_label = 'log₁₀(X)'
        else:
            x = x_sorted
            x_label = 'X'
        
        if use_log_y:
            y_pos = np.maximum(y_for_fitting, eps)
            y = np.log10(y_pos)
            y_label = 'log₁₀(Y)'
        else:
            y = y_for_fitting
            y_label = 'Y'
        
        return {
            'x_sorted': x_sorted,
            'y_sorted': y_sorted,
            'x': x,
            'y': y,
            'y_for_fitting': y_for_fitting,
            'x_label': x_label,
            'y_label': y_label,
            'clipped_points': self.clipped_points,
            'small_values_warning': self.small_values_warning
        }


class DerivativeAnalyzer:
    """Analysis of first and second derivatives for peak detection"""
    
    @staticmethod
    def calculate_derivatives(x, y, window_length=11, polyorder=3):
        """Calculate smoothed derivatives with fallback for small datasets"""
        if len(x) < window_length:
            window_length = len(x) if len(x) % 2 == 1 else len(x) - 1
        
        if window_length < polyorder + 2:
            # Fallback to simple gradient
            dy = np.gradient(y, x)
            d2y = np.gradient(dy, x)
            return dy, d2y, y
        
        try:
            # Savitzky-Golay smoothing
            y_smooth = savgol_filter(y, window_length, polyorder)
            dy = savgol_filter(y, window_length, polyorder, deriv=1, delta=np.mean(np.diff(x)))
            d2y = savgol_filter(y, window_length, polyorder, deriv=2, delta=np.mean(np.diff(x)))
        except Exception as e:
            # Fallback to simple gradient if Savgol fails
            warnings.warn(f"Savitzky-Golay failed, using simple gradient: {e}")
            y_smooth = y
            dy = np.gradient(y, x)
            d2y = np.gradient(dy, x)
        
        return dy, d2y, y_smooth
    
    @staticmethod
    def find_peaks_by_derivatives(x, y, dy, d2y, threshold=0.01):
        """Find peaks by zero crossing of first derivative and negative second derivative"""
        peaks = []
        for i in range(1, len(x) - 1):
            if (dy[i-1] > 0 and dy[i] <= 0) or (dy[i-1] >= 0 and dy[i] < 0):
                if d2y[i] < 0:
                    if y[i] > threshold * np.max(y):
                        peaks.append(i)
        return peaks


class GaussianModel:
    """Model for sum of Gaussians"""
    
    @staticmethod
    def gaussian(x, amp, cen, sigma):
        """Gaussian function with safe sigma"""
        return amp * np.exp(-(x - cen)**2 / (2 * max(sigma, np.finfo(float).eps)**2))
    
    @staticmethod
    def multi_gaussian(x, *params):
        """Sum of multiple Gaussians"""
        n = len(params) // 3
        y = np.zeros_like(x, dtype=float)
        for i in range(n):
            amp = params[3*i]
            cen = params[3*i + 1]
            sigma = abs(params[3*i + 2])
            y += GaussianModel.gaussian(x, amp, cen, sigma)
        return y
    
    @staticmethod
    def calculate_area(amp, sigma):
        """Area under Gaussian"""
        return amp * sigma * np.sqrt(2 * np.pi)
    
    @staticmethod
    def calculate_fwhm(sigma):
        """Full width at half maximum"""
        return 2 * np.sqrt(2 * np.log(2)) * sigma
    
    @staticmethod
    def estimate_sigma_from_peak(x, y, peak_idx):
        """Estimate sigma with fallback methods"""
        try:
            widths, width_heights, left_ips, right_ips = peak_widths(
                y, [peak_idx], rel_height=0.5
            )
            fwhm = widths[0] * np.mean(np.diff(x))
            sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
            return sigma
        except Exception as e:
            # Fallback: estimate from distance to nearest minimum
            left_min = peak_idx
            right_min = peak_idx
            
            # Find left minimum
            for i in range(peak_idx - 1, 0, -1):
                if y[i] < y[i-1] and y[i] < y[i+1]:
                    left_min = i
                    break
            
            # Find right minimum
            for i in range(peak_idx + 1, len(y) - 1):
                if y[i] < y[i-1] and y[i] < y[i+1]:
                    right_min = i
                    break
            
            # Estimate sigma as 1/3 of the width to nearest minima
            width = (right_min - left_min) * np.mean(np.diff(x))
            sigma = width / 3.0
            return max(sigma, 0.01 * (np.max(x) - np.min(x)) / 10)


class FitQualityAnalyzer:
    """Fit quality analysis"""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred, n_params):
        """Calculate quality metrics"""
        residuals = y_true - y_pred
        n = len(y_true)
        
        # R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # AIC and BIC
        rss = ss_res
        aic = n * np.log(rss/n) + 2 * n_params if rss > 0 else -np.inf
        bic = n * np.log(rss/n) + n_params * np.log(n) if rss > 0 else -np.inf
        
        # Chi-squared (reduced)
        chi_squared = rss / (n - n_params) if n > n_params else np.inf
        
        # Maximum error
        max_error = np.max(np.abs(residuals))
        
        # Root mean square error
        rmse = np.sqrt(np.mean(residuals**2))
        
        return {
            'R²': r_squared,
            'AIC': aic,
            'BIC': bic,
            'χ²': chi_squared,
            'Max Error': max_error,
            'RMSE': rmse,
            'Residuals': residuals
        }
    
    @staticmethod
    def detect_autocorrelation(residuals):
        """Detect autocorrelation in residuals"""
        if len(residuals) < 10:
            return False
        
        diff = np.diff(residuals)
        dw = np.sum(diff**2) / np.sum(residuals**2)
        
        return dw < 1.5 or dw > 2.5


class GaussianFitter:
    """Handles Gaussian fitting with multiple optimization methods"""
    
    def __init__(self, method='trf', max_nfev=10000):
        self.method = method
        self.max_nfev = max_nfev
        self.convergence_history = []
        self.fit_progress = 0
    
    def fit(self, x, y_norm, initial_params, y_max, bounds=None, progress_callback=None):
        """Perform fitting with progress tracking"""
        if len(initial_params) == 0:
            return False, None, None
        
        n_peaks = len(initial_params) // 3
        
        # Set default bounds if not provided
        if bounds is None:
            bounds = self._create_default_bounds(x, y_norm, n_peaks)
        
        try:
            # Ensure initial_params are within bounds
            initial_params = np.array(initial_params)
            for i in range(len(initial_params)):
                initial_params[i] = np.clip(initial_params[i], bounds[0][i], bounds[1][i])
            
            # Progress simulation for curve_fit (which doesn't have built-in progress)
            if progress_callback:
                progress_callback(0.3, "Initializing fit...")
            
            # Perform fit with selected method
            popt, pcov = curve_fit(
                GaussianModel.multi_gaussian,
                x,
                y_norm,
                p0=initial_params,
                bounds=bounds,
                method=self.method,
                maxfev=self.max_nfev
            )
            
            if progress_callback:
                progress_callback(0.8, "Calculating components...")
            
            fit_y_norm = GaussianModel.multi_gaussian(x, *popt)
            
            # Extract components
            components = []
            for i in range(n_peaks):
                amp_norm = popt[3*i]
                cen = popt[3*i + 1]
                sigma = abs(popt[3*i + 2])
                
                amp = amp_norm * y_max
                area = GaussianModel.calculate_area(amp_norm, sigma) * y_max
                
                component_y_norm = GaussianModel.gaussian(x, amp_norm, cen, sigma)
                
                components.append({
                    'id': i + 1,
                    'amp_norm': amp_norm,
                    'amp': amp,
                    'cen_log': cen,
                    'sigma_log': sigma,
                    'fwhm': GaussianModel.calculate_fwhm(sigma),
                    'area': area,
                    'fraction': 0,
                    'y_norm': component_y_norm
                })
            
            # Calculate fractions
            total_area = sum([c['area'] for c in components])
            for c in components:
                c['fraction'] = c['area'] / total_area if total_area > 0 else 0
                c['fraction_percent'] = c['fraction'] * 100
            
            if progress_callback:
                progress_callback(1.0, "Fit complete!")
            
            return True, popt, components
            
        except Exception as e:
            if progress_callback:
                progress_callback(1.0, f"Fit failed: {e}")
            return False, None, None
    
    def _create_default_bounds(self, x, y_norm, n_peaks):
        """Create default bounds for fitting"""
        lower_bounds = []
        upper_bounds = []
        x_range = np.max(x) - np.min(x)
        y_range = np.max(y_norm) - np.min(y_norm)
        
        for i in range(n_peaks):
            lower_bounds.extend([0, np.min(x), x_range * 0.001])
            upper_bounds.extend([2 * np.max(y_norm), np.max(x), x_range * 0.5])
        
        return (lower_bounds, upper_bounds)


class SpectrumPlotter:
    """Unified plotting class for all visualizations"""
    
    def __init__(self, scientific_style=True):
        self.scientific_style = scientific_style
    
    def plot_raw_data(self, x, y, use_log_x=False, use_log_y=False, 
                      title="Raw Data", ax=None, figsize=(10, 6)):
        """Plot raw data with optional log scales"""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        # Apply scales
        if use_log_x:
            ax.set_xscale('log')
        if use_log_y:
            ax.set_yscale('log')
        
        # Plot data
        ax.plot(x, y, 'o-', markersize=3, linewidth=1, alpha=0.7, 
                color='black', label='Data', zorder=1)
        
        # Labels and title
        x_label = 'X' + (' (log scale)' if use_log_x else '')
        y_label = 'Y' + (' (log scale)' if use_log_y else '')
        ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Grid and styling
        ax.grid(True, alpha=0.3, linestyle='--')
        
        if self.scientific_style:
            self._apply_scientific_style(ax)
        
        return fig, ax
    
    def plot_with_peaks(self, deconvolver, peak_info, y_smooth, 
                        title="Peak Detection", ax=None, figsize=(10, 6)):
        """Plot data with detected peaks"""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        # Apply scales
        if deconvolver.use_log_x:
            ax.set_xscale('log')
        if deconvolver.use_log_y:
            ax.set_yscale('log')
        
        # Plot original data
        ax.plot(deconvolver.x_sorted, deconvolver.y_sorted, 
                'o-', markersize=3, linewidth=1, alpha=0.7, 
                label='Original Data', color='black', zorder=1)
        
        # Plot smoothed data (converted back to original scale)
        if deconvolver.use_log_y:
            y_smooth_original = 10**(y_smooth * deconvolver.y_max)
        else:
            y_smooth_original = y_smooth * deconvolver.y_max
        
        ax.plot(deconvolver.x_sorted, y_smooth_original, 
                'r-', linewidth=2, label='Smoothed', color='red', zorder=2)
        
        # Mark detected peaks
        for i, info in enumerate(peak_info):
            peak_y_original = info['y_original']
            ax.plot(info['x_linear'], peak_y_original, 
                    'ro', markersize=8, markeredgecolor='darkred', 
                    markerfacecolor='yellow', zorder=3)
            ax.text(info['x_linear'], peak_y_original * 1.05, 
                    f'{i+1}', ha='center', fontweight='bold', 
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", 
                                          facecolor='white', alpha=0.8),
                    zorder=4)
        
        # Labels and title
        x_label = 'X' + (' (log scale)' if deconvolver.use_log_x else '')
        y_label = 'Y' + (' (log scale)' if deconvolver.use_log_y else '')
        ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Legend and grid
        ax.legend(loc='best', fontsize=10, frameon=True, edgecolor='black')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        if self.scientific_style:
            self._apply_scientific_style(ax)
        
        return fig, ax
    
    def plot_deconvolution_result(self, deconvolver, show_components=True,
                                  title="Deconvolution Result", ax=None, figsize=(10, 6)):
        """Plot deconvolution result with components"""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        # Apply scales
        if deconvolver.use_log_x:
            ax.set_xscale('log')
        if deconvolver.use_log_y:
            ax.set_yscale('log')
        
        # Plot original data
        ax.scatter(deconvolver.x_linear, deconvolver.y_original, 
                   s=10, alpha=0.5, color='black', label='Data', zorder=1)
        
        # Generate dense x for smooth curves
        if deconvolver.use_log_x:
            x_min = np.maximum(np.min(deconvolver.x_linear[deconvolver.x_linear>0]), np.finfo(float).eps)
            x_max = np.max(deconvolver.x_linear)
            x_dense = np.logspace(np.log10(x_min), np.log10(x_max), 2000)
            x_dense_log = np.log10(x_dense)
        else:
            x_dense = np.linspace(np.min(deconvolver.x_linear), 
                                  np.max(deconvolver.x_linear), 2000)
            x_dense_log = x_dense
        
        # Plot components
        if show_components and deconvolver.components:
            colors = plt.cm.Set3(np.linspace(0, 1, len(deconvolver.components)))
            for c, color in zip(deconvolver.components, colors):
                y_component = GaussianModel.gaussian(x_dense_log, c['amp_norm'], 
                                                    c['cen_log'], c['sigma_log']) * deconvolver.y_max
                
                # Fill under Gaussian
                ax.fill_between(x_dense, 0, y_component, 
                                color=color, alpha=0.3, linewidth=0)
                
                # Plot line
                ax.plot(x_dense, y_component, '-', color=color, linewidth=2,
                       label=f'Peak {c["id"]}: {c["fraction_percent"]:.1f}%', zorder=2)
        
        # Plot total fit
        if deconvolver.fit_y_norm is not None:
            y_total = GaussianModel.multi_gaussian(x_dense_log, *deconvolver.popt) * deconvolver.y_max
            ax.plot(x_dense, y_total, 'r--', linewidth=2, label='Total Fit', zorder=3)
        
        # Labels and title
        x_label = 'X' + (' (log scale)' if deconvolver.use_log_x else '')
        y_label = 'Y' + (' (log scale)' if deconvolver.use_log_y else '')
        ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add quality metrics to plot if available
        if deconvolver.quality_metrics:
            metrics_text = f"R² = {deconvolver.quality_metrics.get('R²', 0):.4f}\n"
            metrics_text += f"RMSE = {deconvolver.quality_metrics.get('RMSE', 0):.2e}"
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Legend and grid
        ax.legend(loc='upper right', fontsize=8, frameon=True, edgecolor='black')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        if self.scientific_style:
            self._apply_scientific_style(ax)
        
        return fig, ax
    
    def _apply_scientific_style(self, ax):
        """Apply scientific styling to axes"""
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_linewidth(1)
        ax.spines['left'].set_linewidth(1)
        ax.spines['top'].set_linewidth(1)
        ax.spines['right'].set_linewidth(1)
        ax.tick_params(direction='out', length=4, width=1)


class PeakVisualizer:
    """Helper class for peak visualization with original scales"""
    
    @staticmethod
    def plot_peaks_original_scale(deconvolver, peak_info, y_smooth):
        """Plot peaks using original Y scale instead of normalized"""
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Use original Y values, not normalized
        ax.plot(deconvolver.x, deconvolver.y, 
               'o-', markersize=3, alpha=0.5, label='Data', color='black')
        ax.plot(deconvolver.x, y_smooth * deconvolver.y_max, 
               'r-', linewidth=2, label='Smoothed')
        
        for i, info in enumerate(peak_info):
            # Use original Y value
            ax.plot(info['x'], info['y'] * deconvolver.y_max, 'ro', 
                   markersize=8, markeredgecolor='darkred')
            ax.text(info['x'], info['y'] * deconvolver.y_max * 1.05, 
                   f'{i+1}', ha='center', fontweight='bold')
        
        ax.set_xlabel(deconvolver.x_label)
        ax.set_ylabel(deconvolver.y_label)  # Original Y label
        ax.set_title('Detected Peaks (Original Scale)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Scientific styling
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_linewidth(1)
        ax.spines['left'].set_linewidth(1)
        ax.spines['top'].set_linewidth(1)
        ax.spines['right'].set_linewidth(1)
        ax.tick_params(direction='out', length=4, width=1)
        
        return fig


class GaussianDeconvolver:
    """Main class for spectral deconvolution"""
    
    def __init__(self, x_linear, y_original, use_log_x=True, use_log_y=False,
                 clip_negative=True, show_warnings=True):
        # Store original data WITHOUT ANY MODIFICATIONS for display purposes
        self.x_original = np.array(x_linear).copy()
        self.y_original_raw = np.array(y_original).copy()
        
        # Working arrays that may be modified
        self.x_linear = np.array(x_linear)
        self.y_original = np.array(y_original)
        self.use_log_x = use_log_x
        self.use_log_y = use_log_y
        
        # Sort by X to ensure monotonic increasing X
        sort_idx = np.argsort(self.x_linear)
        self.x_linear = self.x_linear[sort_idx]
        self.y_original = self.y_original[sort_idx]
        
        # Store sorted original data for display
        self.x_sorted = self.x_linear.copy()
        self.y_sorted = self.y_original.copy()
        
        # Preprocess data
        self.preprocessor = DataPreprocessor(clip_negative, show_warnings)
        preprocessed = self.preprocessor.preprocess_for_fitting(
            self.x_linear, self.y_original, use_log_x, use_log_y
        )
        
        # Update with preprocessed data
        self.x_sorted = preprocessed['x_sorted']
        self.y_sorted = preprocessed['y_sorted']
        self.x = preprocessed['x']
        self.y = preprocessed['y']
        self.y_for_fitting = preprocessed['y_for_fitting']
        self.x_label = preprocessed['x_label']
        self.y_label = preprocessed['y_label']
        self.clipped_points = preprocessed['clipped_points']
        self.small_values_warning = preprocessed['small_values_warning']
        
        # Store the maximum of y for denormalization
        self.y_max = np.max(self.y_for_fitting) if np.max(self.y_for_fitting) > 0 else 1.0
        
        # For fitting, we normalize but keep track for denormalization
        if self.y_max > 0:
            self.y_norm = self.y / self.y_max
        else:
            self.y_norm = self.y
        
        # Results containers
        self.components = []
        self.fit_y_norm = None
        self.popt = None
        self.quality_metrics = {}
        self.convergence_history = []
        self.total_area = 0
        
        # Fitter
        self.fitter = None
        
        # For compatibility with existing code
        self.multi_gaussian = GaussianModel.multi_gaussian
        self.gaussian = GaussianModel.gaussian
    
    def auto_detect_peaks(self, sensitivity=0.03, min_distance=5):
        """Automatic peak detection using derivatives"""
        # Smoothing
        window_length = min(11, len(self.y_norm) // 5 * 2 + 1)
        if window_length % 2 == 0:
            window_length += 1
        
        if window_length >= 5:
            y_smooth = savgol_filter(self.y_norm, window_length, 3)
        else:
            y_smooth = self.y_norm
        
        # Calculate derivatives
        dy, d2y, y_smooth = DerivativeAnalyzer.calculate_derivatives(self.x, y_smooth)
        
        # Peak search with different methods
        height_threshold = sensitivity * np.max(y_smooth)
        peaks1, _ = find_peaks(y_smooth, height=height_threshold, distance=min_distance)
        peaks2 = DerivativeAnalyzer.find_peaks_by_derivatives(self.x, y_smooth, dy, d2y, sensitivity)
        
        # Combine results
        all_peaks = sorted(set(np.concatenate([peaks1, peaks2])))
        
        # Filter close peaks
        filtered_peaks = []
        for peak in all_peaks:
            if not filtered_peaks or abs(self.x[peak] - self.x[filtered_peaks[-1]]) > min_distance * np.mean(np.diff(self.x)):
                filtered_peaks.append(peak)
        
        # Estimate parameters
        peak_info = []
        initial_params = []
        
        for peak_idx in filtered_peaks:
            cen = self.x[peak_idx]
            amp = y_smooth[peak_idx]
            
            # Estimate sigma with fallback
            sigma = GaussianModel.estimate_sigma_from_peak(self.x, y_smooth, peak_idx)
            sigma = max(sigma, 0.01 * (np.max(self.x) - np.min(self.x)) / max(len(filtered_peaks), 1))
            
            # Get original Y value for display
            if self.use_log_x:
                x_linear = 10**self.x[peak_idx]
            else:
                x_linear = self.x[peak_idx]
            
            # Find closest index in original data - always in linear space
            idx = np.argmin(np.abs(self.x_sorted - x_linear))
            y_original_value = self.y_sorted[idx]
            
            peak_info.append({
                'index': peak_idx,
                'x': self.x[peak_idx],
                'x_linear': x_linear,
                'y': self.y[peak_idx],  # This is in transformed scale (log if use_log_y)
                'y_original': y_original_value,  # This is in original scale
                'amp_est': amp,
                'cen_est': cen,
                'sigma_est': sigma,
                'dy': dy[peak_idx],
                'd2y': d2y[peak_idx]
            })
            
            initial_params.extend([amp, cen, sigma])
        
        return filtered_peaks, peak_info, initial_params, (dy, d2y, y_smooth)
    
    def fit(self, initial_params=None, method='trf', maxfev=10000, progress_callback=None):
        """Perform fitting with selected method"""
        if initial_params is None:
            _, _, initial_params, _ = self.auto_detect_peaks()
        
        if len(initial_params) == 0:
            return False
        
        # Create fitter with selected method
        self.fitter = GaussianFitter(method=method, max_nfev=maxfev)
        
        # Perform fit
        success, popt, components = self.fitter.fit(
            self.x, self.y_norm, initial_params, self.y_max,
            progress_callback=progress_callback
        )
        
        if success:
            self.popt = popt
            self.components = components
            self.fit_y_norm = GaussianModel.multi_gaussian(self.x, *popt)
            
            # Calculate total area
            self.total_area = sum([c['area'] for c in self.components])
            
            # Quality metrics
            self.quality_metrics = FitQualityAnalyzer.calculate_metrics(
                self.y_norm, self.fit_y_norm, len(popt)
            )
            
            return True
        
        return False
    
    def remove_peak(self, peak_id):
        """Remove a peak"""
        if peak_id > len(self.components):
            return False
        
        new_params = []
        for i, c in enumerate(self.components):
            if i != peak_id - 1:
                new_params.extend([c['amp_norm'], c['cen_log'], c['sigma_log']])
        
        if len(new_params) == 0:
            return False
        
        return self.fit(initial_params=new_params, method=self.fitter.method if self.fitter else 'trf')
    
    def split_peak(self, peak_id, split_position):
        """Split a peak into two at specified position"""
        if peak_id > len(self.components):
            return False
        
        peak = self.components[peak_id - 1]
        
        new_params = []
        for i, c in enumerate(self.components):
            if i == peak_id - 1:
                amp1 = c['amp_norm'] * 0.6
                amp2 = c['amp_norm'] * 0.4
                
                # Split at specified position
                cen1 = split_position - c['sigma_log'] * 0.3
                cen2 = split_position + c['sigma_log'] * 0.3
                
                # Ensure centers are within range
                cen1 = np.clip(cen1, np.min(self.x), np.max(self.x))
                cen2 = np.clip(cen2, np.min(self.x), np.max(self.x))
                
                sigma1 = c['sigma_log'] * 0.7
                sigma2 = c['sigma_log'] * 0.7
                
                new_params.extend([amp1, cen1, sigma1])
                new_params.extend([amp2, cen2, sigma2])
            else:
                new_params.extend([c['amp_norm'], c['cen_log'], c['sigma_log']])
        
        return self.fit(initial_params=new_params, method=self.fitter.method if self.fitter else 'trf')
    
    def create_scientific_plotly_figure(self):
        """Create a Plotly figure with scientific styling"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Deconvolution', 'Residuals', 'Components', 'Metrics'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'bar'}, {'type': 'table'}]]
        )
        
        # Scientific styling for Plotly
        plotly_template = {
            'layout': {
                'font': {'family': 'serif', 'size': 12},
                'title': {'font': {'family': 'serif', 'size': 14, 'weight': 'bold'}},
                'xaxis': {
                    'title': {'font': {'family': 'serif', 'size': 13, 'weight': 'bold'}},
                    'tickfont': {'family': 'serif', 'size': 11},
                    'showline': True,
                    'linewidth': 1,
                    'linecolor': 'black',
                    'mirror': True,
                    'ticks': 'outside',
                    'tickwidth': 1,
                    'ticklen': 5,
                    'showgrid': True,
                    'gridwidth': 0.5,
                    'gridcolor': 'lightgray'
                },
                'yaxis': {
                    'title': {'font': {'family': 'serif', 'size': 13, 'weight': 'bold'}},
                    'tickfont': {'family': 'serif', 'size': 11},
                    'showline': True,
                    'linewidth': 1,
                    'linecolor': 'black',
                    'mirror': True,
                    'ticks': 'outside',
                    'tickwidth': 1,
                    'ticklen': 5,
                    'showgrid': True,
                    'gridwidth': 0.5,
                    'gridcolor': 'lightgray'
                },
                'legend': {
                    'font': {'family': 'serif', 'size': 11},
                    'borderwidth': 1,
                    'bordercolor': 'black'
                },
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white'
            }
        }
        
        # Main plot
        fig.add_trace(
            go.Scatter(x=self.x, y=self.y_norm,
                      mode='markers+lines',
                      name='Data',
                      marker=dict(size=4, color='black', symbol='circle'),
                      line=dict(color='black', width=1)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=self.x, y=self.fit_y_norm,
                      mode='lines',
                      name='Total Fit',
                      line=dict(color='red', width=2, dash='solid')),
            row=1, col=1
        )
        
        # Components
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.components)))
        for c, color in zip(self.components, colors):
            rgb_color = f'rgb({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})'
            rgba_color = f'rgba({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}, 0.2)'
            
            fig.add_trace(
                go.Scatter(x=self.x, y=c['y_norm'],
                          mode='lines',
                          name=f'Peak {c["id"]} ({c["fraction_percent"]:.1f}%)',
                          line=dict(color=rgb_color, width=1.5),
                          fill='tozeroy',
                          fillcolor=rgba_color),
                row=1, col=1
            )
        
        # Residuals
        if 'Residuals' in self.quality_metrics:
            residuals = self.quality_metrics['Residuals']
            fig.add_trace(
                go.Scatter(x=self.x, y=residuals,
                          mode='lines+markers',
                          name='Residuals',
                          marker=dict(size=3, color='blue', symbol='circle'),
                          line=dict(color='blue', width=1)),
                row=1, col=2
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red", 
                         line_width=1, row=1, col=2)
        
        # Bar chart of components
        centers = [f"Peak {c['id']}" for c in self.components]
        fractions = [c['fraction_percent'] for c in self.components]
        
        fig.add_trace(
            go.Bar(x=centers, y=fractions,
                  name='Fractions (%)',
                  marker_color='steelblue',
                  marker_line_color='black',
                  marker_line_width=1,
                  opacity=0.8),
            row=2, col=1
        )
        
        # Metrics table
        metrics = self.quality_metrics
        metrics_table = go.Table(
            header=dict(
                values=['Metric', 'Value'],
                fill_color='lightgray',
                align='center',
                font=dict(family='serif', size=12, color='black'),
                line=dict(color='black', width=1)
            ),
            cells=dict(
                values=[
                    ['R²', 'AIC', 'BIC', 'χ²', 'RMSE'],
                    [f"{metrics.get('R²', 0):.6f}", 
                     f"{metrics.get('AIC', 0):.2f}",
                     f"{metrics.get('BIC', 0):.2f}",
                     f"{metrics.get('χ²', 0):.2e}",
                     f"{metrics.get('RMSE', 0):.2e}"]
                ],
                fill_color='white',
                align='center',
                font=dict(family='serif', size=11),
                line=dict(color='black', width=1)
            )
        )
        fig.add_trace(metrics_table, row=2, col=2)
        
        # Update layout with scientific styling
        fig.update_layout(
            height=700,
            showlegend=True,
            title_text="",
            font=dict(family='serif', size=12),
            legend=dict(
                bgcolor='white',
                bordercolor='black',
                borderwidth=1,
                font=dict(family='serif', size=10)
            )
        )
        
        # Update axes
        fig.update_xaxes(
            title_text=self.x_label,
            title_font=dict(family='serif', size=13, weight='bold'),
            tickfont=dict(family='serif', size=11),
            showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True,
            ticks='outside',
            tickwidth=1,
            ticklen=5,
            showgrid=True,
            gridwidth=0.5,
            gridcolor='lightgray',
            row=1, col=1
        )
        
        fig.update_xaxes(
            title_text=self.x_label,
            title_font=dict(family='serif', size=13, weight='bold'),
            tickfont=dict(family='serif', size=11),
            showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True,
            ticks='outside',
            tickwidth=1,
            ticklen=5,
            showgrid=True,
            gridwidth=0.5,
            gridcolor='lightgray',
            row=1, col=2
        )
        
        fig.update_yaxes(
            title_text="Normalized Y",
            title_font=dict(family='serif', size=13, weight='bold'),
            tickfont=dict(family='serif', size=11),
            showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True,
            ticks='outside',
            tickwidth=1,
            ticklen=5,
            showgrid=True,
            gridwidth=0.5,
            gridcolor='lightgray',
            row=1, col=1
        )
        
        fig.update_yaxes(
            title_text="Residuals",
            title_font=dict(family='serif', size=13, weight='bold'),
            tickfont=dict(family='serif', size=11),
            showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True,
            ticks='outside',
            tickwidth=1,
            ticklen=5,
            showgrid=True,
            gridwidth=0.5,
            gridcolor='lightgray',
            row=1, col=2
        )
        
        fig.update_yaxes(
            title_text="Fraction (%)",
            title_font=dict(family='serif', size=13, weight='bold'),
            tickfont=dict(family='serif', size=11),
            showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True,
            ticks='outside',
            tickwidth=1,
            ticklen=5,
            showgrid=True,
            gridwidth=0.5,
            gridcolor='lightgray',
            row=2, col=1
        )
        
        return fig


# ==================== DEFAULT DATA ====================

DEFAULT_DATA = """
0.0000013178292888320056, 0
0.0000016541179235844393, 0
0.000002110204394922548, 0
0.000002736106105123297, 0.0008038829955255381
0.000003490526912068043, 0.006430887335146873
0.000003910611263354757, 0.02331190035401096
0.000004241282522858312, 0.03536979202877925
0.000004599914496045258, 0.048231522541808715
0.000004988871469158237, 0.05787780938726475
0.000005589286330833996, 0.06350485788415045
0.0000059642820221386805, 0.056270131710742306
0.000006468612182886291, 0.04742768370354754
0.000006791450983591507, 0.0401929575301394
0.0000072471086165273385, 0.03376207019499253
0.000007608801014864345, 0.026527344021584475
0.000008252181715593673, 0.02170417852022424
0.000009245330455031805, 0.01688105717612836
0.000010358013722304093, 0.014469452346816188
0.00001141772015716314, 0.011254052836507027
0.000013213995372176845, 0.01205789167476829
0.000015797559733924332, 0.02250806151574978
0.000018886255558434893, 0.03054662652741909
0.000021159213370927343, 0.04180063520666184
0.00002409372325564147, 0.050643083213856695
0.000025710216287767918, 0.05948553122105147
0.00002880446236155555, 0.06672025739445961
0.00003333607997957873, 0.07475886656339312
0.00003858062735057595, 0.07877814906922781
0.00004465026504843482, 0.08199359273680117
0.00005425381140485725, 0.08199359273680117
0.00006592292450584217, 0.07877814906922781
0.00007629414693563464, 0.07154342289581976
0.0000854761352227386, 0.06752414038998505
0.00010218819622484528, 0.0627009748886249
0.00011636038054314034, 0.05948553122105147
0.00014138760731065896, 0.06350485788415045
0.00016903132695428032, 0.07395498356786766
0.00018632472999186478, 0.09003215774847056
0.00020874888121576402, 0.11254021926422034
0.00024956267099068337, 0.14951768896952194
0.0002841737934383662, 0.18569131983656245
0.00031837434759855607, 0.21945339003155503
0.00035669033092745645, 0.25321543814791536
0.00039318326411159053, 0.28938906901495587
0.0004264298325899839, 0.31270096936896685
0.00048557014829716627, 0.3440514568132793
0.000526628714772274, 0.36093249191077553
0.0006194548360801655, 0.38022508768031976
0.0007169096128267397, 0.3882636747706212
0.0008711059961790559, 0.37861736584653305
0.0010246513472112625, 0.3569131873263088
0.0011479669479618167, 0.32395500012684175
0.0013071751518333114, 0.28938906901495587
0.0014409094716578506, 0.26527332982268365
0.0015375841588074382, 0.2500000165589741
0.00169489415884709, 0.21945339003155503
0.0018382100622924278, 0.18327975916451456
0.002059436510444037, 0.15192929379883421
0.002422447059114667, 0.12057878427588965
0.002803554806727549, 0.07958198790748908
0.0032446197429791056, 0.051446966209382154
0.0038789954930106706, 0.02893890469363246
0.004637401984425402, 0.011254052836507027
0.0058208020475350435, 0.0008038829955255381
0.008185466037560749, 0
0.011699171907973411, 0.0016077218337868003
0.013986574587504476, 0.018488779009915163
0.017555721299263825, 0.049035361380069975
0.01966853198801187, 0.07315114472960639
0.02133165261214741, 0.09967848875119079
0.023898934878611392, 0.126205788615511
0.026775191717066393, 0.14308684579163944
0.03149471635492293, 0.1543408544708822
0.03889507892863105, 0.1543408544708822
0.041504659655973816, 0.13987140212406599
0.04357605654836328, 0.12299034494793763
0.04882047146797147, 0.10530549309081229
0.049619493345202464, 0.08762059707642267
0.0520959748208861, 0.06913186222377178
0.05469595856354164, 0.049839244375595435
0.06029189371578275, 0.03376207019499253
0.06330091589098744, 0.02250806151574978
0.07091923416067997, 0.01205789167476829
0.08207649330147837, 0.0016077218337868003
0.10642087310673698, 0
0.14724407114193477, 0
0.20706141239481027, 0
0.34250338264599406, 0
"""


# ==================== SIDEBAR ====================

with st.sidebar:
    st.header("📋 Navigation")
    
    # Step indicator
    steps = {
        1: "1. Data Loading",
        2: "2. Scale Settings",
        3: "3. Peak Detection",
        4: "4. Editing",
        5: "5. Results"
    }
    
    for step_num, step_name in steps.items():
        if step_num < st.session_state.app_state.current_step:
            st.success(f"✅ {step_name}")
        elif step_num == st.session_state.app_state.current_step:
            st.info(f"▶️ {step_name}")
        else:
            st.write(f"⏳ {step_name}")
    
    st.markdown("---")
    
    # Advanced settings (collapsible)
    with st.expander("⚙️ Advanced Settings"):
        st.session_state.app_state.clip_negative = st.checkbox(
            "Clip negative values to 0", 
            value=st.session_state.app_state.clip_negative
        )
        
        st.session_state.app_state.show_warnings = st.checkbox(
            "Show warnings", 
            value=st.session_state.app_state.show_warnings
        )
        
        st.session_state.app_state.fitting_method = st.selectbox(
            "Fitting method",
            options=['trf', 'dogbox', 'lm'],
            index=0,
            help="trf: most robust, dogbox: good for bounds, lm: fast but sensitive"
        )
        
        st.session_state.app_state.max_nfev = st.number_input(
            "Max iterations",
            min_value=1000,
            max_value=100000,
            value=st.session_state.app_state.max_nfev,
            step=1000
        )
    
    # Reset button
    if st.button("🔄 Start Over", use_container_width=True):
        st.session_state.app_state = AppState()
        st.rerun()


# ==================== STEP 1: DATA LOADING ====================

if st.session_state.app_state.current_step == 1:
    st.header("Step 1: Data Loading")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Text area for data input
        data_text = st.text_area(
            "Paste your data (x y separated by space, comma, or tab):",
            height=300,
            value=DEFAULT_DATA
        )
    
    with col2:
        st.subheader("Data Format:")
        st.info(
            """
            Any separators are supported:
            - Space
            - Comma
            - Tab
            
            Examples:
            ```
            1.23, 4.56
            1.23 4.56
            1.23\t4.56
            ```
            """
        )
        
        if st.button("📂 Load Data", type="primary", use_container_width=True):
            x, y = DataParser.parse_text(data_text)
            
            if len(x) > 0:
                st.session_state.app_state.raw_x = x
                st.session_state.app_state.raw_y = y
                st.session_state.app_state.x_range_min = float(np.min(x))
                st.session_state.app_state.x_range_max = float(np.max(x))
                st.session_state.app_state.current_step = 2
                st.rerun()
            else:
                st.error("Could not parse data. Check the format.")
    
    # Preview
    if st.session_state.app_state.raw_x is not None:
        st.subheader("Data Preview:")
        
        plotter = SpectrumPlotter()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        plotter.plot_raw_data(st.session_state.app_state.raw_x, 
                             st.session_state.app_state.raw_y,
                             use_log_x=False, use_log_y=False,
                             title="Linear Scales", ax=ax1)
        
        if np.min(st.session_state.app_state.raw_x[st.session_state.app_state.raw_x > 0]) > 0:
            plotter.plot_raw_data(st.session_state.app_state.raw_x, 
                                 np.maximum(st.session_state.app_state.raw_y, 1e-12),
                                 use_log_x=True, use_log_y=True,
                                 title="Log-Log Scales", ax=ax2)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ==================== STEP 2: SCALE SETTINGS ====================

elif st.session_state.app_state.current_step == 2:
    st.header("Step 2: Scale Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Scale Parameters")
        
        # Auto-detection
        if st.button("🔍 Auto-detect Scales", use_container_width=True):
            suggest_log_x, suggest_log_y = DataParser.auto_detect_scale(
                st.session_state.app_state.raw_x, 
                st.session_state.app_state.raw_y
            )
            st.session_state.app_state.use_log_x = suggest_log_x
            st.session_state.app_state.use_log_y = suggest_log_y
            st.rerun()
        
        # Manual settings
        st.session_state.app_state.use_log_x = st.checkbox(
            "Logarithmic X scale", 
            value=st.session_state.app_state.use_log_x
        )
        st.session_state.app_state.use_log_y = st.checkbox(
            "Logarithmic Y scale", 
            value=st.session_state.app_state.use_log_y
        )
        
        st.markdown("---")
        st.subheader("Range Selection")
        
        # Get current range
        x_min = float(np.min(st.session_state.app_state.raw_x))
        x_max = float(np.max(st.session_state.app_state.raw_x))
        
        # Initialize range if not set
        if st.session_state.app_state.x_range_min is None:
            st.session_state.app_state.x_range_min = x_min
        if st.session_state.app_state.x_range_max is None:
            st.session_state.app_state.x_range_max = x_max
        
        # Range slider
        range_values = st.slider(
            "Select X-axis range:",
            min_value=x_min,
            max_value=x_max,
            value=(st.session_state.app_state.x_range_min, 
                   st.session_state.app_state.x_range_max),
            format="%.2e",
            help="Drag the handles to select the region of interest"
        )
        
        st.session_state.app_state.x_range_min = range_values[0]
        st.session_state.app_state.x_range_max = range_values[1]
        
        # Show selected range statistics
        mask = ((st.session_state.app_state.raw_x >= range_values[0]) & 
                (st.session_state.app_state.raw_x <= range_values[1]))
        points_in_range = np.sum(mask)
        st.info(f"Points in selected range: {points_in_range} / {len(st.session_state.app_state.raw_x)}")
        
        st.markdown("---")
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("⬅️ Back", use_container_width=True):
                st.session_state.app_state.current_step = 1
                st.rerun()
        with col_b:
            if st.button("✅ Apply & Continue", type="primary", use_container_width=True):
                # Apply range selection to data
                x_range, y_range = DataParser.apply_range_selection(
                    st.session_state.app_state.raw_x,
                    st.session_state.app_state.raw_y,
                    st.session_state.app_state.x_range_min,
                    st.session_state.app_state.x_range_max
                )
                
                # Update raw data with selected range
                st.session_state.app_state.raw_x = x_range
                st.session_state.app_state.raw_y = y_range
                
                st.session_state.app_state.current_step = 3
                st.rerun()
    
    with col2:
        st.subheader("Preview:")
        
        plotter = SpectrumPlotter()
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Apply range selection to preview
        x_preview, y_preview = DataParser.apply_range_selection(
            st.session_state.app_state.raw_x,
            st.session_state.app_state.raw_y,
            st.session_state.app_state.x_range_min,
            st.session_state.app_state.x_range_max
        )
        
        plotter.plot_raw_data(
            x_preview, y_preview,
            use_log_x=st.session_state.app_state.use_log_x,
            use_log_y=st.session_state.app_state.use_log_y,
            title="Selected Range Preview",
            ax=ax
        )
        
        # Highlight selected range on full data plot
        if len(x_preview) < len(st.session_state.app_state.raw_x):
            ax.axvspan(st.session_state.app_state.x_range_min,
                      st.session_state.app_state.x_range_max,
                      alpha=0.2, color='green', label='Selected Range')
        
        st.pyplot(fig)
        plt.close()


# ==================== STEP 3: PEAK DETECTION ====================

elif st.session_state.app_state.current_step == 3:
    st.header("Step 3: Peak Detection")
    
    # Create deconvolver if not yet created
    if st.session_state.app_state.deconvolver is None:
        # Store original data separately
        st.session_state.app_state.original_x = st.session_state.app_state.raw_x.copy()
        st.session_state.app_state.original_y = st.session_state.app_state.raw_y.copy()
        
        st.session_state.app_state.deconvolver = GaussianDeconvolver(
            st.session_state.app_state.raw_x,
            st.session_state.app_state.raw_y,
            use_log_x=st.session_state.app_state.use_log_x,
            use_log_y=st.session_state.app_state.use_log_y,
            clip_negative=st.session_state.app_state.clip_negative,
            show_warnings=st.session_state.app_state.show_warnings
        )
        
        # Show warnings if any
        if st.session_state.app_state.deconvolver.clipped_points > 0:
            st.warning(f"Clipped {st.session_state.app_state.deconvolver.clipped_points} negative values to 0")
        if st.session_state.app_state.deconvolver.small_values_warning:
            st.warning("Very small Y values detected. Log transformation may cause artifacts.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Search Parameters")
        
        st.session_state.app_state.sensitivity = st.slider(
            "Sensitivity:",
            min_value=0.001,
            max_value=0.1,
            value=st.session_state.app_state.sensitivity,
            step=0.001,
            format="%.3f"
        )
        
        st.session_state.app_state.min_distance = st.slider(
            "Minimum distance between peaks:",
            min_value=1,
            max_value=20,
            value=st.session_state.app_state.min_distance,
            step=1
        )
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("⬅️ Back", use_container_width=True):
                st.session_state.app_state.current_step = 2
                st.rerun()
        with col_b:
            if st.button("🔍 Find Peaks", type="primary", use_container_width=True):
                with st.spinner("Detecting peaks..."):
                    peaks, peak_info, initial_params, derivatives = st.session_state.app_state.deconvolver.auto_detect_peaks(
                        sensitivity=st.session_state.app_state.sensitivity,
                        min_distance=st.session_state.app_state.min_distance
                    )
                st.session_state.app_state.peak_info = peak_info
                st.session_state.app_state.derivatives = derivatives
                st.session_state.app_state.initial_params = initial_params
                st.success(f"Found {len(peak_info)} peaks!")
        
        if st.session_state.app_state.peak_info is not None:
            if st.button("✅ Confirm Peaks", use_container_width=True):
                with st.spinner("Fitting peaks..."):
                    # Create progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def update_progress(progress, message):
                        progress_bar.progress(progress)
                        status_text.text(message)
                    
                    success = st.session_state.app_state.deconvolver.fit(
                        initial_params=st.session_state.app_state.initial_params,
                        method=st.session_state.app_state.fitting_method,
                        maxfev=st.session_state.app_state.max_nfev,
                        progress_callback=update_progress
                    )
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    if success:
                        st.session_state.app_state.current_step = 4
                        st.rerun()
                    else:
                        st.error("Fitting failed. Try adjusting parameters.")
    
    with col2:
        if (st.session_state.app_state.peak_info is not None and 
            st.session_state.app_state.derivatives is not None):
            st.subheader(f"Peaks found: {len(st.session_state.app_state.peak_info)}")
            
            dy, d2y, y_smooth = st.session_state.app_state.derivatives
            deconv = st.session_state.app_state.deconvolver
            plotter = SpectrumPlotter()
            
            # Create tabs for different plots
            tab1, tab2, tab3 = st.tabs(["📊 Peaks", "📈 Derivatives", "📋 Information"])

            with tab1:
                fig, ax = plt.subplots(figsize=(10, 6))
                plotter.plot_with_peaks(
                    deconv, 
                    st.session_state.app_state.peak_info, 
                    y_smooth,
                    title=f"Peak Detection - {len(st.session_state.app_state.peak_info)} peaks found",
                    ax=ax
                )
                st.pyplot(fig)
                plt.close()
            
            with tab2:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
                
                # First derivative
                ax1.plot(deconv.x, dy, 'b-', linewidth=1.5, label='First derivative')
                ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                ax1.set_xlabel(deconv.x_label)
                ax1.set_ylabel('dy/dx')
                ax1.set_title('First Derivative')
                ax1.grid(True, alpha=0.3)
                ax1.legend()
                
                # Second derivative
                ax2.plot(deconv.x, d2y, 'g-', linewidth=1.5, label='Second derivative')
                ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                ax2.set_xlabel(deconv.x_label)
                ax2.set_ylabel('d²y/dx²')
                ax2.set_title('Second Derivative')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            with tab3:
                # Create a table with peak information in original units
                data = []
                for i, info in enumerate(st.session_state.app_state.peak_info):
                    data.append({
                        'Peak': i + 1,
                        'X Center': f"{info['x_linear']:.4e}",
                        'Y Amplitude': f"{info['y_original']:.4e}",
                        'Estimated Sigma': f"{info['sigma_est']:.4f}",
                    })
                
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)
                
                # Show peak detection statistics
                st.markdown("---")
                st.subheader("Detection Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Peaks Found", len(st.session_state.app_state.peak_info))
                with col2:
                    st.metric("X Range", f"{np.min(deconv.x_sorted):.2e} - {np.max(deconv.x_sorted):.2e}")
                with col3:
                    st.metric("Y Range", f"{np.min(deconv.y_sorted):.2e} - {np.max(deconv.y_sorted):.2e}")


# ==================== STEP 4: EDITING ====================

elif st.session_state.app_state.current_step == 4:
    st.header("Step 4: Peak Editing")
    
    if st.session_state.app_state.deconvolver and st.session_state.app_state.deconvolver.components:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Peak Management")
            
            # Display quality metrics
            metrics = st.session_state.app_state.deconvolver.quality_metrics
            st.info(f"R² = {metrics.get('R²', 0):.4f} | RMSE = {metrics.get('RMSE', 0):.2e}")
            
            st.markdown("---")
            
            # Peak selection
            peak_options = {f"Peak {c['id']}: center = {c['cen_linear']:.2e}, fraction = {c['fraction_percent']:.1f}%": c['id'] 
                           for c in st.session_state.app_state.deconvolver.components}
            
            selected_peak = st.selectbox(
                "Select peak for editing:",
                options=list(peak_options.keys())
            )
            
            if selected_peak:
                peak_id = peak_options[selected_peak]
                
                # Split position slider
                peak = st.session_state.app_state.deconvolver.components[peak_id - 1]
                min_x = np.min(st.session_state.app_state.deconvolver.x)
                max_x = np.max(st.session_state.app_state.deconvolver.x)
                default_pos = peak['cen_log']
                
                split_position = st.slider(
                    "Split position:",
                    min_value=float(min_x),
                    max_value=float(max_x),
                    value=float(default_pos),
                    format="%.4f"
                )
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    if st.button("✂️ Split Peak", use_container_width=True):
                        with st.spinner("Splitting peak..."):
                            if st.session_state.app_state.deconvolver.split_peak(peak_id, split_position):
                                st.rerun()
                
                with col_b:
                    if st.button("🗑️ Remove Peak", use_container_width=True):
                        with st.spinner("Removing peak..."):
                            if st.session_state.app_state.deconvolver.remove_peak(peak_id):
                                st.rerun()
                
                if st.button("🔄 Recalculate All", use_container_width=True):
                    with st.spinner("Recalculating..."):
                        if st.session_state.app_state.deconvolver.fit(
                            initial_params=st.session_state.app_state.deconvolver.popt,
                            method=st.session_state.app_state.fitting_method,
                            maxfev=st.session_state.app_state.max_nfev
                        ):
                            st.rerun()
            
            st.markdown("---")
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("⬅️ Back", use_container_width=True):
                    st.session_state.app_state.current_step = 3
                    st.rerun()
            with col_b:
                if st.button("✅ Finish Editing", type="primary", use_container_width=True):
                    st.session_state.app_state.current_step = 5
                    st.rerun()
        
        with col2:
            st.subheader("Current Deconvolution")
            
            # Use the new scientific Plotly figure
            fig = st.session_state.app_state.deconvolver.create_scientific_plotly_figure()
            st.plotly_chart(fig, use_container_width=True)


# ==================== STEP 5: RESULTS ====================

elif st.session_state.app_state.current_step == 5:
    st.header("Step 5: Results")
    
    if st.session_state.app_state.deconvolver and st.session_state.app_state.deconvolver.components:
        
        # Back button at the top
        col_back, _ = st.columns([1, 5])
        with col_back:
            if st.button("⬅️ Back to Editing", use_container_width=True):
                st.session_state.app_state.current_step = 4
                st.rerun()
        
        st.markdown("---")
        
        # Create tabs for results
        tab1, tab2, tab3 = st.tabs(["📊 Graphs", "📋 Table", "📈 Export"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Deconvolution Result")
                
                plotter = SpectrumPlotter()
                fig, ax = plt.subplots(figsize=(10, 6))
                
                plotter.plot_deconvolution_result(
                    st.session_state.app_state.deconvolver,
                    show_components=True,
                    title="Deconvolution Result",
                    ax=ax
                )
                
                st.pyplot(fig)
                plt.close()
            
            with col2:
                st.subheader("Area Distribution")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Pie chart
                peaks = [f'{c["id"]}' for c in st.session_state.app_state.deconvolver.components]
                fractions = [c['fraction_percent'] for c in st.session_state.app_state.deconvolver.components]
                colors = plt.cm.Set3(np.linspace(0, 1, len(peaks)))
                ax1.pie(fractions, labels=peaks, autopct='%1.1f%%',
                       colors=colors, startangle=90,
                       textprops={'fontweight': 'bold'})
                ax1.set_title('Area Distribution', fontweight='bold')
                
                # Bar chart
                centers = [c['cen_linear'] for c in st.session_state.app_state.deconvolver.components]
                areas = [c['area'] for c in st.session_state.app_state.deconvolver.components]
                
                if st.session_state.app_state.deconvolver.use_log_x:
                    ax2.set_xscale('log')
                
                bars = ax2.bar(range(len(centers)), areas, 
                              tick_label=[f'{c:.2e}' for c in centers],
                              color='steelblue', edgecolor='black', alpha=0.7)
                ax2.set_xlabel('Peak Center', fontweight='bold')
                ax2.set_ylabel('Area', fontweight='bold')
                ax2.set_title('Peak Areas', fontweight='bold')
                ax2.tick_params(axis='x', rotation=45)
                
                # Scientific styling for bar chart
                ax2.spines['top'].set_visible(True)
                ax2.spines['right'].set_visible(True)
                ax2.spines['bottom'].set_linewidth(1)
                ax2.spines['left'].set_linewidth(1)
                ax2.spines['top'].set_linewidth(1)
                ax2.spines['right'].set_linewidth(1)
                ax2.tick_params(direction='out', length=4, width=1)
                ax2.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        
        with tab2:
            st.subheader("Results Table")
            
            data = []
            for c in st.session_state.app_state.deconvolver.components:
                data.append({
                    'Peak': c['id'],
                    'Center': f"{c['cen_linear']:.4e}",
                    'Amplitude': f"{c['amp']:.4e}",
                    'Sigma': f"{c['sigma_log']:.4f}",
                    'FWHM': f"{c['fwhm']:.4f}",
                    'Area': f"{c['area']:.4e}",
                    'Fraction (%)': f"{c['fraction_percent']:.2f}"
                })
            
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
            
            st.markdown("---")
            
            metrics = st.session_state.app_state.deconvolver.quality_metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("R²", f"{metrics.get('R²', 0):.6f}")
            with col2:
                st.metric("AIC", f"{metrics.get('AIC', 0):.2f}")
            with col3:
                st.metric("BIC", f"{metrics.get('BIC', 0):.2f}")
            with col4:
                st.metric("RMSE", f"{metrics.get('RMSE', 0):.2e}")
        
        with tab3:
            st.subheader("Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Export to CSV
                if st.button("📥 Export to CSV", use_container_width=True):
                    # Peak data
                    df_peaks = pd.DataFrame([{
                        'Peak_ID': c['id'],
                        'Center': c['cen_linear'],
                        'Amplitude': c['amp'],
                        'Sigma': c['sigma_log'],
                        'FWHM': c['fwhm'],
                        'Area': c['area'],
                        'Fraction_Percent': c['fraction_percent']
                    } for c in st.session_state.app_state.deconvolver.components])
                    
                    # Convert to CSV
                    csv_peaks = df_peaks.to_csv(index=False)
                    
                    st.download_button(
                        label="Download Peaks CSV",
                        data=csv_peaks,
                        file_name=f"deconvolution_peaks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Fitting data
                    if 'Residuals' in st.session_state.app_state.deconvolver.quality_metrics:
                        df_fit = pd.DataFrame({
                            'X_original': st.session_state.app_state.deconvolver.x_linear,
                            'Y_original': st.session_state.app_state.deconvolver.y_original,
                            'Y_fit': st.session_state.app_state.deconvolver.fit_y_norm * st.session_state.app_state.deconvolver.y_max,
                            'Residuals': st.session_state.app_state.deconvolver.quality_metrics['Residuals'] * st.session_state.app_state.deconvolver.y_max
                        })
                        
                        csv_fit = df_fit.to_csv(index=False)
                        
                        st.download_button(
                            label="Download Fitting CSV",
                            data=csv_fit,
                            file_name=f"deconvolution_fit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
            
            with col2:
                # Export report
                if st.button("📄 Export Report", use_container_width=True):
                    report = f"""GAUSSIAN DECONVOLUTION REPORT
{"="*80}

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Number of points: {len(st.session_state.app_state.deconvolver.x_linear)}
X range: [{st.session_state.app_state.deconvolver.x_linear[0]:.2e}, {st.session_state.app_state.deconvolver.x_linear[-1]:.2e}]
Logarithmic X scale: {st.session_state.app_state.deconvolver.use_log_x}

QUALITY METRICS:
{"-"*40}
R²: {st.session_state.app_state.deconvolver.quality_metrics.get('R²', 0):.6f}
AIC: {st.session_state.app_state.deconvolver.quality_metrics.get('AIC', 0):.2f}
BIC: {st.session_state.app_state.deconvolver.quality_metrics.get('BIC', 0):.2f}
χ²: {st.session_state.app_state.deconvolver.quality_metrics.get('χ²', 0):.2e}
RMSE: {st.session_state.app_state.deconvolver.quality_metrics.get('RMSE', 0):.2e}

COMPONENTS:
{"-"*80}
ID    Center          Amplitude       FWHM        Area           Fraction(%)
{"-"*80}"""
                    
                    for c in st.session_state.app_state.deconvolver.components:
                        report += f"\n{c['id']:<4} {c['cen_linear']:<15.4e} {c['amp']:<15.4e} {c['fwhm']:<12.4f} {c['area']:<15.4e} {c['fraction_percent']:<10.2f}"
                    
                    report += f"\n{'='*80}\nTotal area: {st.session_state.app_state.deconvolver.total_area:.6e}\n{'='*80}"
                    
                    st.download_button(
                        label="Download Report",
                        data=report,
                        file_name=f"deconvolution_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
            
            st.markdown("---")
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🔄 New Analysis", use_container_width=True):
                    st.session_state.app_state = AppState()
                    st.rerun()
