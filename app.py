import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks, peak_widths
from scipy.optimize import curve_fit, least_squares
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from datetime import datetime
import io
import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any, Callable
import time
from dataclasses import dataclass, field

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
    max_nfev: int = 5000
    show_warnings: bool = True
    baseline_method: str = 'none'
    baseline_degree: int = 1
    fit_quality: str = 'balanced'
    last_popt: Optional[np.ndarray] = None
    pending_split: Optional[Tuple[int, float]] = None
    pending_remove: Optional[int] = None
    preview_mode: bool = False
    # Новые параметры для сглаживания и ручного добавления пиков
    smoothing_enabled: bool = False
    smoothing_window: int = 11
    smoothing_polyorder: int = 3
    manual_peak_position: Optional[float] = None
    manual_peak_added: List[Dict] = field(default_factory=list)
    residual_peaks: List[Dict] = field(default_factory=list)
    pending_manual_peaks: List[Dict] = field(default_factory=list)
    pending_residual_peaks: List[int] = field(default_factory=list)

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
    
    def smooth_data(self, x, y, method='savgol', window=11, polyorder=3):
        """Apply smoothing to data for noise reduction"""
        if len(y) < window:
            if self.show_warnings:
                warnings.warn(f"Data length ({len(y)}) less than window ({window}), using original data")
            return y
        
        try:
            if method == 'savgol':
                # Ensure window is odd
                if window % 2 == 0:
                    window += 1
                # Ensure window is not larger than data
                window = min(window, len(y) if len(y) % 2 == 1 else len(y) - 1)
                if window >= polyorder + 2:
                    return savgol_filter(y, window, polyorder)
                else:
                    return y
            elif method == 'gaussian':
                from scipy.ndimage import gaussian_filter1d
                sigma = window / 5
                return gaussian_filter1d(y, sigma)
            elif method == 'median':
                from scipy.signal import medfilt
                kernel_size = window if window % 2 == 1 else window + 1
                return medfilt(y, kernel_size)
            else:
                return y
        except Exception as e:
            if self.show_warnings:
                warnings.warn(f"Smoothing failed: {e}, using original data")
            return y


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
    """Model for sum of Gaussians with baseline correction"""
    
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
    def multi_gaussian_with_baseline(x, n_peaks, peak_params, baseline_params, baseline_method):
        """Sum of Gaussians with baseline correction"""
        # Calculate peaks
        y_peaks = np.zeros_like(x, dtype=float)
        for i in range(n_peaks):
            amp = peak_params[3*i]
            cen = peak_params[3*i + 1]
            sigma = abs(peak_params[3*i + 2])
            y_peaks += GaussianModel.gaussian(x, amp, cen, sigma)
        
        # Calculate baseline
        if baseline_method == "constant" and len(baseline_params) >= 1:
            y_baseline = baseline_params[0]
        elif baseline_method == "linear" and len(baseline_params) >= 2:
            y_baseline = baseline_params[0] + baseline_params[1] * x
        elif baseline_method == "quadratic" and len(baseline_params) >= 3:
            y_baseline = baseline_params[0] + baseline_params[1] * x + baseline_params[2] * x**2
        else:
            y_baseline = 0
        
        return y_peaks + y_baseline
    
    @staticmethod
    def multi_gaussian_with_baseline_flat(x, *params, n_peaks, baseline_method):
        """Flat version for curve_fit"""
        if baseline_method == "none":
            return GaussianModel.multi_gaussian(x, *params)
        
        n_baseline_params = {
            'none': 0,
            'constant': 1,
            'linear': 2,
            'quadratic': 3
        }.get(baseline_method, 0)
        
        peak_params = params[:n_peaks*3]
        baseline_params = params[n_peaks*3:] if n_baseline_params > 0 else []
        
        return GaussianModel.multi_gaussian_with_baseline(
            x, n_peaks, peak_params, baseline_params, baseline_method
        )
    
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
    
    @staticmethod
    def estimate_parameters_from_position(x, y, position):
        """Estimate Gaussian parameters from a given x-position"""
        # Find closest index
        idx = np.argmin(np.abs(x - position))
        
        # Amplitude is the y value at that position
        amp = y[idx]
        
        # Estimate sigma from surrounding region
        # Find left and right where y drops to half of peak
        half_max = amp / 2
        
        left_idx = idx
        right_idx = idx
        
        for i in range(idx, -1, -1):
            if y[i] <= half_max:
                left_idx = i
                break
            left_idx = i
        
        for i in range(idx, len(y)):
            if y[i] <= half_max:
                right_idx = i
                break
            right_idx = i
        
        width = (x[right_idx] - x[left_idx]) if right_idx > left_idx else (x[-1] - x[0]) / 10
        sigma = width / (2 * np.sqrt(2 * np.log(2))) if width > 0 else (x[-1] - x[0]) / 20
        
        return amp, position, max(sigma, 0.001 * (x[-1] - x[0]))


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
    """Handles Gaussian fitting with multiple optimization methods and baseline"""
    
    def __init__(self, method='trf', max_nfev=5000, baseline_method='none', 
                 fit_quality='balanced', last_popt=None):
        self.method = method
        self.max_nfev = max_nfev
        self.baseline_method = baseline_method
        self.fit_quality = fit_quality
        self.last_popt = last_popt
        self.convergence_history = []
        self.fit_progress = 0
        
        # Set tolerances based on quality
        if fit_quality == 'fast':
            self.xtol = 1e-3
            self.ftol = 1e-3
            self.gtol = 1e-3
        elif fit_quality == 'balanced':
            self.xtol = 1e-5
            self.ftol = 1e-5
            self.gtol = 1e-5
        else:  # precise
            self.xtol = 1e-8
            self.ftol = 1e-8
            self.gtol = 1e-8
    
    def get_n_baseline_params(self):
        """Get number of baseline parameters"""
        return {
            'none': 0,
            'constant': 1,
            'linear': 2,
            'quadratic': 3
        }.get(self.baseline_method, 0)
    
    def fit(self, x, y_norm, initial_peak_params, y_max, 
            progress_callback=None, fixed_params=None):
        """Perform fitting with progress tracking"""
        n_peaks = len(initial_peak_params) // 3
        n_baseline = self.get_n_baseline_params()
        
        # Use last good parameters if available (but keep same structure)
        if self.last_popt is not None:
            # Check if structure matches (same number of peaks and baseline)
            expected_len = n_peaks * 3 + n_baseline
            if len(self.last_popt) == expected_len:
                initial_params = self.last_popt.copy()
                if progress_callback:
                    progress_callback(0.1, "Using cached parameters...")
            else:
                # Structure changed, use initial
                initial_params = np.array(initial_peak_params)
                if n_baseline > 0:
                    # Add initial baseline estimates
                    if self.baseline_method == 'constant':
                        baseline_init = [np.percentile(y_norm, 5)]  # 5th percentile as baseline
                    elif self.baseline_method == 'linear':
                        baseline_init = [np.percentile(y_norm, 5), 0]
                    else:  # quadratic
                        baseline_init = [np.percentile(y_norm, 5), 0, 0]
                    initial_params = np.concatenate([initial_params, baseline_init])
        else:
            initial_params = np.array(initial_peak_params)
            if n_baseline > 0:
                # Add initial baseline estimates
                if self.baseline_method == 'constant':
                    baseline_init = [np.percentile(y_norm, 5)]
                elif self.baseline_method == 'linear':
                    baseline_init = [np.percentile(y_norm, 5), 0]
                else:  # quadratic
                    baseline_init = [np.percentile(y_norm, 5), 0, 0]
                initial_params = np.concatenate([initial_params, baseline_init])
        
        if len(initial_params) == 0:
            return False, None, None, None
        
        # Create bounds
        lower_bounds, upper_bounds = self._create_bounds(x, y_norm, n_peaks, n_baseline)
        
        # Ensure initial_params are within bounds
        for i in range(len(initial_params)):
            initial_params[i] = np.clip(initial_params[i], lower_bounds[i], upper_bounds[i])
        
        try:
            if progress_callback:
                progress_callback(0.3, "Initializing fit...")
            
            # Create the model function with correct number of parameters
            def model_func(x, *params):
                return GaussianModel.multi_gaussian_with_baseline_flat(
                    x, *params, n_peaks=n_peaks, baseline_method=self.baseline_method
                )
            
            # Perform fit with selected method and tolerances
            popt, pcov = curve_fit(
                model_func,
                x,
                y_norm,
                p0=initial_params,
                bounds=(lower_bounds, upper_bounds),
                method=self.method,
                maxfev=self.max_nfev,
                xtol=self.xtol,
                ftol=self.ftol,
                gtol=self.gtol
            )
            
            if progress_callback:
                progress_callback(0.8, "Calculating components...")
            
            fit_y_norm = model_func(x, *popt)
            
            # Extract peak parameters
            peak_params = popt[:n_peaks*3]
            baseline_params = popt[n_peaks*3:] if n_baseline > 0 else []
            
            # Extract components
            components = []
            for i in range(n_peaks):
                amp_norm = peak_params[3*i]
                cen = peak_params[3*i + 1]
                sigma = abs(peak_params[3*i + 2])
                
                amp = amp_norm * y_max
                area = GaussianModel.calculate_area(amp_norm, sigma) * y_max
                
                component_y_norm = GaussianModel.gaussian(x, amp_norm, cen, sigma)
                
                # Calculate center in linear space
                if hasattr(x, 'min') and hasattr(x, 'max'):  # Simple check
                    cen_linear = 10**cen if np.any(x < 0) else cen
                else:
                    cen_linear = cen
                
                components.append({
                    'id': i + 1,
                    'amp_norm': amp_norm,
                    'amp': amp,
                    'cen_log': cen,
                    'cen_linear': cen_linear,
                    'sigma_log': sigma,
                    'fwhm': GaussianModel.calculate_fwhm(sigma),
                    'area': area,
                    'fraction': 0,
                    'y_norm': component_y_norm,
                    'source': 'auto'  # Track source of peak
                })
            
            # Calculate fractions
            total_area = sum([c['area'] for c in components])
            for c in components:
                c['fraction'] = c['area'] / total_area if total_area > 0 else 0
                c['fraction_percent'] = c['fraction'] * 100
            
            if progress_callback:
                progress_callback(1.0, "Fit complete!")
            
            return True, popt, components, baseline_params
            
        except Exception as e:
            if progress_callback:
                progress_callback(1.0, f"Fit failed: {e}")
            return False, None, None, None
    
    def _create_bounds(self, x, y_norm, n_peaks, n_baseline):
        """Create bounds for fitting"""
        lower_bounds = []
        upper_bounds = []
        x_range = np.max(x) - np.min(x)
        y_range = np.max(y_norm) - np.min(y_norm)
        
        # Peak bounds
        for i in range(n_peaks):
            lower_bounds.extend([0, np.min(x), x_range * 0.001])
            upper_bounds.extend([2 * np.max(y_norm), np.max(x), x_range * 0.5])
        
        # Baseline bounds
        if n_baseline >= 1:  # constant
            lower_bounds.append(-np.max(y_norm))
            upper_bounds.append(np.max(y_norm))
        if n_baseline >= 2:  # linear term
            lower_bounds.append(-x_range)
            upper_bounds.append(x_range)
        if n_baseline >= 3:  # quadratic term
            lower_bounds.append(-x_range**2)
            upper_bounds.append(x_range**2)
        
        return lower_bounds, upper_bounds
    
    def preview_fit(self, x, peak_params, y_max, baseline_params=None):
        """Preview fit without optimization (fast)"""
        n_peaks = len(peak_params) // 3
        n_baseline = self.get_n_baseline_params()
        
        if baseline_params is None and n_baseline > 0:
            # Use default baseline
            if self.baseline_method == 'constant':
                baseline_params = [0]
            elif self.baseline_method == 'linear':
                baseline_params = [0, 0]
            else:  # quadratic
                baseline_params = [0, 0, 0]
        
        # Calculate fit
        fit_y_norm = GaussianModel.multi_gaussian_with_baseline(
            x, n_peaks, peak_params, baseline_params or [], self.baseline_method
        )
        
        return fit_y_norm


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
                        title="Peak Detection", ax=None, figsize=(10, 6),
                        manual_peak_position=None, pending_manual_peaks=None,
                        residual_peaks=None):
        """Plot data with detected peaks (auto, manual, residual) with color coding"""
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
        
        # Mark automatically detected peaks (green)
        for i, info in enumerate(peak_info):
            peak_y_original = info['y_original']
            ax.plot(info['x_linear'], peak_y_original, 
                    'go', markersize=10, markeredgecolor='darkgreen', 
                    markerfacecolor='lightgreen', zorder=3, label='Auto Peaks' if i == 0 else "")
            ax.text(info['x_linear'], peak_y_original * 1.05, 
                    f'A{i+1}', ha='center', fontweight='bold', 
                    fontsize=10, color='darkgreen',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8),
                    zorder=4)
        
        # Mark manually added peaks (orange)
        if pending_manual_peaks:
            for i, mp in enumerate(pending_manual_peaks):
                ax.plot(mp['x_linear'], mp['y_original'], 
                        'o', markersize=10, markeredgecolor='darkorange', 
                        markerfacecolor='orange', zorder=3, label='Manual Peaks' if i == 0 else "")
                ax.text(mp['x_linear'], mp['y_original'] * 1.05, 
                        f'M{i+1}', ha='center', fontweight='bold', 
                        fontsize=10, color='darkorange',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8),
                        zorder=4)
        
        # Mark residual-based peaks (blue)
        if residual_peaks:
            for i, rp in enumerate(residual_peaks):
                if rp.get('selected', True):
                    ax.plot(rp['x_linear'], rp['y_original'], 
                            's', markersize=8, markeredgecolor='darkblue', 
                            markerfacecolor='lightblue', zorder=3, label='Residual Peaks' if i == 0 else "")
                    ax.text(rp['x_linear'], rp['y_original'] * 1.05, 
                            f'R{i+1}', ha='center', fontweight='bold', 
                            fontsize=10, color='darkblue',
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8),
                            zorder=4)
        
        # Show manual peak preview (red point on cursor)
        if manual_peak_position is not None:
            # Find y-value at this position
            idx = np.argmin(np.abs(deconvolver.x_sorted - manual_peak_position))
            y_at_pos = deconvolver.y_sorted[idx]
            ax.plot(manual_peak_position, y_at_pos, 
                    'ro', markersize=12, markeredgecolor='darkred', 
                    markerfacecolor='red', zorder=5, label='Preview Position')
            ax.axvline(x=manual_peak_position, color='red', linestyle='--', 
                      alpha=0.5, linewidth=1, zorder=4)
        
        # Labels and title
        x_label = 'X' + (' (log scale)' if deconvolver.use_log_x else '')
        y_label = 'Y' + (' (log scale)' if deconvolver.use_log_y else '')
        ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Legend and grid
        ax.legend(loc='best', fontsize=9, frameon=True, edgecolor='black')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        if self.scientific_style:
            self._apply_scientific_style(ax)
        
        return fig, ax
    
    def plot_residual_analysis(self, x, y_original, y_fit, residuals, 
                               title="Residual Analysis", ax=None, figsize=(10, 6)):
        """Plot residual analysis for peak detection"""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        # Plot residuals
        ax.plot(x, residuals, 'b-', linewidth=1.5, alpha=0.7, label='Residuals')
        ax.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.7)
        ax.fill_between(x, 0, residuals, alpha=0.3, color='blue', where=(residuals > 0), 
                        interpolate=True, label='Positive residuals')
        ax.fill_between(x, 0, residuals, alpha=0.3, color='red', where=(residuals < 0), 
                        interpolate=True, label='Negative residuals')
        
        # Highlight potential missed peaks
        from scipy.signal import find_peaks
        peaks_pos, _ = find_peaks(residuals, height=np.std(residuals) * 1.5)
        peaks_neg, _ = find_peaks(-residuals, height=np.std(residuals) * 1.5)
        
        for p in peaks_pos:
            ax.plot(x[p], residuals[p], 'go', markersize=8, 
                   markeredgecolor='darkgreen', markerfacecolor='lightgreen')
        
        for p in peaks_neg:
            ax.plot(x[p], residuals[p], 'ro', markersize=8,
                   markeredgecolor='darkred', markerfacecolor='lightcoral')
        
        ax.set_xlabel('X', fontsize=12, fontweight='bold')
        ax.set_ylabel('Residuals', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        if self.scientific_style:
            self._apply_scientific_style(ax)
        
        return fig, ax
    
    def plot_deconvolution_result(self, deconvolver, show_components=True, show_baseline=True,
                                  title="Deconvolution Result", ax=None, figsize=(10, 6),
                                  preview_mode=False, preview_fit=None):
        """Plot deconvolution result with components and baseline"""
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
        
        # Plot components with color coding based on source
        if show_components and deconvolver.components:
            colors = plt.cm.Set3(np.linspace(0, 1, len(deconvolver.components)))
            for c, color in zip(deconvolver.components, colors):
                # Set line style based on source
                line_style = '-'
                if c.get('source') == 'manual':
                    line_style = '--'
                elif c.get('source') == 'residual':
                    line_style = ':'
                
                y_component = GaussianModel.gaussian(x_dense_log, c['amp_norm'], 
                                                    c['cen_log'], c['sigma_log']) * deconvolver.y_max
                
                # Fill under Gaussian
                ax.fill_between(x_dense, 0, y_component, 
                                color=color, alpha=0.3, linewidth=0)
                
                # Plot line
                ax.plot(x_dense, y_component, line_style, color=color, linewidth=2,
                       label=f'Peak {c["id"]}: {c["fraction_percent"]:.1f}% ({c.get("source", "auto")})', zorder=2)
        
        # Plot baseline if available
        if show_baseline and hasattr(deconvolver, 'baseline_params') and deconvolver.baseline_params:
            if deconvolver.baseline_method == 'constant':
                y_baseline = deconvolver.baseline_params[0] * deconvolver.y_max
                ax.axhline(y=y_baseline, color='gray', linestyle=':', 
                          linewidth=1.5, label='Baseline', zorder=1)
            elif deconvolver.baseline_method == 'linear':
                y_baseline = (deconvolver.baseline_params[0] + 
                            deconvolver.baseline_params[1] * x_dense_log) * deconvolver.y_max
                ax.plot(x_dense, y_baseline, 'gray', linestyle=':', 
                       linewidth=1.5, label='Baseline', zorder=1)
            elif deconvolver.baseline_method == 'quadratic':
                y_baseline = (deconvolver.baseline_params[0] + 
                            deconvolver.baseline_params[1] * x_dense_log +
                            deconvolver.baseline_params[2] * x_dense_log**2) * deconvolver.y_max
                ax.plot(x_dense, y_baseline, 'gray', linestyle=':', 
                       linewidth=1.5, label='Baseline', zorder=1)
        
        # Plot total fit
        if preview_mode and preview_fit is not None:
            y_total = preview_fit * deconvolver.y_max
            ax.plot(x_dense, y_total, 'b--', linewidth=2, 
                   label='Preview (no fit)', zorder=3, alpha=0.7)
        elif deconvolver.fit_y_norm is not None:
            # Reconstruct total fit with baseline
            if hasattr(deconvolver, 'baseline_params') and deconvolver.baseline_params:
                n_peaks = len(deconvolver.components)
                peak_params = []
                for c in deconvolver.components:
                    peak_params.extend([c['amp_norm'], c['cen_log'], c['sigma_log']])
                
                y_total = GaussianModel.multi_gaussian_with_baseline(
                    x_dense_log, n_peaks, peak_params, 
                    deconvolver.baseline_params, deconvolver.baseline_method
                ) * deconvolver.y_max
            else:
                y_total = GaussianModel.multi_gaussian(x_dense_log, *deconvolver.popt) * deconvolver.y_max
            
            ax.plot(x_dense, y_total, 'r--', linewidth=2, label='Total Fit', zorder=3)
        
        # Labels and title
        x_label = 'X' + (' (log scale)' if deconvolver.use_log_x else '')
        y_label = 'Y' + (' (log scale)' if deconvolver.use_log_y else '')
        ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add quality metrics to plot if available
        if deconvolver.quality_metrics and not preview_mode:
            metrics_text = f"R² = {deconvolver.quality_metrics.get('R²', 0):.4f}\n"
            metrics_text += f"RMSE = {deconvolver.quality_metrics.get('RMSE', 0):.2e}"
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        elif preview_mode:
            ax.text(0.02, 0.98, "PREVIEW MODE\n(no fit performed)", 
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
        
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
    """Main class for spectral deconvolution with baseline correction"""
    
    def __init__(self, x_linear, y_original, use_log_x=True, use_log_y=False,
                 clip_negative=True, show_warnings=True, baseline_method='none'):
        # Store original data WITHOUT ANY MODIFICATIONS for display purposes
        self.x_original = np.array(x_linear).copy()
        self.y_original_raw = np.array(y_original).copy()
        
        # Working arrays that may be modified
        self.x_linear = np.array(x_linear)
        self.y_original = np.array(y_original)
        self.use_log_x = use_log_x
        self.use_log_y = use_log_y
        self.baseline_method = baseline_method
        
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
        
        # Normalization - use 95th percentile instead of max for robustness
        self.y_max = np.percentile(self.y_for_fitting, 95) if np.any(self.y_for_fitting > 0) else 1.0
        
        # For fitting, we normalize but keep track for denormalization
        if self.y_max > 0:
            self.y_norm = self.y / self.y_max
        else:
            self.y_norm = self.y
        
        # Results containers
        self.components = []
        self.fit_y_norm = None
        self.popt = None
        self.baseline_params = None
        self.quality_metrics = {}
        self.convergence_history = []
        self.total_area = 0
        
        # Fitter
        self.fitter = None
        
        # For compatibility with existing code
        self.multi_gaussian = GaussianModel.multi_gaussian
        self.gaussian = GaussianModel.gaussian
    
    def apply_smoothing(self, method='savgol', window=11, polyorder=3):
        """Apply smoothing to the data for noise reduction"""
        smoothed_y = self.preprocessor.smooth_data(
            self.x_sorted, self.y_for_fitting, method, window, polyorder
        )
        
        # Update y_for_fitting and recalculate y_norm
        self.y_for_fitting = smoothed_y
        
        # Recalculate normalization
        self.y_max = np.percentile(self.y_for_fitting, 95) if np.any(self.y_for_fitting > 0) else 1.0
        
        # Update y (which may be log transformed)
        if self.use_log_y:
            eps = np.finfo(float).eps
            y_pos = np.maximum(self.y_for_fitting, eps)
            self.y = np.log10(y_pos)
        else:
            self.y = self.y_for_fitting
        
        # Update normalized y
        if self.y_max > 0:
            self.y_norm = self.y / self.y_max
        else:
            self.y_norm = self.y
        
        return True
    
    def auto_detect_peaks(self, sensitivity=0.03, min_distance=5, smoothing_window=None):
        """Automatic peak detection using derivatives with optional smoothing"""
        # Apply smoothing if specified
        y_for_peak_detection = self.y_norm
        if smoothing_window and smoothing_window > 5:
            y_for_peak_detection = self.preprocessor.smooth_data(
                self.x, self.y_norm, 'savgol', smoothing_window, 3
            )
        
        # Smoothing
        window_length = min(11, len(y_for_peak_detection) // 5 * 2 + 1)
        if window_length % 2 == 0:
            window_length += 1
        
        if window_length >= 5:
            y_smooth = savgol_filter(y_for_peak_detection, window_length, 3)
        else:
            y_smooth = y_for_peak_detection
        
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
                'd2y': d2y[peak_idx],
                'source': 'auto'
            })
            
            initial_params.extend([amp, cen, sigma])
        
        return filtered_peaks, peak_info, initial_params, (dy, d2y, y_smooth)
    
    def add_manual_peak(self, position):
        """Add a manual peak at given position"""
        # Estimate parameters from position
        amp, cen, sigma = GaussianModel.estimate_parameters_from_position(
            self.x, self.y_norm, position
        )
        
        # Get original Y value
        if self.use_log_x:
            x_linear = 10**position
        else:
            x_linear = position
        
        idx = np.argmin(np.abs(self.x_sorted - x_linear))
        y_original_value = self.y_sorted[idx]
        
        manual_peak = {
            'index': len(self.components) + len(st.session_state.app_state.pending_manual_peaks) + 1,
            'x': position,
            'x_linear': x_linear,
            'y': self.y_norm[idx],
            'y_original': y_original_value,
            'amp_est': amp,
            'cen_est': cen,
            'sigma_est': sigma,
            'source': 'manual'
        }
        
        return manual_peak
    
    def detect_residual_peaks(self, sensitivity=0.5, min_distance=3):
        """Detect peaks in residuals after initial fit"""
        if self.fit_y_norm is None:
            return []
        
        # Calculate residuals
        residuals = self.y_norm - self.fit_y_norm
        
        # Find peaks in residuals
        height_threshold = sensitivity * np.std(residuals)
        peaks, properties = find_peaks(residuals, height=height_threshold, 
                                       distance=min_distance, prominence=height_threshold)
        
        residual_peaks = []
        for peak_idx in peaks:
            position = self.x[peak_idx]
            amp = residuals[peak_idx]
            
            # Estimate sigma
            sigma = GaussianModel.estimate_sigma_from_peak(self.x, residuals, peak_idx)
            sigma = max(sigma, 0.01 * (np.max(self.x) - np.min(self.x)) / 10)
            
            # Get original Y value
            if self.use_log_x:
                x_linear = 10**position
            else:
                x_linear = position
            
            idx = np.argmin(np.abs(self.x_sorted - x_linear))
            y_original_value = self.y_sorted[idx]
            
            residual_peaks.append({
                'index': peak_idx,
                'x': position,
                'x_linear': x_linear,
                'y': residuals[peak_idx],
                'y_original': y_original_value,
                'amp_est': amp,
                'cen_est': position,
                'sigma_est': sigma,
                'source': 'residual',
                'selected': True
            })
        
        return residual_peaks
    
    def fit(self, initial_params=None, method='trf', maxfev=5000, 
            fit_quality='balanced', last_popt=None, progress_callback=None):
        """Perform fitting with selected method and baseline"""
        if initial_params is None:
            _, _, initial_params, _ = self.auto_detect_peaks()
        
        # Add manual peaks to initial_params
        manual_peaks = st.session_state.app_state.pending_manual_peaks
        for mp in manual_peaks:
            initial_params.extend([mp['amp_est'], mp['cen_est'], mp['sigma_est']])
        
        # Add selected residual peaks to initial_params
        residual_peaks = st.session_state.app_state.residual_peaks
        for i, rp in enumerate(residual_peaks):
            if i in st.session_state.app_state.pending_residual_peaks or rp.get('selected', True):
                initial_params.extend([rp['amp_est'], rp['cen_est'], rp['sigma_est']])
        
        if len(initial_params) == 0:
            return False
        
        # Create fitter with selected parameters
        self.fitter = GaussianFitter(
            method=method, 
            max_nfev=maxfev,
            baseline_method=self.baseline_method,
            fit_quality=fit_quality,
            last_popt=last_popt
        )
        
        # Perform fit
        success, popt, components, baseline_params = self.fitter.fit(
            self.x, self.y_norm, initial_params, self.y_max,
            progress_callback=progress_callback
        )
        
        if success:
            self.popt = popt
            # Assign sources to components
            source_counter = {'auto': 0, 'manual': 0, 'residual': 0}
            for c in components:
                if source_counter['auto'] < len(st.session_state.app_state.peak_info or []):
                    c['source'] = 'auto'
                    source_counter['auto'] += 1
                elif source_counter['manual'] < len(manual_peaks):
                    c['source'] = 'manual'
                    source_counter['manual'] += 1
                else:
                    c['source'] = 'residual'
                    source_counter['residual'] += 1
            
            self.components = components
            self.baseline_params = baseline_params
            
            # Reconstruct full fit with baseline
            n_peaks = len(components)
            peak_params = []
            for c in components:
                peak_params.extend([c['amp_norm'], c['cen_log'], c['sigma_log']])
            
            self.fit_y_norm = GaussianModel.multi_gaussian_with_baseline(
                self.x, n_peaks, peak_params, baseline_params or [], self.baseline_method
            )
            
            # Calculate total area (excluding baseline)
            self.total_area = sum([c['area'] for c in self.components])
            
            # Quality metrics
            self.quality_metrics = FitQualityAnalyzer.calculate_metrics(
                self.y_norm, self.fit_y_norm, len(popt)
            )
            
            return True
        
        return False
    
    def preview_fit(self, initial_params=None):
        """Preview fit without optimization (fast)"""
        if initial_params is None:
            _, _, initial_params, _ = self.auto_detect_peaks()
        
        # Add manual peaks
        for mp in st.session_state.app_state.pending_manual_peaks:
            initial_params.extend([mp['amp_est'], mp['cen_est'], mp['sigma_est']])
        
        # Add residual peaks
        for rp in st.session_state.app_state.residual_peaks:
            if rp.get('selected', True):
                initial_params.extend([rp['amp_est'], rp['cen_est'], rp['sigma_est']])
        
        if len(initial_params) == 0:
            return None
        
        # Create temporary fitter
        fitter = GaussianFitter(
            baseline_method=self.baseline_method
        )
        
        # Preview fit
        n_baseline = fitter.get_n_baseline_params()
        baseline_params = [0] * n_baseline if n_baseline > 0 else None
        
        fit_y_norm = fitter.preview_fit(
            self.x, initial_params, self.y_max, baseline_params
        )
        
        return fit_y_norm
    
    def remove_peak(self, peak_id):
        """Remove a peak (does NOT perform fit, just marks for removal)"""
        if peak_id > len(self.components):
            return False
        
        # Store the operation
        st.session_state.app_state.pending_remove = peak_id
        return True
    
    def split_peak(self, peak_id, split_position):
        """Split a peak into two (does NOT perform fit, just marks for splitting)"""
        if peak_id > len(self.components):
            return False
        
        # Store the operation
        st.session_state.app_state.pending_split = (peak_id, split_position)
        return True
    
    def apply_pending_operations(self, fit_quality='balanced', progress_callback=None):
        """Apply all pending operations and perform fit"""
        # Get current parameters
        if self.components:
            current_params = []
            for c in self.components:
                current_params.extend([c['amp_norm'], c['cen_log'], c['sigma_log']])
        else:
            # Use auto-detected peaks plus manual and residual
            _, _, current_params, _ = self.auto_detect_peaks()
            for mp in st.session_state.app_state.pending_manual_peaks:
                current_params.extend([mp['amp_est'], mp['cen_est'], mp['sigma_est']])
            for rp in st.session_state.app_state.residual_peaks:
                if rp.get('selected', True):
                    current_params.extend([rp['amp_est'], rp['cen_est'], rp['sigma_est']])
        
        # Apply pending remove
        if st.session_state.app_state.pending_remove is not None:
            remove_id = st.session_state.app_state.pending_remove
            new_params = []
            for i, c in enumerate(self.components):
                if i != remove_id - 1:
                    new_params.extend([c['amp_norm'], c['cen_log'], c['sigma_log']])
            current_params = new_params
            st.session_state.app_state.pending_remove = None
        
        # Apply pending split
        if st.session_state.app_state.pending_split is not None:
            peak_id, split_position = st.session_state.app_state.pending_split
            peak = self.components[peak_id - 1]
            
            new_params = []
            for i, c in enumerate(self.components):
                if i == peak_id - 1:
                    amp1 = c['amp_norm'] * 0.6
                    amp2 = c['amp_norm'] * 0.4
                    
                    cen1 = split_position - c['sigma_log'] * 0.3
                    cen2 = split_position + c['sigma_log'] * 0.3
                    
                    cen1 = np.clip(cen1, np.min(self.x), np.max(self.x))
                    cen2 = np.clip(cen2, np.min(self.x), np.max(self.x))
                    
                    sigma1 = c['sigma_log'] * 0.7
                    sigma2 = c['sigma_log'] * 0.7
                    
                    new_params.extend([amp1, cen1, sigma1])
                    new_params.extend([amp2, cen2, sigma2])
                else:
                    new_params.extend([c['amp_norm'], c['cen_log'], c['sigma_log']])
            
            current_params = new_params
            st.session_state.app_state.pending_split = None
        
        # Perform fit
        return self.fit(
            initial_params=current_params,
            method=st.session_state.app_state.fitting_method,
            maxfev=st.session_state.app_state.max_nfev,
            fit_quality=fit_quality,
            last_popt=st.session_state.app_state.last_popt,
            progress_callback=progress_callback
        )
    
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
            
            line_dash = 'solid'
            if c.get('source') == 'manual':
                line_dash = 'dash'
            elif c.get('source') == 'residual':
                line_dash = 'dot'
            
            fig.add_trace(
                go.Scatter(x=self.x, y=c['y_norm'],
                          mode='lines',
                          name=f'Peak {c["id"]} ({c["fraction_percent"]:.1f}%) [{c.get("source", "auto")}]',
                          line=dict(color=rgb_color, width=1.5, dash=line_dash),
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
        
        # Bar chart of components with color coding by source
        centers = [f"Peak {c['id']}\n({c.get('source', 'auto')})" for c in self.components]
        fractions = [c['fraction_percent'] for c in self.components]
        
        bar_colors = []
        for c in self.components:
            if c.get('source') == 'auto':
                bar_colors.append('green')
            elif c.get('source') == 'manual':
                bar_colors.append('orange')
            else:
                bar_colors.append('blue')
        
        fig.add_trace(
            go.Bar(x=centers, y=fractions,
                  name='Fractions (%)',
                  marker_color=bar_colors,
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
    with st.expander("⚙️ Advanced Settings", expanded=False):
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
        
        st.session_state.app_state.fit_quality = st.selectbox(
            "Fit quality",
            options=['fast', 'balanced', 'precise'],
            index=1,
            help="fast: fewer iterations, precise: slower but more accurate"
        )
        
        st.session_state.app_state.max_nfev = st.number_input(
            "Max iterations",
            min_value=1000,
            max_value=100000,
            value=5000 if st.session_state.app_state.fit_quality == 'fast' else 
                  10000 if st.session_state.app_state.fit_quality == 'balanced' else 50000,
            step=1000
        )
        
        st.session_state.app_state.baseline_method = st.selectbox(
            "Baseline correction",
            options=['none', 'constant', 'linear', 'quadratic'],
            index=0,
            help="Remove background before fitting"
        )
        
        st.session_state.app_state.preview_mode = st.checkbox(
            "Preview mode (no fitting)",
            value=st.session_state.app_state.preview_mode,
            help="Show estimated peaks without performing optimization"
        )
        
        st.markdown("---")
        st.subheader("Data Smoothing")
        
        st.session_state.app_state.smoothing_enabled = st.checkbox(
            "Enable data smoothing",
            value=st.session_state.app_state.smoothing_enabled,
            help="Apply smoothing to reduce noise"
        )
        
        if st.session_state.app_state.smoothing_enabled:
            st.session_state.app_state.smoothing_window = st.slider(
                "Smoothing window size",
                min_value=5,
                max_value=51,
                value=st.session_state.app_state.smoothing_window,
                step=2,
                help="Larger window = stronger smoothing (odd numbers only)"
            )
            
            smoothing_method = st.selectbox(
                "Smoothing method",
                options=['savgol', 'gaussian', 'median'],
                index=0,
                help="savgol: Savitzky-Golay, gaussian: Gaussian filter, median: median filter"
            )
            
            if smoothing_method == 'savgol':
                st.session_state.app_state.smoothing_polyorder = st.slider(
                    "Polynomial order",
                    min_value=2,
                    max_value=5,
                    value=st.session_state.app_state.smoothing_polyorder,
                    step=1,
                    help="Higher order = more flexible fit"
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
    
    # Apply smoothing if enabled
    if st.session_state.app_state.smoothing_enabled:
        # Create deconvolver if not yet created
        if st.session_state.app_state.deconvolver is None:
            st.session_state.app_state.original_x = st.session_state.app_state.raw_x.copy()
            st.session_state.app_state.original_y = st.session_state.app_state.raw_y.copy()
            
            st.session_state.app_state.deconvolver = GaussianDeconvolver(
                st.session_state.app_state.raw_x,
                st.session_state.app_state.raw_y,
                use_log_x=st.session_state.app_state.use_log_x,
                use_log_y=st.session_state.app_state.use_log_y,
                clip_negative=st.session_state.app_state.clip_negative,
                show_warnings=st.session_state.app_state.show_warnings,
                baseline_method=st.session_state.app_state.baseline_method
            )
        
        # Apply smoothing to deconvolver data
        smoothing_method = 'savgol'
        st.session_state.app_state.deconvolver.apply_smoothing(
            method=smoothing_method,
            window=st.session_state.app_state.smoothing_window,
            polyorder=st.session_state.app_state.smoothing_polyorder
        )
    
    # Create deconvolver if not yet created
    if st.session_state.app_state.deconvolver is None:
        st.session_state.app_state.original_x = st.session_state.app_state.raw_x.copy()
        st.session_state.app_state.original_y = st.session_state.app_state.raw_y.copy()
        
        st.session_state.app_state.deconvolver = GaussianDeconvolver(
            st.session_state.app_state.raw_x,
            st.session_state.app_state.raw_y,
            use_log_x=st.session_state.app_state.use_log_x,
            use_log_y=st.session_state.app_state.use_log_y,
            clip_negative=st.session_state.app_state.clip_negative,
            show_warnings=st.session_state.app_state.show_warnings,
            baseline_method=st.session_state.app_state.baseline_method
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
                    smoothing_window = st.session_state.app_state.smoothing_window if st.session_state.app_state.smoothing_enabled else None
                    peaks, peak_info, initial_params, derivatives = st.session_state.app_state.deconvolver.auto_detect_peaks(
                        sensitivity=st.session_state.app_state.sensitivity,
                        min_distance=st.session_state.app_state.min_distance,
                        smoothing_window=smoothing_window
                    )
                st.session_state.app_state.peak_info = peak_info
                st.session_state.app_state.derivatives = derivatives
                st.session_state.app_state.initial_params = initial_params
                st.success(f"Found {len(peak_info)} peaks!")
        
        st.markdown("---")
        
        # Manual Peak Addition Section
        st.subheader("✋ Manual Peak Addition")
        st.info("Use the slider below to select position for a new peak")
        
        deconv = st.session_state.app_state.deconvolver
        
        # Get x-axis range for slider (in original scale for user convenience)
        x_min_display = np.min(deconv.x_sorted)
        x_max_display = np.max(deconv.x_sorted)
        
        # Slider for manual peak position (200 steps)
        manual_position = st.slider(
            "Select peak position:",
            min_value=float(x_min_display),
            max_value=float(x_max_display),
            value=float(x_min_display + (x_max_display - x_min_display) * 0.5),
            step=float((x_max_display - x_min_display) / 200),
            format="%.6e"
        )
        
        st.session_state.app_state.manual_peak_position = manual_position
        
        if st.button("➕ Add Peak at This Position", use_container_width=True):
            # Convert to log scale if needed
            if deconv.use_log_x:
                position_log = np.log10(manual_position) if manual_position > 0 else manual_position
            else:
                position_log = manual_position
            
            manual_peak = deconv.add_manual_peak(position_log)
            st.session_state.app_state.pending_manual_peaks.append(manual_peak)
            st.success(f"Manual peak added at x = {manual_position:.4e}")
            st.rerun()
        
        # Show current manual peaks
        if st.session_state.app_state.pending_manual_peaks:
            st.markdown("---")
            st.subheader("Manual Peaks Added")
            for i, mp in enumerate(st.session_state.app_state.pending_manual_peaks):
                st.write(f"{i+1}: x = {mp['x_linear']:.4e}, y = {mp['y_original']:.4e}")
            
            if st.button("🗑️ Clear All Manual Peaks", use_container_width=True):
                st.session_state.app_state.pending_manual_peaks = []
                st.rerun()
        
        st.markdown("---")
        
        # Residual-based Peak Detection Section
        if st.session_state.app_state.deconvolver and st.session_state.app_state.deconvolver.fit_y_norm is not None:
            st.subheader("📊 Residual-Based Peak Detection")
            st.info("Find peaks in residuals after initial fit")
            
            residual_sensitivity = st.slider(
                "Residual sensitivity:",
                min_value=0.1,
                max_value=2.0,
                value=0.5,
                step=0.1,
                help="Higher values detect only stronger residual peaks"
            )
            
            if st.button("🔍 Find Residual Peaks", use_container_width=True):
                with st.spinner("Analyzing residuals..."):
                    residual_peaks = deconv.detect_residual_peaks(
                        sensitivity=residual_sensitivity,
                        min_distance=st.session_state.app_state.min_distance
                    )
                st.session_state.app_state.residual_peaks = residual_peaks
                st.success(f"Found {len(residual_peaks)} peaks in residuals!")
                st.rerun()
            
            # Show residual peaks with selection
            if st.session_state.app_state.residual_peaks:
                st.subheader("Residual Peaks Found")
                for i, rp in enumerate(st.session_state.app_state.residual_peaks):
                    selected = st.checkbox(
                        f"Peak at x = {rp['x_linear']:.4e}, y = {rp['y_original']:.4e}",
                        value=rp.get('selected', True),
                        key=f"residual_peak_{i}"
                    )
                    st.session_state.app_state.residual_peaks[i]['selected'] = selected
                
                if st.button("✅ Add Selected Residual Peaks", use_container_width=True):
                    selected_indices = [i for i, rp in enumerate(st.session_state.app_state.residual_peaks) if rp.get('selected', True)]
                    st.session_state.app_state.pending_residual_peaks = selected_indices
                    st.success(f"Added {len(selected_indices)} residual peaks")
                    st.rerun()
                
                if st.button("🗑️ Clear Residual Peaks", use_container_width=True):
                    st.session_state.app_state.residual_peaks = []
                    st.session_state.app_state.pending_residual_peaks = []
                    st.rerun()
        
        st.markdown("---")
        
        # Confirm button
        if st.session_state.app_state.peak_info is not None or st.session_state.app_state.pending_manual_peaks:
            if st.button("✅ Confirm Peaks", use_container_width=True):
                with st.spinner("Preparing preview..."):
                    if st.session_state.app_state.preview_mode:
                        # Just store params for later
                        st.session_state.app_state.current_step = 4
                        st.rerun()
                    else:
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
                            fit_quality=st.session_state.app_state.fit_quality,
                            progress_callback=update_progress
                        )
                        
                        progress_bar.empty()
                        status_text.empty()
                        
                        if success:
                            # Store last popt for future use
                            st.session_state.app_state.last_popt = st.session_state.app_state.deconvolver.popt
                            st.session_state.app_state.current_step = 4
                            st.rerun()
                        else:
                            st.error("Fitting failed. Try adjusting parameters.")
    
    with col2:
        if (st.session_state.app_state.peak_info is not None or 
            st.session_state.app_state.pending_manual_peaks or 
            st.session_state.app_state.residual_peaks):
            
            st.subheader(f"Peaks found: {len(st.session_state.app_state.peak_info or [])} auto, "
                        f"{len(st.session_state.app_state.pending_manual_peaks)} manual, "
                        f"{len([rp for rp in st.session_state.app_state.residual_peaks if rp.get('selected', True)])} residual")
            
            dy, d2y, y_smooth = st.session_state.app_state.derivatives or (None, None, None)
            deconv = st.session_state.app_state.deconvolver
            plotter = SpectrumPlotter()
            
            # Create tabs for different plots
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Peaks", "📈 Derivatives", "🔍 Residual Analysis", "📋 Information"])
            
            with tab1:
                if dy is not None and d2y is not None and y_smooth is not None:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    plotter.plot_with_peaks(
                        deconv, 
                        st.session_state.app_state.peak_info or [], 
                        y_smooth,
                        title=f"Peak Detection - Auto: {len(st.session_state.app_state.peak_info or [])}, "
                              f"Manual: {len(st.session_state.app_state.pending_manual_peaks)}, "
                              f"Residual: {len([rp for rp in st.session_state.app_state.residual_peaks if rp.get('selected', True)])}",
                        ax=ax,
                        manual_peak_position=st.session_state.app_state.manual_peak_position,
                        pending_manual_peaks=st.session_state.app_state.pending_manual_peaks,
                        residual_peaks=st.session_state.app_state.residual_peaks
                    )
                    st.pyplot(fig)
                    plt.close()
            
            with tab2:
                if dy is not None and d2y is not None:
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
                if deconv.fit_y_norm is not None:
                    residuals = deconv.y_norm - deconv.fit_y_norm
                    fig, ax = plt.subplots(figsize=(10, 6))
                    plotter.plot_residual_analysis(
                        deconv.x, deconv.y_norm, deconv.fit_y_norm, residuals,
                        title="Residual Analysis - Peaks indicate missed components",
                        ax=ax
                    )
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.info("Perform initial fit first to see residual analysis.")
            
            with tab4:
                # Create a table with peak information in original units
                data = []
                # Auto peaks
                if st.session_state.app_state.peak_info:
                    for i, info in enumerate(st.session_state.app_state.peak_info):
                        data.append({
                            'Peak': f"A{i+1}",
                            'Source': 'Auto',
                            'X Center': f"{info['x_linear']:.4e}",
                            'Y Amplitude': f"{info['y_original']:.4e}",
                            'Est. Sigma': f"{info['sigma_est']:.4f}",
                        })
                
                # Manual peaks
                for i, mp in enumerate(st.session_state.app_state.pending_manual_peaks):
                    data.append({
                        'Peak': f"M{i+1}",
                        'Source': 'Manual',
                        'X Center': f"{mp['x_linear']:.4e}",
                        'Y Amplitude': f"{mp['y_original']:.4e}",
                        'Est. Sigma': f"{mp['sigma_est']:.4f}",
                    })
                
                # Residual peaks
                for i, rp in enumerate(st.session_state.app_state.residual_peaks):
                    if rp.get('selected', True):
                        data.append({
                            'Peak': f"R{i+1}",
                            'Source': 'Residual',
                            'X Center': f"{rp['x_linear']:.4e}",
                            'Y Amplitude': f"{rp['y_original']:.4e}",
                            'Est. Sigma': f"{rp['sigma_est']:.4f}",
                        })
                
                if data:
                    df = pd.DataFrame(data)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No peaks detected. Adjust sensitivity or add peaks manually.")
                
                # Show peak detection statistics
                st.markdown("---")
                st.subheader("Detection Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    total_peaks = (len(st.session_state.app_state.peak_info or []) + 
                                  len(st.session_state.app_state.pending_manual_peaks) + 
                                  len([rp for rp in st.session_state.app_state.residual_peaks if rp.get('selected', True)]))
                    st.metric("Total Peaks", total_peaks)
                with col2:
                    st.metric("X Range", f"{np.min(deconv.x_sorted):.2e} - {np.max(deconv.x_sorted):.2e}")
                with col3:
                    st.metric("Y Range", f"{np.min(deconv.y_sorted):.2e} - {np.max(deconv.y_sorted):.2e}")
                
                if st.session_state.app_state.smoothing_enabled:
                    st.info(f"🔹 Smoothing applied: window={st.session_state.app_state.smoothing_window}")


# ==================== STEP 4: EDITING ====================

elif st.session_state.app_state.current_step == 4:
    st.header("Step 4: Peak Editing")
    
    if st.session_state.app_state.deconvolver:
        deconv = st.session_state.app_state.deconvolver
        plotter = SpectrumPlotter()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Peak Management")
            
            # Show pending operations
            if st.session_state.app_state.pending_remove is not None:
                st.warning(f"Pending: Remove Peak {st.session_state.app_state.pending_remove}")
            if st.session_state.app_state.pending_split is not None:
                st.warning(f"Pending: Split Peak {st.session_state.app_state.pending_split[0]}")
            
            # Display quality metrics if available
            if deconv.quality_metrics:
                metrics = deconv.quality_metrics
                st.info(f"R² = {metrics.get('R²', 0):.4f} | RMSE = {metrics.get('RMSE', 0):.2e}")
            
            st.markdown("---")
            
            # Preview mode indicator
            if st.session_state.app_state.preview_mode:
                st.info("🔍 PREVIEW MODE - No fitting performed")
            
            # Peak selection (only if components exist)
            if deconv.components:
                peak_options = {f"Peak {c['id']}: center = {c['cen_linear']:.2e}, fraction = {c['fraction_percent']:.1f}% [{c.get('source', 'auto')}]": c['id'] 
                               for c in deconv.components}
                
                selected_peak = st.selectbox(
                    "Select peak for editing:",
                    options=list(peak_options.keys())
                )
                
                if selected_peak:
                    peak_id = peak_options[selected_peak]
                    
                    # Split position slider
                    peak = deconv.components[peak_id - 1]
                    min_x = np.min(deconv.x)
                    max_x = np.max(deconv.x)
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
                            if deconv.split_peak(peak_id, split_position):
                                st.success(f"Peak {peak_id} marked for splitting")
                                st.rerun()
                    
                    with col_b:
                        if st.button("🗑️ Remove Peak", use_container_width=True):
                            if deconv.remove_peak(peak_id):
                                st.success(f"Peak {peak_id} marked for removal")
                                st.rerun()
            else:
                st.warning("No peaks to edit. Run peak detection first.")
            
            st.markdown("---")
            
            # Apply changes button
            if st.button("🔄 Apply Changes and Recalculate", type="primary", use_container_width=True):
                with st.spinner("Applying changes and recalculating..."):
                    # Create progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def update_progress(progress, message):
                        progress_bar.progress(progress)
                        status_text.text(message)
                    
                    if st.session_state.app_state.preview_mode:
                        # In preview mode, just show preview
                        preview_fit = deconv.preview_fit()
                        if preview_fit is not None:
                            st.session_state.app_state.preview_fit = preview_fit
                            st.success("Preview updated")
                        progress_bar.empty()
                        status_text.empty()
                        st.rerun()
                    else:
                        # Apply pending operations and fit
                        success = deconv.apply_pending_operations(
                            fit_quality=st.session_state.app_state.fit_quality,
                            progress_callback=update_progress
                        )
                        
                        progress_bar.empty()
                        status_text.empty()
                        
                        if success:
                            # Store last popt for future use
                            st.session_state.app_state.last_popt = deconv.popt
                            st.success("Recalculation complete!")
                            st.rerun()
                        else:
                            st.error("Recalculation failed")
            
            # Reset pending operations
            if st.button("🔄 Clear Pending Operations", use_container_width=True):
                st.session_state.app_state.pending_remove = None
                st.session_state.app_state.pending_split = None
                st.rerun()
            
            st.markdown("---")
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("⬅️ Back", use_container_width=True):
                    st.session_state.app_state.current_step = 3
                    st.rerun()
            with col_b:
                if st.button("✅ Finish Editing", use_container_width=True):
                    st.session_state.app_state.current_step = 5
                    st.rerun()
        
        with col2:
            st.subheader("Current Deconvolution")
            
            if st.session_state.app_state.preview_mode and hasattr(st.session_state.app_state, 'preview_fit'):
                # Show preview
                fig, ax = plt.subplots(figsize=(10, 6))
                plotter.plot_deconvolution_result(
                    deconv,
                    show_components=True,
                    show_baseline=True,
                    title="Preview (no fit performed)",
                    ax=ax,
                    preview_mode=True,
                    preview_fit=st.session_state.app_state.preview_fit
                )
                st.pyplot(fig)
                plt.close()
            elif deconv.components:
                # Show actual fit
                fig, ax = plt.subplots(figsize=(10, 6))
                plotter.plot_deconvolution_result(
                    deconv,
                    show_components=True,
                    show_baseline=True,
                    title="Current Deconvolution",
                    ax=ax
                )
                st.pyplot(fig)
                plt.close()
            else:
                st.info("No components to display. Run peak detection first.")


# ==================== STEP 5: RESULTS ====================

elif st.session_state.app_state.current_step == 5:
    st.header("Step 5: Results")
    
    if st.session_state.app_state.deconvolver and st.session_state.app_state.deconvolver.components:
        deconv = st.session_state.app_state.deconvolver
        
        # Back button at the top
        col_back, _ = st.columns([1, 5])
        with col_back:
            if st.button("⬅️ Back to Editing", use_container_width=True):
                st.session_state.app_state.current_step = 4
                st.rerun()
        
        st.markdown("---")
        
        # Create tabs for results
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Graphs", "📈 Normalized View", "📋 Table", "📥 Export"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Deconvolution Result (Original Scale)")
                
                plotter = SpectrumPlotter()
                fig, ax = plt.subplots(figsize=(10, 6))
                
                plotter.plot_deconvolution_result(
                    deconv,
                    show_components=True,
                    show_baseline=True,
                    title="Deconvolution Result - Original Scale",
                    ax=ax
                )
                
                st.pyplot(fig)
                plt.close()
            
            with col2:
                st.subheader("Area Distribution Analysis")
                
                # Create a figure with two subplots
                fig = plt.figure(figsize=(12, 10))
                
                # 1. Bar chart of areas (top left)
                ax1 = plt.subplot(2, 2, 1)
                peaks = [f'Peak {c["id"]}\n({c.get("source", "auto")})' for c in deconv.components]
                areas = [c['area'] for c in deconv.components]
                fractions = [c['fraction_percent'] for c in deconv.components]
                
                # Color coding by source
                colors = []
                for c in deconv.components:
                    if c.get('source') == 'auto':
                        colors.append('green')
                    elif c.get('source') == 'manual':
                        colors.append('orange')
                    else:
                        colors.append('blue')
                
                bars1 = ax1.bar(peaks, areas, color=colors, edgecolor='black', alpha=0.7)
                ax1.set_xlabel('Peak', fontweight='bold')
                ax1.set_ylabel('Area', fontweight='bold')
                ax1.set_title('Peak Areas', fontweight='bold')
                ax1.tick_params(axis='x', rotation=45)
                ax1.grid(True, alpha=0.3, axis='y')
                
                # Add value labels on bars
                for bar, area in zip(bars1, areas):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{area:.2e}',
                            ha='center', va='bottom', fontsize=8, rotation=0)
                
                # 2. Bar chart of fractions (top right)
                ax2 = plt.subplot(2, 2, 2)
                bars2 = ax2.bar(peaks, fractions, color=colors, edgecolor='black', alpha=0.7)
                ax2.set_xlabel('Peak', fontweight='bold')
                ax2.set_ylabel('Fraction (%)', fontweight='bold')
                ax2.set_title('Peak Fractions', fontweight='bold')
                ax2.tick_params(axis='x', rotation=45)
                ax2.grid(True, alpha=0.3, axis='y')
                
                # Add value labels on bars
                for bar, frac in zip(bars2, fractions):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{frac:.1f}%',
                            ha='center', va='bottom', fontsize=8)
                
                # 3. Pie chart (bottom left)
                ax3 = plt.subplot(2, 2, 3)
                wedges, texts, autotexts = ax3.pie(fractions, labels=peaks, autopct='%1.1f%%',
                       colors=colors, startangle=90,
                       textprops={'fontweight': 'bold'})
                ax3.set_title('Area Distribution - Pie Chart', fontweight='bold')
                
                # 4. Horizontal bar chart (bottom right)
                ax4 = plt.subplot(2, 2, 4)
                y_pos = np.arange(len(peaks))
                bars4 = ax4.barh(y_pos, fractions, color=colors, edgecolor='black', alpha=0.7)
                ax4.set_yticks(y_pos)
                ax4.set_yticklabels(peaks)
                ax4.set_xlabel('Fraction (%)', fontweight='bold')
                ax4.set_title('Peak Fractions - Horizontal View', fontweight='bold')
                ax4.grid(True, alpha=0.3, axis='x')
                
                # Add value labels
                for bar, frac in zip(bars4, fractions):
                    width = bar.get_width()
                    ax4.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                            f'{frac:.1f}%',
                            ha='left', va='center', fontsize=8)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            # Summary statistics row
            st.markdown("---")
            st.subheader("Summary Statistics")
            
            col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
            
            with col_sum1:
                total_area = sum([c['area'] for c in deconv.components])
                st.metric("Total Area", f"{total_area:.4e}")
            
            with col_sum2:
                n_peaks = len(deconv.components)
                st.metric("Number of Peaks", f"{n_peaks}")
            
            with col_sum3:
                max_fraction_peak = max(enumerate(deconv.components), key=lambda x: x[1]['fraction'])
                st.metric("Dominant Peak", f"Peak {max_fraction_peak[1]['id']} ({max_fraction_peak[1]['fraction_percent']:.1f}%)")
            
            with col_sum4:
                avg_area = total_area / n_peaks if n_peaks > 0 else 0
                st.metric("Average Area", f"{avg_area:.4e}")
        
        with tab2:
            st.subheader("Normalized View (Max Peak Intensity = 1)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Find maximum amplitude for normalization
                max_amp = max([c['amp'] for c in deconv.components])
                
                # Create normalized plot
                fig_norm, ax_norm = plt.subplots(figsize=(10, 6))
                
                # Apply scales
                if deconv.use_log_x:
                    ax_norm.set_xscale('log')
                
                # Generate dense x for smooth curves
                if deconv.use_log_x:
                    x_min = np.maximum(np.min(deconv.x_linear[deconv.x_linear>0]), np.finfo(float).eps)
                    x_max = np.max(deconv.x_linear)
                    x_dense = np.logspace(np.log10(x_min), np.log10(x_max), 2000)
                    x_dense_log = np.log10(x_dense)
                else:
                    x_dense = np.linspace(np.min(deconv.x_linear), np.max(deconv.x_linear), 2000)
                    x_dense_log = x_dense
                
                # Plot normalized components with color coding
                colors = []
                for c in deconv.components:
                    if c.get('source') == 'auto':
                        colors.append('green')
                    elif c.get('source') == 'manual':
                        colors.append('orange')
                    else:
                        colors.append('blue')
                
                for c, color in zip(deconv.components, colors):
                    y_component_norm = GaussianModel.gaussian(x_dense_log, c['amp_norm'], 
                                                            c['cen_log'], c['sigma_log']) 
                    y_component_norm = y_component_norm * deconv.y_max / max_amp
                    
                    # Fill under Gaussian
                    ax_norm.fill_between(x_dense, 0, y_component_norm, 
                                        color=color, alpha=0.3, linewidth=0)
                    
                    # Plot line
                    line_style = '-'
                    if c.get('source') == 'manual':
                        line_style = '--'
                    elif c.get('source') == 'residual':
                        line_style = ':'
                    
                    ax_norm.plot(x_dense, y_component_norm, line_style, color=color, linewidth=2,
                               label=f'Peak {c["id"]}: {c["fraction_percent"]:.1f}% ({c.get("source", "auto")})', zorder=2)
                
                # Plot normalized total fit
                if deconv.baseline_method != 'none' and deconv.baseline_params:
                    n_peaks = len(deconv.components)
                    peak_params = []
                    for c in deconv.components:
                        peak_params.extend([c['amp_norm'], c['cen_log'], c['sigma_log']])
                    
                    y_total_norm = GaussianModel.multi_gaussian_with_baseline(
                        x_dense_log, n_peaks, peak_params, 
                        deconv.baseline_params, deconv.baseline_method
                    ) * deconv.y_max / max_amp
                else:
                    y_total_norm = GaussianModel.multi_gaussian(x_dense_log, *deconv.popt) * deconv.y_max / max_amp
                
                ax_norm.plot(x_dense, y_total_norm, 'r--', linewidth=2, label='Total Fit', zorder=3)
                
                # Plot original data (normalized)
                y_original_norm = deconv.y_original / max_amp
                ax_norm.scatter(deconv.x_linear, y_original_norm, 
                               s=10, alpha=0.5, color='black', label='Data', zorder=1)
                
                # Labels and title
                x_label = 'X' + (' (log scale)' if deconv.use_log_x else '')
                y_label = 'Normalized Intensity'
                ax_norm.set_xlabel(x_label, fontsize=12, fontweight='bold')
                ax_norm.set_ylabel(y_label, fontsize=12, fontweight='bold')
                ax_norm.set_title('Deconvolution Result - Normalized to Max Peak = 1', fontsize=14, fontweight='bold')
                
                # Add quality metrics
                if deconv.quality_metrics:
                    metrics_text = f"R² = {deconv.quality_metrics.get('R²', 0):.4f}\n"
                    metrics_text += f"RMSE = {deconv.quality_metrics.get('RMSE', 0):.2e}"
                    ax_norm.text(0.02, 0.98, metrics_text, transform=ax_norm.transAxes,
                                fontsize=10, verticalalignment='top',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                
                ax_norm.legend(loc='upper right', fontsize=8, frameon=True, edgecolor='black')
                ax_norm.grid(True, alpha=0.3, linestyle='--')
                
                # Scientific styling
                ax_norm.spines['top'].set_visible(True)
                ax_norm.spines['right'].set_visible(True)
                ax_norm.spines['bottom'].set_linewidth(1)
                ax_norm.spines['left'].set_linewidth(1)
                ax_norm.spines['top'].set_linewidth(1)
                ax_norm.spines['right'].set_linewidth(1)
                ax_norm.tick_params(direction='out', length=4, width=1)
                
                st.pyplot(fig_norm)
                plt.close()
            
            with col2:
                st.subheader("Normalized Components Comparison")
                
                fig_comp_norm, ax_comp_norm = plt.subplots(figsize=(10, 6))
                
                # Plot all normalized components on the same axes without fill
                colors = []
                for c in deconv.components:
                    if c.get('source') == 'auto':
                        colors.append('green')
                    elif c.get('source') == 'manual':
                        colors.append('orange')
                    else:
                        colors.append('blue')
                
                for c, color in zip(deconv.components, colors):
                    y_component_norm = GaussianModel.gaussian(x_dense_log, c['amp_norm'], 
                                                            c['cen_log'], c['sigma_log']) 
                    y_component_norm = y_component_norm * deconv.y_max / max_amp
                    
                    line_style = '-'
                    if c.get('source') == 'manual':
                        line_style = '--'
                    elif c.get('source') == 'residual':
                        line_style = ':'
                    
                    ax_comp_norm.plot(x_dense, y_component_norm, line_style, color=color, linewidth=2,
                                    label=f'Peak {c["id"]} (center: {c["cen_linear"]:.2e}) [{c.get("source", "auto")}]')
                
                # Add vertical lines at peak centers
                for c in deconv.components:
                    ax_comp_norm.axvline(x=c['cen_linear'], color='gray', linestyle=':', alpha=0.5)
                
                ax_comp_norm.set_xlabel(x_label, fontsize=12, fontweight='bold')
                ax_comp_norm.set_ylabel('Normalized Intensity', fontsize=12, fontweight='bold')
                ax_comp_norm.set_title('Normalized Components Overlay', fontsize=14, fontweight='bold')
                ax_comp_norm.legend(loc='upper right', fontsize=8, frameon=True, edgecolor='black')
                ax_comp_norm.grid(True, alpha=0.3, linestyle='--')
                
                if deconv.use_log_x:
                    ax_comp_norm.set_xscale('log')
                
                st.pyplot(fig_comp_norm)
                plt.close()
            
            # Table of normalized values
            st.subheader("Normalized Parameters")
            norm_data = []
            for c in deconv.components:
                norm_data.append({
                    'Peak': c['id'],
                    'Source': c.get('source', 'auto'),
                    'Center': f"{c['cen_linear']:.4e}",
                    'Normalized Amplitude': f"{c['amp'] / max_amp:.4f}",
                    'Original Amplitude': f"{c['amp']:.4e}",
                    'Fraction (%)': f"{c['fraction_percent']:.2f}"
                })
            
            df_norm = pd.DataFrame(norm_data)
            st.dataframe(df_norm, use_container_width=True)
        
        with tab3:
            st.subheader("Results Table - Complete Dataset")
            
            # Main results table
            data = []
            for c in deconv.components:
                data.append({
                    'Peak ID': c['id'],
                    'Source': c.get('source', 'auto'),
                    'Center': c['cen_linear'],
                    'Center (log)': c['cen_log'],
                    'Amplitude': c['amp'],
                    'Amplitude (norm)': c['amp_norm'],
                    'Sigma (log)': c['sigma_log'],
                    'FWHM': c['fwhm'],
                    'Area': c['area'],
                    'Fraction': c['fraction'],
                    'Fraction (%)': c['fraction_percent']
                })
            
            df = pd.DataFrame(data)
            
            # Format for display
            display_df = df.copy()
            for col in ['Center', 'Amplitude', 'Area']:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.4e}")
            display_df['Center (log)'] = display_df['Center (log)'].apply(lambda x: f"{x:.4f}")
            display_df['Amplitude (norm)'] = display_df['Amplitude (norm)'].apply(lambda x: f"{x:.4f}")
            display_df['Sigma (log)'] = display_df['Sigma (log)'].apply(lambda x: f"{x:.4f}")
            display_df['FWHM'] = display_df['FWHM'].apply(lambda x: f"{x:.4f}")
            display_df['Fraction (%)'] = display_df['Fraction (%)'].apply(lambda x: f"{x:.2f}")
            
            st.dataframe(display_df, use_container_width=True)
            
            # Download button for raw data
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Raw Data (CSV)",
                data=csv,
                file_name=f"deconvolution_peaks_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            st.markdown("---")
            
            # Baseline info if used
            if deconv.baseline_method != 'none' and deconv.baseline_params:
                st.subheader("Baseline Parameters")
                baseline_df = pd.DataFrame([{
                    'Method': deconv.baseline_method,
                    'Parameters': ', '.join([f"{p:.4e}" for p in deconv.baseline_params])
                }])
                st.dataframe(baseline_df, use_container_width=True)
            
            st.markdown("---")
            
            # Quality metrics in columns
            st.subheader("Quality Metrics")
            metrics = deconv.quality_metrics
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("R²", f"{metrics.get('R²', 0):.6f}")
            with col_m2:
                st.metric("AIC", f"{metrics.get('AIC', 0):.2f}")
            with col_m3:
                st.metric("BIC", f"{metrics.get('BIC', 0):.2f}")
            with col_m4:
                st.metric("RMSE", f"{metrics.get('RMSE', 0):.2e}")
            
            col_m5, col_m6, col_m7, col_m8 = st.columns(4)
            with col_m5:
                st.metric("χ²", f"{metrics.get('χ²', 0):.2e}")
            with col_m6:
                st.metric("Max Error", f"{metrics.get('Max Error', 0):.2e}")
            with col_m7:
                st.metric("N Parameters", len(deconv.popt) if deconv.popt is not None else 0)
            with col_m8:
                st.metric("N Points", len(deconv.x_linear))
        
        with tab4:
            st.subheader("Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Export peaks to CSV
                if st.button("📥 Export Peaks to CSV", use_container_width=True):
                    df_peaks = pd.DataFrame([{
                        'Peak_ID': c['id'],
                        'Source': c.get('source', 'auto'),
                        'Center': c['cen_linear'],
                        'Center_log': c['cen_log'],
                        'Amplitude': c['amp'],
                        'Amplitude_norm': c['amp_norm'],
                        'Sigma_log': c['sigma_log'],
                        'FWHM': c['fwhm'],
                        'Area': c['area'],
                        'Fraction': c['fraction'],
                        'Fraction_Percent': c['fraction_percent']
                    } for c in deconv.components])
                    
                    csv_peaks = df_peaks.to_csv(index=False)
                    
                    st.download_button(
                        label="Download Peaks CSV",
                        data=csv_peaks,
                        file_name=f"deconvolution_peaks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                # Export fitting data
                if 'Residuals' in deconv.quality_metrics:
                    # Reconstruct fit with baseline if needed
                    if deconv.baseline_method != 'none' and deconv.baseline_params:
                        n_peaks = len(deconv.components)
                        peak_params = []
                        for c in deconv.components:
                            peak_params.extend([c['amp_norm'], c['cen_log'], c['sigma_log']])
                        
                        fit_y_norm = GaussianModel.multi_gaussian_with_baseline(
                            deconv.x, n_peaks, peak_params, 
                            deconv.baseline_params, deconv.baseline_method
                        )
                    else:
                        fit_y_norm = deconv.fit_y_norm
                    
                    # Generate normalized fit data
                    max_amp = max([c['amp'] for c in deconv.components])
                    
                    df_fit = pd.DataFrame({
                        'X_original': deconv.x_linear,
                        'Y_original': deconv.y_original,
                        'Y_fit': fit_y_norm * deconv.y_max,
                        'Y_fit_normalized': fit_y_norm * deconv.y_max / max_amp,
                        'Residuals': deconv.quality_metrics['Residuals'] * deconv.y_max,
                        'Residuals_normalized': deconv.quality_metrics['Residuals'] * deconv.y_max / max_amp
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
                if st.button("📄 Export Detailed Report", use_container_width=True):
                    max_amp = max([c['amp'] for c in deconv.components])
                    
                    report = f"""GAUSSIAN DECONVOLUTION REPORT
{"="*80}

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Number of points: {len(deconv.x_linear)}
X range: [{deconv.x_linear[0]:.2e}, {deconv.x_linear[-1]:.2e}]
Logarithmic X scale: {deconv.use_log_x}
Baseline method: {deconv.baseline_method}
Smoothing applied: {st.session_state.app_state.smoothing_enabled}
Smoothing window: {st.session_state.app_state.smoothing_window if st.session_state.app_state.smoothing_enabled else 'N/A'}

QUALITY METRICS:
{"-"*40}
R²: {deconv.quality_metrics.get('R²', 0):.6f}
AIC: {deconv.quality_metrics.get('AIC', 0):.2f}
BIC: {deconv.quality_metrics.get('BIC', 0):.2f}
χ²: {deconv.quality_metrics.get('χ²', 0):.2e}
RMSE: {deconv.quality_metrics.get('RMSE', 0):.2e}

"""
                    if deconv.baseline_method != 'none' and deconv.baseline_params:
                        report += f"""BASELINE PARAMETERS:
{"-"*40}
Method: {deconv.baseline_method}
Parameters: {', '.join([f'{p:.4e}' for p in deconv.baseline_params])}

"""
                    
                    report += f"""COMPONENTS (ORIGINAL SCALE):
{"-"*80}
ID    Source    Center          Amplitude       FWHM        Area           Fraction(%)
{"-"*80}"""
                    
                    for c in deconv.components:
                        report += f"\n{c['id']:<4} {c.get('source', 'auto'):<8} {c['cen_linear']:<15.4e} {c['amp']:<15.4e} {c['fwhm']:<12.4f} {c['area']:<15.4e} {c['fraction_percent']:<10.2f}"
                    
                    report += f"""

COMPONENTS (NORMALIZED TO MAX PEAK = 1):
{"-"*80}
ID    Source    Center          Norm. Amplitude    Original Amplitude    Fraction(%)
{"-"*80}"""
                    
                    for c in deconv.components:
                        norm_amp = c['amp'] / max_amp
                        report += f"\n{c['id']:<4} {c.get('source', 'auto'):<8} {c['cen_linear']:<15.4e} {norm_amp:<18.4f} {c['amp']:<20.4e} {c['fraction_percent']:<10.2f}"
                    
                    report += f"""
{"="*80}
Total area: {deconv.total_area:.6e}
Maximum amplitude (for normalization): {max_amp:.6e}
{"="*80}"""
                    
                    st.download_button(
                        label="Download Report",
                        data=report,
                        file_name=f"deconvolution_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
            
            st.markdown("---")
            
            # Export figures
            st.subheader("Export Figures")
            
            col_fig1, col_fig2 = st.columns(2)
            
            with col_fig1:
                if st.button("📊 Save Original Scale Figure", use_container_width=True):
                    fig, ax = plt.subplots(figsize=(12, 8))
                    plotter.plot_deconvolution_result(
                        deconv,
                        show_components=True,
                        show_baseline=True,
                        title="Deconvolution Result - Original Scale",
                        ax=ax
                    )
                    
                    # Save to buffer
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                    buf.seek(0)
                    plt.close(fig)
                    
                    st.download_button(
                        label="Download PNG",
                        data=buf,
                        file_name=f"deconvolution_original_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png",
                        use_container_width=True
                    )
            
            with col_fig2:
                if st.button("📈 Save Normalized Scale Figure", use_container_width=True):
                    fig_norm, ax_norm = plt.subplots(figsize=(12, 8))
                    
                    max_amp = max([c['amp'] for c in deconv.components])
                    
                    if deconv.use_log_x:
                        ax_norm.set_xscale('log')
                    
                    x_dense = np.linspace(np.min(deconv.x_linear), np.max(deconv.x_linear), 2000)
                    x_dense_log = x_dense if not deconv.use_log_x else np.log10(x_dense)
                    
                    colors = []
                    for c in deconv.components:
                        if c.get('source') == 'auto':
                            colors.append('green')
                        elif c.get('source') == 'manual':
                            colors.append('orange')
                        else:
                            colors.append('blue')
                    
                    for c, color in zip(deconv.components, colors):
                        y_component_norm = GaussianModel.gaussian(x_dense_log, c['amp_norm'], 
                                                                c['cen_log'], c['sigma_log']) 
                        y_component_norm = y_component_norm * deconv.y_max / max_amp
                        
                        line_style = '-'
                        if c.get('source') == 'manual':
                            line_style = '--'
                        elif c.get('source') == 'residual':
                            line_style = ':'
                        
                        ax_norm.fill_between(x_dense, 0, y_component_norm, 
                                            color=color, alpha=0.3, linewidth=0)
                        ax_norm.plot(x_dense, y_component_norm, line_style, color=color, linewidth=2,
                                   label=f'Peak {c["id"]}: {c["fraction_percent"]:.1f}%')
                    
                    y_original_norm = deconv.y_original / max_amp
                    ax_norm.scatter(deconv.x_linear, y_original_norm, 
                                   s=10, alpha=0.5, color='black', label='Data')
                    
                    ax_norm.set_xlabel('X' + (' (log scale)' if deconv.use_log_x else ''))
                    ax_norm.set_ylabel('Normalized Intensity')
                    ax_norm.set_title('Deconvolution Result - Normalized')
                    ax_norm.legend()
                    ax_norm.grid(True, alpha=0.3)
                    
                    buf = io.BytesIO()
                    fig_norm.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                    buf.seek(0)
                    plt.close(fig_norm)
                    
                    st.download_button(
                        label="Download PNG",
                        data=buf,
                        file_name=f"deconvolution_normalized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png",
                        use_container_width=True
                    )
            
            st.markdown("---")
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🔄 New Analysis", use_container_width=True):
                    st.session_state.app_state = AppState()
                    st.rerun()
