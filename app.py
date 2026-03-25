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
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

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
    manual_peaks: List[float] = field(default_factory=list)
    residual_peaks: List[Dict] = field(default_factory=list)
    smoothing_enabled: bool = False
    smoothing_method: str = 'savgol'
    smoothing_window: int = 11
    smoothing_polyorder: int = 3
    show_smoothing_preview: bool = False
    manual_peak_position: Optional[float] = None
    manual_peak_amplitude: Optional[float] = None
    manual_peak_sigma: Optional[float] = None
    residual_analysis_done: bool = False

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
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.labelweight': 'bold',
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.0,
    'axes.grid': False,
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
    'legend.fontsize': 10,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': 'black',
    'legend.fancybox': False,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'figure.facecolor': 'white',
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'errorbar.capsize': 3,
})

st.title("📊 Gaussian Deconvolution of Spectral Data")
st.markdown("---")

# ==================== CLASSES ====================

class DataSmoother:
    """Data smoothing with multiple methods"""
    
    @staticmethod
    def savgol_smooth(y, window_length=11, polyorder=3):
        """Savitzky-Golay smoothing"""
        if len(y) < window_length:
            window_length = len(y) if len(y) % 2 == 1 else len(y) - 1
        if window_length < polyorder + 2:
            return y
        try:
            return savgol_filter(y, window_length, polyorder)
        except Exception:
            return y
    
    @staticmethod
    def gaussian_smooth(y, sigma=2.0):
        """Gaussian smoothing"""
        if len(y) < 5:
            return y
        try:
            return gaussian_filter1d(y, sigma)
        except Exception:
            return y
    
    @staticmethod
    def median_smooth(y, window_size=5):
        """Median smoothing for outlier removal"""
        from scipy.signal import medfilt
        if len(y) < window_size:
            return y
        try:
            return medfilt(y, window_size)
        except Exception:
            return y
    
    @staticmethod
    def adaptive_smooth(y, noise_level=None):
        """Adaptive smoothing based on estimated noise"""
        if noise_level is None:
            # Estimate noise from high-frequency components
            if len(y) > 10:
                diff = np.diff(y)
                noise_level = np.std(diff) / np.mean(np.abs(y)) if np.mean(np.abs(y)) > 0 else 0.1
            else:
                noise_level = 0.1
        
        if noise_level > 0.05:
            # High noise - use stronger smoothing
            window = max(15, len(y) // 20)
            if window % 2 == 0:
                window += 1
            return DataSmoother.savgol_smooth(y, window, 2)
        elif noise_level > 0.02:
            # Medium noise
            window = max(9, len(y) // 30)
            if window % 2 == 0:
                window += 1
            return DataSmoother.savgol_smooth(y, window, 3)
        else:
            # Low noise - minimal smoothing
            window = min(7, len(y) // 10)
            if window % 2 == 0:
                window += 1
            return DataSmoother.savgol_smooth(y, window, 3)

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
        sort_idx = np.argsort(x_linear)
        x_sorted = x_linear[sort_idx]
        y_sorted = y_original[sort_idx]
        
        if self.clip_negative:
            negative_mask = y_sorted < 0
            self.clipped_points = np.sum(negative_mask)
            if self.clipped_points > 0 and self.show_warnings:
                warnings.warn(f"Clipped {self.clipped_points} negative values to 0")
            y_for_fitting = np.maximum(y_sorted, 0)
        else:
            y_for_fitting = y_sorted
        
        eps = np.finfo(float).eps
        
        if use_log_y and np.any(y_for_fitting < eps * 100):
            self.small_values_warning = True
            if self.show_warnings:
                warnings.warn("Very small Y values detected. Log transformation may cause artifacts.")
        
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
    
    def apply_smoothing(self, x, y, method='savgol', window=11, polyorder=3):
        """Apply smoothing to data"""
        if method == 'savgol':
            return DataSmoother.savgol_smooth(y, window, polyorder)
        elif method == 'gaussian':
            sigma = window / 5.0
            return DataSmoother.gaussian_smooth(y, sigma)
        elif method == 'median':
            return DataSmoother.median_smooth(y, window)
        elif method == 'adaptive':
            return DataSmoother.adaptive_smooth(y)
        else:
            return y


class DerivativeAnalyzer:
    """Analysis of first and second derivatives for peak detection"""
    
    @staticmethod
    def calculate_derivatives(x, y, window_length=11, polyorder=3):
        """Calculate smoothed derivatives with fallback for small datasets"""
        if len(x) < window_length:
            window_length = len(x) if len(x) % 2 == 1 else len(x) - 1
        
        if window_length < polyorder + 2:
            dy = np.gradient(y, x)
            d2y = np.gradient(dy, x)
            return dy, d2y, y
        
        try:
            y_smooth = savgol_filter(y, window_length, polyorder)
            dy = savgol_filter(y, window_length, polyorder, deriv=1, delta=np.mean(np.diff(x)))
            d2y = savgol_filter(y, window_length, polyorder, deriv=2, delta=np.mean(np.diff(x)))
        except Exception as e:
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
        y_peaks = np.zeros_like(x, dtype=float)
        for i in range(n_peaks):
            amp = peak_params[3*i]
            cen = peak_params[3*i + 1]
            sigma = abs(peak_params[3*i + 2])
            y_peaks += GaussianModel.gaussian(x, amp, cen, sigma)
        
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
            left_min = peak_idx
            right_min = peak_idx
            
            for i in range(peak_idx - 1, 0, -1):
                if y[i] < y[i-1] and y[i] < y[i+1]:
                    left_min = i
                    break
            
            for i in range(peak_idx + 1, len(y) - 1):
                if y[i] < y[i-1] and y[i] < y[i+1]:
                    right_min = i
                    break
            
            width = (right_min - left_min) * np.mean(np.diff(x))
            sigma = width / 3.0
            return max(sigma, 0.01 * (np.max(x) - np.min(x)) / 10)
    
    @staticmethod
    def estimate_amplitude_at_position(x, y, position):
        """Estimate amplitude at a given position"""
        idx = np.argmin(np.abs(x - position))
        return y[idx]
    
    @staticmethod
    def estimate_sigma_at_position(x, y, position):
        """Estimate sigma at a given position using local minima"""
        idx = np.argmin(np.abs(x - position))
        left_min = idx
        right_min = idx
        
        for i in range(idx - 1, max(0, idx - 50), -1):
            if i > 0 and i < len(y) - 1:
                if y[i] < y[i-1] and y[i] < y[i+1]:
                    left_min = i
                    break
        
        for i in range(idx + 1, min(len(y), idx + 50)):
            if i > 0 and i < len(y) - 1:
                if y[i] < y[i-1] and y[i] < y[i+1]:
                    right_min = i
                    break
        
        width = (right_min - left_min) * np.mean(np.diff(x))
        sigma = width / 3.0
        return max(sigma, 0.01 * (np.max(x) - np.min(x)) / 20)


class FitQualityAnalyzer:
    """Fit quality analysis"""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred, n_params):
        """Calculate quality metrics"""
        residuals = y_true - y_pred
        n = len(y_true)
        
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        rss = ss_res
        aic = n * np.log(rss/n) + 2 * n_params if rss > 0 else -np.inf
        bic = n * np.log(rss/n) + n_params * np.log(n) if rss > 0 else -np.inf
        
        chi_squared = rss / (n - n_params) if n > n_params else np.inf
        
        max_error = np.max(np.abs(residuals))
        
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
        
        if fit_quality == 'fast':
            self.xtol = 1e-3
            self.ftol = 1e-3
            self.gtol = 1e-3
        elif fit_quality == 'balanced':
            self.xtol = 1e-5
            self.ftol = 1e-5
            self.gtol = 1e-5
        else:
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
        
        if self.last_popt is not None:
            expected_len = n_peaks * 3 + n_baseline
            if len(self.last_popt) == expected_len:
                initial_params = self.last_popt.copy()
                if progress_callback:
                    progress_callback(0.1, "Using cached parameters...")
            else:
                initial_params = np.array(initial_peak_params)
                if n_baseline > 0:
                    if self.baseline_method == 'constant':
                        baseline_init = [np.percentile(y_norm, 5)]
                    elif self.baseline_method == 'linear':
                        baseline_init = [np.percentile(y_norm, 5), 0]
                    else:
                        baseline_init = [np.percentile(y_norm, 5), 0, 0]
                    initial_params = np.concatenate([initial_params, baseline_init])
        else:
            initial_params = np.array(initial_peak_params)
            if n_baseline > 0:
                if self.baseline_method == 'constant':
                    baseline_init = [np.percentile(y_norm, 5)]
                elif self.baseline_method == 'linear':
                    baseline_init = [np.percentile(y_norm, 5), 0]
                else:
                    baseline_init = [np.percentile(y_norm, 5), 0, 0]
                initial_params = np.concatenate([initial_params, baseline_init])
        
        if len(initial_params) == 0:
            return False, None, None, None
        
        lower_bounds, upper_bounds = self._create_bounds(x, y_norm, n_peaks, n_baseline)
        
        for i in range(len(initial_params)):
            initial_params[i] = np.clip(initial_params[i], lower_bounds[i], upper_bounds[i])
        
        try:
            if progress_callback:
                progress_callback(0.3, "Initializing fit...")
            
            def model_func(x, *params):
                return GaussianModel.multi_gaussian_with_baseline_flat(
                    x, *params, n_peaks=n_peaks, baseline_method=self.baseline_method
                )
            
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
            
            peak_params = popt[:n_peaks*3]
            baseline_params = popt[n_peaks*3:] if n_baseline > 0 else []
            
            components = []
            for i in range(n_peaks):
                amp_norm = peak_params[3*i]
                cen = peak_params[3*i + 1]
                sigma = abs(peak_params[3*i + 2])
                
                amp = amp_norm * y_max
                area = GaussianModel.calculate_area(amp_norm, sigma) * y_max
                
                component_y_norm = GaussianModel.gaussian(x, amp_norm, cen, sigma)
                
                if hasattr(x, 'min') and hasattr(x, 'max'):
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
                    'peak_type': 'auto'
                })
            
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
        
        for i in range(n_peaks):
            lower_bounds.extend([0, np.min(x), x_range * 0.001])
            upper_bounds.extend([2 * np.max(y_norm), np.max(x), x_range * 0.5])
        
        if n_baseline >= 1:
            lower_bounds.append(-np.max(y_norm))
            upper_bounds.append(np.max(y_norm))
        if n_baseline >= 2:
            lower_bounds.append(-x_range)
            upper_bounds.append(x_range)
        if n_baseline >= 3:
            lower_bounds.append(-x_range**2)
            upper_bounds.append(x_range**2)
        
        return lower_bounds, upper_bounds
    
    def preview_fit(self, x, peak_params, y_max, baseline_params=None):
        """Preview fit without optimization (fast)"""
        n_peaks = len(peak_params) // 3
        n_baseline = self.get_n_baseline_params()
        
        if baseline_params is None and n_baseline > 0:
            if self.baseline_method == 'constant':
                baseline_params = [0]
            elif self.baseline_method == 'linear':
                baseline_params = [0, 0]
            else:
                baseline_params = [0, 0, 0]
        
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
        
        if use_log_x:
            ax.set_xscale('log')
        if use_log_y:
            ax.set_yscale('log')
        
        ax.plot(x, y, 'o-', markersize=3, linewidth=1, alpha=0.7, 
                color='black', label='Data', zorder=1)
        
        x_label = 'X' + (' (log scale)' if use_log_x else '')
        y_label = 'Y' + (' (log scale)' if use_log_y else '')
        ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        ax.grid(True, alpha=0.3, linestyle='--')
        
        if self.scientific_style:
            self._apply_scientific_style(ax)
        
        return fig, ax
    
    def plot_with_peaks(self, deconvolver, peak_info, y_smooth, 
                        title="Peak Detection", ax=None, figsize=(10, 6),
                        manual_peak_position=None, manual_peak_marker=True):
        """Plot data with detected peaks with color coding"""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        if deconvolver.use_log_x:
            ax.set_xscale('log')
        if deconvolver.use_log_y:
            ax.set_yscale('log')
        
        ax.plot(deconvolver.x_sorted, deconvolver.y_sorted, 
                'o-', markersize=3, linewidth=1, alpha=0.7, 
                label='Original Data', color='black', zorder=1)
        
        if deconvolver.use_log_y:
            y_smooth_original = 10**(y_smooth * deconvolver.y_max)
        else:
            y_smooth_original = y_smooth * deconvolver.y_max
        
        ax.plot(deconvolver.x_sorted, y_smooth_original, 
                'r-', linewidth=2, label='Smoothed', color='red', zorder=2)
        
        for i, info in enumerate(peak_info):
            peak_y_original = info['y_original']
            peak_type = info.get('peak_type', 'auto')
            
            if peak_type == 'auto':
                marker_color = 'green'
                marker_edge = 'darkgreen'
                marker_face = 'lightgreen'
            elif peak_type == 'manual':
                marker_color = 'orange'
                marker_edge = 'darkorange'
                marker_face = 'orange'
            elif peak_type == 'residual':
                marker_color = 'blue'
                marker_edge = 'darkblue'
                marker_face = 'lightblue'
            else:
                marker_color = 'red'
                marker_edge = 'darkred'
                marker_face = 'yellow'
            
            ax.plot(info['x_linear'], peak_y_original, 
                    'o', markersize=8, markeredgecolor=marker_edge, 
                    markerfacecolor=marker_face, zorder=3)
            ax.text(info['x_linear'], peak_y_original * 1.05, 
                    f'{i+1}', ha='center', fontweight='bold', 
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", 
                                          facecolor='white', alpha=0.8),
                    zorder=4)
        
        if manual_peak_position is not None and manual_peak_marker:
            idx = np.argmin(np.abs(deconvolver.x_sorted - manual_peak_position))
            y_at_position = deconvolver.y_sorted[idx]
            ax.plot(manual_peak_position, y_at_position, 
                    'D', markersize=12, markeredgecolor='red', 
                    markerfacecolor='yellow', zorder=5, label='Selected position')
            ax.axvline(x=manual_peak_position, color='red', linestyle='--', 
                      alpha=0.5, linewidth=1, zorder=4)
        
        x_label = 'X' + (' (log scale)' if deconvolver.use_log_x else '')
        y_label = 'Y' + (' (log scale)' if deconvolver.use_log_y else '')
        ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightgreen', edgecolor='darkgreen', label='Auto-detected peaks'),
            Patch(facecolor='orange', edgecolor='darkorange', label='Manual peaks'),
            Patch(facecolor='lightblue', edgecolor='darkblue', label='Residual-based peaks')
        ]
        ax.legend(handles=legend_elements, loc='best', fontsize=10, frameon=True, edgecolor='black')
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
        
        if deconvolver.use_log_x:
            ax.set_xscale('log')
        if deconvolver.use_log_y:
            ax.set_yscale('log')
        
        ax.scatter(deconvolver.x_linear, deconvolver.y_original, 
                   s=10, alpha=0.5, color='black', label='Data', zorder=1)
        
        if deconvolver.use_log_x:
            x_min = np.maximum(np.min(deconvolver.x_linear[deconvolver.x_linear>0]), np.finfo(float).eps)
            x_max = np.max(deconvolver.x_linear)
            x_dense = np.logspace(np.log10(x_min), np.log10(x_max), 2000)
            x_dense_log = np.log10(x_dense)
        else:
            x_dense = np.linspace(np.min(deconvolver.x_linear), 
                                  np.max(deconvolver.x_linear), 2000)
            x_dense_log = x_dense
        
        if show_components and deconvolver.components:
            colors = plt.cm.Set3(np.linspace(0, 1, len(deconvolver.components)))
            for c, color in zip(deconvolver.components, colors):
                y_component = GaussianModel.gaussian(x_dense_log, c['amp_norm'], 
                                                    c['cen_log'], c['sigma_log']) * deconvolver.y_max
                
                ax.fill_between(x_dense, 0, y_component, 
                                color=color, alpha=0.3, linewidth=0)
                
                ax.plot(x_dense, y_component, '-', color=color, linewidth=2,
                       label=f'Peak {c["id"]}: {c["fraction_percent"]:.1f}%', zorder=2)
        
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
        
        if preview_mode and preview_fit is not None:
            y_total = preview_fit * deconvolver.y_max
            ax.plot(x_dense, y_total, 'b--', linewidth=2, 
                   label='Preview (no fit)', zorder=3, alpha=0.7)
        elif deconvolver.fit_y_norm is not None:
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
        
        x_label = 'X' + (' (log scale)' if deconvolver.use_log_x else '')
        y_label = 'Y' + (' (log scale)' if deconvolver.use_log_y else '')
        ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
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
        
        ax.legend(loc='upper right', fontsize=8, frameon=True, edgecolor='black')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        if self.scientific_style:
            self._apply_scientific_style(ax)
        
        return fig, ax
    
    def plot_residual_analysis(self, deconvolver, residual_peaks, title="Residual Analysis", ax=None, figsize=(10, 6)):
        """Plot residuals with detected peaks"""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        if deconvolver.use_log_x:
            ax.set_xscale('log')
        
        residuals = deconvolver.quality_metrics.get('Residuals', None)
        if residuals is not None:
            ax.plot(deconvolver.x_linear, residuals * deconvolver.y_max, 
                   'b-', linewidth=1.5, label='Residuals', alpha=0.7)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            for peak in residual_peaks:
                ax.axvline(x=peak['x_linear'], color='red', linestyle='--', 
                          alpha=0.5, linewidth=1)
                ax.plot(peak['x_linear'], peak['y_residual'], 'ro', 
                       markersize=8, markeredgecolor='darkred', markerfacecolor='red')
                ax.text(peak['x_linear'], peak['y_residual'] * 1.1, 
                       f"{peak['peak_id']}", ha='center', fontweight='bold')
        
        ax.set_xlabel('X' + (' (log scale)' if deconvolver.use_log_x else ''))
        ax.set_ylabel('Residuals')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
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
        
        ax.plot(deconvolver.x, deconvolver.y, 
               'o-', markersize=3, alpha=0.5, label='Data', color='black')
        ax.plot(deconvolver.x, y_smooth * deconvolver.y_max, 
               'r-', linewidth=2, label='Smoothed')
        
        for i, info in enumerate(peak_info):
            ax.plot(info['x'], info['y'] * deconvolver.y_max, 'ro', 
                   markersize=8, markeredgecolor='darkred')
            ax.text(info['x'], info['y'] * deconvolver.y_max * 1.05, 
                   f'{i+1}', ha='center', fontweight='bold')
        
        ax.set_xlabel(deconvolver.x_label)
        ax.set_ylabel(deconvolver.y_label)
        ax.set_title('Detected Peaks (Original Scale)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
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
        self.x_original = np.array(x_linear).copy()
        self.y_original_raw = np.array(y_original).copy()
        
        self.x_linear = np.array(x_linear)
        self.y_original = np.array(y_original)
        self.use_log_x = use_log_x
        self.use_log_y = use_log_y
        self.baseline_method = baseline_method
        
        sort_idx = np.argsort(self.x_linear)
        self.x_linear = self.x_linear[sort_idx]
        self.y_original = self.y_original[sort_idx]
        
        self.x_sorted = self.x_linear.copy()
        self.y_sorted = self.y_original.copy()
        
        self.preprocessor = DataPreprocessor(clip_negative, show_warnings)
        preprocessed = self.preprocessor.preprocess_for_fitting(
            self.x_linear, self.y_original, use_log_x, use_log_y
        )
        
        self.x_sorted = preprocessed['x_sorted']
        self.y_sorted = preprocessed['y_sorted']
        self.x = preprocessed['x']
        self.y = preprocessed['y']
        self.y_for_fitting = preprocessed['y_for_fitting']
        self.x_label = preprocessed['x_label']
        self.y_label = preprocessed['y_label']
        self.clipped_points = preprocessed['clipped_points']
        self.small_values_warning = preprocessed['small_values_warning']
        
        self.y_max = np.percentile(self.y_for_fitting, 95) if np.any(self.y_for_fitting > 0) else 1.0
        
        if self.y_max > 0:
            self.y_norm = self.y / self.y_max
        else:
            self.y_norm = self.y
        
        self.components = []
        self.fit_y_norm = None
        self.popt = None
        self.baseline_params = None
        self.quality_metrics = {}
        self.convergence_history = []
        self.total_area = 0
        
        self.fitter = None
        
        self.multi_gaussian = GaussianModel.multi_gaussian
        self.gaussian = GaussianModel.gaussian
    
    def auto_detect_peaks(self, sensitivity=0.03, min_distance=5):
        """Automatic peak detection using derivatives"""
        window_length = min(11, len(self.y_norm) // 5 * 2 + 1)
        if window_length % 2 == 0:
            window_length += 1
        
        if window_length >= 5:
            y_smooth = savgol_filter(self.y_norm, window_length, 3)
        else:
            y_smooth = self.y_norm
        
        dy, d2y, y_smooth = DerivativeAnalyzer.calculate_derivatives(self.x, y_smooth)
        
        height_threshold = sensitivity * np.max(y_smooth)
        peaks1, _ = find_peaks(y_smooth, height=height_threshold, distance=min_distance)
        peaks2 = DerivativeAnalyzer.find_peaks_by_derivatives(self.x, y_smooth, dy, d2y, sensitivity)
        
        all_peaks = sorted(set(np.concatenate([peaks1, peaks2])))
        
        filtered_peaks = []
        for peak in all_peaks:
            if not filtered_peaks or abs(self.x[peak] - self.x[filtered_peaks[-1]]) > min_distance * np.mean(np.diff(self.x)):
                filtered_peaks.append(peak)
        
        peak_info = []
        initial_params = []
        
        for peak_idx in filtered_peaks:
            cen = self.x[peak_idx]
            amp = y_smooth[peak_idx]
            
            sigma = GaussianModel.estimate_sigma_from_peak(self.x, y_smooth, peak_idx)
            sigma = max(sigma, 0.01 * (np.max(self.x) - np.min(self.x)) / max(len(filtered_peaks), 1))
            
            if self.use_log_x:
                x_linear = 10**self.x[peak_idx]
            else:
                x_linear = self.x[peak_idx]
            
            idx = np.argmin(np.abs(self.x_sorted - x_linear))
            y_original_value = self.y_sorted[idx]
            
            peak_info.append({
                'index': peak_idx,
                'x': self.x[peak_idx],
                'x_linear': x_linear,
                'y': self.y[peak_idx],
                'y_original': y_original_value,
                'amp_est': amp,
                'cen_est': cen,
                'sigma_est': sigma,
                'dy': dy[peak_idx],
                'd2y': d2y[peak_idx],
                'peak_type': 'auto'
            })
            
            initial_params.extend([amp, cen, sigma])
        
        return filtered_peaks, peak_info, initial_params, (dy, d2y, y_smooth)
    
    def add_manual_peak(self, position, amplitude=None, sigma=None):
        """Add a manually selected peak"""
        if self.use_log_x:
            pos_log = np.log10(position) if position > 0 else position
        else:
            pos_log = position
        
        if amplitude is None:
            idx = np.argmin(np.abs(self.x - pos_log))
            amplitude = self.y_norm[idx] * 0.5
        
        if sigma is None:
            sigma = GaussianModel.estimate_sigma_at_position(self.x, self.y_norm, pos_log)
            sigma = max(sigma, 0.01 * (np.max(self.x) - np.min(self.x)) / 20)
        
        new_peak_info = {
            'index': len(self.components) + 1000,
            'x': pos_log,
            'x_linear': position,
            'y': amplitude,
            'y_original': amplitude * self.y_max,
            'amp_est': amplitude,
            'cen_est': pos_log,
            'sigma_est': sigma,
            'peak_type': 'manual'
        }
        
        return new_peak_info, [amplitude, pos_log, sigma]
    
    def analyze_residuals_for_missing_peaks(self, sensitivity=0.05, min_distance=5):
        """Analyze residuals to find missing peaks"""
        if not self.quality_metrics or 'Residuals' not in self.quality_metrics:
            return []
        
        residuals = self.quality_metrics['Residuals']
        residuals_abs = np.abs(residuals)
        
        height_threshold = sensitivity * np.max(residuals_abs)
        residual_peaks_idx, _ = find_peaks(residuals_abs, height=height_threshold, distance=min_distance)
        
        if len(residual_peaks_idx) == 0:
            residual_peaks_idx, _ = find_peaks(residuals, height=sensitivity * np.max(np.abs(residuals)), distance=min_distance)
        
        residual_peaks = []
        for i, idx in enumerate(residual_peaks_idx):
            x_pos = self.x[idx]
            if self.use_log_x:
                x_linear = 10**x_pos
            else:
                x_linear = x_pos
            
            y_residual = residuals[idx]
            y_original_at_pos = self.y_original[idx]
            
            sigma_est = GaussianModel.estimate_sigma_at_position(self.x, self.y_norm, x_pos)
            
            residual_peaks.append({
                'peak_id': i + 1,
                'index': idx,
                'x': x_pos,
                'x_linear': x_linear,
                'y_residual': y_residual,
                'y_original': y_original_at_pos,
                'amplitude_est': np.abs(y_residual),
                'sigma_est': sigma_est,
                'peak_type': 'residual'
            })
        
        return residual_peaks
    
    def fit(self, initial_params=None, method='trf', maxfev=5000, 
            fit_quality='balanced', last_popt=None, progress_callback=None):
        """Perform fitting with selected method and baseline"""
        if initial_params is None:
            _, _, initial_params, _ = self.auto_detect_peaks()
        
        if len(initial_params) == 0:
            return False
        
        self.fitter = GaussianFitter(
            method=method, 
            max_nfev=maxfev,
            baseline_method=self.baseline_method,
            fit_quality=fit_quality,
            last_popt=last_popt
        )
        
        success, popt, components, baseline_params = self.fitter.fit(
            self.x, self.y_norm, initial_params, self.y_max,
            progress_callback=progress_callback
        )
        
        if success:
            self.popt = popt
            self.components = components
            self.baseline_params = baseline_params
            
            n_peaks = len(components)
            peak_params = []
            for c in components:
                peak_params.extend([c['amp_norm'], c['cen_log'], c['sigma_log']])
            
            self.fit_y_norm = GaussianModel.multi_gaussian_with_baseline(
                self.x, n_peaks, peak_params, baseline_params or [], self.baseline_method
            )
            
            self.total_area = sum([c['area'] for c in self.components])
            
            self.quality_metrics = FitQualityAnalyzer.calculate_metrics(
                self.y_norm, self.fit_y_norm, len(popt)
            )
            
            return True
        
        return False
    
    def preview_fit(self, initial_params=None):
        """Preview fit without optimization (fast)"""
        if initial_params is None:
            _, _, initial_params, _ = self.auto_detect_peaks()
        
        if len(initial_params) == 0:
            return None
        
        fitter = GaussianFitter(
            baseline_method=self.baseline_method
        )
        
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
        
        st.session_state.app_state.pending_remove = peak_id
        return True
    
    def split_peak(self, peak_id, split_position):
        """Split a peak into two (does NOT perform fit, just marks for splitting)"""
        if peak_id > len(self.components):
            return False
        
        st.session_state.app_state.pending_split = (peak_id, split_position)
        return True
    
    def apply_pending_operations(self, fit_quality='balanced', progress_callback=None):
        """Apply all pending operations and perform fit"""
        if self.components:
            current_params = []
            for c in self.components:
                current_params.extend([c['amp_norm'], c['cen_log'], c['sigma_log']])
        else:
            return False
        
        if st.session_state.app_state.pending_remove is not None:
            remove_id = st.session_state.app_state.pending_remove
            new_params = []
            for i, c in enumerate(self.components):
                if i != remove_id - 1:
                    new_params.extend([c['amp_norm'], c['cen_log'], c['sigma_log']])
            current_params = new_params
            st.session_state.app_state.pending_remove = None
        
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
        st.subheader("📊 Data Smoothing")
        
        st.session_state.app_state.smoothing_enabled = st.checkbox(
            "Enable data smoothing",
            value=st.session_state.app_state.smoothing_enabled,
            help="Apply smoothing to reduce noise before peak detection"
        )
        
        if st.session_state.app_state.smoothing_enabled:
            st.session_state.app_state.smoothing_method = st.selectbox(
                "Smoothing method",
                options=['savgol', 'gaussian', 'median', 'adaptive'],
                index=0,
                help="savgol: Savitzky-Golay, gaussian: Gaussian filter, median: median filter, adaptive: automatic based on noise"
            )
            
            if st.session_state.app_state.smoothing_method == 'savgol':
                st.session_state.app_state.smoothing_window = st.slider(
                    "Window length",
                    min_value=5,
                    max_value=51,
                    value=11,
                    step=2,
                    help="Must be odd number"
                )
                st.session_state.app_state.smoothing_polyorder = st.slider(
                    "Polynomial order",
                    min_value=1,
                    max_value=5,
                    value=3,
                    step=1,
                    help="Must be less than window length"
                )
            elif st.session_state.app_state.smoothing_method == 'gaussian':
                st.session_state.app_state.smoothing_window = st.slider(
                    "Sigma",
                    min_value=0.5,
                    max_value=10.0,
                    value=2.0,
                    step=0.5,
                    help="Standard deviation for Gaussian kernel"
                )
            elif st.session_state.app_state.smoothing_method == 'median':
                st.session_state.app_state.smoothing_window = st.slider(
                    "Window size",
                    min_value=3,
                    max_value=21,
                    value=5,
                    step=2,
                    help="Median filter window size (must be odd)"
                )
            
            st.session_state.app_state.show_smoothing_preview = st.checkbox(
                "Show smoothing preview",
                value=st.session_state.app_state.show_smoothing_preview,
                help="Preview smoothed data before peak detection"
            )
    
    st.markdown("---")
    
    if st.button("🔄 Start Over", use_container_width=True):
        st.session_state.app_state = AppState()
        st.rerun()


# ==================== STEP 1: DATA LOADING ====================

if st.session_state.app_state.current_step == 1:
    st.header("Step 1: Data Loading")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
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
        
        if st.button("🔍 Auto-detect Scales", use_container_width=True):
        suggest_log_x, suggest_log_y = DataParser.auto_detect_scale(
            st.session_state.app_state.raw_x, 
            st.session_state.app_state.raw_y
        )
        st.session_state.app_state.use_log_x = suggest_log_x
        st.session_state.app_state.use_log_y = suggest_log_y
        st.rerun()
        
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
        
        x_min = float(np.min(st.session_state.app_state.raw_x))
        x_max = float(np.max(st.session_state.app_state.raw_x))
        
        if st.session_state.app_state.x_range_min is None:
        st.session_state.app_state.x_range_min = x_min
        if st.session_state.app_state.x_range_max is None:
        st.session_state.app_state.x_range_max = x_max
        
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
            x_range, y_range = DataParser.apply_range_selection(
                st.session_state.app_state.raw_x,
                st.session_state.app_state.raw_y,
                st.session_state.app_state.x_range_min,
                st.session_state.app_state.x_range_max
            )
            
            st.session_state.app_state.raw_x = x_range
            st.session_state.app_state.raw_y = y_range
            
            st.session_state.app_state.current_step = 3
            st.rerun()
        
        with col2:
        st.subheader("Preview:")
        
        plotter = SpectrumPlotter()
        fig, ax = plt.subplots(figsize=(8, 5))
        
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
        
        if len(x_preview) < len(st.session_state.app_state.raw_x):
        ax.axvspan(st.session_state.app_state.x_range_min,
                  st.session_state.app_state.x_range_max,
                  alpha=0.2, color='green', label='Selected Range')
        
        st.pyplot(fig)
        plt.close()
        
        
        # ==================== STEP 3: PEAK DETECTION ====================
        
        elif st.session_state.app_state.current_step == 3:
        st.header("Step 3: Peak Detection")
        
        if st.session_state.app_state.deconvolver is None:
        st.session_state.app_state.original_x = st.session_state.app_state.raw_x.copy()
        st.session_state.app_state.original_y = st.session_state.app_state.raw_y.copy()
        
        if st.session_state.app_state.smoothing_enabled:
        smoother = DataSmoother()
        if st.session_state.app_state.smoothing_method == 'savgol':
            y_smoothed = smoother.savgol_smooth(
                st.session_state.app_state.raw_y,
                st.session_state.app_state.smoothing_window,
                st.session_state.app_state.smoothing_polyorder
            )
        elif st.session_state.app_state.smoothing_method == 'gaussian':
            y_smoothed = smoother.gaussian_smooth(
                st.session_state.app_state.raw_y,
                st.session_state.app_state.smoothing_window
            )
        elif st.session_state.app_state.smoothing_method == 'median':
            y_smoothed = smoother.median_smooth(
                st.session_state.app_state.raw_y,
                int(st.session_state.app_state.smoothing_window)
            )
        else:
            y_smoothed = smoother.adaptive_smooth(st.session_state.app_state.raw_y)
        
        st.session_state.app_state.raw_y = y_smoothed
        
        st.session_state.app_state.deconvolver = GaussianDeconvolver(
        st.session_state.app_state.raw_x,
        st.session_state.app_state.raw_y,
        use_log_x=st.session_state.app_state.use_log_x,
        use_log_y=st.session_state.app_state.use_log_y,
        clip_negative=st.session_state.app_state.clip_negative,
        show_warnings=st.session_state.app_state.show_warnings,
        baseline_method=st.session_state.app_state.baseline_method
        )
        
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
        
        st.markdown("---")
        st.subheader("✋ Manual Peak Addition")
        
        deconv = st.session_state.app_state.deconvolver
        if deconv.use_log_x:
        x_min_display = 10**np.min(deconv.x)
        x_max_display = 10**np.max(deconv.x)
        slider_min = x_min_display
        slider_max = x_max_display
        else:
        slider_min = np.min(deconv.x)
        slider_max = np.max(deconv.x)
        
        manual_position = st.slider(
        "Select peak position:",
        min_value=float(slider_min),
        max_value=float(slider_max),
        value=float(slider_min + (slider_max - slider_min) / 2),
        step=float((slider_max - slider_min) / 200),
        format="%.4e",
        help="Drag slider to select peak position"
        )
        
        st.session_state.app_state.manual_peak_position = manual_position
        
        col_manual1, col_manual2 = st.columns(2)
        with col_manual1:
        if st.button("➕ Add Manual Peak", use_container_width=True):
            new_peak_info, new_params = deconv.add_manual_peak(manual_position)
            
            if st.session_state.app_state.peak_info is None:
                st.session_state.app_state.peak_info = []
                st.session_state.app_state.initial_params = []
            
            st.session_state.app_state.peak_info.append(new_peak_info)
            st.session_state.app_state.initial_params.extend(new_params)
            st.session_state.app_state.manual_peaks.append(manual_position)
            st.success(f"Manual peak added at position {manual_position:.4e}")
            st.rerun()
        
        with col_manual2:
        if st.button("🗑️ Clear Manual Peaks", use_container_width=True):
            if st.session_state.app_state.peak_info:
                st.session_state.app_state.peak_info = [p for p in st.session_state.app_state.peak_info 
                                                       if p.get('peak_type', 'auto') != 'manual']
                st.session_state.app_state.manual_peaks = []
                st.session_state.app_state.initial_params = []
                _, auto_peaks, auto_params, _ = deconv.auto_detect_peaks(
                    sensitivity=st.session_state.app_state.sensitivity,
                    min_distance=st.session_state.app_state.min_distance
                )
                st.session_state.app_state.peak_info = auto_peaks
                st.session_state.app_state.initial_params = auto_params
            st.success("Manual peaks cleared")
            st.rerun()
        
        st.markdown("---")
        st.subheader("🔍 Residual Analysis")
        
        if st.button("🔎 Find Missing Peaks in Residuals", use_container_width=True):
        with st.spinner("Analyzing residuals..."):
            if not deconv.quality_metrics:
                if st.session_state.app_state.peak_info:
                    initial_params = st.session_state.app_state.initial_params
                    with st.spinner("Performing initial fit..."):
                        success = deconv.fit(
                            initial_params=initial_params,
                            method=st.session_state.app_state.fitting_method,
                            maxfev=st.session_state.app_state.max_nfev // 2,
                            fit_quality='fast'
                        )
                        if success:
                            st.success("Initial fit completed")
                        else:
                            st.warning("Could not perform initial fit")
            
            if deconv.quality_metrics:
                residual_peaks = deconv.analyze_residuals_for_missing_peaks(
                    sensitivity=st.session_state.app_state.sensitivity * 0.5,
                    min_distance=st.session_state.app_state.min_distance
                )
                
                st.session_state.app_state.residual_peaks = residual_peaks
                st.session_state.app_state.residual_analysis_done = True
                
                if residual_peaks:
                    st.success(f"Found {len(residual_peaks)} potential missing peaks")
                else:
                    st.info("No additional peaks found in residuals")
            else:
                st.warning("Please run peak detection first")
        
        if st.session_state.app_state.residual_peaks:
        st.subheader("Suggested Peaks from Residuals")
        for peak in st.session_state.app_state.residual_peaks:
            col_res1, col_res2 = st.columns([3, 1])
            with col_res1:
                st.write(f"Peak at X = {peak['x_linear']:.4e}, amp ≈ {peak['amplitude_est']:.4e}")
            with col_res2:
                if st.button(f"Add", key=f"add_res_{peak['peak_id']}"):
                    new_peak_info, new_params = deconv.add_manual_peak(
                        peak['x_linear'],
                        amplitude=peak['amplitude_est'],
                        sigma=peak['sigma_est']
                    )
                    st.session_state.app_state.peak_info.append(new_peak_info)
                    st.session_state.app_state.initial_params.extend(new_params)
                    st.session_state.app_state.residual_peaks = []
                    st.success(f"Added peak at {peak['x_linear']:.4e}")
                    st.rerun()
        
        st.markdown("---")
        
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
                
                if st.session_state.app_state.manual_peaks:
                    for manual_peak in st.session_state.app_state.manual_peaks:
                        new_info, new_params = deconv.add_manual_peak(manual_peak)
                        peak_info.append(new_info)
                        initial_params.extend(new_params)
                
                st.session_state.app_state.peak_info = peak_info
                st.session_state.app_state.derivatives = derivatives
                st.session_state.app_state.initial_params = initial_params
            st.success(f"Found {len(peak_info)} peaks!")
        
        if st.session_state.app_state.peak_info is not None:
        if st.button("✅ Confirm Peaks", use_container_width=True):
            with st.spinner("Preparing preview..."):
                if st.session_state.app_state.preview_mode:
                    st.session_state.app_state.current_step = 4
                    st.rerun()
                else:
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
                        st.session_state.app_state.last_popt = st.session_state.app_state.deconvolver.popt
                        st.session_state.app_state.current_step = 4
                        st.rerun()
                    else:
                        st.error("Fitting failed. Try adjusting parameters.")
        
        with col2:
        if st.session_state.app_state.peak_info is not None:
        st.subheader(f"Peaks found: {len(st.session_state.app_state.peak_info)}")
        
        if st.session_state.app_state.show_smoothing_preview and st.session_state.app_state.smoothing_enabled:
            st.info(f"Smoothing applied: {st.session_state.app_state.smoothing_method}")
        
        dy, d2y, y_smooth = st.session_state.app_state.derivatives
        deconv = st.session_state.app_state.deconvolver
        plotter = SpectrumPlotter()
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Peaks", "📈 Derivatives", "📋 Information", "🔍 Residuals"])
        
        with tab1:
            fig, ax = plt.subplots(figsize=(10, 6))
            plotter.plot_with_peaks(
                deconv, 
                st.session_state.app_state.peak_info, 
                y_smooth,
                title=f"Peak Detection - {len(st.session_state.app_state.peak_info)} peaks found",
                ax=ax,
                manual_peak_position=st.session_state.app_state.manual_peak_position,
                manual_peak_marker=True
            )
            st.pyplot(fig)
            plt.close()
        
        with tab2:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
            
            ax1.plot(deconv.x, dy, 'b-', linewidth=1.5, label='First derivative')
            ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax1.set_xlabel(deconv.x_label)
            ax1.set_ylabel('dy/dx')
            ax1.set_title('First Derivative')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
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
            data = []
            for i, info in enumerate(st.session_state.app_state.peak_info):
                peak_type = info.get('peak_type', 'auto')
                type_icon = "🟢" if peak_type == 'auto' else "🟠" if peak_type == 'manual' else "🔵"
                data.append({
                    'Peak': i + 1,
                    'Type': f"{type_icon} {peak_type}",
                    'X Center': f"{info['x_linear']:.4e}",
                    'Y Amplitude': f"{info['y_original']:.4e}",
                    'Estimated Sigma': f"{info['sigma_est']:.4f}",
                })
            
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
            
            st.markdown("---")
            st.subheader("Detection Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Peaks Found", len(st.session_state.app_state.peak_info))
            with col2:
                auto_count = sum(1 for p in st.session_state.app_state.peak_info if p.get('peak_type', 'auto') == 'auto')
                st.metric("Auto-detected", auto_count)
            with col3:
                manual_count = sum(1 for p in st.session_state.app_state.peak_info if p.get('peak_type') == 'manual')
                st.metric("Manual", manual_count)
            with col4:
                residual_count = sum(1 for p in st.session_state.app_state.peak_info if p.get('peak_type') == 'residual')
                st.metric("From Residuals", residual_count)
            
            st.markdown("---")
            st.subheader("Peak Type Legend")
            st.markdown("🟢 **Green** - Auto-detected peaks (standard algorithm)")
            st.markdown("🟠 **Orange** - Manually added peaks (user-selected)")
            st.markdown("🔵 **Blue** - Residual-based peaks (found in fit residuals)")
        
        with tab4:
            if st.session_state.app_state.residual_analysis_done and st.session_state.app_state.residual_peaks:
                fig, ax = plt.subplots(figsize=(10, 6))
                plotter.plot_residual_analysis(
                    deconv,
                    st.session_state.app_state.residual_peaks,
                    title="Residual Analysis - Potential Missing Peaks",
                    ax=ax
                )
                st.pyplot(fig)
                plt.close()
                
                st.info("Blue markers indicate potential missing peaks detected in residuals")
            else:
                st.info("Click 'Find Missing Peaks in Residuals' to analyze fit residuals")
        
        
        # ==================== STEP 4: EDITING ====================
        
        elif st.session_state.app_state.current_step == 4:
        st.header("Step 4: Peak Editing")
        
        if st.session_state.app_state.deconvolver:
        deconv = st.session_state.app_state.deconvolver
        plotter = SpectrumPlotter()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
        st.subheader("Peak Management")
        
        if st.session_state.app_state.pending_remove is not None:
            st.warning(f"Pending: Remove Peak {st.session_state.app_state.pending_remove}")
        if st.session_state.app_state.pending_split is not None:
            st.warning(f"Pending: Split Peak {st.session_state.app_state.pending_split[0]}")
        
        if deconv.quality_metrics:
            metrics = deconv.quality_metrics
            st.info(f"R² = {metrics.get('R²', 0):.4f} | RMSE = {metrics.get('RMSE', 0):.2e}")
        
        st.markdown("---")
        
        if st.session_state.app_state.preview_mode:
            st.info("🔍 PREVIEW MODE - No fitting performed")
        
        if deconv.components:
            peak_options = {f"Peak {c['id']}: center = {c['cen_linear']:.2e}, fraction = {c['fraction_percent']:.1f}%": c['id'] 
                           for c in deconv.components}
            
            selected_peak = st.selectbox(
                "Select peak for editing:",
                options=list(peak_options.keys())
            )
            
            if selected_peak:
                peak_id = peak_options[selected_peak]
                
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
        
        if st.button("🔄 Apply Changes and Recalculate", type="primary", use_container_width=True):
            with st.spinner("Applying changes and recalculating..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(progress, message):
                    progress_bar.progress(progress)
                    status_text.text(message)
                
                if st.session_state.app_state.preview_mode:
                    preview_fit = deconv.preview_fit()
                    if preview_fit is not None:
                        st.session_state.app_state.preview_fit = preview_fit
                        st.success("Preview updated")
                    progress_bar.empty()
                    status_text.empty()
                    st.rerun()
                else:
                    success = deconv.apply_pending_operations(
                        fit_quality=st.session_state.app_state.fit_quality,
                        progress_callback=update_progress
                    )
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    if success:
                        st.session_state.app_state.last_popt = deconv.popt
                        st.success("Recalculation complete!")
                        st.rerun()
                    else:
                        st.error("Recalculation failed")
        
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
        
        col_back, _ = st.columns([1, 5])
        with col_back:
        if st.button("⬅️ Back to Editing", use_container_width=True):
            st.session_state.app_state.current_step = 4
            st.rerun()
        
        st.markdown("---")
        
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
            
            fig = plt.figure(figsize=(12, 10))
            
            ax1 = plt.subplot(2, 2, 1)
            peaks = [f'Peak {c["id"]}' for c in deconv.components]
            areas = [c['area'] for c in deconv.components]
            fractions = [c['fraction_percent'] for c in deconv.components]
            colors = plt.cm.Set3(np.linspace(0, 1, len(peaks)))
            
            bars1 = ax1.bar(peaks, areas, color=colors, edgecolor='black', alpha=0.7)
            ax1.set_xlabel('Peak', fontweight='bold')
            ax1.set_ylabel('Area', fontweight='bold')
            ax1.set_title('Peak Areas', fontweight='bold')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3, axis='y')
            
            for bar, area in zip(bars1, areas):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{area:.2e}',
                        ha='center', va='bottom', fontsize=8, rotation=0)
            
            ax2 = plt.subplot(2, 2, 2)
            bars2 = ax2.bar(peaks, fractions, color=colors, edgecolor='black', alpha=0.7)
            ax2.set_xlabel('Peak', fontweight='bold')
            ax2.set_ylabel('Fraction (%)', fontweight='bold')
            ax2.set_title('Peak Fractions', fontweight='bold')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3, axis='y')
            
            for bar, frac in zip(bars2, fractions):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{frac:.1f}%',
                        ha='center', va='bottom', fontsize=8)
            
            ax3 = plt.subplot(2, 2, 3)
            wedges, texts, autotexts = ax3.pie(fractions, labels=peaks, autopct='%1.1f%%',
                   colors=colors, startangle=90,
                   textprops={'fontweight': 'bold'})
            ax3.set_title('Area Distribution - Pie Chart', fontweight='bold')
            
            ax4 = plt.subplot(2, 2, 4)
            y_pos = np.arange(len(peaks))
            bars4 = ax4.barh(y_pos, fractions, color=colors, edgecolor='black', alpha=0.7)
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels(peaks)
            ax4.set_xlabel('Fraction (%)', fontweight='bold')
            ax4.set_title('Peak Fractions - Horizontal View', fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='x')
            
            for bar, frac in zip(bars4, fractions):
                width = bar.get_width()
                ax4.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                        f'{frac:.1f}%',
                        ha='left', va='center', fontsize=8)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
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
            max_amp = max([c['amp'] for c in deconv.components])
            
            fig_norm, ax_norm = plt.subplots(figsize=(10, 6))
            
            if deconv.use_log_x:
                ax_norm.set_xscale('log')
            
            if deconv.use_log_x:
                x_min = np.maximum(np.min(deconv.x_linear[deconv.x_linear>0]), np.finfo(float).eps)
                x_max = np.max(deconv.x_linear)
                x_dense = np.logspace(np.log10(x_min), np.log10(x_max), 2000)
                x_dense_log = np.log10(x_dense)
            else:
                x_dense = np.linspace(np.min(deconv.x_linear), np.max(deconv.x_linear), 2000)
                x_dense_log = x_dense
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(deconv.components)))
            for c, color in zip(deconv.components, colors):
                y_component_norm = GaussianModel.gaussian(x_dense_log, c['amp_norm'], 
                                                        c['cen_log'], c['sigma_log']) 
                y_component_norm = y_component_norm * deconv.y_max / max_amp
                
                ax_norm.fill_between(x_dense, 0, y_component_norm, 
                                    color=color, alpha=0.3, linewidth=0)
                
                ax_norm.plot(x_dense, y_component_norm, '-', color=color, linewidth=2,
                           label=f'Peak {c["id"]}: {c["fraction_percent"]:.1f}%', zorder=2)
            
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
            
            y_original_norm = deconv.y_original / max_amp
            ax_norm.scatter(deconv.x_linear, y_original_norm, 
                           s=10, alpha=0.5, color='black', label='Data', zorder=1)
            
            x_label = 'X' + (' (log scale)' if deconv.use_log_x else '')
            y_label = 'Normalized Intensity'
            ax_norm.set_xlabel(x_label, fontsize=12, fontweight='bold')
            ax_norm.set_ylabel(y_label, fontsize=12, fontweight='bold')
            ax_norm.set_title('Deconvolution Result - Normalized to Max Peak = 1', fontsize=14, fontweight='bold')
            
            if deconv.quality_metrics:
                metrics_text = f"R² = {deconv.quality_metrics.get('R²', 0):.4f}\n"
                metrics_text += f"RMSE = {deconv.quality_metrics.get('RMSE', 0):.2e}"
                ax_norm.text(0.02, 0.98, metrics_text, transform=ax_norm.transAxes,
                            fontsize=10, verticalalignment='top',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            ax_norm.legend(loc='upper right', fontsize=8, frameon=True, edgecolor='black')
            ax_norm.grid(True, alpha=0.3, linestyle='--')
            
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
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(deconv.components)))
            for c, color in zip(deconv.components, colors):
                y_component_norm = GaussianModel.gaussian(x_dense_log, c['amp_norm'], 
                                                        c['cen_log'], c['sigma_log']) 
                y_component_norm = y_component_norm * deconv.y_max / max_amp
                
                ax_comp_norm.plot(x_dense, y_component_norm, '-', color=color, linewidth=2,
                                label=f'Peak {c["id"]} (center: {c["cen_linear"]:.2e})')
            
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
        
        st.subheader("Normalized Parameters")
        norm_data = []
        for c in deconv.components:
            norm_data.append({
                'Peak': c['id'],
                'Center': f"{c['cen_linear']:.4e}",
                'Normalized Amplitude': f"{c['amp'] / max_amp:.4f}",
                'Original Amplitude': f"{c['amp']:.4e}",
                'Fraction (%)': f"{c['fraction_percent']:.2f}"
            })
        
        df_norm = pd.DataFrame(norm_data)
        st.dataframe(df_norm, use_container_width=True)
        
        with tab3:
        st.subheader("Results Table - Complete Dataset")
        
        data = []
        for c in deconv.components:
            data.append({
                'Peak ID': c['id'],
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
        
        display_df = df.copy()
        for col in ['Center', 'Amplitude', 'Area']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4e}")
        display_df['Center (log)'] = display_df['Center (log)'].apply(lambda x: f"{x:.4f}")
        display_df['Amplitude (norm)'] = display_df['Amplitude (norm)'].apply(lambda x: f"{x:.4f}")
        display_df['Sigma (log)'] = display_df['Sigma (log)'].apply(lambda x: f"{x:.4f}")
        display_df['FWHM'] = display_df['FWHM'].apply(lambda x: f"{x:.4f}")
        display_df['Fraction (%)'] = display_df['Fraction (%)'].apply(lambda x: f"{x:.2f}")
        
        st.dataframe(display_df, use_container_width=True)
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Raw Data (CSV)",
            data=csv,
            file_name=f"deconvolution_peaks_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        st.markdown("---")
        
        if deconv.baseline_method != 'none' and deconv.baseline_params:
            st.subheader("Baseline Parameters")
            baseline_df = pd.DataFrame([{
                'Method': deconv.baseline_method,
                'Parameters': ', '.join([f"{p:.4e}" for p in deconv.baseline_params])
            }])
            st.dataframe(baseline_df, use_container_width=True)
        
        st.markdown("---")
        
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
            if st.button("📥 Export Peaks to CSV", use_container_width=True):
                df_peaks = pd.DataFrame([{
                    'Peak_ID': c['id'],
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
            
            if 'Residuals' in deconv.quality_metrics:
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
            if st.button("📄 Export Detailed Report", use_container_width=True):
                max_amp = max([c['amp'] for c in deconv.components])
                
                report = f"""GAUSSIAN DECONVOLUTION REPORT
        {"="*80}
        
        Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Number of points: {len(deconv.x_linear)}
        X range: [{deconv.x_linear[0]:.2e}, {deconv.x_linear[-1]:.2e}]
        Logarithmic X scale: {deconv.use_log_x}
        Baseline method: {deconv.baseline_method}
        Smoothing: {'Enabled' if st.session_state.app_state.smoothing_enabled else 'Disabled'}
        Smoothing method: {st.session_state.app_state.smoothing_method if st.session_state.app_state.smoothing_enabled else 'N/A'}
        
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
        ID    Center          Amplitude       FWHM        Area           Fraction(%)
        {"-"*80}"""
                
                for c in deconv.components:
                    report += f"\n{c['id']:<4} {c['cen_linear']:<15.4e} {c['amp']:<15.4e} {c['fwhm']:<12.4f} {c['area']:<15.4e} {c['fraction_percent']:<10.2f}"
                
                report += f"""
        
        COMPONENTS (NORMALIZED TO MAX PEAK = 1):
        {"-"*80}
        ID    Center          Norm. Amplitude    Original Amplitude    Fraction(%)
        {"-"*80}"""
                
                for c in deconv.components:
                    norm_amp = c['amp'] / max_amp
                    report += f"\n{c['id']:<4} {c['cen_linear']:<15.4e} {norm_amp:<18.4f} {c['amp']:<20.4e} {c['fraction_percent']:<10.2f}"
                
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
                
                colors = plt.cm.Set3(np.linspace(0, 1, len(deconv.components)))
                for c, color in zip(deconv.components, colors):
                    y_component_norm = GaussianModel.gaussian(x_dense_log, c['amp_norm'], 
                                                            c['cen_log'], c['sigma_log']) 
                    y_component_norm = y_component_norm * deconv.y_max / max_amp
                    
                    ax_norm.fill_between(x_dense, 0, y_component_norm, 
                                        color=color, alpha=0.3, linewidth=0)
                    ax_norm.plot(x_dense, y_component_norm, '-', color=color, linewidth=2,
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
