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
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Any
import warnings
from scipy.optimize import OptimizeWarning
import time

# Suppress warnings
warnings.filterwarnings('ignore', category=OptimizeWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ==================== STATE MANAGEMENT ====================

@dataclass
class AppState:
    """Centralized application state"""
    # Data
    raw_x: Optional[np.ndarray] = None
    raw_y: Optional[np.ndarray] = None
    original_x: Optional[np.ndarray] = None
    original_y: Optional[np.ndarray] = None
    
    # Cropping
    crop_range: Optional[Tuple[float, float]] = None
    cropped_x: Optional[np.ndarray] = None
    cropped_y: Optional[np.ndarray] = None
    
    # Scale settings
    use_log_x: bool = True
    use_log_y: bool = False
    clip_negative: bool = True
    log_epsilon: float = 1e-12
    
    # Peak detection
    sensitivity: float = 0.03
    min_distance: int = 5
    peak_info: Optional[List[dict]] = None
    derivatives: Optional[Tuple] = None
    initial_params: Optional[List[float]] = None
    
    # Peak editing
    selected_peak_id: Optional[int] = None
    drag_mode: bool = False
    temp_peak_position: Optional[float] = None
    split_position: Optional[float] = None
    
    # Fitting options
    optimization_method: str = 'trf'
    maxfev: int = 10000
    
    # Results
    deconvolver: Optional['GaussianDeconvolver'] = None
    
    # Navigation
    current_step: int = 1
    
    def __post_init__(self):
        """Auto-update cropped data when raw data or crop range changes"""
        self.update_cropped()
    
    def update_cropped(self):
        """Apply cropping to raw data"""
        if self.raw_x is not None and self.raw_y is not None:
            if self.crop_range is not None:
                mask = (self.raw_x >= self.crop_range[0]) & (self.raw_x <= self.crop_range[1])
                self.cropped_x = self.raw_x[mask].copy()
                self.cropped_y = self.raw_y[mask].copy()
            else:
                self.cropped_x = self.raw_x.copy()
                self.cropped_y = self.raw_y.copy()
    
    def reset_analysis(self):
        """Reset all analysis results but keep raw data"""
        self.peak_info = None
        self.derivatives = None
        self.initial_params = None
        self.selected_peak_id = None
        self.drag_mode = False
        self.temp_peak_position = None
        self.deconvolver = None


# ==================== DATA PARSER ====================

class DataParser:
    """Universal parser for spectral data"""
    
    @staticmethod
    def parse_text(text: str) -> Tuple[np.ndarray, np.ndarray]:
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
        
        if len(x_data) == 0:
            return np.array([]), np.array([])
        
        # Sort by x
        x_data, y_data = zip(*sorted(zip(x_data, y_data)))
        return np.array(x_data), np.array(y_data)
    
    @staticmethod
    def auto_detect_scale(x: np.ndarray, y: np.ndarray) -> Tuple[bool, bool]:
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
    def check_negative_values(y: np.ndarray) -> Tuple[bool, float, int]:
        """Check for negative values in data"""
        negative_mask = y < 0
        n_negative = np.sum(negative_mask)
        if n_negative > 0:
            min_negative = np.min(y)
            return True, min_negative, n_negative
        return False, 0, 0


# ==================== GAUSSIAN MODEL ====================

class GaussianModel:
    """Model for sum of Gaussians"""
    
    @staticmethod
    def gaussian(x: np.ndarray, amp: float, cen: float, sigma: float) -> np.ndarray:
        """Gaussian function with safe sigma"""
        sigma = max(abs(sigma), 1e-12)
        return amp * np.exp(-(x - cen)**2 / (2 * sigma**2))
    
    @staticmethod
    def multi_gaussian(x: np.ndarray, *params: float) -> np.ndarray:
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
    def calculate_area(amp: float, sigma: float) -> float:
        """Area under Gaussian"""
        return amp * sigma * np.sqrt(2 * np.pi)
    
    @staticmethod
    def calculate_fwhm(sigma: float) -> float:
        """Full width at half maximum"""
        return 2 * np.sqrt(2 * np.log(2)) * sigma
    
    @staticmethod
    def estimate_sigma_from_width(x: np.ndarray, y: np.ndarray, peak_idx: int) -> float:
        """Estimate sigma from peak width at half height"""
        try:
            widths = peak_widths(y, [peak_idx], rel_height=0.5)
            fwhm = widths[0][0] * np.mean(np.diff(x))
            return fwhm / (2 * np.sqrt(2 * np.log(2)))
        except:
            # Fallback: use distance to nearest minimum
            left_min = np.argmin(y[:peak_idx]) if peak_idx > 0 else 0
            right_min = peak_idx + np.argmin(y[peak_idx:]) if peak_idx < len(y) - 1 else len(y) - 1
            width = (right_min - left_min) * np.mean(np.diff(x))
            return max(width * 0.2, 0.01)


# ==================== DERIVATIVE ANALYZER ====================

class DerivativeAnalyzer:
    """Analysis of first and second derivatives for peak detection"""
    
    @staticmethod
    def safe_savgol_filter(y: np.ndarray, window_length: int, polyorder: int) -> np.ndarray:
        """Safe Savitzky-Golay filter with fallback"""
        if len(y) < window_length:
            window_length = len(y) if len(y) % 2 == 1 else len(y) - 1
        
        if window_length < polyorder + 2:
            return y  # Return original if can't smooth
        
        try:
            return savgol_filter(y, window_length, polyorder)
        except:
            return y
    
    @staticmethod
    def calculate_derivatives(x: np.ndarray, y: np.ndarray, 
                            window_length: int = 11, polyorder: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate smoothed derivatives with fallback"""
        if len(x) < 5:
            dy = np.gradient(y, x)
            d2y = np.gradient(dy, x)
            return dy, d2y, y
        
        # Adaptive window length
        window_length = min(window_length, len(x) // 5 * 2 + 1)
        if window_length % 2 == 0:
            window_length += 1
        window_length = max(5, window_length)
        
        try:
            # Savitzky-Golay smoothing
            y_smooth = DerivativeAnalyzer.safe_savgol_filter(y, window_length, polyorder)
            dy = savgol_filter(y, window_length, polyorder, deriv=1, 
                              delta=np.mean(np.diff(x)))
            d2y = savgol_filter(y, window_length, polyorder, deriv=2, 
                               delta=np.mean(np.diff(x)))
        except:
            # Fallback to simple differences
            dy = np.gradient(y, x)
            d2y = np.gradient(dy, x)
            y_smooth = y
        
        return dy, d2y, y_smooth
    
    @staticmethod
    def find_peaks_by_derivatives(x: np.ndarray, y: np.ndarray, dy: np.ndarray, 
                                 d2y: np.ndarray, threshold: float = 0.01) -> List[int]:
        """Find peaks by zero crossing of first derivative and negative second derivative"""
        peaks = []
        y_max = np.max(y)
        for i in range(1, len(x) - 1):
            if (dy[i-1] > 0 and dy[i] <= 0) or (dy[i-1] >= 0 and dy[i] < 0):
                if d2y[i] < 0:
                    if y[i] > threshold * y_max:
                        peaks.append(i)
        return peaks


# ==================== FIT QUALITY ANALYZER ====================

class FitQualityAnalyzer:
    """Fit quality analysis"""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                         n_params: int) -> dict:
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
    def detect_autocorrelation(residuals: np.ndarray) -> bool:
        """Detect autocorrelation in residuals"""
        if len(residuals) < 10:
            return False
        
        diff = np.diff(residuals)
        dw = np.sum(diff**2) / np.sum(residuals**2)
        
        return dw < 1.5 or dw > 2.5


# ==================== GAUSSIAN DECONVOLVER ====================

class DataPreprocessor:
    """Handles data preprocessing for deconvolution"""
    
    def __init__(self, x: np.ndarray, y: np.ndarray, 
                 use_log_x: bool = True, use_log_y: bool = False,
                 clip_negative: bool = True, log_epsilon: float = 1e-12):
        self.x_original = x.copy()
        self.y_original = y.copy()
        self.use_log_x = use_log_x
        self.use_log_y = use_log_y
        self.clip_negative = clip_negative
        self.log_epsilon = log_epsilon
        
        # Sort by X
        sort_idx = np.argsort(self.x_original)
        self.x_sorted = self.x_original[sort_idx]
        self.y_sorted = self.y_original[sort_idx]
        
        # Check for negative values
        self.has_negative, self.min_negative, self.n_negative = DataParser.check_negative_values(self.y_sorted)
        
        # Prepare data for fitting
        self._prepare_fitting_data()
    
    def _prepare_fitting_data(self):
        """Prepare data for fitting with appropriate transformations"""
        # Handle negative values
        if self.has_negative and self.clip_negative:
            self.y_for_fitting = np.maximum(self.y_sorted, 0)
            self.negative_clipped = True
        else:
            self.y_for_fitting = self.y_sorted
            self.negative_clipped = False
        
        # Apply log transformations
        if self.use_log_x:
            # For log X, ensure positive values with safe transformation
            x_positive = np.maximum(self.x_sorted, self.log_epsilon)
            self.x = np.log10(x_positive)
            self.x_label = 'log₁₀(X)'
        else:
            self.x = self.x_sorted
            self.x_label = 'X'
        
        if self.use_log_y:
            # For log Y, use log1p for small values to avoid -inf
            y_positive = np.maximum(self.y_for_fitting, 0)
            # Use log1p for better numerical stability with small values
            self.y = np.log1p(y_positive / self.log_epsilon) / np.log(10)
            self.y_label = 'log₁₀(Y+ε)'
        else:
            self.y = self.y_for_fitting
            self.y_label = 'Y'
        
        # Normalize for fitting
        self.y_max = np.max(self.y) if np.max(self.y) > 0 else 1.0
        self.y_norm = self.y / self.y_max
    
    def get_original_y_at_x(self, x_linear: float) -> float:
        """Find original Y value closest to given X in linear space"""
        idx = np.argmin(np.abs(self.x_sorted - x_linear))
        return self.y_sorted[idx]


class PeakDetector:
    """Handles peak detection using various methods"""
    
    def __init__(self, preprocessor: DataPreprocessor):
        self.preprocessor = preprocessor
    
    def detect_peaks(self, sensitivity: float = 0.03, min_distance: int = 5) -> Tuple[List[int], List[dict], List[float], Tuple]:
        """Automatic peak detection using derivatives"""
        x = self.preprocessor.x
        y_norm = self.preprocessor.y_norm
        
        # Calculate derivatives with adaptive smoothing
        window_length = min(11, len(x) // 5 * 2 + 1)
        dy, d2y, y_smooth = DerivativeAnalyzer.calculate_derivatives(x, y_norm, window_length)
        
        # Peak search with different methods
        height_threshold = sensitivity * np.max(y_smooth)
        peaks1, _ = find_peaks(y_smooth, height=height_threshold, distance=min_distance)
        peaks2 = DerivativeAnalyzer.find_peaks_by_derivatives(x, y_smooth, dy, d2y, sensitivity)
        
        # Combine results
        all_peaks = sorted(set(np.concatenate([peaks1, peaks2])))
        
        # Filter close peaks
        filtered_peaks = []
        for peak in all_peaks:
            if not filtered_peaks or abs(x[peak] - x[filtered_peaks[-1]]) > min_distance * np.mean(np.diff(x)):
                filtered_peaks.append(peak)
        
        # Estimate parameters
        peak_info = []
        initial_params = []
        
        for peak_idx in filtered_peaks:
            cen = x[peak_idx]
            amp = y_smooth[peak_idx]
            
            # Estimate sigma
            sigma = GaussianModel.estimate_sigma_from_width(x, y_smooth, peak_idx)
            sigma = max(sigma, 0.01 * (np.max(x) - np.min(x)) / max(len(filtered_peaks), 1))
            
            # Get original values
            if self.preprocessor.use_log_x:
                x_linear = 10**x[peak_idx]
            else:
                x_linear = x[peak_idx]
            
            y_original_value = self.preprocessor.get_original_y_at_x(x_linear)
            
            peak_info.append({
                'index': peak_idx,
                'x': x[peak_idx],
                'x_linear': x_linear,
                'y': y_norm[peak_idx],
                'y_original': y_original_value,
                'amp_est': amp,
                'cen_est': cen,
                'sigma_est': sigma,
                'dy': dy[peak_idx],
                'd2y': d2y[peak_idx]
            })
            
            initial_params.extend([amp, cen, sigma])
        
        return filtered_peaks, peak_info, initial_params, (dy, d2y, y_smooth)


class GaussianFitter:
    """Handles fitting of Gaussian models"""
    
    def __init__(self, preprocessor: DataPreprocessor):
        self.preprocessor = preprocessor
        self.popt = None
        self.fit_y_norm = None
        self.components = []
        self.quality_metrics = {}
        self.convergence_history = []
        self.total_area = 0
    
    def fit(self, initial_params: List[float], method: str = 'trf', 
           maxfev: int = 10000, progress_callback=None) -> bool:
        """Perform fitting with progress tracking"""
        if len(initial_params) == 0:
            return False
        
        x = self.preprocessor.x
        y_norm = self.preprocessor.y_norm
        n_peaks = len(initial_params) // 3
        
        # Set bounds
        lower_bounds = []
        upper_bounds = []
        x_range = np.max(x) - np.min(x)
        
        for i in range(n_peaks):
            lower_bounds.extend([0, np.min(x), x_range * 0.001])
            upper_bounds.extend([2, np.max(x), x_range * 0.5])
        
        try:
            # Ensure initial_params are within bounds
            initial_params = np.array(initial_params)
            for i in range(len(initial_params)):
                initial_params[i] = np.clip(initial_params[i], lower_bounds[i], upper_bounds[i])
            
            # Fit with progress tracking
            if progress_callback:
                progress_callback(0.3, "Optimizing parameters...")
            
            popt, _ = curve_fit(
                GaussianModel.multi_gaussian,
                x,
                y_norm,
                p0=initial_params,
                bounds=(lower_bounds, upper_bounds),
                method=method,
                maxfev=maxfev
            )
            
            if progress_callback:
                progress_callback(0.8, "Calculating components...")
            
            self.popt = popt
            self.fit_y_norm = GaussianModel.multi_gaussian(x, *popt)
            
            # Extract components
            self._extract_components()
            
            # Calculate quality metrics
            self.quality_metrics = FitQualityAnalyzer.calculate_metrics(
                y_norm, self.fit_y_norm, len(popt)
            )
            
            if progress_callback:
                progress_callback(1.0, "Done!")
            
            return True
            
        except Exception as e:
            print(f"Fitting error: {e}")
            return False
    
    def _extract_components(self):
        """Extract individual Gaussian components from fit results"""
        self.components = []
        n_peaks = len(self.popt) // 3
        
        for i in range(n_peaks):
            amp_norm = self.popt[3*i]
            cen = self.popt[3*i + 1]
            sigma = abs(self.popt[3*i + 2])
            
            amp = amp_norm * self.preprocessor.y_max
            
            component_y_norm = GaussianModel.gaussian(self.preprocessor.x, amp_norm, cen, sigma)
            
            # Calculate area
            area = GaussianModel.calculate_area(amp_norm, sigma) * self.preprocessor.y_max
            
            # Convert center to linear if needed
            if self.preprocessor.use_log_x:
                cen_linear = 10**cen
            else:
                cen_linear = cen
            
            self.components.append({
                'id': i + 1,
                'amp_norm': amp_norm,
                'amp': amp,
                'cen_log': cen if self.preprocessor.use_log_x else None,
                'cen_linear': cen_linear,
                'sigma_log': sigma,
                'fwhm': GaussianModel.calculate_fwhm(sigma),
                'area': area,
                'fraction': 0,
                'y_norm': component_y_norm,
                'y_original': component_y_norm * self.preprocessor.y_max
            })
        
        # Calculate fractions
        self.total_area = sum([c['area'] for c in self.components])
        for c in self.components:
            c['fraction'] = c['area'] / self.total_area if self.total_area > 0 else 0
            c['fraction_percent'] = c['fraction'] * 100
    
    def remove_peak(self, peak_id: int) -> bool:
        """Remove a peak and refit"""
        if peak_id > len(self.components):
            return False
        
        new_params = []
        for i, c in enumerate(self.components):
            if i != peak_id - 1:
                new_params.extend([c['amp_norm'], c['cen_log'], c['sigma_log']])
        
        if len(new_params) == 0:
            return False
        
        return self.fit(new_params)
    
    def split_peak(self, peak_id: int, split_position: float) -> bool:
        """Split a peak into two at specified position"""
        if peak_id > len(self.components):
            return False
        
        peak = self.components[peak_id - 1]
        x_range = np.max(self.preprocessor.x) - np.min(self.preprocessor.x)
        
        new_params = []
        for i, c in enumerate(self.components):
            if i == peak_id - 1:
                amp1 = c['amp_norm'] * 0.6
                amp2 = c['amp_norm'] * 0.4
                
                # Split at specified position
                cen1 = split_position - c['sigma_log'] * 0.3
                cen2 = split_position + c['sigma_log'] * 0.3
                
                # Ensure centers are within range
                cen1 = np.clip(cen1, np.min(self.preprocessor.x), np.max(self.preprocessor.x))
                cen2 = np.clip(cen2, np.min(self.preprocessor.x), np.max(self.preprocessor.x))
                
                sigma1 = c['sigma_log'] * 0.7
                sigma2 = c['sigma_log'] * 0.7
                
                new_params.extend([amp1, cen1, sigma1])
                new_params.extend([amp2, cen2, sigma2])
            else:
                new_params.extend([c['amp_norm'], c['cen_log'], c['sigma_log']])
        
        return self.fit(new_params)
    
    def add_peak(self, position: float, amplitude: float = None) -> bool:
        """Add a new peak at specified position"""
        if amplitude is None:
            # Estimate amplitude from data
            idx = np.argmin(np.abs(self.preprocessor.x - position))
            amplitude = self.preprocessor.y_norm[idx] * 0.5
        
        # Estimate sigma based on local width
        x_range = np.max(self.preprocessor.x) - np.min(self.preprocessor.x)
        sigma = x_range * 0.02  # Default 2% of range
        
        new_params = []
        for c in self.components:
            new_params.extend([c['amp_norm'], c['cen_log'], c['sigma_log']])
        
        new_params.extend([amplitude, position, sigma])
        
        return self.fit(new_params)


class GaussianDeconvolver:
    """Main class for spectral deconvolution"""
    
    def __init__(self, x: np.ndarray, y: np.ndarray, 
                 use_log_x: bool = True, use_log_y: bool = False,
                 clip_negative: bool = True, log_epsilon: float = 1e-12):
        
        # Preprocess data
        self.preprocessor = DataPreprocessor(x, y, use_log_x, use_log_y, 
                                            clip_negative, log_epsilon)
        
        # Peak detector
        self.peak_detector = PeakDetector(self.preprocessor)
        
        # Fitter
        self.fitter = GaussianFitter(self.preprocessor)
        
        # Copy properties for backward compatibility
        self.x_original = self.preprocessor.x_original
        self.y_original = self.preprocessor.y_original
        self.x_sorted = self.preprocessor.x_sorted
        self.y_sorted = self.preprocessor.y_sorted
        self.x = self.preprocessor.x
        self.y = self.preprocessor.y
        self.y_norm = self.preprocessor.y_norm
        self.y_max = self.preprocessor.y_max
        self.use_log_x = use_log_x
        self.use_log_y = use_log_y
        self.x_label = self.preprocessor.x_label
        self.y_label = self.preprocessor.y_label
    
    @property
    def components(self):
        return self.fitter.components
    
    @property
    def fit_y_norm(self):
        return self.fitter.fit_y_norm
    
    @property
    def popt(self):
        return self.fitter.popt
    
    @property
    def quality_metrics(self):
        return self.fitter.quality_metrics
    
    @property
    def total_area(self):
        return self.fitter.total_area
    
    def auto_detect_peaks(self, sensitivity=0.03, min_distance=5):
        return self.peak_detector.detect_peaks(sensitivity, min_distance)
    
    def fit(self, initial_params=None, method='trf', maxfev=10000, progress_callback=None):
        if initial_params is None:
            _, _, initial_params, _ = self.auto_detect_peaks()
        return self.fitter.fit(initial_params, method, maxfev, progress_callback)
    
    def remove_peak(self, peak_id):
        return self.fitter.remove_peak(peak_id)
    
    def split_peak(self, peak_id, split_position):
        return self.fitter.split_peak(peak_id, split_position)
    
    def add_peak(self, position, amplitude=None):
        return self.fitter.add_peak(position, amplitude)
    
    def get_warning_messages(self) -> List[str]:
        """Get warning messages about data quality"""
        warnings = []
        if self.preprocessor.has_negative and self.preprocessor.clip_negative:
            warnings.append(f"⚠️ {self.preprocessor.n_negative} negative values were clipped to 0")
        elif self.preprocessor.has_negative:
            warnings.append(f"⚠️ Data contains {self.preprocessor.n_negative} negative values")
        
        if len(self.x) < 10:
            warnings.append(f"⚠️ Very few data points ({len(self.x)}) after cropping")
        
        return warnings


# ==================== PLOTTING UTILITIES ====================

class DeconvolutionPlotter:
    """Unified plotting utility for deconvolution results"""
    
    @staticmethod
    def plot_deconvolution(
        deconvolver: GaussianDeconvolver,
        show_components: bool = True,
        show_residuals: bool = False,
        original_scale: bool = True,
        interactive: bool = False,
        selected_peak_id: Optional[int] = None,
        temp_peak_position: Optional[float] = None,
        fig=None,
        ax=None,
        ax_res=None
    ):
        """
        Unified plotting function for all deconvolution visualizations
        
        Parameters:
        - deconvolver: GaussianDeconvolver with results
        - show_components: show individual peaks
        - show_residuals: show residuals on separate axis
        - original_scale: use original values (not normalized)
        - interactive: editing mode (for Step 4)
        - selected_peak_id: ID of selected peak for highlighting
        - temp_peak_position: temporary position for adding peak
        - fig, ax, ax_res: existing figure and axes to plot on
        """
        if fig is None:
            if show_residuals:
                fig, (ax, ax_res) = plt.subplots(2, 1, figsize=(12, 8), 
                                                gridspec_kw={'height_ratios': [3, 1]})
            else:
                fig, ax = plt.subplots(figsize=(12, 6))
        
        # Prepare data for display
        if original_scale:
            x_display = deconvolver.x_sorted
            y_display = deconvolver.y_sorted
            
            if deconvolver.use_log_x:
                ax.set_xscale('log')
            if deconvolver.use_log_y:
                ax.set_yscale('log')
        else:
            x_display = deconvolver.x
            y_display = deconvolver.y_norm
        
        # Main data
        ax.plot(x_display, y_display, 'o-', markersize=3, alpha=0.5, 
               color='black', label='Data', zorder=1)
        
        # Total fit
        if deconvolver.fit_y_norm is not None:
            y_fit = deconvolver.fit_y_norm
            if original_scale:
                y_fit = y_fit * deconvolver.y_max
            
            ax.plot(x_display, y_fit, 'r-', linewidth=2, 
                   label='Total Fit', zorder=2)
        
        # Components
        if show_components and deconvolver.components:
            colors = plt.cm.Set3(np.linspace(0, 1, len(deconvolver.components)))
            
            for i, (c, color) in enumerate(zip(deconvolver.components, colors)):
                # Generate dense x for smooth curve
                if original_scale and deconvolver.use_log_x:
                    x_dense = np.logspace(
                        np.log10(max(x_display[0], 1e-12)),
                        np.log10(x_display[-1]), 1000
                    )
                    x_dense_log = np.log10(x_dense)
                else:
                    x_dense = np.linspace(x_display[0], x_display[-1], 1000)
                    x_dense_log = x_dense if not deconvolver.use_log_x else np.log10(x_dense)
                
                y_comp = GaussianModel.gaussian(x_dense_log, c['amp_norm'], 
                                               c['cen_log'], c['sigma_log'])
                if original_scale:
                    y_comp = y_comp * deconvolver.y_max
                
                # Highlight selected peak
                linewidth = 3 if interactive and c['id'] == selected_peak_id else 1.5
                alpha = 0.5 if interactive and c['id'] == selected_peak_id else 0.3
                
                ax.fill_between(x_dense, 0, y_comp, color=color, alpha=alpha, zorder=3)
                ax.plot(x_dense, y_comp, '-', color=color, linewidth=linewidth,
                       label=f'Peak {c["id"]}: {c["fraction_percent"]:.1f}%', zorder=4)
        
        # Temporary peak (for adding)
        if interactive and temp_peak_position is not None:
            if deconvolver.use_log_x:
                x_temp = temp_peak_position
                x_temp_display = 10**temp_peak_position
            else:
                x_temp = temp_peak_position
                x_temp_display = temp_peak_position
            
            # Estimate amplitude from data
            idx = np.argmin(np.abs(deconvolver.x - x_temp))
            amp_temp = deconvolver.y_norm[idx] * 0.5
            
            # Plot temporary peak
            x_dense = np.linspace(x_display[0], x_display[-1], 100)
            if deconvolver.use_log_x:
                x_dense_log = np.log10(x_dense)
            else:
                x_dense_log = x_dense
            
            y_temp = GaussianModel.gaussian(x_dense_log, amp_temp, x_temp, 
                                           (np.max(deconvolver.x)-np.min(deconvolver.x))*0.02)
            if original_scale:
                y_temp = y_temp * deconvolver.y_max
            
            ax.plot(x_dense, y_temp, '--', color='purple', linewidth=2,
                   label='New Peak (preview)', zorder=5)
            ax.plot(x_temp_display, amp_temp * (deconvolver.y_max if original_scale else 1), 
                   'p', color='purple', markersize=10, markeredgecolor='black', zorder=6)
        
        # Residuals
        if show_residuals and 'Residuals' in deconvolver.quality_metrics:
            residuals = deconvolver.quality_metrics['Residuals']
            if original_scale:
                residuals = residuals * deconvolver.y_max
            
            ax_res.plot(x_display, residuals, 'o-', markersize=2, 
                       color='blue', alpha=0.5, label='Residuals')
            ax_res.axhline(y=0, color='r', linestyle='--', linewidth=1)
            ax_res.set_ylabel('Residuals')
            ax_res.set_xlabel(deconvolver.x_label)
            ax_res.grid(True, alpha=0.3)
            
            # Add quality metrics to residuals plot
            r2 = deconvolver.quality_metrics.get('R²', 0)
            rmse = deconvolver.quality_metrics.get('RMSE', 0)
            ax_res.text(0.02, 0.95, f'R^2 = {r2:.6f}\nRMSE = {rmse:.2e}',
                       transform=ax_res.transAxes, fontsize=9,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Labels and title
        if original_scale:
            x_label = 'X' + (' (log scale)' if deconvolver.use_log_x else '')
            y_label = 'Y' + (' (log scale)' if deconvolver.use_log_y else '')
        else:
            x_label = deconvolver.x_label
            y_label = 'Normalized Y'
        
        ax.set_xlabel(x_label, fontweight='bold')
        ax.set_ylabel(y_label, fontweight='bold')
        
        if interactive:
            ax.set_title('Peak Editing Mode - Click and drag to adjust peaks', 
                        fontweight='bold')
        
        # Legend
        if len(ax.get_legend_handles_labels()[0]) > 0:
            ax.legend(loc='upper right', fontsize=8, frameon=True, 
                     edgecolor='black', ncol=2 if len(deconvolver.components) > 5 else 1)
        
        ax.grid(True, alpha=0.3)
        
        # Scientific styling
        for axis in [ax, ax_res] if show_residuals else [ax]:
            if axis:
                axis.spines['top'].set_visible(True)
                axis.spines['right'].set_visible(True)
                axis.spines['bottom'].set_linewidth(1)
                axis.spines['left'].set_linewidth(1)
                axis.spines['top'].set_linewidth(1)
                axis.spines['right'].set_linewidth(1)
                axis.tick_params(direction='out', length=4, width=1)
        
        plt.tight_layout()
        return fig


# ==================== STREAMLIT UI COMPONENTS ====================

def render_step_1_data_loading(state: AppState):
    """Render Step 1: Data Loading"""
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
        st.info("""
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
        """)
        
        if st.button("📂 Load Data", type="primary", use_container_width=True):
            x, y = DataParser.parse_text(data_text)
            
            if len(x) > 0:
                state.raw_x = x
                state.raw_y = y
                state.reset_analysis()
                state.crop_range = None  # Reset crop
                state.update_cropped()
                state.current_step = 2
                st.rerun()
            else:
                st.error("Could not parse data. Check the format.")
    
    # Preview
    if state.raw_x is not None:
        st.subheader("Data Preview:")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(state.raw_x, state.raw_y, 'o-', markersize=3, linewidth=1)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title('Linear Scales')
        ax1.grid(True, alpha=0.3)
        
        if np.min(state.raw_x[state.raw_x > 0]) > 0:
            ax2.loglog(state.raw_x, np.maximum(state.raw_y, 1e-12), 
                      'o-', markersize=3, linewidth=1)
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_title('Log-Log Scales')
            ax2.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


def render_step_2_scale_settings(state: AppState):
    """Render Step 2: Scale Settings with cropping"""
    st.header("Step 2: Scale Settings & Cropping")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Scale Parameters")
        
        # Auto-detection
        if st.button("🔍 Auto-detect Scales", use_container_width=True):
            suggest_log_x, suggest_log_y = DataParser.auto_detect_scale(
                state.cropped_x if state.cropped_x is not None else state.raw_x,
                state.cropped_y if state.cropped_y is not None else state.raw_y
            )
            state.use_log_x = suggest_log_x
            state.use_log_y = suggest_log_y
            st.rerun()
        
        # Manual settings
        state.use_log_x = st.checkbox("Logarithmic X scale", value=state.use_log_x)
        state.use_log_y = st.checkbox("Logarithmic Y scale", value=state.use_log_y)
        
        # Negative values handling
        has_neg, min_neg, n_neg = DataParser.check_negative_values(state.cropped_y if state.cropped_y is not None else state.raw_y)
        if has_neg:
            st.warning(f"⚠️ Data contains {n_neg} negative values (min: {min_neg:.2e})")
            state.clip_negative = st.checkbox("Clip negative values to 0", value=True)
        
        st.markdown("---")
        st.subheader("X-Axis Cropping")
        
        if state.raw_x is not None:
            x_min = float(np.min(state.raw_x))
            x_max = float(np.max(state.raw_x))
            
            # Range slider
            crop_range = st.slider(
                "Select X range:",
                min_value=x_min,
                max_value=x_max,
                value=(state.crop_range[0] if state.crop_range else x_min,
                      state.crop_range[1] if state.crop_range else x_max),
                format="%.2e"
            )
            
            if crop_range != state.crop_range:
                state.crop_range = crop_range
                state.update_cropped()
                state.reset_analysis()  # Reset analysis when crop changes
                st.rerun()
            
            if state.crop_range is not None:
                if st.button("↺ Reset Crop", use_container_width=True):
                    state.crop_range = None
                    state.update_cropped()
                    state.reset_analysis()
                    st.rerun()
        
        st.markdown("---")
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("⬅️ Back", use_container_width=True):
                state.current_step = 1
                st.rerun()
        with col_b:
            if st.button("✅ Apply & Continue", type="primary", use_container_width=True):
                state.current_step = 3
                st.rerun()
    
    with col2:
        st.subheader("Preview:")
        
        # Use cropped data if available
        display_x = state.cropped_x if state.cropped_x is not None else state.raw_x
        display_y = state.cropped_y if state.cropped_y is not None else state.raw_y
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Apply selected scales for preview
        if state.use_log_x:
            ax.set_xscale('log')
        if state.use_log_y:
            ax.set_yscale('log')
        
        ax.plot(display_x, display_y, 'o-', markersize=3, linewidth=1, color='black')
        
        # Highlight cropped region on full data
        if state.crop_range is not None and state.raw_x is not None:
            ax.axvspan(state.crop_range[0], state.crop_range[1], 
                      alpha=0.2, color='green', label='Selected region')
        
        ax.set_xlabel('X' + (' (log)' if state.use_log_x else ''))
        ax.set_ylabel('Y' + (' (log)' if state.use_log_y else ''))
        ax.set_title('Data after cropping and scale application')
        ax.grid(True, alpha=0.3)
        if state.crop_range is not None:
            ax.legend()
        
        st.pyplot(fig)
        plt.close()
        
        # Show data statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Points", len(display_x))
        with col2:
            st.metric("X Range", f"{np.min(display_x):.2e} - {np.max(display_x):.2e}")
        with col3:
            st.metric("Y Range", f"{np.min(display_y):.2e} - {np.max(display_y):.2e}")


def render_step_3_peak_detection(state: AppState):
    """Render Step 3: Peak Detection"""
    st.header("Step 3: Peak Detection")
    
    # Create deconvolver if needed
    if state.deconvolver is None:
        state.original_x = state.raw_x.copy()
        state.original_y = state.raw_y.copy()
        
        state.deconvolver = GaussianDeconvolver(
            state.cropped_x if state.cropped_x is not None else state.raw_x,
            state.cropped_y if state.cropped_y is not None else state.raw_y,
            use_log_x=state.use_log_x,
            use_log_y=state.use_log_y,
            clip_negative=state.clip_negative
        )
        
        # Show warnings
        for warning in state.deconvolver.get_warning_messages():
            st.warning(warning)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Search Parameters")
        
        state.sensitivity = st.slider(
            "Sensitivity:",
            min_value=0.001,
            max_value=0.1,
            value=state.sensitivity,
            step=0.001,
            format="%.3f",
            help="Lower values detect more peaks"
        )
        
        state.min_distance = st.slider(
            "Minimum distance between peaks:",
            min_value=1,
            max_value=20,
            value=state.min_distance,
            step=1,
            help="Minimum index distance between peaks"
        )
        
        # Fitting options
        st.markdown("---")
        st.subheader("Fitting Options")
        
        state.optimization_method = st.selectbox(
            "Optimization method:",
            options=['trf', 'dogbox', 'lm'],
            index=0,
            help="trf: robust for bounds, dogbox: good for small problems, lm: Levenberg-Marquardt"
        )
        
        state.maxfev = st.number_input(
            "Max iterations:",
            min_value=1000,
            max_value=50000,
            value=state.maxfev,
            step=1000
        )
        
        st.markdown("---")
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("⬅️ Back", use_container_width=True):
                state.current_step = 2
                st.rerun()
        with col_b:
            if st.button("🔍 Find Peaks", type="primary", use_container_width=True):
                with st.spinner("Detecting peaks..."):
                    peaks, peak_info, initial_params, derivatives = state.deconvolver.auto_detect_peaks(
                        sensitivity=state.sensitivity,
                        min_distance=state.min_distance
                    )
                    state.peak_info = peak_info
                    state.derivatives = derivatives
                    state.initial_params = initial_params
                    st.success(f"Found {len(peak_info)} peaks!")
        
        if state.peak_info is not None:
            if st.button("✅ Fit & Continue", use_container_width=True):
                progress_bar = st.progress(0, text="Starting fit...")
                
                def update_progress(progress, message):
                    progress_bar.progress(progress, text=message)
                
                if state.deconvolver.fit(
                    initial_params=state.initial_params,
                    method=state.optimization_method,
                    maxfev=state.maxfev,
                    progress_callback=update_progress
                ):
                    state.current_step = 4
                    st.rerun()
                else:
                    st.error("Fitting failed. Try adjusting parameters.")
    
    with col2:
        if state.peak_info is not None and state.derivatives is not None:
            st.subheader(f"Peaks found: {len(state.peak_info)}")
            
            dy, d2y, y_smooth = state.derivatives
            
            # Create tabs for different plots
            tab1, tab2, tab3 = st.tabs(["📊 Peaks", "📈 Derivatives", "📋 Information"])
            
            with tab1:
                # Use unified plotting function
                fig, ax = plt.subplots(figsize=(10, 6))
                DeconvolutionPlotter.plot_deconvolution(
                    state.deconvolver,
                    show_components=False,
                    original_scale=True,
                    fig=fig,
                    ax=ax
                )
                
                # Overlay detected peaks
                for i, info in enumerate(state.peak_info):
                    ax.plot(info['x_linear'], info['y_original'], 'ro', 
                           markersize=8, markeredgecolor='darkred', 
                           markerfacecolor='yellow', zorder=5)
                    ax.text(info['x_linear'], info['y_original'] * 1.05, 
                           f'{i+1}', ha='center', fontweight='bold', 
                           fontsize=12, bbox=dict(boxstyle="round,pad=0.3", 
                                                 facecolor='white', alpha=0.8),
                           zorder=6)
                
                st.pyplot(fig)
                plt.close()
            
            with tab2:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
                
                # First derivative
                ax1.plot(state.deconvolver.x, dy, 'b-', linewidth=1.5)
                ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                ax1.set_xlabel(state.deconvolver.x_label)
                ax1.set_ylabel('dy/dx')
                ax1.set_title('First Derivative')
                ax1.grid(True, alpha=0.3)
                
                # Second derivative
                ax2.plot(state.deconvolver.x, d2y, 'g-', linewidth=1.5)
                ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                ax2.set_xlabel(state.deconvolver.x_label)
                ax2.set_ylabel('d²y/dx²')
                ax2.set_title('Second Derivative')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            with tab3:
                # Create table with peak information
                data = []
                for i, info in enumerate(state.peak_info):
                    data.append({
                        'Peak': i + 1,
                        'X Center': f"{info['x_linear']:.4e}",
                        'Y Amplitude': f"{info['y_original']:.4e}",
                        'Sigma Est.': f"{info['sigma_est']:.4f}",
                    })
                
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)
                
                # Statistics
                st.markdown("---")
                st.subheader("Detection Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Peaks", len(state.peak_info))
                with col2:
                    st.metric("X Range", f"{np.min(state.deconvolver.x_sorted):.2e} - {np.max(state.deconvolver.x_sorted):.2e}")
                with col3:
                    st.metric("Y Range", f"{np.min(state.deconvolver.y_sorted):.2e} - {np.max(state.deconvolver.y_sorted):.2e}")


def render_step_4_editing(state: AppState):
    """Render Step 4: Peak Editing with interactive features"""
    st.header("Step 4: Peak Editing")
    
    if state.deconvolver and state.deconvolver.components:
        col1, col2 = st.columns([1, 2.5])
        
        with col1:
            st.subheader("Peak Management")
            
            # Peak selection
            peak_options = {f"Peak {c['id']}: center = {c['cen_linear']:.2e}, fraction = {c['fraction_percent']:.1f}%": c['id'] 
                           for c in state.deconvolver.components}
            
            selected_peak = st.selectbox(
                "Select peak for editing:",
                options=list(peak_options.keys()),
                key="peak_selector"
            )
            
            if selected_peak:
                state.selected_peak_id = peak_options[selected_peak]
                peak = state.deconvolver.components[state.selected_peak_id - 1]
                
                # Split position slider
                min_x = np.min(state.deconvolver.x)
                max_x = np.max(state.deconvolver.x)
                
                split_pos = st.slider(
                    "Split position:",
                    min_value=float(min_x),
                    max_value=float(max_x),
                    value=float(peak['cen_log']),
                    format="%.4f",
                    key="split_slider"
                )
                
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    if st.button("✂️ Split", use_container_width=True):
                        with st.spinner("Splitting peak..."):
                            if state.deconvolver.split_peak(state.selected_peak_id, split_pos):
                                st.rerun()
                
                with col_b:
                    if st.button("🗑️ Remove", use_container_width=True):
                        with st.spinner("Removing peak..."):
                            if state.deconvolver.remove_peak(state.selected_peak_id):
                                st.rerun()
                
                with col_c:
                    if st.button("🔄 Refit", use_container_width=True):
                        with st.spinner("Refitting all peaks..."):
                            if state.deconvolver.fit(
                                initial_params=state.deconvolver.popt,
                                method=state.optimization_method,
                                maxfev=state.maxfev
                            ):
                                st.rerun()
            
            st.markdown("---")
            st.subheader("Add New Peak")
            
            # Interactive peak addition
            st.info("👆 Click on the graph to add a new peak at that position")
            
            if st.button("➕ Add Peak at Current Position", use_container_width=True):
                if state.temp_peak_position is not None:
                    with st.spinner("Adding new peak..."):
                        if state.deconvolver.add_peak(state.temp_peak_position):
                            state.temp_peak_position = None
                            st.rerun()
            
            st.markdown("---")
            
            # Quick fit options
            st.subheader("Quick Actions")
            
            col_opt1, col_opt2 = st.columns(2)
            with col_opt1:
                if st.button("🔄 Refit All", use_container_width=True):
                    with st.spinner("Refitting all peaks..."):
                        if state.deconvolver.fit(
                            initial_params=state.deconvolver.popt,
                            method=state.optimization_method,
                            maxfev=state.maxfev
                        ):
                            st.rerun()
            
            with col_opt2:
                if st.button("📊 Show Residuals", use_container_width=True):
                    state.show_residuals = not getattr(state, 'show_residuals', False)
                    st.rerun()
            
            st.markdown("---")
            
            col_back, col_next = st.columns(2)
            with col_back:
                if st.button("⬅️ Back", use_container_width=True):
                    state.current_step = 3
                    st.rerun()
            with col_next:
                if st.button("✅ Finish", type="primary", use_container_width=True):
                    state.current_step = 5
                    st.rerun()
        
        with col2:
            st.subheader("Deconvolution - Click to add peak")
            
            # Create interactive plot
            fig, ax = plt.subplots(figsize=(14, 8))
            
            DeconvolutionPlotter.plot_deconvolution(
                state.deconvolver,
                show_components=True,
                show_residuals=getattr(state, 'show_residuals', False),
                original_scale=True,
                interactive=True,
                selected_peak_id=state.selected_peak_id,
                temp_peak_position=state.temp_peak_position,
                fig=fig,
                ax=ax
            )
            
            # Add instructions
            ax.text(0.02, 0.98, 
                   "Click on graph to add peak\nDrag existing peaks to adjust",
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
            
            st.pyplot(fig)
            plt.close()
            
            # Simple click simulation (in real app, use plotly for true interactivity)
            st.caption("In production, use Plotly with click/drag events")
            
            # Manual position input as fallback
            col_pos1, col_pos2 = st.columns([3, 1])
            with col_pos1:
                x_min, x_max = ax.get_xlim()
                if state.use_log_x:
                    x_min, x_max = np.log10(x_min), np.log10(x_max)
                
                manual_pos = st.slider(
                    "Manual position:",
                    min_value=float(x_min),
                    max_value=float(x_max),
                    value=float(state.temp_peak_position if state.temp_peak_position is not None else (x_min + x_max)/2),
                    format="%.4f",
                    key="manual_pos_slider"
                )
                state.temp_peak_position = manual_pos
            
            with col_pos2:
                if st.button("Set", use_container_width=True):
                    st.rerun()
            
            # Display quality metrics
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            metrics = state.deconvolver.quality_metrics
            with col_m1:
                st.metric("R^2", f"{metrics.get('R²', 0):.6f}")
            with col_m2:
                st.metric("RMSE", f"{metrics.get('RMSE', 0):.2e}")
            with col_m3:
                st.metric("Peaks", len(state.deconvolver.components))
            with col_m4:
                st.metric("Total Area", f"{state.deconvolver.total_area:.2e}")


def render_step_5_results(state: AppState):
    """Render Step 5: Results"""
    st.header("Step 5: Results")
    
    if state.deconvolver and state.deconvolver.components:
        
        # Back button
        col_back, _ = st.columns([1, 5])
        with col_back:
            if st.button("⬅️ Back to Editing", use_container_width=True):
                state.current_step = 4
                st.rerun()
        
        st.markdown("---")
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["📊 Graphs", "📋 Table", "📈 Export"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Deconvolution Result")
                
                # Main plot
                fig, ax = plt.subplots(figsize=(10, 6))
                DeconvolutionPlotter.plot_deconvolution(
                    state.deconvolver,
                    show_components=True,
                    show_residuals=False,
                    original_scale=True,
                    fig=fig,
                    ax=ax
                )
                st.pyplot(fig)
                plt.close()
                
                # Residuals plot
                if 'Residuals' in state.deconvolver.quality_metrics:
                    st.subheader("Residuals")
                    fig, ax = plt.subplots(figsize=(10, 3))
                    residuals = state.deconvolver.quality_metrics['Residuals'] * state.deconvolver.y_max
                    ax.plot(state.deconvolver.x_sorted, residuals, 'o-', 
                           markersize=2, color='blue', alpha=0.5)
                    ax.axhline(y=0, color='r', linestyle='--')
                    ax.set_xlabel('X' + (' (log)' if state.use_log_x else ''))
                    ax.set_ylabel('Residuals')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close()
            
            with col2:
                st.subheader("Area Distribution")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Pie chart
                peaks = [f'{c["id"]}' for c in state.deconvolver.components]
                fractions = [c['fraction_percent'] for c in state.deconvolver.components]
                colors = plt.cm.Set3(np.linspace(0, 1, len(peaks)))
                ax1.pie(fractions, labels=peaks, autopct='%1.1f%%',
                       colors=colors, startangle=90,
                       textprops={'fontweight': 'bold'})
                ax1.set_title('Area Distribution', fontweight='bold')
                
                # Bar chart
                centers = [c['cen_linear'] for c in state.deconvolver.components]
                areas = [c['area'] for c in state.deconvolver.components]
                
                if state.use_log_x:
                    ax2.set_xscale('log')
                
                bars = ax2.bar(range(len(centers)), areas, 
                              tick_label=[f'{c:.2e}' for c in centers],
                              color='steelblue', edgecolor='black', alpha=0.7)
                ax2.set_xlabel('Peak Center', fontweight='bold')
                ax2.set_ylabel('Area', fontweight='bold')
                ax2.set_title('Peak Areas', fontweight='bold')
                ax2.tick_params(axis='x', rotation=45)
                ax2.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        
        with tab2:
            st.subheader("Results Table")
            
            data = []
            for c in state.deconvolver.components:
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
            
            # Quality metrics
            metrics = state.deconvolver.quality_metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("R^2", f"{metrics.get('R²', 0):.6f}")
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
                    } for c in state.deconvolver.components])
                    
                    csv_peaks = df_peaks.to_csv(index=False)
                    
                    st.download_button(
                        label="Download Peaks CSV",
                        data=csv_peaks,
                        file_name=f"deconvolution_peaks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Fitting data
                    if 'Residuals' in state.deconvolver.quality_metrics:
                        df_fit = pd.DataFrame({
                            'X': state.deconvolver.x_sorted,
                            'Y_original': state.deconvolver.y_sorted,
                            'Y_fit': state.deconvolver.fit_y_norm * state.deconvolver.y_max,
                            'Residuals': state.deconvolver.quality_metrics['Residuals'] * state.deconvolver.y_max
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
Number of points: {len(state.deconvolver.x_sorted)}
X range: [{state.deconvolver.x_sorted[0]:.2e}, {state.deconvolver.x_sorted[-1]:.2e}]
Logarithmic X scale: {state.deconvolver.use_log_x}
Logarithmic Y scale: {state.deconvolver.use_log_y}
Cropped: {state.crop_range is not None}

QUALITY METRICS:
{"-"*40}
R²: {state.deconvolver.quality_metrics.get('R²', 0):.6f}
AIC: {state.deconvolver.quality_metrics.get('AIC', 0):.2f}
BIC: {state.deconvolver.quality_metrics.get('BIC', 0):.2f}
χ²: {state.deconvolver.quality_metrics.get('χ²', 0):.2e}
RMSE: {state.deconvolver.quality_metrics.get('RMSE', 0):.2e}

COMPONENTS:
{"-"*80}
ID    Center          Amplitude       FWHM        Area           Fraction(%)
{"-"*80}"""
                    
                    for c in state.deconvolver.components:
                        report += f"\n{c['id']:<4} {c['cen_linear']:<15.4e} {c['amp']:<15.4e} {c['fwhm']:<12.4f} {c['area']:<15.4e} {c['fraction_percent']:<10.2f}"
                    
                    report += f"\n{'='*80}\nTotal area: {state.deconvolver.total_area:.6e}\n{'='*80}"
                    
                    st.download_button(
                        label="Download Report",
                        data=report,
                        file_name=f"deconvolution_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
            
            st.markdown("---")
            
            if st.button("🔄 New Analysis", use_container_width=True):
                # Reset but keep raw data
                state.reset_analysis()
                state.current_step = 1
                st.rerun()


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
0.00002409372325564147, 0.050643083213856696
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
0.026775191717066393, 0.14308684579163936
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


# ==================== MAIN APP ====================

def main():
    """Main application"""
    
    # Page configuration
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
        'ytick.major.size': 4,
        'legend.fontsize': 10,
        'legend.frameon': True,
        'legend.edgecolor': 'black',
        'figure.dpi': 600,
        'savefig.dpi': 600,
        'figure.facecolor': 'white',
        'lines.linewidth': 1.5,
    })
    
    # Initialize state
    if 'app_state' not in st.session_state:
        st.session_state.app_state = AppState()
    
    state = st.session_state.app_state
    
    # Title
    st.title("📊 Gaussian Deconvolution of Spectral Data")
    st.markdown("---")
    
    # Sidebar navigation
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
            if step_num < state.current_step:
                st.success(f"✅ {step_name}")
            elif step_num == state.current_step:
                st.info(f"▶️ {step_name}")
            else:
                st.write(f"⏳ {step_name}")
        
        st.markdown("---")
        
        if st.button("🔄 Start Over", use_container_width=True):
            # Reset everything
            st.session_state.app_state = AppState()
            st.rerun()
    
    # Render current step
    if state.current_step == 1:
        render_step_1_data_loading(state)
    elif state.current_step == 2:
        render_step_2_scale_settings(state)
    elif state.current_step == 3:
        render_step_3_peak_detection(state)
    elif state.current_step == 4:
        render_step_4_editing(state)
    elif state.current_step == 5:
        render_step_5_results(state)


if __name__ == "__main__":
    main()

