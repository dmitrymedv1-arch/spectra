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
                    # Replace negative values with 0
                    if y < 0:
                        y = 0
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


class DerivativeAnalyzer:
    """Analysis of first and second derivatives for peak detection"""
    
    @staticmethod
    def calculate_derivatives(x, y, window_length=11, polyorder=3):
        """Calculate smoothed derivatives"""
        if len(x) < window_length:
            window_length = len(x) if len(x) % 2 == 1 else len(x) - 1
        
        if window_length < polyorder + 2:
            return np.gradient(y, x), np.gradient(np.gradient(y, x), x), y
        
        # Savitzky-Golay smoothing
        y_smooth = savgol_filter(y, window_length, polyorder)
        dy = savgol_filter(y, window_length, polyorder, deriv=1, delta=np.mean(np.diff(x)))
        d2y = savgol_filter(y, window_length, polyorder, deriv=2, delta=np.mean(np.diff(x)))
        
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
        """Gaussian function"""
        return amp * np.exp(-(x - cen)**2 / (2 * max(sigma, 1e-12)**2))
    
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

class GaussianDeconvolver:
    """Class for spectral deconvolution"""
    
    def __init__(self, x_linear, y_original, use_log_x=True, use_log_y=False):
        self.x_linear = np.array(x_linear)
        self.y_original = np.array(y_original)
        self.use_log_x = use_log_x
        self.use_log_y = use_log_y
        
        # Sort by X
        sort_idx = np.argsort(self.x_linear)
        self.x_linear = self.x_linear[sort_idx]
        self.y_original = self.y_original[sort_idx]
        
        # Replace negative Y with 0
        self.y_original = np.maximum(self.y_original, 0)
        
        # Apply logarithmic transformations
        eps = 1e-12
        if use_log_x:
            self.x = np.log10(np.maximum(self.x_linear, eps))
            self.x_label = 'log₁₀(X)'
        else:
            self.x = self.x_linear
            self.x_label = 'X'
        
        if use_log_y:
            self.y = np.log10(np.maximum(self.y_original, eps))
            self.y_label = 'log₁₀(Y)'
        else:
            self.y = self.y_original
            self.y_label = 'Y'
        
        # Normalization for stability
        self.y_max = np.max(np.abs(self.y))
        if self.y_max > 0:
            self.y_norm = self.y / self.y_max
        else:
            self.y_norm = self.y
        
        # Results
        self.components = []
        self.fit_y_norm = None
        self.popt = None
        self.quality_metrics = {}
        self.convergence_history = []
        self.total_area = 0
        
        # For compatibility
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
            
            try:
                widths = peak_widths(y_smooth, [peak_idx], rel_height=0.5)
                fwhm = widths[0][0] * np.mean(np.diff(self.x))
                sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
            except:
                sigma = np.sqrt(1/abs(d2y[peak_idx])) if abs(d2y[peak_idx]) > 1e-10 else 0.1
            
            sigma = max(sigma, 0.01 * (np.max(self.x) - np.min(self.x)) / max(len(filtered_peaks), 1))
            
            peak_info.append({
                'index': peak_idx,
                'x': self.x[peak_idx],
                'x_linear': 10**self.x[peak_idx] if self.use_log_x else self.x[peak_idx],
                'y': self.y[peak_idx],
                'amp_est': amp,
                'cen_est': cen,
                'sigma_est': sigma,
                'dy': dy[peak_idx],
                'd2y': d2y[peak_idx]
            })
            
            initial_params.extend([amp, cen, sigma])
        
        return filtered_peaks, peak_info, initial_params, (dy, d2y, y_smooth)
    
    def fit(self, initial_params=None, maxfev=10000):
        """Perform fitting"""
        if initial_params is None:
            _, _, initial_params, _ = self.auto_detect_peaks()
        
        if len(initial_params) == 0:
            return False
        
        n_peaks = len(initial_params) // 3
        
        # Set bounds
        lower_bounds = []
        upper_bounds = []
        x_range = np.max(self.x) - np.min(self.x)
        y_range = np.max(self.y_norm) - np.min(self.y_norm)
        
        for i in range(n_peaks):
            lower_bounds.extend([0, np.min(self.x), x_range * 0.001])
            upper_bounds.extend([2 * np.max(self.y_norm), np.max(self.x), x_range * 0.5])
        
        try:
            # Ensure initial_params are within bounds
            initial_params = np.array(initial_params)
            for i in range(len(initial_params)):
                initial_params[i] = np.clip(initial_params[i], lower_bounds[i], upper_bounds[i])
            
            popt, _ = curve_fit(
                GaussianModel.multi_gaussian,
                self.x,
                self.y_norm,
                p0=initial_params,
                bounds=(lower_bounds, upper_bounds),
                maxfev=maxfev
            )
            
            self.popt = popt
            self.fit_y_norm = GaussianModel.multi_gaussian(self.x, *popt)
            
            # Extract components
            self.components = []
            for i in range(n_peaks):
                amp_norm = popt[3*i]
                cen = popt[3*i + 1]
                sigma = abs(popt[3*i + 2])
                
                amp = amp_norm * self.y_max
                area = GaussianModel.calculate_area(amp_norm, sigma) * self.y_max
                
                component_y_norm = GaussianModel.gaussian(self.x, amp_norm, cen, sigma)
                
                self.components.append({
                    'id': i + 1,
                    'amp_norm': amp_norm,
                    'amp': amp,
                    'cen_log': cen if self.use_log_x else None,
                    'cen_linear': 10**cen if self.use_log_x else cen,
                    'sigma_log': sigma,
                    'fwhm': GaussianModel.calculate_fwhm(sigma),
                    'area': area,
                    'fraction': 0,
                    'y_norm': component_y_norm
                })
            
            # Calculate statistics
            self.total_area = sum([c['area'] for c in self.components])
            for c in self.components:
                c['fraction'] = c['area'] / self.total_area if self.total_area > 0 else 0
                c['fraction_percent'] = c['fraction'] * 100
            
            # Quality metrics
            self.quality_metrics = FitQualityAnalyzer.calculate_metrics(
                self.y_norm, self.fit_y_norm, len(popt)
            )
            
            return True
            
        except Exception as e:
            st.error(f"Fitting error: {e}")
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
        
        return self.fit(initial_params=new_params)
    
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
        
        return self.fit(initial_params=new_params)
    
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


# ==================== STATE INITIALIZATION ====================

if 'deconvolver' not in st.session_state:
    st.session_state.deconvolver = None
if 'raw_x' not in st.session_state:
    st.session_state.raw_x = None
if 'raw_y' not in st.session_state:
    st.session_state.raw_y = None
if 'peak_info' not in st.session_state:
    st.session_state.peak_info = None
if 'derivatives' not in st.session_state:
    st.session_state.derivatives = None
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'use_log_x' not in st.session_state:
    st.session_state.use_log_x = True
if 'use_log_y' not in st.session_state:
    st.session_state.use_log_y = False
if 'sensitivity' not in st.session_state:
    st.session_state.sensitivity = 0.03
if 'min_distance' not in st.session_state:
    st.session_state.min_distance = 5
if 'split_position' not in st.session_state:
    st.session_state.split_position = None


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
        if step_num < st.session_state.current_step:
            st.success(f"✅ {step_name}")
        elif step_num == st.session_state.current_step:
            st.info(f"▶️ {step_name}")
        else:
            st.write(f"⏳ {step_name}")
    
    st.markdown("---")
    
    # Reset button
    if st.button("🔄 Start Over", use_container_width=True):
        for key in ['deconvolver', 'raw_x', 'raw_y', 'peak_info', 'derivatives', 'split_position']:
            if key in st.session_state:
                st.session_state[key] = None
        st.session_state.current_step = 1
        st.rerun()


# ==================== STEP 1: DATA LOADING ====================

if st.session_state.current_step == 1:
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
                st.session_state.raw_x = x
                st.session_state.raw_y = y
                st.session_state.current_step = 2
                st.rerun()
            else:
                st.error("Could not parse data. Check the format.")
    
    # Preview
    if st.session_state.raw_x is not None:
        st.subheader("Data Preview:")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(st.session_state.raw_x, st.session_state.raw_y, 'o-', markersize=3, linewidth=1)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title('Linear Scales')
        ax1.grid(True, alpha=0.3)
        
        if np.min(st.session_state.raw_x[st.session_state.raw_x > 0]) > 0:
            ax2.loglog(st.session_state.raw_x, np.maximum(st.session_state.raw_y, 1e-12), 
                      'o-', markersize=3, linewidth=1)
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_title('Log-Log Scales')
            ax2.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ==================== STEP 2: SCALE SETTINGS ====================

elif st.session_state.current_step == 2:
    st.header("Step 2: Scale Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Scale Parameters")
        
        # Auto-detection
        if st.button("🔍 Auto-detect Scales", use_container_width=True):
            suggest_log_x, suggest_log_y = DataParser.auto_detect_scale(
                st.session_state.raw_x, st.session_state.raw_y
            )
            st.session_state.use_log_x = suggest_log_x
            st.session_state.use_log_y = suggest_log_y
            st.rerun()
        
        # Manual settings
        st.session_state.use_log_x = st.checkbox("Logarithmic X scale", value=st.session_state.use_log_x)
        st.session_state.use_log_y = st.checkbox("Logarithmic Y scale", value=st.session_state.use_log_y)
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("⬅️ Back", use_container_width=True):
                st.session_state.current_step = 1
                st.rerun()
        with col_b:
            if st.button("✅ Apply & Continue", type="primary", use_container_width=True):
                st.session_state.current_step = 3
                st.rerun()
    
    with col2:
        st.subheader("Preview:")
        
        # Visualization with selected scales
        fig, ax = plt.subplots(figsize=(8, 5))
        
        x = st.session_state.raw_x
        y = st.session_state.raw_y
        
        # Handle negative values for log scales
        if st.session_state.use_log_x:
            mask = x > 0
            x = x[mask]
            y = y[mask]
            ax.set_xscale('log')
        
        if st.session_state.use_log_y:
            mask = y > 0
            x = x[mask]
            y = y[mask]
            ax.set_yscale('log')
        
        ax.plot(x, y, 'o-', markersize=3, linewidth=1)
        ax.set_xlabel('X' + (' (log)' if st.session_state.use_log_x else ''))
        ax.set_ylabel('Y' + (' (log)' if st.session_state.use_log_y else ''))
        ax.set_title('Data after scale application')
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        plt.close()


# ==================== STEP 3: PEAK DETECTION ====================

elif st.session_state.current_step == 3:
    st.header("Step 3: Peak Detection")
    
    # Create deconvolver if not yet created
    if st.session_state.deconvolver is None:
        st.session_state.deconvolver = GaussianDeconvolver(
            st.session_state.raw_x,
            st.session_state.raw_y,
            use_log_x=st.session_state.use_log_x,
            use_log_y=st.session_state.use_log_y
        )
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Search Parameters")
        
        st.session_state.sensitivity = st.slider(
            "Sensitivity:",
            min_value=0.001,
            max_value=0.1,
            value=st.session_state.sensitivity,
            step=0.001,
            format="%.3f"
        )
        
        st.session_state.min_distance = st.slider(
            "Minimum distance between peaks:",
            min_value=1,
            max_value=20,
            value=st.session_state.min_distance,
            step=1
        )
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("⬅️ Back", use_container_width=True):
                st.session_state.current_step = 2
                st.rerun()
        with col_b:
            if st.button("🔍 Find Peaks", type="primary", use_container_width=True):
                peaks, peak_info, initial_params, derivatives = st.session_state.deconvolver.auto_detect_peaks(
                    sensitivity=st.session_state.sensitivity,
                    min_distance=st.session_state.min_distance
                )
                st.session_state.peak_info = peak_info
                st.session_state.derivatives = derivatives
                st.session_state.initial_params = initial_params
        
        if st.session_state.peak_info is not None:
            if st.button("✅ Confirm Peaks", use_container_width=True):
                if st.session_state.deconvolver.fit(initial_params=st.session_state.initial_params):
                    st.session_state.current_step = 4
                    st.rerun()
                else:
                    st.error("Fitting failed. Try adjusting parameters.")
    
    with col2:
        if st.session_state.peak_info is not None and st.session_state.derivatives is not None:
            st.subheader(f"Peaks found: {len(st.session_state.peak_info)}")
            
            dy, d2y, y_smooth = st.session_state.derivatives
            
            # Create tabs for different plots
            tab1, tab2, tab3 = st.tabs(["📊 Peaks", "📈 Derivatives", "📋 Information"])
            
            with tab1:
                fig, ax = plt.subplots(figsize=(10, 5))
                
                ax.plot(st.session_state.deconvolver.x, st.session_state.deconvolver.y_norm, 
                       'o-', markersize=3, alpha=0.5, label='Data', color='black')
                ax.plot(st.session_state.deconvolver.x, y_smooth, 
                       'r-', linewidth=2, label='Smoothed')
                
                for i, info in enumerate(st.session_state.peak_info):
                    ax.plot(info['x'], info['y'], 'ro', markersize=8, markeredgecolor='darkred')
                    ax.text(info['x'], info['y']*1.05, f'{i+1}', ha='center', fontweight='bold')
                
                ax.set_xlabel(st.session_state.deconvolver.x_label)
                ax.set_ylabel('Normalized Y')
                ax.set_title('Detected Peaks')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                plt.close()
            
            with tab2:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
                
                ax1.plot(st.session_state.deconvolver.x, dy, 'b-', linewidth=1.5)
                ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                ax1.set_xlabel(st.session_state.deconvolver.x_label)
                ax1.set_ylabel('dy/dx')
                ax1.set_title('First Derivative')
                ax1.grid(True, alpha=0.3)
                
                ax2.plot(st.session_state.deconvolver.x, d2y, 'g-', linewidth=1.5)
                ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                ax2.set_xlabel(st.session_state.deconvolver.x_label)
                ax2.set_ylabel('d²y/dx²')
                ax2.set_title('Second Derivative')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            with tab3:
                data = []
                for i, info in enumerate(st.session_state.peak_info):
                    data.append({
                        'Peak': i + 1,
                        'Center (log)': f"{info['x']:.4f}",
                        'Center': f"{info['x_linear']:.2e}",
                        'Amplitude': f"{info['y']:.4f}",
                        'Sigma': f"{info['sigma_est']:.4f}"
                    })
                
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)

# ==================== STEP 4: EDITING ====================

elif st.session_state.current_step == 4:
    st.header("Step 4: Peak Editing")
    
    if st.session_state.deconvolver and st.session_state.deconvolver.components:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Peak Management")
            
            # Peak selection
            peak_options = {f"Peak {c['id']}: center = {c['cen_linear']:.2e}, fraction = {c['fraction_percent']:.1f}%": c['id'] 
                           for c in st.session_state.deconvolver.components}
            
            selected_peak = st.selectbox(
                "Select peak for editing:",
                options=list(peak_options.keys())
            )
            
            if selected_peak:
                peak_id = peak_options[selected_peak]
                
                # Split position slider
                peak = st.session_state.deconvolver.components[peak_id - 1]
                min_x = np.min(st.session_state.deconvolver.x)
                max_x = np.max(st.session_state.deconvolver.x)
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
                        if st.session_state.deconvolver.split_peak(peak_id, split_position):
                            st.rerun()
                
                with col_b:
                    if st.button("🗑️ Remove Peak", use_container_width=True):
                        if st.session_state.deconvolver.remove_peak(peak_id):
                            st.rerun()
                
                if st.button("🔄 Recalculate All", use_container_width=True):
                    if st.session_state.deconvolver.fit(initial_params=st.session_state.deconvolver.popt):
                        st.rerun()
            
            st.markdown("---")
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("⬅️ Back", use_container_width=True):
                    st.session_state.current_step = 3
                    st.rerun()
            with col_b:
                if st.button("✅ Finish Editing", type="primary", use_container_width=True):
                    st.session_state.current_step = 5
                    st.rerun()
        
        with col2:
            st.subheader("Current Deconvolution")
            
            # Use the new scientific Plotly figure
            fig = st.session_state.deconvolver.create_scientific_plotly_figure()
            st.plotly_chart(fig, use_container_width=True)

# ==================== STEP 5: RESULTS ====================

elif st.session_state.current_step == 5:
    st.header("Step 5: Results")
    
    if st.session_state.deconvolver and st.session_state.deconvolver.components:
        
        # Back button at the top
        col_back, _ = st.columns([1, 5])
        with col_back:
            if st.button("⬅️ Back to Editing", use_container_width=True):
                st.session_state.current_step = 4
                st.rerun()
        
        st.markdown("---")
        
        # Create tabs for results
        tab1, tab2, tab3 = st.tabs(["📊 Graphs", "📋 Table", "📈 Export"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Deconvolution Result")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                ax.scatter(st.session_state.deconvolver.x_linear, 
                          st.session_state.deconvolver.y_original, 
                          s=10, alpha=0.5, color='black', label='Data')
                
                if st.session_state.deconvolver.use_log_x:
                    x_dense = np.logspace(np.log10(np.maximum(np.min(st.session_state.deconvolver.x_linear[
                        st.session_state.deconvolver.x_linear>0]), 1e-12)),
                        np.log10(np.max(st.session_state.deconvolver.x_linear)), 1000)
                    x_dense_log = np.log10(x_dense)
                    ax.set_xscale('log')
                else:
                    x_dense = np.linspace(np.min(st.session_state.deconvolver.x_linear), 
                                         np.max(st.session_state.deconvolver.x_linear), 1000)
                    x_dense_log = x_dense
                
                colors = plt.cm.Set3(np.linspace(0, 1, len(st.session_state.deconvolver.components)))
                for c, color in zip(st.session_state.deconvolver.components, colors):
                    y_component = GaussianModel.gaussian(x_dense_log, c['amp_norm'], 
                                                        c['cen_log'], c['sigma_log']) * st.session_state.deconvolver.y_max
                    ax.plot(x_dense, y_component, '-', color=color, linewidth=2,
                           label=f'Peak {c["id"]}: {c["fraction_percent"]:.1f}%')
                
                y_total = GaussianModel.multi_gaussian(x_dense_log, *st.session_state.deconvolver.popt) * st.session_state.deconvolver.y_max
                ax.plot(x_dense, y_total, 'r--', linewidth=2, label='Total Fit')
                
                ax.set_xlabel('X', fontweight='bold')
                ax.set_ylabel('Y', fontweight='bold')
                ax.set_title('Deconvolution Result', fontweight='bold')
                ax.legend(loc='upper right', fontsize=8, frameon=True, edgecolor='black')
                ax.grid(True, alpha=0.3)
                
                # Scientific styling
                ax.spines['top'].set_visible(True)
                ax.spines['right'].set_visible(True)
                ax.spines['bottom'].set_linewidth(1)
                ax.spines['left'].set_linewidth(1)
                ax.spines['top'].set_linewidth(1)
                ax.spines['right'].set_linewidth(1)
                ax.tick_params(direction='out', length=4, width=1)
                
                st.pyplot(fig)
                plt.close()
            
            with col2:
                st.subheader("Area Distribution")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Pie chart
                peaks = [f'{c["id"]}' for c in st.session_state.deconvolver.components]
                fractions = [c['fraction_percent'] for c in st.session_state.deconvolver.components]
                colors = plt.cm.Set3(np.linspace(0, 1, len(peaks)))
                ax1.pie(fractions, labels=peaks, autopct='%1.1f%%',
                       colors=colors, startangle=90,
                       textprops={'fontweight': 'bold'})
                ax1.set_title('Area Distribution', fontweight='bold')
                
                # Bar chart
                centers = [c['cen_linear'] for c in st.session_state.deconvolver.components]
                areas = [c['area'] for c in st.session_state.deconvolver.components]
                
                if st.session_state.deconvolver.use_log_x:
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
            for c in st.session_state.deconvolver.components:
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
            
            metrics = st.session_state.deconvolver.quality_metrics
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
                    } for c in st.session_state.deconvolver.components])
                    
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
                    if 'Residuals' in st.session_state.deconvolver.quality_metrics:
                        df_fit = pd.DataFrame({
                            'X_original': st.session_state.deconvolver.x_linear,
                            'Y_original': st.session_state.deconvolver.y_original,
                            'Y_fit': st.session_state.deconvolver.fit_y_norm * st.session_state.deconvolver.y_max,
                            'Residuals': st.session_state.deconvolver.quality_metrics['Residuals'] * st.session_state.deconvolver.y_max
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
Number of points: {len(st.session_state.deconvolver.x_linear)}
X range: [{st.session_state.deconvolver.x_linear[0]:.2e}, {st.session_state.deconvolver.x_linear[-1]:.2e}]
Logarithmic X scale: {st.session_state.deconvolver.use_log_x}

QUALITY METRICS:
{"-"*40}
R²: {st.session_state.deconvolver.quality_metrics.get('R²', 0):.6f}
AIC: {st.session_state.deconvolver.quality_metrics.get('AIC', 0):.2f}
BIC: {st.session_state.deconvolver.quality_metrics.get('BIC', 0):.2f}
χ²: {st.session_state.deconvolver.quality_metrics.get('χ²', 0):.2e}
RMSE: {st.session_state.deconvolver.quality_metrics.get('RMSE', 0):.2e}

COMPONENTS:
{"-"*80}
ID    Center          Amplitude       FWHM        Area           Fraction(%)
{"-"*80}"""
                    
                    for c in st.session_state.deconvolver.components:
                        report += f"\n{c['id']:<4} {c['cen_linear']:<15.4e} {c['amp']:<15.4e} {c['fwhm']:<12.4f} {c['area']:<15.4e} {c['fraction_percent']:<10.2f}"
                    
                    report += f"\n{'='*80}\nTotal area: {st.session_state.deconvolver.total_area:.6e}\n{'='*80}"
                    
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
                    for key in ['deconvolver', 'raw_x', 'raw_y', 'peak_info', 'derivatives', 'split_position']:
                        if key in st.session_state:
                            st.session_state[key] = None
                    st.session_state.current_step = 1
                    st.rerun()
