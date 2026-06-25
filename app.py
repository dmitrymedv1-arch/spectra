import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks, peak_widths, find_peaks_cwt
from scipy.optimize import curve_fit, least_squares
from scipy.ndimage import gaussian_filter1d
from scipy.special import voigt_profile
from scipy.sparse import diags
from scipy.linalg import solve_banded
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from datetime import datetime
import io
import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any, Callable
import time
import json
import os
from matplotlib.ticker import MaxNLocator

# Попытка импорта pybaselines для улучшенной коррекции фона
try:
    from pybaselines import Baseline
    HAS_PYBASELINES = True
except ImportError:
    HAS_PYBASELINES = False
    warnings.warn("pybaselines not installed. Install with: pip install pybaselines")

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
    point_range_start: Optional[int] = None
    point_range_end: Optional[int] = None
    clip_negative: bool = True
    fitting_method: str = 'trf'
    max_nfev: int = 5000
    show_warnings: bool = True
    baseline_method: str = 'arpls'
    baseline_degree: int = 1
    baseline_lam: float = 1e5
    baseline_p: float = 0.01
    fit_quality: str = 'balanced'
    last_popt: Optional[np.ndarray] = None
    pending_split: Optional[Tuple[int, float]] = None
    pending_remove: Optional[int] = None
    preview_mode: bool = False
    smoothing_level: str = 'adaptive'
    manual_peaks: List[Dict] = field(default_factory=list)
    residuals_peaks: List[Dict] = field(default_factory=list)
    peak_sources: Dict[int, str] = field(default_factory=dict)
    manual_peak_position: Optional[float] = None
    show_smoothing_preview: bool = False
    auto_smooth_suggested: bool = False
    # Новые поля для улучшенной функциональности
    peak_detection_method: str = 'hybrid'
    model_type: str = 'pseudo_voigt'
    use_aic_bic_control: bool = True
    aic_bic_threshold: float = 2.0
    show_residuals_peaks: bool = True
    adaptive_smoothing_factor: float = 1.0
    baseline_iterations: int = 10
    auto_detect_baseline: bool = True
    peak_prominence: float = 0.01
    peak_width_min: float = 0.5
    peak_width_max: float = 50.0
    use_adaptive_smoothing: bool = True
    voigt_sigma_guess: float = 1.0
    voigt_gamma_guess: float = 0.5
    aic_history: List[float] = field(default_factory=list)
    bic_history: List[float] = field(default_factory=list)
    peak_count_history: List[int] = field(default_factory=list)
    peaks_to_remove: List[int] = field(default_factory=list)
    # Новое поле для вычитания минимума
    subtract_minimum: bool = False
    minimum_subtracted_value: Optional[float] = None
    # Новое поле для хранения оригинальных Y данных до вычитания минимума
    original_y_before_subtract: Optional[np.ndarray] = None
    # Новое поле для хранения выбранного диапазона в значениях X
    x_range_selection_min: Optional[float] = None
    x_range_selection_max: Optional[float] = None

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
st.title("📊 Advanced Spectral Deconvolution")
st.markdown("---")

# ==================== NEW CLASSES ====================

class AdaptiveSmoother:
    """
    Адаптивное сглаживание: размер окна зависит от локального шума.
    В областях с большим шумом применяется более сильное сглаживание.
    """
    
    @staticmethod
    def adaptive_savgol(y, base_window=5, noise_threshold=0.1, max_window_factor=5):
        """
        Адаптивное сглаживание Савицким-Голаем.
        
        Parameters:
        -----------
        y : array_like
            Входные данные
        base_window : int
            Базовый размер окна (минимальный)
        noise_threshold : float
            Порог шума для увеличения окна
        max_window_factor : int
            Максимальный множитель для увеличения окна
        
        Returns:
        --------
        y_smooth : array_like
            Сглаженные данные
        """
        n = len(y)
        y_smooth = np.zeros_like(y, dtype=float)
        
        # Оцениваем локальный шум как абсолютную разность между соседними точками
        local_noise = np.abs(np.diff(y))
        # Добавляем первую точку для сохранения размера
        local_noise = np.concatenate([[local_noise[0]], local_noise])
        
        # Нормализуем шум относительно среднего значения
        mean_y = np.mean(np.abs(y)) if np.mean(np.abs(y)) > 0 else 1.0
        noise_level = local_noise / (mean_y + 1e-12)
        
        for i in range(n):
            # Определяем размер окна на основе локального шума
            noise_factor = min(noise_level[i] / noise_threshold, max_window_factor)
            window = base_window + int(noise_factor * 4)
            
            # Обеспечиваем нечетность окна и минимальный размер
            window = max(window, 3)
            if window % 2 == 0:
                window += 1
            window = min(window, n - 1 if n % 2 == 0 else n)
            if window < 3:
                window = 3
            
            # Применяем Savitzky-Golay к окну
            half = window // 2
            start = max(0, i - half)
            end = min(n, i + half + 1)
            
            if end - start < 3:
                y_smooth[i] = y[i]
            else:
                try:
                    # Используем полином 2-й степени для малых окон
                    polyorder = min(2, end - start - 1)
                    if polyorder >= 1:
                        # Для малых окон используем простое среднее
                        if end - start <= 5:
                            y_smooth[i] = np.mean(y[start:end])
                        else:
                            y_smooth[i] = savgol_filter(y[start:end], end - start if (end - start) % 2 == 1 else end - start - 1, polyorder)[i - start]
                    else:
                        y_smooth[i] = np.mean(y[start:end])
                except Exception:
                    y_smooth[i] = y[i]
        
        return y_smooth
    
    @staticmethod
    def estimate_noise_level(y, method='mad'):
        """
        Оценка уровня шума в данных.
        
        Parameters:
        -----------
        y : array_like
            Входные данные
        method : str
            Метод оценки шума: 'mad' (медианное абсолютное отклонение) или 'std'
        
        Returns:
        --------
        noise_level : float
            Оценка уровня шума
        """
        if method == 'mad':
            # Median Absolute Deviation
            median = np.median(y)
            mad = np.median(np.abs(y - median))
            return mad * 1.4826  # Масштабируем для нормального распределения
        else:
            # Стандартное отклонение
            return np.std(y)
    
    @staticmethod
    def suggest_smoothing_level(y, target_snr=10.0):
        """
        Предлагает уровень сглаживания на основе отношения сигнал/шум.
        
        Parameters:
        -----------
        y : array_like
            Входные данные
        target_snr : float
            Целевое отношение сигнал/шум
        
        Returns:
        --------
        suggested_level : str
            Предлагаемый уровень сглаживания ('none', 'light', 'medium', 'strong')
        """
        noise = AdaptiveSmoother.estimate_noise_level(y)
        signal = np.max(y) - np.min(y) if np.max(y) > np.min(y) else 1.0
        current_snr = signal / (noise + 1e-12)
        
        if current_snr > target_snr * 2:
            return 'none'
        elif current_snr > target_snr:
            return 'light'
        elif current_snr > target_snr / 2:
            return 'medium'
        else:
            return 'strong'


class HybridPeakFinder:
    """
    Гибридный поиск пиков с использованием трех методов:
    1. find_peaks (SciPy) с проминенсом
    2. Вторая производная (Savitzky-Golay)
    3. Continuous Wavelet Transform (CWT)
    
    Комбинирование методов позволяет находить пики, которые пропускает каждый метод по отдельности.
    """
    
    @staticmethod
    def find_peaks_hybrid(x, y, sensitivity=0.03, min_distance=5, prominence_factor=0.5):
        """
        Комбинированный поиск пиков.
        
        Parameters:
        -----------
        x : array_like
            Координаты X
        y : array_like
            Координаты Y (нормализованные)
        sensitivity : float
            Чувствительность (0.001 - 0.1)
        min_distance : int
            Минимальное расстояние между пиками в точках
        prominence_factor : float
            Множитель для проминенса (относительно sensitivity)
        
        Returns:
        --------
        peaks : list
            Список индексов найденных пиков
        peak_info : list
            Детальная информация о каждом пике
        """
        peaks = set()
        peak_details = []
        
        # Подготовка данных
        y_smooth = y.copy()
        x_mean_diff = np.mean(np.diff(x)) if len(x) > 1 else 1.0
        
        # Метод 1: find_peaks с проминенсом
        height_threshold = sensitivity * np.max(y)
        prominence_threshold = prominence_factor * sensitivity * np.max(y)
        
        try:
            p1, properties = find_peaks(
                y_smooth, 
                height=height_threshold,
                prominence=prominence_threshold,
                distance=min_distance,
                width=1
            )
            for idx in p1:
                peaks.add(idx)
                peak_details.append({
                    'index': idx,
                    'method': 'find_peaks',
                    'height': properties['peak_heights'][np.where(p1 == idx)[0][0]] if idx in p1 else y[idx],
                    'prominence': properties['prominences'][np.where(p1 == idx)[0][0]] if idx in p1 else 0,
                    'width': properties['widths'][np.where(p1 == idx)[0][0]] if idx in p1 else 0
                })
        except Exception as e:
            warnings.warn(f"find_peaks failed: {e}")
        
        # Метод 2: Вторая производная
        try:
            window = min(11, len(y_smooth) // 5 * 2 + 1)
            if window % 2 == 0:
                window += 1
            if window >= 5:
                d2y = savgol_filter(y_smooth, window, 3, deriv=2, delta=x_mean_diff)
            else:
                d2y = np.gradient(np.gradient(y_smooth, x_mean_diff), x_mean_diff)
            
            # Ищем отрицательные минимумы второй производной
            d2y_min = np.min(d2y) if np.min(d2y) < 0 else -1.0
            threshold = sensitivity * abs(d2y_min)
            
            for i in range(2, len(d2y) - 2):
                if d2y[i] < threshold and d2y[i] < 0:
                    if d2y[i] < d2y[i-1] and d2y[i] < d2y[i+1]:
                        # Проверяем, не слишком ли близко к уже найденным пикам
                        too_close = False
                        for p in peaks:
                            if abs(i - p) < min_distance:
                                too_close = True
                                break
                        if not too_close:
                            peaks.add(i)
                            peak_details.append({
                                'index': i,
                                'method': 'second_derivative',
                                'height': y[i],
                                'prominence': 0,
                                'width': 0
                            })
        except Exception as e:
            warnings.warn(f"Second derivative failed: {e}")
        
        # Метод 3: CWT (Continuous Wavelet Transform)
        try:
            if len(y_smooth) > 50:
                widths = np.arange(2, min(15, len(y_smooth) // 10))
                if len(widths) > 0:
                    p3 = find_peaks_cwt(y_smooth, widths, min_snr=sensitivity * 2)
                    for idx in p3:
                        too_close = False
                        for p in peaks:
                            if abs(idx - p) < min_distance:
                                too_close = True
                                break
                        if not too_close:
                            peaks.add(idx)
                            peak_details.append({
                                'index': idx,
                                'method': 'cwt',
                                'height': y[idx],
                                'prominence': 0,
                                'width': 0
                            })
        except Exception as e:
            warnings.warn(f"CWT failed: {e}")
        
        # Сортируем пики
        peaks_sorted = sorted(peaks)
        
        # Дополнительная фильтрация: удаляем пики, которые слишком близки друг к другу
        filtered_peaks = []
        for p in peaks_sorted:
            if not filtered_peaks or abs(x[p] - x[filtered_peaks[-1]]) > min_distance * x_mean_diff:
                filtered_peaks.append(p)
        
        # Обновляем peak_details только для отфильтрованных пиков
        filtered_details = []
        for p in filtered_peaks:
            # Находим соответствующий detail или создаем новый
            detail = next((d for d in peak_details if d['index'] == p), None)
            if detail is None:
                detail = {
                    'index': p,
                    'method': 'hybrid',
                    'height': y[p],
                    'prominence': 0,
                    'width': 0
                }
            filtered_details.append(detail)
        
        return filtered_peaks, filtered_details
    
    @staticmethod
    def estimate_peak_parameters(x, y, peak_indices):
        """
        Оценка параметров пиков: амплитуда, центр, ширина.
        
        Returns:
        --------
        params : list
            Список параметров [amp, cen, sigma] для каждого пика
        """
        params = []
        for idx in peak_indices:
            cen = x[idx]
            amp = y[idx]
            
            # Оценка ширины
            try:
                widths, _, left_ips, right_ips = peak_widths(y, [idx], rel_height=0.5)
                if len(widths) > 0:
                    fwhm = widths[0] * np.mean(np.diff(x))
                    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
                else:
                    sigma = (np.max(x) - np.min(x)) / 20
            except Exception:
                sigma = (np.max(x) - np.min(x)) / 20
            
            sigma = max(sigma, 0.01 * (np.max(x) - np.min(x)) / 20)
            params.extend([amp, cen, sigma])
        
        return params


class AICBICController:
    """
    Контроль переобучения через AIC (Akaike Information Criterion) и BIC (Bayesian Information Criterion).
    
    AIC = n * ln(RSS/n) + 2k
    BIC = n * ln(RSS/n) + k * ln(n)
    
    где:
    n - число точек
    RSS - остаточная сумма квадратов
    k - число параметров модели
    
    Пик добавляется только если AIC и BIC улучшаются на заданную величину.
    """
    
    @staticmethod
    def calculate_aic_bic(n, rss, k):
        """
        Расчет AIC и BIC.
        
        Parameters:
        -----------
        n : int
            Число точек данных
        rss : float
            Остаточная сумма квадратов (Residual Sum of Squares)
        k : int
            Число параметров модели
        
        Returns:
        --------
        aic : float
            Значение AIC
        bic : float
            Значение BIC
        """
        if rss <= 0 or n <= 0:
            return np.inf, np.inf
        
        aic = n * np.log(rss / n) + 2 * k
        bic = n * np.log(rss / n) + k * np.log(n)
        return aic, bic
    
    @staticmethod
    def should_add_peak(current_aic, new_aic, current_bic, new_bic, threshold=2.0):
        """
        Решение о добавлении пика на основе улучшения AIC/BIC.
        
        Parameters:
        -----------
        current_aic : float
            Текущее значение AIC
        new_aic : float
            Новое значение AIC после добавления пика
        current_bic : float
            Текущее значение BIC
        new_bic : float
            Новое значение BIC после добавления пика
        threshold : float
            Минимальное улучшение для добавления пика
        
        Returns:
        --------
        should_add : bool
            True если пик следует добавить
        improvement_aic : float
            Улучшение AIC
        improvement_bic : float
            Улучшение BIC
        """
        improvement_aic = current_aic - new_aic
        improvement_bic = current_bic - new_bic
        
        # Оба критерия должны улучшиться
        should_add = (improvement_aic > threshold) and (improvement_bic > threshold)
        
        return should_add, improvement_aic, improvement_bic
    
    @staticmethod
    def find_optimal_peak_count(x, y, initial_params, max_peaks=20, model_type='gaussian', 
                                threshold=2.0, method='trf', maxfev=5000):
        """
        Поиск оптимального числа пиков с использованием AIC/BIC.
        
        Parameters:
        -----------
        x : array_like
            Координаты X
        y : array_like
            Координаты Y (нормализованные)
        initial_params : list
            Начальные параметры
        max_peaks : int
            Максимальное число пиков для проверки
        model_type : str
            Тип модели ('gaussian', 'lorentzian', 'pseudo_voigt', 'voigt')
        threshold : float
            Порог улучшения AIC/BIC
        method : str
            Метод оптимизации
        maxfev : int
            Максимальное число итераций
        
        Returns:
        --------
        optimal_params : list
            Оптимальные параметры
        optimal_n_peaks : int
            Оптимальное число пиков
        aic_history : list
            История AIC
        bic_history : list
            История BIC
        """
        n = len(y)
        aic_history = []
        bic_history = []
        peak_count_history = []
        
        # Начинаем с 1 пика
        current_n_peaks = 1
        current_params = initial_params[:3]
        
        # Выполняем фиттинг для текущего числа пиков
        from scipy.optimize import curve_fit
        
        def model_func(x, *params):
            if model_type == 'gaussian':
                return GaussianModel.multi_gaussian(x, *params)
            elif model_type == 'lorentzian':
                return GaussianModel.multi_lorentzian(x, *params)
            elif model_type == 'pseudo_voigt':
                return GaussianModel.multi_pseudo_voigt(x, *params)
            else:  # voigt
                return GaussianModel.multi_voigt(x, *params)
        
        try:
            popt, _ = curve_fit(model_func, x, y, p0=current_params, method=method, maxfev=maxfev)
            y_fit = model_func(x, *popt)
            rss = np.sum((y - y_fit) ** 2)
            k = len(popt)
            aic, bic = AICBICController.calculate_aic_bic(n, rss, k)
            aic_history.append(aic)
            bic_history.append(bic)
            peak_count_history.append(current_n_peaks)
        except Exception:
            return initial_params, 1, [], []
        
        # Пробуем добавлять пики
        for n_peaks in range(2, max_peaks + 1):
            # Добавляем новый пик (берем из начальных параметров если есть)
            if n_peaks * 3 <= len(initial_params):
                # Используем существующие параметры
                new_params = initial_params[:n_peaks * 3]
            else:
                # Добавляем случайный пик
                new_params = list(current_params)
                new_cen = np.mean(x) + np.random.randn() * 0.1 * (np.max(x) - np.min(x))
                new_amp = 0.1 * np.max(y)
                new_sigma = 0.05 * (np.max(x) - np.min(x))
                new_params.extend([new_amp, new_cen, new_sigma])
            
            try:
                popt, _ = curve_fit(model_func, x, y, p0=new_params, method=method, maxfev=maxfev)
                y_fit = model_func(x, *popt)
                rss = np.sum((y - y_fit) ** 2)
                k = len(popt)
                new_aic, new_bic = AICBICController.calculate_aic_bic(n, rss, k)
                
                should_add, imp_aic, imp_bic = AICBICController.should_add_peak(
                    aic_history[-1], new_aic, bic_history[-1], new_bic, threshold
                )
                
                if should_add:
                    current_params = popt
                    current_n_peaks = n_peaks
                    aic_history.append(new_aic)
                    bic_history.append(new_bic)
                    peak_count_history.append(n_peaks)
                else:
                    # Не улучшается - останавливаемся
                    break
                    
            except Exception:
                # Ошибка фиттинга - останавливаемся
                break
        
        return current_params, current_n_peaks, aic_history, bic_history, peak_count_history


class ResidualsAnalyzer:
    """
    Анализ остатков для обнаружения пропущенных пиков.
    
    После фиттинга анализируются остатки (разница между данными и моделью).
    Если в остатках видны структуры, похожие на пики, они предлагаются для добавления.
    """
    
    @staticmethod
    def find_peaks_in_residuals(x, residuals, sensitivity=0.02, min_distance=5):
        """
        Поиск пиков в остатках.
        
        Parameters:
        -----------
        x : array_like
            Координаты X
        residuals : array_like
            Остатки (y_data - y_fit)
        sensitivity : float
            Чувствительность
        min_distance : int
            Минимальное расстояние между пиками в точках
        
        Returns:
        --------
        peaks : list
            Список индексов найденных пиков
        peak_info : list
            Информация о найденных пиках
        """
        # Сглаживаем остатки для уменьшения шума
        window = min(11, len(residuals) // 5 * 2 + 1)
        if window % 2 == 0:
            window += 1
        
        if window >= 5 and len(residuals) >= window:
            residuals_smooth = savgol_filter(residuals, window, 3)
        else:
            residuals_smooth = residuals
        
        # Ищем положительные пики (модель ниже данных)
        height_threshold = sensitivity * np.max(np.abs(residuals))
        
        positive_peaks = []
        negative_peaks = []
        
        try:
            pos_peaks, _ = find_peaks(residuals_smooth, height=height_threshold, distance=min_distance)
            positive_peaks = list(pos_peaks)
        except Exception:
            pass
        
        try:
            neg_peaks, _ = find_peaks(-residuals_smooth, height=height_threshold, distance=min_distance)
            negative_peaks = list(neg_peaks)
        except Exception:
            pass
        
        # Объединяем и фильтруем
        all_peaks = sorted(set(positive_peaks) | set(negative_peaks))
        
        # Фильтруем близкие пики
        filtered_peaks = []
        for p in all_peaks:
            if not filtered_peaks or abs(x[p] - x[filtered_peaks[-1]]) > min_distance * np.mean(np.diff(x)):
                filtered_peaks.append(p)
        
        # Собираем информацию о пиках
        peak_info = []
        for idx in filtered_peaks:
            peak_info.append({
                'index': idx,
                'x': x[idx],
                'residual_value': residuals[idx],
                'amplitude_estimate': abs(residuals[idx]),
                'sign': 'positive' if residuals[idx] > 0 else 'negative'
            })
        
        return filtered_peaks, peak_info
    
    @staticmethod
    def suggest_peaks_from_residuals(x, residuals, peak_info, sensitivity=0.02, min_distance=5):
        """
        Предлагает пики из остатков для добавления.
        
        Returns:
        --------
        suggested_peaks : list
            Список предлагаемых пиков с параметрами
        """
        peaks, info = ResidualsAnalyzer.find_peaks_in_residuals(
            x, residuals, sensitivity, min_distance
        )
        
        suggested = []
        for idx in peaks:
            # Оцениваем параметры пика
            cen = x[idx]
            amp = abs(residuals[idx])
            
            # Оцениваем ширину
            try:
                widths, _, _, _ = peak_widths(residuals, [idx], rel_height=0.5)
                if len(widths) > 0:
                    fwhm = widths[0] * np.mean(np.diff(x))
                    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
                else:
                    sigma = (np.max(x) - np.min(x)) / 20
            except Exception:
                sigma = (np.max(x) - np.min(x)) / 20
            
            sigma = max(sigma, 0.01 * (np.max(x) - np.min(x)) / 20)
            
            suggested.append({
                'index': idx,
                'cen': cen,
                'amp': amp,
                'sigma': sigma,
                'sign': 'positive' if residuals[idx] > 0 else 'negative'
            })
        
        return suggested


class VoigtFitter:
    """
    Полноценный Voigt профиль (свертка Гаусса и Лоренца).
    Использует scipy.special.voigt_profile для точного расчета.
    """
    
    @staticmethod
    def voigt(x, amp, cen, sigma, gamma):
        """
        Voigt профиль.
        
        Parameters:
        -----------
        x : array_like
            Координаты
        amp : float
            Амплитуда
        cen : float
            Центр
        sigma : float
            Ширина Гаусса
        gamma : float
            Ширина Лоренца
        
        Returns:
        --------
        y : array_like
            Значения Voigt профиля
        """
        return amp * voigt_profile(x - cen, sigma, gamma)
    
    @staticmethod
    def multi_voigt(x, *params):
        """
        Сумма Voigt пиков.
        
        Parameters:
        -----------
        x : array_like
            Координаты
        params : list
            Параметры [amp1, cen1, sigma1, gamma1, amp2, cen2, sigma2, gamma2, ...]
        
        Returns:
        --------
        y : array_like
            Сумма Voigt профилей
        """
        n = len(params) // 4
        y = np.zeros_like(x, dtype=float)
        for i in range(n):
            amp = params[4*i]
            cen = params[4*i + 1]
            sigma = abs(params[4*i + 2])
            gamma = abs(params[4*i + 3])
            y += VoigtFitter.voigt(x, amp, cen, sigma, gamma)
        return y
    
    @staticmethod
    def estimate_voigt_parameters(x, y, peak_idx):
        """
        Оценка параметров Voigt для пика.
        
        Returns:
        --------
        sigma_guess : float
            Начальное приближение sigma
        gamma_guess : float
            Начальное приближение gamma
        """
        # Оцениваем общую ширину
        try:
            widths, _, _, _ = peak_widths(y, [peak_idx], rel_height=0.5)
            if len(widths) > 0:
                fwhm = widths[0] * np.mean(np.diff(x))
                # Для Voigt: FWHM ~ sigma * 2.355 + gamma * 2
                sigma_guess = fwhm / 4.0
                gamma_guess = fwhm / 6.0
            else:
                sigma_guess = (np.max(x) - np.min(x)) / 20
                gamma_guess = (np.max(x) - np.min(x)) / 30
        except Exception:
            sigma_guess = (np.max(x) - np.min(x)) / 20
            gamma_guess = (np.max(x) - np.min(x)) / 30
        
        return max(sigma_guess, 0.01), max(gamma_guess, 0.005)


# ==================== EXISTING CLASSES WITH IMPROVEMENTS ====================

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
    def parse_file(file_content):
        """Parse file content (bytes or string) in any format"""
        try:
            # Try to decode as string
            if isinstance(file_content, bytes):
                text = file_content.decode('utf-8')
            else:
                text = str(file_content)
            return DataParser.parse_text(text)
        except Exception as e:
            raise ValueError(f"Failed to parse file: {e}")
    
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
    def apply_range_selection(x, y, start_idx, end_idx, use_log_x=False):
        """Apply range selection to data using point indices"""
        if start_idx is None or end_idx is None:
            return x, y
        
        # Ensure indices are within bounds
        start_idx = max(0, min(start_idx, len(x) - 1))
        end_idx = max(0, min(end_idx, len(x) - 1))
        
        # Select by indices (continuous block of points)
        if start_idx <= end_idx:
            mask = np.zeros(len(x), dtype=bool)
            mask[start_idx:end_idx+1] = True
        else:
            mask = np.zeros(len(x), dtype=bool)
            mask[end_idx:start_idx+1] = True
        
        return x[mask], y[mask]
    
    @staticmethod
    def apply_range_selection_by_x(x, y, x_min, x_max):
        """Apply range selection to data using X values (from min to max)"""
        if x_min is None or x_max is None:
            return x, y
        
        # Ensure x_min <= x_max
        if x_min > x_max:
            x_min, x_max = x_max, x_min
        
        # Create mask for points within range
        mask = (x >= x_min) & (x <= x_max)
        
        return x[mask], y[mask]


class DataPreprocessor:
    """Handles data preprocessing including clipping, log transformations, and baseline correction"""
    
    def __init__(self, clip_negative=True, show_warnings=True):
        self.clip_negative = clip_negative
        self.show_warnings = show_warnings
        self.clipped_points = 0
        self.small_values_warning = False
        self.baseline_removed = False
        self.baseline_values = None
        # Новые поля для отслеживания преобразований
        self.applied_transformations = []
        self.original_max_y = None
        self.preprocessed_max_y = None
    
    def smooth_data(self, x, y, method='savgol', level='none', x_log=False):
        """Smooth data with various methods and levels"""
        if level == 'none' or len(y) < 5:
            return y
        
        # Determine window size based on level and data length
        n_points = len(y)
        if level == 'light':
            window = min(5, n_points - 1 if n_points % 2 == 0 else n_points)
        elif level == 'medium':
            window = min(11, n_points - 1 if n_points % 2 == 0 else n_points)
        elif level == 'strong':
            window = min(21, n_points - 1 if n_points % 2 == 0 else n_points)
        elif level == 'adaptive':
            # Адаптивное сглаживание
            return AdaptiveSmoother.adaptive_savgol(y, base_window=5)
        else:
            return y
        
        # Ensure window is odd
        if window % 2 == 0:
            window += 1
        
        try:
            if method == 'savgol':
                polyorder = min(3, window - 1)
                return savgol_filter(y, window, polyorder)
            elif method == 'gaussian':
                sigma = window / 5
                return gaussian_filter1d(y, sigma)
        except Exception as e:
            if self.show_warnings:
                warnings.warn(f"Smoothing failed: {e}")
            return y
    
    def remove_baseline_arpls(self, y, lam=1e5, p=0.01, niter=10):
        """
        Asymmetric Least Squares (arPLS) baseline correction.
        Золотой стандарт для коррекции фона в рамановских спектрах.
        
        Parameters:
        -----------
        y : array_like
            Входные данные
        lam : float
            Параметр гладкости (чем больше, тем более гладкий фон)
        p : float
            Параметр асимметрии (0.001-0.1, типично 0.01)
        niter : int
            Число итераций
        
        Returns:
        --------
        baseline : array_like
            Оценка фона
        """
        y = np.asarray(y, dtype=float)
        n = len(y)
        
        # Если данные слишком короткие, используем простой метод
        if n < 10:
            return np.percentile(y, 5) * np.ones_like(y)
        
        # Используем pybaselines если доступен
        if HAS_PYBASELINES:
            try:
                baseline_fitter = Baseline(x_data=None)
                # Используем arPLS метод из pybaselines
                baseline, params = baseline_fitter.arpls(y, lam=lam, p=p, max_iter=niter)
                return baseline
            except Exception as e:
                if self.show_warnings:
                    warnings.warn(f"pybaselines.arpls failed: {e}, falling back to manual implementation")
        
        # Ручная реализация arPLS
        try:
            # Построение матрицы второй производной
            from scipy.sparse import spdiags, diags, csr_matrix
            from scipy.sparse.linalg import spsolve
            
            # Создаем матрицу D (вторая производная)
            D = diags([1, -2, 1], [0, 1, 2], shape=(n-2, n)).toarray()
            
            # Итеративный процесс
            z = y.copy()
            for i in range(niter):
                # Веса: больший вес для точек ниже фона
                w = np.ones_like(y)
                w[y > z] = p
                w[y <= z] = 1 - p
                
                # Решаем систему (W + lam * D.T @ D) z = W @ y
                W = np.diag(w)
                A = W + lam * (D.T @ D)
                try:
                    z = np.linalg.solve(A, W @ y)
                except np.linalg.LinAlgError:
                    # Если матрица сингулярна, используем псевдо-обратную
                    z = np.linalg.lstsq(A, W @ y, rcond=None)[0]
            
            return z
            
        except Exception as e:
            if self.show_warnings:
                warnings.warn(f"arPLS manual implementation failed: {e}")
            # Fallback: простая медианная фильтрация
            window = min(21, n // 10 * 2 + 1)
            if window % 2 == 0:
                window += 1
            from scipy.ndimage import median_filter
            return median_filter(y, size=window)
    
    def remove_baseline_polynomial(self, x, y, degree=1):
        """Полиномиальная коррекция фона"""
        if degree < 0:
            return y, np.zeros_like(y)
        
        # Подгоняем полином
        coeffs = np.polyfit(x, y, degree)
        baseline = np.polyval(coeffs, x)
        return y - baseline, baseline
    
    def remove_baseline(self, x, y, method='arpls', degree=1, lam=1e5, p=0.01, niter=10):
        """
        Коррекция фона выбранным методом.
        
        Methods:
        --------
        'none' : Без коррекции
        'polynomial' : Полиномиальная коррекция
        'arpls' : Asymmetric Least Squares
        'constant' : Постоянный фон (медиана)
        """
        if method == 'none':
            return y, np.zeros_like(y)
        elif method == 'polynomial':
            return self.remove_baseline_polynomial(x, y, degree)
        elif method == 'arpls':
            baseline = self.remove_baseline_arpls(y, lam, p, niter)
            return y - baseline, baseline
        elif method == 'constant':
            baseline = np.median(y) * np.ones_like(y)
            return y - baseline, baseline
        else:
            return y, np.zeros_like(y)
    
    def preprocess_for_fitting(self, x_linear, y_original, use_log_x, use_log_y, smoothing_level='none',
                               baseline_method='arpls', baseline_lam=1e5, baseline_p=0.01, 
                               baseline_degree=1, baseline_niter=10):
        """Preprocess data for fitting with proper handling of edge cases"""
        # Sort by X to ensure monotonic increasing X
        sort_idx = np.argsort(x_linear)
        x_sorted = x_linear[sort_idx]
        y_sorted = y_original[sort_idx]
        
        # Сохраняем информацию о преобразованиях
        self.applied_transformations = []
        self.original_max_y = np.max(y_sorted) if len(y_sorted) > 0 else 1.0
        
        # Handle negative values
        if self.clip_negative:
            negative_mask = y_sorted < 0
            self.clipped_points = np.sum(negative_mask)
            if self.clipped_points > 0 and self.show_warnings:
                warnings.warn(f"Clipped {self.clipped_points} negative values to 0")
                self.applied_transformations.append('clip_negative')
            y_for_fitting = np.maximum(y_sorted, 0)
        else:
            y_for_fitting = y_sorted
        
        # Коррекция фона
        if baseline_method != 'none':
            y_corrected, baseline = self.remove_baseline(
                x_sorted, y_for_fitting, baseline_method, 
                baseline_degree, baseline_lam, baseline_p, baseline_niter
            )
            self.baseline_removed = True
            self.baseline_values = baseline
            y_for_fitting = y_corrected
            self.applied_transformations.append(f'baseline_{baseline_method}')
        else:
            self.baseline_removed = False
            self.baseline_values = None
        
        # Apply smoothing if requested
        if smoothing_level != 'none':
            y_for_fitting = self.smooth_data(x_sorted, y_for_fitting, 'savgol', smoothing_level, use_log_x)
            self.applied_transformations.append(f'smoothing_{smoothing_level}')
        
        # Сохраняем максимум после предобработки
        self.preprocessed_max_y = np.max(y_for_fitting) if np.any(y_for_fitting > 0) else 1.0
        
        # Small epsilon for log transformations
        eps = np.finfo(float).eps
        
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
            self.applied_transformations.append('log_y')
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
            'small_values_warning': self.small_values_warning,
            'baseline_removed': self.baseline_removed,
            'baseline_values': self.baseline_values,
            'original_max_y': self.original_max_y,
            'preprocessed_max_y': self.preprocessed_max_y,
            'applied_transformations': self.applied_transformations
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
    
    @staticmethod
    def find_peaks_by_second_derivative(x, y, sensitivity=0.01, min_distance=5):
        """
        Поиск пиков по отрицательным минимумам второй производной.
        Этот метод особенно хорош для обнаружения скрытых пиков.
        """
        # Сглаживаем для уменьшения шума
        window = min(11, len(y) // 4 * 2 + 1)
        if window % 2 == 0:
            window += 1
        
        if window >= 5 and len(y) >= window:
            y_smooth = savgol_filter(y, window, 3)
        else:
            y_smooth = y
        
        # Вторая производная через Savitzky-Golay
        try:
            x_mean_diff = np.mean(np.diff(x)) if len(x) > 1 else 1.0
            d2y = savgol_filter(y_smooth, window, 3, deriv=2, delta=x_mean_diff)
        except Exception:
            d2y = np.gradient(np.gradient(y_smooth, x), x)
        
        # Ищем отрицательные минимумы (пики)
        d2y_min = np.min(d2y) if np.min(d2y) < 0 else -1.0
        threshold = sensitivity * abs(d2y_min)
        
        peaks = []
        for i in range(2, len(d2y) - 2):
            if d2y[i] < threshold and d2y[i] < 0:
                if d2y[i] < d2y[i-1] and d2y[i] < d2y[i+1]:
                    # Проверяем расстояние до предыдущих пиков
                    if not peaks or abs(x[i] - x[peaks[-1]]) > min_distance * np.mean(np.diff(x)):
                        peaks.append(i)
        
        return peaks


class GaussianModel:
    """Model for sum of Gaussians with baseline correction and additional peak shapes"""
    
    @staticmethod
    def gaussian(x, amp, cen, sigma):
        """Gaussian function with safe sigma"""
        return amp * np.exp(-(x - cen)**2 / (2 * max(sigma, np.finfo(float).eps)**2))
    
    @staticmethod
    def lorentzian(x, amp, cen, gamma):
        """Lorentzian function (natural line shape)"""
        return amp * (gamma**2) / ((x - cen)**2 + gamma**2 + np.finfo(float).eps)
    
    @staticmethod
    def pseudo_voigt(x, amp, cen, sigma, eta=0.5):
        """
        Pseudo-Voigt: линейная комбинация Гаусса и Лоренца.
        
        eta = 0 : чистый Гаусс
        eta = 1 : чистый Лоренц
        0 < eta < 1 : смесь
        """
        gauss = GaussianModel.gaussian(x, amp, cen, sigma)
        # Для Лоренца используем gamma = sigma * 1.7 (приближенное соотношение)
        gamma = sigma * 1.7
        lorentz = GaussianModel.lorentzian(x, amp, cen, gamma)
        return eta * lorentz + (1 - eta) * gauss
    
    @staticmethod
    def voigt(x, amp, cen, sigma, gamma):
        """Voigt profile using scipy.special.voigt_profile"""
        return amp * voigt_profile(x - cen, sigma, gamma)
    
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
    def multi_lorentzian(x, *params):
        """Sum of multiple Lorentzians"""
        n = len(params) // 3
        y = np.zeros_like(x, dtype=float)
        for i in range(n):
            amp = params[3*i]
            cen = params[3*i + 1]
            gamma = abs(params[3*i + 2])
            y += GaussianModel.lorentzian(x, amp, cen, gamma)
        return y
    
    @staticmethod
    def multi_pseudo_voigt(x, *params):
        """Sum of multiple Pseudo-Voigt peaks"""
        n = len(params) // 4  # amp, cen, sigma, eta
        y = np.zeros_like(x, dtype=float)
        for i in range(n):
            amp = params[4*i]
            cen = params[4*i + 1]
            sigma = abs(params[4*i + 2])
            eta = np.clip(params[4*i + 3], 0, 1)
            y += GaussianModel.pseudo_voigt(x, amp, cen, sigma, eta)
        return y
    
    @staticmethod
    def multi_voigt(x, *params):
        """Sum of multiple Voigt peaks"""
        n = len(params) // 4
        y = np.zeros_like(x, dtype=float)
        for i in range(n):
            amp = params[4*i]
            cen = params[4*i + 1]
            sigma = abs(params[4*i + 2])
            gamma = abs(params[4*i + 3])
            y += GaussianModel.voigt(x, amp, cen, sigma, gamma)
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
    def calculate_area(amp, sigma, model_type='gaussian', eta=0.5):
        """Calculate area under peak for different models"""
        if model_type == 'gaussian':
            return amp * sigma * np.sqrt(2 * np.pi)
        elif model_type == 'lorentzian':
            return amp * np.pi * sigma  # gamma = sigma * 1.7, площадь = pi * amp * gamma
        elif model_type == 'pseudo_voigt':
            # Взвешенная сумма площадей Гаусса и Лоренца
            area_gauss = amp * sigma * np.sqrt(2 * np.pi)
            area_lorentz = amp * np.pi * sigma * 1.7
            return eta * area_lorentz + (1 - eta) * area_gauss
        else:  # voigt
            # Приближенная площадь для Voigt
            return amp * sigma * np.sqrt(2 * np.pi) * 1.2  # Приближение
        return amp * sigma * np.sqrt(2 * np.pi)  # Fallback
    
    @staticmethod
    def calculate_fwhm(sigma, model_type='gaussian', eta=0.5):
        """Calculate FWHM for different models"""
        if model_type == 'gaussian':
            return 2 * np.sqrt(2 * np.log(2)) * sigma
        elif model_type == 'lorentzian':
            return 2 * sigma  # gamma = sigma * 1.7, FWHM = 2 * gamma
        elif model_type == 'pseudo_voigt':
            # Приближенный FWHM для Pseudo-Voigt
            fwhm_gauss = 2 * np.sqrt(2 * np.log(2)) * sigma
            fwhm_lorentz = 2 * sigma * 1.7
            return eta * fwhm_lorentz + (1 - eta) * fwhm_gauss
        else:  # voigt
            # Приближенный FWHM для Voigt
            fwhm_gauss = 2 * np.sqrt(2 * np.log(2)) * sigma
            fwhm_lorentz = 2 * sigma  # gamma = sigma
            return np.sqrt(fwhm_gauss**2 + fwhm_lorentz**2)  # Приближение
        return 2 * np.sqrt(2 * np.log(2)) * sigma  # Fallback
    
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
    def estimate_eta_from_peak(x, y, peak_idx):
        """Оценка параметра eta для Pseudo-Voigt"""
        # По соотношению высоты и ширины оцениваем форму
        try:
            heights = y[peak_idx]
            widths, _, _, _ = peak_widths(y, [peak_idx], rel_height=0.5)
            if len(widths) > 0:
                fwhm = widths[0] * np.mean(np.diff(x))
                # Если пик острый и с широкими крыльями -> ближе к Лоренцу
                # Если пик более пологий -> ближе к Гауссу
                ratio = fwhm / (heights + 1e-12)
                # Эмпирическое соотношение
                eta = np.clip(1.0 - ratio / 10.0, 0, 1)
                return eta
        except Exception:
            pass
        return 0.5  # По умолчанию 50/50


class FitQualityAnalyzer:
    """Fit quality analysis with AIC and BIC"""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred, n_params):
        """Calculate quality metrics including AIC and BIC"""
        residuals = y_true - y_pred
        n = len(y_true)
        
        # R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # AIC and BIC using AICBICController
        aic, bic = AICBICController.calculate_aic_bic(n, ss_res, n_params)
        
        # Chi-squared (reduced)
        chi_squared = ss_res / (n - n_params) if n > n_params else np.inf
        
        # Maximum error
        max_error = np.max(np.abs(residuals))
        
        # Root mean square error
        rmse = np.sqrt(np.mean(residuals**2))
        
        # Mean absolute error
        mae = np.mean(np.abs(residuals))
        
        return {
            'R²': r_squared,
            'AIC': aic,
            'BIC': bic,
            'χ²': chi_squared,
            'Max Error': max_error,
            'RMSE': rmse,
            'MAE': mae,
            'Residuals': residuals,
            'n_params': n_params,
            'n_points': n,
            'SS_res': ss_res,
            'SS_tot': ss_tot
        }
    
    @staticmethod
    def detect_autocorrelation(residuals):
        """Detect autocorrelation in residuals using Durbin-Watson statistic"""
        if len(residuals) < 10:
            return False
        
        diff = np.diff(residuals)
        dw = np.sum(diff**2) / (np.sum(residuals**2) + 1e-12)
        
        # DW < 1.5: положительная автокорреляция (пропущены пики)
        # DW > 2.5: отрицательная автокорреляция (переобучение)
        return dw < 1.5 or dw > 2.5


class GaussianFitter:
    """Handles Gaussian fitting with multiple optimization methods and baseline"""
    
    def __init__(self, method='trf', max_nfev=5000, baseline_method='none', 
                 fit_quality='balanced', last_popt=None, model_type='gaussian',
                 use_aic_bic_control=False, aic_bic_threshold=2.0):
        self.method = method
        self.max_nfev = max_nfev
        self.baseline_method = baseline_method
        self.fit_quality = fit_quality
        self.last_popt = last_popt
        self.model_type = model_type
        self.use_aic_bic_control = use_aic_bic_control
        self.aic_bic_threshold = aic_bic_threshold
        self.convergence_history = []
        self.fit_progress = 0
        self.aic_history = []
        self.bic_history = []
        self.peak_count_history = []
        
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
    
    def get_model_func(self, n_peaks):
        """Get the appropriate model function based on model_type"""
        if self.model_type == 'gaussian':
            return lambda x, *p: GaussianModel.multi_gaussian(x, *p)
        elif self.model_type == 'lorentzian':
            return lambda x, *p: GaussianModel.multi_lorentzian(x, *p)
        elif self.model_type == 'pseudo_voigt':
            return lambda x, *p: GaussianModel.multi_pseudo_voigt(x, *p)
        elif self.model_type == 'voigt':
            return lambda x, *p: GaussianModel.multi_voigt(x, *p)
        else:
            return lambda x, *p: GaussianModel.multi_gaussian(x, *p)
    
    def get_params_per_peak(self):
        """Get number of parameters per peak for current model"""
        if self.model_type in ['gaussian', 'lorentzian']:
            return 3  # amp, cen, sigma/gamma
        else:  # pseudo_voigt, voigt
            return 4  # amp, cen, sigma, eta/gamma
    
    def fit(self, x, y_norm, initial_peak_params, y_max, normalization_factor=1.0,
            progress_callback=None, fixed_params=None):
        """Perform fitting with progress tracking
        
        Parameters:
        -----------
        x : array_like
            X coordinates (in fitting space)
        y_norm : array_like
            Normalized Y data (0-1 scale)
        initial_peak_params : list
            Initial parameters for peaks (normalized)
        y_max : float
            Maximum value of y_norm (for scaling)
        normalization_factor : float
            Factor to convert from normalized to original scale
        progress_callback : callable
            Progress callback function
        fixed_params : list
            Fixed parameters (not used currently)
        """
        n_peaks = len(initial_peak_params) // self.get_params_per_peak()
        n_baseline = self.get_n_baseline_params()
        
        # Use last good parameters if available
        if self.last_popt is not None:
            expected_len = n_peaks * self.get_params_per_peak() + n_baseline
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
        
        # Create bounds
        lower_bounds, upper_bounds = self._create_bounds(x, y_norm, n_peaks, n_baseline)
        
        # Ensure initial_params are within bounds
        for i in range(len(initial_params)):
            initial_params[i] = np.clip(initial_params[i], lower_bounds[i], upper_bounds[i])
        
        try:
            if progress_callback:
                progress_callback(0.3, f"Initializing {self.model_type} fit...")
            
            # Get the model function
            model_func = self.get_model_func(n_peaks)
            
            # Perform fit
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
            params_per_peak = self.get_params_per_peak()
            peak_params = popt[:n_peaks * params_per_peak]
            baseline_params = popt[n_peaks * params_per_peak:] if n_baseline > 0 else []
            
            # Extract components
            components = []
            for i in range(n_peaks):
                base_idx = i * params_per_peak
                amp_norm = peak_params[base_idx]
                cen = peak_params[base_idx + 1]
                sigma = abs(peak_params[base_idx + 2])
                
                # Store amplitude in normalized form
                amp_norm_scaled = amp_norm
                
                # Get eta for pseudo_voigt or gamma for voigt
                eta = 0.5
                gamma = sigma * 0.5
                if params_per_peak == 4:
                    eta = np.clip(peak_params[base_idx + 3], 0, 1) if self.model_type == 'pseudo_voigt' else 0.5
                    gamma = abs(peak_params[base_idx + 3]) if self.model_type == 'voigt' else sigma * 0.5
                
                # Calculate area and FWHM in normalized space
                area_norm = GaussianModel.calculate_area(amp_norm, sigma, self.model_type, eta)
                fwhm = GaussianModel.calculate_fwhm(sigma, self.model_type, eta)
                
                # Generate component curve (normalized)
                if self.model_type == 'gaussian':
                    component_y_norm = GaussianModel.gaussian(x, amp_norm, cen, sigma)
                elif self.model_type == 'lorentzian':
                    component_y_norm = GaussianModel.lorentzian(x, amp_norm, cen, sigma)
                elif self.model_type == 'pseudo_voigt':
                    component_y_norm = GaussianModel.pseudo_voigt(x, amp_norm, cen, sigma, eta)
                else:  # voigt
                    component_y_norm = GaussianModel.voigt(x, amp_norm, cen, sigma, gamma)
                
                # Calculate center in linear space
                if hasattr(x, 'min') and hasattr(x, 'max'):
                    cen_linear = 10**cen if np.any(x < 0) else cen
                else:
                    cen_linear = cen
                
                # Store component with normalized amplitude
                components.append({
                    'id': i + 1,
                    'amp_norm': amp_norm,
                    'amp_norm_scaled': amp_norm_scaled,
                    'amp_original': amp_norm * normalization_factor,  # Original scale
                    'cen_log': cen,
                    'cen_linear': cen_linear,
                    'sigma_log': sigma,
                    'fwhm': fwhm,
                    'area_norm': area_norm,
                    'area_original': area_norm * normalization_factor,
                    'fraction': 0,
                    'y_norm': component_y_norm,
                    'eta': eta,
                    'gamma': gamma,
                    'model_type': self.model_type,
                    'normalization_factor': normalization_factor
                })
            
            # Calculate fractions
            total_area_norm = sum([c['area_norm'] for c in components])
            for c in components:
                c['fraction'] = c['area_norm'] / total_area_norm if total_area_norm > 0 else 0
                c['fraction_percent'] = c['fraction'] * 100
            
            # Apply AIC/BIC control if enabled
            if self.use_aic_bic_control and len(components) > 1:
                if progress_callback:
                    progress_callback(0.85, "Evaluating AIC/BIC...")
                
                # Calculate metrics for current fit
                n_points = len(x)
                rss = np.sum((y_norm - fit_y_norm) ** 2)
                k = len(popt)
                aic, bic = AICBICController.calculate_aic_bic(n_points, rss, k)
                
                self.aic_history.append(aic)
                self.bic_history.append(bic)
                self.peak_count_history.append(len(components))
                
                # Store for future reference
                self._last_aic = aic
                self._last_bic = bic
                self._last_n_peaks = len(components)
            
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
        
        params_per_peak = self.get_params_per_peak()
        
        # Peak bounds
        for i in range(n_peaks):
            # amp: 0 to 2*max
            lower_bounds.append(0)
            upper_bounds.append(2 * np.max(y_norm))
            
            # cen: within data range
            lower_bounds.append(np.min(x))
            upper_bounds.append(np.max(x))
            
            # sigma/gamma: positive, within reasonable range
            lower_bounds.append(x_range * 0.001)
            upper_bounds.append(x_range * 0.5)
            
            # Additional parameters for pseudo_voigt or voigt
            if params_per_peak == 4:
                if self.model_type == 'pseudo_voigt':
                    # eta: 0 to 1
                    lower_bounds.append(0)
                    upper_bounds.append(1)
                else:  # voigt
                    # gamma: positive
                    lower_bounds.append(x_range * 0.001)
                    upper_bounds.append(x_range * 0.5)
        
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
        n_peaks = len(peak_params) // self.get_params_per_peak()
        n_baseline = self.get_n_baseline_params()
        
        if baseline_params is None and n_baseline > 0:
            if self.baseline_method == 'constant':
                baseline_params = [0]
            elif self.baseline_method == 'linear':
                baseline_params = [0, 0]
            else:
                baseline_params = [0, 0, 0]
        
        # Calculate fit using appropriate model
        model_func = self.get_model_func(n_peaks)
        fit_y_norm = model_func(x, *peak_params)
        
        # Add baseline if provided
        if baseline_params and n_baseline > 0:
            if self.baseline_method == 'constant':
                fit_y_norm += baseline_params[0]
            elif self.baseline_method == 'linear':
                fit_y_norm += baseline_params[0] + baseline_params[1] * x
            elif self.baseline_method == 'quadratic':
                fit_y_norm += baseline_params[0] + baseline_params[1] * x + baseline_params[2] * x**2
        
        return fit_y_norm


class SpectrumPlotter:
    """Unified plotting class for all visualizations"""
    
    def __init__(self, scientific_style=True):
        self.scientific_style = scientific_style
    
    def plot_raw_data(self, x, y, use_log_x=False, use_log_y=False, 
                      title="Raw Data", ax=None, figsize=(10, 6),
                      highlight_range=None):
        """Plot raw data with optional log scales and range highlight"""
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
        
        # Highlight selected range if provided
        if highlight_range is not None:
            x_min, x_max = highlight_range
            if x_min is not None and x_max is not None:
                ax.axvspan(x_min, x_max, alpha=0.2, color='green', 
                          label='Selected range for analysis', zorder=0)
        
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
                        manual_peak_position=None, manual_peak_source=None,
                        peaks_to_remove=None):
        """Plot data with detected peaks and optional manual peak position indicator"""
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
        
        # Plot smoothed data
        # y_smooth is normalized (0-1), convert back to original scale using y_max
        if hasattr(deconvolver, 'y_max') and deconvolver.y_max > 0:
            y_smooth_original = y_smooth * deconvolver.y_max
        else:
            # Fallback
            y_smooth_original = y_smooth * np.max(deconvolver.y_sorted)
        
        ax.plot(deconvolver.x_sorted, y_smooth_original, 
                'r-', linewidth=2, label='Smoothed', color='red', zorder=2)
        
        # Color mapping for peak sources
        source_colors = {
            'auto': 'green',
            'manual': 'orange',
            'residuals': 'blue',
            'find_peaks': 'lime',
            'second_derivative': 'cyan',
            'cwt': 'magenta',
            'hybrid': 'purple'
        }
        
        # Mark detected peaks with source-based colors
        for i, info in enumerate(peak_info):
            source = info.get('source', 'auto')
            color = source_colors.get(source, 'green')
            
            # Check if this peak is marked for removal
            is_marked_for_removal = (i + 1) in peaks_to_remove if peaks_to_remove else False
            
            if is_marked_for_removal:
                # Red color for marked peaks
                marker_color = 'darkred'
                facecolor = 'red'
                marker_size = 12
                marker_style = 's'  # Square for marked peaks
            else:
                # Определяем цвета в зависимости от источника
                if source == 'auto':
                    marker_color = 'darkred'
                    facecolor = 'lime'
                elif source == 'manual':
                    marker_color = 'darkorange'
                    facecolor = 'orange'
                elif source == 'residuals':
                    marker_color = 'darkblue'
                    facecolor = 'lightblue'
                else:
                    marker_color = 'darkgreen'
                    facecolor = 'green'
                
                marker_size = 8
                marker_style = 'o'
            
            # Use method-specific colors for hybrid detection
            if 'method' in info and not is_marked_for_removal:
                method = info.get('method', 'auto')
                method_colors = {
                    'find_peaks': 'green',
                    'second_derivative': 'cyan',
                    'cwt': 'magenta',
                    'hybrid': 'purple'
                }
                facecolor = method_colors.get(method, 'lime')
                # Карта соответствия цветов для темных оттенков
                dark_color_map = {
                    'green': 'darkgreen',
                    'cyan': 'darkcyan',
                    'magenta': 'darkmagenta',
                    'purple': 'darkviolet',
                    'lime': 'darkgreen'
                }
                marker_color = dark_color_map.get(facecolor, 'darkgreen')
            
            peak_y_original = info['y_original']
            ax.plot(info['x_linear'], peak_y_original, 
                    marker_style, markersize=marker_size, markeredgecolor=marker_color, 
                    markerfacecolor=facecolor, zorder=3)
            
            # Add label with strikethrough for marked peaks
            label_text = f'{i+1}'
            if is_marked_for_removal:
                label_text = f'✕{i+1}'
            
            ax.text(info['x_linear'], peak_y_original * 1.05, 
                    label_text, ha='center', fontweight='bold', 
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", 
                                          facecolor='white' if not is_marked_for_removal else 'red', 
                                          alpha=0.8),
                    zorder=4)
        
        # Show manual peak position indicator if provided
        if manual_peak_position is not None:
            # Find Y value at this position
            idx = np.argmin(np.abs(deconvolver.x_sorted - manual_peak_position))
            y_at_position = deconvolver.y_sorted[idx]
            
            # Draw vertical line
            ax.axvline(x=manual_peak_position, color='red', linestyle='--', 
                      linewidth=1.5, alpha=0.7, label='Selected position')
            
            # Draw red dot at the position
            ax.plot(manual_peak_position, y_at_position, 'ro', 
                   markersize=10, markeredgecolor='darkred', 
                   markerfacecolor='red', zorder=5)
            
            # Add annotation
            ax.annotate(f'X: {manual_peak_position:.3e}\nY: {y_at_position:.3e}',
                       xy=(manual_peak_position, y_at_position),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=10, bbox=dict(boxstyle="round,pad=0.3", 
                                             facecolor='yellow', alpha=0.8))
        
        # Add legend for peak sources if available
        if any(info.get('source', 'auto') != 'auto' for info in peak_info):
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='lime', edgecolor='darkgreen', label='Auto-detected peaks'),
                Patch(facecolor='orange', edgecolor='darkorange', label='Manually added peaks'),
                Patch(facecolor='lightblue', edgecolor='darkblue', label='Residuals-found peaks'),
                Patch(facecolor='red', edgecolor='darkred', label='Marked for removal (✕)')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
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
        
        # Determine scaling factor to original data
        if hasattr(deconvolver, 'scale_to_original'):
            scale_factor = deconvolver.scale_to_original
        else:
            # Fallback: try to compute from data
            if deconvolver.y_original is not None and len(deconvolver.y_original) > 0:
                original_max = np.max(deconvolver.y_original)
                if original_max > 0 and deconvolver.y_norm is not None and np.max(deconvolver.y_norm) > 0:
                    scale_factor = original_max / np.max(deconvolver.y_norm)
                else:
                    scale_factor = deconvolver.y_max
            else:
                scale_factor = deconvolver.y_max
        
        # Plot components with proper scaling
        if show_components and deconvolver.components:
            colors = plt.cm.Set3(np.linspace(0, 1, len(deconvolver.components)))
            for c, color in zip(deconvolver.components, colors):
                # Get normalized component Y using x_dense_log
                if deconvolver.model_type == 'gaussian':
                    y_component_norm = GaussianModel.gaussian(x_dense_log, c['amp_norm'], 
                                                              c['cen_log'], c['sigma_log'])
                elif deconvolver.model_type == 'lorentzian':
                    y_component_norm = GaussianModel.lorentzian(x_dense_log, c['amp_norm'], 
                                                                c['cen_log'], c['sigma_log'])
                elif deconvolver.model_type == 'pseudo_voigt':
                    eta = c.get('eta', 0.5)
                    y_component_norm = GaussianModel.pseudo_voigt(x_dense_log, c['amp_norm'], 
                                                                  c['cen_log'], c['sigma_log'], eta)
                else:  # voigt
                    gamma = c.get('gamma', c['sigma_log'] * 0.5)
                    y_component_norm = GaussianModel.voigt(x_dense_log, c['amp_norm'], 
                                                           c['cen_log'], c['sigma_log'], gamma)
                
                # Scale to original data scale
                if deconvolver.use_log_y:
                    y_component = (10 ** (y_component_norm * scale_factor)) if np.any(y_component_norm > 0) else y_component_norm * scale_factor
                else:
                    y_component = y_component_norm * scale_factor
                
                # Ensure we don't have negative values for log scale
                if deconvolver.use_log_y and np.any(y_component < 0):
                    y_component = np.maximum(y_component, 1e-12)
                
                # Fill under Gaussian - используем x_dense (не x_dense_log!)
                ax.fill_between(x_dense, 0, y_component, 
                                color=color, alpha=0.3, linewidth=0)
                
                # Plot line - тоже используем x_dense
                ax.plot(x_dense, y_component, '-', color=color, linewidth=2,
                       label=f'Peak {c["id"]}: {c["fraction_percent"]:.1f}%', zorder=2)
        
        # Plot baseline if available
        if show_baseline and hasattr(deconvolver, 'baseline_params') and deconvolver.baseline_params:
            if deconvolver.baseline_method == 'constant':
                y_baseline = deconvolver.baseline_params[0] * scale_factor
                ax.axhline(y=y_baseline, color='gray', linestyle=':', 
                          linewidth=1.5, label='Baseline', zorder=1)
            elif deconvolver.baseline_method == 'linear':
                y_baseline = (deconvolver.baseline_params[0] + 
                            deconvolver.baseline_params[1] * x_dense_log) * scale_factor
                ax.plot(x_dense, y_baseline, 'gray', linestyle=':', 
                       linewidth=1.5, label='Baseline', zorder=1)
            elif deconvolver.baseline_method == 'quadratic':
                y_baseline = (deconvolver.baseline_params[0] + 
                            deconvolver.baseline_params[1] * x_dense_log +
                            deconvolver.baseline_params[2] * x_dense_log**2) * scale_factor
                ax.plot(x_dense, y_baseline, 'gray', linestyle=':', 
                       linewidth=1.5, label='Baseline', zorder=1)
        
        # Plot total fit
        if preview_mode and preview_fit is not None:
            if deconvolver.use_log_y:
                y_total = (10 ** (preview_fit * scale_factor)) if np.any(preview_fit > 0) else preview_fit * scale_factor
            else:
                y_total = preview_fit * scale_factor
            ax.plot(x_dense, y_total, 'b--', linewidth=2, 
                   label='Preview (no fit)', zorder=3, alpha=0.7)
        elif deconvolver.fit_y_norm is not None:
            from scipy.interpolate import interp1d
            
            # Create interpolation function
            fit_interp = interp1d(deconvolver.x, deconvolver.fit_y_norm, 
                                  kind='linear', fill_value='extrapolate')
            
            # Compute values on dense grid
            y_total_norm = fit_interp(x_dense_log)
            
            # Scale to original data scale
            if deconvolver.use_log_y:
                y_total = (10 ** (y_total_norm * scale_factor)) if np.any(y_total_norm > 0) else y_total_norm * scale_factor
            else:
                y_total = y_total_norm * scale_factor
            
            # Ensure we don't have negative values for log scale
            if deconvolver.use_log_y and np.any(y_total < 0):
                y_total = np.maximum(y_total, 1e-12)
            
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
            metrics_text += f"RMSE = {deconvolver.quality_metrics.get('RMSE', 0):.2e}\n"
            metrics_text += f"AIC = {deconvolver.quality_metrics.get('AIC', 0):.1f}"
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
        
        # Add scaling info
        if hasattr(deconvolver, 'scale_to_original'):
            ax.text(0.02, 0.02, f"Scale factor: {deconvolver.scale_to_original:.3f}", 
                   transform=ax.transAxes, fontsize=8, color='gray')
        
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
        
        # Scale smooth data to original scale
        if hasattr(deconvolver, 'scale_to_original'):
            y_smooth_scaled = y_smooth * deconvolver.scale_to_original
        else:
            y_smooth_scaled = y_smooth * deconvolver.y_max
        
        ax.plot(deconvolver.x, y_smooth_scaled, 
               'r-', linewidth=2, label='Smoothed')
        
        for i, info in enumerate(peak_info):
            # Use original Y value
            ax.plot(info['x'], info['y'] * deconvolver.y_max, 'ro', 
                   markersize=8, markeredgecolor='darkred')
            ax.text(info['x'], info['y'] * deconvolver.y_max * 1.05, 
                   f'{i+1}', ha='center', fontweight='bold')
        
        ax.set_xlabel(deconvolver.x_label)
        ax.set_ylabel(deconvolver.y_label)
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
    """Main class for spectral deconvolution with baseline correction and multiple models"""
    
    def __init__(self, x_linear, y_original, use_log_x=True, use_log_y=False,
                 clip_negative=True, show_warnings=True, baseline_method='arpls',
                 smoothing_level='adaptive', model_type='pseudo_voigt',
                 use_aic_bic_control=True, aic_bic_threshold=2.0,
                 peak_detection_method='hybrid', baseline_lam=1e5, baseline_p=0.01,
                 baseline_iterations=10, adaptive_smoothing_factor=1.0):
        # Store original data WITHOUT ANY MODIFICATIONS for display purposes
        self.x_original = np.array(x_linear).copy()
        self.y_original_raw = np.array(y_original).copy()
        
        # Working arrays that may be modified
        self.x_linear = np.array(x_linear)
        self.y_original = np.array(y_original)
        self.use_log_x = use_log_x
        self.use_log_y = use_log_y
        self.baseline_method = baseline_method
        self.smoothing_level = smoothing_level
        self.model_type = model_type
        self.use_aic_bic_control = use_aic_bic_control
        self.aic_bic_threshold = aic_bic_threshold
        self.peak_detection_method = peak_detection_method
        self.baseline_lam = baseline_lam
        self.baseline_p = baseline_p
        self.baseline_iterations = baseline_iterations
        self.adaptive_smoothing_factor = adaptive_smoothing_factor
        
        # Sort by X to ensure monotonic increasing X
        sort_idx = np.argsort(self.x_linear)
        self.x_linear = self.x_linear[sort_idx]
        self.y_original = self.y_original[sort_idx]
        
        # Store sorted original data for display
        self.x_sorted = self.x_linear.copy()
        self.y_sorted = self.y_original.copy()
        
        # Store original max for scaling
        self.original_max_y = np.max(self.y_original) if np.any(self.y_original > 0) else 1.0
        
        # Preprocess data
        self.preprocessor = DataPreprocessor(clip_negative, show_warnings)
        preprocessed = self.preprocessor.preprocess_for_fitting(
            self.x_linear, self.y_original, use_log_x, use_log_y, smoothing_level,
            baseline_method, baseline_lam, baseline_p, 1, baseline_iterations
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
        self.baseline_removed = preprocessed['baseline_removed']
        self.baseline_values = preprocessed['baseline_values']
        self.preprocessed_max_y = preprocessed.get('preprocessed_max_y', 1.0)
        self.applied_transformations = preprocessed.get('applied_transformations', [])
        
        # Calculate scaling factor from normalized to original scale
        # Use the maximum of y_original (original data) and preprocessed data
        if self.preprocessed_max_y > 0 and self.original_max_y > 0:
            self.scale_to_original = self.original_max_y / self.preprocessed_max_y
        else:
            self.scale_to_original = 1.0
        
        # For fitting, we normalize using the preprocessed max
        if self.preprocessed_max_y > 0:
            # Use the preprocessed max for normalization to keep y_norm in 0-1 range
            self.y_max = self.preprocessed_max_y
            self.y_norm = self.y / self.y_max
        else:
            self.y_max = 1.0
            self.y_norm = self.y
        
        # Results containers
        self.components = []
        self.fit_y_norm = None
        self.popt = None
        self.baseline_params = None
        self.quality_metrics = {}
        self.convergence_history = []
        self.total_area = 0
        self.aic_history = []
        self.bic_history = []
        self.peak_count_history = []
        
        # Fitter
        self.fitter = None
        
        # For compatibility with existing code
        self.multi_gaussian = GaussianModel.multi_gaussian
        self.gaussian = GaussianModel.gaussian
        
        # Store peak_info for sorting
        self.peak_info = []
    
    def get_scaling_factor(self, target='original'):
        """
        Возвращает коэффициент масштабирования для перехода 
        от нормализованных данных к целевому масштабу.
        target: 'original' - к исходным данным, 'preprocessed' - к предобработанным
        """
        if target == 'original':
            return self.scale_to_original
        elif target == 'preprocessed':
            return 1.0
        else:
            return self.scale_to_original
    
    def auto_detect_peaks(self, sensitivity=0.03, min_distance=5, method='hybrid'):
        """
        Automatic peak detection using selected method.
        
        Methods:
        - 'hybrid': Combination of find_peaks, second derivative, and CWT
        - 'find_peaks': SciPy find_peaks with prominence
        - 'second_derivative': Second derivative method
        - 'cwt': Continuous Wavelet Transform
        """
        # Smoothing for peak detection
        if self.smoothing_level != 'none':
            y_smooth = self.preprocessor.smooth_data(
                self.x, self.y_norm, 'savgol', self.smoothing_level, self.use_log_x
            )
        else:
            y_smooth = self.y_norm
        
        # If adaptive smoothing is enabled, use it
        if self.smoothing_level == 'adaptive':
            y_smooth = AdaptiveSmoother.adaptive_savgol(self.y_norm, base_window=5)
        
        if method == 'hybrid':
            peaks, peak_details = HybridPeakFinder.find_peaks_hybrid(
                self.x, y_smooth, sensitivity, min_distance
            )
        elif method == 'find_peaks':
            height_threshold = sensitivity * np.max(y_smooth)
            peaks, properties = find_peaks(
                y_smooth, height=height_threshold, distance=min_distance
            )
            peak_details = []
            for idx in peaks:
                peak_details.append({
                    'index': idx,
                    'method': 'find_peaks',
                    'height': y_smooth[idx],
                    'prominence': properties['prominences'][np.where(peaks == idx)[0][0]] if idx in peaks else 0,
                    'width': properties['widths'][np.where(peaks == idx)[0][0]] if idx in peaks else 0
                })
        elif method == 'second_derivative':
            peaks = DerivativeAnalyzer.find_peaks_by_second_derivative(
                self.x, y_smooth, sensitivity, min_distance
            )
            peak_details = []
            for idx in peaks:
                peak_details.append({
                    'index': idx,
                    'method': 'second_derivative',
                    'height': y_smooth[idx],
                    'prominence': 0,
                    'width': 0
                })
        elif method == 'cwt':
            try:
                widths = np.arange(2, min(15, len(y_smooth) // 10))
                peaks = find_peaks_cwt(y_smooth, widths, min_snr=sensitivity * 2)
                peak_details = []
                for idx in peaks:
                    peak_details.append({
                        'index': idx,
                        'method': 'cwt',
                        'height': y_smooth[idx],
                        'prominence': 0,
                        'width': 0
                    })
            except Exception as e:
                warnings.warn(f"CWT failed: {e}")
                peaks = []
                peak_details = []
        else:
            # Default to hybrid
            peaks, peak_details = HybridPeakFinder.find_peaks_hybrid(
                self.x, y_smooth, sensitivity, min_distance
            )
        
        # Estimate parameters
        peak_info = []
        initial_params = []
        
        for peak_idx in peaks:
            cen = self.x[peak_idx]
            amp = y_smooth[peak_idx]
            
            # Estimate sigma with fallback
            sigma = GaussianModel.estimate_sigma_from_peak(self.x, y_smooth, peak_idx)
            sigma = max(sigma, 0.01 * (np.max(self.x) - np.min(self.x)) / max(len(peaks), 1))
            
            # Get original Y value for display
            if self.use_log_x:
                x_linear = 10**self.x[peak_idx]
            else:
                x_linear = self.x[peak_idx]
            
            # Find closest index in original data
            idx = np.argmin(np.abs(self.x_sorted - x_linear))
            y_original_value = self.y_sorted[idx]
            
            # Get method from peak_details
            method_used = 'auto'
            for detail in peak_details:
                if detail['index'] == peak_idx:
                    method_used = detail.get('method', 'auto')
                    break
            
            peak_info.append({
                'index': peak_idx,
                'x': self.x[peak_idx],
                'x_linear': x_linear,
                'y': self.y[peak_idx],
                'y_original': y_original_value,
                'amp_est': amp,
                'cen_est': cen,
                'sigma_est': sigma,
                'dy': 0,
                'd2y': 0,
                'source': 'auto',
                'method': method_used
            })
            
            # Add parameters based on model type
            if self.model_type in ['gaussian', 'lorentzian']:
                initial_params.extend([amp, cen, sigma])
            else:  # pseudo_voigt or voigt
                eta = GaussianModel.estimate_eta_from_peak(self.x, y_smooth, peak_idx)
                if self.model_type == 'pseudo_voigt':
                    initial_params.extend([amp, cen, sigma, eta])
                else:  # voigt
                    gamma = sigma * 0.5
                    initial_params.extend([amp, cen, sigma, gamma])
        
        # Store peak_info for later use
        self.peak_info = peak_info
        
        # Calculate derivatives for visualization
        dy, d2y, y_smooth_deriv = DerivativeAnalyzer.calculate_derivatives(self.x, y_smooth)
        
        return peaks, peak_info, initial_params, (dy, d2y, y_smooth_deriv)
    
    def add_manual_peak(self, x_position_linear, amplitude=None, sigma_est=None, eta_est=None):
        """Add a peak manually at specified linear X position"""
        # Convert to log space if needed
        if self.use_log_x:
            x_position = np.log10(x_position_linear)
        else:
            x_position = x_position_linear
        
        # Find index for amplitude estimation
        idx = np.argmin(np.abs(self.x_sorted - x_position_linear))
        
        # Estimate amplitude if not provided
        if amplitude is None:
            if self.use_log_x:
                log_idx = np.argmin(np.abs(self.x - x_position))
                amplitude = self.y_norm[log_idx] if log_idx < len(self.y_norm) else 0.1
            else:
                amplitude = self.y_norm[idx] if idx < len(self.y_norm) else 0.1
        
        # Estimate sigma if not provided
        if sigma_est is None:
            if self.use_log_x:
                x_search = self.x
                y_search = self.y_norm
            else:
                x_search = self.x_linear
                y_search = self.y_original / self.y_max
            
            left_idx = idx
            right_idx = idx
            for i in range(idx - 1, 0, -1):
                if i < len(y_search) - 1 and y_search[i] < y_search[i-1] and y_search[i] < y_search[i+1]:
                    left_idx = i
                    break
            for i in range(idx + 1, len(y_search) - 1):
                if y_search[i] < y_search[i-1] and y_search[i] < y_search[i+1]:
                    right_idx = i
                    break
            
            width = (x_search[right_idx] - x_search[left_idx]) if right_idx > left_idx else 0.1
            sigma_est = max(width / 3.0, 0.01 * (np.max(x_search) - np.min(x_search)) / 20)
        
        # Estimate eta if not provided
        if eta_est is None:
            eta_est = 0.5
        
        # Add peak info
        peak_info_entry = {
            'index': idx,
            'x': x_position,
            'x_linear': x_position_linear,
            'y': amplitude,
            'y_original': self.y_sorted[idx],
            'amp_est': amplitude,
            'cen_est': x_position,
            'sigma_est': sigma_est,
            'dy': 0,
            'd2y': 0,
            'source': 'manual',
            'method': 'manual'
        }
        
        # Add parameters based on model type
        if self.model_type in ['gaussian', 'lorentzian']:
            initial_params = [amplitude, x_position, sigma_est]
        else:
            initial_params = [amplitude, x_position, sigma_est, eta_est]
        
        return peak_info_entry, initial_params
    
    def find_missing_peaks_by_residuals(self, peak_info, sensitivity=0.02, min_distance=5):
        """Find missing peaks by analyzing residuals after initial fit"""
        if not peak_info:
            return [], []
        
        # Build initial model with current peaks
        n_peaks = len(peak_info)
        if n_peaks == 0:
            return [], []
        
        # Get parameters based on model type
        params_per_peak = 4 if self.model_type in ['pseudo_voigt', 'voigt'] else 3
        peak_params = []
        for info in peak_info:
            if params_per_peak == 3:
                peak_params.extend([info['amp_est'], info['cen_est'], info['sigma_est']])
            else:
                eta = info.get('eta_est', 0.5)
                peak_params.extend([info['amp_est'], info['cen_est'], info['sigma_est'], eta])
        
        # Calculate initial fit
        if self.model_type == 'gaussian':
            y_initial_fit = GaussianModel.multi_gaussian(self.x, *peak_params)
        elif self.model_type == 'lorentzian':
            y_initial_fit = GaussianModel.multi_lorentzian(self.x, *peak_params)
        elif self.model_type == 'pseudo_voigt':
            y_initial_fit = GaussianModel.multi_pseudo_voigt(self.x, *peak_params)
        else:  # voigt
            y_initial_fit = GaussianModel.multi_voigt(self.x, *peak_params)
        
        # Calculate residuals
        residuals = self.y_norm - y_initial_fit
        
        # Use ResidualsAnalyzer to find missing peaks
        suggested = ResidualsAnalyzer.suggest_peaks_from_residuals(
            self.x, residuals, peak_info, sensitivity, min_distance
        )
        
        missing_peaks = []
        missing_params = []
        
        for suggestion in suggested:
            cen = suggestion['cen']
            amp = suggestion['amp']
            sigma = suggestion['sigma']
            
            # Get original Y value for display
            if self.use_log_x:
                x_linear = 10**cen
            else:
                x_linear = cen
            
            orig_idx = np.argmin(np.abs(self.x_sorted - x_linear))
            y_original_value = self.y_sorted[orig_idx]
            
            # Estimate eta
            eta = GaussianModel.estimate_eta_from_peak(self.x, residuals, suggestion['index'])
            
            missing_peaks.append({
                'index': suggestion['index'],
                'x': cen,
                'x_linear': x_linear,
                'y': amp,
                'y_original': y_original_value,
                'amp_est': amp,
                'cen_est': cen,
                'sigma_est': sigma,
                'dy': 0,
                'd2y': 0,
                'source': 'residuals',
                'method': 'residuals',
                'eta_est': eta,
                'sign': suggestion['sign']
            })
            
            # Add parameters based on model type
            if self.model_type in ['gaussian', 'lorentzian']:
                missing_params.extend([amp, cen, sigma])
            else:
                missing_params.extend([amp, cen, sigma, eta])
        
        return missing_peaks, missing_params
    
    def sort_peaks_by_x(self):
        """
        Sort peaks by X position (in linear space) and renumber them.
        This ensures consistent ordering regardless of detection order.
        Returns sorted peak_info list.
        """
        if not hasattr(self, 'peak_info') or not self.peak_info:
            return self.peak_info
        
        # Sort by x_linear
        sorted_peaks = sorted(self.peak_info, key=lambda p: p['x_linear'])
        
        # Renumber peaks sequentially
        for i, peak in enumerate(sorted_peaks):
            peak['sorted_index'] = i + 1
            # Preserve original ID if needed
            if 'original_id' not in peak:
                peak['original_id'] = peak.get('id', i + 1)
            peak['id'] = i + 1
        
        return sorted_peaks
    
    def fit(self, initial_params=None, method='trf', maxfev=5000, 
            fit_quality='balanced', last_popt=None, progress_callback=None):
        """Perform fitting with selected method and baseline"""
        if initial_params is None:
            _, _, initial_params, _ = self.auto_detect_peaks(method=self.peak_detection_method)
        
        if len(initial_params) == 0:
            return False
        
        # Create fitter with selected parameters
        self.fitter = GaussianFitter(
            method=method, 
            max_nfev=maxfev,
            baseline_method=self.baseline_method,
            fit_quality=fit_quality,
            last_popt=last_popt,
            model_type=self.model_type,
            use_aic_bic_control=self.use_aic_bic_control,
            aic_bic_threshold=self.aic_bic_threshold
        )
        
        # Perform fit with normalization factor
        success, popt, components, baseline_params = self.fitter.fit(
            self.x, self.y_norm, initial_params, self.y_max,
            normalization_factor=self.scale_to_original,
            progress_callback=progress_callback
        )
        
        if success:
            self.popt = popt
            self.components = components
            self.baseline_params = baseline_params
            self.aic_history = self.fitter.aic_history
            self.bic_history = self.fitter.bic_history
            self.peak_count_history = self.fitter.peak_count_history
            
            # Reconstruct full fit with baseline in normalized space
            n_peaks = len(components)
            params_per_peak = self.fitter.get_params_per_peak()
            peak_params = []
            for c in components:
                if params_per_peak == 3:
                    peak_params.extend([c['amp_norm'], c['cen_log'], c['sigma_log']])
                else:
                    peak_params.extend([c['amp_norm'], c['cen_log'], c['sigma_log'], c.get('eta', 0.5)])
            
            # Use appropriate model for total fit
            model_type = self.model_type
            if model_type == 'gaussian':
                self.fit_y_norm = GaussianModel.multi_gaussian(self.x, *peak_params)
            elif model_type == 'lorentzian':
                self.fit_y_norm = GaussianModel.multi_lorentzian(self.x, *peak_params)
            elif model_type == 'pseudo_voigt':
                self.fit_y_norm = GaussianModel.multi_pseudo_voigt(self.x, *peak_params)
            else:  # voigt
                self.fit_y_norm = GaussianModel.multi_voigt(self.x, *peak_params)
            
            # Add baseline if present
            if baseline_params and self.baseline_method != 'none':
                if self.baseline_method == 'constant':
                    self.fit_y_norm += baseline_params[0]
                elif self.baseline_method == 'linear':
                    self.fit_y_norm += baseline_params[0] + baseline_params[1] * self.x
                elif self.baseline_method == 'quadratic':
                    self.fit_y_norm += baseline_params[0] + baseline_params[1] * self.x + baseline_params[2] * self.x**2
            
            # Calculate total area (in original scale)
            self.total_area = sum([c['area_original'] for c in self.components])
            
            # Quality metrics
            self.quality_metrics = FitQualityAnalyzer.calculate_metrics(
                self.y_norm, self.fit_y_norm, len(popt)
            )
            
            # Validate fit quality
            self._validate_fit_quality()
            
            return True
        
        return False
    
    def _validate_fit_quality(self):
        """Validate the quality of the fit and warn if there are issues"""
        if not self.components:
            return
        
        # Check if amplitudes are reasonable
        max_original_y = np.max(self.y_original)
        max_component_amp = max([c['amp_original'] for c in self.components])
        
        # If the max component amplitude is less than 10% of max original Y, warn
        if max_component_amp < 0.1 * max_original_y and max_original_y > 0:
            warnings.warn(f"Max component amplitude ({max_component_amp:.3e}) is less than 10% of max original Y ({max_original_y:.3e}). "
                          f"Check scaling or fit quality.")
        
        # Check if fit R² is reasonable
        if self.quality_metrics and self.quality_metrics.get('R²', 0) < 0.5:
            warnings.warn(f"Fit R² = {self.quality_metrics.get('R²', 0):.3f} is low. Consider adding more peaks or adjusting parameters.")
    
    def preview_fit(self, initial_params=None):
        """Preview fit without optimization (fast)"""
        if initial_params is None:
            _, _, initial_params, _ = self.auto_detect_peaks(method=self.peak_detection_method)
        
        if len(initial_params) == 0:
            return None
        
        # Create temporary fitter
        fitter = GaussianFitter(
            baseline_method=self.baseline_method,
            model_type=self.model_type
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
            params_per_peak = 4 if self.model_type in ['pseudo_voigt', 'voigt'] else 3
            for c in self.components:
                if params_per_peak == 3:
                    current_params.extend([c['amp_norm'], c['cen_log'], c['sigma_log']])
                else:
                    current_params.extend([c['amp_norm'], c['cen_log'], c['sigma_log'], c.get('eta', 0.5)])
        else:
            return False
        
        # Apply pending remove
        if st.session_state.app_state.pending_remove is not None:
            remove_id = st.session_state.app_state.pending_remove
            new_params = []
            for i, c in enumerate(self.components):
                if i != remove_id - 1:
                    if params_per_peak == 3:
                        new_params.extend([c['amp_norm'], c['cen_log'], c['sigma_log']])
                    else:
                        new_params.extend([c['amp_norm'], c['cen_log'], c['sigma_log'], c.get('eta', 0.5)])
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
                    
                    if params_per_peak == 3:
                        new_params.extend([amp1, cen1, sigma1])
                        new_params.extend([amp2, cen2, sigma2])
                    else:
                        eta1 = c.get('eta', 0.5)
                        eta2 = c.get('eta', 0.5)
                        new_params.extend([amp1, cen1, sigma1, eta1])
                        new_params.extend([amp2, cen2, sigma2, eta2])
                else:
                    if params_per_peak == 3:
                        new_params.extend([c['amp_norm'], c['cen_log'], c['sigma_log']])
                    else:
                        new_params.extend([c['amp_norm'], c['cen_log'], c['sigma_log'], c.get('eta', 0.5)])
            
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
                    ['R²', 'AIC', 'BIC', 'χ²', 'RMSE', 'MAE'],
                    [f"{metrics.get('R²', 0):.6f}", 
                     f"{metrics.get('AIC', 0):.2f}",
                     f"{metrics.get('BIC', 0):.2f}",
                     f"{metrics.get('χ²', 0):.2e}",
                     f"{metrics.get('RMSE', 0):.2e}",
                     f"{metrics.get('MAE', 0):.2e}"]
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
        
        st.session_state.app_state.smoothing_level = st.selectbox(
            "Data smoothing",
            options=['none', 'light', 'medium', 'strong', 'adaptive'],
            index=4 if st.session_state.app_state.smoothing_level == 'adaptive' else 
                  0 if st.session_state.app_state.smoothing_level == 'none' else
                  1 if st.session_state.app_state.smoothing_level == 'light' else
                  2 if st.session_state.app_state.smoothing_level == 'medium' else 3,
            help="Smooth noisy data before peak detection. Adaptive automatically adjusts based on noise level."
        )
        
        st.session_state.app_state.peak_detection_method = st.selectbox(
            "Peak detection method",
            options=['hybrid', 'find_peaks', 'second_derivative', 'cwt'],
            index=0 if st.session_state.app_state.peak_detection_method == 'hybrid' else
                  1 if st.session_state.app_state.peak_detection_method == 'find_peaks' else
                  2 if st.session_state.app_state.peak_detection_method == 'second_derivative' else 3,
            help="hybrid: combination of all methods (recommended)"
        )
        
        st.session_state.app_state.model_type = st.selectbox(
            "Peak model",
            options=['gaussian', 'lorentzian', 'pseudo_voigt', 'voigt'],
            index=2 if st.session_state.app_state.model_type == 'pseudo_voigt' else
                  0 if st.session_state.app_state.model_type == 'gaussian' else
                  1 if st.session_state.app_state.model_type == 'lorentzian' else 3,
            help="pseudo_voigt: recommended for Raman spectra"
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
            options=['none', 'arpls', 'polynomial', 'constant'],
            index=1 if st.session_state.app_state.baseline_method == 'arpls' else
                  0 if st.session_state.app_state.baseline_method == 'none' else
                  2 if st.session_state.app_state.baseline_method == 'polynomial' else 3,
            help="arpls: Asymmetric Least Squares (recommended for Raman)"
        )
        
        if st.session_state.app_state.baseline_method == 'arpls':
            st.session_state.app_state.baseline_lam = st.number_input(
                "Lambda (smoothness)",
                min_value=1e3,
                max_value=1e8,
                value=float(st.session_state.app_state.baseline_lam),
                step=1e4,
                format="%.0f",
                help="Higher values give smoother baseline"
            )
            
            st.session_state.app_state.baseline_p = st.slider(
                "p (asymmetry)",
                min_value=0.001,
                max_value=0.1,
                value=float(st.session_state.app_state.baseline_p),
                step=0.001,
                format="%.3f",
                help="Higher values give more aggressive baseline removal"
            )
            
            st.session_state.app_state.baseline_iterations = st.number_input(
                "Iterations",
                min_value=5,
                max_value=50,
                value=st.session_state.app_state.baseline_iterations,
                step=5
            )
        
        st.session_state.app_state.use_aic_bic_control = st.checkbox(
            "AIC/BIC control (auto peak count)",
            value=st.session_state.app_state.use_aic_bic_control,
            help="Automatically determines optimal number of peaks"
        )
        
        if st.session_state.app_state.use_aic_bic_control:
            st.session_state.app_state.aic_bic_threshold = st.slider(
                "AIC/BIC improvement threshold",
                min_value=0.5,
                max_value=5.0,
                value=float(st.session_state.app_state.aic_bic_threshold),
                step=0.5,
                help="Minimum improvement required to add a peak"
            )
        
        st.session_state.app_state.preview_mode = st.checkbox(
            "Preview mode (no fitting)",
            value=st.session_state.app_state.preview_mode,
            help="Show estimated peaks without performing optimization"
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
        # File uploader
        uploaded_file = st.file_uploader(
            "📂 Upload TXT/DAT/CSV file",
            type=['txt', 'dat', 'csv'],
            help="Upload a file with two columns: X and Y data"
        )
        
        # Text area for data input
        data_text = st.text_area(
            "Or paste your data (x y separated by space, comma, or tab):",
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
            
            **File upload also supported:**
            - .txt, .dat, .csv
            - Two columns: X Y
            """
        )
        
        # Process uploaded file if present
        if uploaded_file is not None:
            try:
                # Read file content
                file_content = uploaded_file.read()
                x, y = DataParser.parse_file(file_content)
                
                if len(x) > 0:
                    st.session_state.app_state.raw_x = x
                    st.session_state.app_state.raw_y = y
                    st.session_state.app_state.x_range_min = float(np.min(x))
                    st.session_state.app_state.x_range_max = float(np.max(x))
                    st.session_state.app_state.current_step = 2
                    st.success(f"✅ File loaded successfully: {len(x)} data points")
                    st.rerun()
                else:
                    st.error("Could not parse file. Check the format.")
            except Exception as e:
                st.error(f"Error reading file: {e}")
        
        # Manual load button
        if st.button("📂 Load Data (from text)", type="primary", use_container_width=True):
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
        st.subheader("Subtract Minimum")
        
        # Чекбокс для вычитания минимума
        subtract_minimum = st.checkbox(
            "Subtract minimum intensity (shift spectrum to zero)",
            value=st.session_state.app_state.subtract_minimum,
            help="Subtract the minimum Y value from all data points to shift the spectrum baseline to zero"
        )
        
        # Обработка изменения чекбокса
        if subtract_minimum != st.session_state.app_state.subtract_minimum:
            st.session_state.app_state.subtract_minimum = subtract_minimum
            
            if subtract_minimum:
                # Сохраняем оригинальные Y если еще не сохранены
                if st.session_state.app_state.original_y_before_subtract is None:
                    st.session_state.app_state.original_y_before_subtract = st.session_state.app_state.raw_y.copy()
                
                # Вычитаем минимум
                min_y = np.min(st.session_state.app_state.raw_y)
                st.session_state.app_state.minimum_subtracted_value = min_y
                st.session_state.app_state.raw_y = st.session_state.app_state.raw_y - min_y
                st.success(f"✅ Subtracted minimum value: {min_y:.6e}")
            else:
                # Восстанавливаем оригинальные Y
                if st.session_state.app_state.original_y_before_subtract is not None:
                    st.session_state.app_state.raw_y = st.session_state.app_state.original_y_before_subtract.copy()
                    st.session_state.app_state.minimum_subtracted_value = None
                    st.session_state.app_state.original_y_before_subtract = None
                    st.info("Restored original Y values")
            
            st.rerun()
        
        # Показываем информацию о вычитании минимума
        if st.session_state.app_state.subtract_minimum:
            if st.session_state.app_state.minimum_subtracted_value is not None:
                st.info(f"Minimum subtracted: {st.session_state.app_state.minimum_subtracted_value:.6e}")
            else:
                st.info("Minimum subtraction active")
        
        st.markdown("---")
        st.subheader("Range Selection")
        
        # Get data points count
        n_points = len(st.session_state.app_state.raw_x)
        
        # Get min and max X values for slider
        x_min = np.min(st.session_state.app_state.raw_x)
        x_max = np.max(st.session_state.app_state.raw_x)
        
        # Initialize range selection if not set
        if st.session_state.app_state.x_range_selection_min is None:
            st.session_state.app_state.x_range_selection_min = float(x_min)
        if st.session_state.app_state.x_range_selection_max is None:
            st.session_state.app_state.x_range_selection_max = float(x_max)
        
        # Create slider by X values (from min to max)
        if st.session_state.app_state.use_log_x:
            # For log scale, use log values for slider
            log_min = np.log10(max(x_min, 1e-12))
            log_max = np.log10(max(x_max, 1e-12))
            
            log_range_min = st.session_state.app_state.x_range_selection_min
            log_range_max = st.session_state.app_state.x_range_selection_max
            
            if log_range_min is not None and log_range_min > 0:
                log_start = np.log10(log_range_min)
            else:
                log_start = log_min
            
            if log_range_max is not None and log_range_max > 0:
                log_end = np.log10(log_range_max)
            else:
                log_end = log_max
            
            # Slider in log space
            log_selection = st.slider(
                "Select X range (log scale):",
                min_value=float(log_min),
                max_value=float(log_max),
                value=(float(log_start), float(log_end)),
                step=float((log_max - log_min) / min(200, n_points)),
                format="%.3f",
                help="Select range by X values. All data is shown, range highlights analysis region."
            )
            
            # Convert back to linear
            range_min_linear = 10 ** log_selection[0]
            range_max_linear = 10 ** log_selection[1]
            
            st.session_state.app_state.x_range_selection_min = range_min_linear
            st.session_state.app_state.x_range_selection_max = range_max_linear
            
            st.info(f"Selected X range: {range_min_linear:.3e} - {range_max_linear:.3e}")
            st.info(f"log₁₀(X) range: {log_selection[0]:.3f} - {log_selection[1]:.3f}")
        else:
            # Linear scale slider
            range_selection = st.slider(
                "Select X range:",
                min_value=float(x_min),
                max_value=float(x_max),
                value=(float(st.session_state.app_state.x_range_selection_min), 
                       float(st.session_state.app_state.x_range_selection_max)),
                step=float((x_max - x_min) / min(200, n_points)),
                format="%.3e",
                help="Select range by X values. All data is shown, range highlights analysis region."
            )
            
            st.session_state.app_state.x_range_selection_min = range_selection[0]
            st.session_state.app_state.x_range_selection_max = range_selection[1]
            
            st.info(f"Selected X range: {range_selection[0]:.3e} - {range_selection[1]:.3e}")
        
        # Show selected range statistics
        range_min = st.session_state.app_state.x_range_selection_min
        range_max = st.session_state.app_state.x_range_selection_max
        
        # Count points in range
        mask = (st.session_state.app_state.raw_x >= range_min) & (st.session_state.app_state.raw_x <= range_max)
        points_in_range = np.sum(mask)
        
        st.info(f"Points in selected range: {points_in_range} / {n_points}")
        
        st.markdown("---")
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("⬅️ Back", use_container_width=True):
                st.session_state.app_state.current_step = 1
                st.rerun()
        with col_b:
            if st.button("✅ Apply & Continue", type="primary", use_container_width=True):
                # Apply range selection to data using X values
                x_range, y_range = DataParser.apply_range_selection_by_x(
                    st.session_state.app_state.raw_x,
                    st.session_state.app_state.raw_y,
                    st.session_state.app_state.x_range_selection_min,
                    st.session_state.app_state.x_range_selection_max
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
        
        # Get current data
        x_data = st.session_state.app_state.raw_x
        y_data = st.session_state.app_state.raw_y
        
        # Highlight selected range
        highlight_range = (
            st.session_state.app_state.x_range_selection_min,
            st.session_state.app_state.x_range_selection_max
        )
        
        plotter.plot_raw_data(
            x_data, y_data,
            use_log_x=st.session_state.app_state.use_log_x,
            use_log_y=st.session_state.app_state.use_log_y,
            title="Data with Selected Range Highlighted",
            ax=ax,
            highlight_range=highlight_range
        )
        
        # Add info about subtraction if active
        if st.session_state.app_state.subtract_minimum:
            if st.session_state.app_state.minimum_subtracted_value is not None:
                ax.text(0.02, 0.02, f"Minimum subtracted: {st.session_state.app_state.minimum_subtracted_value:.3e}",
                       transform=ax.transAxes, fontsize=9,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
        
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
            show_warnings=st.session_state.app_state.show_warnings,
            baseline_method=st.session_state.app_state.baseline_method,
            smoothing_level=st.session_state.app_state.smoothing_level,
            model_type=st.session_state.app_state.model_type,
            use_aic_bic_control=st.session_state.app_state.use_aic_bic_control,
            aic_bic_threshold=st.session_state.app_state.aic_bic_threshold,
            peak_detection_method=st.session_state.app_state.peak_detection_method,
            baseline_lam=st.session_state.app_state.baseline_lam,
            baseline_p=st.session_state.app_state.baseline_p,
            baseline_iterations=st.session_state.app_state.baseline_iterations
        )
        
        # Show warnings if any
        if st.session_state.app_state.deconvolver.clipped_points > 0:
            st.warning(f"Clipped {st.session_state.app_state.deconvolver.clipped_points} negative values to 0")
        if st.session_state.app_state.deconvolver.small_values_warning:
            st.warning("Very small Y values detected. Log transformation may cause artifacts.")
        if st.session_state.app_state.deconvolver.baseline_removed:
            st.success(f"Baseline removed using {st.session_state.app_state.baseline_method} method")
    
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
        
        # Расширенный диапазон с 1-20 до 1-50
        st.session_state.app_state.min_distance = st.slider(
            "Minimum distance between peaks:",
            min_value=1,
            max_value=50,
            value=st.session_state.app_state.min_distance,
            step=1
        )
        
        # Show current settings
        st.info(f"Method: {st.session_state.app_state.peak_detection_method}\n"
                f"Model: {st.session_state.app_state.model_type}\n"
                f"Smoothing: {st.session_state.app_state.smoothing_level}\n"
                f"Baseline: {st.session_state.app_state.baseline_method}")
        
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
                        min_distance=st.session_state.app_state.min_distance,
                        method=st.session_state.app_state.peak_detection_method
                    )
                st.session_state.app_state.peak_info = peak_info
                st.session_state.app_state.derivatives = derivatives
                st.session_state.app_state.initial_params = initial_params
                # Clear manual and residuals peaks
                st.session_state.app_state.manual_peaks = []
                st.session_state.app_state.residuals_peaks = []
                # Clear peaks to remove
                st.session_state.app_state.peaks_to_remove = []
                st.success(f"Found {len(peak_info)} peaks!")
        
        if st.session_state.app_state.peak_info is not None:
            st.markdown("---")
            st.subheader("Manual Peak Addition")
            
            # Get current X range for slider
            deconv = st.session_state.app_state.deconvolver
            if deconv.use_log_x:
                x_min_display = 10 ** np.min(deconv.x)
                x_max_display = 10 ** np.max(deconv.x)
                slider_min = np.log10(x_min_display)
                slider_max = np.log10(x_max_display)
            else:
                x_min_display = np.min(deconv.x_linear)
                x_max_display = np.max(deconv.x_linear)
                slider_min = x_min_display
                slider_max = x_max_display
            
            # Calculate number of steps
            n_points = len(deconv.x_linear)
            n_steps = min(200, n_points)
            
            # Create slider in appropriate scale
            if deconv.use_log_x:
                current_position_log = st.slider(
                    "Select peak position (log scale):",
                    min_value=float(slider_min),
                    max_value=float(slider_max),
                    value=float((slider_min + slider_max) / 2),
                    step=float((slider_max - slider_min) / n_steps),
                    format="%.2f",
                    key="manual_peak_slider_log"
                )
                manual_position_linear = 10 ** current_position_log
                st.write(f"Position: {manual_position_linear:.3e}")
            else:
                manual_position_linear = st.slider(
                    "Select peak position:",
                    min_value=float(slider_min),
                    max_value=float(slider_max),
                    value=float((slider_min + slider_max) / 2),
                    step=float((slider_max - slider_min) / n_steps),
                    format="%.3e",
                    key="manual_peak_slider_linear"
                )
                st.write(f"Position: {manual_position_linear:.3e}")
            
            # Store current manual position for visualization
            st.session_state.app_state.manual_peak_position = manual_position_linear
            
            col_add1, col_add2 = st.columns(2)
            with col_add1:
                if st.button("➕ Add peak at this position", use_container_width=True):
                    # Add manual peak
                    new_peak, new_params = deconv.add_manual_peak(manual_position_linear)
                    # Add to peak_info
                    if st.session_state.app_state.peak_info is None:
                        st.session_state.app_state.peak_info = []
                    st.session_state.app_state.peak_info.append(new_peak)
                    st.session_state.app_state.initial_params.extend(new_params)
                    st.session_state.app_state.manual_peaks.append(new_peak)
                    st.success(f"Manual peak added at {manual_position_linear:.3e}")
                    st.rerun()
            
            with col_add2:
                if st.button("🔍 Find missing peaks (residuals)", use_container_width=True):
                    with st.spinner("Analyzing residuals..."):
                        missing_peaks, missing_params = deconv.find_missing_peaks_by_residuals(
                            st.session_state.app_state.peak_info,
                            sensitivity=st.session_state.app_state.sensitivity * 0.5,
                            min_distance=st.session_state.app_state.min_distance
                        )
                    if missing_peaks:
                        st.session_state.app_state.residuals_peaks = missing_peaks
                        # Display found peaks with checkboxes for selection
                        st.subheader("Suggested peaks from residuals:")
                        selected_to_add = []
                        for i, p in enumerate(missing_peaks):
                            col_cb, col_info = st.columns([1, 3])
                            with col_cb:
                                if st.checkbox(f"Add peak {i+1}", key=f"residual_peak_{i}"):
                                    selected_to_add.append(p)
                            with col_info:
                                sign = p.get('sign', 'positive')
                                st.write(f"X: {p['x_linear']:.3e}, Amp: {p['amp_est']:.3e}, Sign: {sign}")
                        
                        if st.button("Add selected peaks", use_container_width=True):
                            for p in selected_to_add:
                                st.session_state.app_state.peak_info.append(p)
                                st.session_state.app_state.initial_params.extend([p['amp_est'], p['cen_est'], p['sigma_est']])
                                st.session_state.app_state.residuals_peaks.append(p)
                            st.success(f"Added {len(selected_to_add)} peaks from residuals")
                            st.rerun()
                    else:
                        st.info("No additional peaks found in residuals")
            
            st.markdown("---")
            
            # AIC/BIC control button
            if st.session_state.app_state.use_aic_bic_control:
                if st.button("🧠 Optimize peak count with AIC/BIC", use_container_width=True):
                    with st.spinner("Optimizing peak count..."):
                        # Use AIC/BIC controller to find optimal number of peaks
                        initial_params = st.session_state.app_state.initial_params
                        n_peaks = len(initial_params) // (4 if st.session_state.app_state.model_type in ['pseudo_voigt', 'voigt'] else 3)
                        
                        # Create temporary fitter for evaluation
                        fitter = GaussianFitter(
                            model_type=st.session_state.app_state.model_type,
                            baseline_method=st.session_state.app_state.baseline_method
                        )
                        
                        params_per_peak = 4 if st.session_state.app_state.model_type in ['pseudo_voigt', 'voigt'] else 3
                        if len(initial_params) < params_per_peak:
                            st.warning("Not enough parameters for AIC/BIC optimization. Please run peak detection first.")
                        else:
                            # Evaluate different numbers of peaks
                            best_params, best_n, aic_hist, bic_hist, count_hist = AICBICController.find_optimal_peak_count(
                                st.session_state.app_state.deconvolver.x,
                                st.session_state.app_state.deconvolver.y_norm,
                                initial_params,
                                max_peaks=min(n_peaks + 5, 30),
                                model_type=st.session_state.app_state.model_type,
                                threshold=st.session_state.app_state.aic_bic_threshold,
                                method=st.session_state.app_state.fitting_method,
                                maxfev=st.session_state.app_state.max_nfev
                            )
                        
                        if aic_hist and bic_hist:
                            # Update parameters
                            st.session_state.app_state.initial_params = list(best_params)
                            st.session_state.app_state.aic_history = aic_hist
                            st.session_state.app_state.bic_history = bic_hist
                            st.session_state.app_state.peak_count_history = count_hist
                            
                            st.success(f"Optimal peak count: {best_n} (was {n_peaks})")
                            
                            # Show AIC/BIC history with integer x-axis
                            fig, ax = plt.subplots(figsize=(8, 4))
                            ax.plot(range(1, len(aic_hist) + 1), aic_hist, 'b-o', label='AIC')
                            ax.plot(range(1, len(bic_hist) + 1), bic_hist, 'r-s', label='BIC')
                            ax.set_xlabel('Number of peaks')
                            ax.set_ylabel('Information Criterion')
                            ax.set_title('AIC/BIC vs Number of Peaks')
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            # Ensure integer ticks on x-axis
                            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                            # Set x-axis limits to include all data points
                            ax.set_xlim(0.5, len(aic_hist) + 0.5)
                            # Set integer ticks
                            ax.set_xticks(range(1, len(aic_hist) + 1))
                            st.pyplot(fig)
                            plt.close()
                            
                            params_per_peak = 4 if st.session_state.app_state.model_type in ['pseudo_voigt', 'voigt'] else 3
                            new_peak_info = []
                            for i in range(best_n):
                                base = i * params_per_peak
                                
                                # Проверяем, что достаточно параметров
                                if base + 2 >= len(st.session_state.app_state.initial_params):
                                    # Если параметров недостаточно, выходим из цикла
                                    break
                                
                                amp = st.session_state.app_state.initial_params[base]
                                cen = st.session_state.app_state.initial_params[base + 1]
                                sigma = st.session_state.app_state.initial_params[base + 2]
                                
                                # Для моделей с 4 параметрами
                                if params_per_peak == 4:
                                    if base + 3 < len(st.session_state.app_state.initial_params):
                                        eta = st.session_state.app_state.initial_params[base + 3]
                                    else:
                                        eta = 0.5  # Значение по умолчанию
                                else:
                                    eta = 0.5
                                
                                # Find original Y
                                if deconv.use_log_x:
                                    x_linear = 10**cen
                                else:
                                    x_linear = cen
                                idx = np.argmin(np.abs(deconv.x_sorted - x_linear))
                                y_original = deconv.y_sorted[idx] if idx < len(deconv.y_sorted) else 0
                                
                                new_peak_info.append({
                                    'index': idx,
                                    'x': cen,
                                    'x_linear': x_linear,
                                    'y': amp,
                                    'y_original': y_original,
                                    'amp_est': amp,
                                    'cen_est': cen,
                                    'sigma_est': sigma,
                                    'dy': 0,
                                    'd2y': 0,
                                    'source': 'auto',
                                    'method': 'aic_bic_optimized',
                                    'eta_est': eta
                                })
                            
                            st.session_state.app_state.peak_info = new_peak_info
                            st.session_state.app_state.peaks_to_remove = []
                            st.rerun()
            
            st.markdown("---")
            
            if st.button("✅ Confirm Peaks", use_container_width=True):
                with st.spinner("Preparing preview..."):
                    # Sort peaks by X position before proceeding
                    if st.session_state.app_state.peak_info:
                        # Sort by x_linear
                        sorted_peaks = sorted(st.session_state.app_state.peak_info, key=lambda p: p['x_linear'])
                        
                        # Renumber peaks sequentially
                        for i, peak in enumerate(sorted_peaks):
                            peak['id'] = i + 1
                            peak['sorted_index'] = i + 1
                        
                        st.session_state.app_state.peak_info = sorted_peaks
                        
                        # Rebuild initial_params based on sorted order
                        params_per_peak = 4 if st.session_state.app_state.model_type in ['pseudo_voigt', 'voigt'] else 3
                        new_initial_params = []
                        for peak in sorted_peaks:
                            if params_per_peak == 3:
                                new_initial_params.extend([peak['amp_est'], peak['cen_est'], peak['sigma_est']])
                            else:
                                eta = peak.get('eta_est', 0.5)
                                new_initial_params.extend([peak['amp_est'], peak['cen_est'], peak['sigma_est'], eta])
                        
                        st.session_state.app_state.initial_params = new_initial_params
                        
                        st.success(f"Peaks sorted by X position and renumbered (1-{len(sorted_peaks)})")
                    
                    if st.session_state.app_state.preview_mode:
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
                            st.session_state.app_state.last_popt = st.session_state.app_state.deconvolver.popt
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
            
            # Get peaks to remove from session state
            peaks_to_remove = getattr(st.session_state.app_state, 'peaks_to_remove', [])
            
            # Create tabs for different plots
            tab1, tab2, tab3 = st.tabs(["📊 Peaks", "📈 Derivatives", "📋 Information"])

            with tab1:
                # Get current manual position for visualization
                manual_pos = getattr(st.session_state.app_state, 'manual_peak_position', None)
                fig, ax = plt.subplots(figsize=(10, 6))
                plotter.plot_with_peaks(
                    deconv, 
                    st.session_state.app_state.peak_info, 
                    y_smooth,
                    title=f"Peak Detection - {len(st.session_state.app_state.peak_info)} peaks found",
                    ax=ax,
                    manual_peak_position=manual_pos,
                    peaks_to_remove=peaks_to_remove
                )
                st.pyplot(fig)
                plt.close()
                
                # Add legend explanation
                st.caption("""
                **Peak color legend:**
                - 🟢 Green: Auto-detected peaks (method varies)
                - 🟠 Orange: Manually added peaks
                - 🔵 Blue: Peaks found by residuals analysis
                - 🟣 Purple: Hybrid method
                - 🟦 Cyan: Second derivative method
                - 🟪 Magenta: CWT method
                - 🔴 Red (✕): Marked for removal
                """)
                
                # ===== НОВАЯ ТАБЛИЦА ДЛЯ УДАЛЕНИЯ ПИКОВ =====
                st.markdown("---")
                st.subheader("🗑️ Peak Removal")
                
                if st.session_state.app_state.peak_info:
                    st.write(f"**Select peaks to remove:** (currently {len(peaks_to_remove)} selected)")
                    
                    # Создаем таблицу с чекбоксами
                    col_names = st.columns([0.5, 1, 2, 2, 1.5, 1.5])
                    with col_names[0]:
                        st.write("**#**")
                    with col_names[1]:
                        st.write("**Select**")
                    with col_names[2]:
                        st.write("**X Center**")
                    with col_names[3]:
                        st.write("**Y Amplitude**")
                    with col_names[4]:
                        st.write("**Source**")
                    with col_names[5]:
                        st.write("**Method**")
                    
                    st.markdown("---")
                    
                    # Отображаем каждый пик с чекбоксом
                    for i, info in enumerate(st.session_state.app_state.peak_info):
                        col1, col2, col3, col4, col5, col6 = st.columns([0.5, 1, 2, 2, 1.5, 1.5])
                        
                        with col1:
                            st.write(f"**{i+1}**")
                        
                        with col2:
                            # Чекбокс для выбора пика на удаление
                            is_selected = (i + 1) in peaks_to_remove
                            checkbox_key = f"remove_peak_{i}"
                            checked = st.checkbox("", value=is_selected, key=checkbox_key)
                            
                            # Обновляем список выбранных пиков в сессии
                            if checked and (i + 1) not in peaks_to_remove:
                                peaks_to_remove.append(i + 1)
                                st.session_state.app_state.peaks_to_remove = peaks_to_remove
                            elif not checked and (i + 1) in peaks_to_remove:
                                peaks_to_remove.remove(i + 1)
                                st.session_state.app_state.peaks_to_remove = peaks_to_remove
                        
                        with col3:
                            st.write(f"{info['x_linear']:.4e}")
                        
                        with col4:
                            st.write(f"{info['y_original']:.4e}")
                        
                        with col5:
                            source = info.get('source', 'auto')
                            source_icon = "🟢" if source == 'auto' else "🟠" if source == 'manual' else "🔵"
                            st.write(f"{source_icon} {source}")
                        
                        with col6:
                            method = info.get('method', 'auto')
                            method_display = {
                                'find_peaks': 'find_peaks',
                                'second_derivative': '2nd deriv',
                                'cwt': 'CWT',
                                'hybrid': 'Hybrid',
                                'residuals': 'Residuals',
                                'manual': 'Manual',
                                'aic_bic_optimized': 'AIC/BIC',
                                'auto': 'auto'
                            }.get(method, method)
                            st.write(method_display)
                    
                    st.markdown("---")
                    
                    # Кнопка подтверждения удаления
                    col_confirm1, col_confirm2, col_confirm3 = st.columns([1, 1, 1])
                    
                    with col_confirm1:
                        if len(peaks_to_remove) > 0:
                            st.warning(f"⚠️ {len(peaks_to_remove)} peak(s) selected for removal")
                        else:
                            st.info("ℹ️ No peaks selected for removal")
                    
                    with col_confirm2:
                        if st.button("🗑️ Confirm and remove selected peaks", type="primary", use_container_width=True):
                            if peaks_to_remove:
                                # Сортируем в обратном порядке для корректного удаления
                                sorted_peaks = sorted(peaks_to_remove, reverse=True)
                                
                                # Удаляем пики из peak_info
                                for peak_id in sorted_peaks:
                                    if peak_id <= len(st.session_state.app_state.peak_info):
                                        del st.session_state.app_state.peak_info[peak_id - 1]
                                
                                # Обновляем параметры
                                # Перестраиваем initial_params на основе оставшихся пиков
                                new_initial_params = []
                                for info in st.session_state.app_state.peak_info:
                                    if st.session_state.app_state.model_type in ['gaussian', 'lorentzian']:
                                        new_initial_params.extend([info['amp_est'], info['cen_est'], info['sigma_est']])
                                    else:
                                        eta = info.get('eta_est', 0.5)
                                        new_initial_params.extend([info['amp_est'], info['cen_est'], info['sigma_est'], eta])
                                
                                st.session_state.app_state.initial_params = new_initial_params
                                st.session_state.app_state.peaks_to_remove = []
                                
                                st.success(f"✅ Removed {len(sorted_peaks)} peaks. Updated to {len(st.session_state.app_state.peak_info)} peaks.")
                                st.rerun()
                            else:
                                st.warning("No peaks selected for removal.")
                    
                    with col_confirm3:
                        if st.button("🔄 Clear all selections", use_container_width=True):
                            st.session_state.app_state.peaks_to_remove = []
                            st.rerun()
                
                else:
                    st.info("No peaks detected yet. Click 'Find Peaks' to start.")
            
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
                # Создаем таблицу с информацией о пиках
                data = []
                for i, info in enumerate(st.session_state.app_state.peak_info):
                    source = info.get('source', 'auto')
                    method = info.get('method', 'auto')
                    source_icon = "🟢" if source == 'auto' else "🟠" if source == 'manual' else "🔵"
                    
                    # Проверяем, отмечен ли пик для удаления
                    is_marked = (i + 1) in peaks_to_remove
                    marker = "🔴 ✕" if is_marked else "✅"
                    
                    method_display = {
                        'find_peaks': 'find_peaks',
                        'second_derivative': '2nd derivative',
                        'cwt': 'CWT',
                        'hybrid': 'Hybrid',
                        'residuals': 'Residuals',
                        'manual': 'Manual',
                        'aic_bic_optimized': 'AIC/BIC',
                        'auto': 'auto'
                    }.get(method, method)
                    
                    data.append({
                        'Peak': f"{marker} {i + 1}",
                        'Source': f"{source_icon} {source}",
                        'Method': method_display,
                        'X Center': f"{info['x_linear']:.4e}",
                        'Y Amplitude': f"{info['y_original']:.4e}",
                        'Estimated Sigma': f"{info['sigma_est']:.4f}",
                        'Marked': "🔴 Yes" if is_marked else "No"
                    })
                
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)
                
                # Show peak detection statistics
                st.markdown("---")
                st.subheader("Detection Statistics")
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Total Peaks Found", len(st.session_state.app_state.peak_info))
                with col2:
                    auto_count = sum(1 for p in st.session_state.app_state.peak_info if p.get('source', 'auto') == 'auto')
                    st.metric("Auto-detected", auto_count)
                with col3:
                    manual_count = sum(1 for p in st.session_state.app_state.peak_info if p.get('source', '') == 'manual')
                    st.metric("Manually added", manual_count)
                with col4:
                    residual_count = sum(1 for p in st.session_state.app_state.peak_info if p.get('source', '') == 'residuals')
                    st.metric("From residuals", residual_count)
                with col5:
                    st.metric("Marked for removal", len(peaks_to_remove))
                
                # Show method breakdown
                st.markdown("---")
                st.subheader("Method Breakdown")
                method_counts = {}
                for p in st.session_state.app_state.peak_info:
                    method = p.get('method', 'auto')
                    method_counts[method] = method_counts.get(method, 0) + 1
                
                method_df = pd.DataFrame([
                    {'Method': m, 'Count': c} for m, c in method_counts.items()
                ])
                st.dataframe(method_df, use_container_width=True)
                
                st.markdown("---")
                st.subheader("Data Info")
                col5, col6 = st.columns(2)
                with col5:
                    st.metric("X Range", f"{np.min(deconv.x_sorted):.2e} - {np.max(deconv.x_sorted):.2e}")
                with col6:
                    st.metric("Y Range", f"{np.min(deconv.y_sorted):.2e} - {np.max(deconv.y_sorted):.2e}")
                
                # AIC/BIC history if available
                if st.session_state.app_state.aic_history:
                    st.markdown("---")
                    st.subheader("AIC/BIC History")
                    history_df = pd.DataFrame({
                        'Peaks': st.session_state.app_state.peak_count_history,
                        'AIC': st.session_state.app_state.aic_history,
                        'BIC': st.session_state.app_state.bic_history
                    })
                    st.dataframe(history_df, use_container_width=True)


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
                if 'AIC' in metrics:
                    st.info(f"AIC = {metrics.get('AIC', 0):.1f} | BIC = {metrics.get('BIC', 0):.1f}")
            
            st.markdown("---")
            
            # Preview mode indicator
            if st.session_state.app_state.preview_mode:
                st.info("🔍 PREVIEW MODE - No fitting performed")
            
            # Peak selection (only if components exist)
            if deconv.components:
                # Display peak sources in selection dropdown
                peak_options = {}
                for c in deconv.components:
                    source_info = ""
                    if hasattr(st.session_state.app_state, 'peak_info'):
                        for p in st.session_state.app_state.peak_info:
                            if abs(p['x_linear'] - c['cen_linear']) / max(p['x_linear'], c['cen_linear']) < 0.01:
                                source = p.get('source', 'auto')
                                method = p.get('method', 'auto')
                                source_icon = "🟢" if source == 'auto' else "🟠" if source == 'manual' else "🔵"
                                source_info = f" [{source_icon} {method}]"
                                break
                    peak_options[f"Peak {c['id']}{source_info}: center = {c['cen_linear']:.2e}, fraction = {c['fraction_percent']:.1f}%"] = c['id']
                
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
                peaks = [f'Peak {c["id"]}' for c in deconv.components]
                areas = [c['area_original'] for c in deconv.components]
                fractions = [c['fraction_percent'] for c in deconv.components]
                colors = plt.cm.Set3(np.linspace(0, 1, len(peaks)))
                
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
                total_area = sum([c['area_original'] for c in deconv.components])
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
            
            # AIC/BIC history if available
            if deconv.aic_history:
                st.markdown("---")
                st.subheader("AIC/BIC Optimization History")
                fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
                ax_hist.plot(deconv.peak_count_history, deconv.aic_history, 'b-o', label='AIC')
                ax_hist.plot(deconv.peak_count_history, deconv.bic_history, 'r-s', label='BIC')
                ax_hist.set_xlabel('Number of Peaks')
                ax_hist.set_ylabel('Information Criterion')
                ax_hist.set_title('AIC/BIC vs Number of Peaks')
                ax_hist.legend()
                ax_hist.grid(True, alpha=0.3)
                # Ensure integer ticks on x-axis
                ax_hist.xaxis.set_major_locator(MaxNLocator(integer=True))
                if len(deconv.peak_count_history) > 0:
                    ax_hist.set_xticks(deconv.peak_count_history)
                    ax_hist.set_xlim(min(deconv.peak_count_history) - 0.5, max(deconv.peak_count_history) + 0.5)
                st.pyplot(fig_hist)
                plt.close()
        
        with tab2:
            st.subheader("Normalized View (Max Peak Intensity = 1)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Find maximum amplitude for normalization
                max_amp = max([c['amp_original'] for c in deconv.components])
                
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
                
                # Plot normalized components
                colors = plt.cm.Set3(np.linspace(0, 1, len(deconv.components)))
                for c, color in zip(deconv.components, colors):
                    # Generate component using appropriate model
                    if c.get('model_type', 'gaussian') == 'gaussian':
                        y_component_norm = GaussianModel.gaussian(x_dense_log, c['amp_norm'], 
                                                                 c['cen_log'], c['sigma_log']) 
                    elif c.get('model_type', 'gaussian') == 'lorentzian':
                        y_component_norm = GaussianModel.lorentzian(x_dense_log, c['amp_norm'], 
                                                                   c['cen_log'], c['sigma_log'])
                    elif c.get('model_type', 'gaussian') == 'pseudo_voigt':
                        eta = c.get('eta', 0.5)
                        y_component_norm = GaussianModel.pseudo_voigt(x_dense_log, c['amp_norm'], 
                                                                     c['cen_log'], c['sigma_log'], eta)
                    else:  # voigt
                        gamma = c.get('gamma', c['sigma_log'] * 0.5)
                        y_component_norm = GaussianModel.voigt(x_dense_log, c['amp_norm'], 
                                                              c['cen_log'], c['sigma_log'], gamma)
                    
                    # Scale to max amplitude for normalization
                    y_component_norm = y_component_norm * c['amp_original'] / max_amp
                    
                    # Fill under Gaussian
                    ax_norm.fill_between(x_dense, 0, y_component_norm, 
                                        color=color, alpha=0.3, linewidth=0)
                    
                    # Plot line
                    ax_norm.plot(x_dense, y_component_norm, '-', color=color, linewidth=2,
                               label=f'Peak {c["id"]}: {c["fraction_percent"]:.1f}%', zorder=2)
                
                # Plot normalized total fit
                if deconv.fit_y_norm is not None:
                    from scipy.interpolate import interp1d
                    
                    # Создаем интерполяционную функцию
                    fit_interp = interp1d(deconv.x, deconv.fit_y_norm, 
                                          kind='linear', fill_value='extrapolate')
                    
                    # Вычисляем значения на плотной сетке и нормализуем
                    y_total_norm = fit_interp(x_dense_log) * deconv.scale_to_original / max_amp
                    
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
                    metrics_text += f"RMSE = {deconv.quality_metrics.get('RMSE', 0):.2e}\n"
                    metrics_text += f"AIC = {deconv.quality_metrics.get('AIC', 0):.1f}"
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
                max_amp = max([c['amp_original'] for c in deconv.components])
                colors = plt.cm.Set3(np.linspace(0, 1, len(deconv.components)))
                for c, color in zip(deconv.components, colors):
                    # Generate component using appropriate model
                    if c.get('model_type', 'gaussian') == 'gaussian':
                        y_component_norm = GaussianModel.gaussian(x_dense_log, c['amp_norm'], 
                                                                 c['cen_log'], c['sigma_log']) 
                    elif c.get('model_type', 'gaussian') == 'lorentzian':
                        y_component_norm = GaussianModel.lorentzian(x_dense_log, c['amp_norm'], 
                                                                   c['cen_log'], c['sigma_log'])
                    elif c.get('model_type', 'gaussian') == 'pseudo_voigt':
                        eta = c.get('eta', 0.5)
                        y_component_norm = GaussianModel.pseudo_voigt(x_dense_log, c['amp_norm'], 
                                                                     c['cen_log'], c['sigma_log'], eta)
                    else:  # voigt
                        gamma = c.get('gamma', c['sigma_log'] * 0.5)
                        y_component_norm = GaussianModel.voigt(x_dense_log, c['amp_norm'], 
                                                              c['cen_log'], c['sigma_log'], gamma)
                    
                    y_component_norm = y_component_norm * c['amp_original'] / max_amp
                    
                    ax_comp_norm.plot(x_dense, y_component_norm, '-', color=color, linewidth=2,
                                    label=f'Peak {c["id"]} (center: {c["cen_linear"]:.2e})')
                
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
            max_amp = max([c['amp_original'] for c in deconv.components])
            norm_data = []
            for c in deconv.components:
                norm_data.append({
                    'Peak': c['id'],
                    'Center': f"{c['cen_linear']:.4e}",
                    'Normalized Amplitude': f"{c['amp_original'] / max_amp:.4f}",
                    'Original Amplitude': f"{c['amp_original']:.4e}",
                    'Fraction (%)': f"{c['fraction_percent']:.2f}",
                    'Model': c.get('model_type', 'gaussian')
                })
            
            df_norm = pd.DataFrame(norm_data)
            st.dataframe(df_norm, use_container_width=True)
        
        with tab3:
            st.subheader("Results Table - Complete Dataset")
            
            # Main results table
            data = []
            for c in deconv.components:
                row = {
                    'Peak ID': c['id'],
                    'Center': c['cen_linear'],
                    'Center (log)': c['cen_log'],
                    'Amplitude (orig)': c['amp_original'],
                    'Amplitude (norm)': c['amp_norm'],
                    'Sigma (log)': c['sigma_log'],
                    'FWHM': c['fwhm'],
                    'Area (orig)': c['area_original'],
                    'Area (norm)': c['area_norm'],
                    'Fraction': c['fraction'],
                    'Fraction (%)': c['fraction_percent'],
                    'Model': c.get('model_type', 'gaussian')
                }
                if 'eta' in c:
                    row['Eta (Gauss/Lorentz)'] = c['eta']
                if 'gamma' in c:
                    row['Gamma (Voigt)'] = c['gamma']
                data.append(row)
            
            df = pd.DataFrame(data)
            
            # Format for display
            display_df = df.copy()
            for col in ['Center', 'Amplitude (orig)', 'Area (orig)']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.4e}")
            for col in ['Center (log)', 'Sigma (log)', 'FWHM', 'Fraction (%)']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
            if 'Amplitude (norm)' in display_df.columns:
                display_df['Amplitude (norm)'] = display_df['Amplitude (norm)'].apply(lambda x: f"{x:.4f}")
            if 'Area (norm)' in display_df.columns:
                display_df['Area (norm)'] = display_df['Area (norm)'].apply(lambda x: f"{x:.4f}")
            
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
                        'Center': c['cen_linear'],
                        'Center_log': c['cen_log'],
                        'Amplitude_original': c['amp_original'],
                        'Amplitude_norm': c['amp_norm'],
                        'Sigma_log': c['sigma_log'],
                        'FWHM': c['fwhm'],
                        'Area_original': c['area_original'],
                        'Area_norm': c['area_norm'],
                        'Fraction': c['fraction'],
                        'Fraction_Percent': c['fraction_percent'],
                        'Model': c.get('model_type', 'gaussian'),
                        'Eta': c.get('eta', None),
                        'Gamma': c.get('gamma', None)
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
                            if c.get('model_type', 'gaussian') in ['gaussian', 'lorentzian']:
                                peak_params.extend([c['amp_norm'], c['cen_log'], c['sigma_log']])
                            else:
                                peak_params.extend([c['amp_norm'], c['cen_log'], c['sigma_log'], c.get('eta', 0.5)])
                        
                        if deconv.model_type == 'gaussian':
                            fit_y_norm = GaussianModel.multi_gaussian(deconv.x, *peak_params)
                        elif deconv.model_type == 'lorentzian':
                            fit_y_norm = GaussianModel.multi_lorentzian(deconv.x, *peak_params)
                        elif deconv.model_type == 'pseudo_voigt':
                            fit_y_norm = GaussianModel.multi_pseudo_voigt(deconv.x, *peak_params)
                        else:
                            fit_y_norm = GaussianModel.multi_voigt(deconv.x, *peak_params)
                        
                        if deconv.baseline_method == 'constant':
                            fit_y_norm += deconv.baseline_params[0]
                        elif deconv.baseline_method == 'linear':
                            fit_y_norm += deconv.baseline_params[0] + deconv.baseline_params[1] * deconv.x
                        elif deconv.baseline_method == 'quadratic':
                            fit_y_norm += deconv.baseline_params[0] + deconv.baseline_params[1] * deconv.x + deconv.baseline_params[2] * deconv.x**2
                    else:
                        fit_y_norm = deconv.fit_y_norm
                    
                    # Generate normalized fit data
                    max_amp = max([c['amp_original'] for c in deconv.components])
                    
                    df_fit = pd.DataFrame({
                        'X_original': deconv.x_linear,
                        'Y_original': deconv.y_original,
                        'Y_fit': fit_y_norm * deconv.scale_to_original,
                        'Y_fit_normalized': fit_y_norm * deconv.scale_to_original / max_amp,
                        'Residuals': deconv.quality_metrics['Residuals'] * deconv.scale_to_original,
                        'Residuals_normalized': deconv.quality_metrics['Residuals'] * deconv.scale_to_original / max_amp
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
                    max_amp = max([c['amp_original'] for c in deconv.components])
                    
                    report = f"""GAUSSIAN DECONVOLUTION REPORT
{"="*80}

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Number of points: {len(deconv.x_linear)}
X range: [{deconv.x_linear[0]:.2e}, {deconv.x_linear[-1]:.2e}]
Logarithmic X scale: {deconv.use_log_x}
Baseline method: {deconv.baseline_method}
Smoothing level: {deconv.smoothing_level}
Model type: {deconv.model_type}
Peak detection method: {deconv.peak_detection_method}
AIC/BIC control: {deconv.use_aic_bic_control}
Minimum subtracted: {st.session_state.app_state.minimum_subtracted_value if st.session_state.app_state.subtract_minimum else 'No'}
Scale factor (normalized -> original): {deconv.scale_to_original:.3f}

QUALITY METRICS:
{"-"*40}
R²: {deconv.quality_metrics.get('R²', 0):.6f}
AIC: {deconv.quality_metrics.get('AIC', 0):.2f}
BIC: {deconv.quality_metrics.get('BIC', 0):.2f}
χ²: {deconv.quality_metrics.get('χ²', 0):.2e}
RMSE: {deconv.quality_metrics.get('RMSE', 0):.2e}
MAE: {deconv.quality_metrics.get('MAE', 0):.2e}

"""
                    if deconv.baseline_method != 'none' and deconv.baseline_params:
                        report += f"""BASELINE PARAMETERS:
{"-"*40}
Method: {deconv.baseline_method}
Parameters: {', '.join([f'{p:.4e}' for p in deconv.baseline_params])}

"""
                    
                    report += f"""COMPONENTS (ORIGINAL SCALE):
{"-"*80}
ID    Center          Amplitude       FWHM        Area           Fraction(%)  Model
{"-"*80}"""
                    
                    for c in deconv.components:
                        model = c.get('model_type', 'gaussian')
                        report += f"\n{c['id']:<4} {c['cen_linear']:<15.4e} {c['amp_original']:<15.4e} {c['fwhm']:<12.4f} {c['area_original']:<15.4e} {c['fraction_percent']:<10.2f} {model}"
                        if 'eta' in c:
                            report += f" (eta={c['eta']:.3f})"
                        if 'gamma' in c:
                            report += f" (gamma={c['gamma']:.3f})"
                    
                    report += f"""

COMPONENTS (NORMALIZED TO MAX PEAK = 1):
{"-"*80}
ID    Center          Norm. Amplitude    Original Amplitude    Fraction(%)
{"-"*80}"""
                    
                    for c in deconv.components:
                        norm_amp = c['amp_original'] / max_amp
                        report += f"\n{c['id']:<4} {c['cen_linear']:<15.4e} {norm_amp:<18.4f} {c['amp_original']:<20.4e} {c['fraction_percent']:<10.2f}"
                    
                    report += f"""
{"="*80}
Total area (original scale): {deconv.total_area:.6e}
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
                    
                    max_amp = max([c['amp_original'] for c in deconv.components])
                    
                    if deconv.use_log_x:
                        ax_norm.set_xscale('log')
                    
                    x_dense = np.linspace(np.min(deconv.x_linear), np.max(deconv.x_linear), 2000)
                    x_dense_log = x_dense if not deconv.use_log_x else np.log10(x_dense)
                    
                    colors = plt.cm.Set3(np.linspace(0, 1, len(deconv.components)))
                    for c, color in zip(deconv.components, colors):
                        if c.get('model_type', 'gaussian') == 'gaussian':
                            y_component_norm = GaussianModel.gaussian(x_dense_log, c['amp_norm'], 
                                                                     c['cen_log'], c['sigma_log']) 
                        elif c.get('model_type', 'gaussian') == 'lorentzian':
                            y_component_norm = GaussianModel.lorentzian(x_dense_log, c['amp_norm'], 
                                                                       c['cen_log'], c['sigma_log'])
                        elif c.get('model_type', 'gaussian') == 'pseudo_voigt':
                            eta = c.get('eta', 0.5)
                            y_component_norm = GaussianModel.pseudo_voigt(x_dense_log, c['amp_norm'], 
                                                                         c['cen_log'], c['sigma_log'], eta)
                        else:
                            gamma = c.get('gamma', c['sigma_log'] * 0.5)
                            y_component_norm = GaussianModel.voigt(x_dense_log, c['amp_norm'], 
                                                                  c['cen_log'], c['sigma_log'], gamma)
                        
                        y_component_norm = y_component_norm * c['amp_original'] / max_amp
                        
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
