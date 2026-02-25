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

# Настройка страницы
st.set_page_config(
    page_title="Гауссова деконволюция спектров",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Заголовок
st.title("📊 Гауссова деконволюция спектральных данных")
st.markdown("---")

# ==================== КЛАССЫ ====================

class DataParser:
    """Универсальный парсер для спектральных данных"""
    
    @staticmethod
    def parse_text(text):
        """Парсинг текста с данными в любом формате"""
        lines = text.strip().split('\n')
        x_data = []
        y_data = []
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith(('#', '//', ';')):
                continue
            
            # Разделяем по любым пробельным символам или запятым
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
        """Автоматическое определение необходимости логарифмических шкал"""
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
    """Анализ первой и второй производных для поиска пиков"""
    
    @staticmethod
    def calculate_derivatives(x, y, window_length=11, polyorder=3):
        """Расчет сглаженных производных"""
        if len(x) < window_length:
            window_length = len(x) if len(x) % 2 == 1 else len(x) - 1
        
        if window_length < polyorder + 2:
            return np.gradient(y, x), np.gradient(np.gradient(y, x), x), y
        
        # Сглаживание Савицкого-Голея
        y_smooth = savgol_filter(y, window_length, polyorder)
        dy = savgol_filter(y, window_length, polyorder, deriv=1, delta=np.mean(np.diff(x)))
        d2y = savgol_filter(y, window_length, polyorder, deriv=2, delta=np.mean(np.diff(x)))
        
        return dy, d2y, y_smooth
    
    @staticmethod
    def find_peaks_by_derivatives(x, y, dy, d2y, threshold=0.01):
        """Поиск пиков по пересечению нуля первой производной и отрицательной второй"""
        peaks = []
        for i in range(1, len(x) - 1):
            if (dy[i-1] > 0 and dy[i] <= 0) or (dy[i-1] >= 0 and dy[i] < 0):
                if d2y[i] < 0:
                    if y[i] > threshold * np.max(y):
                        peaks.append(i)
        return peaks


class GaussianModel:
    """Модель суммы гауссианов"""
    
    @staticmethod
    def gaussian(x, amp, cen, sigma):
        """Гауссиан"""
        return amp * np.exp(-(x - cen)**2 / (2 * max(sigma, 1e-12)**2))
    
    @staticmethod
    def multi_gaussian(x, *params):
        """Сумма нескольких гауссианов"""
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
        """Площадь под гауссианом"""
        return amp * sigma * np.sqrt(2 * np.pi)
    
    @staticmethod
    def calculate_fwhm(sigma):
        """Полная ширина на половине высоты"""
        return 2 * np.sqrt(2 * np.log(2)) * sigma


class FitQualityAnalyzer:
    """Анализ качества фиттинга"""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred, n_params):
        """Расчет метрик качества"""
        residuals = y_true - y_pred
        n = len(y_true)
        
        # R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # AIC и BIC
        rss = ss_res
        aic = n * np.log(rss/n) + 2 * n_params if rss > 0 else -np.inf
        bic = n * np.log(rss/n) + n_params * np.log(n) if rss > 0 else -np.inf
        
        # Chi-squared (редуцированный)
        chi_squared = rss / (n - n_params) if n > n_params else np.inf
        
        # Максимальная ошибка
        max_error = np.max(np.abs(residuals))
        
        # Среднеквадратичная ошибка
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
        """Обнаружение автокорреляции в остатках"""
        if len(residuals) < 10:
            return False
        
        diff = np.diff(residuals)
        dw = np.sum(diff**2) / np.sum(residuals**2)
        
        return dw < 1.5 or dw > 2.5


class GaussianDeconvolver:
    """Класс для деконволюции спектров"""
    
    def __init__(self, x_linear, y_original, use_log_x=True, use_log_y=False):
        self.x_linear = np.array(x_linear)
        self.y_original = np.array(y_original)
        self.use_log_x = use_log_x
        self.use_log_y = use_log_y
        
        # Сортировка по X
        sort_idx = np.argsort(self.x_linear)
        self.x_linear = self.x_linear[sort_idx]
        self.y_original = self.y_original[sort_idx]
        
        # Применение логарифмических преобразований
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
        
        # Нормализация для стабильности
        self.y_max = np.max(np.abs(self.y))
        if self.y_max > 0:
            self.y_norm = self.y / self.y_max
        else:
            self.y_norm = self.y
        
        # Результаты
        self.components = []
        self.fit_y_norm = None
        self.popt = None
        self.quality_metrics = {}
        self.convergence_history = []
        
        # Для совместимости
        self.multi_gaussian = GaussianModel.multi_gaussian
        self.gaussian = GaussianModel.gaussian
    
    def auto_detect_peaks(self, sensitivity=0.03, min_distance=5):
        """Автоматическое определение пиков с использованием производных"""
        # Сглаживание
        window_length = min(11, len(self.y_norm) // 5 * 2 + 1)
        if window_length % 2 == 0:
            window_length += 1
        
        if window_length >= 5:
            y_smooth = savgol_filter(self.y_norm, window_length, 3)
        else:
            y_smooth = self.y_norm
        
        # Расчет производных
        dy, d2y, y_smooth = DerivativeAnalyzer.calculate_derivatives(self.x, y_smooth)
        
        # Поиск пиков разными методами
        height_threshold = sensitivity * np.max(y_smooth)
        peaks1, _ = find_peaks(y_smooth, height=height_threshold, distance=min_distance)
        peaks2 = DerivativeAnalyzer.find_peaks_by_derivatives(self.x, y_smooth, dy, d2y, sensitivity)
        
        # Объединение результатов
        all_peaks = sorted(set(np.concatenate([peaks1, peaks2])))
        
        # Фильтрация слишком близких пиков
        filtered_peaks = []
        for peak in all_peaks:
            if not filtered_peaks or abs(self.x[peak] - self.x[filtered_peaks[-1]]) > min_distance * np.mean(np.diff(self.x)):
                filtered_peaks.append(peak)
        
        # Оценка параметров
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
        """Выполнение фиттинга"""
        if initial_params is None:
            _, _, initial_params, _ = self.auto_detect_peaks()
        
        n_peaks = len(initial_params) // 3
        
        # Установка границ
        lower_bounds = []
        upper_bounds = []
        x_range = np.max(self.x) - np.min(self.x)
        
        for i in range(n_peaks):
            lower_bounds.extend([0, np.min(self.x), x_range * 0.005])
            upper_bounds.extend([2 * np.max(self.y_norm), np.max(self.x), x_range * 0.5])
        
        try:
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
            
            # Извлечение компонентов
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
            
            # Расчет статистики
            total_area = sum([c['area'] for c in self.components])
            for c in self.components:
                c['fraction'] = c['area'] / total_area if total_area > 0 else 0
                c['fraction_percent'] = c['fraction'] * 100
            
            # Метрики качества
            self.quality_metrics = FitQualityAnalyzer.calculate_metrics(
                self.y_norm, self.fit_y_norm, len(popt)
            )
            
            return True
            
        except Exception as e:
            st.error(f"Ошибка фиттинга: {e}")
            return False
    
    def remove_peak(self, peak_id):
        """Удаление пика"""
        if peak_id > len(self.components):
            return False
        
        new_params = []
        for i, c in enumerate(self.components):
            if i != peak_id - 1:
                new_params.extend([c['amp_norm'], c['cen_log'], c['sigma_log']])
        
        if len(new_params) == 0:
            return False
        
        return self.fit(initial_params=new_params)
    
    def split_peak(self, peak_id):
        """Разделение пика на два"""
        if peak_id > len(self.components):
            return False
        
        peak = self.components[peak_id - 1]
        
        new_params = []
        for i, c in enumerate(self.components):
            if i == peak_id - 1:
                amp1 = c['amp_norm'] * 0.6
                amp2 = c['amp_norm'] * 0.4
                cen1 = c['cen_log'] - c['sigma_log'] * 0.5
                cen2 = c['cen_log'] + c['sigma_log'] * 0.5
                sigma1 = c['sigma_log'] * 0.7
                sigma2 = c['sigma_log'] * 0.7
                
                new_params.extend([amp1, cen1, sigma1])
                new_params.extend([amp2, cen2, sigma2])
            else:
                new_params.extend([c['amp_norm'], c['cen_log'], c['sigma_log']])
        
        return self.fit(initial_params=new_params)


# ==================== ДАННЫЕ ПО УМОЛЧАНИЮ ====================

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
0.008185466037560749, -0.00080383883826118
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
0.20706141239481027, -0.0016077218337867182
0.34250338264599406, -0.00080383883826118
"""


# ==================== ИНИЦИАЛИЗАЦИЯ СОСТОЯНИЯ ====================

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


# ==================== БОКОВАЯ ПАНЕЛЬ ====================

with st.sidebar:
    st.header("📋 Навигация")
    
    # Индикатор шагов
    steps = {
        1: "1. Загрузка данных",
        2: "2. Настройка шкал",
        3: "3. Поиск пиков",
        4: "4. Редактирование",
        5: "5. Результаты"
    }
    
    for step_num, step_name in steps.items():
        if step_num < st.session_state.current_step:
            st.success(f"✅ {step_name}")
        elif step_num == st.session_state.current_step:
            st.info(f"▶️ {step_name}")
        else:
            st.write(f"⏳ {step_name}")
    
    st.markdown("---")
    
    # Кнопка сброса
    if st.button("🔄 Начать заново", use_container_width=True):
        for key in ['deconvolver', 'raw_x', 'raw_y', 'peak_info', 'derivatives']:
            if key in st.session_state:
                st.session_state[key] = None
        st.session_state.current_step = 1
        st.rerun()


# ==================== ШАГ 1: ЗАГРУЗКА ДАННЫХ ====================

if st.session_state.current_step == 1:
    st.header("Шаг 1: Загрузка данных")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Текстовое поле для ввода данных
        data_text = st.text_area(
            "Вставьте данные (x y через пробел, запятую или табуляцию):",
            height=300,
            value=DEFAULT_DATA
        )
    
    with col2:
        st.subheader("Формат данных:")
        st.info(
            """
            Поддерживаются любые разделители:
            - Пробел
            - Запятая
            - Табуляция
            
            Примеры:
            ```
            1.23, 4.56
            1.23 4.56
            1.23\t4.56
            ```
            """
        )
        
        if st.button("📂 Загрузить данные", type="primary", use_container_width=True):
            x, y = DataParser.parse_text(data_text)
            
            if len(x) > 0:
                st.session_state.raw_x = x
                st.session_state.raw_y = y
                st.session_state.current_step = 2
                st.rerun()
            else:
                st.error("Не удалось распарсить данные. Проверьте формат.")
    
    # Предпросмотр
    if st.session_state.raw_x is not None:
        st.subheader("Предпросмотр данных:")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(st.session_state.raw_x, st.session_state.raw_y, 'o-', markersize=3, linewidth=1)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title('Линейные шкалы')
        ax1.grid(True, alpha=0.3)
        
        if np.min(st.session_state.raw_x[st.session_state.raw_x > 0]) > 0:
            ax2.loglog(st.session_state.raw_x, np.maximum(st.session_state.raw_y, 1e-12), 
                      'o-', markersize=3, linewidth=1)
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_title('Лог-лог шкалы')
            ax2.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ==================== ШАГ 2: НАСТРОЙКА ШКАЛ ====================

elif st.session_state.current_step == 2:
    st.header("Шаг 2: Настройка шкал")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Параметры шкал")
        
        # Автоопределение
        if st.button("🔍 Автоопределить шкалы", use_container_width=True):
            suggest_log_x, suggest_log_y = DataParser.auto_detect_scale(
                st.session_state.raw_x, st.session_state.raw_y
            )
            st.session_state.use_log_x = suggest_log_x
            st.session_state.use_log_y = suggest_log_y
            st.rerun()
        
        # Ручная настройка
        st.session_state.use_log_x = st.checkbox("Логарифмическая шкала X", value=st.session_state.use_log_x)
        st.session_state.use_log_y = st.checkbox("Логарифмическая шкала Y", value=st.session_state.use_log_y)
        
        if st.button("✅ Применить и продолжить", type="primary", use_container_width=True):
            st.session_state.current_step = 3
            st.rerun()
    
    with col2:
        st.subheader("Предпросмотр:")
        
        # Визуализация с выбранными шкалами
        fig, ax = plt.subplots(figsize=(8, 5))
        
        x = st.session_state.raw_x
        y = st.session_state.raw_y
        
        if st.session_state.use_log_x:
            x = x[x > 0]
            y = y[x > 0]
            ax.set_xscale('log')
        
        if st.session_state.use_log_y:
            y = y[y > 0]
            x = x[y > 0]
            ax.set_yscale('log')
        
        ax.plot(x, y, 'o-', markersize=3, linewidth=1)
        ax.set_xlabel('X' + (' (log)' if st.session_state.use_log_x else ''))
        ax.set_ylabel('Y' + (' (log)' if st.session_state.use_log_y else ''))
        ax.set_title('Данные после применения шкал')
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        plt.close()


# ==================== ШАГ 3: ПОИСК ПИКОВ ====================

elif st.session_state.current_step == 3:
    st.header("Шаг 3: Поиск пиков")
    
    # Создание деконвольвера если еще не создан
    if st.session_state.deconvolver is None:
        st.session_state.deconvolver = GaussianDeconvolver(
            st.session_state.raw_x,
            st.session_state.raw_y,
            use_log_x=st.session_state.use_log_x,
            use_log_y=st.session_state.use_log_y
        )
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Параметры поиска")
        
        st.session_state.sensitivity = st.slider(
            "Чувствительность:",
            min_value=0.001,
            max_value=0.1,
            value=st.session_state.sensitivity,
            step=0.001,
            format="%.3f"
        )
        
        st.session_state.min_distance = st.slider(
            "Минимальное расстояние между пиками:",
            min_value=1,
            max_value=20,
            value=st.session_state.min_distance,
            step=1
        )
        
        if st.button("🔍 Найти пики", type="primary", use_container_width=True):
            peaks, peak_info, initial_params, derivatives = st.session_state.deconvolver.auto_detect_peaks(
                sensitivity=st.session_state.sensitivity,
                min_distance=st.session_state.min_distance
            )
            st.session_state.peak_info = peak_info
            st.session_state.derivatives = derivatives
            
        if st.session_state.peak_info is not None and st.button("✅ Подтвердить пики", use_container_width=True):
            if st.session_state.deconvolver.fit():
                st.session_state.current_step = 4
                st.rerun()
    
    with col2:
        if st.session_state.peak_info is not None and st.session_state.derivatives is not None:
            st.subheader(f"Найдено пиков: {len(st.session_state.peak_info)}")
            
            dy, d2y, y_smooth = st.session_state.derivatives
            
            # Создание вкладок для разных графиков
            tab1, tab2, tab3 = st.tabs(["📊 Пики", "📈 Производные", "📋 Информация"])
            
            with tab1:
                fig, ax = plt.subplots(figsize=(10, 5))
                
                ax.plot(st.session_state.deconvolver.x, st.session_state.deconvolver.y_norm, 
                       'o-', markersize=3, alpha=0.5, label='Данные')
                ax.plot(st.session_state.deconvolver.x, y_smooth, 
                       'r-', linewidth=2, label='Сглаженные')
                
                for i, info in enumerate(st.session_state.peak_info):
                    ax.plot(info['x'], info['y'], 'ro', markersize=8, markeredgecolor='darkred')
                    ax.text(info['x'], info['y']*1.05, f'{i+1}', ha='center', fontweight='bold')
                
                ax.set_xlabel(st.session_state.deconvolver.x_label)
                ax.set_ylabel('Нормализованная Y')
                ax.set_title('Обнаруженные пики')
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
                ax1.set_title('Первая производная')
                ax1.grid(True, alpha=0.3)
                
                ax2.plot(st.session_state.deconvolver.x, d2y, 'g-', linewidth=1.5)
                ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                ax2.set_xlabel(st.session_state.deconvolver.x_label)
                ax2.set_ylabel('d²y/dx²')
                ax2.set_title('Вторая производная')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            with tab3:
                data = []
                for i, info in enumerate(st.session_state.peak_info):
                    data.append({
                        'Пик': i + 1,
                        'Центр (лог)': f"{info['x']:.4f}",
                        'Центр': f"{info['x_linear']:.2e}",
                        'Амплитуда': f"{info['y']:.4f}",
                        'Сигма': f"{info['sigma_est']:.4f}"
                    })
                
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)


# ==================== ШАГ 4: РЕДАКТИРОВАНИЕ ====================

elif st.session_state.current_step == 4:
    st.header("Шаг 4: Редактирование пиков")
    
    if st.session_state.deconvolver and st.session_state.deconvolver.components:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Управление пиками")
            
            # Выбор пика
            peak_options = {f"Пик {c['id']}: центр = {c['cen_linear']:.2e}, доля = {c['fraction_percent']:.1f}%": c['id'] 
                           for c in st.session_state.deconvolver.components}
            
            selected_peak = st.selectbox(
                "Выберите пик для редактирования:",
                options=list(peak_options.keys())
            )
            
            if selected_peak:
                peak_id = peak_options[selected_peak]
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    if st.button("✂️ Разделить пик", use_container_width=True):
                        if st.session_state.deconvolver.split_peak(peak_id):
                            st.rerun()
                
                with col_b:
                    if st.button("🗑️ Удалить пик", use_container_width=True):
                        if st.session_state.deconvolver.remove_peak(peak_id):
                            st.rerun()
                
                if st.button("🔄 Пересчитать все", use_container_width=True):
                    if st.session_state.deconvolver.fit(initial_params=st.session_state.deconvolver.popt):
                        st.rerun()
            
            st.markdown("---")
            
            if st.button("✅ Завершить редактирование", type="primary", use_container_width=True):
                st.session_state.current_step = 5
                st.rerun()
        
        with col2:
            st.subheader("Текущая деконволюция")
            
            # Создание интерактивного графика с Plotly
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Деконволюция', 'Остатки', 'Компоненты', 'Метрики'),
                specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                       [{'type': 'bar'}, {'type': 'table'}]]
            )
            
            # Основной график
            fig.add_trace(
                go.Scatter(x=st.session_state.deconvolver.x, 
                          y=st.session_state.deconvolver.y_norm,
                          mode='markers+lines',
                          name='Данные',
                          marker=dict(size=4, color='black')),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=st.session_state.deconvolver.x, 
                          y=st.session_state.deconvolver.fit_y_norm,
                          mode='lines',
                          name='Суммарный фит',
                          line=dict(color='red', width=2)),
                row=1, col=1
            )
            
            # Компоненты
            colors = plt.cm.Set3(np.linspace(0, 1, len(st.session_state.deconvolver.components)))
            for c, color in zip(st.session_state.deconvolver.components, colors):
                # Используем RGBA формат с прозрачностью
                rgba_color = f'rgba({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}, 0.3)'
                fig.add_trace(
                    go.Scatter(x=st.session_state.deconvolver.x, 
                              y=c['y_norm'],
                              mode='lines',
                              name=f'Пик {c["id"]} ({c["fraction_percent"]:.1f}%)',
                              line=dict(color=f'rgb({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})', width=1.5),
                              fill='tozeroy',
                              fillcolor=rgba_color),  # Используем RGBA с прозрачностью
                    row=1, col=1
                )
            
            # Остатки
            if 'Residuals' in st.session_state.deconvolver.quality_metrics:
                residuals = st.session_state.deconvolver.quality_metrics['Residuals']
                fig.add_trace(
                    go.Scatter(x=st.session_state.deconvolver.x, 
                              y=residuals,
                              mode='lines',
                              name='Остатки',
                              line=dict(color='blue', width=1)),
                    row=1, col=2
                )
                fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
            
            # Гистограмма компонентов
            centers = [c['cen_linear'] for c in st.session_state.deconvolver.components]
            fractions = [c['fraction_percent'] for c in st.session_state.deconvolver.components]
            
            fig.add_trace(
                go.Bar(x=[f"Пик {c['id']}" for c in st.session_state.deconvolver.components], 
                      y=fractions,
                      name='Доли (%)',
                      marker_color='steelblue'),
                row=2, col=1
            )
            
            # Таблица метрик
            metrics = st.session_state.deconvolver.quality_metrics
            metrics_table = go.Table(
                header=dict(values=['Метрика', 'Значение'],
                           fill_color='paleturquoise',
                           align='left'),
                cells=dict(values=[
                    ['R²', 'AIC', 'BIC', 'χ²', 'RMSE'],
                    [f"{metrics.get('R²', 0):.6f}", 
                     f"{metrics.get('AIC', 0):.2f}",
                     f"{metrics.get('BIC', 0):.2f}",
                     f"{metrics.get('χ²', 0):.2e}",
                     f"{metrics.get('RMSE', 0):.2e}"]
                ],
                fill_color='lavender',
                align='left')
            )
            fig.add_trace(metrics_table, row=2, col=2)
            
            fig.update_layout(height=700, showlegend=True, title_text="")
            fig.update_xaxes(title_text=st.session_state.deconvolver.x_label, row=1, col=1)
            fig.update_xaxes(title_text=st.session_state.deconvolver.x_label, row=1, col=2)
            fig.update_yaxes(title_text="Нормализованная Y", row=1, col=1)
            fig.update_yaxes(title_text="Остатки", row=1, col=2)
            fig.update_yaxes(title_text="Доля (%)", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)


# ==================== ШАГ 5: РЕЗУЛЬТАТЫ ====================

elif st.session_state.current_step == 5:
    st.header("Шаг 5: Результаты")
    
    if st.session_state.deconvolver and st.session_state.deconvolver.components:
        
        # Создание вкладок для результатов
        tab1, tab2, tab3 = st.tabs(["📊 Графики", "📋 Таблица", "📈 Экспорт"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Результат деконволюции")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                ax.scatter(st.session_state.deconvolver.x_linear, 
                          st.session_state.deconvolver.y_original, 
                          s=10, alpha=0.5, color='black', label='Данные')
                
                if st.session_state.deconvolver.use_log_x:
                    x_dense = np.logspace(np.log10(np.min(st.session_state.deconvolver.x_linear[
                        st.session_state.deconvolver.x_linear>0])),
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
                           label=f'Пик {c["id"]}: {c["fraction_percent"]:.1f}%')
                
                y_total = GaussianModel.multi_gaussian(x_dense_log, *st.session_state.deconvolver.popt) * st.session_state.deconvolver.y_max
                ax.plot(x_dense, y_total, 'r--', linewidth=2, label='Суммарный фит')
                
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_title('Результат деконволюции')
                ax.legend(loc='upper right', fontsize=8)
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                plt.close()
            
            with col2:
                st.subheader("Распределение площадей")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Круговая диаграмма
                peaks = [f'{c["id"]}' for c in st.session_state.deconvolver.components]
                fractions = [c['fraction_percent'] for c in st.session_state.deconvolver.components]
                colors = plt.cm.Set3(np.linspace(0, 1, len(peaks)))
                ax1.pie(fractions, labels=peaks, autopct='%1.1f%%',
                       colors=colors, startangle=90)
                ax1.set_title('Распределение площадей')
                
                # Столбчатая диаграмма
                centers = [c['cen_linear'] for c in st.session_state.deconvolver.components]
                areas = [c['area'] for c in st.session_state.deconvolver.components]
                
                if st.session_state.deconvolver.use_log_x:
                    ax2.set_xscale('log')
                
                ax2.bar(range(len(centers)), areas, 
                       tick_label=[f'{c:.2e}' for c in centers],
                       color='steelblue', edgecolor='black', alpha=0.7)
                ax2.set_xlabel('Центр пика')
                ax2.set_ylabel('Площадь')
                ax2.set_title('Площади пиков')
                ax2.tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        
        with tab2:
            st.subheader("Таблица результатов")
            
            data = []
            for c in st.session_state.deconvolver.components:
                data.append({
                    'Пик': c['id'],
                    'Центр': f"{c['cen_linear']:.4e}",
                    'Амплитуда': f"{c['amp']:.4e}",
                    'Сигма': f"{c['sigma_log']:.4f}",
                    'FWHM': f"{c['fwhm']:.4f}",
                    'Площадь': f"{c['area']:.4e}",
                    'Доля (%)': f"{c['fraction_percent']:.2f}"
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
            st.subheader("Экспорт результатов")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Экспорт в CSV
                if st.button("📥 Экспорт в CSV", use_container_width=True):
                    # Данные пиков
                    df_peaks = pd.DataFrame([{
                        'Peak_ID': c['id'],
                        'Center': c['cen_linear'],
                        'Amplitude': c['amp'],
                        'Sigma': c['sigma_log'],
                        'FWHM': c['fwhm'],
                        'Area': c['area'],
                        'Fraction_Percent': c['fraction_percent']
                    } for c in st.session_state.deconvolver.components])
                    
                    # Конвертация в CSV
                    csv_peaks = df_peaks.to_csv(index=False)
                    
                    st.download_button(
                        label="Скачать CSV с пиками",
                        data=csv_peaks,
                        file_name=f"deconvolution_peaks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Фиттинг данные
                    if 'Residuals' in st.session_state.deconvolver.quality_metrics:
                        df_fit = pd.DataFrame({
                            'X_original': st.session_state.deconvolver.x_linear,
                            'Y_original': st.session_state.deconvolver.y_original,
                            'Y_fit': st.session_state.deconvolver.fit_y_norm * st.session_state.deconvolver.y_max,
                            'Residuals': st.session_state.deconvolver.quality_metrics['Residuals'] * st.session_state.deconvolver.y_max
                        })
                        
                        csv_fit = df_fit.to_csv(index=False)
                        
                        st.download_button(
                            label="Скачать CSV с фиттингом",
                            data=csv_fit,
                            file_name=f"deconvolution_fit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
            
            with col2:
                # Экспорт отчета
                if st.button("📄 Экспорт отчета", use_container_width=True):
                    report = f"""ОТЧЕТ ПО ДЕКОНВОЛЮЦИИ ГАУССИАНАМИ
{"="*80}

Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Число точек: {len(st.session_state.deconvolver.x_linear)}
Диапазон X: [{st.session_state.deconvolver.x_linear[0]:.2e}, {st.session_state.deconvolver.x_linear[-1]:.2e}]
Логарифмическая шкала X: {st.session_state.deconvolver.use_log_x}

МЕТРИКИ КАЧЕСТВА:
{"-"*40}
R²: {st.session_state.deconvolver.quality_metrics.get('R²', 0):.6f}
AIC: {st.session_state.deconvolver.quality_metrics.get('AIC', 0):.2f}
BIC: {st.session_state.deconvolver.quality_metrics.get('BIC', 0):.2f}
χ²: {st.session_state.deconvolver.quality_metrics.get('χ²', 0):.2e}
RMSE: {st.session_state.deconvolver.quality_metrics.get('RMSE', 0):.2e}

КОМПОНЕНТЫ:
{"-"*80}
ID    Центр           Амплитуда       FWHM        Площадь        Доля(%)
{"-"*80}"""
                    
                    for c in st.session_state.deconvolver.components:
                        report += f"\n{c['id']:<4} {c['cen_linear']:<15.4e} {c['amp']:<15.4e} {c['fwhm']:<12.4f} {c['area']:<15.4e} {c['fraction_percent']:<10.2f}"
                    
                    report += f"\n{'='*80}\nОбщая площадь: {st.session_state.deconvolver.total_area:.6e}\n{'='*80}"
                    
                    st.download_button(
                        label="Скачать отчет",
                        data=report,
                        file_name=f"deconvolution_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
            
            st.markdown("---")
            
            if st.button("🔄 Новый анализ", use_container_width=True):
                for key in ['deconvolver', 'raw_x', 'raw_y', 'peak_info', 'derivatives']:
                    if key in st.session_state:
                        st.session_state[key] = None
                st.session_state.current_step = 1

                st.rerun()

