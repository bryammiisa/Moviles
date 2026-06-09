import time
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image
from scipy import stats as _stats
import customtkinter as ctk
from tkinter import filedialog

# =========================================================
# PALETA BLANCA / LIGHT
# =========================================================
WHITE      = "#FFFFFF"
GRAY_50    = "#F9FAFB"
GRAY_100   = "#F3F4F6"
GRAY_200   = "#E5E7EB"
GRAY_300   = "#D1D5DB"
GRAY_500   = "#6B7280"
GRAY_700   = "#374151"
GRAY_900   = "#111827"

BLUE_600   = "#2563EB"
BLUE_400   = "#60A5FA"
BLUE_100   = "#DBEAFE"

GREEN_600  = "#16A34A"
RED_600    = "#DC2626"
AMBER_600  = "#D97706"

# matplotlib — fondo blanco
FIG_BG   = WHITE
AX_BG    = GRAY_50
TEXT_CLR = GRAY_700
GRID_CLR = GRAY_200

# =========================================================
# CONSTANTES OFDM
# =========================================================
PILOT_VALUE   = 1 + 1j
SMOOTH_KERNEL = np.ones(5) / 5

MODULATION_MAP = {
    "QPSK": {
        "bits_per_symbol": 2,
        "norm": np.sqrt(2),
        "mapping": {
            (0, 0): -1 - 1j, (0, 1): -1 + 1j,
            (1, 1):  1 + 1j, (1, 0):  1 - 1j,
        },
    },
    "16QAM": {
        "bits_per_symbol": 4,
        "norm": np.sqrt(10),
        "mapping": {
            (0,0,0,0): -3-3j, (0,0,0,1): -3-1j, (0,0,1,1): -3+1j, (0,0,1,0): -3+3j,
            (0,1,0,0): -1-3j, (0,1,0,1): -1-1j, (0,1,1,1): -1+1j, (0,1,1,0): -1+3j,
            (1,1,0,0):  1-3j, (1,1,0,1):  1-1j, (1,1,1,1):  1+1j, (1,1,1,0):  1+3j,
            (1,0,0,0):  3-3j, (1,0,0,1):  3-1j, (1,0,1,1):  3+1j, (1,0,1,0):  3+3j,
        },
    },
    "64QAM": {
        "bits_per_symbol": 6,
        "norm": np.sqrt(42),
        "mapping": {
            (0,0,0,0,0,0): -7-7j, (0,0,0,0,0,1): -7-5j, (0,0,0,0,1,1): -7-3j, (0,0,0,0,1,0): -7-1j,
            (0,0,0,1,1,0): -7+1j, (0,0,0,1,1,1): -7+3j, (0,0,0,1,0,1): -7+5j, (0,0,0,1,0,0): -7+7j,
            (0,0,1,0,0,0): -5-7j, (0,0,1,0,0,1): -5-5j, (0,0,1,0,1,1): -5-3j, (0,0,1,0,1,0): -5-1j,
            (0,0,1,1,1,0): -5+1j, (0,0,1,1,1,1): -5+3j, (0,0,1,1,0,1): -5+5j, (0,0,1,1,0,0): -5+7j,
            (0,1,1,0,0,0): -3-7j, (0,1,1,0,0,1): -3-5j, (0,1,1,0,1,1): -3-3j, (0,1,1,0,1,0): -3-1j,
            (0,1,1,1,1,0): -3+1j, (0,1,1,1,1,1): -3+3j, (0,1,1,1,0,1): -3+5j, (0,1,1,1,0,0): -3+7j,
            (0,1,0,0,0,0): -1-7j, (0,1,0,0,0,1): -1-5j, (0,1,0,0,1,1): -1-3j, (0,1,0,0,1,0): -1-1j,
            (0,1,0,1,1,0): -1+1j, (0,1,0,1,1,1): -1+3j, (0,1,0,1,0,1): -1+5j, (0,1,0,1,0,0): -1+7j,
            (1,1,0,0,0,0):  1-7j, (1,1,0,0,0,1):  1-5j, (1,1,0,0,1,1):  1-3j, (1,1,0,0,1,0):  1-1j,
            (1,1,0,1,1,0):  1+1j, (1,1,0,1,1,1):  1+3j, (1,1,0,1,0,1):  1+5j, (1,1,0,1,0,0):  1+7j,
            (1,1,1,0,0,0):  3-7j, (1,1,1,0,0,1):  3-5j, (1,1,1,0,1,1):  3-3j, (1,1,1,0,1,0):  3-1j,
            (1,1,1,1,1,0):  3+1j, (1,1,1,1,1,1):  3+3j, (1,1,1,1,0,1):  3+5j, (1,1,1,1,0,0):  3+7j,
            (1,0,1,0,0,0):  5-7j, (1,0,1,0,0,1):  5-5j, (1,0,1,0,1,1):  5-3j, (1,0,1,0,1,0):  5-1j,
            (1,0,1,1,1,0):  5+1j, (1,0,1,1,1,1):  5+3j, (1,0,1,1,0,1):  5+5j, (1,0,1,1,0,0):  5+7j,
            (1,0,0,0,0,0):  7-7j, (1,0,0,0,0,1):  7-5j, (1,0,0,0,1,1):  7-3j, (1,0,0,0,1,0):  7-1j,
            (1,0,0,1,1,0):  7+1j, (1,0,0,1,1,1):  7+3j, (1,0,0,1,0,1):  7+5j, (1,0,0,1,0,0):  7+7j,
        },
    },
}

BW_MAP = {
    "1.4MHz": 1.4e6, "3MHz": 3e6,  "5MHz": 5e6,
    "10MHz": 10e6,   "15MHz": 15e6, "20MHz": 20e6,
}
SPACING_MAP = {
    "7.5kHz": 7.5e3, "15kHz": 15e3, "30kHz": 30e3, "60kHz": 60e3,
}

# =========================================================
# IMAGEN <-> BITS
# =========================================================
def image_to_bits(image_path):
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    return np.unpackbits(arr.flatten()).astype(int), arr.shape

def bits_to_image(bits, shape, path="imagen_rx.png"):
    total_bits = shape[0] * shape[1] * shape[2] * 8
    Image.fromarray(
        np.packbits(bits[:total_bits]).reshape(shape).astype(np.uint8)
    ).save(path)

# =========================================================
# MODULACION / DEMODULACION
# =========================================================
def modulate_bits(bits, modulation):
    cfg = MODULATION_MAP[modulation]
    bps = cfg["bits_per_symbol"]
    bits = bits[: (len(bits) // bps) * bps]
    symbols = np.array(
        [cfg["mapping"][tuple(b)] for b in bits.reshape(-1, bps)]
    ) / cfg["norm"]
    return bits, symbols

def demodulate_symbols(symbols, modulation):
    cfg = MODULATION_MAP[modulation]
    constellation = np.array(list(cfg["mapping"].values())) / cfg["norm"]
    bit_patterns  = list(cfg["mapping"].keys())
    indices = np.argmin(np.abs(symbols[:, None] - constellation[None, :]), axis=1)
    return np.array([bit for idx in indices for bit in bit_patterns[idx]])

# =========================================================
# OFDM
# =========================================================
def _build_pilot_frame(data_symbols, Nfft, pilot_spacing):
    pilot_idx = np.arange(0, Nfft, pilot_spacing)
    data_idx  = np.setdiff1d(np.arange(Nfft), pilot_idx)
    n_data    = min(len(data_symbols), len(data_idx))
    frame = np.zeros(Nfft, dtype=complex)
    frame[pilot_idx]         = PILOT_VALUE
    frame[data_idx[:n_data]] = data_symbols[:n_data]
    return frame, pilot_idx, n_data

def ofdm_tx(symbols, Nfft, cp_len, pilot_spacing=8):
    tx_signal, pil_list, dcl = [], [], []
    ptr = 0
    while ptr < len(symbols):
        frame, pilot_idx, n_data = _build_pilot_frame(symbols[ptr:], Nfft, pilot_spacing)
        ptr += n_data
        pil_list.append(pilot_idx)
        dcl.append(n_data)
        t = np.fft.ifft(frame)
        tx_signal.extend(np.concatenate([t[-cp_len:], t]))
    return np.array(tx_signal), pil_list, dcl

def ofdm_rx(rx_signal, Nfft, cp_len, pil_list, dcl, snr_db, equalizer_on=True):
    symbol_len = Nfft + cp_len
    n_sym      = len(rx_signal) // symbol_len
    frames     = rx_signal[: n_sym * symbol_len].reshape(n_sym, symbol_len)
    noise_var  = 10 ** (-snr_db / 10)
    recovered  = []
    for i, frame in enumerate(frames):
        freq      = np.fft.fft(frame[cp_len:])
        pilot_idx = pil_list[i]
        data_idx  = np.setdiff1d(np.arange(Nfft), pilot_idx)
        if equalizer_on:
            H_est = freq[pilot_idx] / PILOT_VALUE
            def _smooth(v):
                return np.convolve(
                    np.interp(np.arange(Nfft), pilot_idx, v),
                    SMOOTH_KERNEL, mode="same"
                )
            H_interp = _smooth(np.abs(H_est)) * np.exp(
                1j * _smooth(np.unwrap(np.angle(H_est)))
            )
            freq = (freq * np.conj(H_interp)) / (np.abs(H_interp) ** 2 + noise_var)
        recovered.extend(freq[data_idx[: dcl[i]]])
    return np.array(recovered)

# =========================================================
# CANAL
# =========================================================
def awgn(signal, snr_db):
    p     = np.mean(np.abs(signal) ** 2) / (10 ** (snr_db / 10))
    noise = np.sqrt(p / 2) * (
        np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
    )
    return signal + noise

def multipath_channel(signal):
    h = np.array([
        1.0 + 0j,
        0.5 * np.exp(1j * np.random.uniform(0, 2 * np.pi)),
        0.3 * np.exp(1j * np.random.uniform(0, 2 * np.pi)),
    ])
    return np.convolve(signal, h, mode="same"), h

# =========================================================
# METRICAS
# =========================================================
def calculate_ber(bits_tx, bits_rx):
    e = np.sum(bits_tx != bits_rx)
    return e / len(bits_tx), e

def calculate_papr(signal):
    p = np.abs(signal) ** 2
    return 10 * np.log10(np.max(p) / np.mean(p))

# =========================================================
# PARAMETROS OFDM
# =========================================================
def get_ofdm_parameters(spacing_value, bandwidth_value, cp_mode):
    delta_f   = SPACING_MAP[spacing_value]
    bandwidth = BW_MAP[bandwidth_value]
    useful_bw = 0.90 * bandwidth
    guard_bw  = 0.10 * bandwidth
    Nfft      = 2 ** int(np.ceil(np.log2(int(useful_bw / delta_f))))
    Fs        = Nfft * delta_f
    Tcp       = 4.7e-6 if cp_mode == "Normal" else (33e-6 if delta_f == 7.5e3 else 16.6e-6)
    return delta_f, bandwidth, useful_bw, guard_bw, Nfft, Fs, Tcp, int(Tcp * Fs)

# =========================================================
# SIMULACION BER vs SNR con intervalo de confianza
# =========================================================
def simulate_ber(bits_original, modulation, Nfft, cp_len, snr_range,
                 n_runs=10, confidence=0.95):
    z        = _stats.norm.ppf(0.5 + confidence / 2)
    fragment = 15000
    ber_mean, ci_low, ci_high = [], [], []

    for snr_db in snr_range:
        runs = []
        for _ in range(n_runs):
            idx       = np.random.randint(0, max(1, len(bits_original) - fragment))
            bits_frag = bits_original[idx : idx + fragment]
            bits_f, symbols_f  = modulate_bits(bits_frag, modulation)
            tx_f, pil_f, dcl_f = ofdm_tx(symbols_f, Nfft, cp_len, pilot_spacing=8)
            rx_sym  = ofdm_rx(awgn(tx_f, snr_db), Nfft, cp_len, pil_f, dcl_f, snr_db, False)
            bits_rx = demodulate_symbols(rx_sym, modulation)
            ber, _  = calculate_ber(bits_f, bits_rx)
            runs.append(ber)

        runs   = np.array(runs)
        mean   = max(runs.mean(), 1e-7)
        half_w = z * runs.std(ddof=1) / np.sqrt(n_runs)
        ber_mean.append(mean)
        ci_low.append(max(mean - half_w, 1e-7))
        ci_high.append(mean + half_w)

    return np.array(ber_mean), np.array(ci_low), np.array(ci_high)

# =========================================================
# HELPER MATPLOTLIB — estilo blanco
# =========================================================
def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors=GRAY_700, labelsize=7)
    ax.xaxis.label.set_color(GRAY_500)
    ax.yaxis.label.set_color(GRAY_500)
    ax.title.set_color(GRAY_900)
    ax.title.set_fontsize(8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRAY_300)
    ax.grid(True, color=GRID_CLR, linewidth=0.5)
    if title:  ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel, fontsize=7)
    if ylabel: ax.set_ylabel(ylabel, fontsize=7)

# =========================================================
# GUI
# =========================================================
class OFDMApp(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("Sistema OFDM LTE")
        self.geometry("1340x860")
        ctk.set_appearance_mode("light")
        self.configure(fg_color=GRAY_100)
        self.image_path = None

        self.scrollable = ctk.CTkScrollableFrame(
            self, width=1380, height=950,
            fg_color=GRAY_100,
            scrollbar_button_color=GRAY_300,
        )
        self.scrollable.pack(fill="both", expand=True)

        self.left_frame = ctk.CTkFrame(
            self.scrollable, fg_color=WHITE,
            corner_radius=12, border_width=1, border_color=GRAY_200,
        )
        self.left_frame.grid(row=0, column=0, sticky="nswe", padx=(10, 6), pady=10)

        self.right_frame = ctk.CTkFrame(
            self.scrollable, fg_color=WHITE,
            corner_radius=12, border_width=1, border_color=GRAY_200,
        )
        self.right_frame.grid(row=0, column=1, sticky="nswe", padx=(6, 10), pady=10)

        self._build_controls()
        self._build_metrics()
        self._build_plots()
        self._build_timing_panel()

    # ── helpers sidebar ──────────────────────────────────
    def _section(self, text):
        ctk.CTkFrame(self.left_frame, height=1, fg_color=GRAY_200).pack(
            fill="x", padx=12, pady=(8, 2)
        )
        ctk.CTkLabel(
            self.left_frame, text=text.upper(),
            font=("Arial", 9), text_color=GRAY_500,
        ).pack(anchor="w", padx=14)

    def _labeled_option(self, label, values):
        ctk.CTkLabel(
            self.left_frame, text=label,
            font=("Arial", 11), text_color=GRAY_700,
        ).pack(anchor="w", padx=14, pady=(6, 0))
        w = ctk.CTkOptionMenu(
            self.left_frame, values=values,
            fg_color=GRAY_100,
            button_color=GRAY_300,
            button_hover_color=GRAY_200,
            text_color=GRAY_900,
            dropdown_fg_color=WHITE,
            dropdown_text_color=GRAY_700,
            dropdown_hover_color=GRAY_100,
            font=("Arial", 11),
            corner_radius=6, width=200,
        )
        w.pack(padx=14, pady=(2, 0))
        return w

    def _build_controls(self):
        ctk.CTkLabel(
            self.left_frame, text="OFDM LTE",
            font=("Arial", 20, "bold"), text_color=BLUE_600,
        ).pack(pady=(18, 4))
        ctk.CTkLabel(
            self.left_frame, text="TX / RX ",
            font=("Arial", 10), text_color=GRAY_500,
        ).pack(pady=(0, 10))

        self._section("Imagen")
        ctk.CTkButton(
            self.left_frame, text="Cargar imagen",
            command=self.load_image,
            fg_color=WHITE, hover_color=GRAY_100,
            text_color=BLUE_600, font=("Arial", 11),
            corner_radius=6, border_width=1, border_color=BLUE_400, width=200,
        ).pack(padx=14, pady=6)
        self.image_label = ctk.CTkLabel(self.left_frame, text="")
        self.image_label.pack()

        self._section("Modulacion")
        self.modulation = self._labeled_option("Esquema", ["QPSK", "16QAM", "64QAM"])

        self._section("Parametros OFDM")
        self.spacing   = self._labeled_option("Espaciamiento", list(SPACING_MAP))
        self.bandwidth = self._labeled_option("Ancho de banda", list(BW_MAP))
        self.cp_type   = self._labeled_option("Tipo CP", ["Normal", "Extendido"])

        self._section("Canal")
        self.channel_type = self._labeled_option("Modelo", ["AWGN", "Multipath"])
        self.equalization = self._labeled_option("Ecualizador", ["ON", "OFF"])

        ctk.CTkLabel(
            self.left_frame, text="SNR (dB)",
            font=("Arial", 11), text_color=GRAY_700,
        ).pack(anchor="w", padx=14, pady=(6, 0))
        self.snr = ctk.CTkEntry(
            self.left_frame, fg_color=GRAY_100, border_color=GRAY_300,
            text_color=GRAY_900, font=("Arial", 12), corner_radius=6, width=200,
        )
        self.snr.insert(0, "20")
        self.snr.pack(padx=14, pady=(2, 0))

        self._section("Acciones")
        ctk.CTkButton(
            self.left_frame, text="Ejecutar sistema",
            command=self.run_system,
            fg_color=BLUE_600, hover_color=BLUE_400,
            text_color=WHITE, font=("Arial", 12, "bold"),
            corner_radius=8, width=200, height=36,
        ).pack(padx=14, pady=(8, 4))

        self._section("BER vs SNR")
        ctk.CTkLabel(
            self.left_frame, text="Realizaciones / punto",
            font=("Arial", 11), text_color=GRAY_700,
        ).pack(anchor="w", padx=14, pady=(6, 0))
        self.ber_runs = ctk.CTkOptionMenu(
            self.left_frame, values=["5", "10", "20", "50"],
            fg_color=GRAY_100, button_color=GRAY_300,
            button_hover_color=GRAY_200, text_color=GRAY_900,
            dropdown_fg_color=WHITE, dropdown_text_color=GRAY_700,
            dropdown_hover_color=GRAY_100,
            font=("Arial", 11), corner_radius=6, width=200,
        )
        self.ber_runs.set("10")
        self.ber_runs.pack(padx=14, pady=(2, 0))


        ctk.CTkButton(
            self.left_frame, text="Generar BER vs SNR",
            command=self.generate_ber_plot,
            fg_color=WHITE, hover_color=GRAY_100,
            text_color=BLUE_600, font=("Arial", 11),
            border_width=1, border_color=BLUE_400,
            corner_radius=8, width=200, height=34,
        ).pack(padx=14, pady=(8, 8))

        self._section("Estado")
        self.status = ctk.CTkLabel(
            self.left_frame, text="Esperando...",
            font=("Arial", 10), text_color=GRAY_500, justify="left",
        )
        self.status.pack(anchor="w", padx=14, pady=(6, 16))

    # ── tarjetas metricas ────────────────────────────────
    def _build_metrics(self):
        self.metrics_frame = ctk.CTkFrame(self.right_frame, fg_color=WHITE)
        self.metrics_frame.pack(fill="x", padx=12, pady=(12, 0))

        self._metric_labels = {}
        specs = [
            ("BER", "—"), ("PAPR", "—"), ("Nfft", "—"),
            ("Fs",  "—"), ("BW util", "—"), ("Simb.", "—"),
        ]
        for i, (name, _) in enumerate(specs):
            card = ctk.CTkFrame(
                self.metrics_frame, fg_color=GRAY_50,
                corner_radius=8, border_width=1, border_color=GRAY_200,
            )
            card.grid(row=0, column=i, padx=4, pady=0, sticky="ew")
            self.metrics_frame.columnconfigure(i, weight=1)
            ctk.CTkLabel(card, text=name, font=("Arial", 9), text_color=GRAY_500).pack(pady=(6, 0))
            lbl = ctk.CTkLabel(card, text="—", font=("Arial", 12, "bold"), text_color=GRAY_900)
            lbl.pack(pady=(0, 6))
            self._metric_labels[name] = lbl

    def _update_metrics(self, ber, papr, Nfft, Fs, useful_bw, n_symbols):
        ber_color = GREEN_600 if ber < 1e-4 else (AMBER_600 if ber < 1e-2 else RED_600)
        vals = {
            "BER":     (f"{ber:.2e}", ber_color),
            "PAPR":    (f"{papr:.2f} dB", GRAY_900),
            "Nfft":    (str(Nfft), GRAY_900),
            "Fs":      (f"{Fs/1e6:.2f} MHz", GRAY_900),
            "BW util": (f"{useful_bw/1e6:.2f} MHz", GRAY_900),
            "Simb.":   (str(n_symbols), GRAY_900),
        }
        for name, (txt, clr) in vals.items():
            self._metric_labels[name].configure(text=txt, text_color=clr)

    # ── figura matplotlib ────────────────────────────────
    def _build_plots(self):
        self.fig = Figure(figsize=(12, 17), dpi=95, facecolor=WHITE)

        self.ax_img_tx = self.fig.add_subplot(6, 2, 1)
        self.ax_img_rx = self.fig.add_subplot(6, 2, 2)
        self.ax1       = self.fig.add_subplot(6, 2, 3)
        self.ax2       = self.fig.add_subplot(6, 2, 4)
        self.ax3       = self.fig.add_subplot(6, 2, 5)
        self.ax4       = self.fig.add_subplot(6, 2, 6)
        self.ax5       = self.fig.add_subplot(6, 2, 7)
        self.ax6       = self.fig.add_subplot(6, 2, 8)
        self.ax7       = self.fig.add_subplot(6, 1, 5)
        self.ax8       = self.fig.add_subplot(6, 1, 6)

        self.fig.tight_layout(pad=3.0)
        self.fig.subplots_adjust(hspace=0.55, top=0.97, bottom=0.04)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)

    # ── panel timing ─────────────────────────────────────
    def _build_timing_panel(self):
        outer = ctk.CTkFrame(
            self.right_frame, fg_color=GRAY_50,
            corner_radius=10, border_width=1, border_color=GRAY_200,
        )
        outer.pack(fill="x", padx=12, pady=(0, 12))

        ctk.CTkLabel(
            outer, text="TIMING  &  PARAMETROS OFDM",
            font=("Arial", 9), text_color=GRAY_500,
        ).pack(anchor="w", padx=12, pady=(8, 4))

        self._timing_labels = {}
        grid = ctk.CTkFrame(outer, fg_color="transparent")
        grid.pack(fill="x", padx=8, pady=(0, 10))

        rows = [
            ("_hdr_time",  "TIEMPOS DE EJECUCION",  True),
            ("t_exec",     "Tiempo total",           False),
            ("t_mod",      "Modulacion",             False),
            ("t_ofdm_tx",  "OFDM TX",                False),
            ("t_channel",  "Canal + AWGN",           False),
            ("t_ofdm_rx",  "OFDM RX",                False),
            ("t_demod",    "Demodulacion",            False),
            ("_hdr_ofdm",  "PARAMETROS OFDM",        True),
            ("Nfft",       "Nfft",                   False),
            ("n_symbols",  "Simbolos OFDM",          False),
            ("tcp",        "Tcp  (prefijo ciclico)", False),
            ("t_useful",   "Tu   (tiempo util)",     False),
            ("symbol_dur", "Duracion simbolo",       False),
            ("fs",         "Fs   (frec. muestreo)",  False),
            ("delta_f",    "Delta-f subportadora",   False),
            ("n_subcarr",  "Subportadoras utiles",   False),
            ("throughput", "Throughput estimado",    False),
        ]

        row_idx = 0
        for key, label, is_header in rows:
            if is_header:
                if row_idx > 0:
                    ctk.CTkFrame(grid, height=1, fg_color=GRAY_200).grid(
                        row=row_idx, column=0, columnspan=2,
                        sticky="ew", padx=4, pady=(6, 2)
                    )
                    row_idx += 1
                ctk.CTkLabel(
                    grid, text=label,
                    font=("Arial", 9, "bold"), text_color=BLUE_600, anchor="w",
                ).grid(row=row_idx, column=0, columnspan=2, sticky="w", padx=8, pady=(2, 2))
                row_idx += 1
                continue

            ctk.CTkLabel(
                grid, text=label,
                font=("Arial", 10), text_color=GRAY_500, anchor="w",
            ).grid(row=row_idx, column=0, sticky="w", padx=(12, 4), pady=1)

            lbl = ctk.CTkLabel(
                grid, text="—",
                font=("Arial", 10, "bold"), text_color=GRAY_900, anchor="e",
            )
            lbl.grid(row=row_idx, column=1, sticky="e", padx=(4, 12), pady=1)
            self._timing_labels[key] = lbl
            row_idx += 1

        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)

    def _update_timing(self, t_exec, t_mod, t_ofdm_tx, t_channel, t_ofdm_rx,
                       t_demod, delta_f, Nfft, Fs, Tcp, n_symbols, bps):
        def _fmt(s):
            return f"{s*1e3:.2f} ms" if s >= 1e-3 else f"{s*1e6:.1f} us"

        Tu         = 1.0 / delta_f
        t_sym      = Tu + Tcp
        n_data_sc  = Nfft - Nfft // 8
        throughput = bps * n_data_sc / t_sym / 1e6

        vals = {
            "t_exec":     (_fmt(t_exec),             GRAY_900),
            "t_mod":      (_fmt(t_mod),              GRAY_500),
            "t_ofdm_tx":  (_fmt(t_ofdm_tx),          GRAY_500),
            "t_channel":  (_fmt(t_channel),          GRAY_500),
            "t_ofdm_rx":  (_fmt(t_ofdm_rx),          GRAY_500),
            "t_demod":    (_fmt(t_demod),            GRAY_500),
            "Nfft":       (str(Nfft),                GRAY_900),
            "n_symbols":  (str(n_symbols),           GRAY_900),
            "tcp":        (f"{Tcp*1e6:.2f} us",      GRAY_900),
            "t_useful":   (f"{Tu*1e6:.2f} us",       GRAY_900),
            "symbol_dur": (f"{t_sym*1e6:.2f} us",    GRAY_900),
            "fs":         (f"{Fs/1e6:.3f} MHz",      GRAY_900),
            "delta_f":    (f"{delta_f/1e3:.1f} kHz", GRAY_900),
            "n_subcarr":  (str(n_data_sc),           GRAY_900),
            "throughput": (f"{throughput:.2f} Mbit/s", GREEN_600),
        }
        for key, (txt, clr) in vals.items():
            self._timing_labels[key].configure(text=txt, text_color=clr)

    # ── helpers graficado ────────────────────────────────
    def _plot_power(self, ax, signal, title, samples=3000):
        power = np.abs(signal[:samples]) ** 2
        ax.clear()
        ax.plot(power, color=BLUE_600, linewidth=0.7)
        ax.axhline(np.mean(power), linestyle="--", color=GREEN_600, linewidth=1, label="Promedio")
        ax.axhline(np.max(power),  linestyle="--", color=RED_600,   linewidth=1, label="Pico")
        _style_ax(ax, title, "Muestras", "Potencia")
        ax.legend(fontsize=6, facecolor=AX_BG, labelcolor=TEXT_CLR, edgecolor=GRAY_300)

    def _get_ofdm_params(self):
        return get_ofdm_parameters(
            self.spacing.get(), self.bandwidth.get(), self.cp_type.get()
        )

    # ── cargar imagen ────────────────────────────────────
    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Imagenes", "*.png *.jpg *.jpeg")]
        )
        if path:
            self.image_path = path
            img = Image.open(path).resize((250, 250))
            self.tk_image = ctk.CTkImage(light_image=img, dark_image=img, size=(100, 100))
            self.image_label.configure(image=self.tk_image, text="")
            self.status.configure(text="Imagen cargada", text_color=BLUE_600)

    # ── ejecutar sistema ─────────────────────────────────
    def run_system(self):
        if not self.image_path:
            self.status.configure(text="Cargue una imagen primero", text_color=RED_600)
            return

        bits_tx, shape = image_to_bits(self.image_path)
        delta_f, bw, useful_bw, guard_bw, Nfft, Fs, Tcp, cp_len = self._get_ofdm_params()
        modulation    = self.modulation.get()
        snr_db        = float(self.snr.get())
        use_multipath = self.channel_type.get() == "Multipath"
        equalizer_on  = use_multipath and (self.equalization.get() == "ON")
        bps           = MODULATION_MAP[modulation]["bits_per_symbol"]

        t_start = time.perf_counter()

        t0 = time.perf_counter()
        bits_tx, symbols = modulate_bits(bits_tx, modulation)
        t_mod = time.perf_counter() - t0

        t0 = time.perf_counter()
        tx_signal, pil, dcl = ofdm_tx(symbols, Nfft, cp_len, pilot_spacing=8)
        t_ofdm_tx = time.perf_counter() - t0

        t0 = time.perf_counter()
        if use_multipath:
            ch_signal, h = multipath_channel(tx_signal)
            rx_signal = awgn(ch_signal, snr_db)
        else:
            rx_signal = awgn(tx_signal, snr_db)
            h = np.array([1 + 0j])
        t_channel = time.perf_counter() - t0

        t0 = time.perf_counter()
        rx_symbols = ofdm_rx(rx_signal, Nfft, cp_len, pil, dcl, snr_db, equalizer_on)
        t_ofdm_rx = time.perf_counter() - t0

        t0 = time.perf_counter()
        bits_rx = demodulate_symbols(rx_symbols, modulation)[: len(bits_tx)]
        t_demod = time.perf_counter() - t0

        t_exec = time.perf_counter() - t_start

        ber, bit_errors = calculate_ber(bits_tx, bits_rx)
        papr = calculate_papr(tx_signal)
        bits_to_image(bits_rx, shape)

        self._update_metrics(ber, papr, Nfft, Fs, useful_bw, len(dcl))
        self._update_timing(
            t_exec, t_mod, t_ofdm_tx, t_channel, t_ofdm_rx, t_demod,
            delta_f, Nfft, Fs, Tcp, len(dcl), bps,
        )

        # Imagenes TX / RX
        for ax, path, title in [
            (self.ax_img_tx, self.image_path, "Imagen TX"),
            (self.ax_img_rx, "imagen_rx.png",  f"Imagen RX  BER={ber:.2e}"),
        ]:
            ax.clear()
            ax.set_facecolor(WHITE)
            ax.imshow(Image.open(path).convert("RGB"))
            ax.set_title(title, color=GRAY_700, fontsize=8)
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_edgecolor(GRAY_200)

        # Constelaciones
        for ax, data, title in [
            (self.ax1, symbols,    "Constelacion TX"),
            (self.ax2, rx_symbols, "Constelacion RX"),
        ]:
            ax.clear()
            ax.scatter(np.real(data[:5000]), np.imag(data[:5000]),
                       s=3, color=BLUE_600, alpha=0.5)
            ax.axis("equal")
            _style_ax(ax, title)

        # Potencia
        self._plot_power(self.ax3, tx_signal, f"Potencia TX  PAPR={papr:.2f} dB")
        self._plot_power(self.ax4, rx_signal,  "Potencia RX")

        # Senal x(t)
        self.ax5.clear()
        self.ax5.plot(np.real(tx_signal[:2000]), color=BLUE_600,  linewidth=1.2, label="TX")
        self.ax5.plot(np.real(rx_signal[:2000]), color=AMBER_600, linewidth=0.8,
                      alpha=0.85, linestyle="--", label="RX")
        _style_ax(self.ax5, "Senal x(t)  TX vs RX", "Muestras", "Amplitud")
        self.ax5.legend(fontsize=6, facecolor=AX_BG, labelcolor=TEXT_CLR, edgecolor=GRAY_300)

        # Mapa subportadoras
        pilot_pos = pil[0]
        data_pos  = np.setdiff1d(np.arange(Nfft), pilot_pos)
        fv = np.zeros(Nfft)
        fv[pilot_pos] = 2
        fv[data_pos]  = 1
        self.ax6.clear()
        self.ax6.bar(data_pos,  fv[data_pos],  width=1.0, color=BLUE_400,  label="Datos")
        self.ax6.bar(pilot_pos, fv[pilot_pos], width=1.0, color=AMBER_600, label="Pilotos")
        self.ax6.set_ylim([0, 3])
        _style_ax(self.ax6, "Mapa de Subportadoras", "Indice", "Tipo")
        self.ax6.legend(fontsize=6, facecolor=AX_BG, labelcolor=TEXT_CLR, edgecolor=GRAY_300)

        # Respuesta frecuencia del canal
        H_mag     = 20 * np.log10(np.abs(np.fft.fft(h, Nfft)) + 1e-12)
        freq_axis = np.linspace(-Fs / 2, Fs / 2, Nfft) / 1e6
        self.ax7.clear()
        self.ax7.plot(freq_axis, np.fft.fftshift(H_mag), color=BLUE_600, linewidth=1.5)
        self.ax7.fill_between(freq_axis, np.fft.fftshift(H_mag), alpha=0.08, color=BLUE_400)
        _style_ax(self.ax7, "Respuesta en Frecuencia del Canal", "Frecuencia [MHz]", "Magnitud [dB]")

        self.canvas.draw()

        ber_color = GREEN_600 if ber < 1e-4 else (AMBER_600 if ber < 1e-2 else RED_600)
        self.status.configure(
            text=(
                f"BER           = {ber:.6f}\n"
                f"Bits erroneos = {bit_errors}\n"
                f"Tcp           = {Tcp*1e6:.2f} us\n"
                f"BW guarda     = {guard_bw/1e6:.2f} MHz\n"
                f"Tiempo total  = {t_exec*1e3:.1f} ms"
            ),
            text_color=ber_color,
        )

    # ── BER vs SNR ───────────────────────────────────────
    def generate_ber_plot(self):
        if not self.image_path:
            self.status.configure(text="Cargue una imagen primero", text_color=RED_600)
            return

        n_runs     = int(self.ber_runs.get())
        confidence = 0.95

        self.status.configure(
            text=f"Calculando BER vs SNR\n{n_runs} realizaciones / punto...",
            text_color=BLUE_600,
        )
        self.update()

        bits_original, _ = image_to_bits(self.image_path)
        _, _, _, _, Nfft, _, _, cp_len = self._get_ofdm_params()
        snr_range = np.arange(0, 22, 1)

        styles = {
            "QPSK":  ("o-", BLUE_600),
            "16QAM": ("s-", GREEN_600),
            "64QAM": ("^-", AMBER_600),
        }

        self.ax8.clear()
        for mod, (marker, color) in styles.items():
            mean, ci_low, ci_high = simulate_ber(
                bits_original, mod, Nfft, cp_len, snr_range,
                n_runs=n_runs, confidence=confidence,
            )
            self.ax8.semilogy(snr_range, mean, marker,
                              color=color, linewidth=1.5, markersize=5, label=mod)
            self.ax8.fill_between(snr_range, ci_low, ci_high,
                                  color=color, alpha=0.12, linewidth=0)
            self.ax8.errorbar(
                snr_range, mean,
                yerr=[mean - ci_low, ci_high - mean],
                fmt="none", ecolor=color, elinewidth=0.8,
                capsize=3, capthick=0.8, alpha=0.5,
            )

        _style_ax(
            self.ax8,
            f"BER vs SNR  {int(confidence*100)}%  ({n_runs} realizaciones / punto)",
            "SNR [dB]", "BER",
        )
        self.ax8.grid(True, which="both", color=GRID_CLR, linewidth=0.5)
        self.ax8.legend(fontsize=8, facecolor=AX_BG, labelcolor=TEXT_CLR, edgecolor=GRAY_300)
        self.canvas.draw()
        self.status.configure(
            text=f"BER vs SNR generada\nIC {int(confidence*100)}%  ·  {n_runs} runs/punto",
            text_color=BLUE_600,
        )


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    app = OFDMApp()
    app.mainloop()