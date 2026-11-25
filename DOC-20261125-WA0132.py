# -*- coding: utf-8 -*-
"""
=============================================================================
 PREDICTOR DE VIDA ÚTIL DE TANQUES DE ACERO (LEY DE FARADAY)
=============================================================================
 Universidad Técnica Estatal de Quevedo - Facultad de Ciencias de la Ingeniería
 Carrera de Ingeniería Mecánica

 Descripción:
   Software con interfaz gráfica (Tkinter + Matplotlib) que estima el desgaste
   por corrosión uniforme en el área húmeda de un tanque cilíndrico de acero
   y predice el año de falla estructural bajo tres escenarios cinéticos
   (Pasivado, Moderado y Severo), a partir de la densidad de corriente de
   corrosión (i_corr) aplicando la Ley de Faraday.

 Fundamento teórico:
   1) Ley de Faraday -> masa disuelta por unidad de área y tiempo:
          m / (A·t) = i_corr · M / (n · F)
   2) Dividiendo entre la densidad del acero se obtiene la velocidad de
      penetración (pérdida de espesor) de la corrosión uniforme:
          CR = i_corr · M / (n · F · ρ)            [cm/s]
      Para el hierro (M = 55.845 g/mol, n = 2, ρ = 7.86 g/cm³):
          CR ≈ 0.01162 · i_corr   [mm/año]  con i_corr en µA/cm²
   3) Espesor en función del tiempo (corrosión uniforme, cinética lineal):
          x(t) = x0 - CR · t
   4) Vida útil restante hasta alcanzar el espesor crítico:
          t_falla = (x0 - x_crit) / CR
   5) Área húmeda del tanque cilíndrico vertical (pared lateral mojada +
      fondo en contacto con el líquido):
          A_h = π·D·h + π·D²/4

 Requisitos:  Python 3.8+, numpy, matplotlib  (tkinter viene con Python)
 Ejecución:   python predictor_vida_util_tanques.py
=============================================================================
"""

import math
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import matplotlib
matplotlib.use("TkAgg")                       # Backend para incrustar en Tkinter
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)

# -----------------------------------------------------------------------------
# CONSTANTES FÍSICAS Y DEL MATERIAL (acero al carbono ≈ hierro)
# -----------------------------------------------------------------------------
F = 96485.0              # Constante de Faraday [C/mol]
M_FE = 55.845            # Masa molar del hierro [g/mol]
N_ELECTRONES = 2         # Fe -> Fe²⁺ + 2e⁻
RHO_ACERO = 7.86         # Densidad del acero [g/cm³]
SEG_POR_ANIO = 365.25 * 24 * 3600   # Segundos en un año [s]
T_MAX = 100              # Horizonte del gráfico [años]

# Escenarios cinéticos: nombre, color de la curva y valor por defecto de i_corr
ESCENARIOS = [
    ("Pasivado", "green",      0.5),
    ("Moderado", "darkorange", 5.0),
    ("Severo",   "red",       20.0),
]


# -----------------------------------------------------------------------------
# FUNCIONES DE CÁLCULO (independientes de la interfaz)
# -----------------------------------------------------------------------------
def area_humeda(D, h):
    """Área húmeda [m²] = pared lateral mojada (π·D·h) + fondo (π·D²/4)."""
    return math.pi * D * h + math.pi * D ** 2 / 4.0


def velocidad_corrosion(i_corr_uA_cm2):
    """
    Velocidad de corrosión uniforme [mm/año] por Ley de Faraday.
    i_corr se ingresa en µA/cm² y se convierte a A/cm².
    """
    i_corr = i_corr_uA_cm2 * 1e-6                          # [A/cm²]
    cr_cm_s = i_corr * M_FE / (N_ELECTRONES * F * RHO_ACERO)  # [cm/s]
    return cr_cm_s * 10.0 * SEG_POR_ANIO                   # [mm/año]


def vida_util(x0, x_crit, cr):
    """Años que tarda el espesor en bajar de x0 hasta x_crit."""
    return (x0 - x_crit) / cr


def perdida_masa_anual(cr_mm_anio, area_m2):
    """Masa de acero perdida por año en toda el área húmeda [kg/año]."""
    rho_kg_m3 = RHO_ACERO * 1000.0
    return (cr_mm_anio / 1000.0) * area_m2 * rho_kg_m3


# -----------------------------------------------------------------------------
# INTERFAZ GRÁFICA
# -----------------------------------------------------------------------------
class PredictorApp(tk.Tk):
    """Ventana principal: panel de entradas, tabla de resultados y gráfico."""

    def __init__(self):
        super().__init__()
        self.title("Predictor de Vida Útil de Tanques (Ley de Faraday) - UTEQ")
        self.geometry("1280x830")
        self.minsize(1100, 680)
        self.configure(bg="#f2f4f3")

        self._estilos()
        self._construir_panel_izquierdo()
        self._construir_panel_grafico()

    # ---------------- Estilos visuales ----------------
    def _estilos(self):
        st = ttk.Style(self)
        st.theme_use("clam")
        st.configure("TFrame", background="#f2f4f3")
        st.configure("TLabelframe", background="#f2f4f3")
        st.configure("TLabelframe.Label", background="#f2f4f3",
                     foreground="#1b5e20", font=("DejaVu Sans", 10, "bold"))
        st.configure("TLabel", background="#f2f4f3", font=("DejaVu Sans", 10))
        st.configure("Titulo.TLabel", font=("DejaVu Sans", 13, "bold"),
                     foreground="#1b5e20")
        st.configure("Res.TLabel", font=("DejaVu Sans", 11, "bold"),
                     foreground="#0d47a1")
        st.configure("Accion.TButton", font=("DejaVu Sans", 10, "bold"))
        st.configure("Treeview", font=("DejaVu Sans", 9), rowheight=24)
        st.configure("Treeview.Heading", font=("DejaVu Sans", 9, "bold"))

    # ---------------- Panel de entradas y salidas ----------------
    def _construir_panel_izquierdo(self):
        izq = ttk.Frame(self, padding=10)
        izq.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(izq, text="Predictor de Vida Útil de Tanques",
                  style="Titulo.TLabel").pack(anchor="w")
        ttk.Label(izq, text="Corrosión uniforme · Ley de Faraday (acero, Fe→Fe²⁺)"
                  ).pack(anchor="w", pady=(0, 8))

        # Diccionario que guarda todas las variables de entrada
        self.vars = {}

        # --- Geometría ---
        geo = ttk.LabelFrame(izq, text="1. Geometría del tanque", padding=8)
        geo.pack(fill=tk.X, pady=4)
        self._campo(geo, "D", "Diámetro D [m]:", 3.0, 0)
        self._campo(geo, "h", "Altura del líquido h [m]:", 4.0, 1)

        # --- Espesores ---
        esp = ttk.LabelFrame(izq, text="2. Espesores de pared", padding=8)
        esp.pack(fill=tk.X, pady=4)
        self._campo(esp, "x0", "Espesor inicial x₀ [mm]:", 8.0, 0)
        self._campo(esp, "xcrit", "Espesor crítico x_crit [mm]:", 3.0, 1)

        # --- Densidades de corriente por escenario ---
        cin = ttk.LabelFrame(izq, text="3. Densidad de corriente i_corr [µA/cm²]",
                             padding=8)
        cin.pack(fill=tk.X, pady=4)
        for fila, (nombre, color, defecto) in enumerate(ESCENARIOS):
            self._campo(cin, nombre, f"{nombre}:", defecto, fila, color)

        # --- Botones ---
        bot = ttk.Frame(izq)
        bot.pack(fill=tk.X, pady=8)
        ttk.Button(bot, text="CALCULAR Y GRAFICAR", style="Accion.TButton",
                   command=self.calcular).pack(side=tk.LEFT, expand=True,
                                               fill=tk.X, padx=(0, 4))
        ttk.Button(bot, text="Limpiar", command=self.limpiar
                   ).pack(side=tk.LEFT)

        # --- Salidas ---
        sal = ttk.LabelFrame(izq, text="4. Resultados", padding=8)
        sal.pack(fill=tk.BOTH, expand=True, pady=4)

        self.lbl_area = ttk.Label(sal, text="Área húmeda total: — m²",
                                  style="Res.TLabel")
        self.lbl_area.pack(anchor="w", pady=(0, 6))

        cols = ("esc", "icorr", "cr", "vida", "masa")
        self.tabla = ttk.Treeview(sal, columns=cols, show="headings", height=3)
        encabezados = ["Escenario", "i_corr", "CR [mm/a]", "Vida [años]",
                       "Pérdida [kg/a]"]
        anchos = [80, 58, 82, 86, 104]
        for c, t, w in zip(cols, encabezados, anchos):
            self.tabla.heading(c, text=t)
            self.tabla.column(c, width=w, anchor="center")
        self.tabla.pack(fill=tk.X)
        # Colores de fila según escenario
        for nombre, color, _ in ESCENARIOS:
            self.tabla.tag_configure(nombre, foreground=color)

        self.lbl_nota = ttk.Label(sal, text="", wraplength=430,
                                  foreground="#424242", justify="left")
        self.lbl_nota.pack(anchor="w", pady=(8, 0))

    def _campo(self, padre, clave, etiqueta, defecto, fila, color=None):
        """Crea una etiqueta + caja de texto y registra su variable."""
        lbl = ttk.Label(padre, text=etiqueta, width=26)
        if color:
            lbl.configure(foreground=color, font=("DejaVu Sans", 10, "bold"))
        lbl.grid(row=fila, column=0, sticky="w", pady=2)
        var = tk.StringVar(value=str(defecto))
        ttk.Entry(padre, textvariable=var, width=12, justify="right"
                  ).grid(row=fila, column=1, pady=2, padx=4)
        self.vars[clave] = var

    # ---------------- Panel del gráfico ----------------
    def _construir_panel_grafico(self):
        der = ttk.Frame(self, padding=(0, 10, 10, 10))
        der.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(7.6, 6.4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=der)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, der).update()
        self._grafico_vacio()

    def _grafico_vacio(self):
        self.ax.clear()
        self.ax.set_title("Espesor de pared vs. tiempo")
        self.ax.set_xlabel("Tiempo [años]")
        self.ax.set_ylabel("Espesor de pared [mm]")
        self.ax.set_xlim(0, T_MAX)
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()

    # ---------------- Lógica principal ----------------
    def _leer(self, clave, nombre):
        """Convierte el texto de una caja a float positivo (valida la entrada)."""
        try:
            valor = float(self.vars[clave].get().replace(",", "."))
        except ValueError:
            raise ValueError(f"'{nombre}' debe ser un número.")
        if valor <= 0:
            raise ValueError(f"'{nombre}' debe ser mayor que cero.")
        return valor

    def calcular(self):
        # 1) Lectura y validación de entradas
        try:
            D = self._leer("D", "Diámetro D")
            h = self._leer("h", "Altura h")
            x0 = self._leer("x0", "Espesor inicial x₀")
            xcrit = self._leer("xcrit", "Espesor crítico x_crit")
            icorr = {n: self._leer(n, f"i_corr {n}") for n, _, _ in ESCENARIOS}
            if xcrit >= x0:
                raise ValueError("El espesor crítico debe ser menor que el inicial.")
        except ValueError as e:
            messagebox.showerror("Dato inválido", str(e))
            return

        # 2) Área húmeda
        A = area_humeda(D, h)
        self.lbl_area.configure(text=f"Área húmeda total: {A:,.3f} m²")

        # 3) Cálculo por escenario
        self.tabla.delete(*self.tabla.get_children())
        resultados = []
        for nombre, color, _ in ESCENARIOS:
            cr = velocidad_corrosion(icorr[nombre])
            t_f = vida_util(x0, xcrit, cr)
            masa = perdida_masa_anual(cr, A)
            resultados.append((nombre, color, cr, t_f))
            self.tabla.insert("", tk.END, tags=(nombre,), values=(
                nombre, f"{icorr[nombre]:g}", f"{cr:.4f}", f"{t_f:,.1f}",
                f"{masa:,.2f}"))

        # 4) Nota interpretativa
        fuera = [n for n, _, _, t in resultados if t > T_MAX]
        nota = ("i_corr en µA/cm²; Pérdida = masa de acero disuelta por año "
                "en toda el área húmeda.\n"
                "Modelo: x(t) = x₀ − CR·t,  CR = i_corr·M/(n·F·ρ) "
                "≈ 0.01162·i_corr [mm/año].\n"
                "Margen de corrosión disponible: "
                f"{x0 - xcrit:.2f} mm.")
        if fuera:
            nota += ("\n⚠ " + ", ".join(fuera) +
                     f": la falla ocurre después de {T_MAX} años "
                     "(fuera del horizonte del gráfico).")
        self.lbl_nota.configure(text=nota)

        # 5) Gráfico
        self._graficar(x0, xcrit, resultados)

    def _graficar(self, x0, xcrit, resultados):
        ax = self.ax
        ax.clear()
        t = np.linspace(0, T_MAX, 1001)

        for nombre, color, cr, t_f in resultados:
            # Espesor remanente (no puede ser negativo)
            x = np.clip(x0 - cr * t, 0, None)
            etiqueta = (f"{nombre} (CR={cr:.3f} mm/año, falla en "
                        + (f"{t_f:.1f} años)" if t_f <= T_MAX
                           else f"> {T_MAX} años)"))
            ax.plot(t, x, color=color, linewidth=2.2, label=etiqueta)
            # Marcador "X" en el punto exacto de intersección con x_crit
            if t_f <= T_MAX:
                ax.plot(t_f, xcrit, marker="X", markersize=13, color=color,
                        markeredgecolor="black", zorder=5)
                ax.annotate(f"{t_f:.1f} años", (t_f, xcrit),
                            textcoords="offset points", xytext=(6, 10),
                            fontsize=9, color=color, fontweight="bold")

        # Línea horizontal negra punteada del espesor crítico
        ax.axhline(xcrit, color="black", linestyle=":", linewidth=2,
                   label=f"Espesor crítico x_crit = {xcrit:g} mm")

        ax.set_title("Desgaste por corrosión uniforme – Espesor de pared vs. tiempo",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Tiempo [años]")
        ax.set_ylabel("Espesor de pared [mm]")
        ax.set_xlim(0, T_MAX)
        ax.set_ylim(0, x0 * 1.08)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=8.5, framealpha=0.95)
        self.fig.tight_layout()
        self.canvas.draw()

    def limpiar(self):
        """Restablece valores por defecto y borra resultados."""
        defaults = {"D": 3.0, "h": 4.0, "x0": 8.0, "xcrit": 3.0}
        defaults.update({n: d for n, _, d in ESCENARIOS})
        for k, v in defaults.items():
            self.vars[k].set(str(v))
        self.tabla.delete(*self.tabla.get_children())
        self.lbl_area.configure(text="Área húmeda total: — m²")
        self.lbl_nota.configure(text="")
        self._grafico_vacio()


# -----------------------------------------------------------------------------
# PUNTO DE ENTRADA
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app = PredictorApp()
    app.mainloop()
