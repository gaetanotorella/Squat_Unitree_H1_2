#!/usr/bin/env python3
"""thesis_plot — grafici da CSV per la tesi, con anteprima, testo in LaTeX ed export vettoriale.

    python thesis_plot.py                       # finestra vuota
    python thesis_plot.py a.csv b.csv           # con dei CSV gia' caricati
    python thesis_plot.py --config figura.json  # riapre una figura salvata

La figura in anteprima e' esattamente quella che viene esportata: stessa dimensione fisica, stessi
font. Il rendering e l'export stanno in plotcore.py, che rigenera una figura salvata anche senza
interfaccia:  python plotcore.py figura.json figura.pdf
"""
from __future__ import annotations

import argparse
import copy
import os
import re
import sys
import tkinter as tk
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plotcore as pc  # noqa: E402

HISTORY = re.compile(r"_hm\d+$")


# ==================================================================================================
# piccoli widget di supporto
# ==================================================================================================
class Scrollable(ttk.Frame):
    """Frame con barra di scorrimento verticale (le schede sono piu' alte dello schermo)."""

    def __init__(self, master):
        super().__init__(master)
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        sb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.body = ttk.Frame(self.canvas, padding=8)
        self.body.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self._win = self.canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self._win, width=e.width))
        self.canvas.configure(yscrollcommand=sb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        for w in (self.canvas, self.body):
            w.bind("<Enter>", lambda e: self._wheel(True))
            w.bind("<Leave>", lambda e: self._wheel(False))

    def _wheel(self, on):
        if on:
            self.canvas.bind_all("<Button-4>", lambda e: self.canvas.yview_scroll(-2, "units"))
            self.canvas.bind_all("<Button-5>", lambda e: self.canvas.yview_scroll(2, "units"))
            self.canvas.bind_all("<MouseWheel>", lambda e: self.canvas.yview_scroll(-e.delta // 60, "units"))
        else:
            for ev in ("<Button-4>", "<Button-5>", "<MouseWheel>"):
                self.canvas.unbind_all(ev)


def section(parent, text):
    lf = ttk.LabelFrame(parent, text=text, padding=6)
    lf.pack(fill="x", pady=(0, 8))
    lf.columnconfigure(1, weight=1, minsize=210)
    return lf


def row(parent, r, label, widget, hint=None):
    ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", padx=(0, 6), pady=2)
    widget.grid(row=r, column=1, sticky="ew", pady=2)
    if hint:
        ttk.Label(parent, text=hint, foreground="#666").grid(row=r, column=2, sticky="w", padx=(4, 0))
    return widget


# ==================================================================================================
# applicazione
# ==================================================================================================
class App(tk.Tk):
    PREVIEW_ZOOMS = {"adatta": None, "75 %": 0.75, "100 %": 1.0, "125 %": 1.25, "150 %": 1.5, "200 %": 2.0}

    def __init__(self):
        super().__init__()
        self.title("thesis_plot — grafici per la tesi")
        self.geometry("1500x900")
        self.minsize(1100, 650)
        try:
            ttk.Style(self).theme_use("clam")
        except tk.TclError:
            pass

        self.cfg = pc.PlotConfig(usetex=pc.latex_available())
        self.store = pc.DataStore()
        self.config_path: str | None = None
        self._loading = False          # vero mentre le variabili vengono riempite dal modello
        self._render_job = None
        self._dragging = None          # (indice pannello, legenda) durante il trascinamento
        self._last_export: str | None = None

        self._build()
        self._refresh_all()
        self.bind_all("<Control-o>", lambda e: self.load_csv())
        self.bind_all("<Control-s>", lambda e: self.save_config())
        self.bind_all("<Control-e>", lambda e: self.export("pdf"))

    # ---------------------------------------------------------------------------------------------
    # struttura
    # ---------------------------------------------------------------------------------------------
    def _build(self):
        pw = ttk.PanedWindow(self, orient="horizontal")
        pw.pack(fill="both", expand=True)
        self.after(150, lambda: pw.sashpos(0, 600))
        left = ttk.Frame(pw, width=600)
        right = ttk.Frame(pw)
        pw.add(left, weight=0)
        pw.add(right, weight=1)

        nb = ttk.Notebook(left)
        nb.pack(fill="both", expand=True)
        self.tabs = {}
        for name in ("Dati", "Serie", "Assi", "Legenda", "Figura ed esporta"):
            sc = Scrollable(nb)
            nb.add(sc, text=name)
            self.tabs[name] = sc.body
        self._build_data(self.tabs["Dati"])
        self._build_series(self.tabs["Serie"])
        self._build_axes(self.tabs["Assi"])
        self._build_legend(self.tabs["Legenda"])
        self._build_figure(self.tabs["Figura ed esporta"])
        self._build_preview(right)

    # ---------------------------------------------------------------------------------------------
    # scheda DATI
    # ---------------------------------------------------------------------------------------------
    def _build_data(self, p):
        s = section(p, "File CSV caricati")
        bar = ttk.Frame(s); bar.grid(row=0, column=0, columnspan=3, sticky="ew")
        ttk.Button(bar, text="Carica CSV…  (Ctrl+O)", command=self.load_csv).pack(side="left")
        ttk.Button(bar, text="Rimuovi", command=self.remove_dataset).pack(side="left", padx=4)
        self.ds_tree = ttk.Treeview(s, columns=("alias", "file", "righe"), show="headings", height=4, selectmode="browse")
        for c, w in (("alias", 90), ("file", 230), ("righe", 60)):
            self.ds_tree.heading(c, text=c); self.ds_tree.column(c, width=w, stretch=c == "file")
        self.ds_tree.grid(row=1, column=0, columnspan=3, sticky="ew", pady=4)
        self.ds_tree.bind("<<TreeviewSelect>>", lambda e: self._on_dataset_selected())

        s = section(p, "Dataset selezionato")
        self.v_alias = tk.StringVar(); self.v_xcol = tk.StringVar(); self.v_xoff = tk.StringVar()
        self.v_xscale = tk.StringVar(); self.v_xzero = tk.BooleanVar()
        e = row(s, 0, "Nome (alias)", ttk.Entry(s, textvariable=self.v_alias), "es. reale, sim")
        e.bind("<Return>", lambda ev: self._apply_dataset()); e.bind("<FocusOut>", lambda ev: self._apply_dataset())
        self.cb_xcol = row(s, 1, "Colonna asse x", ttk.Combobox(s, textvariable=self.v_xcol, state="readonly"))
        self.cb_xcol.bind("<<ComboboxSelected>>", lambda ev: self._apply_dataset())
        for r, (lab, var, hint) in enumerate((("Offset x", self.v_xoff, "si somma (allineamento)"),
                                               ("Scala x", self.v_xscale, "es. 1000 per ms")), start=2):
            e = row(s, r, lab, ttk.Entry(s, textvariable=var), hint)
            e.bind("<Return>", lambda ev: self._apply_dataset()); e.bind("<FocusOut>", lambda ev: self._apply_dataset())
        ttk.Checkbutton(s, text="Fai partire x da 0", variable=self.v_xzero,
                        command=self._apply_dataset).grid(row=4, column=1, columnspan=2, sticky="w")

        s = section(p, "Grandezze calcolate")
        self.v_dpreset = tk.StringVar(); self.v_dname = tk.StringVar(); self.v_dexpr = tk.StringVar()
        self.v_djoint = tk.StringVar(value="left_knee"); self.v_dall = tk.BooleanVar(value=True)
        cb = row(s, 0, "Preimpostate", ttk.Combobox(s, textvariable=self.v_dpreset, state="readonly",
                                                    values=list(pc.DERIVED_PRESETS), height=32))
        cb.bind("<<ComboboxSelected>>", lambda e: self._preset_derived())
        cbj = row(s, 1, "Giunto", ttk.Combobox(s, textvariable=self.v_djoint, state="readonly",
                                              values=pc.joint_names(), height=28), "per le voci «Giunto ·»")
        cbj.bind("<<ComboboxSelected>>", lambda e: self._preset_derived())
        row(s, 2, "Nome", ttk.Entry(s, textvariable=self.v_dname))
        row(s, 3, "Espressione", ttk.Entry(s, textvariable=self.v_dexpr))
        ttk.Label(s, foreground="#666", wraplength=520, justify="left",
                  text=pc.RUN_FUNCTIONS_HELP + "  Numpy: sin cos arctan2 sqrt abs hypot gradient cumsum clip where; "
                       "col('nome') per nomi di colonna non validi.").grid(row=4, column=0, columnspan=3, sticky="w")
        ttk.Checkbutton(s, text="Aggiungila a tutti i dataset caricati (es. reale e sim)", variable=self.v_dall).grid(
            row=5, column=0, columnspan=3, sticky="w", pady=(4, 0))
        bar = ttk.Frame(s); bar.grid(row=6, column=0, columnspan=3, sticky="w", pady=(4, 0))
        ttk.Button(bar, text="Aggiungi", command=self.add_derived).pack(side="left")
        ttk.Button(bar, text="Aggiungi e metti nel grafico", command=lambda: self.add_derived(plot=True)).pack(side="left", padx=4)
        ttk.Button(bar, text="Rimuovi selezionata", command=self.remove_derived).pack(side="left")
        self.lb_derived = tk.Listbox(s, height=3, exportselection=False)
        self.lb_derived.grid(row=7, column=0, columnspan=3, sticky="ew", pady=4)

        s = section(p, "Colonne del dataset → serie del grafico")
        self.v_search = tk.StringVar(); self.v_hist = tk.BooleanVar(value=False); self.v_addpanel = tk.IntVar(value=1)
        e = row(s, 0, "Cerca", ttk.Entry(s, textvariable=self.v_search))
        self.v_search.trace_add("write", lambda *a: self._refresh_columns())
        ttk.Checkbutton(s, text="Mostra anche la storia delle osservazioni (_hm1…4)", variable=self.v_hist,
                        command=self._refresh_columns).grid(row=1, column=0, columnspan=3, sticky="w")
        fr = ttk.Frame(s); fr.grid(row=2, column=0, columnspan=3, sticky="nsew", pady=4)
        self.lb_cols = tk.Listbox(fr, selectmode="extended", height=14, exportselection=False)
        sb = ttk.Scrollbar(fr, orient="vertical", command=self.lb_cols.yview)
        self.lb_cols.configure(yscrollcommand=sb.set)
        self.lb_cols.pack(side="left", fill="both", expand=True); sb.pack(side="right", fill="y")
        self.lb_cols.bind("<Double-Button-1>", lambda e: self.add_series())
        bar = ttk.Frame(s); bar.grid(row=3, column=0, columnspan=3, sticky="w")
        ttk.Label(bar, text="nel pannello").pack(side="left")
        ttk.Spinbox(bar, from_=1, to=4, width=3, textvariable=self.v_addpanel).pack(side="left", padx=4)
        ttk.Button(bar, text="Aggiungi al grafico  →", command=self.add_series).pack(side="left", padx=4)
        ttk.Label(s, foreground="#666", text="Doppio clic su una colonna per aggiungerla; Ctrl/Shift per più colonne."
                  ).grid(row=4, column=0, columnspan=3, sticky="w")

    # ---------------------------------------------------------------------------------------------
    # scheda SERIE
    # ---------------------------------------------------------------------------------------------
    def _build_series(self, p):
        s = section(p, "Serie nel grafico (l'ordine è quello della legenda)")
        self.se_tree = ttk.Treeview(s, columns=("pan", "label", "src"), show="headings", height=8, selectmode="browse")
        for c, t, w in (("pan", "pann.", 45), ("label", "etichetta", 170), ("src", "dataset : colonna", 180)):
            self.se_tree.heading(c, text=t); self.se_tree.column(c, width=w, stretch=c != "pan")
        self.se_tree.grid(row=0, column=0, columnspan=3, sticky="ew")
        self.se_tree.bind("<<TreeviewSelect>>", lambda e: self._on_series_selected())
        bar = ttk.Frame(s); bar.grid(row=1, column=0, columnspan=3, sticky="w", pady=4)
        for t, cmd in (("↑", lambda: self.move_series(-1)), ("↓", lambda: self.move_series(1)),
                       ("Duplica", self.duplicate_series), ("Rimuovi", self.remove_series)):
            ttk.Button(bar, text=t, width=8 if len(t) > 1 else 3, command=cmd).pack(side="left", padx=(0, 4))

        s = section(p, "Serie selezionata")
        self.sv = {k: tk.StringVar() for k in ("label", "linestyle", "linewidth", "marker", "markevery",
                                                "drawstyle", "scale", "offset", "panel", "alpha")}
        self.sv_visible = tk.BooleanVar(value=True)
        self.sv_color = "#000000"
        e = row(s, 0, "Etichetta (LaTeX)", ttk.Entry(s, textvariable=self.sv["label"]))
        ttk.Label(s, foreground="#666", text=r"es.  $\theta_\mathrm{reale}$   oppure   reale").grid(
            row=1, column=1, sticky="w")
        fr = ttk.Frame(s); fr.grid(row=2, column=1, sticky="w", pady=2)
        ttk.Label(s, text="Colore").grid(row=2, column=0, sticky="w")
        self.color_swatch = tk.Label(fr, width=4, relief="solid", borderwidth=1, bg=self.sv_color)
        self.color_swatch.pack(side="left")
        ttk.Button(fr, text="Scegli…", command=self.pick_color).pack(side="left", padx=4)
        self.color_hex = ttk.Label(fr, text=self.sv_color); self.color_hex.pack(side="left")
        row(s, 3, "Tratto", ttk.Combobox(s, textvariable=self.sv["linestyle"], state="readonly", values=list(pc.LINESTYLES)))
        row(s, 4, "Spessore [pt]", ttk.Spinbox(s, from_=0.3, to=4, increment=0.1, textvariable=self.sv["linewidth"]))
        row(s, 5, "Marcatore", ttk.Combobox(s, textvariable=self.sv["marker"], state="readonly", values=list(pc.MARKERS)))
        row(s, 6, "…uno ogni N punti", ttk.Spinbox(s, from_=1, to=10000, increment=10, textvariable=self.sv["markevery"]))
        row(s, 7, "Disegno", ttk.Combobox(s, textvariable=self.sv["drawstyle"], state="readonly", values=list(pc.DRAWSTYLES)),
            "gradini per i comandi")
        row(s, 8, "Scala y", ttk.Entry(s, textvariable=self.sv["scale"]), "57.2958 = rad→°")
        row(s, 9, "Offset y", ttk.Entry(s, textvariable=self.sv["offset"]))
        row(s, 10, "Pannello", ttk.Spinbox(s, from_=1, to=4, textvariable=self.sv["panel"]))
        row(s, 11, "Opacità", ttk.Spinbox(s, from_=0.1, to=1.0, increment=0.1, textvariable=self.sv["alpha"]))
        ttk.Checkbutton(s, text="Visibile", variable=self.sv_visible, command=self._apply_series).grid(row=12, column=1, sticky="w")
        for v in self.sv.values():
            v.trace_add("write", lambda *a: self._apply_series())

        s = section(p, "Colori di tutte le serie")
        bar = ttk.Frame(s); bar.grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Button(bar, text="Riapplica la palette a tutte le serie", command=self.apply_palette).pack(side="left")

    # ---------------------------------------------------------------------------------------------
    # scheda ASSI
    # ---------------------------------------------------------------------------------------------
    def _build_axes(self, p):
        s = section(p, "Asse x (comune a tutti i pannelli)")
        self.av = {k: tk.StringVar() for k in ("xlabel", "xmin", "xmax", "title", "npanels")}
        row(s, 0, "Etichetta x", ttk.Entry(s, textvariable=self.av["xlabel"]))
        ttk.Label(s, foreground="#666", text=r"es.  $t$ [\si{\second}]").grid(row=1, column=1, sticky="w")
        row(s, 2, "x min", ttk.Entry(s, textvariable=self.av["xmin"]), "vuoto = automatico")
        row(s, 3, "x max", ttk.Entry(s, textvariable=self.av["xmax"]))
        row(s, 4, "Titolo", ttk.Entry(s, textvariable=self.av["title"]), "di solito vuoto: va nella caption")

        s = section(p, "Pannelli (sovrapposti, asse x condiviso)")
        row(s, 0, "Numero di pannelli", ttk.Spinbox(s, from_=1, to=4, textvariable=self.av["npanels"], width=5))
        self.v_panel_sel = tk.StringVar(value="Pannello 1")
        self.cb_panel = row(s, 1, "Modifica", ttk.Combobox(s, textvariable=self.v_panel_sel, state="readonly"))
        self.cb_panel.bind("<<ComboboxSelected>>", lambda e: self._load_panel())
        self.pv = {k: tk.StringVar() for k in ("ylabel", "ymin", "ymax", "height_ratio", "hlines", "vspans")}
        self.pv_log = tk.BooleanVar()
        row(s, 2, "Etichetta y", ttk.Entry(s, textvariable=self.pv["ylabel"]))
        ttk.Label(s, foreground="#666", text=r"es.  $\theta$ [\si{\degree}]    $h$ [\si{\meter}]").grid(row=3, column=1, sticky="w")
        row(s, 4, "y min", ttk.Entry(s, textvariable=self.pv["ymin"]), "vuoto = automatico")
        row(s, 5, "y max", ttk.Entry(s, textvariable=self.pv["ymax"]))
        row(s, 6, "Altezza relativa", ttk.Entry(s, textvariable=self.pv["height_ratio"]), "es. 2 = doppia")
        ttk.Checkbutton(s, text="Scala logaritmica", variable=self.pv_log, command=self._apply_panel).grid(row=7, column=1, sticky="w")
        row(s, 8, "Linee orizzontali", ttk.Entry(s, textvariable=self.pv["hlines"]), "0.98:nominale, 0.7")
        row(s, 9, "Bande verticali", ttk.Entry(s, textvariable=self.pv["vspans"]), "40-56:squat, 160-170")
        for v in self.pv.values():
            v.trace_add("write", lambda *a: self._apply_panel())
        for v in self.av.values():
            v.trace_add("write", lambda *a: self._apply_axes())

        s = section(p, "Tacche e griglia")
        self.bv = {k: tk.BooleanVar() for k in ("grid", "grid_minor", "ticks_in", "ticks_all_sides", "minor_ticks")}
        for r, (k, t) in enumerate((("grid", "Griglia principale"), ("grid_minor", "Griglia secondaria"),
                                    ("minor_ticks", "Tacche secondarie"), ("ticks_in", "Tacche verso l'interno"),
                                    ("ticks_all_sides", "Tacche su tutti e quattro i lati"))):
            ttk.Checkbutton(s, text=t, variable=self.bv[k], command=self._apply_axes).grid(row=r, column=0, columnspan=2, sticky="w")

        s = section(p, "Zoom")
        ttk.Button(s, text="Usa i limiti dello zoom dell'anteprima", command=self.capture_limits).grid(row=0, column=0, sticky="w")
        ttk.Button(s, text="Limiti automatici", command=self.reset_limits).grid(row=0, column=1, sticky="w", padx=4)
        ttk.Label(s, foreground="#666", wraplength=400, justify="left",
                  text="Ingrandisci con la lente della barra sotto l'anteprima, poi copia qui i limiti: "
                       "così finiscono anche nel file esportato.").grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 0))

    # ---------------------------------------------------------------------------------------------
    # scheda LEGENDA
    # ---------------------------------------------------------------------------------------------
    def _build_legend(self, p):
        s = section(p, "Legenda del pannello")
        self.v_lpanel = tk.StringVar(value="Pannello 1")
        self.cb_lpanel = row(s, 0, "Pannello", ttk.Combobox(s, textvariable=self.v_lpanel, state="readonly"))
        self.cb_lpanel.bind("<<ComboboxSelected>>", lambda e: self._load_legend())
        self.lv = {k: tk.StringVar() for k in ("legend", "legend_ncol", "legend_title", "legend_x", "legend_y")}
        row(s, 1, "Posizione", ttk.Combobox(s, textvariable=self.lv["legend"], state="readonly", values=list(pc.LEGEND_LOCS)))
        row(s, 2, "Colonne", ttk.Spinbox(s, from_=1, to=8, textvariable=self.lv["legend_ncol"], width=5))
        row(s, 3, "Titolo", ttk.Entry(s, textvariable=self.lv["legend_title"]))
        row(s, 4, "x (personalizzata)", ttk.Entry(s, textvariable=self.lv["legend_x"]), "0–1, angolo in basso a sx")
        row(s, 5, "y (personalizzata)", ttk.Entry(s, textvariable=self.lv["legend_y"]))
        for v in self.lv.values():
            v.trace_add("write", lambda *a: self._apply_legend())
        ttk.Label(s, foreground="#1a5", wraplength=400, justify="left",
                  text="Puoi TRASCINARE la legenda direttamente nell'anteprima: la posizione viene "
                       "memorizzata (diventa \"personalizzata\") e usata nell'export.").grid(
            row=6, column=0, columnspan=3, sticky="w", pady=(6, 0))

        s = section(p, "Aspetto (tutte le legende)")
        self.v_lframe = tk.BooleanVar(); self.v_lfont = tk.StringVar()
        ttk.Checkbutton(s, text="Cornice", variable=self.v_lframe, command=self._apply_figure).grid(row=0, column=1, sticky="w")
        row(s, 1, "Dimensione font [pt]", ttk.Entry(s, textvariable=self.v_lfont), "vuoto = testo − 1")
        self.v_lfont.trace_add("write", lambda *a: self._apply_figure())

    # ---------------------------------------------------------------------------------------------
    # scheda FIGURA ED ESPORTA
    # ---------------------------------------------------------------------------------------------
    def _build_figure(self, p):
        s = section(p, "Dimensione (quella reale nel documento)")
        self.fv = {k: tk.StringVar() for k in ("preset", "width_cm", "height_cm", "font_size", "font", "palette")}
        cb = row(s, 0, "Formato", ttk.Combobox(s, textvariable=self.fv["preset"], state="readonly",
                                              values=list(pc.SIZE_PRESETS), width=38))
        cb.bind("<<ComboboxSelected>>", lambda e: self._apply_preset())
        row(s, 1, "Larghezza [cm]", ttk.Entry(s, textvariable=self.fv["width_cm"]))
        row(s, 2, "Altezza [cm]", ttk.Entry(s, textvariable=self.fv["height_cm"]))
        ttk.Label(s, foreground="#666", wraplength=400, justify="left",
                  text="La figura viene esportata esattamente a questa dimensione: in LaTeX includila "
                       "SENZA width=…, così i font restano alla dimensione del testo.").grid(
            row=3, column=0, columnspan=3, sticky="w")

        s = section(p, "Testo")
        row(s, 0, "Dimensione [pt]", ttk.Spinbox(s, from_=6, to=14, increment=0.5, textvariable=self.fv["font_size"]),
            "come il corpo del testo")
        row(s, 1, "Carattere", ttk.Combobox(s, textvariable=self.fv["font"], state="readonly", values=list(pc.FONTS)))
        self.v_usetex = tk.BooleanVar()
        cbx = ttk.Checkbutton(s, text="Testo composto da LaTeX (pdflatex)", variable=self.v_usetex, command=self._apply_figure)
        cbx.grid(row=2, column=1, sticky="w")
        if not pc.latex_available():
            cbx.state(["disabled"])
            ttk.Label(s, foreground="#a33", text="LaTeX non trovato").grid(row=2, column=2)
        ttk.Label(s, foreground="#666", wraplength=400, justify="left",
                  text="Con LaTeX valgono \\si{} di siunitx, amsmath e tutto il resto; senza, si usa il motore "
                       "interno di matplotlib (più rapido, unità di \\si tradotte).").grid(row=3, column=0, columnspan=3, sticky="w")

        s = section(p, "Colori")
        row(s, 0, "Palette", ttk.Combobox(s, textvariable=self.fv["palette"], state="readonly", values=list(pc.PALETTES)))
        for v in self.fv.values():
            v.trace_add("write", lambda *a: self._apply_figure())

        s = section(p, "Esporta")
        self.v_svgpath = tk.BooleanVar(); self.v_transp = tk.BooleanVar()
        ttk.Checkbutton(s, text="SVG: testo convertito in tracciati (resa identica ovunque)",
                        variable=self.v_svgpath, command=self._apply_figure).grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Checkbutton(s, text="Sfondo trasparente", variable=self.v_transp,
                        command=self._apply_figure).grid(row=1, column=0, columnspan=3, sticky="w")
        bar = ttk.Frame(s); bar.grid(row=2, column=0, columnspan=3, sticky="w", pady=6)
        ttk.Button(bar, text="PDF…  (Ctrl+E)", command=lambda: self.export("pdf")).pack(side="left")
        ttk.Button(bar, text="SVG…", command=lambda: self.export("svg")).pack(side="left", padx=4)
        ttk.Button(bar, text="PNG 300 dpi…", command=lambda: self.export("png")).pack(side="left")
        ttk.Label(s, text="Codice LaTeX per includerla:").grid(row=3, column=0, columnspan=3, sticky="w")
        self.txt_latex = tk.Text(s, height=7, width=52, font=("TkFixedFont", 9), relief="solid", borderwidth=1)
        self.txt_latex.grid(row=4, column=0, columnspan=3, sticky="ew")
        ttk.Button(s, text="Copia", command=self.copy_latex).grid(row=5, column=0, sticky="w", pady=4)

        s = section(p, "Configurazione della figura")
        bar = ttk.Frame(s); bar.grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Button(bar, text="Salva…  (Ctrl+S)", command=self.save_config).pack(side="left")
        ttk.Button(bar, text="Apri…", command=self.open_config).pack(side="left", padx=4)
        ttk.Label(s, foreground="#666", wraplength=400, justify="left",
                  text="Il file .json contiene dati, serie, stile e posizione della legenda: la figura si "
                       "riapre identica, o si rigenera senza interfaccia con  python plotcore.py figura.json "
                       "figura.pdf").grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 0))

    # ---------------------------------------------------------------------------------------------
    # ANTEPRIMA
    # ---------------------------------------------------------------------------------------------
    def _build_preview(self, p):
        top = ttk.Frame(p, padding=(8, 6)); top.pack(fill="x")
        ttk.Label(top, text="Anteprima").pack(side="left")
        self.v_zoom = tk.StringVar(value="adatta")
        cb = ttk.Combobox(top, textvariable=self.v_zoom, values=list(self.PREVIEW_ZOOMS), width=7, state="readonly")
        cb.pack(side="left", padx=6); cb.bind("<<ComboboxSelected>>", lambda e: self.schedule_render(0))
        self.v_fast = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="anteprima rapida (senza LaTeX)", variable=self.v_fast,
                        command=lambda: self.schedule_render(0)).pack(side="left", padx=6)
        ttk.Button(top, text="Aggiorna", command=lambda: self.schedule_render(0)).pack(side="left")
        self.lbl_status = ttk.Label(top, text="", foreground="#555")
        self.lbl_status.pack(side="left", padx=10)

        holder = ttk.Frame(p); holder.pack(fill="both", expand=True)
        self.pv_canvas = tk.Canvas(holder, bg="#8a8f96", highlightthickness=0)
        hs = ttk.Scrollbar(holder, orient="horizontal", command=self.pv_canvas.xview)
        vs = ttk.Scrollbar(holder, orient="vertical", command=self.pv_canvas.yview)
        self.pv_canvas.configure(xscrollcommand=hs.set, yscrollcommand=vs.set)
        vs.pack(side="right", fill="y"); hs.pack(side="bottom", fill="x")
        self.pv_canvas.pack(side="left", fill="both", expand=True)

        self.fig = Figure(figsize=(6, 3), dpi=100, layout="constrained")
        self.mpl = FigureCanvasTkAgg(self.fig, master=self.pv_canvas)
        self.mpl_widget = self.mpl.get_tk_widget()
        self._fig_win = self.pv_canvas.create_window(20, 20, window=self.mpl_widget, anchor="nw")
        self.pv_canvas.bind("<Configure>", self._on_preview_resize)

        tb_frame = ttk.Frame(p); tb_frame.pack(fill="x")
        self.toolbar = NavigationToolbar2Tk(self.mpl, tb_frame, pack_toolbar=False)
        self.toolbar.update(); self.toolbar.pack(side="left", fill="x")

        self.mpl.mpl_connect("button_press_event", self._on_press)
        self.mpl.mpl_connect("button_release_event", self._on_release)

    def _on_preview_resize(self, event):
        self._center_preview()
        if self.v_zoom.get() == "adatta":
            self.schedule_render(400)

    def _center_preview(self):
        self.update_idletasks()
        cw, ch = self.pv_canvas.winfo_width(), self.pv_canvas.winfo_height()
        fw, fh = self.mpl_widget.winfo_reqwidth(), self.mpl_widget.winfo_reqheight()
        x, y = max(20, (cw - fw) // 2), max(20, (ch - fh) // 2)
        self.pv_canvas.coords(self._fig_win, x, y)
        self.pv_canvas.configure(scrollregion=(0, 0, max(cw, fw + 40), max(ch, fh + 40)))

    # ==============================================================================================
    # rendering
    # ==============================================================================================
    def schedule_render(self, delay=350):
        if self._render_job:
            self.after_cancel(self._render_job)
        self._render_job = self.after(delay, self._render)

    def _render(self):
        self._render_job = None
        cfg = self.cfg
        if self.v_fast.get() and cfg.usetex:
            cfg = copy.deepcopy(cfg); cfg.usetex = False
        dpi = self._preview_dpi(cfg)
        self.lbl_status.configure(text="rendering…" + (" (LaTeX)" if cfg.usetex else ""), foreground="#555")
        self.config(cursor="watch"); self.update_idletasks()
        try:
            res = pc.render(cfg, self.store, fig=self.fig, dpi=dpi)
            w = int(round(cfg.width_cm / 2.54 * dpi)); h = int(round(cfg.height_cm / 2.54 * dpi))
            self.mpl_widget.configure(width=w, height=h)
            self.fig.set_size_inches(cfg.width_cm / 2.54, cfg.height_cm / 2.54, forward=False)
            self.mpl.draw()
            self.toolbar.update()       # nuova "home" per lo zoom
            self._legends = res.legends
            for leg in res.legends:
                if leg is not None:
                    leg.set_draggable(True)
            msg = f"{cfg.width_cm:g} × {cfg.height_cm:g} cm · {len([s for s in self.cfg.series if s.visible])} serie"
            if res.warnings:
                self.lbl_status.configure(text="⚠ " + " | ".join(res.warnings)[:220], foreground="#b35c00")
            else:
                self.lbl_status.configure(text=msg, foreground="#2a6")
        except Exception as e:
            msg = self._short_error(e)
            if "latex" in str(e).lower():
                culprit = pc.find_latex_culprit(cfg)
                if culprit:
                    msg = "errore LaTeX nella " + culprit
            self.lbl_status.configure(text="✗ " + msg[:240], foreground="#c00")
        finally:
            self.config(cursor="")
            self._center_preview()

    def _preview_dpi(self, cfg):
        z = self.PREVIEW_ZOOMS.get(self.v_zoom.get(), None)
        if z is not None:
            return 96 * z
        # "adatta": la figura riempie l'area di anteprima mantenendo le proporzioni
        cw, ch = self.pv_canvas.winfo_width(), self.pv_canvas.winfo_height()
        if cw < 50 or ch < 50:
            return 120
        return max(40.0, min((cw - 40) / (cfg.width_cm / 2.54), (ch - 40) / (cfg.height_cm / 2.54)))

    @staticmethod
    def _short_error(e: Exception) -> str:
        t = str(e)
        if "latex" in t.lower():
            # dall'output di LaTeX la riga utile e' quella che comincia con "!"
            bang = [l for l in t.splitlines() if l.startswith("!")]
            return "errore LaTeX in un'etichetta: " + (bang[0] if bang else t.splitlines()[0])[:180]
        return t.splitlines()[0][:200] if t else type(e).__name__

    # -- legenda trascinabile --------------------------------------------------------------------
    def _on_press(self, event):
        self._dragging = None
        if self.toolbar.mode:          # lente o mano attive: il clic e' per lo zoom, non per la legenda
            return
        for i, leg in enumerate(getattr(self, "_legends", [])):
            if leg is not None and leg.contains(event)[0]:
                self._dragging = (i, leg)

    def _on_release(self, event):
        if not self._dragging:
            return
        i, leg = self._dragging
        self._dragging = None
        ax = leg.axes
        bb = leg.get_window_extent()
        x, y = ax.transAxes.inverted().transform((bb.x0, bb.y0))
        p = self.cfg.panels[i]
        p.legend = "personalizzata (trascina)"
        p.legend_x, p.legend_y = round(float(x), 3), round(float(y), 3)
        if self._panel_index(self.v_lpanel.get()) == i:
            self._load_legend()
        self.schedule_render(50)

    # ==============================================================================================
    # DATI
    # ==============================================================================================
    def load_csv(self, paths=None):
        if paths is None:
            paths = filedialog.askopenfilenames(title="Carica CSV", filetypes=[("CSV", "*.csv"), ("Tutti", "*")],
                                                initialdir=self._initial_dir())
        for path in paths:
            try:
                df = self.store.load(path)
            except Exception as e:
                messagebox.showerror("CSV", f"Impossibile leggere {path}:\n{e}"); continue
            alias = self._unique_alias(self._guess_alias(path))
            self.cfg.datasets.append(pc.Dataset(path=str(Path(path).resolve()), alias=alias,
                                                xcol=pc.DataStore.default_xcol(df)))
        self._refresh_datasets(select=len(self.cfg.datasets) - 1)
        self.schedule_render()

    def _initial_dir(self):
        for d in (Path.home() / "unitree_ws" / "sim_logs", Path.home()):
            if d.exists():
                return str(d)
        return "."

    @staticmethod
    def _guess_alias(path):
        parent = Path(path).parent.name
        if parent.startswith("exp_"):
            return "reale"
        if "replay" in parent or parent.startswith("sim_"):
            return "sim"
        return Path(path).stem

    def _unique_alias(self, base):
        names = {d.alias for d in self.cfg.datasets}
        if base not in names:
            return base
        k = 2
        while f"{base}{k}" in names:
            k += 1
        return f"{base}{k}"

    def remove_dataset(self):
        i = self._sel_index(self.ds_tree)
        if i is None:
            return
        alias = self.cfg.datasets[i].alias
        n = sum(1 for s in self.cfg.series if s.dataset == alias)
        if n and not messagebox.askyesno("Rimuovi", f"Rimuovere '{alias}' e le sue {n} serie?"):
            return
        del self.cfg.datasets[i]
        self.cfg.series = [s for s in self.cfg.series if s.dataset != alias]
        self._refresh_all()

    def _current_dataset(self):
        i = self._sel_index(self.ds_tree)
        return None if i is None else self.cfg.datasets[i]

    def _on_dataset_selected(self):
        ds = self._current_dataset()
        self._loading = True
        if ds:
            df = self.store.frame(ds)
            self.cb_xcol.configure(values=["__indice__"] + list(df.columns))
            self.v_alias.set(ds.alias); self.v_xcol.set(ds.xcol or pc.DataStore.default_xcol(df))
            self.v_xoff.set(f"{ds.x_offset:g}"); self.v_xscale.set(f"{ds.x_scale:g}"); self.v_xzero.set(ds.x_from_zero)
            self.lb_derived.delete(0, "end")
            for k, v in ds.derived.items():
                self.lb_derived.insert("end", f"{k} = {v}")
        self._loading = False
        self._refresh_columns()

    def _apply_dataset(self):
        ds = self._current_dataset()
        if ds is None or self._loading:
            return
        new_alias = self.v_alias.get().strip() or ds.alias
        if new_alias != ds.alias and new_alias in {d.alias for d in self.cfg.datasets}:
            messagebox.showwarning("Alias", f"'{new_alias}' è già usato"); self.v_alias.set(ds.alias); return
        for s in self.cfg.series:
            if s.dataset == ds.alias:
                s.dataset = new_alias
        ds.alias = new_alias
        ds.xcol = self.v_xcol.get()
        ds.x_offset = self._float(self.v_xoff.get(), 0.0)
        ds.x_scale = self._float(self.v_xscale.get(), 1.0)
        ds.x_from_zero = self.v_xzero.get()
        i = self.cfg.datasets.index(ds)
        self._refresh_datasets(select=i, keep_form=True)
        self._refresh_series()
        self.schedule_render()

    def _preset_derived(self):
        key = self.v_dpreset.get()
        if key not in pc.DERIVED_PRESETS:
            return
        name, expr, _, _ = pc.preset(key, self.v_djoint.get() or "left_knee")
        self.v_dname.set(name); self.v_dexpr.set(expr)

    def add_derived(self, plot=False):
        cur = self._current_dataset()
        if cur is None:
            messagebox.showinfo("Grandezza calcolata", "Seleziona prima un dataset."); return
        name, expr = self.v_dname.get().strip(), self.v_dexpr.get().strip()
        if not name.isidentifier():
            messagebox.showwarning("Grandezza calcolata", "Il nome deve essere un identificatore (lettere, cifre, _)."); return
        targets = list(self.cfg.datasets) if self.v_dall.get() else [cur]
        ok, failed = [], []
        self.config(cursor="watch"); self.update_idletasks()
        for ds in targets:
            try:
                pc.eval_expression(self.store.load(ds.path), expr, self.store.context(ds.path).functions())
                ds.derived[name] = expr
                ok.append(ds)
            except Exception as e:
                failed.append(f"{ds.alias}: {str(e).splitlines()[0][:160]}")
        self.config(cursor="")
        if failed:
            (messagebox.showwarning if ok else messagebox.showerror)(
                "Grandezza calcolata", ("Aggiunta a: " + ", ".join(d.alias for d in ok) + "\n\n" if ok else "")
                + "Non calcolabile su:\n" + "\n".join(failed))
        if plot and ok:
            pal = pc.PALETTES.get(self.cfg.palette, pc.PALETTES["Okabe-Ito (daltonici)"])
            panel = max(1, min(4, self._int(self.v_addpanel.get(), 1))) - 1
            if panel >= len(self.cfg.panels):
                self._set_npanels(panel + 1)
            label, ylabel = pc.labels_for(name)
            if ylabel and not self.cfg.panels[panel].ylabel:
                self.cfg.panels[panel].ylabel = ylabel
                self._load_panel()
            for ds in ok:
                if len(ok) > 1:
                    label = pc.latex_escape(ds.alias)
                self.cfg.series.append(pc.Series(dataset=ds.alias, column=name, label=label,
                                                 color=self._auto_color(ds.alias, panel), panel=panel,
                                                 linestyle="continua" if ds is ok[0] else "tratteggiata"))
            self._refresh_series(select=len(self.cfg.series) - 1)
        self._on_dataset_selected()
        self.schedule_render()

    def remove_derived(self):
        ds = self._current_dataset()
        sel = self.lb_derived.curselection()
        if ds is None or not sel:
            return
        name = self.lb_derived.get(sel[0]).split(" = ", 1)[0]
        ds.derived.pop(name, None)
        self._on_dataset_selected()
        self.schedule_render()

    def _refresh_columns(self):
        self.lb_cols.delete(0, "end")
        ds = self._current_dataset()
        if ds is None:
            return
        try:
            df = self.store.frame(ds)
        except Exception as e:
            self.lb_cols.insert("end", f"(errore: {e})"); return
        q = self.v_search.get().strip().lower()
        derived = list(ds.derived)
        for c in derived + [c for c in df.columns if c not in ds.derived]:
            if not self.v_hist.get() and HISTORY.search(str(c)):
                continue
            if q and q not in str(c).lower():
                continue
            self.lb_cols.insert("end", ("ƒ  " if c in ds.derived else "") + str(c))

    def _auto_color(self, alias, panel):
        """Stesso dataset -> stesso colore in tutti i pannelli (reale blu, sim arancio...), cosi' il
        confronto si legge anche dove la legenda non c'e'. Dentro uno stesso pannello, se il dataset
        ha gia' una serie, si passa al colore successivo libero."""
        pal = pc.PALETTES.get(self.cfg.palette, pc.PALETTES["Okabe-Ito (daltonici)"])
        used_here = {s.color for s in self.cfg.series if s.panel == panel}
        others = {s.color for s in self.cfg.series if s.dataset != alias}
        for c in [s.color for s in self.cfg.series if s.dataset == alias]:   # il colore del dataset
            if c not in used_here:
                return c
        for c in pal:                                                         # un colore libero
            if c not in used_here and c not in others:
                return c
        return pal[len(self.cfg.series) % len(pal)]

    def add_series(self):
        ds = self._current_dataset()
        if ds is None:
            messagebox.showinfo("Serie", "Seleziona prima un dataset."); return
        cols = [self.lb_cols.get(i).replace("ƒ  ", "") for i in self.lb_cols.curselection()]
        if not cols:
            return
        pal = pc.PALETTES.get(self.cfg.palette, pc.PALETTES["Okabe-Ito (daltonici)"])
        panel = max(1, min(4, self._int(self.v_addpanel.get(), 1))) - 1
        if panel >= len(self.cfg.panels):
            self._set_npanels(panel + 1)
        multi = len(self.cfg.datasets) > 1
        for c in cols:
            label, ylabel = pc.labels_for(c)
            if multi:
                label = f"{pc.latex_escape(ds.alias)}: {label}"
            if ylabel and not self.cfg.panels[panel].ylabel:
                self.cfg.panels[panel].ylabel = ylabel
                self._load_panel()
            self.cfg.series.append(pc.Series(dataset=ds.alias, column=c, label=label,
                                             color=self._auto_color(ds.alias, panel), panel=panel))
        self._refresh_series(select=len(self.cfg.series) - 1)
        self.schedule_render()

    # ==============================================================================================
    # SERIE
    # ==============================================================================================
    def _current_series(self):
        i = self._sel_index(self.se_tree)
        return None if i is None else self.cfg.series[i]

    def _on_series_selected(self):
        s = self._current_series()
        if s is None:
            return
        self._loading = True
        for k in self.sv:
            v = getattr(s, k)
            self.sv[k].set(str(v + 1) if k == "panel" else (f"{v:g}" if isinstance(v, float) else str(v)))
        self.sv_visible.set(s.visible)
        self._set_swatch(s.color)
        self._loading = False

    def _apply_series(self):
        s = self._current_series()
        if s is None or self._loading:
            return
        s.label = self.sv["label"].get()
        s.linestyle = self.sv["linestyle"].get() or s.linestyle
        s.linewidth = self._float(self.sv["linewidth"].get(), s.linewidth)
        s.marker = self.sv["marker"].get() or s.marker
        s.markevery = self._int(self.sv["markevery"].get(), s.markevery)
        s.drawstyle = self.sv["drawstyle"].get() or s.drawstyle
        s.scale = self._float(self.sv["scale"].get(), s.scale)
        s.offset = self._float(self.sv["offset"].get(), s.offset)
        s.panel = max(0, min(3, self._int(self.sv["panel"].get(), s.panel + 1) - 1))
        if s.panel >= len(self.cfg.panels):
            self._set_npanels(s.panel + 1)
        s.alpha = max(0.05, min(1.0, self._float(self.sv["alpha"].get(), s.alpha)))
        s.visible = self.sv_visible.get()
        self._refresh_series(select=self.cfg.series.index(s), keep_form=True)
        self.schedule_render()

    def pick_color(self):
        s = self._current_series()
        if s is None:
            return
        rgb, hexc = colorchooser.askcolor(color=s.color, title="Colore della serie")
        if hexc:
            s.color = hexc.upper(); self._set_swatch(s.color); self.schedule_render(0)

    def _set_swatch(self, c):
        self.color_swatch.configure(bg=c); self.color_hex.configure(text=c)

    def move_series(self, d):
        i = self._sel_index(self.se_tree)
        if i is None or not (0 <= i + d < len(self.cfg.series)):
            return
        L = self.cfg.series
        L[i], L[i + d] = L[i + d], L[i]
        self._refresh_series(select=i + d); self.schedule_render()

    def duplicate_series(self):
        s = self._current_series()
        if s is None:
            return
        self.cfg.series.append(copy.deepcopy(s))
        self._refresh_series(select=len(self.cfg.series) - 1); self.schedule_render()

    def remove_series(self):
        i = self._sel_index(self.se_tree)
        if i is None:
            return
        del self.cfg.series[i]
        self._refresh_series(select=min(i, len(self.cfg.series) - 1)); self.schedule_render()

    def apply_palette(self):
        pal = pc.PALETTES.get(self.cfg.palette, pc.PALETTES["Okabe-Ito (daltonici)"])
        for k, s in enumerate(self.cfg.series):
            s.color = pal[k % len(pal)]
        self._on_series_selected(); self.schedule_render(0)

    # ==============================================================================================
    # ASSI / PANNELLI / LEGENDA / FIGURA
    # ==============================================================================================
    def _set_npanels(self, n):
        n = max(1, min(4, n))
        while len(self.cfg.panels) < n:
            self.cfg.panels.append(pc.Panel(legend="alto a destra"))
        del self.cfg.panels[n:]
        self._loading = True
        self.av["npanels"].set(str(n))
        names = [f"Pannello {i + 1}" for i in range(n)]
        self.cb_panel.configure(values=names); self.cb_lpanel.configure(values=names)
        if self._panel_index(self.v_panel_sel.get()) >= n:
            self.v_panel_sel.set(names[0])
        if self._panel_index(self.v_lpanel.get()) >= n:
            self.v_lpanel.set(names[0])
        self._loading = False
        self._load_panel(); self._load_legend()

    @staticmethod
    def _panel_index(name):
        try:
            return int(name.split()[-1]) - 1
        except Exception:
            return 0

    def _apply_axes(self):
        if self._loading:
            return
        c = self.cfg
        c.xlabel = self.av["xlabel"].get(); c.xmin = self.av["xmin"].get(); c.xmax = self.av["xmax"].get()
        c.title = self.av["title"].get()
        n = self._int(self.av["npanels"].get(), len(c.panels))
        if n != len(c.panels):
            self._set_npanels(n)
        for k in self.bv:
            setattr(c, k, self.bv[k].get())
        self.schedule_render()

    def _load_panel(self):
        p = self.cfg.panels[min(self._panel_index(self.v_panel_sel.get()), len(self.cfg.panels) - 1)]
        self._loading = True
        for k in self.pv:
            v = getattr(p, k)
            self.pv[k].set(f"{v:g}" if isinstance(v, float) else str(v))
        self.pv_log.set(p.log)
        self._loading = False

    def _apply_panel(self):
        if self._loading:
            return
        p = self.cfg.panels[min(self._panel_index(self.v_panel_sel.get()), len(self.cfg.panels) - 1)]
        p.ylabel = self.pv["ylabel"].get(); p.ymin = self.pv["ymin"].get(); p.ymax = self.pv["ymax"].get()
        p.height_ratio = self._float(self.pv["height_ratio"].get(), p.height_ratio)
        p.hlines = self.pv["hlines"].get(); p.vspans = self.pv["vspans"].get(); p.log = self.pv_log.get()
        self.schedule_render()

    def _load_legend(self):
        p = self.cfg.panels[min(self._panel_index(self.v_lpanel.get()), len(self.cfg.panels) - 1)]
        self._loading = True
        for k in self.lv:
            v = getattr(p, k)
            self.lv[k].set(f"{v:g}" if isinstance(v, float) else str(v))
        self._loading = False

    def _apply_legend(self):
        if self._loading:
            return
        p = self.cfg.panels[min(self._panel_index(self.v_lpanel.get()), len(self.cfg.panels) - 1)]
        p.legend = self.lv["legend"].get() or p.legend
        p.legend_ncol = self._int(self.lv["legend_ncol"].get(), p.legend_ncol)
        p.legend_title = self.lv["legend_title"].get()
        p.legend_x = self._float(self.lv["legend_x"].get(), p.legend_x)
        p.legend_y = self._float(self.lv["legend_y"].get(), p.legend_y)
        self.schedule_render()

    def _apply_preset(self):
        w, h = pc.SIZE_PRESETS[self.fv["preset"].get()]
        self.fv["width_cm"].set(f"{w:g}"); self.fv["height_cm"].set(f"{h:g}")

    def capture_limits(self):
        axes = self.fig.axes
        if not axes:
            return
        lo, hi = axes[-1].get_xlim()
        self.av["xmin"].set(f"{lo:.4g}"); self.av["xmax"].set(f"{hi:.4g}")
        for p, ax in zip(self.cfg.panels, axes):
            lo, hi = ax.get_ylim(); p.ymin, p.ymax = f"{lo:.4g}", f"{hi:.4g}"
        self._load_panel(); self.schedule_render(0)

    def reset_limits(self):
        self.av["xmin"].set(""); self.av["xmax"].set("")
        for p in self.cfg.panels:
            p.ymin = p.ymax = ""
        self._load_panel(); self.schedule_render(0)

    def _apply_figure(self):
        if self._loading:
            return
        c = self.cfg
        c.width_cm = max(2.0, self._float(self.fv["width_cm"].get(), c.width_cm))
        c.height_cm = max(1.5, self._float(self.fv["height_cm"].get(), c.height_cm))
        c.font_size = max(5.0, self._float(self.fv["font_size"].get(), c.font_size))
        c.font = self.fv["font"].get() or c.font
        c.palette = self.fv["palette"].get() or c.palette
        c.usetex = self.v_usetex.get()
        c.legend_frame = self.v_lframe.get()
        c.legend_fontsize = self._float(self.v_lfont.get(), 0.0)
        c.svg_text_as_path = self.v_svgpath.get(); c.transparent = self.v_transp.get()
        self.schedule_render()

    # ==============================================================================================
    # EXPORT E CONFIGURAZIONE
    # ==============================================================================================
    def export(self, fmt):
        if not self.cfg.series:
            messagebox.showinfo("Esporta", "Il grafico non contiene serie."); return
        base = Path(self.config_path).stem if self.config_path else "figura"
        path = filedialog.asksaveasfilename(title=f"Esporta {fmt.upper()}", defaultextension=f".{fmt}",
                                            initialfile=f"{base}.{fmt}", filetypes=[(fmt.upper(), f"*.{fmt}")])
        if not path:
            return
        self.config(cursor="watch"); self.update_idletasks()
        try:
            warnings = pc.export(self.cfg, self.store, path)
        except Exception as e:
            self.config(cursor="")
            culprit = pc.find_latex_culprit(self.cfg) if "latex" in str(e).lower() else None
            messagebox.showerror("Esporta", ("Errore LaTeX nella " + culprit) if culprit else self._short_error(e)); return
        self.config(cursor="")
        self._last_export = path
        self._update_latex_snippet(path)
        self.lbl_status.configure(text=f"esportato: {path}", foreground="#2a6")
        if warnings:
            messagebox.showwarning("Esporta", "Esportato, ma con avvisi:\n" + "\n".join(warnings))
        self.schedule_render(0)   # l'export ha usato la risoluzione di stampa: ripristina l'anteprima

    def _update_latex_snippet(self, path):
        name = Path(path).stem
        inc = f"figure/{Path(path).name}"
        if path.endswith(".svg"):
            body = (f"% SVG: \\usepackage{{svg}} nel preambolo e compilazione con --shell-escape (serve Inkscape).\n"
                    f"\\begin{{figure}}[htbp]\n  \\centering\n  \\includesvg{{figure/{name}}}\n")
        else:
            body = f"\\begin{{figure}}[htbp]\n  \\centering\n  \\includegraphics{{{inc}}}\n"
        body += f"  \\caption{{…}}\n  \\label{{fig:{name}}}\n\\end{{figure}}\n"
        self.txt_latex.delete("1.0", "end"); self.txt_latex.insert("1.0", body)

    def copy_latex(self):
        self.clipboard_clear(); self.clipboard_append(self.txt_latex.get("1.0", "end").strip())
        self.lbl_status.configure(text="codice LaTeX copiato", foreground="#2a6")

    def save_config(self):
        path = filedialog.asksaveasfilename(title="Salva configurazione", defaultextension=".json",
                                            initialfile=Path(self.config_path).name if self.config_path else "figura.json",
                                            filetypes=[("Configurazione", "*.json")])
        if not path:
            return
        Path(path).write_text(self.cfg.to_json())
        self.config_path = path
        self.title(f"thesis_plot — {Path(path).name}")
        self.lbl_status.configure(text=f"salvato: {path}", foreground="#2a6")

    def open_config(self, path=None):
        path = path or filedialog.askopenfilename(title="Apri configurazione", filetypes=[("Configurazione", "*.json")])
        if not path:
            return
        try:
            cfg = pc.PlotConfig.from_json(Path(path).read_text())
            missing = [d.path for d in cfg.datasets if not Path(d.path).exists()]
            if missing:
                messagebox.showwarning("Configurazione", "File non trovati:\n" + "\n".join(missing))
        except Exception as e:
            messagebox.showerror("Configurazione", str(e)); return
        self.cfg = cfg
        self.config_path = path
        self.title(f"thesis_plot — {Path(path).name}")
        self._refresh_all()

    # ==============================================================================================
    # aggiornamento dei widget dal modello
    # ==============================================================================================
    def _refresh_all(self):
        c = self.cfg
        self._loading = True
        for k in ("xlabel", "xmin", "xmax", "title"):
            self.av[k].set(getattr(c, k))
        for k in self.bv:
            self.bv[k].set(getattr(c, k))
        self.fv["width_cm"].set(f"{c.width_cm:g}"); self.fv["height_cm"].set(f"{c.height_cm:g}")
        self.fv["font_size"].set(f"{c.font_size:g}"); self.fv["font"].set(c.font); self.fv["palette"].set(c.palette)
        match = [k for k, v in pc.SIZE_PRESETS.items() if v == (c.width_cm, c.height_cm)]
        self.fv["preset"].set(match[0] if match else "")
        self.v_usetex.set(c.usetex); self.v_lframe.set(c.legend_frame)
        self.v_lfont.set(f"{c.legend_fontsize:g}" if c.legend_fontsize else "")
        self.v_svgpath.set(c.svg_text_as_path); self.v_transp.set(c.transparent)
        self._loading = False
        self._set_npanels(len(c.panels))
        self._refresh_datasets(select=0 if c.datasets else None)
        self._refresh_series(select=0 if c.series else None)
        self.schedule_render(0)

    def _refresh_datasets(self, select=None, keep_form=False):
        self.ds_tree.delete(*self.ds_tree.get_children())
        for i, d in enumerate(self.cfg.datasets):
            try:
                n = len(self.store.load(d.path))
            except Exception:
                n = "?"
            self.ds_tree.insert("", "end", iid=str(i), values=(d.alias, f"{Path(d.path).parent.name}/{Path(d.path).name}", n))
        if select is not None and 0 <= select < len(self.cfg.datasets):
            self.ds_tree.selection_set(str(select))
            if not keep_form:
                self._on_dataset_selected()
        else:
            self.lb_cols.delete(0, "end")

    def _refresh_series(self, select=None, keep_form=False):
        cur = self._sel_index(self.se_tree) if select is None else select
        self.se_tree.delete(*self.se_tree.get_children())
        for i, s in enumerate(self.cfg.series):
            self.se_tree.insert("", "end", iid=str(i), values=(s.panel + 1, ("" if s.visible else "(nascosta) ") + s.label,
                                                               f"{s.dataset} : {s.column}"))
        if cur is not None and 0 <= cur < len(self.cfg.series):
            self.se_tree.selection_set(str(cur))
            if not keep_form:
                self._on_series_selected()

    # ==============================================================================================
    # utilita'
    # ==============================================================================================
    @staticmethod
    def _sel_index(tree):
        sel = tree.selection()
        return int(sel[0]) if sel else None

    @staticmethod
    def _float(s, default):
        try:
            return float(str(s).replace(",", "."))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _int(s, default):
        try:
            return int(float(str(s)))
        except (TypeError, ValueError):
            return default


def main():
    ap = argparse.ArgumentParser(description="Grafici da CSV per la tesi (anteprima, LaTeX, export PDF/SVG).")
    ap.add_argument("csv", nargs="*", help="CSV da caricare all'avvio")
    ap.add_argument("--config", help="configurazione .json da riaprire")
    args = ap.parse_args()
    app = App()
    if args.config:
        app.open_config(args.config)
    if args.csv:
        app.load_csv(args.csv)
    app.mainloop()


if __name__ == "__main__":
    main()
