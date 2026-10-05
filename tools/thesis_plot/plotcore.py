"""Motore di rendering di thesis_plot: configurazione, dati, stile scientifico, export.

Non dipende dall'interfaccia grafica: la stessa configurazione salvata in JSON dalla GUI si
rigenera da riga di comando, cosi' le figure della tesi restano riproducibili.

    python plotcore.py figura.json figura.pdf      # o .svg / .png
"""
from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.figure import Figure

# fontTools (usato da matplotlib per incorporare i font) segnala come "timestamp troppo basso" la data
# di creazione dei font Computer Modern, che e' del 1990: innocuo, ma riempie il terminale.
logging.getLogger("fontTools").setLevel(logging.ERROR)

# --------------------------------------------------------------------------------------------------
# Palette (colori in hex). Okabe-Ito e Tol sono distinguibili anche da chi ha deficit cromatici.
# --------------------------------------------------------------------------------------------------
PALETTES = {
    "Okabe-Ito (daltonici)": ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"],
    "Tol bright": ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB"],
    "Tol muted": ["#332288", "#88CCEE", "#44AA99", "#117733", "#999933", "#DDCC77", "#CC6677", "#882255", "#AA4499"],
    "Tableau 10": ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF"],
    "Scala di grigi (stampa B/N)": ["#000000", "#555555", "#888888", "#AAAAAA"],
}

LINESTYLES = {"continua": "-", "tratteggiata": "--", "punteggiata": ":", "tratto-punto": "-."}
MARKERS = {"nessuno": "", "cerchio": "o", "quadrato": "s", "triangolo": "^", "rombo": "D", "croce": "x", "più": "+", "punto": "."}
DRAWSTYLES = {"linea": "default", "gradini": "steps-post"}

LEGEND_LOCS = {
    "nessuna": None,
    "automatica (migliore)": "best",
    "alto a destra": "upper right", "alto a sinistra": "upper left", "alto al centro": "upper center",
    "basso a destra": "lower right", "basso a sinistra": "lower left", "basso al centro": "lower center",
    "centro a destra": "center right", "centro a sinistra": "center left", "centro": "center",
    "fuori, a destra": "OUT_RIGHT", "fuori, sopra": "OUT_TOP", "fuori, sotto": "OUT_BOTTOM",
    "personalizzata (trascina)": "CUSTOM",
}

# Dimensioni di riferimento [cm]. Le figure si esportano a dimensione ESATTA: nel documento si
# includono senza riscalarle (\includegraphics senza width), cosi' i font restano alla dimensione
# del testo.
SIZE_PRESETS = {
    "Tesi A4 — larghezza testo (15 × 8 cm)": (15.0, 8.0),
    "Tesi A4 — larghezza testo, alta (15 × 11 cm)": (15.0, 11.0),
    "Tesi A4 — mezza larghezza (7.4 × 6 cm)": (7.4, 6.0),
    "IEEE — una colonna (8.89 × 6 cm)": (8.89, 6.0),
    "IEEE — due colonne (18.2 × 8 cm)": (18.2, 8.0),
    "Elsevier — una colonna (9 × 6.5 cm)": (9.0, 6.5),
    "Elsevier — due colonne (19 × 9 cm)": (19.0, 9.0),
}

FONTS = {  # nome -> (preambolo LaTeX, famiglia mathtext di ripiego)
    "Computer Modern (LaTeX standard)": ("", "cm"),
    "Latin Modern": (r"\usepackage{lmodern}", "cm"),
    "Times": (r"\usepackage{mathptmx}", "stix"),
    "Palatino": (r"\usepackage{mathpazo}", "stix"),
    "Helvetica (sans-serif)": (r"\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}\usepackage{sfmath}", "dejavusans"),
}

# Colonne calcolate pronte all'uso. {J} viene sostituito con il giunto scelto nell'interfaccia.
# Convenzione degli angoli: REP-103 (ROS), la stessa di analyze_sim.py -- x avanti, y a sinistra,
# z in alto; pitch POSITIVO = busto in avanti, roll POSITIVO = inclinato verso destra.
# Le funzioni q, qd, qerr, tau, fk_height, gt, ... sono quelle di RunContext (sotto).
DERIVED_PRESETS = {   # voce -> (nome colonna, espressione, etichetta in legenda, etichetta asse y)
    "Base · pitch da IMU [°]": ("pitch", "deg(pitch_imu())", r"$\theta$", r"$\theta$ [\si{\degree}]"),
    "Base · roll da IMU [°]": ("roll", "deg(roll_imu())", r"$\phi$", r"$\phi$ [\si{\degree}]"),
    "Base · velocità angolare x [rad/s]": ("omega_x", "base_ang_vel_0 / 0.2", r"$\omega_x$", r"$\omega$ [\si{\radian\per\second}]"),
    "Base · velocità angolare y [rad/s]": ("omega_y", "base_ang_vel_1 / 0.2", r"$\omega_y$", r"$\omega$ [\si{\radian\per\second}]"),
    "Base · velocità angolare z [rad/s]": ("omega_z", "base_ang_vel_2 / 0.2", r"$\omega_z$", r"$\omega$ [\si{\radian\per\second}]"),
    "Quota · comandata [m]": ("h_cmd", "base_height_command_0 + 0.98", r"$h_\mathrm{cmd}$", r"$h$ [\si{\meter}]"),
    "Quota · stimata da FK, encoder + IMU [m]": ("h_fk", "fk_height()", r"$h_\mathrm{FK}$", r"$h$ [\si{\meter}]"),
    "Quota · errore stimato FK − comandata [cm]": ("h_err_fk", "100 * (fk_height() - (base_height_command_0 + 0.98))",
                                                   r"$h_\mathrm{FK} - h_\mathrm{cmd}$", r"$e_h$ [\si{\centi\meter}]"),
    "Comandi · vx [m/s]": ("v_x_cmd", "velocity_commands_0", r"$v_x^\mathrm{cmd}$", r"$v$ [\si{\meter\per\second}]"),
    "Comandi · vy [m/s]": ("v_y_cmd", "velocity_commands_1", r"$v_y^\mathrm{cmd}$", r"$v$ [\si{\meter\per\second}]"),
    "Comandi · ωz [rad/s]": ("omega_z_cmd", "velocity_commands_2", r"$\omega_z^\mathrm{cmd}$", r"$\omega_z$ [\si{\radian\per\second}]"),
    "Comandi · norma velocità lineare [m/s]": ("v_cmd", "sqrt(velocity_commands_0**2 + velocity_commands_1**2)",
                                               r"$\|v^\mathrm{cmd}\|$", r"$v$ [\si{\meter\per\second}]"),
    "Giunto · posizione assoluta [°]": ("q_{J}", 'deg(q("{J}"))', r"$q_\mathrm{{JL}}$", r"$q$ [\si{\degree}]"),
    "Giunto · posizione assoluta [rad]": ("q_{J}_rad", 'q("{J}")', r"$q_\mathrm{{JL}}$", r"$q$ [\si{\radian}]"),
    "Giunto · velocità [rad/s]": ("qd_{J}", 'qd("{J}")', r"$\dot q_\mathrm{{JL}}$", r"$\dot q$ [\si{\radian\per\second}]"),
    "Giunto · target della policy [°]": ("qt_{J}", 'deg(qtarget("{J}"))', r"$q^\mathrm{des}_\mathrm{{JL}}$", r"$q$ [\si{\degree}]"),
    "Giunto · errore di inseguimento target − misura [°]": ("qerr_{J}", 'deg(qerr("{J}"))', r"$e_{q,\mathrm{{JL}}}$", r"$e_q$ [\si{\degree}]"),
    "Giunto · coppia stimata kp·err − kd·q̇ [N·m]": ("tau_{J}", 'tau("{J}")', r"$\tau_\mathrm{{JL}}$", r"$\tau$ [\si{\newton\meter}]"),
    "Caviglie · coppia pitch media sx/dx [N·m]": ("tau_ankle_pitch", '(tau("left_ankle_pitch") + tau("right_ankle_pitch")) / 2',
                                                   r"$\tau_\mathrm{ankle}$", r"$\tau$ [\si{\newton\meter}]"),
    "Policy · variazione dell'azione ‖Δa‖": ("action_rate", "action_rate()", r"$\|\Delta a\|$", r"$\|\Delta a\|$"),
    "Sim (verità) · altezza base_z [m]": ("h_true", 'gt("base_z")', r"$h_\mathrm{gt}$", r"$h$ [\si{\meter}]"),
    "Sim (verità) · errore di quota vero − comandata [cm]": ("h_err_true", '100 * (gt("base_z") - (base_height_command_0 + 0.98))',
                                                             r"$h_\mathrm{gt} - h_\mathrm{cmd}$", r"$e_h$ [\si{\centi\meter}]"),
    "Sim (verità) · pitch [°]": ("pitch_true", "deg(gt_pitch())", r"$\theta_\mathrm{gt}$", r"$\theta$ [\si{\degree}]"),
    "Sim (verità) · roll [°]": ("roll_true", "deg(gt_roll())", r"$\phi_\mathrm{gt}$", r"$\phi$ [\si{\degree}]"),
    "Sim (verità) · yaw [°]": ("yaw_true", "deg(gt_yaw())", r"$\psi_\mathrm{gt}$", r"$\psi$ [\si{\degree}]"),
    "Sim (verità) · velocità avanti nel corpo [m/s]": ("v_x_true", "gt_vx_body()", r"$v_x$", r"$v$ [\si{\meter\per\second}]"),
    "Sim (verità) · velocità laterale nel corpo [m/s]": ("v_y_true", "gt_vy_body()", r"$v_y$", r"$v$ [\si{\meter\per\second}]"),
    "Sim (verità) · velocità di imbardata [rad/s]": ("omega_z_true", 'gt("wz")', r"$\omega_z$", r"$\omega_z$ [\si{\radian\per\second}]"),
    "Sim (verità) · posizione x dall'inizio [m]": ("x_true", 'gt("base_x") - gt("base_x")[0]', r"$x$", r"$x$ [\si{\meter}]"),
    "Sim (verità) · posizione y dall'inizio [m]": ("y_true", 'gt("base_y") - gt("base_y")[0]', r"$y$", r"$y$ [\si{\meter}]"),
}


def preset(key: str, joint: str = "left_knee") -> tuple:
    """(nome, espressione, etichetta, etichetta y) con il giunto sostituito. {JL} e' il nome del giunto
    leggibile in formula (left_knee -> left\\,knee)."""
    name, expr, label, ylabel = DERIVED_PRESETS[key]
    jl = joint.replace("_", r"\,")
    return (name.replace("{J}", joint), expr.replace("{J}", joint),
            label.replace("{JL}", jl), ylabel.replace("{JL}", jl))


def labels_for(name: str) -> tuple:
    """(etichetta legenda, etichetta y) per una colonna: quella del preset da cui il nome deriva
    (anche con il giunto sostituito, es. tau_left_ankle_pitch), altrimenti auto_label e y vuota."""
    for _, (pname, _e, label, ylabel) in DERIVED_PRESETS.items():
        if "{J}" not in pname and pname == name:
            return label, ylabel
    joints = joint_names()
    for key, (pname, _e, _l, _y) in DERIVED_PRESETS.items():
        if "{J}" in pname:
            pre, post = pname.split("{J}")
            if name.startswith(pre) and name.endswith(post) and len(name) > len(pre) + len(post):
                j = name[len(pre):len(name) - len(post)]
                if j in joints:
                    _, _, label, ylabel = preset(key, j)
                    return label, ylabel
    return auto_label(name), ""


_GREEK = {"alpha", "beta", "gamma", "delta", "epsilon", "eta", "theta", "kappa", "lambda", "mu", "rho",
          "sigma", "tau", "phi", "psi", "omega"}


def auto_label(name: str) -> str:
    """Etichetta di default per una colonna: simbolo con pedice se il nome lo suggerisce, altrimenti
    testo leggibile. h_cmd -> $h_\\mathrm{cmd}$, tau_ankle -> $\\tau_\\mathrm{ankle}$,
    left_knee_pos -> left knee pos (mai underscore letterali in legenda)."""
    toks = [t for t in str(name).split("_") if t]
    if not toks:
        return latex_escape(str(name))
    head, rest = toks[0], toks[1:]
    ok = all(t.isalnum() for t in toks)
    if ok and head.lower() in _GREEK:
        sym = "\\" + head.lower()
        return f"${sym}_\\mathrm{{{','.join(rest)}}}$" if rest else f"${sym}$"
    if ok and len(head) <= 2 and head.isalpha():
        return f"${head}_\\mathrm{{{','.join(rest)}}}$" if rest else f"${head}$"
    return latex_escape(" ".join(toks))


# Funzioni disponibili nelle espressioni per i log del controller (obs.csv di un run sim_* / exp_*)
RUN_FUNCTIONS_HELP = (
    "Per i log del controller (obs.csv): q(g), qd(g), qtarget(g), qerr(g), tau(g), kp(g), kd(g) "
    "con g = nome del giunto (es. \"left_knee\"); pitch_imu(), roll_imu(), fk_height(), action_rate(). "
    "Solo simulazione: gt(\"base_z\"), gt_pitch(), gt_roll(), gt_yaw(), gt_vx_body(), gt_vy_body(). "
    "Angoli in radianti: deg(...) per i gradi."
)


# --------------------------------------------------------------------------------------------------
# Configurazione
# --------------------------------------------------------------------------------------------------
@dataclass
class Dataset:
    path: str
    alias: str
    xcol: str = ""
    x_offset: float = 0.0
    x_scale: float = 1.0
    x_from_zero: bool = False
    derived: dict = field(default_factory=dict)   # nome -> espressione


@dataclass
class Series:
    dataset: str
    column: str
    label: str = ""
    color: str = "#0072B2"
    linestyle: str = "continua"
    linewidth: float = 1.2
    marker: str = "nessuno"
    markevery: int = 50
    drawstyle: str = "linea"
    scale: float = 1.0
    offset: float = 0.0
    panel: int = 0
    alpha: float = 1.0
    visible: bool = True


@dataclass
class Panel:
    ylabel: str = ""
    ymin: str = ""
    ymax: str = ""
    log: bool = False
    legend: str = "alto a destra"
    legend_x: float = 0.02
    legend_y: float = 0.70
    legend_ncol: int = 1
    legend_title: str = ""
    hlines: str = ""     # "0.98, 0.7:cutoff"   valore[:etichetta]
    vspans: str = ""     # "40-55, 160-170:spinta"
    height_ratio: float = 1.0


@dataclass
class PlotConfig:
    datasets: list = field(default_factory=list)
    series: list = field(default_factory=list)
    panels: list = field(default_factory=lambda: [Panel()])
    xlabel: str = r"$t$ [s]"
    xmin: str = ""
    xmax: str = ""
    title: str = ""
    width_cm: float = 15.0
    height_cm: float = 8.0
    font_size: float = 10.0
    font: str = "Computer Modern (LaTeX standard)"
    usetex: bool = True
    palette: str = "Okabe-Ito (daltonici)"
    grid: bool = True
    grid_minor: bool = False
    ticks_in: bool = True
    ticks_all_sides: bool = True
    minor_ticks: bool = True
    legend_frame: bool = True
    legend_fontsize: float = 0.0          # 0 = font_size - 1
    svg_text_as_path: bool = True
    transparent: bool = False

    # -- serializzazione ---------------------------------------------------------------------------
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    @staticmethod
    def from_json(text: str) -> "PlotConfig":
        d = json.loads(text)
        cfg = PlotConfig(**{k: v for k, v in d.items() if k not in ("datasets", "series", "panels")})
        cfg.datasets = [Dataset(**x) for x in d.get("datasets", [])]
        cfg.series = [Series(**x) for x in d.get("series", [])]
        cfg.panels = [Panel(**x) for x in d.get("panels", [])] or [Panel()]
        return cfg


# --------------------------------------------------------------------------------------------------
# Dati
# --------------------------------------------------------------------------------------------------
_EXPR_NS = {n: getattr(np, n) for n in (
    "sin", "cos", "tan", "arcsin", "arccos", "arctan", "arctan2", "sqrt", "abs", "exp", "log", "log10",
    "minimum", "maximum", "clip", "sign", "hypot", "gradient", "cumsum", "pi", "where", "mean", "std")}
_EXPR_NS.update({"deg": np.degrees, "rad": np.radians, "np": np})


class DataStore:
    """CSV caricati una volta sola, con le colonne calcolate applicate su richiesta (e in cache)."""

    def __init__(self):
        self._raw: dict[str, pd.DataFrame] = {}
        self._frames: dict = {}
        self._ctx: dict[str, "RunContext"] = {}

    def load(self, path: str) -> pd.DataFrame:
        path = str(Path(path).expanduser().resolve())
        if path not in self._raw:
            self._raw[path] = pd.read_csv(path)
        return self._raw[path]

    def context(self, path: str) -> "RunContext":
        path = str(Path(path).expanduser().resolve())
        if path not in self._ctx:
            self._ctx[path] = RunContext(path, self.load(path))
        return self._ctx[path]

    @staticmethod
    def default_xcol(df: pd.DataFrame) -> str:
        for c in ("ep_t", "sim_t", "t", "time", "wall_t"):
            if c in df.columns:
                return c
        return "__indice__"

    def frame(self, ds: Dataset) -> pd.DataFrame:
        df = self.load(ds.path)
        if not ds.derived:
            return df
        key = (str(Path(ds.path).expanduser().resolve()), tuple(ds.derived.items()))
        if key not in self._frames:
            df = df.copy()
            extra = self.context(ds.path).functions()
            for name, expr in ds.derived.items():
                df[name] = eval_expression(df, expr, extra)
            self._frames[key] = df
        return self._frames[key]

    def xy(self, ds: Dataset, s: Series) -> tuple[np.ndarray, np.ndarray]:
        df = self.frame(ds)
        if s.column not in df.columns:
            raise KeyError(f"colonna '{s.column}' assente in {ds.alias}")
        xcol = ds.xcol or self.default_xcol(df)
        x = np.arange(len(df), dtype=float) if xcol == "__indice__" else df[xcol].to_numpy(float)
        if ds.x_from_zero and len(x):
            x = x - x[0]
        x = x * ds.x_scale + ds.x_offset
        y = df[s.column].to_numpy(float) * s.scale + s.offset
        return x, y


def eval_expression(df: pd.DataFrame, expr: str, extra: dict | None = None) -> np.ndarray:
    ns = dict(_EXPR_NS)
    if extra:
        ns.update(extra)
    ns["col"] = lambda name: df[name].to_numpy(float)
    ns.update({c: df[c].to_numpy(float) for c in df.columns if str(c).isidentifier()
               and pd.api.types.is_numeric_dtype(df[c])})
    out = eval(expr, {"__builtins__": {}}, ns)  # espressioni scritte dall'utente, in locale
    out = np.asarray(out, dtype=float)
    if out.shape == ():
        out = np.full(len(df), float(out))
    if len(out) != len(df):
        raise ValueError(f"l'espressione restituisce {len(out)} valori invece di {len(df)}")
    return out


class RunContext:
    """Grandezze fisiche ricavabili dai log del controller h1_2_ctrl_squat.

    Un run (sim_<ts>/ o exp_<ts>/) contiene obs.csv, errors.csv e commands.csv, scritti allo stesso
    passo di policy (stesse righe, stesso ep_t), e in simulazione anche ground_truth.csv, scritto dal
    simulatore a 100 Hz su un altro orologio: quello si riallinea su wall_t, l'unico comune.

    obs.csv contiene le osservazioni COME LE VEDE LA POLICY, cioe' scalate: base_ang_vel x0.2,
    joint_vel_rel x0.05, joint_pos_rel relative alla posa di default. Le funzioni qui restituiscono
    le grandezze fisiche (rad, rad/s, N.m, m).
    """

    VEL_SCALE = 0.05

    def __init__(self, path: str, obs: pd.DataFrame):
        self.dir = Path(path).parent
        self.obs = obs
        self._cache: dict = {}

    # -- file fratelli ---------------------------------------------------------------------------
    def _sibling(self, name: str) -> pd.DataFrame:
        if name not in self._cache:
            f = self.dir / name
            if not f.exists():
                raise FileNotFoundError(f"{name} non trovato accanto al CSV"
                                        + (" (il ground truth esiste solo in simulazione)" if name == "ground_truth.csv" else ""))
            df = pd.read_csv(f)
            if name != "ground_truth.csv" and len(df) != len(self.obs):
                # stesse righe di obs.csv per costruzione; se no, riallinea su ep_t
                df = df.set_index("ep_t").reindex(self.obs["ep_t"], method="nearest").reset_index()
            self._cache[name] = df
        return self._cache[name]

    @staticmethod
    def _tools():
        import importlib, sys
        tools = str(Path(__file__).resolve().parents[1])
        if tools not in sys.path:
            sys.path.insert(0, tools)
        return importlib.import_module("analyze_sim")

    def _col(self, df, name):
        if name not in df.columns:
            raise KeyError(f"colonna '{name}' assente (giunto scritto bene? es. left_knee, right_ankle_pitch)")
        return df[name].to_numpy(float)

    # -- giunti -----------------------------------------------------------------------------------
    def q(self, j):
        A = self._tools()
        if j not in A.JOINT_NAMES:
            raise KeyError(f"giunto '{j}' sconosciuto")
        return self._col(self.obs, f"{j}_pos") + A.DEFAULT_JOINT_POS[A.JOINT_NAMES.index(j)]

    def qd(self, j):
        return self._col(self.obs, f"{j}_vel") / self.VEL_SCALE

    def qtarget(self, j):
        return self._col(self._sibling("commands.csv"), f"{j}_q_target")

    def qerr(self, j):
        return self._col(self._sibling("errors.csv"), f"{j}_q_err")

    def kp(self, j):
        return self._col(self._sibling("commands.csv"), f"{j}_kp")

    def kd(self, j):
        return self._col(self._sibling("commands.csv"), f"{j}_kd")

    def tau(self, j):
        """Coppia del PD del controller: kp*(q_target - q) - kd*qd. E' la coppia COMANDATA al motore
        (il PD gira sul driver), non una misura; da ferma coincide con la coppia di equilibrio."""
        return self.kp(j) * self.qerr(j) - self.kd(j) * self.qd(j)

    # -- base ----------------------------------------------------------------------------------------
    def _g(self):
        return (self._col(self.obs, "projected_gravity_0"), self._col(self.obs, "projected_gravity_1"),
                self._col(self.obs, "projected_gravity_2"))

    def pitch_imu(self):
        gx, gy, gz = self._g()
        return np.arctan2(gx, np.sqrt(gy ** 2 + gz ** 2))

    def roll_imu(self):
        gx, gy, gz = self._g()
        return np.arctan2(-gy, -gz)

    def fk_height(self):
        """Altezza della base da cinematica diretta (encoder + IMU), come la stima analyze_sim.py.
        Validata contro il ground truth: bias +0.3 cm, std 0.2 cm."""
        if "fk" not in self._cache:
            A = self._tools()
            jabs = np.stack([self.q(j) for j in A.JOINT_NAMES], axis=1)
            h = A.estimate_base_height_fk(jabs, self.roll_imu(), self.pitch_imu())
            if h is None:
                raise RuntimeError("stima FK non disponibile (mujoco o modello MJCF mancanti)")
            self._cache["fk"] = h
        return self._cache["fk"]

    def action_rate(self):
        cols = [c for c in self.obs.columns if c.endswith("_act") and "_hm" not in c]
        a = self.obs[cols].to_numpy(float)
        d = np.linalg.norm(np.diff(a, axis=0), axis=1)
        return np.concatenate([[d[0] if len(d) else 0.0], d])

    # -- verita' della simulazione ------------------------------------------------------------------
    def gt(self, name):
        g = self._sibling("ground_truth.csv")
        if name not in g.columns:
            raise KeyError(f"'{name}' non e' in ground_truth.csv ({', '.join(g.columns)})")
        return np.interp(self._col(self.obs, "wall_t"), g["wall_t"].to_numpy(float), g[name].to_numpy(float))

    def _quat(self):
        return [self.gt(c) for c in ("qw", "qx", "qy", "qz")]

    def gt_roll(self):
        w, x, y, z = self._quat()
        return np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))

    def gt_pitch(self):
        w, x, y, z = self._quat()
        return np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))

    def gt_yaw(self):
        w, x, y, z = self._quat()
        return np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def gt_vx_body(self):
        yaw = self.gt_yaw()
        return np.cos(yaw) * self.gt("vx") + np.sin(yaw) * self.gt("vy")

    def gt_vy_body(self):
        yaw = self.gt_yaw()
        return -np.sin(yaw) * self.gt("vx") + np.cos(yaw) * self.gt("vy")

    def functions(self) -> dict:
        names = ("q", "qd", "qtarget", "qerr", "kp", "kd", "tau", "pitch_imu", "roll_imu", "fk_height",
                 "action_rate", "gt", "gt_roll", "gt_pitch", "gt_yaw", "gt_vx_body", "gt_vy_body")
        return {n: getattr(self, n) for n in names}


def joint_names() -> list:
    try:
        return list(RunContext._tools().JOINT_NAMES)
    except Exception:
        return []


# --------------------------------------------------------------------------------------------------
# Stile
# --------------------------------------------------------------------------------------------------
def latex_available() -> bool:
    try:
        return subprocess.run(["latex", "--version"], capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


def style_rc(cfg: PlotConfig) -> dict:
    fs = cfg.font_size
    pre, mathfont = FONTS.get(cfg.font, FONTS["Computer Modern (LaTeX standard)"])
    sans = "sans" in cfg.font.lower()
    rc = {
        "text.usetex": cfg.usetex,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{siunitx}" + pre,
        "font.family": "sans-serif" if sans else "serif",
        "font.serif": ["Computer Modern Roman", "cmr10", "DejaVu Serif"],
        "mathtext.fontset": mathfont,
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": True,
        "font.size": fs, "axes.labelsize": fs, "axes.titlesize": fs,
        "xtick.labelsize": fs - 1, "ytick.labelsize": fs - 1,
        "legend.fontsize": cfg.legend_fontsize or fs - 1, "legend.title_fontsize": cfg.legend_fontsize or fs - 1,
        "axes.linewidth": 0.6,
        "xtick.direction": "in" if cfg.ticks_in else "out", "ytick.direction": "in" if cfg.ticks_in else "out",
        "xtick.top": cfg.ticks_all_sides, "ytick.right": cfg.ticks_all_sides,
        "xtick.minor.visible": cfg.minor_ticks, "ytick.minor.visible": cfg.minor_ticks,
        "xtick.major.size": 3.5, "ytick.major.size": 3.5, "xtick.minor.size": 2.0, "ytick.minor.size": 2.0,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.45, "ytick.minor.width": 0.45,
        "lines.linewidth": 1.2, "lines.markersize": 4,
        "legend.frameon": cfg.legend_frame, "legend.framealpha": 1.0, "legend.edgecolor": "0.25",
        "legend.fancybox": False, "legend.borderpad": 0.4, "legend.handlelength": 1.8, "legend.labelspacing": 0.3,
        "patch.linewidth": 0.5,
        "grid.linewidth": 0.4, "grid.alpha": 0.6, "grid.linestyle": ":", "grid.color": "0.6",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "path" if cfg.svg_text_as_path else "none",
        "axes.prop_cycle": mpl.cycler(color=PALETTES.get(cfg.palette, PALETTES["Okabe-Ito (daltonici)"])),
        "figure.constrained_layout.h_pad": 0.02, "figure.constrained_layout.w_pad": 0.02,
    }
    if cfg.usetex:
        rc["font.serif"] = ["Computer Modern Roman"]
    else:
        # senza LaTeX: Computer Modern di matplotlib per testo e formule (resa molto vicina a LaTeX)
        rc["font.serif"] = ["cmr10", "DejaVu Serif"]
    return rc


# Unita' di siunitx -> testo semplice, per l'anteprima SENZA LaTeX (il motore interno di matplotlib non
# conosce \si e, con il font cmr10, renderebbe "\" come una virgoletta). Con LaTeX vero non serve.
_SI_UNITS = {
    # i simboli non ASCII passano dal motore matematico: cmr10 non ha i glifi ° ² · µ
    "meter": "m", "metre": "m", "second": "s", "degree": r"$^\circ$", "radian": "rad", "newton": "N",
    "kilogram": "kg", "gram": "g", "hertz": "Hz", "volt": "V", "ampere": "A", "watt": "W",
    "milli": "m", "centi": "c", "kilo": "k", "micro": r"$\mu$", "percent": r"$\%$", "joule": "J",
}


def _si_to_plain(text: str) -> str:
    import re

    def units(body: str) -> str:
        toks = re.findall(r"\\(per|squared|cubed|[A-Za-z]+)|([^\\]+)", body)
        out, per, prefix = [], False, ""
        for name, lit in toks:
            if lit:
                out.append(lit); continue
            if name == "per":
                per = True; continue
            if name in ("squared", "cubed"):
                if out: out[-1] += "$^2$" if name == "squared" else "$^3$"
                continue
            sym = _SI_UNITS.get(name, name)
            if name in ("milli", "centi", "kilo", "micro"):
                prefix = sym; continue
            sym = prefix + sym; prefix = ""
            out.append(("/" if per else (r"$\cdot$" if out else "")) + sym); per = False
        return "".join(out)

    text = re.sub(r"\\SI\{([^}]*)\}\{([^}]*)\}", lambda m: m.group(1) + " " + units(m.group(2)), text)
    text = re.sub(r"\\si\{([^}]*)\}", lambda m: units(m.group(1)), text)
    return text


def latex_escape(text: str) -> str:
    """Nome di colonna -> testo sicuro per LaTeX (left_knee_pos -> left\\_knee\\_pos)."""
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        text = text.replace(a, b)
    return text


def _plain(cfg, text: str) -> str:
    if cfg.usetex:
        return text
    text = _si_to_plain(text)
    for a, b in ((r"\_", "_"), (r"\%", "%"), (r"\&", "&"), (r"\#", "#"), (r"\textbackslash{}", "/")):
        text = text.replace(a, b)
    return text


def _num(s: str):
    s = str(s).strip().replace(",", ".")
    return float(s) if s else None


def _parse_hlines(text: str):
    out = []
    for tok in [t.strip() for t in text.split(";" if ";" in text else ",") if t.strip()]:
        v, _, lab = tok.partition(":")
        out.append((float(v.replace(" ", "")), lab.strip()))
    return out


def _parse_vspans(text: str):
    out = []
    for tok in [t.strip() for t in text.split(";" if ";" in text else ",") if t.strip()]:
        rng, _, lab = tok.partition(":")
        a, b = rng.split("-", 1) if rng.count("-") == 1 or not rng.startswith("-") else rng[1:].split("-", 1)
        out.append((float(a), float(b), lab.strip()))
    return out


# --------------------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------------------
class RenderResult:
    def __init__(self, fig, axes, legends, warnings):
        self.fig, self.axes, self.legends, self.warnings = fig, axes, legends, warnings


def render(cfg: PlotConfig, store: DataStore, fig: Figure | None = None, dpi: float = 100) -> RenderResult:
    """Disegna la figura. Le rcParams vengono applicate globalmente: va chiamata prima di draw/savefig."""
    mpl.rcParams.update(style_rc(cfg))
    w, h = cfg.width_cm / 2.54, cfg.height_cm / 2.54
    if fig is None:
        fig = Figure(figsize=(w, h), dpi=dpi, layout="constrained")
    else:
        fig.clear()
        fig.set_size_inches(w, h, forward=True)
        fig.set_dpi(dpi)
        fig.set_layout_engine("constrained")

    n = max(1, len(cfg.panels))
    ratios = [max(0.1, float(p.height_ratio or 1.0)) for p in cfg.panels[:n]]
    axes = fig.subplots(n, 1, sharex=True, squeeze=False, gridspec_kw={"height_ratios": ratios})[:, 0]
    datasets = {d.alias: d for d in cfg.datasets}
    warnings = []

    for s in cfg.series:
        if not s.visible:
            continue
        ds = datasets.get(s.dataset)
        if ds is None:
            warnings.append(f"serie '{s.label or s.column}': dataset '{s.dataset}' non caricato"); continue
        try:
            x, y = store.xy(ds, s)
        except Exception as e:  # colonna assente, espressione sbagliata...
            warnings.append(f"serie '{s.label or s.column}': {e}"); continue
        ax = axes[min(max(s.panel, 0), n - 1)]
        mk = MARKERS.get(s.marker, "")
        ax.plot(x, y, color=s.color, linestyle=LINESTYLES.get(s.linestyle, "-"), linewidth=s.linewidth,
                marker=mk or None, markevery=max(1, int(s.markevery)) if mk else None,
                drawstyle=DRAWSTYLES.get(s.drawstyle, "default"), alpha=s.alpha,
                label=_plain(cfg, s.label) if s.label else None)

    legends = []
    for i, (ax, p) in enumerate(zip(axes, cfg.panels)):
        for (a, b, lab) in _safe(_parse_vspans, p.vspans, warnings, f"pannello {i+1}, bande"):
            ax.axvspan(a, b, color="0.85", alpha=0.6, linewidth=0, zorder=0)
            if lab:
                ax.text((a + b) / 2, 0.98, _plain(cfg, lab), transform=ax.get_xaxis_transform(), ha="center", va="top",
                        fontsize=cfg.font_size - 2)
        hl = _safe(_parse_hlines, p.hlines, warnings, f"pannello {i+1}, linee")
        for (v, lab) in hl:
            ax.axhline(v, color="0.35", linestyle="--", linewidth=0.7, zorder=1)
        if p.ylabel:
            ax.set_ylabel(_plain(cfg, p.ylabel))
        if p.log:
            ax.set_yscale("log")
        lo, hi = _num(p.ymin), _num(p.ymax)
        if lo is not None or hi is not None:
            ax.set_ylim(lo, hi)
        # etichette delle linee: a limiti definitivi, sotto la linea se sta nella meta' alta
        ylo, yhi = ax.get_ylim()
        for (v, lab) in hl:
            if not lab:
                continue
            upper = (v - ylo) / ((yhi - ylo) or 1.0) > 0.5
            ax.annotate(_plain(cfg, lab), xy=(0.005, v), xycoords=ax.get_yaxis_transform(),
                        xytext=(0, -1.5 if upper else 1.5), textcoords="offset points",
                        ha="left", va="top" if upper else "bottom", fontsize=cfg.font_size - 2, color="0.25")
        if cfg.grid:
            ax.grid(True, which="major")
            if cfg.grid_minor:
                ax.grid(True, which="minor", alpha=0.3)
        legends.append(_legend(ax, p, cfg))

    axes[-1].set_xlabel(_plain(cfg, cfg.xlabel))
    lo, hi = _num(cfg.xmin), _num(cfg.xmax)
    if lo is not None or hi is not None:
        axes[-1].set_xlim(lo, hi)
    else:
        axes[-1].autoscale(axis="x", tight=True)
    if cfg.title:
        axes[0].set_title(_plain(cfg, cfg.title))
    return RenderResult(fig, list(axes), legends, warnings)


def _safe(fn, text, warnings, where):
    try:
        return fn(text) if text.strip() else []
    except Exception:
        warnings.append(f"{where}: formato non valido ('{text}')")
        return []


def _legend(ax, p: Panel, cfg: PlotConfig):
    mode = LEGEND_LOCS.get(p.legend, "upper right")
    handles, labels = ax.get_legend_handles_labels()
    if mode is None or not handles:
        return None
    kw = dict(ncol=max(1, int(p.legend_ncol)), title=_plain(cfg, p.legend_title) or None)
    if mode == "OUT_RIGHT":
        leg = ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), borderaxespad=0, **kw)
    elif mode == "OUT_TOP":
        kw["ncol"] = max(kw["ncol"], len(handles)) if p.legend_ncol <= 1 else kw["ncol"]
        leg = ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), borderaxespad=0, **kw)
    elif mode == "OUT_BOTTOM":
        kw["ncol"] = max(kw["ncol"], len(handles)) if p.legend_ncol <= 1 else kw["ncol"]
        leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), borderaxespad=0, **kw)
    elif mode == "CUSTOM":
        leg = ax.legend(loc=(p.legend_x, p.legend_y), **kw)
    else:
        leg = ax.legend(loc=mode, **kw)
    leg.get_frame().set_linewidth(0.5)
    return leg


def find_latex_culprit(cfg: PlotConfig) -> str | None:
    """Se LaTeX fallisce, prova le etichette una per una e restituisce dove sta quella sbagliata."""
    from matplotlib.texmanager import TexManager
    mpl.rcParams.update(style_rc(cfg))     # stesso preambolo (siunitx, font) del rendering vero
    texts = [("etichetta x", cfg.xlabel), ("titolo", cfg.title)]
    for i, p in enumerate(cfg.panels):
        texts += [(f"etichetta y del pannello {i + 1}", p.ylabel), (f"titolo legenda pannello {i + 1}", p.legend_title)]
        for v in (p.hlines, p.vspans):
            texts += [(f"etichetta di una linea/banda del pannello {i + 1}", t.partition(":")[2])
                      for t in v.replace(";", ",").split(",") if ":" in t]
    texts += [(f"serie «{s.dataset}: {s.column}»", s.label) for s in cfg.series if s.visible]
    for where, t in texts:
        if not t or not t.strip():
            continue
        try:
            TexManager().make_dvi(t, cfg.font_size)
        except Exception as e:
            bang = [l for l in str(e).splitlines() if l.startswith("!")]
            return f"{where}: «{t}» → {bang[0] if bang else 'errore LaTeX'}"
    return None


def export(cfg: PlotConfig, store: DataStore, path: str, dpi: int = 300) -> list:
    """Esporta a dimensione ESATTA (larghezza x altezza della configurazione). Restituisce gli avvisi."""
    res = render(cfg, store, dpi=dpi)
    fmt = Path(path).suffix.lower().lstrip(".")
    res.fig.savefig(path, format=fmt, dpi=dpi, transparent=cfg.transparent,
                    metadata={"Creator": "thesis_plot"} if fmt in ("pdf", "svg") else None)
    return res.warnings


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(1)
    cfg = PlotConfig.from_json(Path(sys.argv[1]).read_text())
    for w in export(cfg, DataStore(), sys.argv[2]):
        print("avviso:", w)
    print("scritto", sys.argv[2])
