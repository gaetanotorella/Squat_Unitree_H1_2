# thesis_plot

Grafici da CSV per la tesi: testo composto da LaTeX, stile da articolo scientifico, anteprima
uguale al file esportato, export vettoriale PDF/SVG.

```bash
cd ~/unitree_ws/Squat_Unitree_H1_2/tools/thesis_plot
../../../venv_sim/bin/python thesis_plot.py                                  # finestra vuota
../../../venv_sim/bin/python thesis_plot.py reale/obs.csv sim/obs.csv        # con CSV già caricati
../../../venv_sim/bin/python thesis_plot.py --config figure/pitch.json       # riapre una figura
```

## Flusso tipico

1. **Dati** — *Carica CSV* (anche più file: reale e simulazione). Ogni file ha un nome breve
   (`reale`, `sim`), la colonna dell'asse x (di default `ep_t`), un offset per allinearli.
   Le **colonne calcolate** creano grandezze che nel CSV non ci sono, ad esempio il pitch in gradi
   da `projected_gravity` (ci sono delle preimpostate). Poi cerca la colonna, scegli il pannello e
   *Aggiungi al grafico* (o doppio clic).
2. **Serie** — etichetta (LaTeX), colore, tratto, spessore, marcatore, linea o *gradini* (adatti ai
   comandi), scala e offset (es. `57.2958` per radianti → gradi), pannello, opacità.
3. **Assi** — etichette, limiti, fino a 4 **pannelli sovrapposti** con l'asse x condiviso,
   **linee orizzontali** di riferimento (`0.98:nominale`) e **bande verticali** (`40-56:squat`).
   Puoi ingrandire con la lente sotto l'anteprima e copiare i limiti nella figura.
4. **Legenda** — posizione per ogni pannello (dentro, fuori a destra/sopra/sotto, o personalizzata).
   La legenda si **trascina** direttamente nell'anteprima: la posizione resta memorizzata.
5. **Figura ed esporta** — formato (tesi A4, IEEE, Elsevier o libero), dimensione e carattere del
   testo, palette, **PDF / SVG / PNG**. Il codice LaTeX per includerla è già pronto da copiare.
   *Salva…* registra tutto in un `.json`.

## Grandezze calcolate

Nella scheda **Dati → Grandezze calcolate** ci sono le grandezze dei log del controller già pronte.
Con *Aggiungila a tutti i dataset caricati* la stessa grandezza va su reale e simulazione in un colpo;
*Aggiungi e metti nel grafico* la disegna subito (stesso colore per lo stesso dataset in tutti i pannelli).

| gruppo | grandezze | reale | sim |
|---|---|---|---|
| Base | pitch e roll da IMU [°], velocità angolare x/y/z [rad/s] | ✓ | ✓ |
| Quota | comandata [m], **stimata da FK (encoder + IMU)** [m], errore stimato [cm] | ✓ | ✓ |
| Comandi | vx, vy, ωz, norma della velocità lineare | ✓ | ✓ |
| Giunto (a scelta) | posizione assoluta [° / rad], velocità [rad/s], target della policy [°], errore di inseguimento [°], coppia stimata kp·err − kd·q̇ [N·m] | ✓ | ✓ |
| Caviglie | coppia pitch media sinistra/destra [N·m] | ✓ | ✓ |
| Policy | variazione dell'azione ‖Δa‖ | ✓ | ✓ |
| Sim (verità) | altezza base_z e errore vero [m, cm], pitch/roll/yaw veri [°], velocità avanti/laterale nel corpo [m/s], velocità di imbardata [rad/s], posizione x/y [m] | — | ✓ |

Le espressioni si possono anche scrivere a mano, con le funzioni `q(g)`, `qd(g)`, `qtarget(g)`,
`qerr(g)`, `tau(g)`, `kp(g)`, `kd(g)` (g = nome del giunto, es. `"left_knee"`), `pitch_imu()`,
`roll_imu()`, `fk_height()`, `action_rate()` e, in simulazione, `gt("base_z")`, `gt_pitch()`,
`gt_roll()`, `gt_yaw()`, `gt_vx_body()`, `gt_vy_body()`. Le funzioni leggono da sole `errors.csv`,
`commands.csv` e `ground_truth.csv` accanto a `obs.csv` (il ground truth viene riallineato su
`wall_t`, perché il simulatore lo scrive su un altro orologio).

**Unità.** `obs.csv` registra le osservazioni come le vede la policy, cioè scalate:
`base_ang_vel` ×0.2 e `joint_vel_rel` ×0.05. Le grandezze preimpostate le riportano in rad/s.

**Convenzione degli angoli: REP-103 (ROS)**, la stessa dei grafici di `analyze_sim.py` —
x avanti, y a sinistra, z in alto; **pitch positivo = busto in avanti**, roll positivo = inclinato
verso destra. In squat profondo il pitch è quindi positivo (~+22°).

**Verifiche** (replay di `exp_2026-09-22_14-45-16` in simulazione, stima contro verità):
pitch da IMU −0.013 ± 0.016°, roll +0.030 ± 0.051°, quota da FK +0.20 ± 0.09 cm.

**Coppia stimata.** È la coppia del PD del controller, kp·(q_target − q) − kd·q̇: quella
*comandata* al motore, non una misura. Da fermo coincide con la coppia di equilibrio.

**Etichette già in LaTeX.** Ogni grandezza preimpostata entra in legenda con il suo simbolo
(`$h_\mathrm{cmd}$`, `$h_\mathrm{FK}$`, `$\tau_\mathrm{left\,ankle\,pitch}$`, …) e, se l'asse y del
pannello è ancora vuoto, gli dà etichetta e unità (`$h$ [\si{\meter}]`). Le colonne grezze del CSV
diventano un simbolo con pedice quando il nome lo permette (`h_cmd` → $h_\mathrm{cmd}$,
`omega_z` → $\omega_\mathrm{z}$), altrimenti testo leggibile (`left_knee_pos` → «left knee pos»).
Nessun underscore letterale in legenda. Tutto resta modificabile in **Serie → Etichetta (LaTeX)**:
il testo tra `$…$` è in modo matematico, il resto è testo normale. Le configurazioni `.json` salvate
prima di questa modifica conservano le vecchie etichette (`h\_cmd`): vanno corrette a mano o
riaggiungendo la serie.

**Etichette LaTeX sbagliate.** Se un'etichetta non compila, la barra sopra l'anteprima dice quale
(es. «etichetta y del pannello 2: … → ! Missing { inserted.») invece di un errore generico.

## Stile

Computer Modern (o Latin Modern, Times, Palatino, Helvetica) composto da `pdflatex`, con `amsmath`
e `siunitx`: le unità si scrivono `[\si{\meter\per\second}]`. Tacche verso l'interno su tutti e
quattro i lati, tacche secondarie, griglia leggera punteggiata, legenda con cornice sottile. Palette
Okabe-Ito di default (leggibile anche da chi ha deficit cromatici); c'è anche la scala di grigi per
la stampa in bianco e nero.

**Dimensione esatta.** La figura si esporta esattamente alla larghezza e altezza impostate e i font
sono alla dimensione scelta. In LaTeX va inclusa **senza** `width=…`:

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics{figure/pitch.pdf}
  \caption{…}
  \label{fig:pitch}
\end{figure}
```

Il PDF contiene i font Type 1 incorporati (verificato con `pdffonts`). Per la tesi il formato
consigliato è il **PDF**; l'SVG serve per presentazioni o per ritoccare a mano (in LaTeX
richiederebbe il pacchetto `svg` e Inkscape).

**Anteprima.** È la stessa figura che viene esportata. Il primo rendering con LaTeX richiede qualche
secondo, poi matplotlib usa una cache. *Anteprima rapida (senza LaTeX)* usa il motore interno di
matplotlib: istantaneo, molto vicino al risultato finale, utile mentre si sistemano i dati.

## Riprodurre le figure

Ogni figura salvata come `.json` si rigenera senza interfaccia, per esempio dopo aver cambiato i
dati:

```bash
../../../venv_sim/bin/python plotcore.py figure/pitch.json figure/pitch.pdf
```

## File

- `thesis_plot.py` — interfaccia (tkinter)
- `plotcore.py` — configurazione, dati, stile, rendering ed export; non dipende dall'interfaccia
