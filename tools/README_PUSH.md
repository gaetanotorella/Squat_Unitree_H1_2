# Spinte manuali (test di robustezza)

Durante una normale sessione sim2sim col joystick fisico, premi un tasto nella finestra di MuJoCo
e il simulatore applica una forza o una coppia a un corpo del robot. Nessun legame col banco
automatico: è per provare a mano la robustezza della policy.

## Uso

1. In `unitree_mujoco/simulate/config.yaml`: `disturbance_script: "disturbances.yaml"` (già
   impostato; `""` per disattivare).
2. I preset sono in `unitree_mujoco/simulate/disturbances.yaml`. All'avvio il simulatore stampa la
   lista dei tasti.
3. Sessione come sempre (controller, simulatore, `LT+su`, `RB+X`, tasto `9`), poi con la finestra
   MuJoCo **in primo piano** premi il tasto: la spinta parte subito, nel terminale compare
   `[spinta] tasto 1 a t=... s`.

Preset iniziali: `1`–`4` 20 N·s avanti/indietro/sinistra/destra sul torso, `5`/`6` 30 N·s
avanti/indietro, `Q`/`W` coppia di imbardata ±30 N·m, `E` spinta continua 60 N per 2 s.
`7`, `8`, `9` sono della corda elastica: non usarli.

## Formato dei preset

```yaml
presets:
  - {key: "1", body: torso_link, force: [100, 0, 0], dur: 0.2}                    # 20 N·s avanti
  - {key: "Q", body: pelvis,     torque: [0, 0, 30], dur: 0.2}                    # coppia di imbardata
  - {key: "A", body: torso_link, force: [60, 0, 0],  dur: 0.2, point: [0, 0, 0.3]}  # 30 cm sopra l'origine
  - {key: "S", body: torso_link, force: [60, 0, 0],  dur: 0.2, frame: world}      # assi del mondo
```

`force` in N, `torque` in N·m, `dur` in secondi (`0` = un solo passo fisico). `frame`: `body`
(default — "avanti" è l'avanti del robot anche se ha girato) o `world`. `point`: metri nel
riferimento del corpo, default baricentro; un punto diverso aggiunge la coppia equivalente
`(P − CoM) × F`. Corpi utili: `pelvis`, `torso_link`, `*_shoulder_pitch_link`, `*_knee_link`.

20 N·s su 79.5 kg = 0.25 m/s di variazione di velocità del baricentro.

## Analisi

Ogni spinta è loggata in `sim_logs/sim_<ts>/disturbance_events.csv`. Dopo la sessione:

```bash
../../venv_sim/bin/python report_pushes.py ~/unitree_ws/sim_logs/sim_<ts>
```

Per ogni spinta: Δv di picco del baricentro contro quella attesa dall'impulso (conferma che la
forza è arrivata), picco di pitch/roll, spostamento massimo e finale, imbardata, tempo di
assestamento (|Δv| < 0.05 m/s e |Δtilt| < 1° per 0.5 s), caduta.

## Come funziona (per chi tocca il codice)

`unitree_mujoco/simulate/src/push_disturbance.h`. `mjData::xfrc_applied[6*body]` = forza + coppia
al baricentro, nel mondo. `Simulate::Sync()` azzera **tutto** `xfrc_applied` a ogni frame di
rendering per pulire le perturbazioni del mouse: le spinte vanno riscritte in valore assoluto a
ogni passo fisico, come fa la corda elastica (se il corpo è lo stesso, il valore della corda viene
conservato e la spinta gli si somma). Il tasto arriva dal thread della finestra, la fisica gira in
un altro: la coda degli eventi è protetta da un mutex. Verificato con il modello vero e
l'azzeramento di `Sync()` simulato: 100 passi con forza, 19.96 N·s integrati su 20 attesi, zero
dopo.
