# Replay in simulazione degli esperimenti sul robot reale

```bash
cd ~/unitree_ws/Squat_Unitree_H1_2/tools
../../venv_sim/bin/python replay_experiment.py ~/unitree_ws/sim_logs/FINALE/exp_2026-09-22_14-45-16
../../venv_sim/bin/python replay_experiment.py ~/unitree_ws/sim_logs/FINALE/exp_*      # tutti
```

Per ogni esperimento crea `sim_logs/sim_<ts>_replay_<exp>/` con gli stessi file di ogni sim2sim
(`obs.csv`, `commands.csv`, `errors.csv`, `ground_truth.csv`, `plots/`) più:

| file | contenuto |
|---|---|
| `replay.yaml` | la timeline del joystick ricostruita dall'esperimento |
| `source_run_info.md` | il `run_info.md` dell'esperimento reale |
| `replay_fidelity.md` | quanto i comandi visti dalla policy in sim coincidono con quelli del reale |

La policy è quella scritta nel `run_info.md` dell'esperimento (`--policy` per cambiarla). Ci vuole
la durata dell'esperimento più ~15 s; serve il display (si apre la finestra di MuJoCo).

## Metodo

In simulazione gira **lo stesso controller** del robot (stesso codice, policy, kp/kd, `deploy.yaml`):
cambia solo la sorgente del joystick, che invece del radiocomando è il joystick scriptato del
simulatore. La sua timeline si ricostruisce da `obs.csv` dell'esperimento, che registra esattamente
i comandi visti dalla policy a ogni passo.

- **Velocità.** Nel controller passano da un filtro EMA (`x ← 0.97x + 0.03u` a 1 kHz) prima di
  essere registrate. Il replay lo pre-compensa: per il passo k manda
  `u_k = (y_k − 0.97²⁰·y_{k−1}) / (1 − 0.97²⁰)`, così dopo i 20 aggiornamenti del filtro fra due passi
  l'uscita è esattamente il valore registrato. Il filtro lato simulatore è disattivato
  (`stick_smoothing: false`), perché sul robot il pacchetto del radiocomando arriva senza filtri.
- **Quota.** Il controller la integra di 1 mm per passo mentre `RT` (scende) o `B` (sale) è premuto.
  La timeline riporta la quota registrata passo per passo e il joystick tiene `RT`/`B` esattamente
  per quei passi; le voci di discesa sono anticipate dei ~23 ms che il filtro di `RT` impiega a
  superare la soglia nel controller.
- **Orologio.** Il controller scandisce i passi sull'orologio di parete, la simulazione gira a
  rtf ≈ 0.9995: su 270 s sarebbero 7 passi di deriva. Dopo l'ingresso in RL il replay va
  sull'orologio di parete (`clock: wall_after_rl`).
- **Avvio.** FixStand, poi la corda viene rilasciata 0.3 s prima dell'ingresso in RL: sul reale la
  fune era lasca e la policy parte con il robot interamente sui piedi. Prima non si può: in sim la
  posa FixStand senza corda crolla in ~1.5 s.

Verificato prima con un emulatore della catena completa (joystick → pacchetto → filtri del
controller → passi di policy): quota riprodotta esattamente su 42 000 passi, velocità con RMS ≤ 0.0004.

## Fedeltà sui quattro esperimenti FINALE (2026-09-22, policy R9)

| esperimento | sfasamento | RMS vx / vy / wz | max | quota RMS / max |
|---|---|---|---|---|
| `14-45-16` | 0 passi | 0.0002 / 0 / 0.0005 | 0.019 rad/s | 0 / 0 mm |
| `14-52-57` | 0 passi | 0.0001 / 0.0002 / 0.0002 | 0.012 rad/s | 0.6 / 1.0 mm |
| `15-01-32` | 0 passi | 0.0000 / 0.0001 / 0.0002 | 0.012 rad/s | 0 / 0 mm |
| `15-08-45` | — | nessun comando | — | — |

I massimi stanno sulle transizioni brusche della levetta (un passo); il singolo millimetro di quota
in `14-52-57` è un passo di rampa perso per la latenza DDS.

## Limiti da dichiarare

- Si riproducono i **comandi dell'operatore**, non le perturbazioni esterne. `15-08-45` sul reale ha
  ~12 spinte a mano (picchi del giroscopio fino a 1.2 rad/s): non sono misurate e nel replay non ci
  sono — quel replay è solo 212 s di stazione eretta.
- **Stato iniziale**: posizioni dei giunti al primo passo entro 0.6–0.9° in media; pitch del busto
  in sim +1.2° contro −1.6 ÷ +0.1° sul reale (è l'offset IMU/baricentro sim→reale già noto).
- La sim registra ~10 s in più del reale alla fine (stazione eretta dopo l'ultimo comando): per i
  grafici allineare su `ep_t` e tagliare alla durata del reale.
