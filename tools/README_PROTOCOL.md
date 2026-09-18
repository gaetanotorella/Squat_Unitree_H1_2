# Procedura di test sim2sim automatizzata

Sostituisce la sessione manuale (avvia controller → avvia MuJoCo → `LT+su` → `RB+X` → tasto `9` →
joystick a mano → chiudi → `analyze_sim.py`) con un'unica invocazione che comanda **sempre le
stesse cose negli stessi istanti**. Serve a rendere confrontabili policy addestrate in momenti
diversi: finché il joystick lo muove una persona, metà delle differenze fra due run sono la
persona.

```bash
cd ~/unitree_ws/Squat_Unitree_H1_2/tools
../../venv_sim/bin/python run_protocol.py --label round7
```

## Come funziona

Il controller legge il joystick da `lowstate->joystick`, cioè dal campo `wireless_remote` del
messaggio **LowState, che pubblica il simulatore**. Il topic `WirelessController` esiste ma il
controller non lo usa. Quindi un publisher DDS esterno non può iniettare comandi: l'unico punto
di innesto è dentro MuJoCo. (L'altra strada — un gamepad virtuale via `uinput` — richiede root su
`/dev/uinput`, quindi è esclusa.)

`unitree_mujoco/simulate/src/scripted_joystick.h` aggiunge un backend `joystick_type: "script"`
che riproduce un protocollo YAML **contro il tempo di simulazione**, non contro l'orologio di
parete: due esecuzioni su macchine diverse comandano le stesse cose agli stessi istanti fisici, e
il tempo di simulazione è anche l'asse di `ground_truth.csv`.

`run_protocol.py` fa il resto: fissa la policy sotto test, lancia i due processi, aspetta, li
chiude, e produce metriche e grafici.

## File prodotti nel run

| file | contenuto |
|---|---|
| `protocol.yaml` | copia esatta del protocollo usato — il run è autocontenuto |
| `run_meta.json` | policy, commit, se l'albero era sporco, real-time factor |
| `metrics.json` | le metriche fisse, per `compare_runs.py` |
| `report.md` | le stesse metriche in forma leggibile |
| `plots/` | i grafici di `analyze_sim.py` |

## Confronto fra run

```bash
../../venv_sim/bin/python compare_runs.py ~/unitree_ws/sim_logs/sim_A ~/unitree_ws/sim_logs/sim_B
```

Confronta segmento per segmento e chiude con l'elenco dei **peggioramenti oltre soglia**. Si
rifiuta di confrontare run di protocolli diversi.

## I protocolli

`protocols/standard_v2.yaml` è quello da usare. `standard_v1.yaml` è conservato **immutato** perché
i cinque run di confronto del 2026-09-13 (`sim_logs/automatic/`) lo usano, e `compare_runs.py`
rifiuta di confrontare protocolli diversi.

**Perché esiste v2 — un difetto del banco, non del robot.** Nei cinque run di confronto del
2026-09-13 le quote intermedie arrivavano più alte del richiesto (0.85 → 0.860, 0.74 → 0.761,
0.66 → 0.692, **0.60 → 0.641**), con l'errore che si accumulava a ogni rampa. Avevo attribuito la
cosa al ciclo del controller "a ~46 Hz": **era sbagliato**. La causa vera, trovata il 2026-09-18:
nel joystick scriptato `RT` veniva scritto due volte per ciclo (`RT(0)` "tutto rilasciato", poi
`RT(1)`), e `RT` è un `Axis` con filtro EMA — due aggiornamenti per ciclo portano il punto fisso a
0.508, esattamente sulla soglia 0.5 con cui il controller decide `R2.pressed`. Il bit sfarfallava
per arrotondamento: 8% di perdita in quei run, 67% in un altro. Ora ogni tasto è scritto **una
volta per ciclo** e il bit è stabile; il modello di quota del joystick coincide col controller al
millimetro. L'`overhold` e la ri-ancoratura a 0.98 di v2 restano: innocui, e proteggono da
qualsiasi altra deriva futura.

Identico in tutti i run, quindi il confronto fra le policy del 09-13 resta valido fra loro. Le
metriche misurano comunque contro il **target loggato dal controller** (`base_height_command_0`,
già in `obs.csv`), non contro il valore del protocollo, e riportano entrambe le quote nel report
(colonne `h prot` e `h ctrl`).

**Le rampe del controller sono nel protocollo.** `descend_rate` / `rise_rate` (oggi 0.05 m/s,
commit `7e3bb51`) devono coincidere con `observations.h` del controller: il joystick scriptato le
usa per sapere quanto tenere RT/B. Se il controller cambia, si cambiano lì, altrimenti le quote
arrivano sbagliate e il check "scarto del banco" lo segnala.

## Struttura del protocollo

`protocols/standard_v1.yaml`. Stato **appiccicoso**: `vx/vy/wz/height` valgono finché non li
cambia una voce successiva; `do` è invece un impulso. Le ampiezze sono a fondo corsa dei range di
training, *prima* dello scaling per profondità che applica il controller.

Struttura: A) fermo in piedi · B) i sei assi in piedi · C) scala di quote da fermo, con 22 s a
fondo squat · D) i sei assi a 0.74 m · E) ritorno e fermo finale. Durata 262 s.

**Il protocollo non si modifica per adattarlo a una policy.** Se serve una manovra nuova si crea
`standard_v2.yaml`: i confronti valgono solo fra run dello stesso protocollo, e `compare_runs.py`
lo impone.

## Controlli di validità

`report_metrics.py` invalida il run se:

- il **real-time factor** esce da [0.97, 1.03]. Il joystick scriptato integra la quota sul tempo
  di simulazione, il controller sul suo orologio a 50 Hz: se MuJoCo rallenta i due divergono e le
  quote comandate non sono più quelle scritte nel protocollo;
- l'**inseguimento di quota** del robot supera 1 cm rispetto al target del controller (era 5 cm
  contro il valore del protocollo: a quella soglia uno scarto di 4.1 cm passava in silenzio);
- il **target del controller** differisce da quello chiesto dal protocollo di oltre 1 cm. Non
  invalida il confronto fra policy, ma va visto; con v2 resta sotto il centimetro.

## Vincoli da non rompere

- Le rampe `kDescendRate` / `kRiseRate` in `scripted_joystick.h` devono restare uguali a
  `descend_rate` / `rise_rate` in `deploy/include/isaaclab/envs/mdp/observations/observations.h`.
  Il joystick scriptato tiene RT/B finché il **suo modello** del target raggiunge la quota: se le
  due rampe divergono, il controllo sulla quota misurata lo segnala.
- Il joystick fisico azzera tutto sotto 0.15 (deadzone in `physics_joystick.h`, messa lì per un
  difetto del pad). Quello scriptato **non** la applica — iniettare valori esatti è meglio — ma
  avvisa al caricamento se il protocollo usa un valore in quella fascia, perché quell'istante non
  sarebbe riproducibile a mano.
## Separazione dal percorso manuale e dal robot reale

I due metodi sono **completamente separati**: `run_protocol.py` non tocca nessun file di
produzione. Genera due config usa-e-getta in `/tmp` e li passa con `--config`, opzione aggiunta a
entrambi i binari (default invariato: il file accanto all'eseguibile).

Questo perche' il config del controller, `deploy/robots/h1_2_squat/config/config.yaml`, e' **lo
stesso file usato con `--network <eth>` sul robot reale**. Una versione precedente di questo
strumento lo riscriveva e lo ripristinava alla fine: un test interrotto a forza lo avrebbe
lasciato puntato alla policy sotto test, e il giro successivo sul robot vero avrebbe caricato
quella. Ora quel rischio non esiste.

| | manuale | automatico |
|---|---|---|
| simulatore | `./unitree_mujoco` | `--config <tmp>/sim_config.yaml` |
| controller | `./h1_2_ctrl_squat --network lo` | `--config <tmp>/ctrl_config.yaml` |
| joystick | fisico, `/dev/input/js0` | scriptato, dal protocollo |
| policy | la piu' recente | fissata esplicitamente |
| file di produzione | — | **mai toccati** |

Il percorso manuale e quello sul robot reale funzionano esattamente come prima, senza rimettere a
posto nulla. L'unica modifica al binario del controller e' l'aggiunta dell'opzione `--config`:
senza di essa il comportamento e' identico riga per riga a prima.
