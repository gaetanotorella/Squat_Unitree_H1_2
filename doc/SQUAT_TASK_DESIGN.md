# Squat Task — Design Document (H1_2)

Task RL per far **camminare, stare in piedi e accovacciarsi (squat)** l'Unitree H1_2, costruito
sopra il task `velocity` già validato. Documento di riferimento per la tesi.

> Stato: **scheletro implementato**, da smoke-testare su macchina Isaac e da tunare. Le reward/pesi
> sono punti di partenza, non valori finali.

---

## 1. Obiettivo e scelte di progetto

- **Capacità target:** walk + stand + **squat sul posto** (a comando di velocità ~0). Lo squat-mentre-cammina
  è previsto come estensione (Fase B), non come primo obiettivo.
- **From-scratch** (non warm-start da `model_9000`): riproducibilità totale, nessuna dipendenza da un `.pt`
  esterno per rigenerare la policy.
- **Walk+squat insieme** (non "solo standing+squat"): la locomozione dà l'equilibrio che rende robusto lo
  squat. Il vecchio approccio "solo standing+squat" era fragile (instabilità caviglie).
- **Curriculum a stadi:** il robot impara **prima** a camminare/stare in piedi (altezza nominale), **poi**
  ad abbassarsi; la profondità dello squat cresce solo quando la locomozione è padroneggiata.
- **Controllore di deploy separato** `h1_2_ctrl_squat`, così il controllore `velocity` resta intatto.

| Config / env | Controllore deploy |
|---|---|
| standing + cammino | `velocity_env_cfg.py` → `h1_2_ctrl` |
| standing + cammino + **squat** | `squat_env_cfg.py` → **`h1_2_ctrl_squat`** |

---

## 2. Actor-Critic asimmetrico (importante per lo squat)

Il template è **asimmetrico (critic privilegiato)**:
- **Actor** (deployabile, con rumore): solo osservazioni misurabili sul robot reale.
- **Critic** (solo training, senza rumore): in più `base_lin_vel`, `joint_effort` e — **nuovo per lo squat** —
  l'**altezza vera della base**.

Conseguenza sfruttata qui: la reward e il critic possono usare l'**altezza vera** (`root_pos_w[:,2]`),
mentre l'**actor NON la osserva** (sul robot reale non c'è un sensore di altezza assoluta) e la inferisce
via FK dai `joint_pos`.

---

## 3. Osservazioni

Actor squat = actor velocity (92) **+ `base_height_command` (1) = 93**. Il gait clock resta (gated, Option C)
perché il robot cammina ancora.

| Obs (actor) | Fonte | Note sim2real |
|---|---|---|
| base_ang_vel, projected_gravity, velocity_commands, joint_pos_rel, joint_vel_rel, last_action, gait_phase | ereditate | invariate |
| **base_height_command** | `mdp.base_height_command_rel` | target − nominale; comando operatore, **riproducibile** |

Critic squat = critic velocity **+ base_height_command + base_height (vera, privilegiata)**.

> `base_height_command_rel` restituisce `target − 0.98` (centrato su 0: 0 = in piedi, negativo = squat).
> `base_height_rel` (solo critic) restituisce `altezza_vera − 0.98`.

**NON** aggiungere l'altezza vera all'actor: romperebbe il sim2real.

---

## 4. Comando di altezza + curriculum

Nuovo command term `UniformHeightCommand` (`mdp/commands/height_command.py`):
- campiona un'altezza target assoluta in `ranges.height`;
- **`only_when_standing=True`**: la profondità di squat è campionata solo sugli env fermi
  (`is_standing_env` del comando di velocità); gli env in cammino restano a `nominal_height` → squat a
  velocità ~0 (Fase A). Metti `False` per la Fase B (squat+cammino);
- `resampling_time_range=(5,8)s` → alcuni cicli squat per episodio.

**Curriculum staged** (`mdp.height_cmd_levels`): abbassa `ranges.height[0]` di `step=0.05`
verso `limit_ranges.height[0]` (0.55 m) **solo quando**:
- `track_lin_vel_xy` > 0.8·peso (**locomozione appresa**), **e**
- `track_base_height` > 0.8·peso (altezza corrente inseguita).

Partenza: `ranges.height = (0.98, 0.98)` → nessuno squat finché il robot non cammina bene. Questo realizza
esattamente "prima cammina/stai in piedi, poi squat". Il curriculum velocity/terreno resta attivo in parallelo.

---

## 5. Reward

Ereditate tutte quelle del task velocity (regolarizzazione, orientamento, piedi, gait gated), con queste
**modifiche/aggiunte** (`SquatRewardsCfg`). Pesi = punti di partenza.

| Termine | Funzione | Peso | Scopo |
|---|---|---:|---|
| **track_base_height** | `track_base_height_exp` (std 0.1) | **+3.0** | task principale: insegue l'altezza comandata |
| **settle_height** | `lin_vel_z_gated` (tol 0.05) | −2.0 | penalizza vz **solo vicino al target** (non combatte la discesa) |
| **squat_symmetry** | `joint_mirror` (hip_pitch, knee, ankle_pitch) | −0.5 | simmetria L/R della discesa |
| **double_support** | `feet_contact_without_cmd` | +0.5 | entrambi i piedi a terra da fermo |
| **feet_too_near** | `feet_too_near` (0.2 m) | −2.0 | evita che i piedi collassino insieme |
| ~~base_height~~ | (disabilitata) | — | sostituita da track_base_height (target variabile) |
| ~~base_linear_velocity~~ (lin_vel_z_l2) | (disabilitata) | — | penalizzava vz sempre → in conflitto con la discesa; sostituita da settle_height |

Anti-spaccata: la deviazione di `hip_roll/hip_yaw` da 0 è già coperta da `joint_deviation_hips` (ereditata);
`squat_symmetry` copre la simmetria della discesa (pitch/knee/ankle_pitch).

**Regola di sicurezza (lezioni imparate):** la reward netta per-step di un robot *vivo e in piedi* deve
restare **positiva**, altrimenti la policy impara a "suicidarsi" (terminare subito). Tenere `alive` + task
sopra le penalità; non alzare troppo penalità che esplodono a inizio training (es. action_rate). Vedi §8.

---

## 6. Deploy — `h1_2_ctrl_squat` (IMPLEMENTATO)

La pipeline obs del deploy è **data-driven** dal `deploy.yaml` esportato: il C++ chiama la
`REGISTER_OBSERVATION(<nome-attributo-obs>)` corrispondente.

**Controllore creato:** `deploy/robots/h1_2_squat/` (copia di `h1_2/`) con:
- `config/config.yaml`: stato **`Squat`** (type `RLBase`), `policy_dir: ../../../logs/rsl_rl/unitree_h1_2_squat`,
  transizioni **`RB + X`** (FixStand → Squat) e **`LT + B`** (Squat → Passive).
- `CMakeLists.txt`: target eseguibile **`h1_2_ctrl_squat`** (project `h1_2_squat_controller`).

**Comando altezza (RT + B).** RT/R2 sul pad Unitree è **digitale** (bit R2, non analogico), quindi:
- **Tieni `RT` (R2)** → il robot **scende** progressivamente (rampa `descend_rate` 0.30 m/s; più tieni, più scende).
- **Tieni `B`** → **si rialza** progressivamente (rampa `rise_rate` 0.15 m/s, più lenta della discesa: evita
  l'overshoot/ringing del nominale che dava barcollio alla risalita).
- Stato dell'altezza in `env->squat_target` (aggiunto a `manager_based_rl_env.h`), letto/scritto in
  `REGISTER_OBSERVATION(base_height_command)` (`observations.h`); output = `target − nominale`, identico al
  training (`mdp.base_height_command_rel`). Innocua per il controllore velocity (che non elenca l'obs).
- Gli stick continuano a **camminare/girare**; lo squat si applica a velocità ~0 (`only_when_standing`).
- Tarabili: `descend_rate` (0.30 m/s), `rise_rate` (0.15 m/s). Verificare che `RT.pressed`/`B.pressed`
  diano il verso atteso.

**Fix export (necessario):** `export_deploy_cfg.py` serializzava solo `base_velocity`; ora serializza anche
`commands.base_height` (con `limit_ranges` = 0.55–0.98). Inoltre `play.py` **rigenera** il `deploy.yaml`
dopo l'export. ⇒ **ri-lanciare `play` su `model_3500`** per ottenere un `deploy.yaml` con
`commands.base_height` (senza, il C++ che legge `cfg["commands"]["base_height"]["ranges"]["height"]` fallisce).

**Cartelle di log (sim vs robot reale).** Il controllore decide da `--network`:

| Contesto | `--network` | Cartella creata sotto `unitree_ws/sim_logs/` | `ground_truth.csv` |
|---|---|---|---|
| sim2sim MuJoCo | `lo` | `sim_<ts>/` (creata dal simulatore, agganciata dal controllore) | sì |
| **test robot reale** | `eth0`, `enp3s0`, … | **`exp_<ts>/`** (creata dal controllore) | **no** |
| controllore senza simulatore | `lo` | `ctrl_only_<ts>/` | no |

**Tracciabilità della policy.** Ogni run scrive anche un **`run_info.md`** nella propria cartella, con
la policy effettivamente caricata e il contesto:

```markdown
- **policy_dir**: .../logs/rsl_rl/unitree_h1_2_squat/2026-09-01_19-20-07
- **policy_name**: 2026-09-01_19-20-07
- **controller**: h1_2_ctrl_squat
- **network**: enp3s0  -> ROBOT REALE
- **n_obs**: 93     - **base_height_range**: [0.55, 0.98]
```

Serve perché il controllore, quando `policy_dir` in `config.yaml` non è esplicita, **sceglie da solo**
la run più recente con `exported/`: senza questo file, dato un log non c'era modo di risalire alla
policy che l'ha prodotto. Il percorso è preso da `param::policy_dir`, valorizzato in
`parser_policy_dir()`, quindi vale per **tutti** i controllori. `tools/analyze_sim.py` lo legge e lo
stampa in testa a ogni analisi (e avvisa se manca, come nei log precedenti a questa modifica).

Il simulatore viene riconosciuto dalla presenza di `ground_truth.csv` **scritto da meno di 120 s**
(si guarda la mtime del file, non della cartella: appendere non aggiorna la mtime della directory).
Prima il controllore agganciava semplicemente la `sim_*` più recente e **sovrascriveva i CSV di un run
vecchio** se lanciato senza far ripartire il simulatore.

`tools/analyze_sim.py` riconosce entrambi i prefissi e, sui run `exp_*`, **omette automaticamente**
tutto ciò che deriva dal ground truth (posizione XYZ, yaw). Al suo posto usa solo grandezze
ricostruibili a bordo:

| Grandezza | Come si ottiene sul robot reale | Accuratezza (vs ground truth in sim) |
|---|---|---|
| roll / pitch | da `projected_gravity` (IMU) | RMS **0.06° / 0.03°** |
| **altezza della base** | **cinematica diretta**: modello MuJoCo nella posa misurata (encoder) + orientazione IMU, si prende il punto più basso dei due piedi e si assume appoggiato a terra | bias **+0.33 cm**, std **0.23 cm**, corr **0.9995** (fermo: std 0.07–0.12 cm) |
| yaw, posizione XYZ | **non ricostruibili** (servirebbe motion capture / state estimator) | — |

La stima di altezza è la stessa informazione da cui l'actor inferisce la propria altezza (FK dai
`joint_pos`), quindi è coerente col design del task. **Ipotesi: suolo piatto e almeno un piede in
appoggio** — degrada su terreno inclinato o irregolare. Richiede `mujoco` e il MJCF di H1_2
(cercato in `$H1_2_MJCF`, poi in `~/unitree_ws/unitree_mujoco/...`); se mancano, il plot mostra
solo l'altezza comandata. Nei run in simulazione la stessa stima è sovrapposta come **controprova**
tratteggiata accanto al ground truth.

**Build + avvio (PC deploy):**
```bash
cd deploy/robots/h1_2_squat && mkdir -p build && cd build && cmake .. && make
./h1_2_ctrl_squat --network lo        # 'lo' = loopback per sim2sim; interfaccia reale per sim2real
```
Uso: `[L2 + Up]` → FixStand, poi `[R1 + X]` → Squat; quindi **tieni R2** per accovacciarti (si ferma a `cutoff_height` + 1 cm = 0.71 m, la quota minima di camminata), **A + R2** per scendere oltre fino a h_min, **B** per rialzarti.

> ⚠️ Il C++ non è stato compilato in fase di design: se `make` dà errori (candidati: accessi
> `joystick->RT.pressed` / `joystick->B.pressed`), vanno risolti sul PC deploy.

---

## 7. Come si esegue

Training (macchina Isaac):
```bash
./unitree_rl_lab.sh -t --task Unitree-H1_2-Squat --headless --max_iterations 8000
```
I log vanno in `logs/rsl_rl/unitree_h1_2_squat/` (separati dal velocity, automatico dal nome task).

Play / export di uno specifico checkpoint:
```bash
./unitree_rl_lab.sh -p --task Unitree-H1_2-Squat \
  --checkpoint logs/rsl_rl/unitree_h1_2_squat/<run>/model_<iter>.pt --num_envs 32
```

Segnali sani da sorvegliare (prime ~300 iter): `mean_episode_length` cresce, `base_contact` scende,
`Curriculum/terrain_levels` e `lin_vel_cmd_levels` salgono. La **profondità squat** (`height_cmd_levels`,
minimo altezza) inizia a scendere **solo dopo** che la locomozione è appresa.

---

## 8. Rischi noti e punti di tuning (dalle lezioni del task velocity)

- **Collasso "suicida"**: se le penalità superano `alive`+task a inizio training, la policy termina subito
  (ep_len→minimo, base_contact→100%). Bilanciare i pesi; introdurre lo squat via curriculum (già fatto).
- **Deadlock curriculum**: non gatare penalità su `cmd<soglia` che si sovrappone al range iniziale del
  curriculum. Qui il curriculum altezza parte da (0.98,0.98) e scende solo a locomozione appresa → nessun
  deadlock.
- **Over-training**: i finetune lunghi possono derivare su un optimum peggiore; usare checkpoint al picco.
- **Segno joystick altezza**: da verificare sul pad reale (commento nel C++).
- **`std` di track_base_height (0.1)** e i pesi: da tunare dopo il primo run.
- **`only_when_standing`**: accoppia il comando altezza con `is_standing_env` del comando velocità; se in
  futuro serve squat+cammino (Fase B), metterlo `False`.

---

## 9. File toccati

Training:
- `mdp/commands/height_command.py` (nuovo) — `UniformHeightCommand(Cfg)`
- `mdp/commands/__init__.py` — export
- `mdp/observations.py` — `base_height_command_rel`, `base_height_rel`
- `mdp/rewards.py` — `track_base_height_exp`, `lin_vel_z_gated`
- `mdp/curriculums.py` — `height_cmd_levels`
- `robots/h1_2/squat_env_cfg.py` (nuovo) — `SquatEnvCfg` / `SquatPlayEnvCfg`
- `robots/h1_2/__init__.py` — registrazione `Unitree-H1_2-Squat`

Deploy:
- `deploy/include/isaaclab/envs/mdp/observations/observations.h` — `REGISTER_OBSERVATION(base_height_command)`
- `deploy/robots/h1_2_squat/` — **da creare** sulla macchina deploy (§6)

Building block riusati (già in `mdp/rewards.py`): `joint_mirror`, `feet_contact_without_cmd`, `feet_too_near`.
