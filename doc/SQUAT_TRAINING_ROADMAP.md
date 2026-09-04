# Squat Task — Roadmap modifiche di training

Obiettivo: policy **robusta e non oscillante**, con **cammino fluido**, **squat fluido in discesa e
risalita**, e **comando di yaw (giro) funzionante**. Consolida le proposte precedenti + la diagnosi
dei run sim2sim `sim_2026-08-29_01-10-54` e `sim_2026-08-31_13-08-05`.

> Principio guida: **nessuna nuova osservazione dell'actor** → il `deploy.yaml` e il controllore
> `h1_2_ctrl_squat` restano invariati (tutte le modifiche sono su reward / curriculum / pesi / attuatori
> di training). Vale sempre la **regola anti-suicidio** (reward netta per-step di un robot vivo > 0) e il
> **curriculum a stadi** (prima cammina/stai in piedi, poi squat).

> **STATO: modifiche P1–P4 IMPLEMENTATE il 2026-08-31.** Retrain consigliato **DA ZERO** (non finetune):
> `model_3500` ha convergito *contro* i due fix chiave — ha imparato un'andatura **che non gira**
> (ang_vel_z congelato a ±0.1 per tutto il run) e il **ciclo-limite di caviglia** come strategia di
> equilibrio; un finetune dovrebbe *disimparare* abitudini radicate (lento/instabile) e al resume il
> curriculum si riazzera comunque (mismatch policy-esperta / comandi-facili). Il curriculum va plasmato
> dall'inizializzazione → **from-scratch, 5000 iter.**

---

## 1. Diagnosi sintetica (problema → causa → prova)

| # | Sintomo | Causa-radice | Prova (dati) |
|---|---|---|---|
| A | **Barcollio da fermo** (beccheggio ~0.8 Hz) | La policy **comanda un target caviglia oscillante** (±0.23 rad) che il giunto non segue (kp=20); la coppia modulata fa beccheggiare il corpo → **ciclo-limite ad anello chiuso**. Invisibile ai penalizzatori di *posizione* (il giunto è quasi fermo, std 0.01). | `12_ankle_decomp.png`: target std 0.23 vs misura std 0.010; `wy` a 0.83 Hz in fase con la caviglia |
| B | **Yaw ignorato** (giro non risponde) | Il curriculum `lin_vel_cmd_levels` espande **solo** `lin_vel_x/y`, **mai** `ang_vel_z` → il giro resta a ±0.1 rad/s in training, ma il deploy comanda ±0.5 → out-of-distribution. | corr(cmd,reale)=−0.17; `track_ang_vel_z` a fine training solo 17% del max |
| C | **Squat undershoot ~4 cm** in profondità | `track_base_height` con `std=0.1` troppo largo (poco gradiente sotto 5 cm) **+** `stand_still_zero_cmd` (−0.5) che frena lo scostamento dalla default proprio negli env che squattano. | target 0.55 → `base_z` si ferma ~0.59 |
| D | **Cammino "chattering" + err giunti alto** | In parte fisiologico: passo dinamico a 1.25 Hz (anti-fase L/R) + impatti al contatto + kp caviglia basso. Non è rumore HF (potenza >3 Hz solo 5–24%). | `10_zoom_walk.png`; f_picco err = 1.15–1.30 Hz = gait 1.25 Hz |
| E | **Asimmetria L/R ~5°** (hip_pitch, ankle_pitch) → deriva yaw lenta | Simmetria non imposta abbastanza in cammino. | asym hip_pitch +5.1°, ankle_pitch −5.6°; yaw −69° in 75 s non comandato |

**Conflitto chiave (lega A e C):** `stand_still_zero_cmd` penalizza `|q − q_default|` sugli
`is_standing_env`, ma con `only_when_standing=True` quegli stessi env ricevono il target di squat →
penalizza lo squat **e** non tocca il ciclo-limite (che è nell'azione). Va ridotto/ridisegnato.

---

## 2. Roadmap per priorità

### Priorità 1 — Eliminare il ciclo-limite della caviglia (barcollio) [Sintomo A]

**1a. Nuova reward: `joint_action_rate` mirata sulle caviglie, gated su `is_standing_env`.**
Colpisce la patologia dove vive davvero (nell'azione), senza toccare il passo (gate su fermo) né la
discesa lenta dello squat (che ha action-rate piccola).

`mdp/rewards.py` — nuova funzione:
```python
def joint_action_rate_standing(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Action-rate (prima differenza) su un sottoinsieme di giunti, SOLO sugli env fermi.

    Mirato al ciclo-limite di caviglia da fermo: vive nell'azione (il target oscilla) ma NON nella
    posizione misurata (giunto troppo morbido per seguirlo), quindi i penalizzatori di posizione lo
    mancano. Gate su is_standing_env -> non penalizza il passo; l'action-rate della discesa squat
    (lenta, rampata) e' piccola -> non ostacola lo squat.
    """
    idx = asset_cfg.joint_ids  # risolti da joint_names (ordine articolazione == ordine azione con joint_names=[".*"])
    da = env.action_manager.action[:, idx] - env.action_manager.prev_action[:, idx]
    is_standing = env.command_manager.get_term(command_name).is_standing_env
    return torch.sum(torch.square(da), dim=1) * is_standing
```

`squat_env_cfg.py` — in `SquatRewardsCfg`:
```python
ankle_action_smoothness = RewTerm(
    func=mdp.joint_action_rate_standing,
    weight=-0.02,  # DA TUNARE: partire piccolo, alzare finche' wy std da fermo < ~0.02 rad/s
    params={
        "command_name": "base_velocity",
        "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"]),
    },
)
```
> Esporre `joint_action_rate_standing` in `mdp/__init__.py` (o dal modulo rewards già importato come `mdp`).
> Se il barcollio resiste, estendere i `joint_names` a `.*_knee_joint` e/o alzare il peso a −0.03/−0.05.
> Rischio: peso troppo alto → caviglie "congelate" → equilibrio peggiore. Alzare gradualmente.

**1b. Ridurre il conflitto `stand_still_zero_cmd`.**
`squat_env_cfg.py`, `SquatRewardsCfg.__post_init__`:
```python
def __post_init__(self):
    self.base_height = None
    self.base_linear_velocity = None
    # stand_still penalizza |q-default| sugli env fermi = quelli che squattano -> frena lo squat e NON
    # tocca il ciclo-limite (che e' nell'azione). Ridotto: l'anti-barcollio ora e' 1a; standing-tall e'
    # gia' buono (sway 0.03, err altezza 0.35cm). Portare a 0 se lo squat resta poco profondo.
    self.stand_still_zero_cmd.weight = -0.1
```

**1c. (Opzionale, test rapido SENZA retrain)** alzare il `kd` delle caviglie nel deploy
`robots/h1_2_squat/config/config.yaml` (stato `Squat`/RLBase erediterà i kp/kd esportati): aumentare il
`kd` di `ankle_pitch/ankle_roll` smorza il beccheggio del corpo. Mitiga il **sintomo**, non la causa;
utile per confermare la diagnosi prima di rialleniare.

### Priorità 2 — Comando di yaw funzionante [Sintomo B]

**2a. Estendere `lin_vel_cmd_levels` per espandere anche `ang_vel_z`** (gated sul suo reward).
`mdp/curriculums.py`, dentro `lin_vel_cmd_levels`, aggiungere il blocco simmetrico:
```python
# --- espansione del comando di giro (yaw), gated sul suo tracking ---
ang_reward = torch.mean(env.reward_manager._episode_sums["track_ang_vel_z"][env_ids]) / env.max_episode_length_s
ang_w = env.reward_manager.get_term_cfg("track_ang_vel_z").weight
if env.common_step_counter % env.max_episode_length == 0 and ang_reward > ang_w * 0.8:
    delta = torch.tensor([-0.1, 0.1], device=env.device)
    ranges.ang_vel_z = torch.clamp(
        torch.tensor(ranges.ang_vel_z, device=env.device) + delta,
        limit_ranges.ang_vel_z[0], limit_ranges.ang_vel_z[1],
    ).tolist()
```
> Così il giro parte da ±0.1 e cresce fino a ±0.5 **solo** quando il robot lo insegue bene → in-distribution
> al deploy, nessun collasso da comando troppo grande all'inizio.

**2b. Alzare il peso di `track_ang_vel_z`** da `0.5` a `1.0` (pari a `track_lin_vel_xy`).
`velocity_env_cfg.py` riga ~241: dà alla policy incentivo sufficiente a imparare il giro.
> Alternativa più semplice ma meno pulita: partire con `ang_vel_z=(-0.3, 0.3)` in `CommandsCfg.ranges`
> (giro allenato da subito, senza curriculum dedicato). Consigliata la 2a+2b.

### Priorità 3 — Cammino fluido + simmetria [Sintomi D, E]

**3a. Rafforzare la simmetria L/R** (riduce i 5° e la deriva di yaw, rende il passo più regolare).
`squat_env_cfg.py`: alzare il peso di `squat_symmetry` (`joint_mirror`) da `-0.5` a `-1.0`. È già attivo
sempre (non gated), copre hip_pitch/knee/ankle_pitch (giunti a segno concorde → penalizza `(L−R)²`).
> Non aggiungere i giunti *roll* al mirror senza verificare la gestione del segno in `joint_mirror`
> (default L/R a segno opposto). Tenere il set pitch/knee/ankle_pitch.

**3b. (vx sottostima ~50%)** in gran parte gap sim2sim; **non** forzare con pesi aggressivi. Se serve,
piccolo aumento di `track_lin_vel_xy` (1.0 → 1.2) o semplicemente più iterazioni (il curriculum era a
0.9, tracking al picco 0.81). Priorità bassa.

### Priorità 4 — Profondità e fluidità dello squat [Sintomo C]

**4a. Stringere `track_base_height`**: `std` `0.1 → 0.07` (più gradiente sotto i 5 cm → arriva alla
profondità comandata). `squat_env_cfg.py` riga ~84. Tenere `weight=3.0` (eventuale 4.0 se serve).
**4b.** La `ankle_action_smoothness` (1a) e la simmetria (3a) rendono discesa/risalita più lisce.
La rampa di risalita è già nel deploy (`rise_rate` 0.15) — invariata.

---

## 3. Interazioni e rischi

- **1a vs squat**: il gate `is_standing_env` include gli env che squattano; la discesa è lenta (action-rate
  piccola) mentre il ciclo-limite è veloce → il peso colpisce più l'oscillazione che lo squat. Tunare dal
  basso e verificare la profondità squat (4a) dopo.
- **1b + 4a** insieme sbloccano la profondità: rimuovono/riducono la forza che frena lo scostamento dalla
  default e aumentano il richiamo verso il target.
- **2a**: gate su `track_ang_vel_z` con soglia `0.8*weight`; avendo alzato il peso a 1.0 (2b), la soglia è
  0.8 → il giro si espande solo a competenza raggiunta (nessun deadlock, come per lin_vel).
- **Anti-suicidio**: nessuna di queste penalità è illimitata sui 27 giunti; `1a` è ristretta a 2–4 giunti e
  gated su fermo. Sorvegliare comunque `ep_len` e `base_contact` nelle prime ~300 iter.
- **Deploy invariato**: nessuna nuova obs/azione → stesso `deploy.yaml`/controllore. Ricordarsi solo di
  **ri-lanciare `play`** sul checkpoint scelto per rigenerare `deploy.yaml` e **ricompilare** `h1_2_ctrl_squat`.

---

## 4. Protocollo di training e validazione

1. Applicare gli edit (ordine: P1 → P2 → P3 → P4; sono indipendenti ma conviene un unico retrain).
2. Smoke-test sintassi + primo avvio su Isaac (poche iter, `--num_envs` ridotto) per intercettare errori
   runtime (accesso `is_standing_env`, indici azione, chiavi reward nel curriculum).
3. Training completo: `./unitree_rl_lab.sh -t --task Unitree-H1_2-Squat --headless --max_iterations 5000`.
4. Segnali sani (TensorBoard):
   - `track_lin_vel_xy` sale e plateau ~0.8; **`track_ang_vel_z` ora deve salire** (>0.4) → giro appreso;
   - `Curriculum/lin_vel_cmd_levels` → ~1.0, **`ang_vel_z`** cresce, `height_cmd_levels` → ~0.55;
   - `base_contact` basso, `mean_std` non deve risalire → **usare il checkpoint al PICCO** (cap ~1000 iter
     dopo il plateau; niente over-training — lezione dei run precedenti).
5. Play + export sul checkpoint scelto (rigenera `deploy.yaml`), rebuild `h1_2_ctrl_squat`.
6. Validazione MuJoCo con `tools/analyze_sim.py`: verificare
   - `wy std` da fermo < ~0.02 rad/s (barcollio); nessun ciclo-limite 0.8 Hz in `11_zoom_stand`-like;
   - `corr(wz_cmd, wz_reale)` > 0.7 (giro); `corr(vx)` alto e modulo meno sottostimato;
   - err altezza squat < ~1.5 cm, discesa/risalita lisce.

---

## 4-bis. ROUND 2 — modifiche dopo la validazione sim2sim `sim_2026-09-03_20-49-35`

Il round 1 ha funzionato: yaw corr **−0.17 → +0.86**, sway da fermo **0.030 → 0.016**, errore altezza
squat **3.55 → 1.70 cm**, asimmetria posa **~dimezzata**, zero cadute. Restavano 5 punti, tutti applicati
il **2026-09-03**:

| # | Problema misurato | Modifica | File |
|---|---|---|---|
| 1 | Squat satura a **0.581 m**: `ankle_pitch` a fondo corsa (HW [−0.897,+0.524], soft −0.826); target comandato −1.00 = oltre il limite | `DEEPEST_SQUAT` 0.55 → **0.60** | `squat_env_cfg.py` |
| 2a | Yaw gira ma a **guadagno 0.40**; il range smetteva di espandersi (gate 0.8 vs tracking che plafona a ~0.5) | gate espansione `ang_vel_z` **0.8 → 0.55**·peso | `curriculums.py` |
| 2b | Girare sul posto = manovra peggiore (roll −9.1°, **vy spuria 0.45 m/s**) | nuova `lateral_drift_on_turn` (peso **−0.5**), attiva solo con comando lineare ~0 e yaw > 0.2 | `rewards.py`, `squat_env_cfg.py` |
| 3 | Asimmetria di risposta: vy a sinistra **0.13** vs destra 0.73; wz sinistra 0.33 vs destra 0.45 | nuova `joint_mirror_antisym` su hip_roll/ankle_roll (peso **−0.3**) | `rewards.py`, `squat_env_cfg.py` |
| 4 | Ciclo-limite caviglia ridotto ma presente (tgt 0.172 / mis 0.044) | `ankle_action_smoothness` −0.02 → **−0.04** | `squat_env_cfg.py` |
| 5 | **12% del run** era squat-mentre-cammina = fuori distribuzione | `only_when_standing` True → **False** (Fase B) | `squat_env_cfg.py` |

**Perché `joint_mirror_antisym` e non `joint_mirror`:** i giunti *roll* sono **antisimmetrici** (L ≈ −R:
misurato hip_roll L −0.012 / R +0.010), quindi la simmetria corretta è `(L+R)²`, non `(L−R)²`. Inoltre
`joint_mirror` tiene la cache in **un unico attributo sull'env**: un secondo termine che riusasse quella
funzione erediterebbe silenziosamente i giunti del primo → serve una funzione con cache propria.
**Non** includere `hip_yaw`: la rotazione richiede legittimamente asimmetria di yaw.

**Rischio principale del round 2:** la #5 (squat in cammino) alza parecchio la difficoltà. Il freno di
sicurezza è il curriculum di altezza, che approfondisce solo se `track_lin_vel_xy` **e**
`track_base_height` sono entrambi buoni: se il regime è troppo duro, la profondità semplicemente non
cresce (nessun collasso). Sorvegliare `base_contact` e `height_cmd_levels` nelle prime ~500 iter.

**Non fatto di proposito:** stringere lo `std` di `track_ang_vel_z` (attualmente √0.25). Interagisce con
la soglia del curriculum appena abbassata (std più stretto ⇒ reward più bassa ⇒ gate più difficile):
meglio cambiare una variabile per volta. È il primo knob se il guadagno yaw resta < 0.7.

**Criteri di successo del round 2 (in sim2sim):**
- guadagno yaw **> 0.7** (era 0.40) e nessun picco di roll > 5° girando sul posto;
- rapporto vy sinistra/destra **> 0.6** (era 0.13/0.73);
- errore altezza squat **< 1 cm su tutto il range 0.60–0.98** (niente saturazione);
- sway `wy` da fermo **< 0.012** (era 0.016);
- squat in cammino: `wy` comparabile al cammino eretto (era 0.029 vs 0.020).

---

## 4-ter. ROUND 4 — "bacino dietro le caviglie" (dopo `exp_2026-09-04_18-06-20` / `18-13-04`)

Sul robot reale, con la policy `2026-09-04_15-03-01`, il bacino resta **dietro** le caviglie e arretra
scendendo. Prima di intervenire ho ricostruito il pitch della base **implicato dalla cinematica a piede
piatto**, sia dagli angoli misurati sia dagli angoli *target*, per capire se la policy stesse rifiutando
di inclinarsi in avanti o se non ci riuscisse:

| | pitch IMU | da q **misurati** | da q **TARGET** |
|---|---|---|---|
| REALE, in piedi 0.98 | −2.1° | +1.2° | **+15.8°** |
| REALE, squat 0.73 | −1.6° | +4.0° | **+37.8°** |
| SIM, in piedi 0.98 | +0.5° | +0.5° | +3.6° |
| SIM, squat 0.73 | +3.2° | +3.2° | +15.6° |

**La policy chiede già 2.4–4× più lean in avanti che in sim e non lo ottiene**: il collo di bottiglia non
è la reward, è il **cedimento del PD**. Errori target−misurato da fermo:

| | hip_pitch (kp 250) | knee (kp 400) | ankle_pitch (kp 80) |
|---|---|---|---|
| REALE 0.73 | −1.3° | −13.9° | **−18.6°** (−26 N·m) |
| SIM 0.73 | +0.4° | −12.2° | **−0.6°** |

Il **ginocchio cede uguale** in sim e sul reale → è modellato bene. La divergenza è tutta nella
**caviglia**, e si autoalimenta: CoM indietro → coppia → cedimento (kp 80) → il corpo ruota indietro →
CoM ancora più indietro. Due caviglie danno 160 N·m/rad contro una rigidezza da pendolo inverso
`M·g·h` = 493 N·m/rad (**32%**): l'equilibrio lo tiene solo il loop attivo della policy, quindi ogni
offset di CoM non modellato si presenta **amplificato**. Riscontro indipendente: lo scarto fra IMU e
cinematica-a-piede-piatto è 0.0° in sim e **−3.4° in piedi → −5.6° a 0.73 m** sul reale — il piede che
ruota punte-in-su appoggiandosi sul tallone, cioè il dondolio osservato a occhio.

| # | Problema misurato | Modifica | File |
|---|---|---|---|
| 1 | La randomizzazione CoM su `pelvis`+`torso_link` copre solo il **35.5%** della massa → ±8 cm impostati = **±2.8 cm effettivi**, sotto il disallineamento reale (3–7 cm): il round 3 addestrava **fuori distribuzione** | `body_names` → `.*` (rapporto **1:1**), range x **±0.06** | `velocity_env_cfg.py` |
| 2 | `flat_orientation_l2` (−5.0) penalizza uguale avanti e indietro, ma il poligono d'appoggio è asimmetrico (punta +17.3, tallone −8.6 cm) e il lean avanti è la **manovra di recupero** | nuova `flat_orientation_asym`: rollio e lean **indietro** penalizzati sempre; lean **avanti** libero dentro una banda morta ∝ profondità (0° in piedi → ~20° a fondo squat) | `rewards.py`, `squat_env_cfg.py` |

**Perché ±0.06 e non ±0.08:** l'*errore di modello* misurato è ~3–4 cm (stessa policy da fermo: il CoM si
assesta a +0.6/+1.3 cm in sim e a −2.4/−3.0 cm sul reale) → fattore di sicurezza 1.5–2×. Con ±8 cm
**effettivi** sommati alla posa di squat profondo molti episodi sarebbero irrecuperabili (margine al
tallone: 8.6 cm) → collasso suicida.

**Perché la randomizzazione delle masse NON sostituisce quella del CoM:** su 20 000 estrazioni della
scala 0.85–1.15 iid per link, il CoM globale si sposta in orizzontale di **σ = 0.08 cm** (max 0.29) —
il **4%** di quel che serve — perché la distribuzione di massa è quasi simmetrica avanti/indietro e le
perturbazioni si mediano. Sposta il CoM in **verticale** (σ 0.87 cm). Le due randomizzazioni sono
proprietà **ortogonali** dello stesso corpo rigido (PhysX tiene `mass`, `com`, `inertia` separate) e si
compongono senza conflitto: con lo stesso offset D su tutti i link, `Σ(mᵢsᵢD)/Σ(mᵢsᵢ) = D` per ogni sᵢ.

**Perché offset condiviso e non indipendente per link:** l'effetto globale di offset indipendenti si
media come 1/√N — con 28 corpi servirebbero ±40 cm per link per ottenere ±8 cm complessivi.

**Non fatto di proposito:** alzare la `kp` di caviglia (80 → 160). È l'attacco più diretto al cedimento,
ma va alzata *insieme* in training e deploy, `randomize_actuator_gains` (0.8–1.2) non copre un fattore 2,
e c'è una storia documentata di ciclo-limite sulla caviglia. Leva di riserva per il round 5.

**Criteri di successo (sim2sim con CoM spostato, poi reale):**
- da fermo in piedi, coppia di caviglia **< 15 N·m** totali (era 16 sul reale, ma con CoM randomizzato
  deve restare tale *anche* con offset iniettato);
- CoM a **0.73 m** entro **−4 cm** dalla caviglia (era −7.0, margine 1.6 cm);
- scendere a **0.60 m** sul reale senza dondolio delle punte (mai sceso sotto 0.728);
- nessuna regressione su `track_lin_vel_xy`, guadagno yaw, sway da fermo.

---

## 5. Checklist edit (file → modifica)

- [x] `mdp/rewards.py` — aggiunta `joint_action_rate_standing` (P1a)
- [x] `mdp/__init__.py` — export automatico via `from .rewards import *` (nessuna modifica necessaria)
- [x] `robots/h1_2/squat_env_cfg.py` — `ankle_action_smoothness` −0.02 (P1a); `stand_still_zero_cmd.weight=-0.1`
      (P1b); `squat_symmetry` peso −1.0 (P3a); `track_base_height` std 0.07 (P4a)
- [x] `mdp/curriculums.py` — espansione `ang_vel_z` in `lin_vel_cmd_levels`, gate indipendente (P2a)
- [x] `robots/h1_2/velocity_env_cfg.py` — `track_ang_vel_z` peso 1.0 (P2b). P3b (`track_lin_vel_xy` 1.2) NON
      applicata (priorità bassa, gap sim2sim; da valutare solo se vx resta troppo bassa)
- [ ] (opz. deploy) `robots/h1_2_squat/config/config.yaml` — kd caviglie (P1c, test sintomatico, non necessario)

**Round 4:**

- [x] `mdp/rewards.py` — aggiunta `flat_orientation_asym` (banda morta in avanti ∝ profondità)
- [x] `mdp/events.py` — docstring: scaling per frazione di massa, perché offset condiviso e non per-link
- [x] `robots/h1_2/squat_env_cfg.py` — `flat_orientation` (−5.0) al posto di `flat_orientation_l2` (→ `None`)
- [x] `robots/h1_2/velocity_env_cfg.py` — `randomize_base_com`: `body_names=".*"`, x ±0.06, y/z ±0.02

> Tutte le modifiche superano `python -m py_compile`. Nessuna tocca obs actor / azioni / deploy.

> Nessuna modifica a osservazioni actor, spazio azioni, o pipeline deploy.
