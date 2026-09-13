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

## 4-quater. ROUND 5 — tracking dei comandi (dopo `exp_2026-09-07_11-27-08`)

Il round 4 ha chiuso il problema del baricentro sul robot reale (coppia di caviglia a 0.73 m da
−52.1 a **−16.8 N·m**, margine da 0.7 a **6.0 cm**, squat fino a **0.662 m**). Resta il tracking dei
comandi. La diagnosi non è "la policy non ce la fa", è **quanta reward c'è in palio**.

Entrambi i termini usano `exp(−err²/std²)` con **std = 0.5**, contro comandi di ±0.3 m/s laterali e
±0.5 rad/s: la campana è quasi piatta e restare fermi rende quasi quanto inseguire.

| canale | range | reward in palio (media) | guadagno misurato sim2sim |
|---|---|---|---|
| vx | [−0.3, +1.0] | 0.455 | 0.77 |
| wz | [−0.5, +0.5] | 0.253 | 0.73 |
| vy | [−0.3, +0.3] | **0.108** | **0.63** |

L'ordine dei guadagni riproduce esattamente l'ordine della reward disponibile.

> **Correzione (misurata su `sim_2026-09-07_14-23-00`).** In una prima stesura avevo scritto che per
> vy la posta media (0.108) è inferiore al costo di `joint_deviation_hips` (0.120): era un confronto
> sbagliato, fra la posta *mediata sul range* e il valore *totale* del termine. Il costo che conta è
> quello **marginale**, misurato: `|q−q_default|` sui fianchi passa da 0.037 da fermo a 0.089 con
> vy=0.30 e 0.098 con wz=0.50, cioè **+0.045 e +0.055** — ben sotto la posta al comando massimo
> (0.302 e 0.632). Quindi `joint_deviation_hips` **non è il freno dominante**: lo è la campana piatta
> (P1). P2 resta giustificata su una base più debole — ai comandi *tipici* di metà range la posta
> scende a 0.086 e il costo dei fianchi pesa in proporzione molto di più, e penalizzare `hip_yaw`,
> che serve solo quando il giro è comandato, non ha comunque alcun beneficio.

| # | Problema misurato | Modifica | File |
|---|---|---|---|
| P0 | `ankle_torque_margin` spento sotto 0.93 m: dove è acceso la policy compensa il **69%** di uno sbilanciamento di CoM, dove è spento il **43%** — e non è il fine corsa (restano 4.9°), è mancanza di incentivo | gate da "vicino al nominale" a **"assestato alla quota comandata"** (`|h − h_cmd| < tol`) | `rewards.py`, `squat_env_cfg.py` |
| P1 | campana di tracking quasi piatta; non si può stringere lo `std` perché i curriculum leggono quei termini **per nome** e girano al limite del gate (0.841 vs 0.800; 0.609 vs 0.550) | due termini **aggiuntivi** `track_*_fine` con std 0.25, peso 0.5 → posta ×1.7 (vx), ×2.1 (wz), ×2.5 (vy) | `velocity_env_cfg.py` |
| P2 | `joint_deviation_hips` (−1.0) penalizzava hip_roll **e** hip_yaw, cioè i giunti che servono a eseguire vy e wz | **split**: `hip_roll` resta −1.0 (guardia anti-spaccata), `hip_yaw` scende a **−0.25** | `velocity_env_cfg.py` |
| P3 | girare sul posto capitava nel ~2% dei campioni (vx/vy/wz estratti indipendenti + env fermi azzerati) | nuovo `rel_turn_in_place_envs = 0.15` in `UniformLevelVelocityCommand` | `commands/velocity_command.py`, `velocity_env_cfg.py` |
| P4 | camminare accovacciati è fuori distribuzione (`only_when_standing=True`): guadagni 0.09–0.28 a mezzo squat contro 0.6–0.77 in piedi | **Fase B parziale**: `rel_squat_while_walking = 0.35` | `commands/height_command.py`, `squat_env_cfg.py` |
| P5 | `action_rate` è la penalità più grossa in assoluto (−0.323) e tassa ogni movimento | −0.05 → **−0.035** | `velocity_env_cfg.py` |
| P7 | il 35% degli env in cammino campionava profondità fino a 0.60 m, dove i guadagni misurati sono −0.03 / −0.01 / +0.13 e lo sway triplica: gradiente speso su un compito che non riesce | `min_height_while_walking = 0.70` (rimappa affine, non troncamento) | `commands/height_command.py`, `squat_env_cfg.py` |
| P6 | a 0.60 m la caviglia ha solo **5.8°** di dorsiflessione residua, quindi l'inclinazione del busto è l'unico grado di libertà rimasto per ricentrare il baricentro — e la banda morta lo taglia proprio lì (lean misurato +20.9° contro banda 20.5°) | `forward_deadband` 0.35 → **0.42** (~25°) | `squat_env_cfg.py` |

**Perché vy è strutturalmente svantaggiata:** `track_lin_vel_xy_yaw_frame_exp` usa la **norma**
dell'errore su (vx, vy) in un unico esponenziale. Quando l'errore su vx domina — e domina, perché vx
arriva a 1.0 contro 0.3 — il gradiente rispetto a vy si schiaccia: con errore vx 0.2 e vy 0.1,
azzerare vy porta la reward da 0.82 a 0.85. Se P1 non basta, il passo successivo è un termine
dedicato al solo vy.

**Rischi da sorvegliare nelle prime ~500 iterazioni:**
- P1 + P4 spostano peso sulla locomozione a scapito dello squat (`track_base_height` vale 2.94, il
  termine positivo dominante). Il freno è il curriculum di altezza, che approfondisce solo se
  `track_base_height > 2.4`: guardare `height_cmd_levels`.
- P5 interagisce con P1: se `action_rate` risale sopra ~0.40 e la posa torna nervosa, è il primo
  knob da riportare indietro.
- P3 riduce la frazione di env che cammina in linea retta: se `terrain_levels` crolla, abbassare
  `rel_turn_in_place_envs` a 0.10.

**Criteri di successo (sim2sim, con segmenti di comando costante da almeno 8 s):**

| metrica | oggi | obiettivo |
|---|---|---|
| guadagno vx in piedi | 0.77 | > 0.85 |
| guadagno vy in piedi | 0.63 | > 0.80 |
| guadagno wz in piedi | 0.73 | > 0.85 |
| guadagno vx / vy a mezzo squat | 0.19–0.28 / 0.09–0.19 | > 0.50 |
| errore altezza | < 1 cm | invariato |
| margine al tallone a 0.60 (modello CoM −3.5 cm) | 6.3 cm | ≥ 5 cm |

---

## 4-quinquies. ROUND 6 — velocità proporzionale alla profondità, e P5 annullata

Il round 5 ha centrato l'obiettivo sul tracking (misure su `sim_2026-09-07_23-16-17`):

| | eretto 0.98 | half 0.78 | full 0.60 |
|---|---|---|---|
| vx avanti | 1.02 → **1.06** | 0.55 → **0.82** | 0.28 → **0.39** |
| vx indietro | 0.60 → **0.80** | 0.16 → **0.69** | −0.03 → **0.18** |
| vy | 0.63 → **0.79** | — | −0.01 → 0.03 |
| wz | 0.73 → 0.69 | 0.47 → **0.58** | 0.13 → 0.05 |

A mezzo squat l'andatura è tornata (ampiezza del ginocchio 1.4–1.7×). A fondo squat no, e non tornerà:
a 0.60 m il robot ha `hip_pitch` −96°, `knee` +118°, busto +24° e **5.8°** di dorsiflessione residua.

| # | Problema misurato | Modifica | File |
|---|---|---|---|
| P5⁻¹ | `action_rate` **non** penalizza il movimento ma il RUMORE: azioni consecutive indipendenti contribuiscono `2σ²·n`, che spiega il **98%** (round 4) e il **95%** (round 5) del valore misurato. Abbassarla ha alzato σ del 32% e degradato la policy **esportata**: action_rate grezza da fermo +137%, sway `wy` da fermo +79%, `\|wy\|` max +68%, `gait` −23% | `action_rate` −0.035 → **−0.05** (annulla P5) | `velocity_env_cfg.py` |
| P8 | P7 impedisce agli env in cammino di scendere sotto 0.70, ma a 0.70 possono ancora ricevere `vx = 1.0`: stesso spreco di gradiente, un piano più su. E al deploy la levetta a fondo squat manda il robot nella sua condizione meno stabile (sway 0.057, roll −4.9°) senza che si muova | ampiezza del comando **∝ profondità**: 1.0 sopra 0.95 m, 0 sotto 0.62 m. Applicata IDENTICA in training e nel controller C++ | `commands/velocity_command.py`, `export_deploy_cfg.py`, `deploy/.../observations.h`, `velocity_env_cfg.py` |

**Fattore risultante:** 0.85 a 0.90 m, 0.70 a 0.85, **0.50 a 0.785**, 0.24 a 0.70, 0 sotto 0.62.

⚠️ **P8 tocca il deploy, ed è obbligatorio che lo faccia.** Se si scalasse solo in addestramento si
creerebbe esattamente il disallineamento di distribuzione che la modifica vuole rimuovere. I due
numeri viaggiano in `deploy.yaml` (`commands.base_velocity.speed_scale`) generato dalla config di
training, quindi la formula è scritta una volta sola. Va applicata in **`velocity_commands` e in
`gait_phase`**: quest'ultima usa il comando come soglia di congelamento del clock, e in addestramento
legge il comando già scalato. Per le policy precedenti la chiave è assente e il fattore vale 1.0.

---

## 4-sexies. ROUND 7 — banda morta a mezzo squat e rettilineità (dopo `sim_2026-09-10_18-08-39`)

Policy di partenza: `2026-09-08_00-25-08` (round 6). Il round 6 è stato il migliore su quasi
tutto — reward 109.7 → 115.4, `error_vel_yaw` −21%, σ 0.451 → 0.375, e soprattutto lo squat
profondo da fermo è diventato quasi rigido (rollSD 0.767 → **0.050°**, sway 0.0156 → **0.0023**,
gyro 0.0085 → **0.0019**, tenuto 22 s a 0.596 m). Restano due difetti, entrambi diagnosticati
quantitativamente.

### P10 — soglie "comando ~ zero" proporzionali alla scala di profondità

**Sintomo.** A 0.74 m, con lo stick a fondo corsa: vx avanti guadagno 0.79 (0.29 m/s), ma vx
indietro 0.09, vy 0.13–0.23, wz 0.35–0.40. Velocità del ginocchio 0.013–0.018 rad/s contro 0.034
in avanti alla stessa quota e 0.045 in piedi: **non fa il passo**, oscilla sul posto. Non è un
limite fisico — il round 5, senza scaling, a 0.785 m faceva −0.20 m/s indietro e ±0.23 rad/s.

**Causa.** Lo scaling lineare del round 6 interagisce con tre soglie fisse a 0.1 sulla norma del
comando, tarate sulla stazione eretta:

| termine | soglia | si spegne (o accende) sotto |
|---|---|---|
| `gait_phase` (obs + deploy C++) | `‖cmd‖ > 0.1` | h = 0.730 (vy, vx−) / 0.686 (wz) |
| `feet_gait` (+0.5) | `‖cmd‖ > 0.1` | idem |
| `double_support` (+0.5) | `‖cmd‖ < 0.1` | si **accende** sotto 0.730 |

*(`feet_clearance` usa `foot_clearance_reward`, che NON ha gate sul comando: non è coinvolta.
`feet_height_body` ce l'ha ma non è in config.)*

Con `lin_vel_y ∈ [−0.3, 0.3]` e scale lineare, il laterale a fondo corsa supera 0.1 solo sopra
0.713 m, e a 0.74 m solo l'8% dei comandi campionati lo supera. Nella fascia 0.62–0.73 il robot
riceve un target di velocità non nullo mentre il clock del passo è congelato e `double_support`
lo paga per restare piantato. La vx avanti si salva solo perché il range è asimmetrico
`[−0.3, 1.0]`. Corollario coerente: `terrain_levels` 4.63 → 3.86, perché gli env squat-walk non
percorrono più metà della distanza comandata.

**Fix.** `command_gate_threshold(env, name, 0.1)` = `0.1 · max(scale, threshold_min_scale)`, con
`threshold_min_scale = 0.2`. Il pavimento è necessario: a fondo squat `scale = 0` e una soglia
nulla renderebbe falso `cmd_norm < thr`, spegnendo il doppio appoggio proprio dove stare fermi è
tutto ciò che vogliamo. Con il pavimento la soglia a fondo squat è 0.02 e il comando è
identicamente 0 → clock congelato e doppio appoggio attivo, come prima. Replicato in C++
(`squat_gait_threshold`), con i parametri esportati in `deploy.yaml`.

### P11 — curva di scaling non lineare

`scale = clamp((h−h0)/(h1−h0))^0.5`. Estremi invariati (0.62 → 0 esatto, 0.95 → 1 esatto), ma
la fascia intermedia recupera autorità:

| h | lineare | sqrt | vy pieno | wz pieno | soglia |
|---|---|---|---|---|---|
| 0.85 | 0.697 | 0.835 | 0.250 | 0.417 | 0.083 |
| 0.80 | 0.545 | 0.739 | 0.222 | 0.369 | 0.074 |
| 0.74 | 0.364 | **0.603** | 0.181 | 0.302 | 0.060 |
| 0.70 | 0.242 | **0.492** | 0.148 | 0.246 | 0.049 |

Con P10+P11 il gate si apre su tutti e quattro gli assi fino a 0.62 m, mentre nel round 6 vy e
vx− erano già chiusi a 0.72 e wz a 0.66. Verificato che Python e C++ producono numeri identici
bit per bit su tutta la scala.

### P12 — rettilineità: `heading_hold` e `path_hold`

**Sintomo.** Su percorsi di 2.3–3.0 m il robot esce dalla retta di 15–52 cm. Scomponendo lo
scostamento fra rotazione non comandata e strisciamento nel body frame:

| bout (h=0.98) | devMONDO | devCORPO | quota imbardata | Δyaw |
|---|---|---|---|---|
| vx +0.33, 9.0 s | −0.255 m | −0.188 m | 26% | −2.4° |
| vx −0.30, 11.1 s | +0.523 m | +0.079 m | **85%** | **+21.0°** |
| vy −0.30, 8.5 s | +0.155 m | +0.214 m | −38% | −3.3° |
| vy +0.20, 12.1 s | +0.353 m | −0.229 m | **165%** | **+22.7°** |

**Causa A — deriva di imbardata.** Regressione sui bout senza comando di yaw:
`ω [°/s] ≈ +0.72 − 4.45·vx + 6.52·vy` (residuo RMS 0.80). Bias e coefficienti cambiano segno da
una policy all'altra a parità di modello MuJoCo (R4 −2.53 / R5 +0.80 / R6 +0.72) → è la policy,
non il simulatore. Meccanismo: le due semifasi del passo differiscono di 2–7 °/s e la deriva è il
residuo non cancellato di quell'oscillazione.

**Causa B — creep costante.** (+0.02 avanti, −0.02 a destra) m/s nel body frame, in ogni modalità
di cammino, assente da fermo. Spiega anche il comando di yaw: v costante nel body frame + ω
costante = arco di raggio |v|/ω. Previsto R = 0.108 / 0.093 m e spostamento netto 0.209 / 0.175 m
su 152° / 139°; **misurato 0.208 / 0.182**. Accordo entro il 4%.

**Perché i termini esistenti non lo vedono.** Con std 0.5/0.25, una deriva di 2 °/s mette in palio
0.0146/step e un creep di 0.02 m/s ne mette 0.0048, su ~115 di reward totale. È sotto la
risoluzione del reward, non una questione di peso. E **stringere lo std sul rate sarebbe
controproducente**: l'oscillazione sana dentro il passo (2–7 °/s) è più grande della deriva, quindi
si punirebbe l'andatura normale.

**Fix.** Errore **integrato** sulla finestra di comando (10 s), dove l'oscillazione per-passo si
media a zero e resta solo la deriva. Accumulatori dentro `UniformLevelVelocityCommand`, che è
l'unico oggetto che sa quando fa il resample:

- `heading_err += (ω_z − ω_cmd)·dt` → `heading_hold = exp(−err²/0.25²)`, peso 0.5
- `path_err += perp(v_body − v_cmd)·dt` → `path_hold = exp(−‖err‖²/0.20²)`, peso 0.3

Solo la componente **perpendicolare** al comando: quella parallela è il normale ritardo di
inseguimento (guadagno 0.87 → 0.4 m integrati su 10 s) e penalizzarla renderebbe questo un
`track_lin_vel` più duro invece che un termine di rettilineità. A comando lineare nullo la
direzione è indefinita e si accumula tutto il vettore — cioè esattamente il cerchio.

`drift_grace_steps = 25` (0.5 s): senza, ogni termine porterebbe un offset incomprimibile, perché
raggiungere un comando appena estratto integra ~0.12 rad anche per una policy perfetta.

Montepremi verificato numericamente: con la deriva misurata la reward media sulla finestra è 0.628
(heading) e 0.626 (path) contro 1.000 di una policy senza deriva → **37% in palio**, contro lo
0.013% e 0.004% di prima.

### P13 — `lateral_drift_on_turn` riattivato

Era inerte nel 2026-09-04 (reward ferma a −0.000) perché "comando lineare ~0 + yaw significativo"
era il ~2% dei campioni. Dal round 5 `rel_turn_in_place_envs = 0.15` rende lo scenario esplicito:
la condizione posta allora è soddisfatta. Peso −2.0. Divisione dei compiti con `path_hold`:
`path_hold` lavora sull'integrale e cattura il creep costante, questo lavora sull'istante e smorza
le sbandate di picco. La soglia sul comando lineare ora scala con la profondità; quella sullo yaw
resta assoluta (0.2) perché misura l'entità della manovra — e con la curva sqrt il wz a fondo
corsa vale 0.30 a 0.74 m e 0.25 a 0.70 m, quindi resta sopra 0.2 su tutta la fascia in cui il
robot ha il permesso di camminare accovacciato.

### Verifiche fatte prima del lancio

- `compileall` su tutto `source/unitree_rl_lab`: OK.
- `g++ -fsyntax-only` su `deploy/robots/h1_2_squat/main.cpp` con tutti gli include: OK.
- Curva e soglie: Python e C++ identici bit per bit su 0.60–0.98.
- Trascrizione numpy di `_update_command` + `_accumulate_drift` su 500 step: comando **non**
  decade (nessun compounding come nel bug P8), env fermi restano a zero esatto, grace rispettato
  (19.0° = 2 °/s × 9.5 s), reset pulito, proiezione perpendicolare corretta (0.19 m camminando
  dritto contro 0.269 m in rotazione pura).
- **Non** testato in esecuzione reale: su questa macchina non c'è torch né IsaacLab.

### Cosa misurare al prossimo sim2sim

1. Rifare la regressione `ω = a + b·vx + c·vy`: il bias +0.72 °/s e i coefficienti −4.45 / +6.52
   devono scendere in modulo. Target: |ω| < 0.5 °/s su tutti i bout.
2. Spostamento netto durante 145° di rotazione: 0.19 m → target < 0.08 m.
3. Guadagni a 0.74 m su vy / vx− / wz: 0.13–0.23 / 0.09 / 0.35–0.40 → target > 0.5 su tutti.
4. Controllare che lo squat profondo da fermo **non** sia regredito: rollSD 0.050°, sway 0.0023,
   gyro 0.0019 sono il riferimento da difendere.
5. `terrain_levels` deve risalire verso 4.6.

## 4-septies. ROUND 8 — dopo il confronto a cinque policy sullo stesso protocollo

Primo confronto fatto con il banco automatico (`tools/run_protocol.py`, protocollo `standard_v1`):
cinque policy, `rtf = 1.0000` su tutte, `wall_seconds` identico, 21 segmenti confrontabili, zero
controlli falliti. I run sono in `sim_logs/automatic/`, 2026-09-13.

### Verdetto sul round 7

| modifica | esito |
|---|---|
| P10 soglie proporzionali + P11 curva sqrt | **funzionano**: a 0.74 m vx avanti 0.326 → 0.522 m/s (+60%), wz 0.097 → 0.238 e −0.078 → −0.217 (+145% / +178%) |
| P12 `heading_hold` | **a metà**: accoppiamenti collassati (vy +6.21 → −1.07, vx −1.54 → −0.17) ma \|ω\| solo 1.70±0.18 → 1.23±0.22 °/s |
| P12 `path_hold` | **controproducente**: creep laterale 0.0165±0.0014 → 0.0309±0.0033 m/s, 4σ di peggioramento |
| P13 `lateral_drift_on_turn` | **inerte**: −0.0041 su 122 di reward (0.003%); raggio del giro 0.042/0.067 → 0.136/0.118 m |

Il laterale a 0.74 m resta a guadagno 0.25 **nonostante** il comando (0.190) superi ampiamente la
soglia del clock (0.060): non è più un problema di gate.

### P14 — `heading_hold` e `path_hold` su errore MEDIO, non accumulato

Due difetti della versione a integrale, entrambi misurati:

1. **Satura.** A 1.7 °/s su 9.5 s l'errore accumulato è 0.28 rad contro std 0.25 → reward 0.29 con
   gradiente già quasi piatto. E il valore dipende dalla lunghezza della finestra: la stessa deriva
   vale 0.96 su 3 s e 0.37 su 20 s.
2. **Dà reward pieno all'immobilità.** Con `err = v − cmd` e la proiezione che toglie la componente
   lungo il comando, un robot fermo dà `err_perp = −cmd + (cmd·u)u = 0` **esatto**: il termine non
   distingueva "cammina dritto" da "non cammina".

Nuova forma: `exp(−(err/T)²/σ²)` con `T` = tempo attivo (pavimento 1 s), std in unità fisiche.
Verificato numericamente: il reward è costante da 1.5 s in poi, indipendente dalla finestra.

| | vecchia (in palio) | nuova (in palio) | |
|---|---|---|---|
| heading, da R6 (1.70 °/s) a 0.30 °/s | 0.287 | 0.823 | 2.9× |
| path, da R7 (0.0309) a R6 (0.0165 m/s) | 0.074 | 0.393 | 5.3× |

`path_hold` passa da peso 0.3 a **1.0**: a 0.3 la policy lo barattava via — il suo valore in
addestramento è *calato* (0.183 → 0.168, cioè creep implicito 0.0292 → 0.0324 m/s, che coincide con
i 0.0309±0.0033 misurati in deploy).

**`heading_hold` è a due scale** (larga σ=0.06 peso 0.3, stretta σ=0.02 peso 0.5), come
track_lin_vel dopo P1 e per la stessa ragione. Invertendo il valore del round 7 a iterazione 1000,
la deriva implicita a inizio addestramento è **3.46 °/s**, dove σ=0.02 darebbe reward 0.000 e
gradiente 0.0: una zona morta che sprecherebbe metà del round.

| deriva | larga (0.06) | stretta (0.02) | totale in palio |
|---|---|---|---|
| 3.46 °/s | 0.298 | 0.000 | 0.089 |
| 1.70 °/s | 0.783 | 0.111 | 0.291 |
| 1.23 °/s | 0.880 | 0.316 | 0.422 |
| 0.50 °/s | 0.979 | 0.827 | 0.708 |

`path_hold` resta a scala singola: il creep implicito a inizio addestramento è 0.0292 m/s, dove
σ=0.03 vale 0.34 con gradiente 23.7 — nessuna zona morta.

### P15 — `lateral_drift_on_turn` rimosso dalla config

A −2.0 vale lo 0.003% della reward; per contare servirebbe un peso ~50× scelto a tentoni. Il
compito lo fa già `path_hold` riformulato: a comando lineare nullo la proiezione degenera e
accumula l'**intero** vettore velocità, cioè esattamente la sbandata durante la rotazione.
Verificato numericamente: env in giro sul posto → `|pm| = 0.0283` = creep pieno; env in marcia
dritta → `|pm| = 0.0200` = sola componente ortogonale. La funzione resta in `rewards.py`.

### P16 — densità di campionamento dello squat-in-cammino

`rel_squat_while_walking` 0.35 → **0.50** e nuovo `max_height_while_walking = 0.85`. Sopra 0.85 il
robot cammina già bene (guadagni 0.87–1.01 in piedi): quei campioni non insegnano nulla. La
rimappatura affine già usata per il pavimento vale ora su tutta la finestra, quindi la
distribuzione resta uniforme (bin massimo 1.72%, cioè 1/60).

Frazione di env in cammino con quota in [0.71, 0.77]: **7.5% → 20.0%** (2.7×).

### Non toccato di proposito

**vx indietro** (−0.144 m/s in piedi, outlier basso: R3 −0.209, R5 −0.237, R6 −0.209) e **lean a
fondo squat** (−17.4° contro −22.2° di R6, coppia di caviglia +1.86 contro +2.74). Entrambi sono
regressioni reali ma ho un solo seed per policy e non posso distinguerle da varianza di
addestramento. Sono i due numeri da guardare per primi nel round 8; se il lean resta compresso su
due round consecutivi, allora è uno dei nuovi termini a comprimerlo e va gated.

### P17 — audit dei reward: `stand_still_zero_cmd` rimosso

33 termini attivi. Il candidato problematico è uno solo: `stand_still_zero_cmd` non ha `asset_cfg`,
quindi vale su **tutti e 27 i giunti**, ed è gated su `is_standing_env` — che con
`only_when_standing=True` sono esattamente gli env che ricevono la profondità di squat.

| quota | somma \|q − q_default\| | penalità/step | % di `track_base_height` (3.0) |
|---|---|---|---|
| 0.980 | 1.78 rad | −0.178 | 5.9% |
| 0.761 | 5.68 | −0.568 | 18.9% |
| 0.641 | 7.45 | −0.745 | **24.8%** |

(verifica incrociata: il modello prevede un valore di episodio 0.1017, misurato 0.1046)

**Non è stato rimosso perché "frena lo squat":** l'equilibrio fra questa tassa e il gradiente di
`track_base_height` dà un offset di appena **1.3 mm**, e l'errore di inseguimento vero è ≤5 mm a
ogni quota. È stato rimosso perché (a) **non fa il suo lavoro** — era l'anti-barcollio, ma il
ciclo-limite della caviglia vive nell'azione, non nella posizione misurata, ed è per questo che
esiste `ankle_action_smoothness`; (b) ogni sua parte legittima è già coperta da un termine più
mirato (caviglie, braccia, busto, hip_roll/yaw, doppio appoggio). Resta solo un bias sulla postura
di squat di cui non controlliamo il verso.

Riferimento da difendere al prossimo run (segmento t=14, h=0.98, comando nullo): sway 0.0013 m/s,
rollSD 0.027°, gyro 0.0009 rad/s.

### P18 — `joint_deviation_hip_yaw` −0.25 → −0.15

Nel round 7 la deviazione di hip_yaw è salita del 68% (0.0223 → 0.0375) proprio mentre
`heading_hold` veniva ottimizzato (0.193 → 0.328): hip_yaw è il giunto con cui la policy sterza, e
il penalty le rema contro. Stessa logica già nel docstring di `joint_mirror_antisym`.

### Da sapere, non cambiato

- **Il curriculum del terreno è un no-op**: `sub_terrains` contiene solo `flat`, quindi le 9×21
  tessere sono identiche e promuovere un env non cambia nulla di fisico. `Curriculum/terrain_levels`
  **non è una metrica di robustezza** — l'ho citata come tale più volte. Misura solo se gli env
  percorrono la distanza comandata.
- **`feet_clearance` vale esattamente 1.0 da fermo** (`foot_velocity_tanh` = 0 → `exp(0)`), cioè è
  un secondo bonus di sopravvivenza. Con `track_base_height` (2.90) e `alive` (0.15) significa che
  **4.03 dei 7.32 di budget positivo — il 55% — sono incassabili stando perfettamente fermi**.
- **`ankle_torque_margin` vale lo 0.05% del budget** ed è il termine che codifica il margine del
  baricentro sulle caviglie. Non alzato ora: la rimozione di P17 cambia già la postura di squat.
- Cinque termini di regolarità sovrapposti (`action_rate`, `joint_vel`, `joint_acc`, `energy`,
  `ankle_action_smoothness`) per l'8.5% del budget. Correlati fra loro, ma l'equilibrio funziona.
- Inerti **per costruzione e corretti così**: `feet_too_near`, `undesired_contacts`,
  `dof_pos_limits`, `squat_symmetry`. Valgono ~0 perché il vincolo è rispettato.

### Miglioria del protocollo, da fare

Lo scatter di una singola manovra è ±1 °/s sul rate di imbardata: differenze di quell'ordine fra
due policy non sono risolvibili con un tratto solo. Per `standard_v2`: tratti da 15 s invece di 9,
oppure le manovre più rumorose (vx avanti, giro sul posto) ripetute in due punti del protocollo.
Costa ~90 s e raddoppia la potenza statistica.

## 4-octies. ROUND 9 — ripartenza pulita da R4, solo con ciò che ha evidenza

### Perché

Il round 8 (`2026-09-13_16-15-47`) è collassato sullo stesso protocollo: locomozione laterale e
rotatoria a zero (vy 0.29 → 0.01, wz 0.50 → 0.05), e a 0.98 m un ciclo-limite della caviglia a
1.5 Hz (azione std ×44, coppia ±16 N·m, sway da fermo ×36). Due cause distinte:

1. **`stand_still_zero_cmd` rimosso** (P17) → la postura in piedi non è più ancorata → il
   baricentro va sull'asse della caviglia (pitch −4.5°, coppia +7.8 → −0.8 N·m) → caviglia scarica
   → chatter. La mia diagnosi di P17 era letteralmente vera ("non vede il ciclo-limite") e
   sostanzialmente sbagliata: il termine era l'**àncora della postura**, e quella teneva la
   caviglia precaricata. A quote di squat la caviglia è comunque carica e il fenomeno non compare.
2. **`path_hold` a peso 1.0** penalizza lo sway che il passo laterale/rotatorio produce
   inevitabilmente: per vy=0.3 la perdita (0.97) supera il guadagno di tracking (0.68), per vx=1.0
   no (0.47 contro 1.48). La policy ha smesso di fare il passo per tutto ciò che non è marcia in
   avanti. In training: `gait` −38%, `double_support` +76%.

Sei policy confrontate sullo stesso protocollo (`sim_logs/automatic/`, `standard_v1`):

| | R4 | R5 | R6 | R7 | R8 |
|---|---|---|---|---|---|
| wz in piedi (cmd 0.5) | 0.36 | 0.38 | 0.44 | **0.50** | 0.05 |
| vy in piedi (cmd 0.3) | 0.17/0.20 | 0.22/0.25 | 0.27/0.27 | 0.22/**0.29** | 0.01 |
| vx+ a 0.75 m | 0.50 | **0.94** | 0.33* | 0.52* | 0.47* |
| wz a 0.75 m | 0.16 | **0.27** | 0.10* | 0.24* | 0.00 |
| rollSD camminata [°] | **1.44** | 2.11 | 2.00 | 2.32 | — |
| sway fermo 0.98 | 0.002 | 0.001 | 0.001 | 0.001 | **0.046** |
| pitch fondo squat [°] | −17.5 | −20.5 | **−22.2** | −17.4 | −25.7 |
| coppia caviglia fondo [N·m] | −0.55 | −0.14 | **+2.74** | +1.86 | +1.87 |
| deriva yaw \|ω\| [°/s] | 1.73 | 3.60 | **1.49** | 1.20 | — |

\* comando scalato dal controller; R3–R5 ricevono il comando pieno.

### Verdetti

**Tenuto** — CoM su tutti i link + `flat_orientation_asym` (validati sul reale); P0 gate di
`ankle_torque_margin` (test con CoM spostato); P1 fine tracking + P2 hip separati + P3 turn-in-place
(vy/wz +30–40% in tre round; costo rollSD +0.6°, scelta esplicita di tenerli); P4+P7 squat-in-cammino
0.35 con pavimento 0.70 (mobilità a 0.75 m da 2× a 5×, l'evidenza più netta di tutte); P6 deadband
0.42; `stand_still_zero_cmd` a −0.1 (àncora della postura); `action_rate` −0.05.

**Scartato** — scaling della velocità con la quota (riduce per costruzione la mobilità a mezzo
squat, e ha trascinato P10/P11); `heading_hold`, `path_hold`, `lateral_drift_on_turn`; squat-in-cammino
0.50/[0.70, 0.85]; hip_yaw −0.15; curriculum del terreno (no-op: solo `flat`).

**Il config risultante è R5 con `action_rate` a −0.05 — cioè R6 senza lo scaling.** Mai addestrato.

### P19 — cutoff a 0.70 m nel deploy, al posto dello scaling

In training gli env che camminano non scendono mai sotto `min_height_while_walking = 0.70`: sotto
quella quota la policy non ha **mai** visto un comando di velocità non nullo. `export_deploy_cfg`
esporta il pavimento come `cutoff_height` e `squat_speed_scale()` in `observations.h` restituisce 0
sotto quella quota, 1 sopra — per entrambi i chiamanti (`velocity_commands` e `gait_phase`). Non è
uno scaling: è una guardia che tiene il deploy dentro la distribuzione di training. Verificato con
un harness sulle tre generazioni di `deploy.yaml`: policy R4/R5 (nessuna chiave) → 1.0 ovunque,
soglia 0.1; round 9 → 0 sotto 0.70, 1 sopra; R7 → il suo sqrt intatto.

La macchineria di scaling e degli accumulatori di deriva resta nel codice ma spenta
(`speed_scale_zero_height=None`): scale = 1, soglia 0.1, nessuna chiave `speed_scale` in deploy.yaml.
Equivalente a R4/R5 su entrambi i lati. Pulizia del codice: dopo la validazione.

### Verifica prima del lancio

`diff params/env.yaml` del nuovo run contro `2026-09-07_20-47-22` (R5) deve mostrare solo
`action_rate −0.035 → −0.05`, `terrain_levels` assente, e i campi inerti del command term
(`speed_scale_zero_height: null`, `drift_grace_steps`, ecc.). Il `deploy.yaml` deve contenere
`cutoff_height: 0.7` e nessuna chiave `speed_scale`.

### Obiettivi al protocollo (`standard_v1` per il confronto con i sei run, `standard_v2` come riferimento)

- fermo a 0.98: sway ≤ 0.002, gyro ≤ 0.002, coppia caviglia > +3 N·m (niente ciclo-limite)
- camminata in piedi: rollSD ≤ 2.0°, vy ≥ 0.22, wz ≥ 0.42
- fondo squat: rollSD ≤ 0.03°, coppia ≥ 0, pitch ≤ −20°
- mezzo squat a comando pieno: vx+ ≥ 0.85, vy ≥ 0.10, wz ≥ 0.25 (livello R5), sway ≤ 0.10
- deriva yaw |ω| ≤ 1.7 °/s (livello R6)

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

**Round 5:**

- [x] `mdp/rewards.py` — `ankle_torque_standing`: gate "assestato alla quota comandata", rimosso `nominal_height` (P0)
- [x] `mdp/commands/velocity_command.py` — `UniformLevelVelocityCommand` con `rel_turn_in_place_envs` (P3)
- [x] `mdp/commands/height_command.py` — `rel_squat_while_walking` (P4)
- [x] `robots/h1_2/velocity_env_cfg.py` — `track_*_fine` (P1); split `hip_roll`/`hip_yaw` (P2); `rel_turn_in_place_envs=0.15` (P3); `action_rate` −0.035 (P5)
- [x] `robots/h1_2/squat_env_cfg.py` — `rel_squat_while_walking=0.35` (P4); `ankle_torque_margin` senza `nominal_height` (P0); `forward_deadband=0.42` (P6); `min_height_while_walking=0.70` (P7)
- [x] `mdp/commands/height_command.py` — `min_height_while_walking` con rimappa affine (P7)

**Round 6:**

- [x] `robots/h1_2/velocity_env_cfg.py` — `action_rate` −0.05 (annulla P5); `speed_scale_zero_height=0.62`, `speed_scale_full_height=0.95` (P8)
- [x] `mdp/commands/velocity_command.py` — `_update_command` scala il comando con la profondità (P8)
- [x] `utils/export_deploy_cfg.py` — esporta `commands.base_velocity.speed_scale` (P8)
- [x] `deploy/include/isaaclab/envs/mdp/observations/observations.h` — `squat_speed_scale()` applicata in `velocity_commands` **e** `gait_phase` (P8)

**Rimandate alla misura post-training:**

- [ ] range `lin_vel_x` asimmetrico [−0.3, +1.0] → vx all'indietro rende 0.60 contro 1.02 in avanti già da eretto. P1 dovrebbe bastare (posta a |0.3| da 0.302 a 0.68): allargare il range solo se resta sotto 0.80.
- [ ] asimmetria di imbardata che cresce con la profondità (0.74/0.73 eretto → 0.24/0.03 a 0.60), con `hip_yaw` L−R da −0.5° a −1.3°. Nessun termine copre la simmetria durante il movimento (`squat_symmetry` è gated sugli env fermi, `roll_symmetry` copre solo i roll). La risposta corretta sarebbe un termine sull'asimmetria **media nel tempo**, ma P3+P4 portano quelle situazioni in distribuzione: rimisurare prima di aggiungere codice.

> Tutte le modifiche superano `python -m py_compile`. Nessuna tocca obs actor / azioni / deploy.

> Nessuna modifica a osservazioni actor, spazio azioni, o pipeline deploy.
