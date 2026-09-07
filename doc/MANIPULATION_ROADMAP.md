# Manipolazione — Roadmap

> Stato: **proposta concordata**, non ancora avviata. Prerequisito: chiusura del round 5 sullo squat.
> Documento fratello di [SQUAT_TRAINING_ROADMAP.md](SQUAT_TRAINING_ROADMAP.md), stessa disciplina
> train → sim2sim → sim2real con gate di validazione fra le fasi.

## 1. Obiettivo e perimetro

Demo bersaglio: **cammina → individua l'oggetto → lo raccoglie dal tavolo → si sposta → squatta → risale**.

| voce | scelta | conseguenza |
|---|---|---|
| "rialzarsi" | **risalita dallo squat** | Già fatto e validato. Il recupero da terra resta fuori perimetro. |
| braccia | **una (destra)** | 7 giunti. Il braccio sinistro resta sulla posa di default. |
| oggetto | **< 0.3 kg** | Vale il 4% dell'effetto sull'equilibrio (§2a): irrilevante. |
| piano d'appoggio | **0.70–0.80 m** | Raggiungibile stando eretti: raggiungimento e squat **disaccoppiati**. |
| percezione | **marker ArUco ammesso** | Sul **tavolo**, non sull'oggetto (Fase 3). |
| autonomia | **fasi innescate dal joystick** | L'operatore avanza l'FSM. Niente pianificazione. |
| **sotto RL** | **solo il raggiungimento** | La presa è una primitiva; la percezione sta fuori dal loop di controllo. |
| tempi | nessuna scadenza | Percorso robusto, un gate per fase. |

### 1.1 Piattaforma di calcolo di bordo (da documentazione H1-2)

Sensori: mani **FTP sensorizzate**, telecamera **RGB-D**, **LiDAR Livox MID-360**.

| | Motion Control Unit (PC1) | Development Unit (PC4) |
|---|---|---|
| CPU | i5-1235U | **i7-1255U / i7-1265U** |
| core / thread | 10 / 12 | 10 / 12 |
| RAM | 8 GB LPDDR5 | **16 / 32 GB LPDDR5** |
| GPU | — | **Intel Iris Xe integrata**, 1.20 GHz |
| accelerazione DL | GNA 3.0, DL Boost (VNNI) | GNA 3.0, DL Boost (VNNI) |
| indirizzo | non accessibile | **192.168.123.164** |

**Nessuna CUDA, da nessuna parte**: entrambe le unità sono Intel U-series a 15 W con grafica integrata.
Qualsiasi inferenza a bordo passa da CPU o OpenVINO su Iris Xe. (La GNA 3.0 non aiuta: è per modelli
audio minuscoli.) **PC1 non è accessibile**, quindi tutto il nostro codice vive su PC4.

**Il controllo però non è il problema.** L'MLP dell'actor è `465 → 512 → 256 → 128 → 27`, cioè
**0.405 MFLOP-MAC per inferenza** e 20.3 MMAC/s a 50 Hz: meno dello **0.5% di un core**. Il conto dei
parametri (406 299 → 1587 KB float32) coincide con i 1 626 343 byte del `policy.onnx` reale, quindi la
stima è verificata. Anche ArUco su un frame 720p costa pochi millisecondi.

**Ripartizione dei processi** (verificata sul codice): `h1_2_ctrl_squat` gira **sul portatile
dell'operatore**, collegato via ethernet al robot, con `ChannelFactory::Init(0, network)` — dominio DDS
**0** — sui topic di default dell'SDK: `rt/lowcmd` (comandi ai 27 motori), `rt/lowstate` (stato + IMU),
`rt/wirelesscontroller` (joystick). La percezione girerà invece **su PC4**, perché è lì che la
telecamera è collegata via USB.

Conseguenza utile: controllo e percezione stanno su **macchine diverse**, quindi non c'è contenzione di
CPU e il jitter sul loop a 50 Hz non è un problema. (Lo sarebbe se entrambi girassero su PC4: in quel
caso servirebbe affinità del controllo su un P-core con `taskset` + `chrt`.)

## 2. I numeri che inquadrano il problema

Misurati per cinematica diretta su **`h1_2_with_FTP_hand.urdf`** (79.54 kg), campionando 30 pose valide
nello spazio di lavoro con la mano davanti a > 32 cm, ad altezza di un tavolo 0.70–0.85 m:

| grandezza | valore |
|---|---|
| massa delle due braccia (spalla→polso) | 12.70 kg |
| massa di **una mano FTP** | **3.278 kg** (di cui 3.023 nel `base_link`) |
| lunghezza del braccio (spalla→mano) | **0.683 m** |
| CoM **sagittale** raggiungendo | mediana **+2.22 cm**, p90 +2.83, peggiore **+3.10 cm → 24.2 N·m** |
| CoM **laterale** raggiungendo | mediana +0.65 cm, p90 +2.36, peggiore **+2.91 cm → 22.7 N·m** |
| contributo del carico da 0.3 kg | +0.10 cm = **4%** dell'effetto totale |

**(a) L'oggetto non conta, conta il braccio.** Il 96% della perturbazione viene da braccio e mano. Non
serve randomizzare il carico: serve **coprire le pose**.

**(b) Lo sbilanciamento laterale NON è trascurabile.** Il caso peggiore è **2.91 cm** contro i ±2 cm
oggi in `randomize_base_com`. Portare il range su `y` a **±3 cm**.

**(c) Raggiungere in avanti spinge il baricentro dalla parte SICURA.** Il poligono è asimmetrico —
punta **+17.3 cm**, tallone **−8.6 cm** — e tutto il progetto ha inseguito un CoM che scappava
*all'indietro*. Allungare il braccio lo porta avanti e carica la caviglia in plantarflessione, il verso
forte. Per la scala: 24 N·m nel caso peggiore è **il 105%** dell'errore di modello del baricentro che è
costato i round 3 e 4 (3.5 cm = 23.0 N·m) — ma qui è **noto e comandato** invece che ignoto.

## 3. Decisioni architetturali

### D1 — Sotto RL va il RAGGIUNGIMENTO; presa e percezione stanno fuori

La policy riceve la **posa dell'oggetto** come osservazione e produce direttamente gli angoli dei
giunti. Non c'è cinematica inversa nel loop di controllo: la ridondanza dei 7 DoF viene risolta
dall'apprendimento con l'obiettivo vero (raggiungi *e* resta in equilibrio), e sbilanciarsi, spostare il
peso e allungarsi diventano un unico comportamento coordinato.

Restano **fuori**, ciascuno per un motivo preciso:

- **la chiusura delle dita** → primitiva apri/chiudi su `HandCmd_`. Le FTP hanno
  `actuatorfrcrange="-1 1"`, cioè 1 N·m per giunto: fanno presa di forza, non manipolazione destra.
  L'RL non aggiungerebbe nulla, e porterebbe dentro il gap sim2real del contatto, il secondo peggiore
  dopo la visione. In simulazione l'oggetto è un **bersaglio da raggiungere, non da afferrare**: niente
  contatto dita-oggetto, e l'oggetto costa quasi nulla nel simulatore.
- **la percezione** → processo separato a 5–10 Hz che pubblica su `rt/object_pose`. Se si pianta, il
  robot resta in piedi sull'ultimo valore valido e `reach_enable` va a 0.

⚠️ Nota: la policy consuma la **posa** (3–4 numeri), non i pixel. In addestramento è ground truth,
quindi **non serve rendering** e il throughput resta quello di oggi (73 924 step/s, 1.85 h per round).

*Riscontro esterno:* Unitree stessa non fa manipolazione con RL. `unitree_lerobot` è apprendimento per
imitazione da teleoperazione (ACT, Diffusion, Pi0, GR00T) sul **solo busto**, senza gambe;
`unitree_sim_isaaclab` è teleoperazione e replay. E `rt/arm_sdk` (vedi D3) è il loro modo di
sovrapporre comandi di braccio a un controllore di locomozione: **lo stesso disaccoppiamento**.

### D2 — Modello a 27 giunti con la massa delle mani accorpata

`h1_2_with_FTP_hand.urdf` resta la **sorgente di verità** per masse e montaggio sensori, ma non è il
modello di addestramento: 88 corpi contro 28, 51 DoF contro 27, **58 mesh di collisione in più**, per
4096 ambienti. E 24 DoF minuscoli (dita da 28–43 g, limiti da 1 N·m) in un solver GPU sono una fonte
nota di instabilità, per giunta mai comandati dalla policy.

**L'accorpamento non è un'approssimazione:** se le dita non si muovono, la mano È un corpo rigido, e un
corpo rigido è descritto esattamente da massa, baricentro e tensore d'inerzia:

```
.*_wrist_yaw_link : + 3.2783 kg
   CoM dal polso    (+0.12018, ∓0.00194, +0.00142) m      [y opposta fra L e R]
   inerzia attorno al proprio CoM (kg m^2):
      +0.000700  -0.000156  -0.000090
      -0.000156  +0.002951  -0.000166
      -0.000090  -0.000166  +0.002559
torso_link        : + 6.000 kg   (elettronica delle mani FTP)
                    -> 79.541 kg, che coincide con l'URDF completo
```

La sola massa puntiforme sbaglierebbe del 4% (inerzia propria 0.00207 contro trasporto 0.04737): col
tensore l'accorpamento è **esatto**. Da generare con uno script sul modello di
`tools/make_com_test_model.py`, con la verifica incorporata che la massa risultante faccia 79.541 kg.

**Perché conta:** i test reali dei round 3 e 4 sono stati fatti **senza mani** (smontate per sicurezza),
quindi il modello a 66.98 kg era corretto e le stime `d = τ/(M·g)` con M = 67 kg restano valide.
Montando le mani il robot passa a **79.54 kg (+18.7%)** e un braccio da 6.35 a **9.63 kg (+52%)**.
`randomize_link_masses` + `add_base_mass` mostrano alla policy **67.97 ± 2.22 kg**: 79.54 sta a
**+5.2 sigma**, zero episodi su 50 000. **La Fase 1 va addestrata con le mani nel modello.**

Il baricentro invece quasi non si muove: −2.46 → −2.08 cm, perché le mani (+3.4 cm dalla caviglia) e
l'elettronica del torso (−3.8 cm) si annullano. **La massa in più non spiega il baricentro arretrato**
misurato sul reale: ipotesi chiusa.

### D3 — I guadagni dei polsi vanno alzati (da `unitree_sdk2`)

`example/h1/high_level/h1_2_arm_sdk_dds_example.cpp` è il riferimento ufficiale per comandare le
braccia dell'H1_2: topic `rt/arm_sdk`, 15 giunti (14 braccia + waist yaw), **`control_dt = 0.02`, cioè
50 Hz come noi**, peso di miscelazione in `motor_cmd[27].q()` rampato a 0.2/s, e
**`max_joint_velocity = 0.5 rad/s`**.

Non possiamo usare `rt/arm_sdk` — è pensato per sovrapporsi al controllore interno di Unitree su PC1,
mentre noi pubblichiamo su `rt/lowcmd` con la nostra policy: sono mutuamente esclusivi. Ma i suoi
**parametri sono tarati sul robot vero** e vanno presi:

| giunto | Unitree `arm_sdk` | nostro `deploy.yaml` | |
|---|---|---|---|
| shoulder_pitch | 120 | 250 | 2.1× |
| shoulder_roll | 120 | 400 | 3.3× |
| shoulder_yaw | 80 | 80 | = |
| elbow | 50 | 80 | 1.6× |
| **wrist ×3** | **50** | **20** | **0.4×** |
| torso | 200 | 600 | 3.0× |

I nostri polsi sono **2.5× più cedevoli**. Oggi è irrilevante (senza mani la coppia sul polso è
~0.05 N·m), ma con la mano FTP montata:

```
coppia sul polso = 3.278 kg x 9.81 x 0.120 m = 3.87 Nm
   kp = 20 (nostro)  -> cedimento 11.1 deg -> 3.9 cm di errore sulla punta delle dita
   kp = 50 (Unitree) -> cedimento  4.4 deg -> 1.5 cm
   kp = 80           -> cedimento  2.8 deg -> 1.0 cm
```

**11° di cedimento del polso rendono impossibile una presa al centimetro.** È lo stesso meccanismo del
cedimento di caviglia dei round 3–4, trasferito al polso — ma stavolta lo vediamo *prima*. I guadagni
vanno alzati **in addestramento e nel deploy insieme**, come parte della Fase 1.

Adottare anche il **limite di velocità di 0.5 rad/s** sul movimento del braccio (0.57° per passo a
50 Hz; un raggiungimento da 1.5 rad richiede 3 s), identico in addestramento e al deploy.

### D4 — La posa dell'oggetto va addestrata IMPERFETTA *(il punto che decide tutto)*

La policy dipende dalla posa dell'oggetto **a ogni passo di controllo**. In simulazione quella posa
sarebbe perfetta, istantanea e sempre presente; sul robot no. Addestrare con la posa perfetta insegna a
fidarsene ciecamente — ed è **letteralmente la storia del baricentro**, dove una perturbazione non
modellata è costata due round.

| imperfezione | come modellarla | perché |
|---|---|---|
| rumore | σ 1–2 cm sulla posizione, qualche grado sullo yaw | ArUco + profondità non danno millimetri |
| **frequenza** | aggiornamento a 5–10 Hz, tenuta dell'ultimo valore | il controllo gira a **50 Hz**: la policy vede lo stesso numero per 5–10 passi |
| latenza | 100–200 ms | acquisizione, elaborazione, DDS |
| dropout | perdita casuale di aggiornamenti | il marker esce dal campo, il clustering fallisce |
| **occlusione finale** | congelamento dell'ultima posa valida quando la mano si avvicina | negli ultimi 5–10 cm la mano copre l'oggetto: **succede sempre** |

L'ultima riga è la più importante e la più facile da dimenticare: senza di essa la policy imparerebbe a
fare la correzione finale su un'informazione che sul robot vero non avrà.

Trattata così, la posa dell'oggetto diventa una quantità **inferita dalla storia** — lo stesso
meccanismo con cui la policy ha imparato a compensare il 69% dello spostamento del baricentro.

### D5 — Diagnosticabilità senza cinematica inversa nel loop

Rinunciando all'IK si perdono due gate indipendenti (l'IK verificabile al millimetro, l'inseguimento del
braccio misurabile da solo). Si recuperano per altra via, e queste tre cose vanno previste **da subito**:

1. **Registrare l'errore di raggiungimento.** Cinematica diretta sui giunti misurati → posa della mano;
   confrontata con la posa dell'oggetto → errore campione per campione. Come già facciamo per l'altezza
   e per la coppia di caviglia.
2. **Calcolare l'IK offline come riferimento, in analisi** (non nel loop). Se la mano manca di 5 cm,
   l'IK dice se quella posa era raggiungibile: separa *"la policy ha raggiunto male"* da *"il bersaglio
   era fuori portata"*.
3. **Iniettare la posa a mano nei test reali.** Prima di collegare la percezione, pubblicare su
   `rt/object_pose` una posa misurata a metro: isola gli errori di controllo da quelli di percezione.
   È l'equivalente del modello con il CoM spostato — un esperimento che discrimina.

## 4. Fasi

### Fase 0 — Prerequisito
Chiusura del round 5 sullo squat (togliere il gate di profondità ad `ankle_torque_margin`).
**Non si apre questo fronte con il baricentro ancora aperto sul reale.**

### Fase 1 — Raggiungimento sotto RL *(la fase che decide il progetto)*

**Modello:** variante `h1_2_handless_ftp.urdf` generata da script (D2), 79.541 kg. Guadagni dei polsi
alzati a 50 (D3). `randomize_base_com` con `y` a ±3 cm (§2b). Stesso trattamento sull'MJCF per il
sim2sim.

**Azioni: invariate**, 27 `JointPositionAction`. Le dita non sono azioni.

**Osservazioni: 465 → 490**

| termine | dim | |
|---|---|---|
| gli 8 termini attuali | 93 | invariati |
| `object_pose_rel` (posizione nel frame base) | 3 | nuovo |
| yaw dell'oggetto | 1 | nuovo (omettibile se a simmetria di rotazione) |
| `reach_enable` | 1 | nuovo |
| | **98 × 5 = 490** | |

`reach_enable` è il **fallback sicuro**: a 0 il braccio torna a riposo e il robot cammina e squatta come
oggi; al deploy va a 0 quando la percezione tace oltre il timeout.

**Critico (privilegiato):** posa vera dell'oggetto, posa vera della mano, errore mano-oggetto.

**Reward:**
- `reach_object` — esponenziale sulla distanza mano-oggetto, gated su `reach_enable`;
- `hand_orientation` — allineamento del palmo alla direzione di avvicinamento (senza, la mano arriva
  sul punto giusto con l'orientamento sbagliato e la presa fallisce);
- `joint_deviation_arms` — gated su **NON** raggiungimento, così il braccio torna a riposo da solo;
- tutto il resto invariato. *Sinergia:* `ankle_torque_margin` spinge a ricentrare il baricentro
  arretrando i fianchi, cioè esattamente la compensazione che serve quando il braccio si allunga.

**Curriculum `reach_levels`**, sullo schema di `height_cmd_levels`: si attiva solo quando
`track_lin_vel_xy` **e** `track_base_height` sono buoni, poi allarga la scatola di posizioni
dell'oggetto dal punto centrale a tutto lo spazio di lavoro.

**Scena:** tavolo e oggetto per ambiente, posizione randomizzata. Nessun marker ArUco in sim (la posa è
ground truth), nessun contatto dita-oggetto. Modello di percezione imperfetta secondo D4.

**Gate:** in sim2sim errore mano-oggetto **< 2 cm** su tutto lo spazio di lavoro, con la posa
imperfetta; **nessuna regressione** su `track_lin_vel_xy`, guadagno yaw e `track_base_height` rispetto
al round 5; poi lo stesso sul robot reale con la posa iniettata a mano (D5.3).
**Se questa fase non passa, il resto non serve.**

### Fase 2 — Mani
Le FTP hanno 12 giunti per mano e — la parte che conta — **17 link di sensore di forza per mano**
(palmo più 3–4 falangi per dito). Il rilevamento della presa diventa **tattile invece che visivo**,
quindi immune all'occlusione della mano. È il segnale con cui l'FSM passa da `Grasp` a `Retreat` e con
cui si accorge se l'oggetto scivola. Canale DDS separato, primitive apri/chiudi, nessun RL.

⚠️ **Interfaccia da verificare empiricamente.** Nessun repository Unitree copre H1_2 + FTP:
`unitree_sim_isaaclab` ha H1-2 con mani **Inspire**, `unitree_lerobot` non supporta l'H1_2, e
`g1_hand_sdk_example.cpp` usa `rt/hand_sdk` sul G1. `HandCmd_` contiene un `std::vector<MotorCmd_>` a
lunghezza variabile, quindi il messaggio si adatta, ma topic e semantica per le FTP non sono
documentati. Un pomeriggio di ispezione dei topic DDS sul robot **prima** di progettare questa fase:
quale topic pubblica lo stato delle mani, quanti motori riporta, se accetta posizione o forza, e come si
leggono i sensori (probabilmente `PressSensorState_`).

**Gate:** l'oggetto resta in mano stando in piedi e camminando, e il contatto è rilevato dai sensori di
forza senza ricorrere alla visione.

### Fase 3 — Percezione

**La telecamera è già nel modello del robot** (l'MJCF no, l'URDF FTP sì):

```
camera_joint  su torso_link, xyz = (0.111, 0.018, 0.688), pitch +50.8 deg verso il basso
lidar_joint   su torso_link, xyz = (0.049, 0.000, 0.680), pitch +13.8 deg
```

Col bacino a 0.98 m la telecamera sta a **1.67 m**; su un tavolo a 0.75 m l'asse ottico cade a
**0.86 m** davanti al bacino e (assumendo FOV verticale ~58°, tipico D435i — **da verificare**) il
piano è inquadrato **da 0.28 a 2.41 m**. L'oggetto a 0.45 m è dentro con margine.

**Il marker NON va sull'oggetto, va sul tavolo.** I due ruoli si separano:

| ruolo | strumento | perché |
|---|---|---|
| riferimento e piano del tavolo | **marker 10–15 cm sul tavolo**, fuori dalla zona di presa | fisso, mai occluso, rende la calibrazione mano-occhio verificabile in isolamento |
| posizione dell'oggetto | **profondità RGB-D**: piano per RANSAC + clustering dei punti sopra | zero addestramento, su CPU, non richiede di toccare l'oggetto |

Un marker *sull'oggetto* è la seconda scelta: la precisione degrada col rapporto distanza/lato (regola
pratica: lato ≈ 1/10 della distanza, cioè ~10 cm a 1 m — non ci stanno su un oggetto piccolo);
l'ambiguità di posa planare rende l'orientamento rumoroso; e **la mano lo occlude proprio negli ultimi
centimetri**. Eccezione legittima: se per la demo si **fabbrica** l'oggetto (un cubo leggero con marker
su più facce), è la scorciatoia più rapida.

**Semplificazione che regge la fase:** con l'oggetto su un piano di altezza nota la posa ha **3 DoF**
(x, y, yaw), e per un oggetto a simmetria di rotazione solo **2**.

**Un solo script, tre sorgenti di posa.** Il rendering è impraticabile in addestramento (4096
ambienti) ma **gratuito in sim2sim**, dove gira un robot solo in tempo reale. Quindi il sim2sim diventa
il posto dove la percezione si valida davvero:

| | sorgente immagini | posa |
|---|---|---|
| addestramento | — | ground truth IsaacLab + modello D4 |
| **sim2sim** | **renderer MuJoCo** | **percezione vera, stesso codice del reale** |
| reale | RealSense | percezione vera |

Lo script si scrive **una volta**, con un'astrazione sulla sorgente immagini. In sim2sim hai anche la
posa vera dell'oggetto dal simulatore, quindi puoi misurare l'errore della tua percezione **contro la
verità** — un gate che sul robot reale non potresti mai avere. Stesso schema del modello col CoM
spostato: un esperimento che discrimina, fatto a terra.

Servono due aggiunte al MJCF: la telecamera col montaggio dell'URDF, e (se si testa ArUco) una texture
del marker sul piano del tavolo.

**Realizzazione:**

- **linguaggio: Python.** `opencv-python` (ArUco), `pyrealsense2`, `numpy`/`open3d` per piano e
  clustering. Il carico è di pochi ms per frame a 5–10 Hz: il C++ non darebbe nulla. Per il DDS c'è
  `unitree_sdk2_python`, già presente in `~/unitree_ws/`.
- **macchina: PC4**, dove la telecamera è collegata (da verificare con `lsusb | grep -i intel` su
  192.168.123.164).
- **topic: `rt/object_pose`**, accanto a `rt/lowcmd` e `rt/lowstate`, **stesso dominio 0**. I topic DDS
  sono indipendenti: un partecipante in più non disturba gli altri. Due condizioni: il dominio deve
  coincidere (`ChannelFactory.Initialize(0, iface)` anche lato Python) e **il tipo di messaggio deve
  essere identico sulle due sponde** — con IDL diversi il match non avviene e *non arriva alcun errore*,
  semplicemente non ricevi nulla, che è il modo peggiore di sbagliare.
- **messaggio:** scorciatoia consigliata per il primo giro — riusare un tipo già nell'SDK invece di
  generare un IDL nuovo su entrambi i lati. `MotorStates_` è un vettore a lunghezza variabile in cui
  stanno comodamente x, y, z, yaw, confidenza e timestamp. Brutto da leggere, ma funzionante in un
  pomeriggio; l'IDL pulito si definisce quando il progetto si consolida.
- ⚠️ **timestamp e flag di validità sono obbligatori** nel messaggio: il controller ne ha bisogno per il
  timeout che porta `reach_enable` a 0 e per replicare la logica di congelamento con cui la policy è
  stata addestrata (D4). Senza timestamp non distingui "posa vecchia di 2 secondi" da "posa appena
  aggiornata", e quella distinzione **è** la sicurezza del sistema.

**Il LiDAR resta fuori.** Il Livox MID-360 è uno scanner rado per navigazione e mappatura: su un
oggetto piccolo a mezzo metro dà pochissimi punti, e la RGB-D lì è molto più densa. Aggiunge
complessità senza migliorare la stima.

**Gate:** errore di stima < 2 cm contro la ground truth del **simulatore** (misurabile in sim2sim), e
ripetibilità equivalente sul reale con l'oggetto su un punto misurato a metro.

### Fase 4 — Orchestrazione FSM
Nuovi stati (`Reach`, `Grasp`, `Retreat`) con transizioni joystick, coerenti con l'FSM attuale
(`LT + B` → Passive, `RB + X` → Squat): `RB + Y` → Reach. Il framework è minuscolo e data-driven — 6
header in `deploy/include/FSM/` e `State_RLBase.cpp` è **31 righe** — quindi qui è configurazione.

## 5. Rischi, in ordine

| # | rischio | livello | mitigazione |
|---|---|---|---|
| 1 | Modello di percezione imperfetta insufficiente | **alto** | D4: rumore, frequenza, latenza, dropout, congelamento |
| 2 | Precisione del raggiungimento (leva di 0.68 m) | **alto** | Servono < 1.7° per giunto per 2 cm alla mano. Gate di Fase 1 |
| 3 | Cedimento del polso con la mano FTP | **alto** | D3: kp da 20 a 50, in training e deploy insieme |
| 4 | Regressione della locomozione | medio | Gate di Fase 1 contro le metriche del round 5 |
| 5 | Interferenza fra i tre curriculum | medio | §6: renderli salvabili nel checkpoint prima di arrivarci |
| 6 | Interfaccia DDS delle mani FTP ignota | medio | Ispezione empirica prima della Fase 2 |
| 7 | Calibrazione mano-occhio | medio | Marker fisso sul tavolo, verificabile in isolamento |
| 7b | Mismatch del tipo di messaggio DDS | basso ma insidioso | Fallisce in silenzio: verificare la ricezione prima di ogni altra cosa |
| 8 | Equilibrio con braccio esteso | **basso** | 24 N·m nel caso peggiore, verso sicuro del poligono (§2c) |
| 9 | Presa | basso | Primitive apri/chiudi con conferma tattile |

**Sui curriculum (rischio 5):** avresti tre stadi — terreno, altezza, raggiungimento. Al round 4 quando
quello dell'altezza avanzava `terrain_levels` è crollato da 4.5 a 2.4 prima di risalire. E i curriculum
**non stanno nel checkpoint**: al resume del round 4 si sono perse 375 iterazioni a ricalcare la scala.
Con tre curriculum quel costo triplica: **renderli salvabili è un prerequisito, non un miglioramento.**

## 6. Cosa NON facciamo (e perché)

- **Presa sotto RL** — le dita hanno `actuatorfrcrange="-1 1"` (1 N·m): presa di forza, non
  manipolazione destra. L'RL non aggiungerebbe nulla e porterebbe dentro il gap sim2real del contatto.
  La ricompensa sarebbe inoltre sparsa, e il rimedio standard (shaping verso una posa desiderata della
  mano) **reintrodurrebbe la cinematica inversa** come reward shaping, per giunta approssimata.
- **Visione nella policy** — un RGB-D 64×64 sono 16 384 valori per frame, **35×** l'osservazione
  attuale, e richiede rendering in addestramento (crollo del throughput) e una CNN al deploy su una
  macchina **senza CUDA** che deve garantire il passo a 50 Hz.
- **URDF a 51 giunti come modello di addestramento** — vedi D2. Resta sorgente di verità.
- **`rt/arm_sdk`** — mutuamente esclusivo con la nostra policy su `rt/lowcmd`. Ne prendiamo i parametri
  (D3), non il meccanismo.
- **Contatto dita-oggetto in simulazione** — l'oggetto è un bersaglio da raggiungere; la presa è una
  primitiva innescata dall'FSM e confermata dai sensori tattili.
- **Randomizzazione del carico** — vale il 4% dell'effetto (§2a). Si randomizzano le pose, non il peso.
- **Navigazione autonoma verso il tavolo** — fuori perimetro: l'avvicinamento resta a joystick.
