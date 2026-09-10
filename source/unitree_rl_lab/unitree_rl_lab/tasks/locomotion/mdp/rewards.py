from __future__ import annotations

import torch
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from .commands.velocity_command import command_gate_threshold

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
Joint penalties.
"""


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def stand_still(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < 0.1)


def stand_still_zero_cmd(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint deviation from default pose ONLY on the true standing envs.

    Gates on the command term's ``is_standing_env`` mask (envs whose command is forced to zero),
    NOT on ``cmd_norm < threshold``. This avoids penalizing the leg motion needed to TRACK small
    commands, so it never deadlocks with the command curriculum. Meant for anti-sway finetuning.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    is_standing = env.command_manager.get_term(command_name).is_standing_env
    return reward * is_standing


def joint_action_rate_standing(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Action-rate (first difference) on a joint SUBSET, ONLY on the true standing envs.

    Targets the standing ankle limit-cycle: it lives in the ACTION (the policy commands an
    oscillating target) but NOT in the measured joint position (the joint is too soft, kp low, to
    follow it), so position-based penalties (stand_still) miss it entirely. Restricting to a few
    joints and gating on ``is_standing_env`` means:
      - it does NOT penalize the leg swing needed while walking;
      - the slow, ramped squat descent has tiny action-rate, so it does not fight the squat, while the
        fast (~0.8 Hz) limit-cycle has large per-step action change and IS penalized.

    NOTE: assumes the action space is JointPositionAction over ``joint_names=[".*"]`` so that the
    action index equals the articulation joint index (thus ``asset_cfg.joint_ids`` indexes the action).
    """
    idx = asset_cfg.joint_ids
    da = env.action_manager.action[:, idx] - env.action_manager.prev_action[:, idx]
    is_standing = env.command_manager.get_term(command_name).is_standing_env
    return torch.sum(torch.square(da), dim=1) * is_standing


def ankle_torque_standing(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    height_command_name: str = "base_height",
    height_tol: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize STEADY ankle torque while standing tall -> keeps the CoM over the ankles.

    Rationale (measured on the real robot): at equilibrium the ankle torque IS the horizontal
    offset of the centre of mass, ``tau = M*g*d``. A posture that needs little ankle torque is a
    posture whose CoM sits over the ankle, i.e. with the MAXIMUM margin before tipping onto the
    heels. Nothing in the reward set rewarded that margin, and the real robot stood with the CoM
    4.5 cm behind the ankle -- 4.1 cm from the heel edge -- which the squat then consumed.

    It also attacks the mechanism directly: the foot tilt was proportional to the ankle tracking
    error (r = 0.980), and that error is exactly ``tau / kp``.

    Gated on standing envs AND on being SETTLED at the commanded height -- at ANY depth, not only
    near the nominal one. The original version also required a near-nominal height command, on the
    assumption that "in a squat a large ankle torque is legitimate and unavoidable". The CoM-shifted
    sim2sim of 2026-09-06 disproved that: on the nominal model the policy holds full squat (0.60 m)
    with only -2.1 N*m of total ankle torque, and where this term is ACTIVE it absorbs 69% of an
    injected CoM offset against 43% where it is gated off. The gate was leaving performance on the
    table exactly where the stability margin is thinnest.

    The settle gate (|h - h_cmd| < height_tol) still keeps the term out of the squat TRANSITION,
    where a large ankle torque really is unavoidable.

    NOTE: uses L1 (|tau|) and not L2 on purpose. The goal is to shape the STEADY-STATE posture,
    not to punish transients: with the square, a 60 Nm recovery spike would cost ~0.7 (about 5x
    the ``alive`` bonus) and would discourage the robot from catching itself. L1 keeps the same
    nominal pressure while staying ~10x gentler on spikes.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    is_standing = env.command_manager.get_term(command_name).is_standing_env
    height_cmd = env.command_manager.get_command(height_command_name).squeeze(-1)
    settled = torch.abs(asset.data.root_pos_w[:, 2] - height_cmd) < height_tol
    return torch.sum(torch.abs(torque), dim=1) * is_standing * settled


def flat_orientation_asym(
    env: ManagerBasedRLEnv,
    height_command_name: str = "base_height",
    nominal_height: float = 0.98,
    deepest_height: float = 0.60,
    forward_deadband: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Asymmetric replacement for ``flat_orientation_l2``: free forward lean while squatting.

    WHY (support polygon geometry, measured from the MJCF): the foot spans +17.3 cm at the toe
    but only -8.6 cm at the heel about the ankle. Leaning BACKWARD is therefore twice as costly
    in stability margin as leaning forward, and forward lean is the RECOVERY direction when the
    centre of mass sits behind the ankles (the failure mode measured on the real robot).
    Penalizing both directions equally, as ``flat_orientation_l2`` does, contradicts that
    asymmetry: it taxes the very posture the deep squat needs.

    Shape:
      * roll (``g_y``) is penalized quadratically at all times -- it is never useful;
      * backward pitch (``g_x < 0``) is penalized quadratically at all times;
      * forward pitch (``g_x > 0``) is free inside a dead-band that grows LINEARLY with the
        commanded squat depth, and quadratic beyond it.

    At the nominal standing height the dead-band is exactly 0, so this term is bit-for-bit
    identical to ``flat_orientation_l2`` while standing and walking tall -- the walking posture
    learned so far is preserved, and there is no discontinuity when the squat starts.

    Args:
        forward_deadband: dead-band at the DEEPEST squat, in units of ``g_x`` (= sin(pitch)).
            0.35 ~= 20 deg. Measured reference: the policy leans +6.5 deg at full squat in sim.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    grav = asset.data.projected_gravity_b
    # g_x > 0 <=> nose-down (forward) lean; g_x < 0 <=> backward lean
    gx, gy = grav[:, 0], grav[:, 1]

    height_cmd = env.command_manager.get_command(height_command_name).squeeze(-1)
    depth_frac = ((nominal_height - height_cmd) / (nominal_height - deepest_height)).clamp(0.0, 1.0)
    deadband = forward_deadband * depth_frac

    backward = gx.clamp(max=0.0)
    forward_excess = (gx - deadband).clamp(min=0.0)
    return gy.square() + backward.square() + forward_excess.square()


"""
Squat (base height) rewards.
"""


def track_base_height_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "base_height",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track the commanded base height with an exponential kernel: exp(-(h - h_cmd)^2 / std^2).

    Uses the TRUE base height (privileged) in the reward -- fine, rewards are training-only.
    This is the primary squat task term.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    target = env.command_manager.get_command(command_name).squeeze(-1)  # (N,)
    base_h = asset.data.root_pos_w[:, 2]
    error = torch.square(base_h - target)
    return torch.exp(-error / std**2)


def lin_vel_z_gated(
    env: ManagerBasedRLEnv,
    command_name: str = "base_height",
    height_tol: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize vertical base velocity ONLY when already near the commanded height.

    During the squat transition the base MUST move vertically, so penalizing vz there would
    fight the task. This term activates (penalizes vz^2) only once |h - h_cmd| < height_tol,
    encouraging the robot to settle and hold the target height without bouncing.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    target = env.command_manager.get_command(command_name).squeeze(-1)
    base_h = asset.data.root_pos_w[:, 2]
    near_target = (torch.abs(base_h - target) < height_tol).float()
    return torch.square(asset.data.root_lin_vel_b[:, 2]) * near_target


"""
Robot.
"""


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)  # cosine distance
    normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
    return torch.square(normalized)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


"""
Feet rewards.
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footpos_translated[:, i, :])
        footvel_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footvel_translated[:, i, :])
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > command_gate_threshold(
        env, command_name, 0.1
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """
    Reward for feet contact when the command is zero.
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    # ROUND 7: soglia proporzionale alla scala di profondita'. Con la soglia fissa questo termine
    # si accendeva su tutti gli env accovacciati che AVEVANO un comando di cammino (sotto 0.73 m
    # il laterale a fondo corsa valeva 0.109), pagandoli 0.5/step per tenere i piedi piantati
    # mentre track_lin_vel ne metteva in palio 0.135: vinceva stare fermi.
    return reward * (command_norm < command_gate_threshold(env, command_name, 0.1))


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


"""
Feet Gait rewards.
"""


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > command_gate_threshold(env, command_name, 0.1)
    return reward


"""
Other rewards.
"""


def joint_mirror(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    mirror_joints: list[list[str]],
    command_name: str | None = None,
) -> torch.Tensor:
    """Penalize L/R asymmetry ``(L-R)^2`` on joint pairs that mirror with the SAME sign.

    ``command_name`` (optional): if given, the penalty is applied ONLY on the standing envs of
    that velocity command. This matters for walking: a gait is ANTI-PHASE by construction, so an
    ungated ``(L-R)^2`` penalizes exactly what makes stepping work. Measured on the 2026-09-04
    run: the term sat at -0.023/-0.031, i.e. actively violated by the gait, against a
    ``track_lin_vel_xy`` of +0.73 (~4% drag). Gating it on standing envs keeps the intent
    (a symmetric squat) and removes the conflict.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    if command_name is not None:
        reward = reward * env.command_manager.get_term(command_name).is_standing_env
    return reward


def joint_mirror_antisym(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    """L/R symmetry for joints whose mirrored configuration is ANTI-symmetric (left = -right).

    Use for ROLL joints (hip_roll, ankle_roll), where a symmetric stance means left and right hold
    OPPOSITE values (verified in the logs: standing L hip_roll -0.012 vs R +0.010). ``joint_mirror``
    penalizes ``(L-R)^2`` and is for joints that mirror with the SAME sign (hip_pitch, knee,
    ankle_pitch); using it on roll joints would penalize the correct posture.

    Do NOT apply to hip_yaw: turning legitimately needs L/R yaw asymmetry.

    NOTE: keeps its OWN cache attribute -- ``joint_mirror`` stores its cache on the env under a
    single name, so a second term reusing that function would silently share the first term's joints.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_antimirror_joints_cache") or env.joint_antimirror_joints_cache is None:
        env.joint_antimirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    for joint_pair in env.joint_antimirror_joints_cache:
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] + asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward


def heading_hold(env: ManagerBasedRLEnv, command_name: str = "base_velocity", std: float = 0.25) -> torch.Tensor:
    """Reward per la RETTILINEITA' angolare: errore di imbardata INTEGRATO sulla finestra di comando.

    Misurato in sim2sim (sim_2026-09-10_18-08-39), regredendo il rate di imbardata sui bout SENZA
    comando di yaw:

        omega_deriva [deg/s] ~= +0.72 - 4.45*vx + 6.52*vy      (residuo RMS 0.80)

    cioe' un bias costante piu' un accoppiamento con la direzione di marcia. Camminando di lato a
    0.30 m/s la deriva vale +2.7 deg/s: in 10 s sono 27 gradi, che curvano visibilmente la
    traiettoria (fino a 52 cm di scostamento su 2.3 m percorsi). Bias e coefficienti cambiano
    segno da una policy all'altra a parita' di modello MuJoCo -> e' la policy, non il simulatore.

    Perche' INTEGRATO e non istantaneo: scomponendo il rate sulle due semifasi del passo, le due
    meta' del ciclo differiscono di 2-7 deg/s, cioe' MOLTO PIU' della deriva netta. La deriva e'
    il residuo non cancellato di quell'oscillazione. Una penalita' sul rate punirebbe soprattutto
    l'andatura sana; sull'integrale l'oscillazione si media a zero e resta solo la deriva.

    Perche' i termini esistenti non lo vedono: con std 0.5 e 0.25, una deriva di 2 deg/s
    (0.035 rad/s) mette in palio 0.0146/step su ~115 di reward totale, cioe' lo 0.013%. E'
    sotto la risoluzione del reward, non un problema di peso.

    ``std`` in radianti sull'errore accumulato: 0.25 rad = 14 gradi. Con una deriva di 2 deg/s e
    una finestra di 10 s l'errore finale e' 0.35 rad e la reward media sulla finestra scende a
    ~0.60 contro 1.0 di una policy senza deriva -- il 40% in palio, contro lo 0.013% di prima.
    """
    term = env.command_manager.get_term(command_name)
    return torch.exp(-torch.square(term.heading_err) / std**2)


def path_hold(env: ManagerBasedRLEnv, command_name: str = "base_velocity", std: float = 0.20) -> torch.Tensor:
    """Reward per la RETTILINEITA' lineare: scostamento laterale INTEGRATO dalla traiettoria comandata.

    Sovrapposto a ogni modalita' di camminata, in sim2sim si misura un creep costante nel
    riferimento del corpo di (+0.02 avanti, -0.02 a destra) m/s, indipendente dal comando e dal suo
    segno, assente da fermo. Vale il 7-11% del comando nominale.

    E' anche la spiegazione del giro sul posto: velocita' costante nel body frame + omega costante
    = arco di cerchio di raggio |v|/omega. Previsto 0.108 e 0.093 m, spostamento netto previsto
    0.209 e 0.175 m su 152 e 139 gradi di rotazione; misurato 0.208 e 0.182. Accordo entro il 4%.

    L'accumulo (in ``UniformLevelVelocityCommand._accumulate_drift``) usa solo la componente
    PERPENDICOLARE al comando: quella parallela e' il normale ritardo di inseguimento e
    penalizzarla renderebbe questo un track_lin_vel piu' duro invece che un termine di
    rettilineita'.

    ``std`` in metri: 0.20 m. Con il creep misurato l'errore a fine finestra e' 0.2 m e la reward
    media sulla finestra ~0.75, cioe' il 25% in palio.
    """
    term = env.command_manager.get_term(command_name)
    return torch.exp(-torch.sum(torch.square(term.path_err), dim=1) / std**2)


def lateral_drift_on_turn(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    lin_cmd_thresh: float = 0.1,
    ang_cmd_thresh: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize spurious LATERAL body velocity while the command is a pure turn (yaw only).

    Turning in place was the most destabilizing manoeuvre in sim2sim: with ``wz=-0.5`` and zero
    linear command the robot lurched sideways at up to 0.45 m/s and rolled to -9.1 deg, instead of
    rotating on the spot. Gated so it is active ONLY when the linear command is ~0 and the yaw
    command is significant -> it never fights a commanded lateral motion.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    lin_cmd_norm = torch.norm(command[:, :2], dim=1)
    # "comando lineare ~ zero" e' una soglia sul comando -> scala con la profondita', altrimenti a
    # meta' squat un comando lineare piccolo-ma-vero verrebbe scambiato per assenza di comando.
    # "yaw significativo" resta assoluta: e' una soglia sull'ENTITA' della manovra da penalizzare.
    pure_turn = (lin_cmd_norm < command_gate_threshold(env, command_name, lin_cmd_thresh)) & (
        torch.abs(command[:, 2]) > ang_cmd_thresh
    )
    lateral_vel = asset.data.root_lin_vel_b[:, 1]
    return torch.square(lateral_vel) * pure_turn
