// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "isaaclab/envs/manager_based_rl_env.h"

#include <algorithm>
#include <cmath>
#include <spdlog/spdlog.h>

namespace isaaclab
{
namespace mdp
{

REGISTER_OBSERVATION(base_ang_vel)
{
    auto & asset = env->robot;
    auto & data = asset->data.root_ang_vel_b;
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(projected_gravity)
{
    auto & asset = env->robot;
    auto & data = asset->data.projected_gravity_b;
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(joint_pos)
{
    auto & asset = env->robot;
    std::vector<float> data;

    std::vector<int> joint_ids;
    try {
        joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();
    } catch(const std::exception& e) {
    }

    if(joint_ids.empty())
    {
        data.resize(asset->data.joint_pos.size());
        for(size_t i = 0; i < asset->data.joint_pos.size(); ++i)
        {
            data[i] = asset->data.joint_pos[i];
        }
    }
    else
    {
        data.resize(joint_ids.size());
        for(size_t i = 0; i < joint_ids.size(); ++i)
        {
            data[i] = asset->data.joint_pos[joint_ids[i]];
        }
    }

    return data;
}

REGISTER_OBSERVATION(joint_pos_rel)
{
    auto & asset = env->robot;
    std::vector<float> data;

    data.resize(asset->data.joint_pos.size());
    for(size_t i = 0; i < asset->data.joint_pos.size(); ++i) {
        data[i] = asset->data.joint_pos[i] - asset->data.default_joint_pos[i];
    }

    try {
        std::vector<int> joint_ids;
        joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();
        if(!joint_ids.empty()) {
            std::vector<float> tmp_data;
            tmp_data.resize(joint_ids.size());
            for(size_t i = 0; i < joint_ids.size(); ++i){
                tmp_data[i] = data[joint_ids[i]];
            }
            data = tmp_data;
        }
    } catch(const std::exception& e) {
    
    }

    return data;
}

REGISTER_OBSERVATION(joint_vel_rel)
{
    auto & asset = env->robot;
    auto data = asset->data.joint_vel;

    try {
        const std::vector<int> joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();

        if(!joint_ids.empty()) {
            data.resize(joint_ids.size());
            for(size_t i = 0; i < joint_ids.size(); ++i) {
                data[i] = asset->data.joint_vel[joint_ids[i]];
            }
        }
    } catch(const std::exception& e) {
    }
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(last_action)
{
    auto data = env->action_manager->action();
    return std::vector<float>(data.data(), data.data() + data.size());
};

// Fattore di scala del comando di velocita' in funzione della profondita' dello squat.
// DEVE essere identico a UniformLevelVelocityCommand._update_command nel training: i due numeri
// arrivano da deploy.yaml (generato dalla config di training), quindi la formula sta scritta una
// volta sola e non puo' divergere. Se `speed_scale` manca -- policy addestrata prima di questa
// modifica -- il fattore e' 1.0 e il comportamento e' quello di prima.
//
// Perche' (misurato in sim2sim, protocollo completo a tre quote):
//     quota   vx avanti  vx indietro   vy     wz
//     0.98      1.06        0.80      0.79   0.69
//     0.78      0.82        0.69       --    0.58
//     0.60      0.39        0.18      0.03   0.05
// A fondo squat la locomozione non esiste: chiedergliela lo porta nella sua condizione meno
// stabile di tutta la sessione (sway 0.057, roll -4.9 deg) senza che si muova.
//
// ROUND 7: la curva e' NON LINEARE (esponente esportato, 0.5 = radice). Con la rampa lineare a
// 0.74 m il fattore valeva 0.364 e il comando laterale/indietro a fondo corsa scendeva a 0.109,
// appena sopra la soglia 0.1 del clock del passo -- e sotto 0.713 m sempre sotto. Misurato a
// 0.74 m: guadagno 0.79 in avanti contro 0.09 indietro, 0.13-0.23 laterale, 0.35-0.40 in
// imbardata. Con esponente 0.5 il fattore a 0.74 diventa 0.60. Estremi invariati.
//
// ROUND 9: CUTOFF, non scaling. Lo scaling e' stato tolto dal training (riduceva per costruzione
// la mobilita' a mezzo squat: vx avanti a 0.75 m da 0.94 a comando pieno a 0.33/0.52 scalato). Al
// suo posto `cutoff_height`, esportato da export_deploy_cfg a partire da min_height_while_walking
// del training: sotto quella quota la policy NON ha mai visto un comando di velocita' non nullo
// (gli env che camminano non scendono sotto 0.70), quindi il controller lo azzera e il deploy resta
// dentro la distribuzione di training. Sopra, comando pieno come in training. Vale per entrambi i
// chiamanti (velocity_commands e gait_phase): con comando 0 il clock del passo resta congelato.
// Se manca la chiave -- policy senza pavimento -- nessun cutoff.
inline float squat_speed_scale(isaaclab::ManagerBasedRLEnv * env)
{
    const auto cmds = env->cfg["commands"]["base_velocity"];
    if (cmds["cutoff_height"] && env->squat_target < cmds["cutoff_height"].as<float>())
        return 0.0f;
    if (!cmds["speed_scale"])
        return 1.0f;

    const float h0 = cmds["speed_scale"]["zero_height"].as<float>();
    const float h1 = cmds["speed_scale"]["full_height"].as<float>();
    if (h1 <= h0)
        return 1.0f;

    const float s = std::clamp((env->squat_target - h0) / (h1 - h0), 0.0f, 1.0f);
    const float e = cmds["speed_scale"]["exponent"] ? cmds["speed_scale"]["exponent"].as<float>() : 1.0f;
    return (e == 1.0f) ? s : std::pow(s, e);
}

// Soglia "comando ~ zero" del clock del passo, proporzionale alla scala di profondita'.
// DEVE combaciare con command_gate_threshold nel training: la soglia fissa a 0.1 era tarata sulla
// stazione eretta e con lo scaling apriva una fascia (0.62-0.73 m) in cui il robot riceveva un
// target di velocita' non nullo con il clock congelato. Il pavimento serve a NON annullare la
// soglia a fondo squat, dove la scala e' 0: li' il comando e' identicamente 0 e il clock deve
// restare congelato come prima.
inline float squat_gait_threshold(isaaclab::ManagerBasedRLEnv * env, float base_threshold)
{
    const auto cmds = env->cfg["commands"]["base_velocity"];
    if (!cmds["speed_scale"])
        return base_threshold;

    const float floor_ = cmds["speed_scale"]["threshold_min_scale"]
                             ? cmds["speed_scale"]["threshold_min_scale"].as<float>()
                             : 0.2f;
    return base_threshold * std::max(squat_speed_scale(env), floor_);
}

REGISTER_OBSERVATION(velocity_commands)
{
    std::vector<float> obs(3);
    auto & joystick = env->robot->data.joystick;

    const auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    obs[0] = std::clamp(joystick->ly(), cfg["lin_vel_x"][0].as<float>(), cfg["lin_vel_x"][1].as<float>());
    obs[1] = std::clamp(-joystick->lx(), cfg["lin_vel_y"][0].as<float>(), cfg["lin_vel_y"][1].as<float>());
    obs[2] = std::clamp(-joystick->rx(), cfg["ang_vel_z"][0].as<float>(), cfg["ang_vel_z"][1].as<float>());

    const float scale = squat_speed_scale(env);
    obs[0] *= scale;
    obs[1] *= scale;
    obs[2] *= scale;

    return obs;
}

REGISTER_OBSERVATION(gait_phase)
{
    float period = params["period"].as<float>();
    float delta_phase = env->step_dt * (1.0f / period);

    env->global_phase += delta_phase;
    env->global_phase = std::fmod(env->global_phase, 1.0f);

    std::vector<float> obs(2);
    obs[0] = std::sin(env->global_phase * 2 * M_PI);
    obs[1] = std::cos(env->global_phase * 2 * M_PI);

    // Opzione C: congela il clock a (0,0) quando il comando joystick e' ~0 (soglia 0.1).
    // IDENTICO al training (mdp.gait_phase con command_name="base_velocity"): il clock interno
    // continua ad avanzare, ma l'output e' azzerato da fermo -> niente dondolio 1.25 Hz.
    auto & joystick = env->robot->data.joystick;
    const auto cmd_cfg = env->cfg["commands"]["base_velocity"]["ranges"];
    float vx = std::clamp(joystick->ly(), cmd_cfg["lin_vel_x"][0].as<float>(), cmd_cfg["lin_vel_x"][1].as<float>());
    float vy = std::clamp(-joystick->lx(), cmd_cfg["lin_vel_y"][0].as<float>(), cmd_cfg["lin_vel_y"][1].as<float>());
    float wz = std::clamp(-joystick->rx(), cmd_cfg["ang_vel_z"][0].as<float>(), cmd_cfg["ang_vel_z"][1].as<float>());
    // La soglia va applicata al comando SCALATO, come nel training: mdp.gait_phase legge il comando
    // dal command manager, che e' gia' scalato dalla profondita'. Usare qui quello grezzo terrebbe
    // il clock in moto a fondo squat mentre in addestramento e' congelato -> disallineamento.
    const float gait_scale = squat_speed_scale(env);
    vx *= gait_scale;
    vy *= gait_scale;
    wz *= gait_scale;
    if (std::sqrt(vx * vx + vy * vy + wz * wz) <= squat_gait_threshold(env, 0.1f))
    {
        obs[0] = 0.0f;
        obs[1] = 0.0f;
    }
    return obs;
}

// Squat task: target base height relative to nominal (target - nominal), matching the training
// obs mdp.base_height_command_rel. Used ONLY by the squat controller (h1_2_ctrl_squat); harmless
// for the velocity controller, whose deploy.yaml does not list it.
REGISTER_OBSERVATION(base_height_command)
{
    const float nominal = params["nominal_height"].as<float>();
    const auto range = env->cfg["commands"]["base_height"]["ranges"]["height"];
    const float h_min = range[0].as<float>();  // deepest squat (e.g. 0.55)
    const float h_max = range[1].as<float>();  // nominal / standing (e.g. 0.98)

    // Fermo alla quota minima di camminata: R2 da solo si ferma 1 cm sopra cutoff_height (0.70 ->
    // 0.71), A + R2 scende oltre fino a h_min. Il centimetro di margine c'e' perche' la rampa procede
    // a 1 mm per passo in float: un fermo esattamente sul cutoff potrebbe lasciare il target a 0.6999
    // e il cutoff azzererebbe i comandi di velocita' senza che si veda. Policy senza cutoff_height
    // nel deploy.yaml -> nessun fermo (h_stop = h_min), comportamento di prima.
    const auto vel_cmds = env->cfg["commands"]["base_velocity"];
    const bool has_cutoff = static_cast<bool>(vel_cmds["cutoff_height"]);
    const float cutoff = has_cutoff ? vel_cmds["cutoff_height"].as<float>() : h_min;
    const float h_stop = has_cutoff ? std::clamp(cutoff + 0.01f, h_min, h_max) : h_min;

    auto & joystick = env->robot->data.joystick;
    const float descend_rate = 0.05f;  // [m/s] descent speed while RT is held
    const float rise_rate = 0.05f;     // [m/s] stand-up speed while B is held (slower -> no overshoot)

    // RT (R2) is digital on this pad -> "hold to squat": the longer RT is held, the deeper.
    // B -> stand up, RAMPED (not a snap) so the base does not overshoot the nominal height and
    // ring afterwards. Held state lives in env->squat_target.
    const float prev = env->squat_target;
    if (joystick->B.pressed)
        env->squat_target += rise_rate * env->step_dt;
    else if (joystick->RT.pressed)
    {
        // Sotto il fermo R2 da solo non fa nulla (il limite non deve mai riportare il target in su):
        // per scendere ancora serve A.
        const float lower = joystick->A.pressed ? h_min : std::min(h_stop, prev);
        env->squat_target = std::max(prev - descend_rate * env->step_dt, lower);
    }
    env->squat_target = std::clamp(env->squat_target, h_min, h_max);

    // Quota nel terminale, perche' durante il test non si vede altrimenti: gli eventi subito, durante
    // la rampa al massimo ogni 0.25 s, e sempre il valore finale quando la rampa si ferma.
    const float h = env->squat_target;
    if (env->squat_print_cooldown > 0)
        env->squat_print_cooldown--;
    const bool stop_reached = has_cutoff && prev > h_stop && h == h_stop;
    const bool went_below = has_cutoff && prev >= cutoff && h < cutoff;
    const bool came_back = has_cutoff && prev < cutoff && h >= cutoff;
    const bool ramp_tick = h != prev && env->squat_print_cooldown == 0;
    const bool ramp_end = h == prev && h != env->squat_shown;
    if (stop_reached)
        spdlog::info("Squat: height {:.3f} m, stopped at the walking floor (hold A + R2 to go down to {:.2f} m)", h, h_min);
    else if (went_below)
        spdlog::warn("Squat: height {:.3f} m, below {:.2f} m: velocity commands DISABLED", h, cutoff);
    else if (came_back)
        spdlog::info("Squat: height {:.3f} m, back above {:.2f} m: velocity commands enabled", h, cutoff);
    else if (ramp_tick || ramp_end)
        spdlog::info("Squat: height {:.3f} m", h);
    if (stop_reached || went_below || came_back || ramp_tick || ramp_end)
    {
        env->squat_shown = h;
        env->squat_print_cooldown = static_cast<int>(std::lround(0.25f / env->step_dt));
    }

    std::vector<float> obs(1);
    obs[0] = env->squat_target - nominal;  // matches training obs base_height_command_rel
    return obs;
}

}
}