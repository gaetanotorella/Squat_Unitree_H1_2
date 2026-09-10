// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "isaaclab/envs/manager_based_rl_env.h"

#include <algorithm>
#include <cmath>

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
inline float squat_speed_scale(isaaclab::ManagerBasedRLEnv * env)
{
    const auto cmds = env->cfg["commands"]["base_velocity"];
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

    auto & joystick = env->robot->data.joystick;
    const float descend_rate = 0.15f;  // [m/s] descent speed while RT is held
    const float rise_rate = 0.15f;     // [m/s] stand-up speed while B is held (slower -> no overshoot)

    // RT (R2) is digital on this pad -> "hold to squat": the longer RT is held, the deeper.
    // B -> stand up, RAMPED (not a snap) so the base does not overshoot the nominal height and
    // ring afterwards. Held state lives in env->squat_target.
    if (joystick->B.pressed)
        env->squat_target += rise_rate * env->step_dt;
    else if (joystick->RT.pressed)
        env->squat_target -= descend_rate * env->step_dt;
    env->squat_target = std::clamp(env->squat_target, h_min, h_max);

    std::vector<float> obs(1);
    obs[0] = env->squat_target - nominal;  // matches training obs base_height_command_rel
    return obs;
}

}
}