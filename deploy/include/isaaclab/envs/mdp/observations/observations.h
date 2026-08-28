// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "isaaclab/envs/manager_based_rl_env.h"

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

REGISTER_OBSERVATION(velocity_commands)
{
    std::vector<float> obs(3);
    auto & joystick = env->robot->data.joystick;

    const auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    obs[0] = std::clamp(joystick->ly(), cfg["lin_vel_x"][0].as<float>(), cfg["lin_vel_x"][1].as<float>());
    obs[1] = std::clamp(-joystick->lx(), cfg["lin_vel_y"][0].as<float>(), cfg["lin_vel_y"][1].as<float>());
    obs[2] = std::clamp(-joystick->rx(), cfg["ang_vel_z"][0].as<float>(), cfg["ang_vel_z"][1].as<float>());

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
    if (std::sqrt(vx * vx + vy * vy + wz * wz) <= 0.1f)
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
    const float h_min = range[0].as<float>();
    const float h_max = range[1].as<float>();

    auto & joystick = env->robot->data.joystick;
    // Continuous height axis on the right-stick Y: +1 (up) -> stand, -1 (down) -> deepest squat.
    // NOTE: verify axis + sign on the real pad; flip if needed.
    float u = std::clamp((joystick->ry() + 1.0f) * 0.5f, 0.0f, 1.0f);  // [0,1]
    float target = h_min + u * (h_max - h_min);

    // TODO(deploy): "preset" button -> snap target to nominal (stand up) while held.
    // e.g. if (joystick->A()) target = nominal;  // wire to the actual button API

    std::vector<float> obs(1);
    obs[0] = target - nominal;
    return obs;
}

}
}