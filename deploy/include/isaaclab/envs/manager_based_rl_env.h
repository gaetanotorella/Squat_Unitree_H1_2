// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <eigen3/Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include "isaaclab/manager/observation_manager.h"
#include "isaaclab/manager/action_manager.h"
#include "isaaclab/assets/articulation/articulation.h"
#include "isaaclab/algorithms/algorithms.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include "isaaclab/utils/utils.h"

namespace isaaclab
{

class ObservationManager;
class ActionManager;

class ManagerBasedRLEnv
{
public:
    // Constructor
    ManagerBasedRLEnv(YAML::Node cfg, std::shared_ptr<Articulation> robot_)
    :cfg(cfg), robot(std::move(robot_))
    {
        // Parse configuration
        this->step_dt = cfg["step_dt"].as<float>();
        robot->data.joint_ids_map = cfg["joint_ids_map"].as<std::vector<float>>();
        robot->data.joint_pos.resize(robot->data.joint_ids_map.size());
        robot->data.joint_vel.resize(robot->data.joint_ids_map.size());

        { // default joint positions
            auto default_joint_pos = cfg["default_joint_pos"].as<std::vector<float>>();
            robot->data.default_joint_pos = Eigen::VectorXf::Map(default_joint_pos.data(), default_joint_pos.size());
        }
        { // joint stiffness and damping
            robot->data.joint_stiffness = cfg["stiffness"].as<std::vector<float>>();
            robot->data.joint_damping = cfg["damping"].as<std::vector<float>>();
        }

        robot->update();

        // load managers
        action_manager = std::make_unique<ActionManager>(cfg["actions"], this);
        observation_manager = std::make_unique<ObservationManager>(cfg["observations"], this);
    }

    void reset()
    {
        global_phase = 0;
        episode_length = 0;
        robot->update();
        action_manager->reset();
        observation_manager->reset();
    }

    void step()
    {
        episode_length += 1;
        robot->update();
        auto obs = observation_manager->compute();
        auto action = alg->act(obs);
        action_manager->process_action(action);

        log_step(obs);
    }

    // === LOG STRUTTURATO: obs.csv / commands.csv / errors.csv ===
    // Scrive nella cartella sim_logs/sim_<ts>/ creata dal simulatore (piu' recente);
    // fallback a sim_controller_only/ se il simulatore non ne ha create.
    // Allineamento col ground_truth via wall_t (orologio di sistema).
    void log_step(std::unordered_map<std::string, std::vector<float>> & obs)
    {
        static std::ofstream obs_out, cmd_out, err_out;
        static bool init = false;

        auto q_target = action_manager->processed_actions();
        auto & kp = robot->data.joint_stiffness;
        auto & kd = robot->data.joint_damping;

        if (!init) {
            init = true;
            const char * home = std::getenv("HOME");
            std::filesystem::path base = std::string(home ? home : ".") + "/sim_logs";
            std::filesystem::path run_dir;
            std::error_code ec;
            std::string best;
            if (std::filesystem::exists(base, ec)) {
                for (auto & e : std::filesystem::directory_iterator(base, ec)) {
                    if (!e.is_directory()) continue;
                    std::string name = e.path().filename().string();
                    if (name.rfind("sim_", 0) == 0 && name > best) best = name;
                }
            }
            if (!best.empty()) run_dir = base / best;
            else { run_dir = base / "sim_controller_only"; std::filesystem::create_directories(run_dir, ec); }

            obs_out.open((run_dir / "obs.csv").string());
            cmd_out.open((run_dir / "commands.csv").string());
            err_out.open((run_dir / "errors.csv").string());

            obs_out << "wall_t,ep_t";
            for (auto & kv : obs)
                for (size_t i = 0; i < kv.second.size(); i++) obs_out << "," << kv.first << "_" << i;
            obs_out << "\n";

            cmd_out << "wall_t,ep_t";
            for (size_t i = 0; i < q_target.size(); i++) cmd_out << ",q_target_" << i;
            for (size_t i = 0; i < kp.size(); i++) cmd_out << ",kp_" << i;
            for (size_t i = 0; i < kd.size(); i++) cmd_out << ",kd_" << i;
            cmd_out << "\n";

            err_out << "wall_t,ep_t";
            for (size_t i = 0; i < q_target.size(); i++) err_out << ",q_err_" << i;
            err_out << "\n";

            std::cout << "[deploy_logger] obs/commands/errors -> " << run_dir << std::endl;
        }

        double wall_t = std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        double ep_t = episode_length * step_dt;

        obs_out << std::setprecision(7) << wall_t << "," << ep_t;
        for (auto & kv : obs)
            for (size_t i = 0; i < kv.second.size(); i++) obs_out << "," << kv.second[i];
        obs_out << "\n"; obs_out.flush();

        cmd_out << std::setprecision(7) << wall_t << "," << ep_t;
        for (size_t i = 0; i < q_target.size(); i++) cmd_out << "," << q_target[i];
        for (size_t i = 0; i < kp.size(); i++) cmd_out << "," << kp[i];
        for (size_t i = 0; i < kd.size(); i++) cmd_out << "," << kd[i];
        cmd_out << "\n"; cmd_out.flush();

        err_out << std::setprecision(7) << wall_t << "," << ep_t;
        for (size_t i = 0; i < q_target.size(); i++) {
            float q_meas = (i < (size_t)robot->data.joint_pos.size()) ? robot->data.joint_pos[i] : 0.0f;
            err_out << "," << (q_target[i] - q_meas);
        }
        err_out << "\n"; err_out.flush();
    }

    float step_dt;
    
    YAML::Node cfg;

    std::unique_ptr<ObservationManager> observation_manager;
    std::unique_ptr<ActionManager> action_manager;
    std::shared_ptr<Articulation> robot;
    std::unique_ptr<Algorithms> alg;
    long episode_length = 0;
    float global_phase = 0.0f;
};

};