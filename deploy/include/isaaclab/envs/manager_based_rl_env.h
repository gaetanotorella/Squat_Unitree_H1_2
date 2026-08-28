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
#include <cstdio>
#include <cmath>
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
    // Scrive nella cartella unitree_ws/sim_logs/sim_<ts>/ creata dal simulatore (piu' recente);
    // fallback a sim_controller_only/ se il simulatore non ne ha create.
    // Allineamento col ground_truth via wall_t (orologio di sistema).
    void log_step(std::unordered_map<std::string, std::vector<float>> & obs)
    {
        static std::ofstream obs_out, cmd_out, err_out;
        static bool init = false;
        static std::vector<int> jmap;          // policy i -> indice SDK (per rimappare kp/kd)

        auto q_target = action_manager->processed_actions();
        auto & kp = robot->data.joint_stiffness; // in ordine SDK
        auto & kd = robot->data.joint_damping;   // in ordine SDK
        const size_t nj = q_target.size();

        // gruppo obs (chiave "obs", oppure l'unico gruppo presente)
        const std::vector<float> * obsvec = nullptr;
        { auto it = obs.find("obs"); obsvec = (it != obs.end()) ? &it->second
                                             : (obs.empty() ? nullptr : &obs.begin()->second); }

        if (!init) {
            init = true;

            // nomi giunti in ordine-policy (senza suffisso _joint); fallback a indici
            std::vector<std::string> jn;
            if (cfg["joint_names"]) {
                for (auto n : cfg["joint_names"]) {
                    std::string s = n.as<std::string>();
                    if (s.size() > 6 && s.compare(s.size() - 6, 6, "_joint") == 0) s.resize(s.size() - 6);
                    jn.push_back(s);
                }
            }
            if (jn.size() != nj) { jn.clear(); for (size_t i = 0; i < nj; i++) jn.push_back(std::to_string(i)); }

            // joint_ids_map (policy -> SDK) per rimappare kp/kd in ordine-policy
            jmap.clear();
            for (auto v : robot->data.joint_ids_map) jmap.push_back((int)std::lround(v));
            if (jmap.size() != nj) { jmap.clear(); for (size_t i = 0; i < nj; i++) jmap.push_back((int)i); }

            // nomi colonne obs: per-termine da cfg["observations"] (stesso ordine della concat)
            auto sfx = [](const std::string & t) -> std::string {
                if (t.find("pos") != std::string::npos) return "pos";
                if (t.find("vel") != std::string::npos) return "vel";
                if (t.find("action") != std::string::npos) return "act";
                return t;
            };
            std::vector<std::string> obs_cols;
            if (cfg["observations"]) {
                for (auto it = cfg["observations"].begin(); it != cfg["observations"].end(); ++it) {
                    std::string term = it->first.as<std::string>();
                    if (term == "scale_first" || term == "use_gym_history") continue;
                    size_t dim = it->second["scale"] ? it->second["scale"].size() : 0;
                    if (dim == nj) { std::string s = sfx(term); for (auto & j : jn) obs_cols.push_back(j + "_" + s); }
                    else for (size_t k = 0; k < dim; k++) obs_cols.push_back(term + "_" + std::to_string(k));
                }
            }
            // guardia: se il conteggio non torna, ricadi su nomi generici
            if (obsvec && obs_cols.size() != obsvec->size()) {
                obs_cols.clear();
                for (size_t i = 0; obsvec && i < obsvec->size(); i++) obs_cols.push_back("obs_" + std::to_string(i));
            }

            const char * home = std::getenv("HOME");
            std::filesystem::path base = std::string(home ? home : ".") + "/unitree_ws/sim_logs";
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
            for (auto & c : obs_cols) obs_out << "," << c;
            obs_out << "\n";

            cmd_out << "wall_t,ep_t";
            for (auto & j : jn) cmd_out << "," << j << "_q_target";
            for (auto & j : jn) cmd_out << "," << j << "_kp";
            for (auto & j : jn) cmd_out << "," << j << "_kd";
            cmd_out << "\n";

            err_out << "wall_t,ep_t";
            for (auto & j : jn) err_out << "," << j << "_q_err";
            err_out << "\n";

            std::cout << "[deploy_logger] obs/commands/errors -> " << run_dir << std::endl;
        }

        double wall_t = std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        double ep_t = episode_length * step_dt;
        char wbuf[32];
        std::snprintf(wbuf, sizeof(wbuf), "%.6f", wall_t); // parte intera + microsec

        obs_out << wbuf << "," << std::setprecision(7) << ep_t;
        if (obsvec) for (float v : *obsvec) obs_out << "," << v;
        obs_out << "\n"; obs_out.flush();

        // kp/kd rimappati in ordine-policy: kp_policy[i] = kp_sdk[jmap[i]]
        cmd_out << wbuf << "," << std::setprecision(7) << ep_t;
        for (size_t i = 0; i < nj; i++) cmd_out << "," << q_target[i];
        for (size_t i = 0; i < nj; i++) cmd_out << "," << kp[jmap[i]];
        for (size_t i = 0; i < nj; i++) cmd_out << "," << kd[jmap[i]];
        cmd_out << "\n"; cmd_out.flush();

        err_out << wbuf << "," << std::setprecision(7) << ep_t;
        for (size_t i = 0; i < nj; i++) {
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