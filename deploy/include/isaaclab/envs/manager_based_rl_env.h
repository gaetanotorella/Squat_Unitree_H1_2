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
#include <ctime>
#include "param.h"
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
    // Destinazione (sotto unitree_ws/sim_logs/), decisa da --network:
    //   - ROBOT REALE (interfaccia != "lo") -> crea exp_<ts>/  (nessun ground_truth: non esiste)
    //   - sim2sim ("lo")                    -> usa la sim_<ts>/ del simulatore, se fresca (<120 s)
    //   - sim2sim senza simulatore attivo   -> sim_controller_only_<ts>/
    // Allineamento col ground_truth (solo in sim2sim) via wall_t (orologio di sistema).
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
            std::filesystem::create_directories(base, ec);

            auto stamp = []() {
                std::time_t t = std::time(nullptr);
                char buf[32];
                std::strftime(buf, sizeof(buf), "%Y-%m-%d_%H-%M-%S", std::localtime(&t));
                return std::string(buf);
            };

            // "lo" (o vuota) = sim2sim; qualunque altra interfaccia = ROBOT REALE.
            const std::string & iface = param::network_iface;
            const bool real_robot = !(iface.empty() || iface == "lo");

            if (real_robot) {
                // Sul robot reale non c'e' nessun simulatore che crei la cartella, e non esiste un
                // ground_truth: il controllore crea la PROPRIA cartella exp_<ts>.
                run_dir = base / ("exp_" + stamp());
                std::filesystem::create_directories(run_dir, ec);
            } else {
                // sim2sim: aggancia la cartella del simulatore, identificata da ground_truth.csv
                // (che SOLO il simulatore scrive) e "fresca" -- cioe' con ground_truth.csv scritto
                // da meno di 120 s, segno che il simulatore sta girando ADESSO.
                // Senza questo controllo, lanciare il controllore senza far ripartire il simulatore
                // riscriveva i CSV di un run VECCHIO (e' cosi' che sim_2026-08-31 ha perso una colonna).
                // NB: si guarda la mtime del FILE, non della directory: appendere a un file non
                // aggiorna la mtime della cartella che lo contiene.
                std::string best;
                std::filesystem::file_time_type best_t{};
                for (auto & e : std::filesystem::directory_iterator(base, ec)) {
                    if (!e.is_directory()) continue;
                    std::filesystem::path g = e.path() / "ground_truth.csv";
                    if (!std::filesystem::exists(g, ec)) continue;
                    std::string name = e.path().filename().string();
                    if (name > best) { best = name; best_t = std::filesystem::last_write_time(g, ec); }
                }
                bool fresh = false;
                if (!best.empty()) {
                    auto age = std::filesystem::file_time_type::clock::now() - best_t;
                    fresh = std::chrono::duration_cast<std::chrono::seconds>(age).count() < 120;
                }
                if (fresh) {
                    run_dir = base / best;
                } else {
                    // nessun simulatore attivo: cartella dedicata e TIMESTAMPATA (non sovrascrive
                    // nulla, e senza ground_truth.csv non verra' mai riagganciata da un run futuro)
                    run_dir = base / ("ctrl_only_" + stamp());
                    std::filesystem::create_directories(run_dir, ec);
                }
            }

            // --- run_info.md: da quale POLICY viene questo log ---
            // Senza questo, dato un sim_log/exp_log non c'e' modo di risalire alla policy che lo
            // ha prodotto (le cartelle in logs/ sono piu' d'una e il controllore, se policy_dir
            // non e' esplicita, sceglie da solo la piu' recente con exported/).
            {
                std::ofstream info((run_dir / "run_info.md").string());
                auto yaml_range = [&](const char * cmd, const char * key) {
                    try {
                        auto n = cfg["commands"][cmd]["ranges"][key];
                        char b[64];
                        std::snprintf(b, sizeof(b), "[%.3g, %.3g]", n[0].as<float>(), n[1].as<float>());
                        return std::string(b);
                    } catch (...) { return std::string("n/d"); }
                };
                std::time_t tt = std::time(nullptr);
                char tbuf[64];
                std::strftime(tbuf, sizeof(tbuf), "%Y-%m-%d %H:%M:%S", std::localtime(&tt));

                info << "# Run info\n\n";
                info << "- **policy_dir**: " << param::policy_dir.string() << "\n";
                info << "- **policy_name**: " << param::policy_dir.filename().string() << "\n";
                info << "- **controller**: " << param::bin_path.filename().string() << "\n";
                info << "- **network**: " << (param::network_iface.empty() ? "(default)" : param::network_iface)
                     << (real_robot ? "  -> ROBOT REALE" : "  -> sim2sim") << "\n";
                info << "- **started**: " << tbuf << "\n";
                info << "- **n_obs**: " << (obsvec ? obsvec->size() : 0) << "\n";
                info << "- **n_joints**: " << nj << "\n";
                info << "- **base_height_range**: " << yaml_range("base_height", "height") << "\n";
                info << "- **lin_vel_x_range**: " << yaml_range("base_velocity", "lin_vel_x") << "\n";
                info << "- **lin_vel_y_range**: " << yaml_range("base_velocity", "lin_vel_y") << "\n";
                info << "- **ang_vel_z_range**: " << yaml_range("base_velocity", "ang_vel_z") << "\n";
                std::error_code fec;
                auto onnx = param::policy_dir / "exported" / "policy.onnx";
                if (std::filesystem::exists(onnx, fec))
                    info << "- **policy_onnx_bytes**: " << std::filesystem::file_size(onnx, fec) << "\n";
                info.close();
                std::cout << "[deploy_logger] policy = " << param::policy_dir.string() << std::endl;
            }

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
    float squat_target = 0.98f;  // squat task: current commanded base height [m] (held between steps)
};

};