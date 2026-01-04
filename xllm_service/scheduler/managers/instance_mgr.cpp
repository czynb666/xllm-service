/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm-service/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "instance_mgr.h"

#include <absl/strings/str_join.h>
#include <brpc/controller.h>
#include <glog/logging.h>

#include <chrono>
#include <iostream>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <limits>
#include <shared_mutex>

#include "common/global_gflags.h"
#include "common/types.h"
#include "common/utils.h"

namespace xllm_service {

static std::unordered_map<InstanceType, std::string> ETCD_KEYS_PREFIX_MAP = {
    {InstanceType::DEFAULT, "XLLM:DEFAULT:"},
    {InstanceType::PREFILL, "XLLM:PREFILL:"},
    {InstanceType::DECODE, "XLLM:DECODE:"},
    {InstanceType::MIX, "XLLM:MIX:"},
};
static std::string ETCD_ALL_KEYS_PREFIX = "XLLM:";
static std::string ETCD_LOADMETRICS_PREFIX = "XLLM:LOADMETRICS:";

InstanceMgr::InstanceMgr(const Options& options,
                         const std::shared_ptr<EtcdClient>& etcd_client,
                         const bool is_master_service)
    : options_(options),
      is_master_service_(is_master_service),
      etcd_client_(etcd_client) {
  auto handle_instance_metainfo =
      std::bind(&InstanceMgr::update_instance_metainfo,
                this,
                std::placeholders::_1,
                std::placeholders::_2);
  for (auto& it : ETCD_KEYS_PREFIX_MAP) {
    etcd_client_->add_watch(it.second, handle_instance_metainfo);
  }
  if (!is_master_service_) {
    auto handle_load_metrics = std::bind(&InstanceMgr::update_load_metrics,
                                         this,
                                         std::placeholders::_1,
                                         std::placeholders::_2);
    etcd_client_->add_watch(ETCD_LOADMETRICS_PREFIX, handle_load_metrics);
  }

  init();
}

void InstanceMgr::init() {
  init_model_memory_specs();

  {
    std::unique_lock<std::shared_mutex> lock(inst_mutex_);
    for (auto& it : ETCD_KEYS_PREFIX_MAP) {
      etcd_client_->get_prefix(it.second, &instances_);
    }
    // create ttft predictor and request metrics for each instance
    {
      std::lock_guard<std::mutex> time_predictor_lock(time_predictor_mutex_);
      std::lock_guard<std::mutex> request_metrics_lock(request_metrics_mutex_);
      for (auto& pair : instances_) {
        time_predictors_.insert_or_assign(
            pair.first,
            TimePredictor(pair.second.ttft_profiling_data,
                          pair.second.tpot_profiling_data));
        request_metrics_.insert_or_assign(pair.first, RequestMetrics());
      }
    }
    LOG(INFO) << "Load instance info from etcd:" << instances_.size();
    std::vector<std::string> channel_creat_fail_insts;
    prefill_index_.reserve(instances_.size());
    decode_index_.reserve(instances_.size());

    for (auto& ist : instances_) {
      if (!create_channel(ist.first)) {
        channel_creat_fail_insts.emplace_back(ist.first);
      } else {
        switch (ist.second.type) {
          case InstanceType::DEFAULT:
          case InstanceType::PREFILL:
            ist.second.instance_index = prefill_index_.size();
            prefill_index_.emplace_back(ist.first);
            LOG(INFO) << "Register a new prefill instance, instance name : "
                      << ist.first;
            break;
          case InstanceType::DECODE:
            ist.second.instance_index = decode_index_.size();
            decode_index_.emplace_back(ist.first);
            LOG(INFO) << "Register a new decode instance, instance name : "
                      << ist.first;
            break;
          case InstanceType::MIX:
            // In the initial state, we set the first MIX type instance as a
            // decode instance, while all subsequent instances are set as
            // prefill instances.
            if (decode_index_.size() > 0) {
              ist.second.instance_index = prefill_index_.size();
              ist.second.current_type = InstanceType::PREFILL;
              prefill_index_.emplace_back(ist.first);
              LOG(INFO) << "Register a new prefill instance, instance name : "
                        << ist.first;
            } else {
              ist.second.instance_index = decode_index_.size();
              ist.second.current_type = InstanceType::DECODE;
              decode_index_.emplace_back(ist.first);
              LOG(INFO) << "Register a new decode instance, instance name : "
                        << ist.first;
            }
            break;
          default:
            LOG(WARNING) << "Unknown InstanceType: " << int(ist.second.type);
            channel_creat_fail_insts.emplace_back(ist.first);
            break;
        }
      }
    }
    for (auto& name : channel_creat_fail_insts) {
      instances_.erase(name);
      {
        std::lock_guard<std::mutex> time_predictor_lock(time_predictor_mutex_);
        std::lock_guard<std::mutex> request_metrics_lock(
            request_metrics_mutex_);
        time_predictors_.erase(name);
        request_metrics_.erase(name);
      }
    }
  }
  {
    std::unique_lock<std::shared_mutex> lock(load_metric_mutex_);
    etcd_client_->get_prefix(ETCD_LOADMETRICS_PREFIX, &load_metrics_);
  }

  for (int i = 0; i < prefill_index_.size(); i++) {
    LOG(INFO) << i << " : " << prefill_index_[i];
  }
}

InstanceMgr::~InstanceMgr() { exited_ = true; }

InstanceMetaInfo InstanceMgr::get_instance_info(
    const std::string& instance_name) {
  std::shared_lock<std::shared_mutex> lock(inst_mutex_);
  if (instances_.find(instance_name) == instances_.end()) {
    LOG(ERROR) << "Get instance info failed, instance is not registered, "
                  "instance_name: "
               << instance_name;
    return InstanceMetaInfo();
  }
  return instances_[instance_name];
}

bool InstanceMgr::get_next_instance_pair(const std::string& model_id, Routing* routing) {
  std::unique_lock<std::shared_mutex> lock(inst_mutex_);
  if (prefill_index_.empty()) {
    LOG(ERROR) << "No prefill or default instance found for model " << model_id;
    return false;
  }

  {
    std::shared_lock<std::shared_mutex> lock(instance_model_state_mutex_);
    int32_t now_prefill_index = next_prefill_index_[model_id];
    while (instance_model_states_[prefill_index_[now_prefill_index]][model_id] != ModelState::WAKEUP) {
      now_prefill_index = (now_prefill_index + 1) % prefill_index_.size();
    }
    routing->prefill_name = prefill_index_[now_prefill_index];
    next_prefill_index_[model_id] = (now_prefill_index + 1) % prefill_index_.size();
  }

  if (decode_index_.empty()) {
    return true;
  }

  {
    std::shared_lock<std::shared_mutex> lock(instance_model_state_mutex_);
    int32_t now_decode_index = next_decode_index_[model_id];
    while (instance_model_states_[decode_index_[now_decode_index]][model_id] != ModelState::WAKEUP) {
      now_decode_index = (now_decode_index + 1) % decode_index_.size();
    }
    routing->decode_name = decode_index_[now_decode_index];
    next_decode_index_[model_id] = (now_decode_index + 1) % decode_index_.size();
  }

  return true;
}

// TODO: refactor later, currently return all decode instances
std::vector<std::string> InstanceMgr::get_static_decode_list(
    const std::string& instance_name) {
  std::vector<std::string> decode_list;
  std::shared_lock<std::shared_mutex> lock(inst_mutex_);
  for (auto& inst : instances_) {
    if (inst.second.type == InstanceType::DECODE) {
      decode_list.emplace_back(inst.second.name);
    }
  }

  return decode_list;
}

// TODO: refactor later, currently return all prefill instances
std::vector<std::string> InstanceMgr::get_static_prefill_list(
    const std::string& instance_name) {
  std::vector<std::string> prefill_list;
  std::shared_lock<std::shared_mutex> lock(inst_mutex_);
  for (auto& inst : instances_) {
    if (inst.second.type == InstanceType::PREFILL ||
        inst.second.type == InstanceType::DEFAULT) {
      prefill_list.emplace_back(inst.second.name);
    }
  }

  return prefill_list;
}

void InstanceMgr::fork_master_and_sleep(
    const std::string& instance_name,
    std::shared_ptr<brpc::Channel> channel) {
  for (const auto& model : MODELS) {
    // 1. Fork Master
    nlohmann::json fork_body;
    fork_body["model_path"] = model.second;
    fork_body["master_node_addr"] = "127.0.0.1:" + std::to_string(++master_node_port);
    fork_body["master_status"] = 0;

    if (!send_http_request(channel, "/fork_master", fork_body.dump())) {
      LOG(ERROR) << "Failed to fork master for model " << model.first << " on "
                 << instance_name;
      continue;
    }

    // 2. Sleep
    nlohmann::json sleep_body;
    sleep_body["model_id"] = model.first;
    sleep_body["master_status"] = 1;

    if (send_http_request(channel, "/sleep", sleep_body.dump())) {
      std::unique_lock<std::shared_mutex> lock(instance_model_state_mutex_);
      instance_model_states_[instance_name][model.first] = ModelState::SLEEP;  // Sleep
      LOG(INFO) << "Model " << model.first << " on " << instance_name
                << " is now SLEEPING";
    } else {
      LOG(ERROR) << "Failed to sleep model " << model.first << " on "
                 << instance_name;
    }
  }
}

bool InstanceMgr::send_http_request(const std::string& instance_name,
                                    const std::string& uri,
                                    const std::string& request_body) {
  std::shared_ptr<brpc::Channel> channel = get_channel(instance_name);
  if (!channel) {
    LOG(ERROR) << "Channel not found for " << instance_name;
    return false;
  }
  return send_http_request(channel, uri, request_body);
}

bool InstanceMgr::send_http_request(std::shared_ptr<brpc::Channel> channel,
                                    const std::string& uri,
                                    const std::string& request_body) {
  brpc::Controller cntl;
  cntl.http_request().uri() = uri;  // brpc channel already has host:port
  cntl.http_request().set_method(brpc::HTTP_METHOD_POST);
  cntl.http_request().set_content_type("application/json");
  cntl.request_attachment().append(request_body);

  channel->CallMethod(nullptr, &cntl, nullptr, nullptr, nullptr);

  if (cntl.Failed()) {
    LOG(ERROR) << "HTTP request failed: " << cntl.ErrorText();
    return false;
  }
  return true;
}

void InstanceMgr::get_load_metrics(LoadBalanceInfos* infos) {
  std::shared_lock<std::shared_mutex> inst_lock(inst_mutex_);
  std::shared_lock<std::shared_mutex> metric_lock(load_metric_mutex_);

  for (auto name : infos->overlap_scores.instances) {
    auto it = load_metrics_.find(name);
    if (it == load_metrics_.end()) {
      continue;
    }
    auto instance_it = instances_.find(name);
    if (instance_it == instances_.end()) {
      continue;
    }

    if (instance_it->second.type == InstanceType::DECODE) {
      infos->decode_load_metrics.insert(std::make_pair(name, it->second));
      infos->decode_max_waiting_requests_num =
          std::max(infos->decode_max_waiting_requests_num,
                   it->second.waiting_requests_num);
    } else {
      infos->prefill_load_metrics.insert(std::make_pair(name, it->second));
      infos->prefill_max_waiting_requests_num =
          std::max(infos->prefill_max_waiting_requests_num,
                   it->second.waiting_requests_num);
    }
  }

  std::string least_loaded_prefill_instance;
  float least_loaded_prefill_gpu_cache_usage_perc = 1;
  std::string least_loaded_decode_instance;
  float least_loaded_decode_gpu_cache_usage_perc = 1;

  if (infos->prefill_load_metrics.size() == 0 ||
      infos->decode_load_metrics.size() == 0) {
    for (const auto& metric : load_metrics_) {
      auto instance_it = instances_.find(metric.first);
      if (instance_it != instances_.end()) {
        if (instance_it->second.type != InstanceType::DECODE) {
          if (metric.second.gpu_cache_usage_perc <
              least_loaded_prefill_gpu_cache_usage_perc) {
            least_loaded_prefill_gpu_cache_usage_perc =
                metric.second.gpu_cache_usage_perc;
            least_loaded_prefill_instance = metric.first;
          }
        } else {
          if (metric.second.gpu_cache_usage_perc <
              least_loaded_decode_gpu_cache_usage_perc) {
            least_loaded_decode_gpu_cache_usage_perc =
                metric.second.gpu_cache_usage_perc;
            least_loaded_decode_instance = metric.first;
          }
        }
      }
    }
  }

  if (infos->prefill_load_metrics.size() == 0 &&
      !least_loaded_prefill_instance.empty()) {
    infos->prefill_load_metrics.insert(
        std::make_pair(least_loaded_prefill_instance,
                       load_metrics_[least_loaded_prefill_instance]));
  }

  if (infos->decode_load_metrics.size() == 0 &&
      !least_loaded_decode_instance.empty()) {
    infos->decode_load_metrics.insert(
        std::make_pair(least_loaded_decode_instance,
                       load_metrics_[least_loaded_decode_instance]));
  }
}

void InstanceMgr::record_load_metrics_update(
    const std::string& instance_name,
    const proto::LoadMetrics& load_metrics) {
  std::lock_guard<std::mutex> lock(update_mutex_);

  updated_metrics_.insert_or_assign(
      instance_name,
      LoadMetrics(load_metrics.waiting_requests_num(),
                  load_metrics.gpu_cache_usage_perc()));
}

bool InstanceMgr::upload_load_metrics() {
  std::lock_guard<std::mutex> lock(update_mutex_);
  bool status = etcd_client_->set(ETCD_LOADMETRICS_PREFIX, updated_metrics_);
  status =
      status && etcd_client_->rm(ETCD_LOADMETRICS_PREFIX, removed_instance_);
  {
    std::unique_lock<std::shared_mutex> lock(inst_mutex_);
    for (auto& iter : updated_metrics_) {
      load_metrics_.insert_or_assign(iter.first, std::move(iter.second));
    }
    for (auto& iter : removed_instance_) {
      load_metrics_.erase(iter);
    }
  }
  updated_metrics_.clear();
  removed_instance_.clear();

  return status;
}

void InstanceMgr::set_as_master() {
  is_master_service_ = true;
  etcd_client_->remove_watch(ETCD_LOADMETRICS_PREFIX);
}

void InstanceMgr::on_heartbeat(const std::string& instance_name) {
  InstanceMetaInfo metainfo;
  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    auto it = pending_infos_.find(instance_name);
    if (it == pending_infos_.end()) {
      return;
    }
    metainfo = std::move(it->second);
    pending_infos_.erase(it);
  }

  LOG(INFO) << "Received heartbeat from pending instance: " << instance_name;
  threadpool_.schedule([this, instance_name, metainfo = std::move(metainfo)]() {
    register_instance(instance_name, metainfo);
  });
}

void InstanceMgr::register_instance(const std::string& instance_name,
                                    InstanceMetaInfo metainfo) {
  std::unique_lock<std::shared_mutex> lock(inst_mutex_);
  if (instances_.find(instance_name) != instances_.end()) {
    LOG(ERROR) << "Instance is already registered, instance_name: "
               << instance_name;
    return;
  }
  
  if (!create_channel(instance_name)) {
    LOG(ERROR) << "create channel fail: " << instance_name;
    return;
  }

  {
    std::unique_lock<std::shared_mutex> lock(instance_model_state_mutex_);
    instance_model_states_.emplace(instance_name,
                                   std::unordered_map<std::string, ModelState>());
  }
  
  // Note: we can't call fork_master_and_sleep here if we are holding
  // inst_mutex_ and fork_master_and_sleep calls send_http_request which calls
  // get_channel which acquires inst_mutex_ again (deadlock).
  auto channel = cached_channels_[instance_name];
  threadpool_.schedule([this, instance_name, channel]() {
    fork_master_and_sleep(instance_name, channel);
  });
  {
    std::lock_guard<std::mutex> mem_lock(instance_memory_mutex_);
    instance_memory_usage_[instance_name] = 0.0;
  }

  {
    std::lock_guard<std::mutex> time_predictor_lock(time_predictor_mutex_);
    std::lock_guard<std::mutex> request_metrics_lock(request_metrics_mutex_);
    // create ttft predictor for instance
    time_predictors_.emplace(instance_name,
                             TimePredictor(metainfo.ttft_profiling_data,
                                           metainfo.tpot_profiling_data));

    // create request metrics for instance
    request_metrics_.emplace(instance_name, RequestMetrics());
    for (auto& model : MODELS) {
      request_metrics_[instance_name].model_metrics.try_emplace(model.first);
    }
  }

  switch (metainfo.type) {
    case InstanceType::DEFAULT:
    case InstanceType::PREFILL:
      metainfo.instance_index = prefill_index_.size();
      prefill_index_.emplace_back(instance_name);
      LOG(INFO) << "Register a new prefill instance, instance name : "
                << instance_name;
      break;
    case InstanceType::DECODE:
      metainfo.instance_index = decode_index_.size();
      decode_index_.emplace_back(instance_name);
      LOG(INFO) << "Register a new decode instance, instance name : "
                << instance_name;
      break;
    case InstanceType::MIX:
      // In the initial state, we set the first MIX type instance as a
      // decode instance, while all subsequent instances are set as
      // prefill instances.
      if (decode_index_.size() > 0) {
        metainfo.instance_index = prefill_index_.size();
        metainfo.current_type = InstanceType::PREFILL;
        prefill_index_.emplace_back(instance_name);
        LOG(INFO) << "Register a new prefill instance, instance name : "
                  << instance_name;
      } else {
        metainfo.instance_index = decode_index_.size();
        metainfo.current_type = InstanceType::DECODE;
        decode_index_.emplace_back(instance_name);
        LOG(INFO) << "Register a new decode instance, instance name : "
                  << instance_name;
      }
      break;
    default:
      LOG(WARNING) << "Unknown InstanceType: " << int(metainfo.type);
      break;
  }

  instances_.insert(std::make_pair(instance_name, std::move(metainfo)));
}

std::shared_ptr<brpc::Channel> InstanceMgr::get_channel(
    const std::string& instance_name) {
  std::shared_lock<std::shared_mutex> lock(inst_mutex_);
  auto iter = cached_channels_.find(instance_name);
  if (iter == cached_channels_.end()) {
    return nullptr;
  }
  return iter->second;
}

bool InstanceMgr::create_channel(const std::string& instance_name) {
  if (cached_channels_.find(instance_name) == cached_channels_.end()) {
    auto channel = std::make_shared<brpc::Channel>();
    brpc::ChannelOptions options;
    // Add to params
    options.protocol = "http";
    options.timeout_ms = options_.timeout_ms(); /*milliseconds*/
    options.max_retry = 3;
    std::string load_balancer = "";
    if (channel->Init(instance_name.c_str(), load_balancer.c_str(), &options) !=
        0) {
      LOG(ERROR) << "Fail to initialize channel for " << instance_name;
      return false;
    }
    cached_channels_[instance_name] = std::move(channel);
  }

  return true;
}

void InstanceMgr::update_instance_metainfo(const etcd::Response& response,
                                           const uint64_t& prefix_len) {
  if (response.events().empty() || exited_) {
    return;
  }

  threadpool_.schedule([this,
                        response = std::move(response),
                        prefix_len = std::move(prefix_len)] {
    if (exited_) return;
    std::unordered_map<std::string, InstanceMetaInfo> put_map;
    std::vector<std::string> delete_list;

    for (const auto& event : response.events()) {
      std::string instance_name = event.kv().key().substr(prefix_len);

      if (event.event_type() == etcd::Event::EventType::PUT) {
        InstanceMetaInfo metainfo;
        auto json_str = event.kv().as_string();
        if (!metainfo.parse_from_json(json_str)) {
          LOG(ERROR) << "pase json:" << json_str << " error!";
          continue;
        }
        put_map.insert(std::make_pair(instance_name, std::move(metainfo)));

      } else if (event.event_type() == etcd::Event::EventType::DELETE_) {
        delete_list.push_back(instance_name);
      }
    }

    {
      std::unique_lock<std::shared_mutex> lock(inst_mutex_);
      for (auto& iter : put_map) {
        if (instances_.find(iter.first) != instances_.end()) {
          // Update existing instance profiling data
          LOG(INFO) << "Update instance profiling data, instance_name: " << iter.first;
          auto& exist_info = instances_[iter.first];
          auto& new_info = iter.second;
          
          // Merge TTFT profiling data
          for (auto& [model_id, data] : new_info.ttft_profiling_data) {
            exist_info.ttft_profiling_data[model_id] = std::move(data);
          }
          
          // Merge TPOT profiling data
          for (auto& [model_id, data] : new_info.tpot_profiling_data) {
            exist_info.tpot_profiling_data[model_id] = std::move(data);
          }

          // Update TimePredictor
          {
            std::lock_guard<std::mutex> time_predictor_lock(time_predictor_mutex_);
            time_predictors_.insert_or_assign(
                iter.first,
                TimePredictor(exist_info.ttft_profiling_data,
                              exist_info.tpot_profiling_data));
          }
          continue;
        }

        {
          std::lock_guard<std::mutex> lock(pending_mutex_);
          if (pending_infos_.count(iter.first)) {
            LOG(INFO) << "Instance is pending, instance_name: " << iter.first;
            continue;
          }
          pending_infos_.insert(
              std::make_pair(iter.first, std::move(iter.second)));
          LOG(INFO) << "Add instance to pending list and wait for heartbeat: "
                    << iter.first;
        }
      }

      for (auto& iter : delete_list) {
        LOG(INFO) << "delete instance: " << iter;
        {
          std::lock_guard<std::mutex> lock(pending_mutex_);
          if (pending_infos_.count(iter)) {
            pending_infos_.erase(iter);
            LOG(INFO) << "Delete pending instance: " << iter;
            continue;
          }
        }
        if (instances_.find(iter) == instances_.end()) {
          LOG(ERROR) << "Instance is already deleted, instance_name: " << iter;
          continue;
        }
        // TODO: notify cache manager to clear expire cache
        uint64_t index = instances_[iter].instance_index;

        switch (instances_[iter].type) {
          case InstanceType::DEFAULT:
          case InstanceType::PREFILL:
            if (index == -1 || index >= prefill_index_.size()) {
              break;
            }
            std::swap(prefill_index_[index], prefill_index_.back());
            instances_[prefill_index_[index]].instance_index = index;
            prefill_index_.pop_back();
            break;
          case InstanceType::DECODE:
            if (index == -1 || index >= decode_index_.size()) {
              break;
            }
            std::swap(decode_index_[index], decode_index_.back());
            instances_[decode_index_[index]].instance_index = index;
            decode_index_.pop_back();
            break;
          case InstanceType::MIX:
            if (index == -1) {
              break;
            }
            if (instances_[iter].current_type == InstanceType::PREFILL) {
              if (index >= prefill_index_.size()) {
                break;
              }
              std::swap(prefill_index_[index], prefill_index_.back());
              instances_[prefill_index_[index]].instance_index = index;
              prefill_index_.pop_back();
            } else {
              if (index >= decode_index_.size()) {
                break;
              }
              std::swap(decode_index_[index], decode_index_.back());
              instances_[decode_index_[index]].instance_index = index;
              decode_index_.pop_back();
            }
            break;
          default:
            LOG(WARNING) << "Unknown InstanceType: "
                         << int(instances_[iter].type);
            break;
        }

        instances_.erase(iter);
        cached_channels_.erase(iter);
        {
          std::lock_guard<std::mutex> time_predictor_lock(
              time_predictor_mutex_);
          std::lock_guard<std::mutex> request_metrics_lock(
              request_metrics_mutex_);
          time_predictors_.erase(iter);
          request_metrics_.erase(iter);
        }
        {
          std::unique_lock<std::shared_mutex> lock(instance_model_state_mutex_);
          instance_model_states_.erase(iter);
        }
        {
          std::lock_guard<std::mutex> lock(update_mutex_);
          updated_metrics_.erase(iter);
          removed_instance_.insert(iter);
        }
      }
    }
  });
}

void InstanceMgr::update_load_metrics(const etcd::Response& response,
                                      const uint64_t& prefix_len) {
  if (response.events().empty() || exited_) {
    return;
  }
  threadpool_.schedule([this,
                        response = std::move(response),
                        prefix_len = std::move(prefix_len)] {
    if (exited_) return;
    std::unordered_map<std::string, LoadMetrics> put_map;
    std::vector<std::string> delete_list;

    for (const auto& event : response.events()) {
      std::string instance_name = event.kv().key().substr(prefix_len);

      if (event.event_type() == etcd::Event::EventType::PUT) {
        LoadMetrics load_metrics;
        auto json_str = event.kv().as_string();
        if (!load_metrics.parse_from_json(json_str)) {
          LOG(ERROR) << "pase json:" << json_str << " error!";
          continue;
        }

        put_map.insert(std::make_pair(instance_name, std::move(load_metrics)));

      } else if (event.event_type() == etcd::Event::EventType::DELETE_) {
        delete_list.push_back(instance_name);
      }
    }

    {
      std::unique_lock<std::shared_mutex> lock(load_metric_mutex_);
      for (auto& iter : put_map) {
        load_metrics_.insert_or_assign(iter.first, std::move(iter.second));
      }

      for (auto& iter : delete_list) {
        load_metrics_.erase(iter);
      }
    }
  });
}

void InstanceMgr::update_latency_metrics(
    const std::string& instance_name,
    const proto::LatencyMetrics& latency_metrics) {
  std::lock_guard<std::mutex> lock(latency_metrics_mutex_);

  LatencyMetrics metrics;
  for (const auto& entry : latency_metrics.model_metrics()) {
    const std::string& model_id = entry.first;
    const auto& proto_model_metrics = entry.second;
    
    LatencyMetrics::ModelLatencyMetrics model_metrics;
    model_metrics.recent_max_ttft = proto_model_metrics.recent_max_ttft();
    model_metrics.recent_max_tbt = proto_model_metrics.recent_max_tbt();
    
    metrics.model_metrics[model_id] = model_metrics;
  }
  
  latency_metrics_.insert_or_assign(instance_name, std::move(metrics));
}

void InstanceMgr::update_request_metrics(std::shared_ptr<Request> request,
                                         RequestAction action) {
  std::lock_guard<std::mutex> lock(request_metrics_mutex_);

  auto prefill_instance_it = request_metrics_.find(request->routing.prefill_name);
  if (prefill_instance_it == request_metrics_.end()) {
    LOG(ERROR) << "Failed to find prefill instance request metrics, instance name : "
               << request->routing.prefill_name;
    return;
  }

  auto prefill_model_it = prefill_instance_it->second.model_metrics.find(
      request->model);
  if (prefill_model_it == prefill_instance_it->second.model_metrics.end()) {
    LOG(ERROR) << "Failed to find prefill model request metrics, instance name : "
               << request->routing.prefill_name
               << ", model id : " << request->model;
    return;
  }

  if (request->routing.decode_name.empty()) {
    request->routing.decode_name = request->routing.prefill_name;
  }

  auto decode_instance_it = request_metrics_.find(request->routing.decode_name);
  if (decode_instance_it == request_metrics_.end()) {
    LOG(ERROR) << "Failed to find decode instance request metrics, instance name : "
               << request->routing.decode_name;
    return;
  }

  auto decode_model_it = decode_instance_it->second.model_metrics.find(
      request->model);
  if (decode_model_it == decode_instance_it->second.model_metrics.end()) {
    LOG(ERROR) << "Failed to find decode model request metrics, instance name : "
               << request->routing.decode_name
               << ", model id : " << request->model;
    return;
  }

  int64_t num_prompt_tokens = request->token_ids.size();
  int64_t num_generated_tokens = request->num_generated_tokens;
  switch (action) {
    case RequestAction::SCHEDULE:
      // update the request metrics for prefill and decode instances when
      // request is scheduled
      prefill_model_it->second.prefill_request_num += 1;
      prefill_model_it->second.prefill_token_num += num_prompt_tokens;

      decode_model_it->second.decode_request_num += 1;
      decode_model_it->second.decode_token_num += num_prompt_tokens;
      break;
    case RequestAction::FINISH_PREFILL:
      // update the request metrics for prefill and decode instance when request
      // finishes the prefill phase
      prefill_model_it->second.prefill_request_num -= 1;
      prefill_model_it->second.prefill_token_num -= num_prompt_tokens;
      prefill_instance_it->second.estimated_prefill_time -= request->estimated_ttft;

      decode_model_it->second.decode_token_num += 1;
      break;
    case RequestAction::GENERATE:
      // update the request metrics for decode instance when request generate a
      // token
      decode_model_it->second.decode_token_num += 1;
      break;
    case RequestAction::FINISH_DECODE:
      // update the request metrics for decode instance when request finishes
      // the decode phase
      decode_model_it->second.decode_request_num -= 1;
      decode_model_it->second.decode_token_num -=
          (num_prompt_tokens + num_generated_tokens);
      break;
    case RequestAction::CANCEL:
      // update the request metrics for prefill and decode instances when
      // request is cancelled
      prefill_model_it->second.prefill_request_num -= 1;
      prefill_model_it->second.prefill_token_num -= num_prompt_tokens;
      prefill_instance_it->second.estimated_prefill_time -= request->estimated_ttft;

      decode_model_it->second.decode_request_num -= 1;
      decode_model_it->second.decode_token_num -=
          (num_prompt_tokens + num_generated_tokens);
      break;
    default:
      LOG(ERROR) << "Unknown RequestAction: " << static_cast<int32_t>(action);
      break;
  }

  if (action == RequestAction::FINISH_PREFILL ||
      action == RequestAction::FINISH_DECODE ||
      action == RequestAction::CANCEL) {

    if (prefill_model_it->second.prefill_request_num == 0 &&
        prefill_model_it->second.decode_request_num == 0) {
      prefill_model_it->second.cv_idle.notify_all();
    }

    if (request->routing.prefill_name != request->routing.decode_name &&
        decode_model_it->second.prefill_request_num == 0 &&
        decode_model_it->second.decode_request_num == 0) {
      decode_model_it->second.cv_idle.notify_all();
    }

  }

  /*
  if (options_.load_balance_policy() == "SLO_AWARE" &&
      decode_it->second.decode_request_num == 0) {
    std::unique_lock<std::shared_mutex> instance_lock(inst_mutex_);
    flip_decode_to_prefill(request->routing.decode_name);
  }
  */
}

bool InstanceMgr::select_instance_pair_on_slo(
    std::shared_ptr<Request> request) {
  std::unique_lock<std::shared_mutex> lock(inst_mutex_);
  std::lock_guard<std::mutex> request_metrics_lock(request_metrics_mutex_);
  auto awake_instances = get_awake_instances(request->model);
  if (awake_instances.empty()) {
    LOG(ERROR) << "No awake instance found for model " << request->model;
    return false;
  }

  // get min prefill time instance from request metrics
  auto best_instance = awake_instances[0];
  int64_t min_prefill_time = std::numeric_limits<int64_t>::max();
  for (auto& instance : awake_instances) {
    int64_t prefill_time = request_metrics_[instance].estimated_prefill_time;
    if (prefill_time < min_prefill_time) {
      best_instance = instance;
      min_prefill_time = prefill_time;
    }
  }

  request->routing.prefill_name = best_instance;
  auto& time_predictor = get_time_predictor(best_instance);
  request->estimated_ttft =
      time_predictor.predict_ttft(request->model, request->token_ids.size());
  request_metrics_[best_instance].estimated_prefill_time +=
      request->estimated_ttft;

  return true;
}

void InstanceMgr::flip_prefill_to_decode(std::string& instance_name) {
  if (prefill_index_.size() <= 1) {
    // Ensure there is at least one prefill instance.
    return;
  }

  if (instances_.find(instance_name) == instances_.end()) {
    LOG(ERROR) << "Can't find instance, instance_name: " << instance_name;
    return;
  }

  // delete instance name from prefill_index_
  uint64_t index = instances_[instance_name].instance_index;
  std::swap(prefill_index_[index], prefill_index_.back());
  instances_[prefill_index_[index]].instance_index = index;
  prefill_index_.pop_back();

  // insert instance name to decode_index_
  instances_[instance_name].instance_index = decode_index_.size();
  instances_[instance_name].current_type = InstanceType::DECODE;
  decode_index_.emplace_back(instance_name);

  LOG(INFO) << "Flip prefill to decode, instance name : " << instance_name;
}

void InstanceMgr::flip_decode_to_prefill(std::string& instance_name) {
  if (decode_index_.size() <= 1) {
    // Ensure there is at least one decode instance.
    return;
  }

  if (instances_.find(instance_name) == instances_.end()) {
    LOG(ERROR) << "Can't find instance, instance_name: " << instance_name;
    return;
  }

  // delete instance name from decode_index_
  uint64_t index = instances_[instance_name].instance_index;
  std::swap(decode_index_[index], decode_index_.back());
  instances_[decode_index_[index]].instance_index = index;
  decode_index_.pop_back();

  // insert instance name to prefill_index
  instances_[instance_name].instance_index = prefill_index_.size();
  instances_[instance_name].current_type = InstanceType::PREFILL;
  prefill_index_.emplace_back(instance_name);

  LOG(INFO) << "Flip decode to prefill, instance name : " << instance_name;
}

TimePredictor& InstanceMgr::get_time_predictor(
    const std::string& instance_name) {
  std::lock_guard<std::mutex> lock(time_predictor_mutex_);

  auto it = time_predictors_.find(instance_name);
  if (it == time_predictors_.end()) {
    LOG(FATAL) << "Find TimePredictor failed, instance name : "
               << instance_name;
  }
  return it->second;
}

void InstanceMgr::send_model_sleep(const std::string& instance_name,
                                   const std::string& model_id) {

  if (instance_name.empty() || instance_name == "all") {
    LOG(ERROR) << "Only support fixed instance_name for model trigger now.";
    return;
  }

  // Use unique mutex for (instance_name, model_id) to serialize operations
  std::mutex* op_mutex = get_op_mutex(instance_name, model_id);
  std::lock_guard<std::mutex> lock(*op_mutex);

  {
    std::shared_lock<std::shared_mutex> lock(instance_model_state_mutex_);
    if (instance_model_states_[instance_name].count(model_id) &&
        instance_model_states_[instance_name][model_id] == ModelState::SLEEP) {
      LOG(INFO) << "Model " << model_id << " on " << instance_name
                << " is already sleeping.";
      return;  // Already sleep
    }
  }

  nlohmann::json sleep_body;
  sleep_body["model_id"] = model_id;
  sleep_body["master_status"] = 1;

  if (send_http_request(instance_name, "/sleep", sleep_body.dump())) {
    LOG(INFO) << "Model " << model_id << " on " << instance_name
              << " trigger sleep success.";
    std::unique_lock<std::shared_mutex> model_state_lock(instance_model_state_mutex_);
    std::unique_lock<std::shared_mutex> model_count_lock(model_count_mutex_);
    std::lock_guard<std::mutex> mem_lock(instance_memory_mutex_);
    
    if (instance_model_states_[instance_name][model_id] == ModelState::WAKEUP) {
      // Only decrease model count if the model is not DRAINING
      model_waking_up_counts_[model_id] -= 1;
      model_count_[model_id] -= 1;
    }
    
    instance_model_states_[instance_name][model_id] = ModelState::SLEEP;  // Sleep
    instance_memory_usage_[instance_name] -= get_model_memory_size(model_id);

    if (instance_memory_usage_[instance_name] < 0) {
      LOG(WARNING) << "Instance " << instance_name
                   << " memory usage negative, reset to 0.";
      instance_memory_usage_[instance_name] = 0;
    }

  } else {
    LOG(ERROR) << "Failed to sleep model " << model_id << " on "
               << instance_name;
  }
}

void InstanceMgr::send_model_wakeup(const std::string& instance_name,
                                    const std::string& model_id,
                                    bool memory_increased_in_advance) {// memory_increased_in_advance: for race conditions

  if (instance_name.empty() || instance_name == "all") {
    LOG(ERROR) << "Only support fixed instance_name for model trigger now.";
    return;
  }

  // Use unique mutex for (instance_name, model_id) to serialize operations
  std::mutex* op_mutex = get_op_mutex(instance_name, model_id);
  std::lock_guard<std::mutex> lock(*op_mutex);

  {
    std::unique_lock<std::shared_mutex> lock(instance_model_state_mutex_);
    if (instance_model_states_[instance_name].count(model_id) &&
        (instance_model_states_[instance_name][model_id] == ModelState::WAKEUP ||
         instance_model_states_[instance_name][model_id] == ModelState::SENDING_WAKEUP_REQUEST)) {
      LOG(INFO) << "Model " << model_id << " on " << instance_name
                << " is already wakeup or sending wakeup request.";
      return;  // Already wakeup
    }
    instance_model_states_[instance_name][model_id] = ModelState::SENDING_WAKEUP_REQUEST;
  }

  nlohmann::json wakeup_body;
  wakeup_body["model_id"] = model_id;
  wakeup_body["master_status"] = 0;

  if (send_http_request(instance_name, "/wakeup", wakeup_body.dump())) {
    LOG(INFO) << "Model " << model_id << " on " << instance_name
              << " trigger wakeup success.";
    std::unique_lock<std::shared_mutex> inst_lock(instance_model_state_mutex_);
    std::unique_lock<std::shared_mutex> model_count_lock(model_count_mutex_);
    instance_model_states_[instance_name][model_id] = ModelState::WAKEUP;  // Wakeup
    model_count_[model_id] += 1;

    if (!memory_increased_in_advance) {
      std::lock_guard<std::mutex> mem_lock(instance_memory_mutex_);
      instance_memory_usage_[instance_name] += get_model_memory_size(model_id);
    }

    if (instance_memory_usage_[instance_name] > kMaxInstanceMemoryGB) {
      LOG(WARNING) << "Instance " << instance_name
                   << " memory usage exceeds max limit after waking up model "
                   << model_id << ".";
    }
  } else {
    LOG(ERROR) << "Failed to wakeup model " << model_id
               << " on " << instance_name;
    {
      // Wakeup failed, revert status to SLEEP or handle accordingly
      std::unique_lock<std::shared_mutex> inst_lock(instance_model_state_mutex_);
      instance_model_states_[instance_name][model_id] = ModelState::SLEEP;
    }
    if (memory_increased_in_advance) {
      std::lock_guard<std::mutex> mem_lock(instance_memory_mutex_);
      instance_memory_usage_[instance_name] -= get_model_memory_size(model_id);
    }
  }
}

void InstanceMgr::init_model_memory_specs() {
    // Hardcoded memory specs: x * 2 + 5 GB. 5GB = 3GB KV cache + 2GB overhead

    // just for test
    model_memory_specs_["Qwen3-4B"] = 25.0;
    model_memory_specs_["Qwen2-7B"] = 25.0;
    model_memory_specs_["Qwen3-8B"] = 25.0;
    

    // model_memory_specs_["Qwen3-8B"] = 8.0 * 2 + 5.0;// 21.0GB
    // model_memory_specs_["Qwen2-7B"] = 7.0 * 2 + 5.0;// 19.0GB
    // model_memory_specs_["Qwen2-7B-Instruct"] = 7.0 * 2 + 5.0;// 19.0GB
    // model_memory_specs_["Qwen2.5-14B"] = 14.0 * 2 + 5.0;// 33.0GB
    // model_memory_specs_["Qwen3-4B"] = 4.0 * 2 + 5.0;// 13.0GB
    // model_memory_specs_["Qwen2.5-3b"] = 3.0 * 2 + 5.0;// 11.0GB
    // model_memory_specs_["Qwen3-30B-A3B-Instruct-2507"] = 57.0 + 5.0;// 62.0GB
    // model_memory_specs_["Qwen3-30B-A3B-W8A8"] = 30.0 + 5.0;// 35.0GB
    // model_memory_specs_["Qwen3-32B-W8A8"] = 40.0 + 5.0;// 45.0GB
}

// TODO: support dynamic instance memory specs, rather than hardcoded.
double InstanceMgr::get_model_memory_size(const std::string& model_id) {
    if (model_memory_specs_.count(model_id)) {
        return model_memory_specs_[model_id];
    }
    LOG(WARNING) << "Unknown model ID for memory spec: " << model_id << ", using default 20GB";
    return 20.0; 
}

bool InstanceMgr::is_model_waking_up(const std::string& model_id) {
  std::shared_lock<std::shared_mutex> lock(instance_model_state_mutex_);
  for (auto& pair : instance_model_states_) {
    auto it = pair.second.find(model_id);
    if (it != pair.second.end() &&
        (it->second == ModelState::WAKING_UP ||
         it->second == ModelState::SENDING_WAKEUP_REQUEST)) {
      return true;
    }
  }
  return false;
}

std::vector<std::string> InstanceMgr::get_awake_instances(const std::string& model_id) {
  std::shared_lock<std::shared_mutex> lock(instance_model_state_mutex_);
  std::vector<std::string> awake_instances;
  for (const auto& instance_pair : instance_model_states_) {
    const std::string& instance_name = instance_pair.first;
    const auto& model_states = instance_pair.second;
    
    if (model_states.count(model_id) && model_states.at(model_id) == ModelState::WAKEUP) {
      awake_instances.push_back(instance_name);
    }
  }
  return awake_instances;
}

int32_t InstanceMgr::get_model_count(const std::string& model_id) {
  std::shared_lock<std::shared_mutex> lock(model_count_mutex_);
  return model_count_[model_id];
}

std::string InstanceMgr::wait_for_model_wakeup(const std::string& model_id,
                                               std::chrono::milliseconds timeout_ms) {
  std::unique_lock<std::mutex> lock(wakeup_mutex_);
  bool not_timeout = wakeup_cv_.wait_for(lock, timeout_ms, [this, model_id]() {
    return !is_model_waking_up(model_id);
  });
  if (!not_timeout) {
    return ""; // Timeout, return empty instance_name
  }
  std::string instance_name;
  auto it = wakeup_instance_name_.find(model_id);
  if (it != wakeup_instance_name_.end()) {
    return it->second;
  }
  return "";
}

void InstanceMgr::notify_model_wakeup(const std::string& model_id,
                                      const std::string& instance_name) {
  // Just notify all waiting threads to re-check
  {
    std::lock_guard<std::mutex> lock(wakeup_mutex_);
    wakeup_instance_name_[model_id] = instance_name;
  }
  wakeup_cv_.notify_all();
}

void InstanceMgr::update_model_heat(const std::string& model_id, int64_t token_count) {
  std::lock_guard<std::mutex> lock(model_heat_mutex_);
  prune_model_heat_locked(model_id);
  model_heat_records_[model_id].push_back({std::chrono::steady_clock::now(), token_count});
  global_model_heat_[model_id] += token_count;
}

std::string InstanceMgr::allocate_instance_for_model(const std::string& model_id,
                                                     int32_t target_model_count) {
  
  std::unique_lock<std::mutex> allocation_lock(allocation_mutex_);
  // check for race conditions
  // (multiple entrance in allocate_instance_for_model)
  {
    std::shared_lock<std::shared_mutex> inst_lock(inst_mutex_);
    std::shared_lock<std::shared_mutex> model_state_lock(instance_model_state_mutex_);
    if (target_model_count == model_waking_up_counts_[model_id]) {
      LOG(INFO) << "Model " << model_id << " is already being allocated to target count "
                << target_model_count << ". Give up allocation.";
      for (auto &inst_pair : instances_) {
        std::string instance_name = inst_pair.first;
        if (instance_model_states_[instance_name][model_id] == ModelState::WAKEUP) {
          LOG(INFO) << "Model " << model_id << " on " << instance_name
                    << " is already awake. Give up allocation.";
          return instance_name;
        } else if (instance_model_states_[instance_name][model_id] == ModelState::WAKING_UP ||
                   instance_model_states_[instance_name][model_id] == ModelState::SENDING_WAKEUP_REQUEST) {
          LOG(INFO) << "Model " << model_id << " on " << instance_name
                    << " is waking up. Give up allocation.";
          allocation_lock.unlock();
          return wait_for_model_wakeup(model_id, std::chrono::milliseconds(kMaxWakeupTimeoutms));
        }
      }
      LOG(ERROR) << "Model " << model_id << " fails to find existing waking-up instances.";
      return ""; // Already being allocated to target cou
    }
  }

  double model_size = get_model_memory_size(model_id);
  
  LOG(INFO) << "Allocating instance for model " << model_id 
            << " with size " << model_size << " GB.";

  // First, try to find an instance with enough free space
  {
    std::unique_lock<std::mutex> mem_lock(instance_memory_mutex_);
    for (const auto& pair : instance_memory_usage_) {
      const std::string& instance_name = pair.first;

      {
        std::shared_lock<std::shared_mutex> lock(instance_model_state_mutex_);
        if (instance_model_states_[instance_name][model_id] != ModelState::SLEEP) {
          // Model not ready to wake up.
          // Note that DRAINING does not contribute to model_waking_up_counts,
          // but expecting DRAINING->SLEEP->WAKEUP_AGAIN yields too much race conditions.
          // We would rather expect a temporary -1 margin on model_waking_up_counts,
          // which is to be fixed by triggering allocate_instance_for_model again during the next auto_scaling.
          continue;
        }
      }

      double current_usage = pair.second;

      LOG(INFO) << "Instance " << instance_name 
                << " current memory usage: " << current_usage << " GB.";

      if (current_usage + model_size <= kMaxInstanceMemoryGB) {
        std::unique_lock<std::shared_mutex> model_state_lock(instance_model_state_mutex_);
        instance_memory_usage_[instance_name] += model_size;
        instance_model_states_[instance_name][model_id] = ModelState::WAKING_UP;
        model_waking_up_counts_[model_id] += 1;
        model_state_lock.unlock();
        mem_lock.unlock();
        allocation_lock.unlock();
        send_model_wakeup(instance_name, model_id, /*memory_increased_in_advance*/ true);
        return instance_name;
      }
    }
  }

  // If no instance has enough space, we need to evict.
  // Strategy: Find the instance where we can free up enough space by evicting the *coldest* models.
  
  std::string best_candidate_instance;
  std::vector<std::string> best_eviction_plan;
  uint64_t min_evicted_heat_sum = std::numeric_limits<uint64_t>::max();

  {
    std::shared_lock<std::shared_mutex> inst_lock(inst_mutex_); // Iterate instances safely
    for (const auto& inst_pair : instances_) {
      std::string instance_name = inst_pair.first;

      {
        std::shared_lock<std::shared_mutex> lock(instance_model_state_mutex_);
        if (instance_model_states_[instance_name][model_id] != ModelState::SLEEP) {
          // Model not ready.
          // This double check is for race conditions. (After allocation_lock, are there race conditions?)
          continue;
        }
      }
      
      // Check current usage
      double current_usage = 0;
      {
        std::lock_guard<std::mutex> mem_lock(instance_memory_mutex_);
        if (instance_memory_usage_.count(instance_name)) {
          current_usage = instance_memory_usage_[instance_name];
        }
      }

      double space_needed = model_size - (kMaxInstanceMemoryGB - current_usage);

      auto candidates = select_eviction_candidates(instance_name, space_needed);
      if (candidates.empty()) {
        continue; // Cannot free enough space on this instance
      }

      // Calculate total heat of candidates
      uint64_t current_plan_heat = 0;
      {
        std::lock_guard<std::mutex> heat_lock(model_heat_mutex_);
        for (const auto& mod : candidates) {
          if (global_model_heat_.count(mod)) {
            current_plan_heat += global_model_heat_[mod];
          }
        }
      }

      if (current_plan_heat < min_evicted_heat_sum) {
        min_evicted_heat_sum = current_plan_heat;
        best_candidate_instance = instance_name;
        best_eviction_plan = candidates;
      }
    }
  }

  if (!best_candidate_instance.empty()) {
    {
      std::unique_lock<std::shared_mutex> inst_lock(instance_model_state_mutex_);
      std::unique_lock<std::shared_mutex> count_lock(model_count_mutex_);
      
      // early mark as WAKING_UP, for race conditions 
      // (multiple entrance in allocate_instance_for_model)
      instance_model_states_[best_candidate_instance][model_id] = ModelState::WAKING_UP;
      model_waking_up_counts_[model_id] += 1;
      for (const auto& model_to_sleep : best_eviction_plan) {
        instance_model_states_[best_candidate_instance][model_to_sleep] = ModelState::DRAINING;
        model_waking_up_counts_[model_to_sleep] -= 1;
        model_count_[model_to_sleep] -= 1;
      }
    }

    // Execute eviction

    std::vector<std::thread> eviction_threads;

    for (const auto& model_to_sleep : best_eviction_plan) {

      eviction_threads.emplace_back([this, best_candidate_instance, model_to_sleep]() {
        std::unique_lock<std::mutex> request_metrics_lock(request_metrics_mutex_);
        auto instance_it = request_metrics_.find(best_candidate_instance);
        if (instance_it == request_metrics_.end()) {
          LOG(ERROR) << "Failed to find request metrics for instance "
                     << best_candidate_instance << " during eviction.";
          return;
        }
        auto model_it = instance_it->second.model_metrics.find(model_to_sleep);
        if (model_it == instance_it->second.model_metrics.end()) {
          LOG(ERROR) << "Failed to find request metrics for model "
                     << model_to_sleep << " on instance "
                     << best_candidate_instance << " during eviction.";
          return;
        }
        
        auto& metrics = model_it->second;
        metrics.cv_idle.wait(request_metrics_lock, [&metrics]() {
          return metrics.prefill_request_num == 0 && metrics.decode_request_num == 0;
        });
        
        request_metrics_lock.unlock();
        send_model_sleep(best_candidate_instance, model_to_sleep);
      });
      
    }

    for (auto& thread : eviction_threads) {
      if (thread.joinable()) {
        thread.join();
      }
    }

    {
      std::lock_guard<std::mutex> mem_lock(instance_memory_mutex_);
      instance_memory_usage_[best_candidate_instance] += model_size;
    }

    allocation_lock.unlock();
    send_model_wakeup(best_candidate_instance, model_id, /*memory_increased_in_advance*/ true);
    return best_candidate_instance;
  } else {
    LOG(INFO) << "Failed to allocate instance for model " << model_id
              << ": no suitable instance found for eviction.";
  }

  return ""; // Failed to allocate
}

// Select models to evict out >= required_space, meanwhile minimizing sum(heat)
std::vector<std::string> InstanceMgr::select_eviction_candidates(const std::string& instance_name, 
                                                                 double required_space) {
    
  // Get all awake models on this instance
  std::vector<std::string> awake_models;
  {
    std::shared_lock<std::shared_mutex> model_state_lock(instance_model_state_mutex_);
    std::shared_lock<std::shared_mutex> model_count_lock(model_count_mutex_);
    if (instance_model_states_.count(instance_name)) {
      for (const auto& pair : instance_model_states_.at(instance_name)) {
        if (pair.second == ModelState::WAKEUP &&
            model_count_[pair.first] > 1) {
          awake_models.push_back(pair.first);
        }
        if (pair.second == ModelState::WAKEUP &&
            model_count_[pair.first] == 1) {
          std::lock_guard<std::mutex> lock(model_heat_mutex_);
          prune_model_heat_locked(pair.first);
          if (global_model_heat_[pair.first] == 0) {
            awake_models.push_back(pair.first);
          }
        }
      }
    }
  }

  // Select models to evict enough space for the new model, meanwhile minimizing sum(heat)
  std::vector<uint64_t> awake_model_heats;
  {
    std::lock_guard<std::mutex> lock(model_heat_mutex_);
    for (const auto& model_id : awake_models) {
      prune_model_heat_locked(model_id);
      awake_model_heats.push_back(global_model_heat_[model_id]);
    }
  }

  uint64_t min_sum_heat = std::numeric_limits<uint64_t>::max();
  size_t best_subset = 0;
  for (size_t subset = 0; subset < 1 << awake_models.size(); ++subset) {
    double current_sum_space = 0;
    uint64_t current_sum_heat = 0;
    for (size_t i = 0; i < awake_models.size(); ++i) {
      if (subset & (1 << i)) {
        current_sum_space += get_model_memory_size(awake_models[i]);
        current_sum_heat += awake_model_heats[i];
      }
    }
    if (current_sum_space >= required_space && 
        current_sum_heat < min_sum_heat) {
      min_sum_heat = current_sum_heat;
      best_subset = subset;
    }
  }

  if (best_subset == 0) {
    return {};// Cannot free enough space even if we evict everything
  }

  std::vector<std::string> candidates;
  for (size_t i = 0; i < awake_models.size(); ++i) {
    if (best_subset & (1 << i)) {
      candidates.push_back(awake_models[i]);
    }
  }

  return candidates;
}
  
std::mutex* InstanceMgr::get_op_mutex(const std::string& instance_name,
                                      const std::string& model_id) {
  std::string key = instance_name + ":" + model_id;
  std::lock_guard<std::mutex> lock(op_mutex_map_mutex_);
  if (op_mutexes_.find(key) == op_mutexes_.end()) {
    op_mutexes_[key] = std::make_unique<std::mutex>();
  }
  return op_mutexes_[key].get();
}

void InstanceMgr::auto_scaling() {
  // 1. Identify the hottest model
  std::string hottest_model = "";
  int64_t max_heat = 0;

  std::string second_hottest_model = "";
  int64_t second_max_heat = 0;
  
  {
    std::lock_guard<std::mutex> lock(model_heat_mutex_);
    for (const auto& pair : global_model_heat_) {
      prune_model_heat_locked(pair.first);
      if (pair.second > max_heat) {

        second_max_heat = max_heat;
        second_hottest_model = hottest_model;

        max_heat = pair.second;
        hottest_model = pair.first;

      } else if (pair.second > second_max_heat) {
        second_max_heat = pair.second;
        second_hottest_model = pair.first;
      }
    }
  }

  if (hottest_model.empty()) {
    return;
  }
  
  LOG(INFO) << "Current model heats:";
  {
    std::lock_guard<std::mutex> lock(model_heat_mutex_);
    for (const auto& pair : global_model_heat_) {
      LOG(INFO) << "Model " << pair.first << ": Heat = " << pair.second;
    }
  }

  LOG(INFO) << "Hottest model: " << (hottest_model.empty() ? "None" : hottest_model) << " with heat: " << max_heat;

  bool hottest_model_needs_scale_up = true;

  {
    std::shared_lock<std::shared_mutex> lock(model_count_mutex_);
    if (model_count_[hottest_model] == 2) {
      LOG(INFO) << "No need to scale up, hottest model " << hottest_model << " already has 2 instances.";
      hottest_model_needs_scale_up = false;
    }
  }

  if (hottest_model_needs_scale_up) {
    auto instance_name = allocate_instance_for_model(hottest_model, /*target_model_count*/ 2);

    if (instance_name.empty()) {
      LOG(ERROR) << "Auto scaling failed: unable to allocate instance for hottest model " << hottest_model;
    } else {
      LOG(INFO) << "Auto scaling: allocated instance " << instance_name << " for hottest model " << hottest_model;
    }
  }

  if (second_hottest_model.empty()) {
    return;
  }

  bool second_hottest_model_needs_scale_up = true;

  LOG(INFO) << "Second hottest model: " << second_hottest_model << " with heat: " << second_max_heat;

  {
    std::shared_lock<std::shared_mutex> lock(model_count_mutex_);
    if (model_count_[second_hottest_model] == 2) {
      LOG(INFO) << "No need to scale up, second hottest model " << second_hottest_model << " already has 2 instances.";
      second_hottest_model_needs_scale_up = false;
    }
  }

  if (second_hottest_model_needs_scale_up) {
    auto instance_name = allocate_instance_for_model(second_hottest_model, /*target_model_count*/ 2);

    if (instance_name.empty()) {
      LOG(ERROR) << "Auto scaling failed: unable to allocate instance for second hottest model " << second_hottest_model;
    } else {
      LOG(INFO) << "Auto scaling: allocated instance " << instance_name << " for second hottest model " << second_hottest_model;
    }
  }
}

void InstanceMgr::prune_model_heat_locked(const std::string& model_id) {
  auto& records = model_heat_records_[model_id];
  auto now = std::chrono::steady_clock::now();
  while (!records.empty()) {
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - records.front().timestamp).count();
    if (duration > 5) {
      global_model_heat_[model_id] -= records.front().token_count;
      records.pop_front();
    } else {
      break;
    }
  }
  if (global_model_heat_[model_id] < 0) {
    global_model_heat_[model_id] = 0;
  }
}

}  // namespace xllm_service
