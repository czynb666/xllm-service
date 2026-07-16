/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

// Unit tests for GlobalKVCacheMgr::match() — Phase 3 (§3.3-2).
// Covers HBM/DRAM/SSD partial-tier hits, empty-set skipping (no wrong
// hbm.begin() indexing), instance-down behaviour, block-boundary alignment and
// the n_tokens==0 early return.
//
// Strategy: with enable_peer_service=true the ctor skips all etcd access, so we
// pass a null EtcdClient and seed the index directly through
// record_updated_kvcaches / replace_instance_kvcaches. Index keys are computed
// with the same chained xxh3_128bits_hash that match() uses, so a seeded token
// sequence is discoverable by match().

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "common/hash_util.h"
#include "common/options.h"
#include "common/slice.h"
#include "scheduler/managers/global_kvcache_mgr.h"
#include "xllm_rpc_service.pb.h"

namespace xllm_service {
namespace {

constexpr int32_t kBlockSize = 128;

Options make_options() {
  Options options;
  options.block_size(kBlockSize).enable_peer_service(true);
  return options;
}

// Deterministic token ids spanning `n_blocks` full blocks. Distinct per block
// so each block hashes to a distinct key.
std::vector<int32_t> make_tokens(int32_t n_blocks) {
  std::vector<int32_t> tokens;
  tokens.reserve(n_blocks * kBlockSize);
  for (int32_t b = 0; b < n_blocks; ++b) {
    for (int32_t i = 0; i < kBlockSize; ++i) {
      tokens.push_back(b * 1000 + i);
    }
  }
  return tokens;
}

// Chained per-block hash keys, identical to how match() walks the tokens.
std::vector<XXH3Key> block_keys(const std::vector<int32_t>& tokens) {
  Slice<int32_t> slice(tokens);
  const size_t n_tokens = (tokens.size() / kBlockSize) * kBlockSize;
  std::vector<XXH3Key> keys;
  keys.reserve(n_tokens / kBlockSize);
  XXH3Key key;
  for (size_t i = 0; i < n_tokens; i += kBlockSize) {
    if (i == 0) {
      xxh3_128bits_hash(nullptr, slice.slice(i, i + kBlockSize), key.data);
    } else {
      xxh3_128bits_hash(key.data, slice.slice(i, i + kBlockSize), key.data);
    }
    keys.emplace_back(key.data);
  }
  return keys;
}

// Seed the first `n_blocks` block keys as HBM-stored on `instance_name`.
void seed_hbm(GlobalKVCacheMgr* mgr,
              const std::string& instance_name,
              const std::vector<XXH3Key>& keys,
              int32_t n_blocks) {
  proto::KvCacheEvent event;
  for (int32_t i = 0; i < n_blocks; ++i) {
    event.add_stored_cache(reinterpret_cast<const char*>(keys[i].data),
                           XXH3_128BITS_HASH_VALUE_LEN);
  }
  mgr->record_updated_kvcaches(instance_name, event);
}

// Push already-HBM keys down one tier (HBM->DRAM, or DRAM->SSD) via offload.
void offload_once(GlobalKVCacheMgr* mgr,
                  const std::string& instance_name,
                  const std::vector<XXH3Key>& keys,
                  int32_t n_blocks) {
  proto::KvCacheEvent event;
  for (int32_t i = 0; i < n_blocks; ++i) {
    event.add_offload_cache(reinterpret_cast<const char*>(keys[i].data),
                            XXH3_128BITS_HASH_VALUE_LEN);
  }
  mgr->record_updated_kvcaches(instance_name, event);
}

GlobalKVCacheMgr make_mgr() {
  return GlobalKVCacheMgr(
      make_options(), std::shared_ptr<EtcdClient>(nullptr), /*is_master=*/false);
}

}  // namespace

// n_tokens == 0 (fewer than one block) -> early return, no scores.
TEST(GlobalKVCacheMgrMatchTest, EmptyBelowOneBlockReturnsEarly) {
  GlobalKVCacheMgr mgr = make_mgr();
  std::vector<int32_t> tokens(kBlockSize - 1, 7);  // < 1 block
  Slice<int32_t> slice(tokens);

  OverlapScores scores;
  mgr.match(slice, &scores);

  EXPECT_EQ(scores.max_block_num, 0u);
  EXPECT_EQ(scores.max_matched_block_num, 0u);
  EXPECT_TRUE(scores.instances.empty());
  EXPECT_TRUE(scores.max_matched_instance_name.empty());
}

// Empty index -> aligned block count reported, but no matches.
TEST(GlobalKVCacheMgrMatchTest, EmptyIndexNoMatch) {
  GlobalKVCacheMgr mgr = make_mgr();
  std::vector<int32_t> tokens = make_tokens(3);
  Slice<int32_t> slice(tokens);

  OverlapScores scores;
  mgr.match(slice, &scores);

  EXPECT_EQ(scores.max_block_num, 3u);
  EXPECT_EQ(scores.max_matched_block_num, 0u);
  EXPECT_TRUE(scores.instances.empty());
}

// Full HBM hit on all 3 blocks.
TEST(GlobalKVCacheMgrMatchTest, HbmFullHit) {
  GlobalKVCacheMgr mgr = make_mgr();
  std::vector<int32_t> tokens = make_tokens(3);
  std::vector<XXH3Key> keys = block_keys(tokens);
  seed_hbm(&mgr, "inst_a", keys, /*n_blocks=*/3);

  OverlapScores scores;
  mgr.match(Slice<int32_t>(tokens), &scores);

  EXPECT_EQ(scores.max_block_num, 3u);
  EXPECT_EQ(scores.max_matched_block_num, 3u);
  EXPECT_EQ(scores.max_matched_instance_name, "inst_a");
  ASSERT_EQ(scores.hbm_instance_score.count("inst_a"), 1u);
  EXPECT_EQ(scores.hbm_instance_score.at("inst_a"), 3u);
  EXPECT_TRUE(scores.dram_instance_score.empty());
  EXPECT_TRUE(scores.ssd_instance_score.empty());
}

// Partial prefix hit: only first 2 of 3 blocks stored -> match stops at 2.
TEST(GlobalKVCacheMgrMatchTest, HbmPartialPrefixHit) {
  GlobalKVCacheMgr mgr = make_mgr();
  std::vector<int32_t> tokens = make_tokens(3);
  std::vector<XXH3Key> keys = block_keys(tokens);
  seed_hbm(&mgr, "inst_a", keys, /*n_blocks=*/2);  // 3rd block absent

  OverlapScores scores;
  mgr.match(Slice<int32_t>(tokens), &scores);

  EXPECT_EQ(scores.max_block_num, 3u);
  EXPECT_EQ(scores.max_matched_block_num, 2u);
  ASSERT_EQ(scores.hbm_instance_score.count("inst_a"), 1u);
  EXPECT_EQ(scores.hbm_instance_score.at("inst_a"), 2u);
}

// DRAM-only tier hit: keys offloaded HBM->DRAM. match() must read the DRAM set,
// NOT wrongly index the empty hbm_instance_set (the fixed begin() bug).
TEST(GlobalKVCacheMgrMatchTest, DramOnlyTierHit) {
  GlobalKVCacheMgr mgr = make_mgr();
  std::vector<int32_t> tokens = make_tokens(2);
  std::vector<XXH3Key> keys = block_keys(tokens);
  seed_hbm(&mgr, "inst_a", keys, /*n_blocks=*/2);
  offload_once(&mgr, "inst_a", keys, /*n_blocks=*/2);  // HBM -> DRAM

  OverlapScores scores;
  mgr.match(Slice<int32_t>(tokens), &scores);

  EXPECT_EQ(scores.max_matched_block_num, 2u);
  EXPECT_EQ(scores.max_matched_instance_name, "inst_a");
  ASSERT_EQ(scores.dram_instance_score.count("inst_a"), 1u);
  EXPECT_EQ(scores.dram_instance_score.at("inst_a"), 2u);
  EXPECT_TRUE(scores.hbm_instance_score.empty());  // HBM set is empty -> skipped
  EXPECT_TRUE(scores.ssd_instance_score.empty());
}

// SSD-only tier hit: keys offloaded twice HBM->DRAM->SSD.
TEST(GlobalKVCacheMgrMatchTest, SsdOnlyTierHit) {
  GlobalKVCacheMgr mgr = make_mgr();
  std::vector<int32_t> tokens = make_tokens(2);
  std::vector<XXH3Key> keys = block_keys(tokens);
  seed_hbm(&mgr, "inst_a", keys, /*n_blocks=*/2);
  offload_once(&mgr, "inst_a", keys, /*n_blocks=*/2);  // HBM -> DRAM
  offload_once(&mgr, "inst_a", keys, /*n_blocks=*/2);  // DRAM -> SSD

  OverlapScores scores;
  mgr.match(Slice<int32_t>(tokens), &scores);

  EXPECT_EQ(scores.max_matched_block_num, 2u);
  ASSERT_EQ(scores.ssd_instance_score.count("inst_a"), 1u);
  EXPECT_EQ(scores.ssd_instance_score.at("inst_a"), 2u);
  EXPECT_TRUE(scores.hbm_instance_score.empty());
  EXPECT_TRUE(scores.dram_instance_score.empty());
}

// Two instances, same prefix in different tiers: both surface with per-tier
// scores; the empty tiers of each are skipped (no wrong-set indexing).
TEST(GlobalKVCacheMgrMatchTest, MultiInstanceMixedTiers) {
  GlobalKVCacheMgr mgr = make_mgr();
  std::vector<int32_t> tokens = make_tokens(2);
  std::vector<XXH3Key> keys = block_keys(tokens);
  seed_hbm(&mgr, "inst_hbm", keys, /*n_blocks=*/2);   // stays in HBM
  seed_hbm(&mgr, "inst_dram", keys, /*n_blocks=*/2);
  offload_once(&mgr, "inst_dram", keys, /*n_blocks=*/2);  // this one -> DRAM

  OverlapScores scores;
  mgr.match(Slice<int32_t>(tokens), &scores);

  EXPECT_EQ(scores.max_matched_block_num, 2u);
  ASSERT_EQ(scores.hbm_instance_score.count("inst_hbm"), 1u);
  EXPECT_EQ(scores.hbm_instance_score.at("inst_hbm"), 2u);
  ASSERT_EQ(scores.dram_instance_score.count("inst_dram"), 1u);
  EXPECT_EQ(scores.dram_instance_score.at("inst_dram"), 2u);
  EXPECT_EQ(scores.instances.count("inst_hbm"), 1u);
  EXPECT_EQ(scores.instances.count("inst_dram"), 1u);
}

// Instance-down: after clear_instance_cache the entries vanish -> no match.
TEST(GlobalKVCacheMgrMatchTest, InstanceDownClearsMatch) {
  GlobalKVCacheMgr mgr = make_mgr();
  std::vector<int32_t> tokens = make_tokens(3);
  std::vector<XXH3Key> keys = block_keys(tokens);
  seed_hbm(&mgr, "inst_a", keys, /*n_blocks=*/3);

  OverlapScores before;
  mgr.match(Slice<int32_t>(tokens), &before);
  ASSERT_EQ(before.max_matched_block_num, 3u);

  mgr.clear_instance_cache("inst_a");  // instance goes down

  OverlapScores after;
  mgr.match(Slice<int32_t>(tokens), &after);
  EXPECT_EQ(after.max_matched_block_num, 0u);
  EXPECT_TRUE(after.instances.empty());
}

// Snapshot rebuild: replace_instance_kvcaches fully replaces an instance's
// footprint; stale keys drop and only the snapshot's keys match.
TEST(GlobalKVCacheMgrMatchTest, SnapshotReplaceRebuildsIndex) {
  GlobalKVCacheMgr mgr = make_mgr();
  std::vector<int32_t> old_tokens = make_tokens(2);
  std::vector<int32_t> new_tokens = make_tokens(3);
  // shift new_tokens so its block keys differ from old_tokens
  for (int32_t& t : new_tokens) {
    t += 500000;
  }
  std::vector<XXH3Key> old_keys = block_keys(old_tokens);
  std::vector<XXH3Key> new_keys = block_keys(new_tokens);
  seed_hbm(&mgr, "inst_a", old_keys, /*n_blocks=*/2);

  // Snapshot carries only the new keys.
  proto::KvCacheEvent snapshot;
  for (int32_t i = 0; i < 3; ++i) {
    snapshot.add_stored_cache(reinterpret_cast<const char*>(new_keys[i].data),
                              XXH3_128BITS_HASH_VALUE_LEN);
  }
  mgr.replace_instance_kvcaches("inst_a", snapshot);

  // Old prefix no longer matches.
  OverlapScores old_scores;
  mgr.match(Slice<int32_t>(old_tokens), &old_scores);
  EXPECT_EQ(old_scores.max_matched_block_num, 0u);

  // New prefix matches all 3 blocks.
  OverlapScores new_scores;
  mgr.match(Slice<int32_t>(new_tokens), &new_scores);
  EXPECT_EQ(new_scores.max_matched_block_num, 3u);
  EXPECT_EQ(new_scores.max_matched_instance_name, "inst_a");
}

}  // namespace xllm_service
