include(cc_library)
include(cc_test)

cc_library(
  NAME
   managers
  HDRS
    instance_mgr.h
    global_kvcache_mgr.h
    kv_event_subscriber.h
  SRCS
    instance_mgr.cpp
    global_kvcache_mgr.cpp
    kv_event_subscriber.cpp
  DEPS
    :chat_template
    :common
    :etcd_client
    :request
    absl::random_random
    absl::strings
    glog::glog
    proto::proto_rpc_service
    proto_xllm
)
target_link_libraries(managers PRIVATE brpc-static cppzmq)

# Phase 3 §3.3-2: unit tests for GlobalKVCacheMgr::match() (built only with
# -DBUILD_TESTING=ON; cc_test is a no-op otherwise).
cc_test(
  NAME
    global_kvcache_mgr_match_test
  SRCS
    global_kvcache_mgr_match_test.cpp
  DEPS
    :managers
    :common
    :etcd_client
    GTest::gtest_main
    glog::glog
    proto::proto_rpc_service
)
target_link_libraries(global_kvcache_mgr_match_test PRIVATE brpc-static cppzmq)
