py_test(
    name = "py_runtime_rl_tests",
    srcs = ["py_runtime_rl_tests.py"],
    data = ["@bark_project//bark:generate_core",
            "//tests/data:bark_data"],
    imports = ["../external/bark_project/python/"],
    deps = ["@bark_project//bark/runtime:runtime",
            "@bark_project//bark/runtime/scenario/scenario_generation:scenario_generation",
            "//src:rl_runtime",
            "//src/agents:agents"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "py_runtime_tfa_tests",
    srcs = ["py_runtime_tfa_tests.py"],
    data = ["@bark_project//bark:generate_core",
            "//tests/data:bark_data"],
    imports = ["../external/bark_project/python/"],
    deps = ["@bark_project//bark/runtime:runtime",
            "@bark_project//bark/runtime/scenario/scenario_generation:scenario_generation",
            "//src:rl_runtime",
            "//src/agents:agents"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "py_agent_tests",
    srcs = ["py_agent_tests.py"],
    data = ["@bark_project//bark:generate_core",
            "//tests/data:bark_data"],
    imports = ["../external/bark_project/python/"],
    deps = ["@bark_project//bark/runtime:runtime",
            "@bark_project//bark/runtime/scenario/scenario_generation:scenario_generation",
            "//src:rl_runtime",
            "//src/agents:agents"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "py_runner_tests",
    srcs = ["py_runner_tests.py"],
    data = ["@bark_project//bark:generate_core",
            "//tests/data:bark_data"],
    imports = ["../external/bark_project/python/"],
    deps = ["@bark_project//bark/runtime:runtime",
            "@bark_project//bark/runtime/scenario/scenario_generation:scenario_generation",
            "//src:rl_runtime",
            "//src/agents:agents",
            "//src/runners:runners"],
    visibility = ["//visibility:public"],
)

py_test(
  name = "py_runtime_tests",
  srcs = ["py_runtime_tests.py"],
  data = ["@bark_project//bark:generate_core",
          "//tests/data:bark_data"],
  imports = ["../external/bark_project/python/"],
  deps = ["@bark_project//bark/runtime:runtime",
          "@bark_project//bark/runtime/scenario/scenario_generation:scenario_generation"],
)

cc_test(
  name = "observer_tests",
  srcs = ["observer_tests_new.cc"],
  copts = ["-Iexternal/gtest/include"],
  deps = [
    "//src/observers:nearest_observer",
    "@bark_project//bark/world/tests:make_test_world",
    "@gtest//:main",
  ]
)

cc_test(
    name="model_loader_test",
    srcs=["model_loader_test.cc"],
    copts = ["-Iexternal/gtest/include"],
    deps=[
        "//src/model_loader:ModelLoader",
        "@libtensorflow-RL-MCTS//libtensorflow:tensorflowlib",
        "@gtest//:main",
    ],
)