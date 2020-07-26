py_library(
    name = "rl_runtime",
    srcs = ["rl_runtime.py"],
    data = ["@bark_project//bark:generate_core"],
    imports = ["../external/bark_project/python/"],
    deps = ["@bark_project//bark/runtime:runtime",
            "//src/wrappers:wrappers",
            "//src/evaluators:evaluators",
            "//src/observers:observers"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "test_env",
    srcs = ["test_env.py"],
    data = ["@bark_project//bark:generate_core"],
    imports = ["../external/bark_project/python/"],
    deps = ["@bark_project//bark/runtime:runtime",
            "//src/wrappers:wrappers",
            "//src/evaluators:evaluators",
            "//src/observers:observers"],
    visibility = ["//visibility:public"],
)