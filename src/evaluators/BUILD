py_library(
    name = "evaluators",
    srcs = ["goal_reached.py",
            "evaluator.py"],
    data = ["@bark_project//bark:generate_core"],
    imports = ["../external/bark_project/python/"],
    deps = ["@bark_project//bark/runtime:runtime"],
    visibility = ["//visibility:public"],
)