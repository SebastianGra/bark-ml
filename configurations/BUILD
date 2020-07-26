py_library(
    name = "bark_agent",
    srcs = ["bark_agent.py"],
    data = ["@bark_project//bark:generate_core"],
    imports = ["../external/bark_project/python/"],
    deps = ["@bark_project//bark/runtime:runtime",
            "//configurations/highway:configuration"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "base_configuration",
    srcs = ["base_configuration.py"],
    data = ["@bark_project//bark:generate_core"],
    imports = ["../external/bark_project/python/"],
    deps = ["@bark_project//bark/runtime:runtime"],
    visibility = ["//visibility:public"],
)