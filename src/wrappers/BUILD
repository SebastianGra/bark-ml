py_library(
    name = "wrappers",
    srcs = ["action_wrapper.py",
						"tfa_wrapper.py",
            "dynamic_model.py",
            "motion_primitives.py"],
    data = ["@bark_project//bark:generate_core",
            "//python:bark_ml.so",
            "//src/commons:py_spaces"],
    imports = ["../external/bark_project/python/",
               "../../python/"],
		deps = ["@bark_project//bark/runtime:runtime"],
		visibility = ["//visibility:public"],
)