py_library(
    name = "observers",
    srcs = ["nearest_state_observer.py",
            "simple_observer.py",
            "observer.py"],
    data = ["@bark_project//bark:generate_core"],
    imports = ["../external/bark_project/python/"],
    deps = ["@bark_project//bark/runtime:runtime"],
    visibility = ["//visibility:public"],
)

cc_library(
  name = "nearest_observer",
  hdrs = [
    "nearest_observer_new.hpp",
  ],
  deps = [
    "@bark_project//bark/world:world",
    "//src/commons:spaces",
    "//src/commons:commons"
  ],
  visibility = ["//visibility:public"],
)
