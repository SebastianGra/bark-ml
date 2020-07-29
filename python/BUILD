package(default_visibility = ["//visibility:public"])

cc_binary(
  name = "bark_ml.so",
  srcs = glob(["*.cpp", "*.hpp"]),
  deps = [
    "//src/observers:nearest_observer",
    "//src/commons:spaces",
    # external
    "@python_linux//:python-lib",
    "@pybind11//:pybind11",
    "@com_github_eigen_eigen//:eigen",
    "@libtensorflow-RL-MCTS//libtensorflow:tensorflowlib",
  ],
  linkshared = 1,
  linkstatic = 1
)
