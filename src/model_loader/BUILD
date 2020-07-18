

cc_library(
    name="ModelLoader",
    srcs=["ModelLoader.cpp"],
    hdrs=["ModelLoader.hpp"],
    copts = ["-Iexternal/libtensorflow-RL-MCTS"],
    deps= ["@libtensorflow-RL-MCTS//libtensorflow:tensorflowlib"],
    visibility = ["//visibility:public"],
)