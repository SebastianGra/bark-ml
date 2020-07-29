load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def _maybe(repo_rule, name, **kwargs):
    if name not in native.existing_rules():
        repo_rule(name = name, **kwargs)

def load_bark():
  _maybe(
    git_repository,
    name = "bark_project",
    commit="2eb8646a0a991aece1e83af71930a58fccb72ae2",
    remote = "https://github.com/bark-simulator/bark",
  )
