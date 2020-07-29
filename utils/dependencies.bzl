load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def _maybe(repo_rule, name, **kwargs):
    if name not in native.existing_rules():
        repo_rule(name = name, **kwargs)

def load_bark():
  _maybe(
    git_repository,
    name = "bark_project",
    commit="24f9937f6574f63fd955dd497497dfc68f47f2da",
    remote = "https://github.com/bark-simulator/bark",
  )
