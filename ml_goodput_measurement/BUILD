# Python package build.
load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")
load("//third_party/py/etils:build_defs.bzl", "glob_py_srcs")

package(
    default_applicable_licenses = ["//third_party/py/cloud_tpu_goodput:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

pytype_strict_library(
    name = "cloud_tpu_goodput",
    # Returns all non-test python files recursively
    srcs = glob_py_srcs(),
    visibility = ["//visibility:public"],
    # Project dependencies (matching the `pip install` deps defined in `pyproject.toml`)
    deps = [
    ],
)
