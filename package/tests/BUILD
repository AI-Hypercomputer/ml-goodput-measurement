load("//third_party/bazel_rules/rules_python/python:py_test.bzl", "py_test")

package(
    default_applicable_licenses = ["//third_party/py/cloud_tpu_goodput:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

py_test(
    name = "goodput_test",
    srcs = ["goodput_test.py"],
    deps = [
        "//testing/pybase",
        "//third_party/py/cloud_tpu_goodput/package/src:cloud_tpu_goodput",
    ],
)
