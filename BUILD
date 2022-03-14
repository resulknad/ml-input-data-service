load("@hedron_compile_commands//:refresh_compile_commands.bzl", "refresh_compile_commands")

refresh_compile_commands(
    name = "refresh_compile_commands",

    # Specify the targets of interest.
    # For example, specify a dict of targets and their arguments:
    targets = {
    #"//tensorflow/core/data:dataset_utils":"",
    "//tensorflow/tools/pip_package:build_pip_package": "",
    },
    # For more details, feel free to look into refresh_compile_commands.bzl if you want.
)

exports_files([
    "configure",
    "configure.py",
    "ACKNOWLEDGEMENTS",
    "LICENSE",
])
