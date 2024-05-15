import os
import sys
import subprocess
import argparse


def cmake_exe_path(build_tool_path):
    path = build_tool_path["cmake"][0]
    for p in build_tool_path["cmake"][1:]:
        path = os.path.join(path, p)
    path = os.path.join(path, "bin")
    exe_name = "cmake" + os.path.splitext(sys.executable)[-1]
    path = os.path.join(path, exe_name)
    return path


def cuda_toolkit_path(build_tool_path):
    path = build_tool_path["cuda_toolkit"][0]
    for p in build_tool_path["cuda_toolkit"][1:]:
        path = os.path.join(path, p)
    return '"' + path + '"'


def assemble_cmake_cmd(src_dir, build_tool_path, cmake_options):
    cmd = cmake_exe_path(build_tool_path=build_tool_path)
    cmd += " "
    cmd += src_dir
    for key, value in cmake_options.items():
        cmd += " " + "-D" + key + "=" + str(value)
    return cmd


def cmake(build_dir, build_tool_path, cmake_options):
    if not os.path.exists(build_dir):
        os.mkdir(build_dir)
    os.chdir(build_dir)
    src_dir = os.path.dirname(__file__)
    cmd = assemble_cmake_cmd(
        src_dir=src_dir, build_tool_path=build_tool_path, cmake_options=cmake_options
    )
    subprocess.run(cmd)
    os.chdir(src_dir)


def build(build_dir):
    assert os.path.exists(build_dir)
    os.chdir(build_dir)
    subprocess.run("cmake --build .")
    os.chdir(os.path.dirname(__file__))


build_tool_path = {
    "cmake": ["C:\\", "Program Files", "CMake"],
    "cuda_toolkit": [
        "C:\\",
        "Program Files",
        "NVIDIA GPU Computing Toolkit",
        "CUDA",
        "v12.4",
    ],
}

cmake_options = {
    "CMAKE_CUDA_ARCHITECTURES": 61,
    "CMAKE_C_STANDARD": 99,
    "CMAKE_CXX_STANDARD": 11,
    "CMAKE_BUILD_TYPE": "Debug",
    "CUDAToolkit_ROOT": cuda_toolkit_path(build_tool_path),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d", "--build-dir", default="build", help="cmake and build directory"
    )

    args = parser.parse_args()

    cmake(
        build_dir=args.build_dir,
        build_tool_path=build_tool_path,
        cmake_options=cmake_options,
    )
    build(build_dir=args.build_dir)
