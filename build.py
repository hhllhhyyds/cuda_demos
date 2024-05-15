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
    return path


def cmake(src_dir, build_dir, cmake_exe_path, cmake_options):
    if not os.path.exists(build_dir):
        os.mkdir(build_dir, mode=777)
    os.chdir(build_dir)
    cmd = [cmake_exe_path, src_dir]
    for key, value in cmake_options.items():
        cmd.append("-D" + key + "=" + str(value))
    process = subprocess.Popen(cmd)
    process.wait()


def build(build_dir, cmake_exe_path):
    assert os.path.exists(build_dir)
    os.chdir(build_dir)
    process = subprocess.Popen([cmake_exe_path, "--build", "."])
    process.wait()


def cmake_options(build_tool_path):
    return {
        "CMAKE_CUDA_ARCHITECTURES": 61,
        "CMAKE_C_STANDARD": 99,
        "CMAKE_CXX_STANDARD": 11,
        "CMAKE_BUILD_TYPE": "Debug",
        "CUDAToolkit_ROOT": cuda_toolkit_path(build_tool_path),
    }


build_tool_path_win = {
    "cmake": ["C:\\", "Program Files", "CMake"],
    "cuda_toolkit": [
        "C:\\",
        "Program Files",
        "NVIDIA GPU Computing Toolkit",
        "CUDA",
        "v12.4",
    ],
}

build_tool_path_linux = {
    "cmake": ["/udata", "liuxiaonan", "downloads", "cmake-3.29.3-linux-x86_64"],
    "cuda_toolkit": [
        "/udata",
        "pangen",
        "pangen_build_env2",
        "cuda-8.0",
    ],
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--build-dir", default="build", help="cmake and build directory"
    )
    args = parser.parse_args()

    if "win" in sys.platform.lower():
        build_tool_path = build_tool_path_win
    elif "linux" in sys.platform.lower():
        build_tool_path = build_tool_path_linux
    else:
        raise Exception("Unsupported platform = " + sys.platform)

    src_dir = os.getcwd()
    build_dir = os.path.abspath(args.build_dir)
    cmake_exe = cmake_exe_path(build_tool_path=build_tool_path)
    cmake_args = cmake_options(build_tool_path=build_tool_path)

    cmake(
        src_dir=src_dir,
        build_dir=build_dir,
        cmake_exe_path=cmake_exe,
        cmake_options=cmake_args,
    )
    build(build_dir=build_dir, cmake_exe_path=cmake_exe)
