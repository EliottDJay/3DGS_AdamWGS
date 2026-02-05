
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="diff_gauss",
    packages=['diff_gauss'],
    ext_modules=[
        CUDAExtension(
            name="diff_gauss._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "cuda_rasterizer/utils.cu",
            "cuda_rasterizer/adam.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args={
                "nvcc": [
                "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/"),
                "--disable-warnings"
                ]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
