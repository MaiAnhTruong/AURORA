from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="fused_ssim",
    packages=['fused_ssim'],
    ext_modules=[
        CUDAExtension(
            name="fused_ssim_cuda",
            sources=[
                "ssim.cu",
                "ext.cpp"
            ],
            extra_compile_args={
                "nvcc": ["-allow-unsupported-compiler"]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)