from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='simple_roipool3d',
    ext_modules=[
        CUDAExtension('simple_roipool3d_cuda', [
            'src/simple_roipool3d.cpp',
            'src/simple_roipool3d_kernel.cu',
        ],
        extra_compile_args={'cxx': ['-g'],
                            'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension})
