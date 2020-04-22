
from distutils.core import setup, Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy.distutils.misc_util

from Cython.Compiler.Options import get_directive_defaults
directive_defaults              = get_directive_defaults()
directive_defaults['linetrace'] = True
directive_defaults['binding']   = True
include_dirs                    = numpy.distutils.misc_util.get_numpy_include_dirs()
scripts                         = ['methods','models','models_video','models_long_experiment','models_video_long','MF_models']
for i in range(0, len(scripts), 1):
    script_name  = scripts[i]
    extensions   = [Extension(script_name, [script_name +'.pyx'], define_macros=[('CYTHON_TRACE', '1')], include_dirs=include_dirs)]
    setup (name = script_name, version = "0.1", ext_modules = cythonize(extensions), cmdclass = {'build_ext': build_ext})
