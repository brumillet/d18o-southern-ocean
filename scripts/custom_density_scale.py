import numpy as np
import matplotlib.ticker as ticker
import matplotlib.transforms as mtransforms
import matplotlib.scale as mscale

# Here we create a custom neutral density scale which starts from 27 to 28.6 
# in a logarithmic visual aspec.

class CustomScaleTransform(mtransforms.Transform):
    input_dims = output_dims = 1

    def __init__(self):
        mtransforms.Transform.__init__(self)

    def transform_non_affine(self, a):
        return np.log1p(28.6 - a)  # Custom transformation: log1p scales higher densities more

    def inverted(self):
        return InvertedCustomScaleTransform()

class InvertedCustomScaleTransform(mtransforms.Transform):
    input_dims = output_dims = 1

    def transform_non_affine(self, a):
        return np.expm1(a) + 27  # Inverse of the log1p transformation

    def inverted(self):
        return CustomScaleTransform()

class CustomScale(mscale.ScaleBase):
    name = 'custom_scale'

    def get_transform(self):
        return CustomScaleTransform()

    def set_default_locators_and_formatters(self, axis, major=True):
        axis.set_major_locator(ticker.AutoLocator())
        axis.set_major_formatter(ticker.ScalarFormatter())

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return max(vmin, 27), min(vmax, 28.6)

# Register the custom scale
mscale.register_scale(CustomScale)