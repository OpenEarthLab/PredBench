from collections.abc import Sequence

from mmengine.registry import build_from_cfg
from mmcv.transforms import BaseTransform

from predbench.registry import TRANSFORMS
from .augmentations import PytorchVideoTrans, TorchvisionTrans


@TRANSFORMS.register_module()
class Compose(BaseTransform):
    """Compose a data pipeline with a sequence of transforms.
    Args:
        transforms (list[dict | callable]):
            Either config dicts of transforms or transform objects.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                if transform['type'].startswith('torchvision.'):
                    trans_type = transform.pop('type')[12:]
                    transform = TorchvisionTrans(trans_type, **transform)
                elif transform['type'].startswith('pytorchvideo.'):
                    trans_type = transform.pop('type')[13:]
                    transform = PytorchVideoTrans(trans_type, **transform)
                else:
                    transform = build_from_cfg(transform, TRANSFORMS)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError(f'transform must be callable or a dict, '
                                f'but got {type(transform)}')

    def transform(self, data):
        """Call function to apply transforms sequentially.
        Args:
            data (dict): A result dict contains the data to transform.
        Returns:
            dict: Transformed data.
        """
        for t in self.transforms:
            data = t(data)
            # print(t)
            # print((data.keys()))
            # print(data)
            # break
            # if 'imgs' in data:
            #     print(type(data['imgs']))
            #     print(len(data['imgs']))
            #     print(data['imgs'][0].shape)
            #     print(data['imgs'][0].dtype)
                # raise NotImplementedError
            if data is None:
                return None
        # raise NotImplementedError
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string