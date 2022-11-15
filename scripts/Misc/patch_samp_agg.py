import torch
import numpy as np
import warnings
from typing import Optional, Tuple, Generator, Union

import scipy
import torch
import numpy as np

import trackpy as tp
import pandas as pd
import torchio as tio
import monai
from tqdm import tqdm
from pathlib2 import Path

import colloidoscope

TypeTripletInt = Tuple[int, int, int]

TypeSpatialShape = Union[int, TypeTripletInt]

LOCATION = 'location'
CHANNELS_DIMENSION = 1

TypeBounds = Union[
    int,
    TypeTripletInt,
    TypeSixBounds,
    None,
]


class SpatialTransform(Transform):
    """Transform that modifies image bounds or voxels positions."""

    def get_images(self, subject: Subject) -> List[Image]:
        images = subject.get_images(
            intensity_only=False,
            include=self.include,
            exclude=self.exclude,
        )
        return images

class BoundsTransform(SpatialTransform):
    """Base class for transforms that change image bounds.
    Args:
        bounds_parameters: The meaning of this argument varies according to the
            child class.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.
    """
    def __init__(
            self,
            bounds_parameters: TypeBounds,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.bounds_parameters = self.parse_bounds(bounds_parameters)

    def is_invertible(self):
        return True

class Pad(BoundsTransform):
    r"""Pad an image.
    Args:
        padding: Tuple
            :math:`(w_{ini}, w_{fin}, h_{ini}, h_{fin}, d_{ini}, d_{fin})`
            defining the number of values padded to the edges of each axis.
            If the initial shape of the image is
            :math:`W \times H \times D`, the final shape will be
            :math:`(w_{ini} + W + w_{fin}) \times (h_{ini} + H + h_{fin})
            \times (d_{ini} + D + d_{fin})`.
            If only three values :math:`(w, h, d)` are provided, then
            :math:`w_{ini} = w_{fin} = w`,
            :math:`h_{ini} = h_{fin} = h` and
            :math:`d_{ini} = d_{fin} = d`.
            If only one value :math:`n` is provided, then
            :math:`w_{ini} = w_{fin} = h_{ini} = h_{fin} =
            d_{ini} = d_{fin} = n`.
        padding_mode: See possible modes in `NumPy docs`_. If it is a number,
            the mode will be set to ``'constant'``.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.
    .. seealso:: If you want to pass the output shape instead, please use
        :class:`~torchio.transforms.CropOrPad` instead.
    .. _NumPy docs: https://numpy.org/doc/stable/reference/generated/numpy.pad.html
    """  # noqa: E501

    PADDING_MODES = (
        'empty',
        'edge',
        'wrap',
        'constant',
        'linear_ramp',
        'maximum',
        'mean',
        'median',
        'minimum',
        'reflect',
        'symmetric',
    )

    def __init__(
            self,
            padding: TypeBounds,
            padding_mode: Union[str, float] = 0,
            **kwargs
    ):
        super().__init__(padding, **kwargs)
        self.padding = padding
        self.check_padding_mode(padding_mode)
        self.padding_mode = padding_mode
        self.args_names = ['padding', 'padding_mode']

    @classmethod
    def check_padding_mode(cls, padding_mode):
        is_number = isinstance(padding_mode, Number)
        is_callable = callable(padding_mode)
        if not (padding_mode in cls.PADDING_MODES or is_number or is_callable):
            message = (
                f'Padding mode "{padding_mode}" not valid. Valid options are'
                f' {list(cls.PADDING_MODES)}, a number or a function'
            )
            raise KeyError(message)

    def apply_transform(self, subject: Subject) -> Subject:
        assert self.bounds_parameters is not None
        low = self.bounds_parameters[::2]
        for image in self.get_images(subject):
            new_origin = nib.affines.apply_affine(image.affine, -np.array(low))
            new_affine = image.affine.copy()
            new_affine[:3, 3] = new_origin
            kwargs: Dict[str, Union[str, float]]
            if isinstance(self.padding_mode, Number):
                kwargs = {
                    'mode': 'constant',
                    'constant_values': self.padding_mode,
                }
            else:
                kwargs = {'mode': self.padding_mode}
            pad_params = self.bounds_parameters
            paddings = (0, 0), pad_params[:2], pad_params[2:4], pad_params[4:]
            padded = np.pad(image.data, paddings, **kwargs)  # type: ignore[call-overload]  # noqa: E501
            image.set_data(torch.as_tensor(padded))
            image.affine = new_affine
        return subject

    def inverse(self):
        from .crop import Crop
        return Crop(self.padding)

class Crop(BoundsTransform):
    r"""Crop an image.
    Args:
        cropping: Tuple
            :math:`(w_{ini}, w_{fin}, h_{ini}, h_{fin}, d_{ini}, d_{fin})`
            defining the number of values cropped from the edges of each axis.
            If the initial shape of the image is
            :math:`W \times H \times D`, the final shape will be
            :math:`(- w_{ini} + W - w_{fin}) \times (- h_{ini} + H - h_{fin})
            \times (- d_{ini} + D - d_{fin})`.
            If only three values :math:`(w, h, d)` are provided, then
            :math:`w_{ini} = w_{fin} = w`,
            :math:`h_{ini} = h_{fin} = h` and
            :math:`d_{ini} = d_{fin} = d`.
            If only one value :math:`n` is provided, then
            :math:`w_{ini} = w_{fin} = h_{ini} = h_{fin}
            = d_{ini} = d_{fin} = n`.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.
    .. seealso:: If you want to pass the output shape instead, please use
        :class:`~torchio.transforms.CropOrPad` instead.
    """
    def __init__(
            self,
            cropping: TypeBounds,
            **kwargs
    ):
        super().__init__(cropping, **kwargs)
        self.cropping = cropping
        self.args_names = ['cropping']

    def apply_transform(self, sample) -> Subject:
        assert self.bounds_parameters is not None
        low = self.bounds_parameters[::2]
        high = self.bounds_parameters[1::2]
        index_ini = low
        index_fin = np.array(sample.spatial_shape) - high
        for image in self.get_images(sample):
            new_origin = nib.affines.apply_affine(image.affine, index_ini)
            new_affine = image.affine.copy()
            new_affine[:3, 3] = new_origin
            i0, j0, k0 = index_ini
            i1, j1, k1 = index_fin
            image.set_data(image.data[:, i0:i1, j0:j1, k0:k1].clone())
            image.affine = new_affine
        return sample

    def inverse(self):
        from .pad import Pad
        return Pad(self.cropping)

class PatchSampler:
    r"""Base class for TorchIO samplers.
    Args:
        patch_size: Tuple of integers :math:`(w, h, d)` to generate patches
            of size :math:`w \times h \times d`.
            If a single number :math:`n` is provided, :math:`w = h = d = n`.
    .. warning:: This is an abstract class that should only be instantiated
        using child classes such as :class:`~torchio.data.UniformSampler` and
        :class:`~torchio.data.WeightedSampler`.
    """
    def __init__(self, patch_size: TypeSpatialShape):
        patch_size_array = np.array(to_tuple(patch_size, length=3))
        for n in patch_size_array:
            if n < 1 or not isinstance(n, (int, np.integer)):
                message = (
                    'Patch dimensions must be positive integers,'
                    f' not {patch_size_array}'
                )
                raise ValueError(message)
        self.patch_size = patch_size_array.astype(np.uint16)

    def extract_patch(
            self,
            subject: Subject,
            index_ini: TypeTripletInt,
    ) -> Subject:
        cropped_subject = self.crop(subject, index_ini, self.patch_size)  # type: ignore[arg-type]  # noqa: E501
        return cropped_subject

    def crop(
            self,
            subject: Subject,
            index_ini: TypeTripletInt,
            patch_size: TypeTripletInt,
    ) -> Subject:
        transform = self._get_crop_transform(subject, index_ini, patch_size)
        cropped_subject = transform(subject)
        index_ini_array = np.asarray(index_ini)
        patch_size_array = np.asarray(patch_size)
        index_fin = index_ini_array + patch_size_array
        location = index_ini_array.tolist() + index_fin.tolist()
        cropped_subject[LOCATION] = torch.as_tensor(location)
        cropped_subject.update_attributes()
        return cropped_subject

    @staticmethod
    def _get_crop_transform(
            subject,
            index_ini: TypeTripletInt,
            patch_size: TypeSpatialShape,
    ):
        from ...transforms.preprocessing.spatial.crop import Crop
        shape = np.array(subject.spatial_shape, dtype=np.uint16)
        index_ini_array = np.array(index_ini, dtype=np.uint16)
        patch_size_array = np.array(patch_size, dtype=np.uint16)
        assert len(index_ini_array) == 3
        assert len(patch_size_array) == 3
        index_fin = index_ini_array + patch_size_array
        crop_ini = index_ini_array.tolist()
        crop_fin = (shape - index_fin).tolist()
        start = ()
        cropping = sum(zip(crop_ini, crop_fin), start)
        return Crop(cropping)  # type: ignore[arg-type]

    def __call__(
            self,
            subject: Subject,
            num_patches: Optional[int] = None,
    ) -> Generator[Subject, None, None]:
        subject.check_consistent_space()
        if np.any(self.patch_size > subject.spatial_shape):
            message = (
                f'Patch size {tuple(self.patch_size)} cannot be'
                f' larger than image size {tuple(subject.spatial_shape)}'
            )
            raise RuntimeError(message)
        kwargs = {} if num_patches is None else {'num_patches': num_patches}
        return self._generate_patches(subject, **kwargs)

    def _generate_patches(
            self,
            subject: Subject,
            num_patches: Optional[int] = None,
    ) -> Generator[Subject, None, None]:
        raise 

class GridAggregator:

    r"""Aggregate patches for dense inference.

    This class is typically used to build a volume made of patches after
    inference of batches extracted by a :class:`~torchio.data.GridSampler`.

    Args:
        sampler: Instance of :class:`~torchio.data.GridSampler` used to
            extract the patches.
        overlap_mode: If ``'crop'``, the overlapping predictions will be
            cropped. If ``'average'``, the predictions in the overlapping areas
            will be averaged with equal weights. If ``'hann'``, the predictions
            in the overlapping areas will be weighted with a Hann window
            function. See the `grid aggregator tests`_ for a raw visualization
            of the three modes.

    .. _grid aggregator tests: https://github.com/fepegar/torchio/blob/main/tests/data/inference/test_aggregator.py

    .. note:: Adapted from NiftyNet. See `this NiftyNet tutorial
        <https://niftynet.readthedocs.io/en/dev/window_sizes.html>`_ for more
        information about patch-based sampling.
    """  # noqa: E501
    def __init__(self, sampler: GridSampler, overlap_mode: str = 'crop'):
        subject = sampler.subject
        self.volume_padded = sampler.padding_mode is not None
        self.spatial_shape = subject.spatial_shape
        self._output_tensor: Optional[torch.Tensor] = None
        self.patch_overlap = sampler.patch_overlap
        self.patch_size = sampler.patch_size
        self._parse_overlap_mode(overlap_mode)
        self.overlap_mode = overlap_mode
        self._avgmask_tensor: Optional[torch.Tensor] = None
        self._hann_window: Optional[torch.Tensor] = None

    @staticmethod
    def _parse_overlap_mode(overlap_mode):
        if overlap_mode not in ('crop', 'average', 'hann'):
            message = (
                'Overlap mode must be "crop", "average" or "hann" but '
                f' "{overlap_mode}" was passed'
            )
            raise ValueError(message)

    def _crop_patch(
            self,
            patch: torch.Tensor,
            location: np.ndarray,
            overlap: np.ndarray,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        half_overlap = overlap // 2  # overlap is always even in grid sampler
        index_ini, index_fin = location[:3], location[3:]

        # If the patch is not at the border, we crop half the overlap
        crop_ini = half_overlap.copy()
        crop_fin = half_overlap.copy()

        # If the volume has been padded, we don't need to worry about cropping
        if self.volume_padded:
            pass
        else:
            crop_ini *= index_ini > 0
            crop_fin *= index_fin != self.spatial_shape

        # Update the location of the patch in the volume
        new_index_ini = index_ini + crop_ini
        new_index_fin = index_fin - crop_fin
        new_location = np.hstack((new_index_ini, new_index_fin))

        patch_size = patch.shape[-3:]
        i_ini, j_ini, k_ini = crop_ini
        i_fin, j_fin, k_fin = patch_size - crop_fin
        cropped_patch = patch[:, i_ini:i_fin, j_ini:j_fin, k_ini:k_fin]
        return cropped_patch, new_location

    def _initialize_output_tensor(self, batch: torch.Tensor) -> None:
        if self._output_tensor is not None:
            return
        num_channels = batch.shape[CHANNELS_DIMENSION]
        self._output_tensor = torch.zeros(
            num_channels,
            *self.spatial_shape,
            dtype=batch.dtype,
        )

    def _initialize_avgmask_tensor(self, batch: torch.Tensor) -> None:
        if self._avgmask_tensor is not None:
            return
        num_channels = batch.shape[CHANNELS_DIMENSION]
        self._avgmask_tensor = torch.zeros(
            num_channels,
            *self.spatial_shape,
            dtype=batch.dtype,
        )

    @staticmethod
    def _get_hann_window(patch_size):
        hann_window_3d = torch.as_tensor([1])
        # create a n-dim hann window
        for spatial_dim, size in enumerate(patch_size):
            window_shape = np.ones_like(patch_size)
            window_shape[spatial_dim] = size
            hann_window_1d = torch.hann_window(
                size + 2,
                periodic=False,
            )
            hann_window_1d = hann_window_1d[1:-1].view(*window_shape)
            hann_window_3d = hann_window_3d * hann_window_1d
        return hann_window_3d

    def _initialize_hann_window(self) -> None:
        if self._hann_window is not None:
            return
        self._hann_window = self._get_hann_window(self.patch_size)

    def add_batch(
            self,
            batch_tensor: torch.Tensor,
            locations: torch.Tensor,
    ) -> None:
        """Add batch processed by a CNN to the output prediction volume.

        Args:
            batch_tensor: 5D tensor, typically the output of a convolutional
                neural network, e.g. ``batch['image'][torchio.DATA]``.
            locations: 2D tensor with shape :math:`(B, 6)` representing the
                patch indices in the original image. They are typically
                extracted using ``batch[torchio.LOCATION]``.
        """
        batch = batch_tensor.cpu()
        locations = locations.cpu().numpy()
        patch_sizes = locations[:, 3:] - locations[:, :3]
        # There should be only one patch size
        assert len(np.unique(patch_sizes, axis=0)) == 1
        input_spatial_shape = tuple(batch.shape[-3:])
        target_spatial_shape = tuple(patch_sizes[0])
        if input_spatial_shape != target_spatial_shape:
            message = (
                f'The shape of the input batch, {input_spatial_shape},'
                ' does not match the shape of the target location,'
                f' which is {target_spatial_shape}'
            )
            raise RuntimeError(message)
        self._initialize_output_tensor(batch)
        assert isinstance(self._output_tensor, torch.Tensor)
        if self.overlap_mode == 'crop':
            for patch, location in zip(batch, locations):
                cropped_patch, new_location = self._crop_patch(
                    patch,
                    location,
                    self.patch_overlap,
                )
                i_ini, j_ini, k_ini, i_fin, j_fin, k_fin = new_location
                self._output_tensor[
                    :,
                    i_ini:i_fin,
                    j_ini:j_fin,
                    k_ini:k_fin,
                ] = cropped_patch
        elif self.overlap_mode == 'average':
            self._initialize_avgmask_tensor(batch)
            assert isinstance(self._avgmask_tensor, torch.Tensor)
            for patch, location in zip(batch, locations):
                i_ini, j_ini, k_ini, i_fin, j_fin, k_fin = location
                self._output_tensor[
                    :,
                    i_ini:i_fin,
                    j_ini:j_fin,
                    k_ini:k_fin,
                ] += patch
                self._avgmask_tensor[
                    :,
                    i_ini:i_fin,
                    j_ini:j_fin,
                    k_ini:k_fin,
                ] += 1
        elif self.overlap_mode == 'hann':
            # To handle edge and corners avoid numerical problems, we save the
            # hann window in a different tensor
            # At the end, it will be filled with ones (or close values) where
            # there is overlap and < 1 where there is not
            # When we divide, the multiplication will be canceled in areas that
            # do not overlap
            self._initialize_avgmask_tensor(batch)
            self._initialize_hann_window()

            if self._output_tensor.dtype != torch.float32:
                self._output_tensor = self._output_tensor.float()

            assert isinstance(self._avgmask_tensor, torch.Tensor)  # for mypy
            if self._avgmask_tensor.dtype != torch.float32:
                self._avgmask_tensor = self._avgmask_tensor.float()

            for patch, location in zip(batch, locations):
                i_ini, j_ini, k_ini, i_fin, j_fin, k_fin = location

                patch = patch * self._hann_window
                self._output_tensor[
                    :,
                    i_ini:i_fin,
                    j_ini:j_fin,
                    k_ini:k_fin,
                ] += patch
                self._avgmask_tensor[
                    :,
                    i_ini:i_fin,
                    j_ini:j_fin,
                    k_ini:k_fin,
                ] += self._hann_window


    def get_output_tensor(self) -> torch.Tensor:
        """Get the aggregated volume after dense inference."""
        assert isinstance(self._output_tensor, torch.Tensor)
        if self._output_tensor.dtype == torch.int64:
            message = (
                'Medical image frameworks such as ITK do not support int64.'
                ' Casting to int32...'
            )
            warnings.warn(message, RuntimeWarning)
            self._output_tensor = self._output_tensor.type(torch.int32)
        if self.overlap_mode in ['average', 'hann']:
            assert isinstance(self._avgmask_tensor, torch.Tensor)  # for mypy
            # true_divide is used instead of / in case the PyTorch version is
            # old and one the operands is int:
            # https://github.com/fepegar/torchio/issues/526
            output = torch.true_divide(
                self._output_tensor, self._avgmask_tensor,
            )
        else:
            output = self._output_tensor
        if self.volume_padded:
            from ...transforms import Crop
            border = self.patch_overlap // 2
            cropping = border.repeat(2)
            crop = Crop(cropping)  # type: ignore[arg-type]
            return crop(output)  # type: ignore[return-value]
        else:
            return output



class GridSampler(PatchSampler):
    r"""Extract patches across a whole volume.

    Grid samplers are useful to perform inference using all patches from a
    volume. It is often used with a :class:`~torchio.data.GridAggregator`.

    Args:
        subject: Instance of :class:`~torchio.data.Subject`
            from which patches will be extracted. This argument should only be
            used before instantiating a :class:`~torchio.data.GridAggregator`,
            or to precompute the number of patches that would be generated from
            a subject.
        patch_size: Tuple of integers :math:`(w, h, d)` to generate patches
            of size :math:`w \times h \times d`.
            If a single number :math:`n` is provided,
            :math:`w = h = d = n`.
            This argument is mandatory (it is a keyword argument for backward
            compatibility).
        patch_overlap: Tuple of even integers :math:`(w_o, h_o, d_o)`
            specifying the overlap between patches for dense inference. If a
            single number :math:`n` is provided, :math:`w_o = h_o = d_o = n`.
        padding_mode: Same as :attr:`padding_mode` in
            :class:`~torchio.transforms.Pad`. If ``None``, the volume will not
            be padded before sampling and patches at the border will not be
            cropped by the aggregator.
            Otherwise, the volume will be padded with
            :math:`\left(\frac{w_o}{2}, \frac{h_o}{2}, \frac{d_o}{2} \right)`
            on each side before sampling. If the sampler is passed to a
            :class:`~torchio.data.GridAggregator`, it will crop the output
            to its original size.

    Example:

        >>> import torchio as tio
        >>> sampler = tio.GridSampler(patch_size=88)
        >>> colin = tio.datasets.Colin27()
        >>> for i, patch in enumerate(sampler(colin)):
        ...     patch.t1.save(f'patch_{i}.nii.gz')
        ...
        >>> # To figure out the number of patches beforehand:
        >>> sampler = tio.GridSampler(subject=colin, patch_size=88)
        >>> len(sampler)
        8

    .. note:: Adapted from NiftyNet. See `this NiftyNet tutorial
        <https://niftynet.readthedocs.io/en/dev/window_sizes.html>`_ for more
        information about patch based sampling. Note that
        :attr:`patch_overlap` is twice :attr:`border` in NiftyNet
        tutorial.
    """
    def __init__(
            self,
            subject: Optional[Subject] = None,
            patch_size: TypeSpatialShape = None,
            patch_overlap: TypeSpatialShape = (0, 0, 0),
            padding_mode: Union[str, float, None] = None,
    ):
        if patch_size is None:
            raise ValueError('A value for patch_size must be given')
        super().__init__(patch_size)
        self.patch_overlap = np.array(to_tuple(patch_overlap, length=3))
        self.padding_mode = padding_mode
        if subject is not None and not isinstance(subject, Subject):
            raise ValueError('The subject argument must be None or Subject')
        assert subject is not None
        self.subject = self._pad(subject)
        self.locations = self._compute_locations(self.subject)

    def __len__(self):
        return len(self.locations)

    def __getitem__(self, index):
        # Assume 3D
        location = self.locations[index]
        index_ini = location[:3]
        cropped_subject = self.crop(self.subject, index_ini, self.patch_size)
        return cropped_subject

    def _pad(self, subject: Subject) -> Subject:
        if self.padding_mode is not None:
            from ...transforms import Pad
            border = self.patch_overlap // 2
            padding = border.repeat(2)
            pad = Pad(padding, padding_mode=self.padding_mode)  # type: ignore[arg-type]  # noqa: E501
            subject = pad(subject)  # type: ignore[assignment]
        return subject

    def _compute_locations(self, subject: Subject):
        if subject is None:
            return None
        sizes = subject.spatial_shape, self.patch_size, self.patch_overlap
        self._parse_sizes(*sizes)  # type: ignore[arg-type]
        return self._get_patches_locations(*sizes)  # type: ignore[arg-type]

    def _generate_patches(  # type: ignore[override]
            self,
            subject: Subject,
    ) -> Generator[Subject, None, None]:
        subject = self._pad(subject)
        sizes = subject.spatial_shape, self.patch_size, self.patch_overlap
        self._parse_sizes(*sizes)  # type: ignore[arg-type]
        locations = self._get_patches_locations(*sizes)  # type: ignore[arg-type]  # noqa: E501
        for location in locations:
            index_ini = location[:3]
            yield self.extract_patch(subject, index_ini)

    @staticmethod
    def _parse_sizes(
            image_size: TypeTripletInt,
            patch_size: TypeTripletInt,
            patch_overlap: TypeTripletInt,
    ) -> None:
        image_size_array = np.array(image_size)
        patch_size_array = np.array(patch_size)
        patch_overlap_array = np.array(patch_overlap)
        if np.any(patch_size_array > image_size_array):
            message = (
                f'Patch size {tuple(patch_size_array)} cannot be'
                f' larger than image size {tuple(image_size_array)}'
            )
            raise ValueError(message)
        if np.any(patch_overlap_array >= patch_size_array):
            message = (
                f'Patch overlap {tuple(patch_overlap_array)} must be smaller'
                f' than patch size {tuple(patch_size_array)}'
            )
            raise ValueError(message)
        if np.any(patch_overlap_array % 2):
            message = (
                'Patch overlap must be a tuple of even integers,'
                f' not {tuple(patch_overlap_array)}'
            )
            raise ValueError(message)

    @staticmethod
    def _get_patches_locations(
            image_size: TypeTripletInt,
            patch_size: TypeTripletInt,
            patch_overlap: TypeTripletInt,
    ) -> np.ndarray:
        # Example with image_size 10, patch_size 5, overlap 2:
        # [0 1 2 3 4 5 6 7 8 9]
        # [0 0 0 0 0]
        #       [1 1 1 1 1]
        #           [2 2 2 2 2]
        # Locations:
        # [[0, 5],
        #  [3, 8],
        #  [5, 10]]
        indices = []
        zipped = zip(image_size, patch_size, patch_overlap)
        for im_size_dim, patch_size_dim, patch_overlap_dim in zipped:
            end = im_size_dim + 1 - patch_size_dim
            step = patch_size_dim - patch_overlap_dim
            indices_dim = list(range(0, end, step))
            if indices_dim[-1] != im_size_dim - patch_size_dim:
                indices_dim.append(im_size_dim - patch_size_dim)
            indices.append(indices_dim)
        indices_ini = np.array(np.meshgrid(*indices)).reshape(3, -1).T
        indices_ini = np.unique(indices_ini, axis=0)
        indices_fin = indices_ini + np.array(patch_size)
        locations = np.hstack((indices_ini, indices_fin))
        return np.array(sorted(locations.tolist()))


def detect(array:np.ndarray, diameter:Union[int, list]=1, model:torch.nn.Module=None, weights_path:Union[str, Path] = None, 
			patch_overlap:tuple=(16, 16, 16), roiSize:tuple=(64,64,64), post_processing:str="tp", threshold:float=0.5, 
			debug:bool=False, device=None, batch_size=4) -> pd.DataFrame:
	"""Detect 3d spheres from confocal microscopy

	Args:
		array (np.ndarray): Image for particles to be detected from.
		diameter (Union[int, list], optional): Diameter of particles to feed to TrackPy, can be int or list the same length as image dimensions. Defaults to 5. If post_processing == str(max) this has to be int it will be min_distance.
		model (torch.nn.Module, optional): Pytorch model. Defaults to None.
		weights_path (Union[str, Path], optional): Path to model weights file. Defaults to None.
		patch_overlap (tuple, optional): Overlap for patch based inference, overlap must be diff between input and output shape (if they are not the same). Defaults to (16, 16, 16).
		roiSize (tuple, optional): Size of ROI for model. Defaults to (64,64,64).
		debug (bool, optional): Option to return model output and positions in format for testing. Defaults to False.

	Returns:
		pd.DataFrame: TrackPy positions dataframe
	"""	

	# TODO write asserts

	if post_processing not in ["tp", "max", "log", "classic"]:
		raise ValueError(f"post_processing can be str(tp), or str(max) but you provided {post_processing}")
	
	# initialise torch device
	if device is None: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f'predicting on {device}')

	# model
	if model is None:
		model = monai.networks.nets.AttentionUnet(
			spatial_dims=3,
			in_channels=1,
			out_channels=1,
			channels=[32, 64, 128],
			strides=[2,2],
			# act=params['activation'],
			# norm=params["norm"],
			padding='valid',
		)

	if device == "cuda": model = torch.nn.DataParallel(model, device_ids=None) # parallelise model
	elif device == "cpu": 
		model = torch.nn.DataParallel(model, device_ids=None) # parallelise model

	if weights_path is not None:
		model_weights = torch.load(weights_path, map_location=device) # read trained weights
		model.load_state_dict(model_weights) # add weights to model
	
	if device == "cuda": model = model.to(device)
	elif device == "cpu": model = model.module.to(device)
    
	
	array = np.array(array/array.max(), dtype=np.float32) # normalise input
	array = np.expand_dims(array, 0) # add batch axis
	# tensor = torch.from_numpy(array)
	# tensor = tensor.unsqueeze(1)

	# print(tensor.shape, tensor.max(), tensor.min())
	# print(path)

	# TODO NORMALISE BRIGHTNESS HISTOGRAM BEFORE PREDICITON
	subject_dict = {'scan' : tio.ScalarImage(tensor=array, type=tio.INTENSITY, path=None),}
	subject = tio.Subject(subject_dict) # use torchio subject to enable using grid sampling
	grid_sampler = tio.inference.GridSampler(subject, patch_size=roiSize, patch_overlap=patch_overlap, padding_mode='mean')
	patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=batch_size)
	aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode='crop') # average for bc, crop for normal
	
	# patch_iter = PatchIter(patch_size=roiSize, start_pos=(0, 0, 0))
	# ds = GridPatchDataset(data=[array],
    #                           patch_iter=patch_iter,
    #                           transform=None)
	# patch_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size)
	# output_tensor = np.zeros_like(array)

	model.eval()
	with torch.no_grad():
		for i, patch_batch in tqdm(enumerate(patch_loader)):
			# input_tensor = patch_batch['scan'][tio.DATA]
			# locations = patch_batch[tio.LOCATION]
			input_tensor, locations = patch_batch[0], patch_batch[1]
			print(input_tensor.shape, locations.shape, locations)
			input_tensor.to(device)
			out = model(input_tensor)  # send through model/network
			out_sigmoid = torch.sigmoid(out)  # perform sigmoid on output because logits
			# print(out_sigmoid.shape, input_tensor.shape)
			
			# blank = torch.zeros_like(input_tensor) # because tio doesnt accept outputs of different sizes
			# out_sigmoid = insert_in_center(blank, out_sigmoid)
			# out_sigmoid = out_sigmoid.cpu().numpy()  # send to cpu and transform to numpy.ndarray
			# for pred, loc in zip(out_sigmoid, locations):
			# 	output_tensor[	loc[0,0]:loc[0,1], 
			# 					loc[1,0]:loc[1,1], 
			# 					loc[2,0]:loc[2,1], 
			# 					loc[3,0]:loc[3,1]] = pred

			aggregator.add_batch(out_sigmoid, locations)

	# post process to numpy array
	output_tensor = aggregator.get_output_tensor()
	result = output_tensor.cpu().numpy()  # send to cpu and transform to numpy.ndarray
	result = np.squeeze(output_tensor)  # remove batch dim and channel dim -> [H, W]

	# find positions from label


	if post_processing == "tp":
		positions = run_trackpy(result*255, diameter=diameter)
	# elif post_processing == "max":
	# 	if isinstance(diameter, list): diameter = diameter[0]
	# 	positions = peak_local_max(result*255, min_distance=int((diameter/2)))
	# elif post_processing == "log":
	# 	result[result<threshold]=0
	# 	if isinstance(diameter, list): 
	# 		sigma = int((diameter[0]/2)/math.sqrt(3))
	# 		positions = blob_log(result, min_sigma=sigma, max_sigma=sigma, overlap=0.1)[:,:-1]
	# 	else:
	# 		sigma = int((diameter/2)/math.sqrt(3))
	# 		positions = blob_log(result, min_sigma=sigma, max_sigma=sigma, overlap=0.1)[:,:-1]
	# elif post_processing == "classic":
	# 	positions = find_positions(result, threshold)
	

	d = {
		'x' : positions[:,1],
		'y' : positions[:,2],
		'z' : positions[:,0],
		}
	df = pd.DataFrame().from_dict(d) #, orient='index')

	if debug:
		return df, positions, result
	else:
		return df

def run_trackpy(array, diameter=5, *args, **kwargs):
	df = None
	df = tp.locate(array, diameter=diameter, *args, **kwargs)
	f = list(zip(df['z'], df['y'], df['x']))
	tp_predictions = np.array(f, dtype='float32')

	return tp_predictions


if __name__ == "__main__":
    dc = colloidoscope.DeepColloid()
    array = dc.read_tif('examples/Data/emily.tiff')
    array = dc.crop3d(array, roiSize=(128,128,128))
    print(array.shape)
    df, positions, label = dc.detect(array, diameter=13, weights_path='output/weights/attention_unet_202206.pt', debug=True, device="cpu")
    print(df)

    x,y = dc.get_gr(positions, 50,50)
    plt.plot(x,y,)
    plt.savefig("output/test/gr.png")