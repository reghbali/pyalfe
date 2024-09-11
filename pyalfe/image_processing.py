import shutil
from abc import ABC, abstractmethod
import re
from collections import defaultdict

import nibabel as nib
import scipy.ndimage
import nilearn.image
import nilearn.masking
import nilearn.regions
import numpy as np

from pyalfe.interfaces.c3d import C3D


class ImageProcessor(ABC):
    """
    Abstract class for ImageProcessors.
    """

    @staticmethod
    @abstractmethod
    def threshold(
        image, output, lower_bound, upper_bound, inside_target, outside_target
    ):
        """Maps any pixel with value inside
        ``[lower_bound , upper_bound]`` to
        ``inside_target`` and any other pixel is mapped to ``outsize_target``.
        The thresholded image is written to `output` path.

        Parameters
        ----------
        image: str or Path
            Path to the input image.
        output: str or Path
            Path to the output image.
        lower_bound: int or float
            The lower_bound for the threshold.
        upper_bound: int or float
            The upper_bound for the threshold.
        inside_target: int or float
            The target value for pixels that are inside
             ``[lower_bound , upper_bound]``.
        outside_target: int or float
            The target value for the pixels that are not inside
             ``[lower_bound , upper_bound]``.

        Returns
        -------

        """
        pass

    @staticmethod
    def binarize(image, output):
        """Converts an image to binary by mapping all non-zero values to 1.

        Parameters
        ----------
        image: str or Path
            Path to the input image.
        output: str or Path
            Path to the output image.

        Returns
        -------

        """
        pass

    @staticmethod
    @abstractmethod
    def mask(image, mask, output):
        """Masks the input image by setting the pixels outside the mask to zero.

        Parameters
        ----------
        image: str or Path
            Path to the input image.
        mask: str or Path
            Path to the mask.
        output: str or Path
            Path to the output image.

        Returns
        -------

        """
        pass

    @staticmethod
    @abstractmethod
    def largest_mask_comp(image, output):
        """Finds the largest connected component of an input mask image.

        Parameters
        ----------
        image: str or Path
            Path to the input image.
        output: str or Path
            Path to the output image.

        Returns
        -------

        """
        pass

    @staticmethod
    @abstractmethod
    def holefill(binary_image, output):
        """Fills the holes in a binary image.

        Parameters
        ----------
        binary_image: str or Path
            Path to the input binary image.
        output: str or Path
            Path to the output image.

        Returns
        -------

        """
        pass

    @staticmethod
    @abstractmethod
    def reslice_to_ref(ref_image, moving_image, output):
        """Reslice the moving_image to the ref_image space.

        Parameters
        ----------
        ref_image: str or Path
            Path to the reference image.
        moving_image: str or Path
            Path to the moving image.
        output: str or Path
            Path to the output image.

        Returns
        -------

        """
        pass

    @staticmethod
    @abstractmethod
    def resample_new_dim(image, output, dim1, dim2, dim3, percent=True):
        """Resample the image to new dimensions keeping the bounding box
         the same, but changing the number of voxels in the image.

        Parameters
        ----------
        image: str or Path
            Path to the reference image.
        output: str or Path
            Path to the output image.
        dim1: int
            By default the percentage of the original number of voxels
            along the first dimension. For example, to double
            the number of voxel along the first dimension, set
            ``dim1 = 200``. If ``percent`` is False, then ``dim1``
            is interpreted as the number of voxels in the output image.
        dim2: int
            Similar to ``dim1`` but for the second axis.
        dim3: int
            Similar to ``dim1`` but for the third axis.
        percent: bool
            If True, the ``dim1``, ``dim2``, and ``dim3`` will be
            interpreted as percentages of the original dimensions
            instead of raw dimensions.

        Returns
        -------

        """
        pass

    @staticmethod
    @abstractmethod
    def get_dims(image):
        """

        Parameters
        ----------
        image: str or Path
            The path to the image.

        Returns
        -------
        list
            Returns a list containing the dimensions of the image.

        """
        pass

    @staticmethod
    @abstractmethod
    def trim_largest_comp(image, output, trim_margin_vec):
        """Finds the largest component of the image and trims the margins.
        This function is usefull for trimming neck.

        Parameters
        ----------
        image: str or Path
            The path to the image.
        output: str or Path
            Path to the output image.
        trim_margin_vec: tuple or list
            The trim margins along the three dimensions in voxels. For example,
            (5, 5, 5) means 5 voxels of background are kept in each side of
            each dimension.

        Returns
        -------

        """
        pass

    @staticmethod
    @abstractmethod
    def set_subtract(binary_image_1, binary_image_2, output):
        """Performs set subtraction between of second binary image
        from the first binary image.

        Parameters
        ----------
        binary_image_1: str or Path
            The path to the first image.
        binary_image_2: str or Path
            The path to the second image.
        output: str or Path
            Path to the output image.

        Returns
        -------

        """
        pass

    @staticmethod
    @abstractmethod
    def dilate(binary_image, rad, output):
        """Dilates binary image by a certain radius given in voxels.

        Parameters
        ----------
        binary_image: str or Path
            The path to the binary image.
        rad: int
            The dilation radius in voxels.
        output

        Returns
        -------

        """
        pass

    @staticmethod
    @abstractmethod
    def union(binary_image_1, binary_image_2, output):
        """Takes the union of two binary images.

        Parameters
        ----------
        binary_image_1: str or Path
            The path to the first image.
        binary_image_2: str or Path
            The path to the second image.
        output: str or Path
            Path to the output image.

        Returns
        -------

        """
        pass

    @staticmethod
    @abstractmethod
    def distance_transform(binary_image, output):
        """Computes the distance transform of a binary image.

        Parameters
        ----------
        binary_image: str or Path
            The path to the binary image.
        output: str or Path
            Path to the output image.

        Returns
        -------

        """
        pass

    @staticmethod
    @abstractmethod
    def label_mask_comp(binary_image, output):
        """Assigns discrete labels to the connected component of a 1 region in
        a binary image. The largest region is assigned label 1,
        the second largest is assigned label 2 and so forth.

        Parameters
        ----------
        binary_image: str or Path
            The path to the binary image.
        output: str or Path
            Path to the output image.

        Returns
        -------

        """
        pass

    @staticmethod
    @abstractmethod
    def remap_labels(multi_label_image, label_map, output):
        """Maps all the labels if a multi label image according to a label map
        the `label_map` is a dictionary that maps input labels to output label.
        For example, {1: 1, 2: 1, 3: 2} maps labels 1 and 2 in the input to
        label 1 in the output and label 3 in the input to label 2 in the output.
        Any other label is mapped to 0.

        Parameters
        ----------
        multi_label_image: str or Path
            The path to the multi label image
        label_map: dict
            The label map
        output: str or Path
            Path to the output image.

        Returns
        -------

        """


class Convert3DProcessor(ImageProcessor):
    @staticmethod
    def threshold(
        image, output, lower_bound, upper_bound, inside_target, outside_target
    ):
        c3d = C3D()
        (
            c3d.operand(image)
            .thresh(lower_bound, upper_bound, inside_target, outside_target)
            .out(output)
            .run()
        )

    @staticmethod
    def binarize(image, output):
        c3d = C3D()
        c3d.operand(image).binarize().out(output).run()

    @staticmethod
    def mask(image, mask, output):
        c3d = C3D()
        c3d.operand(image, mask).multiply().out(output).run()

    @staticmethod
    def largest_mask_comp(image, output):
        c3d = C3D()
        (
            c3d.operand(image)
            .popas('S')
            .push('S')
            .thresh(1, 1, 1, 0)
            .comp()
            .popas('C')
            .push('C')
            .thresh(1, 1, 1, 0)
            .push('S')
            .multiply()
            .out(output)
        ).run()

    @staticmethod
    def holefill(binary_image, output):
        c3d = C3D()
        c3d.operand(binary_image).holefill(1, 0).out(output).run()

    @staticmethod
    def reslice_to_ref(ref_image, moving_image, output):
        c3d = C3D()
        c3d.operand(ref_image, moving_image).reslice_identity().out(output).run()

    @staticmethod
    def resample_new_dim(image, output, dim1, dim2, dim3, percent=True):
        c3d = C3D()
        reasmple_arg = f'{str(dim1)}x{str(dim2)}x{str(dim3)}'
        if percent:
            reasmple_arg = reasmple_arg + '%'
        c3d.operand(image).resample().operand(reasmple_arg).out(output).run()

    @staticmethod
    def get_dims(image):
        c3d = C3D()
        output = c3d.operand(image).info().check_output()
        match_list = re.findall(r'dim = \[.*?\]', output)
        if len(match_list) == 0:
            raise RuntimeError(f'Cannot find dim in {image} header')
        return eval(match_list[0].strip('dim = '))

    @staticmethod
    def trim_largest_comp(image, output, trim_margin_vec):
        c3d = C3D()
        largest_comp_cmd = c3d.operand(image).dup().comp().thresh(1, 1, 1, 0).multiply()
        trim_cmd = largest_comp_cmd.trim(*trim_margin_vec).out(output)
        # trim_cmd = trim_cmd.operand(image).reslice_identity().out(output)
        trim_cmd.run()

    @staticmethod
    def set_subtract(binary_image_1, binary_image_2, output):
        c3d = C3D()
        c3d.operand(binary_image_1, binary_image_2).scale(-1).add().thresh(
            1, 1, 1, 0
        ).out(output).run()

    @staticmethod
    def dilate(binary_image, rad, output):
        c3d = C3D()
        if rad >= 0:
            c3d.operand(binary_image).dilate(label=1, r1=rad, r2=rad, r3=rad).out(
                output
            ).run()
        else:
            c3d.operand(binary_image).dilate(label=0, r1=-rad, r2=-rad, r3=-rad).out(
                output
            ).run()

    @staticmethod
    def union(binary_image_1, binary_image_2, output):
        c3d = C3D()
        c3d.operand(binary_image_1, binary_image_2).add().thresh(1, 2, 1, 0).out(
            output
        ).run()

    @staticmethod
    def distance_transform(binary_image, output):
        c3d = C3D()
        c3d.operand(binary_image).sdt().clip(0, 'inf').out(output).run()

    @staticmethod
    def label_mask_comp(binary_image, output):
        c3d = C3D()
        c3d.operand(binary_image).comp().out(output).run()

    @staticmethod
    def remap_labels(multi_label_image, label_map, output):
        reverse_map = defaultdict(list)
        for input_label, output_label in label_map.items():
            reverse_map[output_label].append(input_label)

        c3d = C3D()
        c3d.operand(multi_label_image).assign('input')

        first = True
        for output_label, labels in reverse_map.items():
            c3d.push('input').retain_labels(labels).thresh(
                min(labels), max(labels), output_label, 0
            )
            if not first:
                c3d.add()
            else:
                first = False

        c3d.out(output)
        c3d.run()


class NilearnProcessor(ImageProcessor):
    @staticmethod
    def save(nib_image, file):
        # np.int32 is problematic. 1s can turn into 0.9999999
        nib_image.set_data_dtype(np.float32)
        nib.save(nib_image, file)

    @staticmethod
    def _crop_img_to(image, slices, copy=True):

        data = nilearn.image.get_data(image)
        affine = image.affine

        cropped_data = data[tuple(slices)]
        if copy:
            cropped_data = cropped_data.copy()

        linear_part = affine[:3, :3]
        old_origin = affine[:3, 3]
        new_origin_voxel = np.array([s.start for s in slices])
        new_origin = old_origin + linear_part.dot(new_origin_voxel)

        new_affine = np.eye(4)
        new_affine[:3, :3] = linear_part
        new_affine[:3, 3] = new_origin
        return nilearn.image.new_img_like(image, cropped_data, new_affine)

    @staticmethod
    def crop_img(image, rtol=1e-8, copy=True, pad=(0, 0, 0)):

        data = nilearn.image.get_data(image)
        infinity_norm = max(-data.min(), data.max())
        passes_threshold = np.logical_or(
            data < -rtol * infinity_norm, data > rtol * infinity_norm
        )

        coords = np.array(np.where(passes_threshold))

        # Sets full range if no data are found along the axis
        if coords.shape[1] == 0:
            start, end = [0, 0, 0], list(data.shape)
        else:
            start = coords.min(axis=1)
            end = coords.max(axis=1) + 1

        start = np.maximum(start - pad, 0)
        end = np.minimum(end + pad, data.shape)

        slices = [slice(s, e) for s, e in zip(start, end)]
        cropped_im = NilearnProcessor._crop_img_to(image, slices, copy=copy)
        return cropped_im

    @staticmethod
    def threshold(
        image, output, lower_bound, upper_bound, inside_target, outside_target
    ):
        nib_image = nilearn.image.load_img(image)
        data = nib_image.get_fdata()
        threshold_data = np.where(
            (data >= lower_bound) & (data <= upper_bound), inside_target, outside_target
        ).astype(np.int16)
        threshold_image = nib.Nifti1Image(threshold_data, nib_image.affine)
        NilearnProcessor.save(threshold_image, output)

    @staticmethod
    def binarize(image, output):
        nib_image = nilearn.image.load_img(image)
        NilearnProcessor.save(nilearn.image.binarize_img(nib_image), output)

    @staticmethod
    def mask(image, mask, output):
        nib_image = nilearn.image.load_img(image)
        nib_mask = nilearn.image.load_img(mask)
        masked_image = nib.Nifti1Image(
            nib_image.get_fdata() * nib_mask.get_fdata(), nib_image.affine
        )
        NilearnProcessor.save(masked_image, output)

    @staticmethod
    def largest_mask_comp(image, output):
        try:
            nilearn.image.largest_connected_component_img(image).to_filename(output)
        except ValueError as e:
            if str(e).startswith('No non-zero values: no connected components'):
                shutil.copyfile(image, output)
            else:
                raise ValueError(e)

    @staticmethod
    def holefill(binary_image, output):
        nib_image = nilearn.image.load_img(binary_image)
        data = nib_image.get_fdata()
        holefilled_data = scipy.ndimage.binary_fill_holes(data).astype('int32')
        holefilled_image = nib.Nifti1Image(holefilled_data, nib_image.affine)
        NilearnProcessor.save(holefilled_image, output)

    @staticmethod
    def reslice_to_ref(ref_image, moving_image, output):
        nilearn.image.resample_to_img(moving_image, ref_image).to_filename(output)

    @staticmethod
    def resample_new_dim(image, output, dim1, dim2, dim3, percent=True):
        nib_image = nilearn.image.load_img(image)
        dims = nib_image.get_fdata().shape
        if percent:
            ratios = np.array([dim1 * 0.01, dim2 * 0.01, dim3 * 0.01, 1])
        else:
            ratios = np.array([dim1 / dims[0], dim2 / dims[1], dim3 / dims[2], 1])
        new_affine = nib_image.affine.dot(np.diag(1.0 / ratios))
        new_dims = (
            int(ratios[0] * dims[0]),
            int(ratios[1] * dims[1]),
            int(ratios[2] * dims[2]),
        )
        resampled_image = nilearn.image.resample_img(
            image,
            target_affine=new_affine,
            target_shape=new_dims,
            interpolation='nearest',
        )
        NilearnProcessor.save(resampled_image, output)

    @staticmethod
    def get_dims(image):
        return nilearn.image.load_img(image).get_fdata().shape

    @staticmethod
    def trim_largest_comp(image, output, trim_margin_vec):
        nib_image = nilearn.image.load_img(image)
        largest_comp_mask_image = nilearn.image.largest_connected_component_img(
            nib_image
        )
        largest_comp_image = nilearn.image.math_img(
            'img1 * img2', img1=nib_image, img2=largest_comp_mask_image
        )

        trimmed_largest_comp_image = NilearnProcessor.crop_img(
            largest_comp_image, pad=trim_margin_vec
        )
        NilearnProcessor.save(trimmed_largest_comp_image, output)

    @staticmethod
    def set_subtract(binary_image_1, binary_image_2, output):
        subtract_image = nilearn.image.binarize_img(
            nilearn.image.math_img(
                'np.maximum(img1 - img2, 0)', img1=binary_image_1, img2=binary_image_2
            )
        )
        NilearnProcessor.save(subtract_image, output)

    @staticmethod
    def dilate(binary_image, rad, output):
        nib_image = nilearn.image.load_img(binary_image)
        data = nib_image.get_fdata()
        if rad >= 0:
            dilated_data = scipy.ndimage.binary_dilation(
                data, structure=np.ones(3 * (2 * rad + 1,))
            ).astype(data.dtype)
        else:
            dilated_data = scipy.ndimage.binary_erosion(
                data, structure=np.ones(3 * (-2 * rad + 1,))
            ).astype(data.dtype)
        dilated_image = nib.Nifti1Image(dilated_data, nib_image.affine)
        NilearnProcessor.save(dilated_image, output)

    @staticmethod
    def union(binary_image_1, binary_image_2, output):
        union_image = nilearn.image.binarize_img(
            nilearn.image.math_img(
                'img1 + img2', img1=binary_image_1, img2=binary_image_2
            )
        )
        NilearnProcessor.save(union_image, output)

    @staticmethod
    def distance_transform(binary_image, output):
        nib_image = nilearn.image.load_img(binary_image)
        data = nib_image.get_fdata()
        dist_data = scipy.ndimage.distance_transform_edt(1 - data)
        dist_image = nib.Nifti1Image(dist_data, nib_image.affine)
        NilearnProcessor.save(dist_image, output)

    @staticmethod
    def label_mask_comp(binary_image, output):
        nib_image = nilearn.image.load_img(binary_image)
        comp_image = nilearn.regions.connected_label_regions(nib_image)
        comp_image_data = comp_image.get_fdata()
        labels, freq = np.unique(comp_image_data, return_counts=True)
        sorted_labels = labels[1:][np.argsort(freq[1:])[::-1]]
        sorted_comp_data = np.zeros_like(comp_image_data)
        for index, label in enumerate(sorted_labels):
            sorted_comp_data[comp_image_data == label] = index + 1

        sorted_comp_image = nib.Nifti1Image(sorted_comp_data, nib_image.affine)
        NilearnProcessor.save(sorted_comp_image, output)

    @staticmethod
    def remap_labels(multi_label_image, label_map, output):
        reverse_map = defaultdict(list)
        for input_label, output_label in label_map.items():
            reverse_map[output_label].append(input_label)

        formula = '0'
        for output_label, labels in reverse_map.items():
            formula += f' + {output_label} * np.isin(img, {labels})'
        remapped_image = nilearn.image.math_img(formula, img=multi_label_image)
        NilearnProcessor.save(remapped_image, output)
