# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import glob
from tqdm import tqdm
import numpy as np
import paddle
import cv2


class FaceDetector(object):
    """An abstract class representing a face detector.

    Any other face detection implementation must subclass it. All subclasses
    must implement ``detect_from_image``, that return a list of detected
    bounding boxes. Optionally, for speed considerations detect from path is
    recommended.
    """

    def __init__(self, verbose):
        self.verbose = verbose

    def detect_from_image(self, tensor_or_path):
        """Detects faces in a given image.

        This function detects the faces present in a provided BGR(usually)
        image. The input can be either the image itself or the path to it.

        Args:
            tensor_or_path {numpy.ndarray, paddle.tensor or string} -- the path
            to an image or the image itself.

        Example::

            >>> path_to_image = 'data/image_01.jpg'
            ...   detected_faces = detect_from_image(path_to_image)
            [A list of bounding boxes (x1, y1, x2, y2)]
            >>> image = cv2.imread(path_to_image)
            ...   detected_faces = detect_from_image(image)
            [A list of bounding boxes (x1, y1, x2, y2)]

        """
        raise NotImplementedError

    def detect_from_directory(self,
                              path,
                              extensions=['.jpg', '.png'],
                              recursive=False,
                              show_progress_bar=True):
        """Detects faces from all the images present in a given directory.

        Ars:
            path {string} -- a string containing a path that points to the folder containing the images
            extensions {list} -- list of string containing the extensions to be
            consider in the following format: ``.extension_name`` (default:
            {['.jpg', '.png']}) recursive {bool} -- option wherever to scan the
            folder recursively (default: {False}) show_progress_bar {bool} --
            display a progressbar (default: {True})

        Example:
        >>> directory = 'data'
        ...   detected_faces = detect_from_directory(directory)
        {A dictionary of [lists containing bounding boxes(x1, y1, x2, y2)]}

        """
        if self.verbose:
            logger = logging.getLogger(__name__)

        if len(extensions) == 0:
            if self.verbose:
                logger.error(
                    "Expected at list one extension, but none was received.")
            raise ValueError

        if self.verbose:
            logger.info("Constructing the list of images.")
        additional_pattern = '/**/*' if recursive else '/*'
        files = []
        for extension in extensions:
            files.extend(
                glob.glob(
                    path + additional_pattern + extension, recursive=recursive))

        if self.verbose:
            logger.info("Finished searching for images. %s images found",
                        len(files))
            logger.info("Preparing to run the detection.")

        predictions = {}
        for image_path in tqdm(files, disable=not show_progress_bar):
            if self.verbose:
                logger.info("Running the face detector on image: %s",
                            image_path)
            predictions[image_path] = self.detect_from_image(image_path)

        if self.verbose:
            logger.info("The detector was successfully run on all %s images",
                        len(files))

        return predictions

    @property
    def reference_scale(self):
        raise NotImplementedError

    @property
    def reference_x_shift(self):
        raise NotImplementedError

    @property
    def reference_y_shift(self):
        raise NotImplementedError

    @staticmethod
    def tensor_or_path_to_ndarray(tensor_or_path, rgb=True):
        """Convert path (represented as a string) or paddle.tensor to a numpy.ndarray

        Args:
            tensor_or_path {numpy.ndarray, paddle.tensor or string} -- path to the image, or the image itself
        """
        if isinstance(tensor_or_path, str):
            return cv2.imread(tensor_or_path) if not rgb else cv2.imread(
                tensor_or_path)[..., ::-1]
        elif isinstance(tensor_or_path,
                        (paddle.static.Variable, paddle.Tensor)):
            # Call cpu in case its coming from cuda
            return tensor_or_path.numpy()[
                ..., ::-1].copy() if not rgb else tensor_or_path.numpy()
        elif isinstance(tensor_or_path, np.ndarray):
            return tensor_or_path[
                ..., ::-1].copy() if not rgb else tensor_or_path
        else:
            raise TypeError
