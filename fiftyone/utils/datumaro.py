"""
Utilities for working with datasets in
`Datumaro format <https://open-edge-platform.github.io/datumaro/latest/docs/data-formats/formats/datumaro.html>`_.

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
from collections import defaultdict
import csv
from datetime import datetime
from itertools import groupby
import logging
import multiprocessing.dummy
import os
import random
import shutil
import warnings

import numpy as np
from skimage import measure

import eta.core.image as etai
import eta.core.serial as etas
import eta.core.utils as etau
import eta.core.web as etaw

import fiftyone.core.fields as fof
import fiftyone.core.labels as fol
import fiftyone.core.metadata as fom
import fiftyone.core.storage as fos
import fiftyone.core.utils as fou
import fiftyone.utils.data as foud
import fiftyone.utils.eta as foue
from fiftyone import ViewField as F

mask_utils = fou.lazy_import(
    "pycocotools.mask", callback=lambda: fou.ensure_import("pycocotools")
)


logger = logging.getLogger(__name__)


def add_datumaro_labels(
    sample_collection,
    label_field,
    labels_or_path,
    label_categories,
    label_types="detections",
    include_annotation_id=False,
    ann_attrs=True,
    item_attrs=True,
    tag_attributes=["uuid"],
    use_polylines=False,
    tolerance=None,
    overwrite_labels=True
):
    """Adds the given datumaro labels to the collection.

    The ``labels_or_path`` argument can be any of the following:

    -   a list of datumaro annotations in the format below
    -   the path to a JSON file containing a list of datumaro annotations
    -   the path to a JSON file whose ``"annotations"`` key contains a list of
        datumaro annotations

    When ``label_type="detections"``, the labels should have format::

        [
            {
                "id": 1,
                "type": "bbox",
                "label_id": 1,
                "bbox": [260, 177, 231, 199],

                # optional
                "group": 1,
                "z_order": 2,

                # extra attrs
                attributes = {}
            },
            ...
        ]

    When ``label_type="segmentations"``, the labels should have format::

        [
            {
                "id": 1,
                "type": "mask",
                "label_id": 1,
                "rle": {
                    "counts": "ncdfas34fc42000",
                    "size": [ 1536, 2048]},

                # optional
                "group": 1,
                "z_order": 2,

                # extra attrs
                attributes = {}
            },
            ...
        ]

    When ``label_type="keypoints"``, the labels should have format::

        [
            {
                "id": 1,
                "type": "points",
                "label_id": 1,
                "points": [260, 177, 231, 199],

                # optional
                "group": 1,
                "z_order": 2,
                "visibility": 2,

                # extra attrs
                attributes = {}
            },
            ...
        ]

    See `this page <https://open-edge-platform.github.io/datumaro/latest/docs/data-formats/formats/datumaro.html>`_ for more
    information about the datumaro data format.

    Args:
        sample_collection: a
            :class:`fiftyone.core.collections.SampleCollection`
        label_field: controls the field(s) in which imported labels are
            stored. If the importer produces a
            single :class:`fiftyone.core.labels.Label` instance per
            sample/frame, this argument specifies the string prefix of the field to use;
            the default is ``"ground_truth"``. If the importer produces a
            dictionary of labels per sample, this argument can be either a
            string prefix to prepend to each label key or a dict mapping label
            keys to field names; the default in this case is to directly use
            the keys of the imported label dictionaries as field names
        labels_or_path: a list of datumaro annotations or the path to a JSON file
            containing such data on disk
        label_categories: can be any of the following:
            -   a list of labels dicts in the format of
                :meth:`parse_datumaro_label_categories` specifying the classes and their
                label IDs
            -   a dict mapping class IDs to class labels
            -   a list of class labels whose 1-based ordering is assumed to
                correspond to the labels IDs in the provided datumaro labels
        label_types (None): a label type or list of label types to load. The
            supported values are
            ``("classifications", "detections", "segmentations", "keypoints")``.
            By default, all label types are loaded
        include_annotation_id (False): whether to include the ID of each
            annotation in the loaded labels
        extra_attrs (True): whether to load extra annotation attributes onto
            the imported labels. Supported values are:
            -   ``True``: load all extra attributes found
            -   ``False``: do not load extra attributes
            -   a name or list of names of specific attributes to load
        use_polylines (False): whether to represent segmentations as
            :class:`fiftyone.core.labels.Polylines` instances rather than
            :class:`fiftyone.core.labels.Detections` with dense masks
        tolerance (None): a tolerance, in pixels, when generating approximate
            polylines for instance masks. Typical values are 1-3 pixels
        overwrite_labels (True): whether existing labels with the same tag for all items
            specified in the datumaro annotations should be deleted and replaced by the new added labels
    """
    if etau.is_str(labels_or_path):
        labels = etas.load_json(labels_or_path)
        if isinstance(labels, dict):
            labels = labels["items"]
    else:
        labels = labels_or_path

    (
        items,
        item_attributes,
        annotations,
    ) = _parse_datumaro_items(labels, ann_attrs=ann_attrs, item_attrs=item_attrs, tag_attributes=tag_attributes)

    datumaro_items_map = defaultdict(list)
    for item_id, ann_dict in annotations.items():
        datumaro_obj = DatumaroObject.from_anno_dict(ann_dict, ann_attrs=ann_attrs, tag_attributes=tag_attributes)
        datumaro_items_map[item_id].append(datumaro_obj)

    # Use field `item_id` as key to match labels with samples
    item_ids_sample_collection = sample_collection.values(["item_id"])

    item_ids = sorted(datumaro_items_map.keys())
    bad_ids = set(item_ids) - set(item_ids_sample_collection)
    if bad_ids:
        item_ids = [_id for _id in item_ids if _id not in bad_ids]
        logger.warning(
            "Ignoring %d labels with nonexistent Item IDs (eg %s)",
            len(bad_ids),
            next(iter(bad_ids)),
        )

    # prepare inputs
    if isinstance(label_categories, dict):
        classes_map = label_categories
    elif not label_categories:
        classes_map = {}
    elif isinstance(label_categories[0], dict):
        classes_map = parse_datumaro_label_categories(label_categories)
    else:
        classes_map = {i: label for i, label in enumerate(label_categories, 1)}

    _label_types = _parse_label_types(label_types)

    if isinstance(label_field, dict):
        label_field_key = lambda k: label_field.get(k, k)
    elif label_field is not None:
        label_field_key = lambda k: label_field + "_" + k
    else:
        label_field = "ground_truth"
        label_field_key = lambda k: label_field + "_" + k

    # iterate through samples with item_id
    for item_id in item_ids:
        sample_view = sample_collection.select_by("item_id", item_id, ordered=True)
        sample_view.compute_metadata()

        width, height = sample_view.values(["metadata.width", "metadata.height"])
        frame_size = (width, height)
        
        item_label = {}
        datumaro_objects = datumaro_items_map[item_id]

        ## read item attributes into a seperate label of custom type "Item_attributes"
        if item_attrs:
            if item_attributes[item_id]:
                item_label["item_attributes"] = Item_attributes.from_dict(item_attributes[item_id])

        if "detections" in _label_types:
            detections = _datumaro_objects_to_detections(
                datumaro_objects,
                frame_size,
                classes_map,
                False,  # no segmentations
                include_annotation_id
            )
            if detections is not None:
                item_label["detections"] = detections

        if "segmentations" in _label_types:
            if use_polylines:
                segmentations = _datumaro_objects_to_detections(
                    datumaro_objects,
                    frame_size,
                    classes_map,
                    tolerance,
                    include_annotation_id
                )
            else:
                segmentations = _datumaro_objects_to_detections(
                    datumaro_objects,
                    frame_size,
                    classes_map,
                    True,  # load segmentations
                    include_annotation_id
                )

            if segmentations is not None:
                item_label["segmentations"] = segmentations

        if "keypoints" in _label_types:
            keypoints = _datumaro_objects_to_keypoints(
                datumaro_objects,
                frame_size,
                classes_map,
                include_annotation_id,
            )

            if keypoints is not None:
                labels["keypoints"] = keypoints

        if "classifications" in _label_types:
            classifications = _datumaro_objects_to_classifications(
                datumaro_objects,
                classes_map,
                include_annotation_id,
            )

            if classifications is not None:
                labels["classifications"] = classifications

        if item_label:
            for label_key, label_values in item_label.items():
                full_label_field = label_field_key(label_key)
                tag = label_values[0].tag
                if overwrite_labels:
                    filtered_sample_view = sample_view.filter_labels(full_label_field, F("tag") != tag)
                    if sample_view[full_label_field]:
                        sample_view[full_label_field] = filtered_sample_view + label_values
                    else:
                        sample_view[full_label_field] = label_values
                else:
                    if sample_view[full_label_field]:
                        sample_view[full_label_field] = sample_view[full_label_field] + label_values
                    else:
                        sample_view[full_label_field] = label_values

        sample_view.save()


class DatumaroDatasetImporter(
    foud.LabeledImageDatasetImporter, foud.ImportPathsMixin
):
    """Importer for Datumaro annotated datasets stored on disk.

    See :ref:`this page <DatumaroDataset-import>` for format details.

    Args:
        dataset_dir (None): the dataset directory. If omitted, ``data_path``
            and/or ``labels_path`` must be provided
        data_path (None): an optional parameter that enables explicit control
            over the location of the media. Can be any of the following:

            -   a folder name like ``"data"`` or ``"data/"`` specifying a
                subfolder of ``dataset_dir`` where the media files reside
            -   an absolute directory path where the media files reside. In
                this case, the ``dataset_dir`` has no effect on the location of
                the data
            -   a filename like ``"data.json"`` specifying the filename of the
                JSON data manifest file in ``dataset_dir``
            -   an absolute filepath specifying the location of the JSON data
                manifest. In this case, ``dataset_dir`` has no effect on the
                location of the data
            -   a dict mapping filenames to absolute filepaths

            If None, this parameter will default to whichever of ``data/`` or
            ``data.json`` exists in the dataset directory
        labels_path (None): an optional parameter that enables explicit control
            over the location of the labels. Can be any of the following:

            -   a filename like ``"labels.json"`` specifying the location of
                the labels in ``dataset_dir``
            -   an absolute filepath to the labels. In this case,
                ``dataset_dir`` has no effect on the location of the labels

            If None, the parameter will default to ``labels.json``
        label_types (None): a label type or list of label types to load. The
            supported values are
            ``("classifications", "detections", "segmentations", "keypoints")``.
            By default, all label types are loaded
        classes (None): a string or list of strings specifying required classes
            to load. Only samples containing at least one instance of a
            specified class will be loaded
        item_ids (None): an optional list of specific item IDs to load. Can
            be provided in any of the following formats:

            -   a list of ``<item-id>`` strings
            -   a list of ``<split>/<item-id>`` strings
            -   the path to a text (newline-separated), JSON, or CSV file
                containing the list of item IDs to load in either of the first
                two formats
        include_annotation_id (False): whether to include the ID of each
            annotation in the loaded labels
        ann_attrs (True): whether to load annotation attributes onto
            the imported labels. Supported values are:
            -   ``True``: load all attributes found
            -   ``False``: do not load attributes
            -   a name or list of names of specific attributes to load
        item_attrs (True): whether to load item attributes into a seperate label.
                            Supported values are:
            -   ``True``: load all item attributes found
            -   ``False``: do not load item attributes
            -   a name or list of names of specific attributes to load
        tag_attributes (None): a list of attributes names that will be concatenated\
                               with a seperating underscore to create a tag string.
        only_matching (False): whether to only load labels that match the
            ``classes`` requirement that you provide (True), or to load all
            labels for samples that match the requirements (False)
        use_polylines (False): whether to represent segmentations as
            :class:`fiftyone.core.labels.Polylines` instances rather than
            :class:`fiftyone.core.labels.Detections` with dense masks
        tolerance (None): a tolerance, in pixels, when generating approximate
            polylines for instance masks. Typical values are 1-3 pixels
        shuffle (False): whether to randomly shuffle the order in which the
            samples are imported
        seed (None): a random seed to use when shuffling
        max_samples (None): a maximum number of samples to load. If
            ``label_types`` and/or ``classes`` are also specified, first
            priority will be given to samples that contain all of the specified
            label types and/or classes, followed by samples that contain at
            least one of the specified labels types or classes. The actual
            number of samples loaded may be less than this maximum value if the
            dataset does not contain sufficient samples matching your
            requirements. By default, all matching samples are loaded
        read_metadata_from_file (False): whether to load metadata informations that are
            stored in the .png-file in the tEXT format and store them as additional labels
        read_uuid_from_filename (False): whether to read the uuid from the filename
            and store them as an additional annotation attribute
    """

    def __init__(
        self,
        dataset_dir=None,
        data_path=None,
        labels_path=None,
        label_types=None,
        classes=None,
        item_ids=None,
        include_annotation_id=False,
        ann_attrs=True,
        item_attrs=True,
        only_matching=False,
        use_polylines=False,
        tolerance=None,
        shuffle=False,
        seed=None,
        max_samples=None,
        tag_attributes=None,
        read_metadata_from_file=False,
        read_uuid_from_filename=False
    ):
        if dataset_dir is None and data_path is None and labels_path is None:
            raise ValueError(
                "At least one of `dataset_dir`, `data_path`, and "
                "`labels_path` must be provided"
            )

        data_path = self._parse_data_path(
            dataset_dir=dataset_dir,
            data_path=data_path,
            default="images/default/",
        )

        labels_path = self._parse_labels_path(
            dataset_dir=dataset_dir,
            labels_path=labels_path,
            default="/annotations/default.json",
        )

        _label_types = _parse_label_types(label_types)

        super().__init__(
            dataset_dir=dataset_dir,
            shuffle=shuffle,
            seed=seed,
            max_samples=max_samples,
        )

        self.data_path = data_path
        self.labels_path = labels_path
        self.label_types = label_types
        self.classes = classes
        self.item_ids = item_ids
        self.include_annotation_id = include_annotation_id
        self.ann_attrs = ann_attrs
        self.item_attrs = item_attrs
        self.only_matching = only_matching
        self.use_polylines = use_polylines
        self.tolerance = tolerance
        self.tag_attributes = tag_attributes
        self.read_metadata_from_file = read_metadata_from_file
        self.read_uuid_from_filename = read_uuid_from_filename

        self._label_types = _label_types
        self._info = None
        self._classes_map = None
        self._class_ids = None
        self._item_paths_map = None
        self._annotations = None
        self._filenames = None
        self._iter_filenames = None
        self._item_attributes = None

    def __iter__(self):
        self._iter_filenames = iter(self._filenames)
        return self

    def __len__(self):
        return len(self._filenames)

    def __next__(self):
        filename = next(self._iter_filenames)

        if os.path.isabs(filename):
            item_path = filename
        else:
            item_path = self._item_paths_map[filename]

        item_metadata = fom.ImageMetadata.build_for(item_path)

        item_id = ".".join(filename.split(".")[:-1])
        width = item_metadata.width
        height = item_metadata.height

        labels = {}
        labels.update({"item_id": item_id})

        # read uuid from filename
        if self.read_uuid_from_filename:
            labels.update({"uuid": "".join(filename.split(".")[:-1]).split("_")[-1]})

        # read metadata (like location, sensor settings, etc.) from file into seperate fields
        if self.read_metadata_from_file:
            labels.update(read_metadata_from_image_file(item_path))

        ## read item attributes into a seperate label of custom type "Item_attributes"
        if self.item_attrs:
            if self._item_attributes[item_id]:
                labels["item_attributes"] = Item_attributes.from_dict(self._item_attributes[item_id])

        if self._annotations is not None:
            datumaro_objects = self._annotations.get(item_id, [])
            frame_size = (width, height)

            if self.only_matching and self._class_ids is not None:
                datumaro_objects = _get_matching_objects(
                    datumaro_objects, self._class_ids
                )

            if "detections" in self._label_types:
                detections = _datumaro_objects_to_detections(
                    datumaro_objects,
                    frame_size,
                    self._classes_map,
                    False,  # no segmentations
                    self.include_annotation_id
                )
                if detections is not None:
                    labels["detections"] = detections

            if "segmentations" in self._label_types:
                if self.use_polylines:
                    segmentations = _datumaro_objects_to_polylines(
                        datumaro_objects,
                        frame_size,
                        self._classes_map,
                        self.tolerance,
                        self.include_annotation_id,
                    )
                else:
                    segmentations = _datumaro_objects_to_detections(
                        datumaro_objects,
                        frame_size,
                        self._classes_map,
                        True,  # load segmentations
                        self.include_annotation_id,
                    )

                if segmentations is not None:
                    labels["segmentations"] = segmentations

            if "keypoints" in self._label_types:
                keypoints = _datumaro_objects_to_keypoints(
                    datumaro_objects,
                    frame_size,
                    self._classes_map,
                    self.include_annotation_id,
                )

                if keypoints is not None:
                    labels["keypoints"] = keypoints

            if "classifications" in self._label_types:
                classifications = _datumaro_objects_to_classifications(
                    datumaro_objects,
                    self._classes_map,
                    self.include_annotation_id,
                )

                if classifications is not None:
                    labels["classifications"] = classifications

            if "datumaro_item_id" in self._label_types:
                labels["datumaro_item_id"] = item_id

        if self._has_scalar_labels:
            labels = next(iter(labels.values())) if labels else None

        return item_path, item_metadata, labels

    @property
    def has_dataset_info(self):
        return True

    @property
    def has_image_metadata(self):
        return True

    @property
    def _has_scalar_labels(self):
        return len(self._label_types) == 1

    @property
    def label_cls(self):
        seg_type = fol.Polylines if self.use_polylines else fol.Detections
        types = {
            "classifications": fol.Classifications,
            "detections": fol.Detections,
            "segmentations": seg_type,
            "keypoints": fol.Keypoints,
            "item_attributes": Item_attributes
        }

        if self._has_scalar_labels:
            return types[self._label_types[0]]

        return {k: v for k, v in types.items() if k in self._label_types}

    def setup(self):
        item_paths_map = self._load_data_map(self.data_path, recursive=True)

        if self.labels_path is not None and os.path.isfile(self.labels_path) and item_paths_map:
            (
                info,
                classes_map,
                items,
                item_attributes,
                annotations,
            ) = load_datumaro_items(
                self.labels_path, ann_attrs=self.ann_attrs, item_attrs=self.item_attrs, tag_attributes=self.tag_attributes
            )

            if classes_map is not None:
                info["classes"] = _to_classes(classes_map)

            item_ids = _get_matching_item_ids(
                classes_map,
                items,
                annotations,
                item_ids=self.item_ids,
                classes=self.classes,
                shuffle=self.shuffle,
                seed=self.seed,
                max_samples=self.max_samples,
            )
            item_ids = set(item_ids)

            filenames = fos.normpath(list(item_paths_map.keys()))
            if not self.load_all_images_without_labels:
                # check if image files exist for every item_id
                item_ids_with_file = [".".join(filename.split(".")[:-1]) for filename in filenames]
                indices = [index for index, value in enumerate(item_ids_with_file) if value in item_ids]
                filenames_exist = [filenames[i] for i in indices]
                filenames = filenames_exist

        else:
            info = {}
            classes_map = None
            items = None
            item_attributes = None
            annotations = None
            filenames = fos.normpath(list(item_paths_map.keys()))

        if self.only_matching and self.classes is not None:
            class_ids = _get_class_ids(self.classes, classes_map)
        else:
            class_ids = None

        self._info = info
        self._classes_map = classes_map
        self._class_ids = class_ids
        self._items = items
        self._item_attributes = item_attributes
        self._item_paths_map = item_paths_map
        self._annotations = annotations
        self._filenames = filenames

    def get_dataset_info(self):
        return self._info


class DatumaroDatasetExporter(
    foud.LabeledImageDatasetExporter, foud.ExportPathsMixin
):
    """Exporter that writes Datumaro datasets to disk.

    See :ref:`this page <DatumaroDataset-import>` for format details.

    Args:
        export_dir (None): the directory to write the export. This has no
            effect if ``data_path`` and ``labels_path`` are absolute paths
        data_path (None): an optional parameter that enables explicit control
            over the location of the exported media. Can be any of the
            following:
            -   a folder name like ``"data"`` or ``"data/"`` specifying a
                subfolder of ``export_dir`` in which to export the media
            -   an absolute directory path in which to export the media. In
                this case, the ``export_dir`` has no effect on the location of
                the data
            -   a JSON filename like ``"data.json"`` specifying the filename of
                the manifest file in ``export_dir`` generated when
                ``export_media`` is ``"manifest"``
            -   an absolute filepath specifying the location to write the JSON
                manifest file when ``export_media`` is ``"manifest"``. In this
                case, ``export_dir`` has no effect on the location of the data
            If None, the default value of this parameter will be chosen based
            on the value of the ``export_media`` parameter
        labels_path (None): an optional parameter that enables explicit control
            over the location of the exported labels. Can be any of the
            following:
            -   a filename like ``"labels.json"`` specifying the location in
                ``export_dir`` in which to export the labels
            -   an absolute filepath to which to export the labels. In this
                case, the ``export_dir`` has no effect on the location of the
                labels
            If None, the labels will be exported into ``export_dir`` using the
            default filename
        export_media (None): controls how to export the raw media. The
            supported values are:
            -   ``True``: copy all media files into the output directory
            -   ``False``: don't export media
            -   ``"move"``: move all media files into the output directory
            -   ``"symlink"``: create symlinks to the media files in the output
                directory
            -   ``"manifest"``: create a ``data.json`` in the output directory
                that maps UUIDs used in the labels files to the filepaths of
                the source media, rather than exporting the actual media
            If None, the default value of this parameter will be chosen based
            on the value of the ``data_path`` parameter
        rel_dir (None): an optional relative directory to strip from each input
            filepath to generate a unique identifier for each image. When
            exporting media, this identifier is joined with ``data_path`` to
            generate an output path for each exported image. This argument
            allows for populating nested subdirectories that match the shape of
            the input paths. The path is converted to an absolute path (if
            necessary) via :func:`fiftyone.core.storage.normalize_path`
        abs_paths (False): whether to store absolute paths to the images in the
            exported labels
        image_format (None): the image format to use when writing in-memory
            images to disk. By default, ``fiftyone.config.default_image_ext``
            is used
        classes (None): the list of possible class labels
        label_categories (None): a list of label category dicts in the format of
            :meth:`parse_datumaro_label_categories` specifying the classes and their
            label IDs
        info (None): a dict of info as returned by
            :meth:`load_datumaro_items` to include in the exported
            JSON. If not provided, this info will be extracted when
            :meth:`log_collection` is called, if possible
        extra_attrs (True): whether to include extra object attributes in the
            exported labels. Supported values are:
            -   ``True``: export all extra attributes found
            -   ``False``: do not export extra attributes
            -   a name or list of names of specific attributes to export
        ####coco_id (None): the name of a sample field containing the COCO IDs of
            each image####
        annotation_id (None): the name of a label field containing the COCO
            annotation ID of each label
        num_decimals (None): an optional number of decimal places at which to
            round bounding box pixel coordinates. By default, no rounding is
            done
        tolerance (None): a tolerance, in pixels, when generating approximate
            polylines for instance masks. Typical values are 1-3 pixels
    """

    def __init__(
        self,
        export_dir=None,
        data_path=None,
        labels_path=None,
        export_media=None,
        rel_dir=None,
        abs_paths=False,
        image_format=None,
        classes=None,
        label_categories=None,
        info=None,
        extra_attrs=True,
        coco_id=None,
        annotation_id=None,
        num_decimals=None,
        tolerance=None,
    ):
        data_path, export_media = self._parse_data_path(
            export_dir=export_dir,
            data_path=data_path,
            export_media=export_media,
            default="images/default/",
        )

        labels_path = self._parse_labels_path(
            export_dir=export_dir,
            labels_path=labels_path,
            default="/annotations/default.json",
        )

        super().__init__(export_dir=export_dir)

        self.data_path = data_path
        self.labels_path = labels_path
        self.export_media = export_media
        self.rel_dir = rel_dir
        self.abs_paths = abs_paths
        self.image_format = image_format
        self.classes = classes
        self.label_categories = label_categories
        self.info = info
        self.extra_attrs = extra_attrs
        self.coco_id = coco_id
        self.annotation_id = annotation_id
        self.num_decimals = num_decimals
        self.tolerance = tolerance

        self._image_id = None
        self._image_id_map = None
        self._anno_id = None
        self._images = None
        self._annotations = None
        self._classes = None
        self._dynamic_classes = None
        self._labels_map_rev = None
        self._has_labels = None
        self._media_exporter = None

    @property
    def requires_image_metadata(self):
        return True

    @property
    def label_cls(self):
        return (fol.Detections, fol.Polylines, fol.Keypoints, fol.Classifications, Item_attributes)

    def setup(self):
        self._image_id = 0
        self._anno_id = 0
        self._images = []
        self._annotations = []
        self._has_labels = False

        self._parse_classes()

        self._media_exporter = foud.ImageExporter(
            self.export_media,
            export_path=self.data_path,
            rel_dir=self.rel_dir,
            default_ext=self.image_format,
        )
        self._media_exporter.setup()

    def log_collection(self, sample_collection):
        if self.info is None:
            self.info = sample_collection.info

        if self.coco_id is not None:
            self._image_id_map = dict(
                zip(*sample_collection.values(["filepath", self.coco_id]))
            )

    def export_sample(self, image_or_path, label, metadata=None):
        out_image_path, uuid = self._media_exporter.export(image_or_path)

        if metadata is None:
            metadata = fom.ImageMetadata.build_for(image_or_path)

        if self.abs_paths:
            file_name = out_image_path
        else:
            file_name = uuid

        if self._image_id_map is not None:
            image_id = self._image_id_map.get(image_or_path, None)
            if image_id is None:
                msg = (
                    "Ignoring sample with filepath '%s' that has no image ID"
                    % image_or_path
                )
                warnings.warn(msg)
                return

            image_id = int(image_id)
        else:
            self._image_id += 1
            image_id = self._image_id

        ## not needed for datumaro
        self._images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "height": metadata.height,
                "width": metadata.width,
                "license": None,
                "coco_url": None,
            }
        )

        ## return only images with labels or not?
        if label is None:
            return

        self._has_labels = True

        if isinstance(label, fol.Detections):
            labels = label.detections
        elif isinstance(label, fol.Polylines):
            labels = label.polylines
        elif isinstance(label, fol.Keypoints):
            labels = label.keypoints
        else:
            raise ValueError(
                "Unsupported label type %s. The supported types are %s"
                % (type(label), self.label_cls)
            )

        for label in labels:
            _label = label.label

            if self._dynamic_classes:
                category_id = _label  # will be converted to int later
                self._classes.add(_label)
            else:
                if _label not in self._labels_map_rev:
                    msg = (
                        "Ignoring object with label '%s' not in provided "
                        "classes" % _label
                    )
                    warnings.warn(msg)
                    continue

                category_id = self._labels_map_rev[_label]

            self._anno_id += 1

            obj = COCOObject.from_label(
                label,
                metadata,
                image_id=image_id,
                category_id=category_id,
                extra_attrs=self.extra_attrs,
                id_attr=self.annotation_id,
                iscrowd=self.iscrowd,
                num_decimals=self.num_decimals,
                tolerance=self.tolerance,
            )

            if obj.id is None:
                obj.id = self._anno_id

            self._annotations.append(obj.to_anno_dict())

    def close(self, *args):
        if self._dynamic_classes:
            labels_map_rev = _to_labels_map_rev(sorted(self._classes))
            for anno in self._annotations:
                anno["category_id"] = labels_map_rev[anno["category_id"]]
        elif self.categories is None:
            labels_map_rev = _to_labels_map_rev(self.classes)

        _info = self.info or {}
        _date_created = datetime.now().replace(microsecond=0).isoformat()

        info = {
            "year": _info.get("year", ""),
            "version": _info.get("version", ""),
            "contributor": _info.get("contributor", ""),
            "url": _info.get("url", "https://voxel51.com/fiftyone"),
            "date_created": _info.get("date_created", _date_created),
        }

        licenses = _info.get("licenses", [])

        if self.categories is not None:
            categories = self.categories
        else:
            categories = [
                {
                    "id": i,
                    "name": c,
                    "supercategory": None,
                }
                for c, i in sorted(labels_map_rev.items(), key=lambda t: t[1])
            ]

        labels = {
            "info": info,
            "licenses": licenses,
            "categories": categories,
            "images": self._images,
        }

        if self._has_labels:
            labels["annotations"] = self._annotations

        etas.write_json(labels, self.labels_path)

        self._media_exporter.close()

    def _parse_classes(self):
        if self.categories is not None:
            self._labels_map_rev = _parse_label_categories(
                self.label_categories, classes=self.classes
            )
            self._dynamic_classes = False
        elif self.classes is None:
            self._classes = set()
            self._dynamic_classes = True
        else:
            self._labels_map_rev = _to_labels_map_rev(self.classes)
            self._dynamic_classes = False


class DatumaroObject(object):
    """An object in Datumaro format.

    Args:
        id (None): the ID of the annotation
        type (None): the type of the annotation, supported are:\
                     label, mask, bbox, polygon, points
        label_id (None): the labels ID of the annotation matching\
                         to the "labels"-list in the categories dict
        attributes (None): dict with custom attributes
        z_order (None): the z-order of the annotation
        group (None): groupnumber the annotation belongs to
        visibility (None): the visibility of annotations of type "points"
        rle (None): a binary mask for the annotation in\
                    ``[dict] run-length encoding`` format
        points (None): a list of points for the annotation in\
                       ``[x, y, ...]`` format
        bbox (None): a bounding box for the annotation in\
                     ``[xmin, ymin, width, height]`` format
        tag_attributes (None): a list of attributes names that will be concatenated\
                               with a seperating underscore to create a tag string.
    """

    def __init__(
        self,
        id=None,
        type=None,
        label_id=None,
        attributes=None,
        z_order=0,
        group=0,
        visibility=[0],
        rle=None,
        points=None,
        bbox=None,
        tag_attributes=None,
    ):
        self.id = id
        self.type = type
        self.label_id = label_id
        self.attributes = attributes
        self.z_order = z_order
        self.group = group
        self.visibility = visibility
        self.rle = rle
        self.points = points
        self.bbox = bbox
        if tag_attributes is not None:
            self.tag = ["_".join(list(str(attributes.get[tag_attribute, "NN"]) for tag_attribute in tag_attributes))]
        else:
            self.tag = []

    def to_polyline(
        self,
        frame_size,
        classes_map=None,
        tolerance=None,
        include_id=False,
    ):
        """Returns a :class:`fiftyone.core.labels.Polyline` representation of
        the object.

        Args:
            frame_size: the ``(width, height)`` of the image
            classes_map (None): a dict mapping class IDs to class labels
            tolerance (None): a tolerance, in pixels, when generating
                approximate polylines for instance masks. Typical values are
                1-3 pixels
            include_id (False): whether to include the ID of the object as
                a label attribute

        Returns:
            a :class:`fiftyone.core.labels.Polyline`, or None if no
            segmentation data is available
        """
        if not self.rle:
            return None

        label, attributes = self._get_object_label_and_attributes(
            classes_map, include_id
        )
        attributes.update(self.attributes)
        attributes.update(self.z_order, self.group)

        points = _get_polygons_for_segmentation(
            self.rle, frame_size, tolerance
        )

        return fol.Polyline(
            label=label,
            points=points,
            closed=False,
            filled=True,
            tags=self.tag,
            **attributes,
        )

    def to_classification(
        self,
        classes_map=None,
        include_id=False
    ):
        """Returns a :class:`fiftyone.core.labels.Classification` representation of
        the object.

        Args:
            classes_map (None): a dict mapping class IDs to class labels
            include_id (False): whether to include the ID of the object as
                a label attribute

        Returns:
            a :class:`fiftyone.core.labels.Classification`
        """

        label, attributes = self._get_object_label_and_attributes(
            classes_map, include_id
        )
        attributes.update(self.attributes)
        attributes.update(self.z_order, self.group)

        return fol.Classification(
            label=label,
            tags=self.tag,
            **attributes,
        )

    def to_keypoints(
        self,
        frame_size,
        classes_map=None,
        include_id=False,
    ):
        """Returns a :class:`fiftyone.core.labels.Keypoint` representation of
        the object.

        Args:
            frame_size: the ``(width, height)`` of the image
            classes_map (None): a dict mapping class IDs to class labels
            include_id (False): whether to include the ID of the object as
                a label attribute

        Returns:
            a :class:`fiftyone.core.labels.Keypoint`, or None if no keypoints
            data is available
        """
        if self.points is None:
            return None

        label, attributes = self._get_object_label_and_attributes(
            classes_map, include_id
        )
        attributes.update(self.attributes)
        attributes.update(self.z_order, self.group, self.visibility)

        width, height = frame_size

        points = []
        for x, y in fou.iter_batches(self.points, 2):
                points.append((x / width, y / height))

        return fol.Keypoint(
            label=label, points=points, **attributes
        )

    def to_detection(
        self,
        frame_size,
        classes_map=None,
        load_segmentation=False,
        include_id=False
    ):
        """Returns a :class:`fiftyone.core.labels.Detection` representation of
        the object.

        Args:
            frame_size: the ``(width, height)`` of the image
            classes_map (None): a dict mapping class IDs to class labels
            load_segmentation (False): whether to load the segmentation mask
                for the object, if available
            include_id (False): whether to include the ID of the object as
                a label attribute

        Returns:
            a :class:`fiftyone.core.labels.Detection`, or None if no bbox data
            is available
        """
        if self.type == "mask" and self.rle:
            self.bbox = mask_utils.toBbox(self.rle)
        
        if self.bbox is None:
            return None

        label, attributes = self._get_object_label_and_attributes(
            classes_map, include_id
        )
        attributes.update(self.attributes)
        attributes.update(self.z_order, self.group)

        width, height = frame_size
        x, y, w, h = self.bbox
        bounding_box = [x / width, y / height, w / width, h / height]

        if load_segmentation and self.rle:
            mask = _datumaro_segmentation_to_mask(
                self.rle, self.bbox, frame_size
            )
        else:
            mask = None

        return fol.Detection(
            label=label,
            bounding_box=bounding_box,
            mask=mask,
            tags=self.tag,
            **attributes,
        )

    def to_anno_dict(self):
        """Returns a Datumaro annotation dictionary representation of the object.

        Returns:
            a Datumaro annotation dict
        """
        d = {
            "id": self.id,
            "type": self.type,
            "label_id": self.label_id
        }

        if self.bbox is not None:
            d["bbox"] = self.bbox

        if self.points is not None:
            d["points"] = self.points

        if self.rle is not None:
            d["rle"] = self.rle

        if self.group is not None:
            d["group"] = self.group

        if self.z_order is not None:
            d["z_order"] = self.z_order

        if self.visibility is not None:
            d["visibility"] = self.visibility

        if self.attributes:
            d["attributes"] = self.attributes

        return d

    @classmethod
    def from_anno_dict(cls, d, ann_attrs=True, tag_attributes=None):
        """Creates a :class:`DatumaroObject` from a Datumaro annotation dict.

        Args:
            d: a Datumaro annotation dict
            ann_attrs (True): whether to load annotation attributes.
                Supported values are:

                -   ``True``: load all attributes
                -   ``False``: do not load attributes
                -   a name or list of names of specific attributes to load
            tag_attributes (None): a list of attributes names that will be concatenated\
                with a seperating underscore to create a tag string.
        Returns:
            a :class:`DatumaroObject`
        """
        if ann_attrs is True:
            attributes = d.get("attributes", {})
        else:
            attributes = {}

        if etau.is_str(ann_attrs):
            ann_attrs = [ann_attrs]

        if isinstance(ann_attrs, list):
            attributes = {f: d["attributes"].get(f, None) for f in ann_attrs}
            
        return cls(
            id=d.get("id", None),
            type=d.get("type", None),
            label_id=d.get("label_id", None),
            attributes=attributes,
            z_order=d.get("z_order", None),
            group=d.get("group", None),
            visibility=d.get("visibility", None),
            rle=d.get("rle", None),
            points=d.get("points", None),
            bbox=d.get("bbox", None),
            tag_attributes=tag_attributes
        )

    @classmethod
    def from_label(
        cls,
        label,
        metadata,
        label_id=None,
        extra_attrs=True,
        id_attr=None,
        num_decimals=None,
        treat_polyline_as_segmentation=None
    ):
        """Creates a :class:`DatumaroObject` from a compatible
        :class:`fiftyone.core.labels.Label`.

        Args:
            label: a :class:`fiftyone.core.labels.Detection`,
                :class:`fiftyone.core.labels.Polyline`,
                :class:`fiftyone.core.labels.Classification` or
                :class:`fiftyone.core.labels.Keypoint`
            metadata: a :class:`fiftyone.core.metadata.ImageMetadata` for the
                image
            label_id (None): the label ID for the object
            extra_attrs (True): whether to include extra attributes from the
                object. Supported values are:
                -   ``True``: include all extra attributes found
                -   ``False``: do not include extra attributes
                -   a name or list of names of specific attributes to include
            id_attr (None): the name of the attribute containing the annotation
                ID of the label, if any
            num_decimals (None): an optional number of decimal places at which
                to round bounding box pixel coordinates. By default, no
                rounding is done
            treat_polyline_as_segmentation (None): whether to convert a polygon into a mask
                
        Returns:
            a :class:`DatumaroObject`
        """
        width = metadata.width
        height = metadata.height
        frame_size = (width, height)

        bbox = None
        rle = None
        points = None
        attributes = {}
        visibility = [0]

        if isinstance(label, fol.Detection):
            x, y, w, h = label.bounding_box
            bbox = [x * width, y * height, w * width, h * height]
            type = "bbox"

            if label.has_mask:
                rle = _instance_to_datumaro_segmentation(
                    label, frame_size
                )
                type = "mask"
        elif isinstance(label, fol.Polyline):
            if treat_polyline_as_segmentation:
                rle = _polyline_to_datumaro_segmentation(
                    label, frame_size
                )
                type = "mask"
            else:
                points = np.concatenate(label.points, axis=0)
                type = "polygon"

        elif isinstance(label, fol.Classification):
            type="label"
            
        elif isinstance(label, fol.Keypoint):
            points = label.points
            num_points = len(points)
            visibility = [None] * num_points
            type="points"

        else:
            raise ValueError("Unsupported label type %s" % type(label))

        if bbox is not None:
            if num_decimals is not None:
                bbox = [round(p, num_decimals) for p in bbox]

        if id_attr is not None:
            _id = label.get_attribute_value(id_attr, 0)
        else:
            _id = 0

        attributes = _get_attributes(label, extra_attrs)
        attributes.pop(id_attr, None)  # okay if `id_attr` is None
        z_order = attributes.pop(z_order, 0)
        group = attributes.pop(group, 0)
        attributes.pop(visibility, [0])

        return cls(
            id=_id,
            type=type,
            label_id=label_id,
            attributes=attributes,
            bbox=bbox,
            rle=rle,
            points=points,
            z_order = z_order,
            group = group,
            visibility = visibility
        )

    def _get_label(self, classes):
        if classes:
            return classes[self.label_id]

        return str(self.label_id)

    def _get_object_label_and_attributes(
        self, classes_map, include_id
    ):
        if classes_map:
            label = classes_map[self.label_id]
        else:
            label = str(self.label_id)

        attributes = {}

        if self.include_id:
            attributes["annotation_id"] = self.id

        return label, attributes


class Item_attributes(fol._HasID, fol.Label):
    """
    
    """
    tags = []


def read_metadata_from_image_file(image_path: str) -> dict:
    """
    read metadata from .png file in tEXT format
    """
    from PIL import Image
    import json, fiftyone
    
    img = Image.open(image_path)
    metadata_dict = img.info
    metadata_fields_dict = {}

    for meta_object in metadata_dict:
        if meta_object == "recording_location":
            location = json.loads(metadata_dict[meta_object])
            metadata_fields_dict["recording_location"] = fiftyone.GeoLocation(point=[location["lon"], location["lat"]])
        
        elif meta_object == "recording_timestamp":
            metadata_fields_dict["recording_timestamp"] = datetime.datetime.fromtimestamp(float(metadata_dict[meta_object]))
        
        elif meta_object == "camera_name":
            metadata_fields_dict["camera_name"] = metadata_dict[meta_object]

        # metadata is a dict formatted in a json string
        else:
            meta_dict = json.loads(metadata_dict[meta_object])
            if meta_object == "weather":
                meta_dict = flatten_dict(meta_dict)
            metadata_fields_dict[meta_object] = fiftyone.DynamicEmbeddedDocument().from_dict(meta_dict)


def flatten_dict(input_dict, parent_key='', sep='_') -> dict:
    """
    Flatten a nested dictionary.

    Parameters:
    - input_dict: A dictionary to flatten.
    - parent_key: The base key string (used for recursive calls).
    - sep: The separator used for the concatenated keys.

    Returns:
    - A flattened dictionary.
    """
    items = {}
    for k, v in input_dict.items():
        # Create new key for the flattened dictionary
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        # If the value is a dictionary, recursively flatten it
        if isinstance(v, list):
            if isinstance(v[0], dict):
                items.update(flatten_dict(v[0], new_key, sep=sep))
        else:
            items[new_key] = v  # Otherwise, just assign the value
    return items


def load_datumaro_items(json_path, ann_attrs=True, item_attrs=True, tag_attributes=None):
    """Loads the Datumaro items from the given JSON file.

    See :ref:`this page <DatumaroDataset-import>` for format details.

    Args:
        json_path: the path to the datumaro JSON file
        ann_attrs (True): whether to load annotation attributes.
            Supported values are:
            -   ``True``: load all attributes found
            -   ``False``: do not load attributes
            -   a name or list of names of specific attributes to load
        item_attrs (True): whether to load image attributes.
                Supported values are:
                -   ``True``: load all image attributes
                -   ``False``: do not load image attributes
                -   a name or list of names of specific attributes to load
        tag_attributes (None): a list of attributes names that will be concatenated\
            with a seperating underscore to create a tag string.

    Returns:
        a tuple of
        -   info: a dict of dataset info
        -   classes_map: a dict mapping label IDs to labels
        -   items: a list of item IDs of all items contained in the JSON file
        -   item_attributes: a dict mapping item IDs to a dict of item_attributes or ``None``
        -   annotations: a dict mapping item IDs to list of
            :class:`DatumaroObject` instances, or ``None`` for unlabeled datasets
    """
    datumaro_json = etas.load_json(json_path)

    info = datumaro_json.get("info", None)
    categories = datumaro_json.get("categories", None)
    if info is None:
        info = {}

    if categories is not None:
        info["categories"] = categories

    # Load classes
    if categories is not None:
        classes_map = parse_datumaro_label_categories(categories)
    else:
        classes_map = None

    items, item_attributes, annotations = _parse_datumaro_items(datumaro_json["items"], ann_attrs=ann_attrs, item_attrs=item_attrs,
                                                                tag_attributes=tag_attributes)

    return info, classes_map, items, item_attributes, annotations


def _parse_datumaro_items(d, ann_attrs=True, item_attrs=True, tag_attributes=None):
    # Load items and annotation attributes
    _items = d.get("items", None)
    items = []
    if _items is not None:
        annotations = defaultdict(list)
        item_attributes = defaultdict(list)
        for i in _items:
            items.append(i["id"])
            if i["annotations"] is not None:
                for a in i["annotations"]:
                    annotations[i["id"]].append(DatumaroObject.from_anno_dict(a, ann_attrs=ann_attrs, tag_attributes=tag_attributes))
            if i["attr"] is not None:
                if item_attrs is True:
                    item_attributes[i["id"]].append(i["attr"])
                else:
                    item_attributes[i["id"]] = {}
                
                if etau.is_str(item_attrs):
                    item_attrs = [item_attrs]

                if isinstance(item_attrs, list):
                    item_attributes[i["id"]].append({f: i["attr"].get(f, None) for f in item_attrs})       

        if not len(annotations) == 0:
            annotations = dict(annotations)
        else:
            annotations = None
        if not len(item_attributes) == 0:
            image_attributes = dict(image_attributes)
        else:
            image_attributes = None
    else:
        annotations = None
        item_attributes = None

    return items, item_attributes, annotations


def parse_datumaro_label_categories(labels):
    """Parses the Datumaro categories labels list.

    Args:
        labels: a list of dict of the form::
            [
                ...
                {
                    "name": "",
                    "parent": "",
                    "attributes": []
                },
                ...
            ]

    Returns:
        a tuple of
        -   classes_map: a dict mapping label IDs to labels
    """
    classes_map = {
        pos: label["name"] for pos, label in enumerate(labels)
    }

    return classes_map


def _parse_label_types(label_types):
    if label_types is None:
        return _SUPPORTED_LABEL_TYPES

    if etau.is_str(label_types):
        label_types = [label_types]
    else:
        label_types = list(label_types)

    bad_types = [l for l in label_types if l not in _SUPPORTED_LABEL_TYPES]

    if len(bad_types) == 1:
        raise ValueError(
            "Unsupported label type '%s'. Supported types are %s"
            % (bad_types[0], _SUPPORTED_LABEL_TYPES)
        )

    if len(bad_types) > 1:
        raise ValueError(
            "Unsupported label types %s. Supported types are %s"
            % (bad_types, _SUPPORTED_LABEL_TYPES)
        )

    return label_types


def _get_matching_item_ids(
    classes_map,
    items,
    annotations,
    item_ids=None,
    classes=None,
    shuffle=False,
    seed=None,
    max_samples=None,
):
    if item_ids is not None:
        item_ids = _parse_item_ids(item_ids, items)
    else:
        item_ids = items

    if classes is not None:
        all_ids, any_ids = _get_items_with_classes(
            item_ids, annotations, classes, classes_map
        )
    else:
        all_ids = item_ids
        any_ids = []

    all_ids = sorted(all_ids)
    any_ids = sorted(any_ids)

    if shuffle:
        if seed is not None:
            random.seed(seed)

        random.shuffle(all_ids)
        random.shuffle(any_ids)

    item_ids = all_ids + any_ids

    if max_samples is not None:
        return item_ids[:max_samples]

    return item_ids


def _get_items_with_classes(
    item_ids, annotations, target_classes, classes_map
):
    if annotations is None:
        logger.warning("Dataset is unlabeled; ignoring classes requirement")
        return item_ids, []

    if etau.is_str(target_classes):
        target_classes = [target_classes]

    labels_map_rev = {c: i for i, c in classes_map.items()}

    bad_classes = [c for c in target_classes if c not in labels_map_rev]
    if bad_classes:
        raise ValueError("Unsupported classes: %s" % bad_classes)

    class_ids = {labels_map_rev[c] for c in target_classes}

    all_ids = []
    any_ids = []
    for item_id in item_ids:
        coco_objects = annotations.get(item_id, None)
        if not coco_objects:
            continue

        oids = set(o.label_id for o in coco_objects)
        if class_ids.issubset(oids):
            all_ids.append(item_id)
        elif class_ids & oids:
            any_ids.append(item_id)

    return all_ids, any_ids


def _parse_item_ids(raw_item_ids, items, split=None):
    # Load IDs from file
    if etau.is_str(raw_item_ids):
        item_ids_path = raw_item_ids
        ext = os.path.splitext(item_ids_path)[-1]
        if ext == ".txt":
            raw_image_ids = _load_item_ids_txt(item_ids_path)
        elif ext == ".json":
            raw_image_ids = _load_item_ids_json(item_ids_path)
        elif ext == ".csv":
            raw_image_ids = _load_item_ids_csv(item_ids_path)
        else:
            raise ValueError(
                "Invalid item ID file '%s'. Supported formats are .txt, "
                ".csv, and .json" % ext
            )

    item_ids = []
    for raw_id in raw_item_ids:
        if etau.is_str(raw_id):
            if "/" in raw_id:
                _split, raw_id = raw_id.split("/")
                if split and _split != split:
                    continue

            raw_id = int(raw_id.strip())

        item_ids.append(raw_id)

    # Validate that IDs exist
    invalid_ids = [_id for _id in item_ids if _id not in items]
    if invalid_ids:
        raise ValueError(
            "Found %d invalid IDs, ex: %s" % (len(invalid_ids), invalid_ids[0])
        )

    return item_ids


def _load_item_ids_txt(txt_path):
    with open(txt_path, "r") as f:
        return [l.strip() for l in f.readlines()]


def _load_item_ids_csv(csv_path):
    with open(csv_path, "r", newline="") as f:
        dialect = csv.Sniffer().sniff(f.read(10240))
        f.seek(0)
        if dialect.delimiter in _CSV_DELIMITERS:
            reader = csv.reader(f, dialect)
        else:
            reader = csv.reader(f)

        item_ids = [row for row in reader]

    if isinstance(item_ids[0], list):
        # Flatten list
        item_ids = [_id for ids in item_ids for _id in ids]

    return item_ids


def _load_item_ids_json(json_path):
    return [_id for _id in etas.load_json(json_path)]


def _to_labels_map_rev(classes):
    return {c: i for i, c in enumerate(classes, 1)}


def _to_classes(classes_map):
    return [classes_map[i] for i in sorted(classes_map.keys())]


def _get_class_ids(classes, classes_map):
    if etau.is_str(classes):
        classes = [classes]

    labels_map_rev = {c: i for i, c in classes_map.items()}
    class_ids = {labels_map_rev[c] for c in classes}

    return class_ids


def _get_matching_objects(datumaro_objects, class_ids):
    return [obj for obj in datumaro_objects if obj.label_id in class_ids]


def _parse_label_categories(label_categories, classes=None):
    classes_map, _ = parse_datumaro_label_categories(label_categories)

    if classes is None:
        return {c: i for i, c in classes_map.items()}

    if etau.is_str(classes):
        classes = {classes}
    else:
        classes = set(classes)

    return {c: i for i, c in classes_map.items() if c in classes}


def _datumaro_objects_to_polylines(
    datumaro_objects,
    frame_size,
    classes_map,
    tolerance,
    include_id,
):
    polylines = []
    for datumaro_object in datumaro_objects:
        polyline = datumaro_object.to_polyline(
            frame_size,
            classes_map=classes_map,
            tolerance=tolerance,
            include_id=include_id,
        )

        if polyline is not None:
            polylines.append(polyline)

    if not polylines:
        return None

    return fol.Polylines(polylines=polylines)


def _datumaro_objects_to_detections(
    datumaro_objects,
    frame_size,
    classes_map,
    load_segmentations,
    include_id
):
    detections = []
    for datumaro_obj in datumaro_objects:
        detection = datumaro_obj.to_detection(
            frame_size,
            classes_map=classes_map,
            include_id=include_id
        )

        if detection is not None and (
            not load_segmentations or detection.has_mask
        ):
            detections.append(detection)

    if not detections:
        return None

    return fol.Detections(detections=detections)


def _datumaro_objects_to_keypoints(
    datumaro_objects,
    frame_size,
    classes_map,
    include_id,
):
    keypoints = []
    for datumaro_object in datumaro_objects:
        keypoint = datumaro_object.to_keypoints(
            frame_size,
            classes_map=classes_map,
            include_id=include_id,
        )

        if keypoint is not None:
            keypoints.append(keypoint)

    if not keypoints:
        return None

    return fol.Keypoints(keypoints=keypoints)


def _datumaro_objects_to_classifications(
    datumaro_objects,
    classes_map,
    include_id,
):
    classifications = []
    for datumaro_object in datumaro_objects:
        classification = datumaro_object.to_classification(
            classes_map=classes_map,
            include_id=include_id,
        )

        if classification is not None:
            classifications.append(classification)

    if not classifications:
        return None

    return fol.Classifications(classifications=classifications)


def _get_attributes(label, extra_attrs):
    if extra_attrs == True:
        return dict(label.iter_attributes())

    if extra_attrs == False:
        return {}

    if etau.is_str(extra_attrs):
        extra_attrs = [extra_attrs]

    return {
        name: label.get_attribute_value(name, None) for name in extra_attrs
    }


#
# The methods below are taken, in part, from:
# https://github.com/waspinator/pycococreator/blob/207b4fa8bbaae22ebcdeb3bbf00b724498e026a7/pycococreatortools/pycococreatortools.py
#


def _get_polygons_for_segmentation(segmentation, frame_size, tolerance):
    width, height = frame_size

    # Convert to [[x1, y1, x2, y2, ...]] polygons
    if isinstance(segmentation, list):
        abs_points = segmentation
    else:
        if isinstance(segmentation["counts"], list):
            # Uncompressed RLE
            rle = mask_utils.frPyObjects(segmentation, height, width)
        else:
            # RLE
            rle = segmentation

        mask = mask_utils.decode(rle)
        abs_points = _mask_to_polygons(mask, tolerance)

    # Convert to [[(x1, y1), (x2, y2), ...]] in relative coordinates

    rel_points = []
    for apoints in abs_points:
        rel_points.append(
            [(x / width, y / height) for x, y, in _pairwise(apoints)]
        )

    return rel_points


def _pairwise(x):
    y = iter(x)
    return zip(y, y)


def _datumaro_segmentation_to_mask(segmentation, bbox, frame_size):
    x, y, w, h = bbox
    width, height = frame_size

    if isinstance(segmentation, list):
        # Polygon -- a single object might consist of multiple parts, so merge
        # all parts into one mask RLE code
        segmentation = _normalize_coco_segmentation(segmentation)
        if len(segmentation) == 0:
            return None

        rle = mask_utils.merge(
            mask_utils.frPyObjects(segmentation, height, width)
        )
    elif isinstance(segmentation["counts"], list):
        # Uncompressed RLE
        rle = mask_utils.frPyObjects(segmentation, height, width)
    else:
        # RLE
        rle = segmentation

    mask = mask_utils.decode(rle).astype(bool)

    return mask[
        int(round(y)) : int(round(y + h)),
        int(round(x)) : int(round(x + w)),
    ]


def _normalize_coco_segmentation(segmentation):
    # Filter out empty segmentations
    # For polygons of 4 points (1 pixel), duplicate to convert to valid polygon
    _segmentation = []
    for seg in segmentation:
        if len(seg) == 0:
            continue

        if len(seg) == 4:
            seg *= 4

        _segmentation.append(seg)

    return _segmentation


def _polyline_to_datumaro_segmentation(polyline, frame_size):

    seg = polyline.to_segmentation(frame_size=frame_size, target=1)
    return _mask_to_rle(seg.mask)


def _instance_to_datumaro_segmentation(
    detection, frame_size
):
    dobj = foue.to_detected_object(detection, extra_attrs=False)

    try:
        mask = etai.render_instance_image(
            dobj.mask, dobj.bounding_box, frame_size
        )
    except:
        # Either mask or bounding box is too small to render
        width, height = frame_size
        mask = np.zeros((height, width), dtype=bool)

    return _mask_to_rle(mask)


def _mask_to_rle(mask):
    counts = []
    for i, (value, elements) in enumerate(groupby(mask.ravel(order="F"))):
        if i == 0 and value == 1:
            counts.append(0)

        counts.append(len(list(elements)))

    return {"counts": counts, "size": list(mask.shape)}


def _mask_to_polygons(mask, tolerance):
    if tolerance is None:
        tolerance = 2

    # Pad mask to close contours of shapes which start and end at an edge
    padded_mask = np.pad(mask, pad_width=1, mode="constant", constant_values=0)

    contours = measure.find_contours(padded_mask, 0.5)
    contours = [c - 1 for c in contours]  # undo padding

    polygons = []
    for contour in contours:
        contour = _close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue

        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()

        # After padding and subtracting 1 there may be -0.5 points
        segmentation = [0 if i < 0 else i for i in segmentation]

        polygons.append(segmentation)

    return polygons


def _close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))

    return contour


_SUPPORTED_LABEL_TYPES = ["classifications", "detections", "segmentations", "keypoints"]


_CSV_DELIMITERS = [",", ";", ":", " ", "\t", "\n"]
