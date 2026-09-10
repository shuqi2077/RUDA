use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use globwalk::DirEntry;

use super::{
    AnnotationRaw, ImageDatasetItem, ImageDatasetItemRaw, ImageFolderDataset, ImageLoaderError,
    PathToImageDatasetItem,
};
use crate::transform::MapperDataset;
use crate::{Dataset, InMemDataset};

const SUPPORTED_FILES: [&str; 4] = ["bmp", "jpg", "jpeg", "png"];

impl Dataset<ImageDatasetItem> for ImageFolderDataset {
    fn get(&self, index: usize) -> Option<ImageDatasetItem> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl ImageFolderDataset {
    /// Create an image classification dataset from the root folder.
    ///
    /// # Arguments
    ///
    /// * `root` - Dataset root folder.
    ///
    /// # Returns
    /// A new dataset instance.
    pub fn new_classification<P: AsRef<Path>>(root: P) -> Result<Self, ImageLoaderError> {
        // New dataset containing any of the supported file types
        ImageFolderDataset::new_classification_with(root, &SUPPORTED_FILES)
    }

    /// Create an image classification dataset from the root folder.
    /// The included images are filtered based on the provided extensions.
    ///
    /// # Arguments
    ///
    /// * `root` - Dataset root folder.
    /// * `extensions` - List of allowed extensions.
    ///
    /// # Returns
    /// A new dataset instance.
    pub fn new_classification_with<P, S>(
        root: P,
        extensions: &[S],
    ) -> Result<Self, ImageLoaderError>
    where
        P: AsRef<Path>,
        S: AsRef<str>,
    {
        // Glob all images with extensions
        let walker = globwalk::GlobWalkerBuilder::from_patterns(
            root.as_ref(),
            &[format!(
                "*.{{{}}}", // "*.{ext1,ext2,ext3}
                extensions
                    .iter()
                    .map(Self::check_extension)
                    .collect::<Result<Vec<_>, _>>()?
                    .join(",")
            )],
        )
        .follow_links(true)
        .sort_by(|p1: &DirEntry, p2: &DirEntry| p1.path().cmp(p2.path())) // order by path
        .build()
        .map_err(|err| ImageLoaderError::Unknown(format!("{err:?}")))?
        .filter_map(Result::ok);

        // Get all dataset items
        let mut items = Vec::new();
        let mut classes = HashSet::new();
        for img in walker {
            let image_path = img.path();

            // Label name is represented by the parent folder name
            let label = image_path
                .parent()
                .ok_or_else(|| {
                    ImageLoaderError::IOError("Could not resolve image parent folder".to_string())
                })?
                .file_name()
                .ok_or_else(|| {
                    ImageLoaderError::IOError(
                        "Could not resolve image parent folder name".to_string(),
                    )
                })?
                .to_string_lossy()
                .into_owned();

            classes.insert(label.clone());

            items.push(ImageDatasetItemRaw::new(
                image_path,
                AnnotationRaw::Label(label),
            ))
        }

        // Sort class names
        let mut classes = classes.into_iter().collect::<Vec<_>>();
        classes.sort();

        Self::with_items(items, &classes)
    }

    /// Create an image classification dataset with the specified items.
    ///
    /// # Arguments
    ///
    /// * `items` - List of dataset items, each item represented by a tuple `(image path, label)`.
    /// * `classes` - Dataset class names.
    ///
    /// # Returns
    /// A new dataset instance.
    pub fn new_classification_with_items<P: AsRef<Path>, S: AsRef<str>>(
        items: Vec<(P, String)>,
        classes: &[S],
    ) -> Result<Self, ImageLoaderError> {
        // Parse items and check valid image extension types
        let items = items
            .into_iter()
            .map(|(path, label)| {
                // Map image path and label
                let path = path.as_ref();
                let label = AnnotationRaw::Label(label);

                Self::check_extension(&path.extension().unwrap().to_str().unwrap())?;

                Ok(ImageDatasetItemRaw::new(path, label))
            })
            .collect::<Result<Vec<_>, _>>()?;

        Self::with_items(items, classes)
    }

    /// Create a multi-label image classification dataset with the specified items.
    ///
    /// # Arguments
    ///
    /// * `items` - List of dataset items, each item represented by a tuple `(image path, labels)`.
    /// * `classes` - Dataset class names.
    ///
    /// # Returns
    /// A new dataset instance.
    pub fn new_multilabel_classification_with_items<P: AsRef<Path>, S: AsRef<str>>(
        items: Vec<(P, Vec<String>)>,
        classes: &[S],
    ) -> Result<Self, ImageLoaderError> {
        // Parse items and check valid image extension types
        let items = items
            .into_iter()
            .map(|(path, labels)| {
                // Map image path and multi-label
                let path = path.as_ref();
                let labels = AnnotationRaw::MultiLabel(labels);

                Self::check_extension(&path.extension().unwrap().to_str().unwrap())?;

                Ok(ImageDatasetItemRaw::new(path, labels))
            })
            .collect::<Result<Vec<_>, _>>()?;

        Self::with_items(items, classes)
    }

    /// Create an image segmentation dataset with the specified items.
    ///
    /// # Arguments
    ///
    /// * `items` - List of dataset items, each item represented by a tuple `(image path, annotation path)`.
    /// * `classes` - Dataset class names.
    ///
    /// # Returns
    /// A new dataset instance.
    pub fn new_segmentation_with_items<P: AsRef<Path>, S: AsRef<str>>(
        items: Vec<(P, P)>,
        classes: &[S],
    ) -> Result<Self, ImageLoaderError> {
        // Parse items and check valid image extension types
        let items = items
            .into_iter()
            .map(|(image_path, mask_path)| {
                // Map image path and segmentation mask path
                let image_path = image_path.as_ref();
                let annotation = AnnotationRaw::SegmentationMask(mask_path.as_ref().to_path_buf());

                Self::check_extension(&image_path.extension().unwrap().to_str().unwrap())?;

                Ok(ImageDatasetItemRaw::new(image_path, annotation))
            })
            .collect::<Result<Vec<_>, _>>()?;

        Self::with_items(items, classes)
    }

    /// Create an image dataset with the specified items.
    ///
    /// # Arguments
    ///
    /// * `items` - Raw dataset items.
    /// * `classes` - Dataset class names.
    ///
    /// # Returns
    /// A new dataset instance.
    fn with_items<S: AsRef<str>>(
        items: Vec<ImageDatasetItemRaw>,
        classes: &[S],
    ) -> Result<Self, ImageLoaderError> {
        // NOTE: right now we don't need to validate the supported image files since
        // the method is private. We assume it's already validated.
        let dataset = InMemDataset::new(items);

        // Class names to index map
        let classes = classes.iter().map(|c| c.as_ref()).collect::<Vec<_>>();
        let classes_map: HashMap<_, _> = classes
            .into_iter()
            .enumerate()
            .map(|(idx, cls)| (cls.to_string(), idx))
            .collect();

        let mapper = PathToImageDatasetItem {
            classes: classes_map,
        };
        let dataset = MapperDataset::new(dataset, mapper);

        Ok(Self { dataset })
    }

    /// Check if extension is supported.
    fn check_extension<S: AsRef<str>>(extension: &S) -> Result<String, ImageLoaderError> {
        let extension = extension.as_ref();
        if !SUPPORTED_FILES.contains(&extension) {
            Err(ImageLoaderError::InvalidFileExtensionError(
                extension.to_string(),
            ))
        } else {
            Ok(extension.to_string())
        }
    }
}
