use faiss::cluster::{Clustering, ClusteringParameters};
use faiss::{FlatIndex, Index};
use h5::{Dataset, File};
use h5::types::VarLenUnicode;
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{s, Array1, Array2, ArrayView2, Axis};
use std::path::PathBuf;
use structopt::StructOpt;

type DynResult<T> = Result<T, Box<dyn std::error::Error>>;

#[derive(Debug, StructOpt)]
enum App {
    /// Generate a feature vocabulary
    #[structopt(name = "vocabulary")]
    Vocabulary(VocabularyArgs),
    /// Generate bags of features
    #[structopt(name = "quantize", alias = "bows")]
    Quantize(QuantizeArgs),
}

#[derive(Debug, StructOpt)]
pub struct VocabularyArgs {
    /// The hdf5 file containing the features
    #[structopt(name = "FEATURES", parse(from_os_str))]
    features: PathBuf,
    /// Group path where the features are
    #[structopt(long = "name", default_value = "data")]
    dataset_name: String,
    /// The size of the codebook
    #[structopt(short = "k", long = "size")]
    size: u32,
    /// The hdf5 file to store the k centroids
    #[structopt(
        short = "o",
        long = "out",
        parse(from_os_str),
        default_value = "vocabulary.h5"
    )]
    out: PathBuf,
    /// Only use `n` features for clustering
    #[structopt(short = "N")]
    n: Option<usize>,
    /// Number of k-means clustering iterations
    #[structopt(long = "niter")]
    niter: Option<u32>,
}

#[derive(Debug, StructOpt)]
pub struct QuantizeArgs {
    /// The hdf5 file containing the codebook
    #[structopt(name = "VOCABULARY", parse(from_os_str))]
    vocabulary: PathBuf,
    /// The hdf5 file containing the features
    #[structopt(name = "FEATURES", parse(from_os_str))]
    features: PathBuf,
    /// Group path where the features are
    #[structopt(long = "name", default_value = "data")]
    features_dataset_name: String,
    /// Group path where the item IDs are defined for each feature
    #[structopt(long = "item_id", alias = "id_slice", default_value = "item_id")]
    item_id: String,
    /// Group path where the names (or textual IDs) are defined for each item
    #[structopt(long = "item_name", default_value = "id_volume")]
    item_name: String,
    /// Features file represents a single item (don't read item_id nor item_name)
    #[structopt(long = "single_item", alias = "single_volume")]
    single_item: bool,
    /// The hdf5 file to store the bags
    #[structopt(
        short = "o",
        long = "out",
        parse(from_os_str),
        default_value = "bows.h5"
    )]
    out: PathBuf,
}

fn main() -> DynResult<()> {
    match App::from_args() {
        App::Vocabulary(args) => generate_vocabulary(args)?,
        App::Quantize(args) => generate_descriptors(args)?,
    }

    Ok(())
}

fn generate_vocabulary(args: VocabularyArgs) -> DynResult<()> {
    let file = File::open(args.features, "r")?;

    let data = file.dataset(&args.dataset_name)?;

    let k = args.size;

    let progress = ProgressBar::new_spinner();
    progress.set_message("Loading features to memory...");
    progress.enable_steady_tick(100);

    let features: Array2<f32> = if let Some(n) = args.n {
        data.read_slice_2d(s![0..n, ..])?
    } else {
        data.read_2d()?
    };
    let d = features.shape()[1] as u32;
    let mut params = ClusteringParameters::new();
    if let Some(niter) = args.niter {
        params.set_niter(niter);
    }
    let mut cluster = Clustering::new_with_params(d, k, &params)?;
    let mut index = FlatIndex::new_l2(d)?;

    progress.set_message(&format!(
        "Clustering {} descriptors into {} components ...",
        features.shape()[0],
        k
    ));
    progress.enable_steady_tick(300);

    cluster.train(
        features
            .as_slice()
            .expect("array must be in standard order"),
        &mut index,
    )?;

    println!(
        "Done. Final objective loss: {}",
        cluster
            .objectives()?
            .last()
            .cloned()
            .unwrap_or(std::f32::INFINITY)
    );
    println!("Saving centroids to {} ...", args.out.display());

    let vocabulary_shape = (k as usize, d as usize);

    let file = File::with_options().mode("w").open(&args.out)?;
    let data = file
        .new_dataset::<f32>()
        .no_chunk()
        .create("data", vocabulary_shape)?;

    let centroids: ArrayView2<f32> = ArrayView2::from_shape(vocabulary_shape, index.xb())?;

    data.write(centroids)?;

    Ok(())
}

fn generate_descriptors(args: QuantizeArgs) -> DynResult<()> {
    let progress = ProgressBar::new_spinner();

    progress.set_message("Reading data ...");
    let codebook: Array2<f32> = {
        let file = File::open(args.vocabulary, "r")?;
        let vocabulary_dset = file.dataset("data")?;
        vocabulary_dset.read_2d()?
    };
    let d = codebook.shape()[1] as u32;
    let mut index = FlatIndex::new_l2(d)?;
    index.add(
        codebook
            .as_slice()
            .expect("codebook should be in standard layout"),
    )?;

    let file = File::open(args.features, "r")?;
    let features_dset = file.dataset(&args.features_dataset_name)?;

    let bows: Array2<_> = if args.single_item {
        drop(progress);

        let progress = ProgressBar::new(features_dset.shape()[0] as u64);
        progress.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:50} {pos:>7}/{len:7} {msg}"),
        );
        progress.set_message("Building bags ...");
        let bows = construct_bows_one(&features_dset, &mut index, |n| {
            progress.inc(u64::from(n));
        })?;

        bows.insert_axis(Axis(0))
    } else {
        let id_slice_dset = file.dataset(&args.item_id)?;

        // peek at item_name to identify the number of items
        let n_items = {
            let id_item_dset = file.dataset(&args.item_name)?;
            id_item_dset.shape()[0]
        };

        drop(progress);

        let progress = ProgressBar::new(id_slice_dset.shape()[0] as u64);
        progress.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:50} {pos:>7}/{len:7} {msg}"),
        );
        progress.set_message("Building bags ...");
        construct_bows(&features_dset, &id_slice_dset, n_items, &mut index, |n| {
            progress.inc(u64::from(n));
        })?
    };
    // save them
    let progress = ProgressBar::new_spinner();
    progress.set_message("Saving to file ...");

    let out = File::open(&args.out, "w")?;
    let bows_dset = out
        .new_dataset::<f32>()
        .no_chunk()
        .create("data", bows.dim())?;
    bows_dset.write(bows.view())?;

    let n_items = bows_dset.shape()[0];

    if !args.single_item {
        // write sequential range to `id_slice`
        let id_slice_dset_out = out
            .new_dataset::<u32>()
            .no_chunk()
            .create(&args.item_id, (n_items,))?;
        id_slice_dset_out.write_raw(&(0..n_items).collect::<Vec<_>>())?;

        // replicate `id_item` to the output file
        let id_item_dset_in = file.dataset(&args.item_name)?;
        let id_item_in: Vec<VarLenUnicode> = id_item_dset_in.read_raw()?;
        let id_item_dset_out = out
            .new_dataset::<VarLenUnicode>()
            .no_chunk()
            .create(&args.item_name, id_item_dset_in.shape())?;
        id_item_dset_out.write_raw(&id_item_in)?;
    }

    progress.finish_with_message(&format!("Bags saved: {}", args.out.display()));
    Ok(())
}

fn batched_1d<'a, T>(dset: &'a Dataset, batch_size: usize) -> impl Iterator<Item = Array1<T>> + 'a
where
    T: h5::H5Type,
{
    let batch_offset = dset.shape()[0] % batch_size;
    let nbatches = dset.shape()[0] / batch_size + if batch_offset > 0 { 1 } else { 0 };

    (0..nbatches).map(move |i| {
        let begin = i * batch_size;
        let end = usize::min(begin + batch_size, dset.shape()[0]);
        dset.read_slice_1d::<T, _>(s![begin..end])
            .expect("out of range")
    })
}

fn batched_2d<'a, T>(dset: &'a Dataset, batch_size: usize) -> impl Iterator<Item = Array2<T>> + 'a
where
    T: h5::H5Type,
{
    let total = dset.shape()[0];
    let batch_offset = total % batch_size;
    let nbatches = total / batch_size + if batch_offset > 0 { 1 } else { 0 };

    (0..nbatches).map(move |i| {
        let begin = i * batch_size;
        let end = usize::min(begin + batch_size, total);
        dset.read_slice_2d::<T, _>(s![begin..end, ..])
            .expect("out of range")
    })
}

fn construct_bows_one<F>(
    features_dset: &Dataset,
    index: &mut Index,
    tick_fn: F,
) -> DynResult<Array1<u32>>
where
    F: Fn(u32),
{
    let batch_size = 1024;
    let mut bows = Array1::<u32>::zeros([index.ntotal() as usize]);
    for feature_batch in batched_2d::<f32>(&features_dset, batch_size) {
        let b_size = feature_batch.shape()[0];
        let nearest = index.assign(
            feature_batch
                .as_slice()
                .expect("features should be in standard layout"),
            1,
        )?;
        for b in nearest.labels.into_iter() {
            if b >= 0 {
                *bows
                    .get_mut([b as usize])
                    .unwrap_or_else(|| panic!("invalid BoW index ({})", b)) += 1_u32;
            }
        }

        tick_fn(b_size as u32);
    }
    Ok(bows)
}

fn construct_bows<F>(
    features_dset: &Dataset,
    id_slice_dset: &Dataset,
    n_items: usize,
    index: &mut Index,
    tick_fn: F,
) -> DynResult<Array2<u32>>
where
    F: Fn(u32),
{
    let batch_size = 1024;
    let mut bows = Array2::<u32>::zeros([n_items, index.ntotal() as usize]);
    for (feature_batch, item_batch) in Iterator::zip(
        batched_2d::<f32>(&features_dset, batch_size),
        batched_1d::<u32>(&id_slice_dset, batch_size),
    ) {
        let b_size = feature_batch.shape()[0];
        // build bows
        let nearest = index.assign(
            feature_batch
                .as_slice()
                .expect("features should be in standard layout"),
            1,
        )?;
        for (b, vol_id) in Iterator::zip(nearest.labels.into_iter(), item_batch.into_iter()) {
            if b >= 0 {
                *bows
                    .get_mut((*vol_id as usize, b as usize))
                    .unwrap_or_else(|| panic!("invalid BoW index ({}, {})", *vol_id, b)) += 1_u32;
            }
        }

        tick_fn(b_size as u32);
    }
    Ok(bows)
}
