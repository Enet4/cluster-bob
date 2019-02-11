# Cluster Bob
[![Latest Version](https://img.shields.io/crates/v/cluster-bob.svg)](https://crates.io/crates/cluster-bob) [![Build Status](https://travis-ci.org/Enet4/cluster-bob.svg?branch=master)](https://travis-ci.org/Enet4/cluster-bob) [![dependency status](https://deps.rs/repo/github/Enet4/cluster-bob/status.svg)](https://deps.rs/repo/github/Enet4/cluster-bob)

A framework for dense vector clustering and quantization into bags of features.

## About

Cluster Bob is a CLI tool for generating feature vocabularies and quantizing feature vectors into bags of features. It was originally intended for creating image descriptors in content-based image retrieval.

## Using

Please see the [Data Format](#input-data-format) section in order to make data sets which comply with this tool.

```
USAGE:
    cluster-bob <SUBCOMMAND>

FLAGS:
    -h, --help       Prints help information
    -V, --version    Prints version information

SUBCOMMANDS:
    help          Prints this message or the help of the given subcommand(s)
    quantize      Generate bags of features
    vocabulary    Generate a feature vocabulary
```

### Vocabulary

The first step is to generate a vocabulary (or codebook), with a clustering algorithm. For example, to use 5000 features from "dataset.h5" to create a vocabulary of 256 components:

```
cluster-bob vocabulary dataset.h5 -N 5000 -k 256 -o codebook.h5
```

### Generating Bags

Afterwards, we can create a descriptor using the bags of features algorithm. Each feature is tested against a previously established codebook by determining the nearest component (by the L2-norm Euclidean distance) to that feature vector.

## Data format

All inputs and outputs are HDF5 files. By default, the features should be available in the `/data` group as a two-dimensional HDF5 data set of floating point numbers, with the shape `NxD`, where _N_ is the number of features and _D_ is their dimensionality.

Moreover, each feature is assigned to an _item_. An item can have multiple features. In order to specify this, two additional one-dimensional data sets are specified:

- `/item_name` is a Unicode string data set containing as many elements as the number of items.
- `/item_id` should have the exact length _N_, and contains unsigned integers for mapping each feature in `/data` to an item defined in `/item_name`. The feature of index `i` in `data` is mapped to an item in `item_name` by `item_id[i]`.

In most cases, the names of these data sets are configurable via CLI options.

If the `single_item` flag is enabled, it is assumed that all features in the data set belong to the same data set, in which case both `/item_name` and `/item_id` are no longer required.

## License

Licensed under either of

* Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
