
use std::collections::HashMap;
use std::thread::current;
use serde::Serialize;
use serde::Deserialize;

mod readfromfile;

use tch::Tensor;
use tch::Kind;
use tch::nn::*;
use tch::nn;

mod uservectormodel;

mod ratingprediction;

mod mds;

pub use mds::majorize;

mod differencemajorize;

pub use differencemajorize::difference_majorize;

pub use mds::differencemdsmodel;