
use tch::Device;

use std::collections::HashMap;
use std::thread::current;
use serde::Serialize;
use serde::Deserialize;


use tch::Tensor;


use tch::Kind;

use tch::nn::*;
use tch::nn;


pub struct MDSModel{

    layers: Sequential,

    opt: Optimizer,
}

impl MDSModel{

    pub fn new( inputsize: i64, outputsize: i64, ) -> MDSModel{

        let vs = nn::VarStore::new(Device::Cpu);

        let mut opt = nn::Adam::default().build(&vs, 1e-4).unwrap();

        let vs = vs.root();


        println!("F");

        let mut layers = nn::seq()
        .add(nn::linear(
            &vs,
            inputsize,
            10,
            Default::default(),
        ))
        .add(nn::linear(&vs, 10, 7, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(&vs, 7, 5, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(&vs, 5, outputsize, Default::default()));
        // // .add_fn(|xs| xs.relu())
        // .add(nn::linear(&vs, 10, 10, Default::default()))
        // .add_fn(|xs| xs.relu())
        // .add(nn::linear(&vs, 10, outputsize, Default::default()));

        println!("G");

        MDSModel {
            layers: layers,
            opt: opt        
        }

    }



    // pub fn set_lr(&mut self, lr: f64){
    //     self.opt.set_lr( lr );
    // }

    pub fn tensor_forward(&self, tensor: &Tensor) -> Tensor{

        let prediction = self.layers.forward(tensor);

        return prediction;
    }

    pub fn forward(&self, inputvector: Vec<f64>) -> Tensor{

        //println!("C");
        let user_input = Tensor::f_of_slice( &inputvector ).unwrap().to_kind( Kind::Float );

        let prediction = self.layers.forward(&user_input);

        //println!("D");
        return prediction;
    }







    //user, item, rating
    pub fn train_step(&mut self, input: Vec<f64>, target: Vec<f64> ) {

        //println!("A");
        let differencetensor = self.forward( input );

        let label = Tensor::f_of_slice( &target ).unwrap().to_kind( Kind::Float );

        let loss = differencetensor.f_mse_loss( &label, tch::Reduction::Mean ).unwrap();

        self.opt.backward_step( &loss );

        //println!("B");
    }



}




// pub fn normalize<T>( ratings: Vec<(T, f64)>) ->  Vec<(T, f64)>{

//     if ratings.len() == 0{
//         return Vec::new();
//     }


//     let totalratings = ratings.len();

//     let mut ratingsorder = Vec::new();

//     for (index, rating) in ratings{
//         ratingsorder.push(  (index, rating) );
//     }
    
//     ratingsorder.sort_by(|(_, ratinga), (_, ratingb)|   ratingb.partial_cmp( ratinga ).unwrap() );
    


//     let mut ratingsgroups: Vec<Vec<T>> = Vec::new();

//     let mut currentgrouprating = ratingsorder[0].1;

//     let mut currentgroup = Vec::new();

//     for (movie, rating) in ratingsorder{

//         currentgroup.push( movie );

//         if rating != currentgrouprating{
//             currentgrouprating = rating;

//             let temp = currentgroup;
//             currentgroup = Vec::new();
//             ratingsgroups.push( temp );
//         }

//     }

//     //this is sorted from highest group to lowest group
//     ratingsgroups.push(  currentgroup );


//     //println!("Ratings order{:?}", ratingsgroups);
//     let mut toreturn: Vec<(T, f64)> = Vec::new();

//     let mut currentscore = 0.5;


//     for ratingsgroup in ratingsgroups{

//         //how many total ratings are there
//         //from the range of -0.5 to 0.5
//         let ratingsgroupsize = ratingsgroup.len();

//         //how much of the -1.0 to 1.0 does this take up
//         let sizepercentage = ratingsgroupsize as f64 / totalratings as f64;

//         let grouprating = currentscore - (sizepercentage / 2.0);

//         for index in ratingsgroup{
//             toreturn.push( (index, (grouprating ) * 10.0 ) );
//         }

//         currentscore = currentscore - sizepercentage;

//     }


//     /*
//     let mut totalrating = 0.0;
//     for (_, rating) in &toreturn{
//         totalrating += rating;
//     }
//     println!("average rating {:?}", totalrating / toreturn.len() as f64);
//     This correctly prints "0.00000124123"
//     */


//     return toreturn;
// }

