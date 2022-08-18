use tch::Device;
use std::collections::HashMap;
use std::thread::current;
use serde::Serialize;
use serde::Deserialize;
use tch::Tensor;
use tch::Kind;
use tch::nn::*;
use tch::nn;


pub struct DifferenceMDSModel{

    embeddings: Embedding,

    itemcount: i64,

    layers: Sequential,

    //mdsmodel: super::MDSModel,

    embeddingsfrozen: bool,


    layeropt: Optimizer,
    layervs: VarStore,
    
    embeddingopt: Optimizer,
    embeddingvs: VarStore,

}

impl DifferenceMDSModel{

    pub fn new(itemcount: i64, internalembeddingsize: i64, embeddingsize: i64) -> DifferenceMDSModel{

        //let mut opt = nn::Adam::default().build(&vs, 1e-3).unwrap();

        let layervs = nn::VarStore::new(Device::Cpu);
        let layeropt = nn::Adam::default().build(&layervs, 1e-4).unwrap();

        
        let layers = nn::seq()
        .add(nn::linear(
            &layervs.root(),
            internalembeddingsize,
            5,
            Default::default(),
        ))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(&layervs.root(), 5, embeddingsize, Default::default()));
        // .add_fn(|xs| xs.relu())
        //.add(nn::linear(&layervs.root(), 7, embeddingsize, Default::default()));
        // .add_fn(|xs| xs.relu())
        // .add(nn::linear(&vs, 10, embeddingsize, Default::default()));


        let embeddingvs = nn::VarStore::new(Device::Cpu);
        let embeddingopt = nn::Adam::default().build(&embeddingvs, 1e-3).unwrap();


        let config = EmbeddingConfig{
            //sparse: true,
            ..Default::default()
        };
        let embeddings = embedding( &embeddingvs.root(), itemcount, internalembeddingsize, config );

        println!("made");

        DifferenceMDSModel {
            embeddingvs,
            embeddingopt,
            layervs,
            layeropt,

            itemcount,
            layers,
            embeddings,

            embeddingsfrozen: false,
        }

    }


    pub fn toggle_frozen_layer(&mut self){
        
        self.embeddingsfrozen = !self.embeddingsfrozen;

        if self.embeddingsfrozen{
            self.embeddingvs.freeze();
            self.layervs.unfreeze();
        }
        else{
            self.embeddingvs.unfreeze();
            self.layervs.freeze();
        }
    }


    
    pub fn get_vectors(&self) -> Vec<Vec<f32>>{

        let mut toreturn = Vec::new();

        for x in 0..self.itemcount{
            let internalembedding = self.embeddings.forward( &Tensor::f_of_slice( &[x] ).unwrap()  );

            let embedding = self.layers.forward( &internalembedding );
            //let embedding = internalembedding;

            let embedding = Vec::<f32>::from( embedding );
            toreturn.push( embedding );
        }

        return toreturn;
    }



    pub fn set_lr(&mut self, lr: f64){

        self.embeddingopt.set_lr( lr );
        self.layeropt.set_lr( lr / 50.0 );
    }




    pub fn batch_train(&mut self, data: Vec<(i64, i64, f64)> ) {

        self.layeropt.zero_grad();
        for (user1, user2, value) in data.clone(){
            let loss = self.get_loss( (user1, user2), value );
            loss.backward();
        }
        self.layeropt.step();


        self.embeddingopt.zero_grad();               
        for (user1, user2, value) in data{

            let loss = self.get_loss( (user1, user2), value );
            loss.backward();
        }
        self.embeddingopt.step();
    }

    
    pub fn get_loss_value(&mut self, input: (i64, i64), target: f64 )  -> f32{

        let loss = self.get_loss( input, target );
        //println!("loss {:?}", loss.size() );
        return f32::from( loss )
    }


    // pub fn tensor_difference(tensor1: &Tensor, tensor2: &Tensor) -> Tensor{
    //     return tensor1 - tensor2;
    // }


    pub fn get_loss(&mut self, input: (i64, i64), target: f64  ) -> Tensor{

        let embedding1 = self.embeddings.forward( &Tensor::f_of_slice( &[input.0] ).unwrap()  ).to_kind( Kind::Float );
        let embedding2 = self.embeddings.forward( &Tensor::f_of_slice( &[input.1] ).unwrap()  ).to_kind( Kind::Float );

        let embedding1 = self.layers.forward( &embedding1 );
        let embedding2 = self.layers.forward( &embedding2 );
        
        
        let distance = embedding1.f_subtract( &embedding2 ).unwrap().abs().to_kind( Kind::Float );
        //let distance = distance.f_square().unwrap();
        let distance = distance.sum( Kind::Float );
        //let distance = distance.sqrt();
        let distance = distance.f_reshape(&[1]).unwrap().to_kind( Kind::Float );


        let targetdistance = Tensor::f_of_slice( &[target] ).unwrap().to_kind( Kind::Float );

        let loss = distance.f_mse_loss( &targetdistance, tch::Reduction::Mean ).unwrap().to_kind( Kind::Float );

        return loss;
    }



    //user, item, rating
    // pub fn train_step(&mut self, input: (i64, i64), target: f64 )  -> f32{
    //     //a list of inputs and a list of targets
    //     let loss = self.get_loss( input, target );
    //     self.opt.backward_step( &loss );
    //     return f32::from( loss )
    // }



}


