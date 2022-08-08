
//train on a list of embeddings and the ratings

use tch::Tensor;
use tch::Kind;
use tch::nn::*;
use tch::nn;
use tch::Device;
pub struct RatingPredictionsModel{

    layers: Sequential,

    opt: Optimizer,
}

impl RatingPredictionsModel{

    pub fn new() -> RatingPredictionsModel{

        let vs = nn::VarStore::new(Device::Cpu);

        let adam = nn::Adam::default();

        let opt = adam.build(&vs, 5e-3).unwrap();

        let vs = vs.root();

        let mut layers = nn::seq()
        .add(nn::linear(
            &vs,
            20,
            5,
            Default::default(),
        ))
        // // .add_fn(|xs| xs.relu())
        // // .add(nn::linear(&vs, 20, 20, Default::default()))
        // .add_fn(|xs| xs.relu())
        // .add(nn::linear(&vs, 12, 7, Default::default()))
        // .add_fn(|xs| xs.relu())
        .add(nn::linear(&vs, 5, 1, Default::default()));


        RatingPredictionsModel {
            layers,
            opt,
        }
    }


    pub fn predict_rating(&self, embedding: Vec<f32>) -> f32{

        let user_input = Tensor::f_of_slice( &embedding ).unwrap();

        let prediction = self.layers.forward(&user_input);

        let prediction = prediction.to_kind(Kind::Float);

        let prediction = f32::from( prediction );

        //println!("predicted score {}", prediction);

        return prediction;
    }

    pub fn train_batch(&mut self, itemtoratings: Vec<(Vec<f32>, f32)> ){
    
        let mut totalscoredifference = 0.0;
        let totalnumber = itemtoratings.len();

        self.opt.zero_grad();

        for (embedding, rating) in itemtoratings{

            let targetrating = Tensor::from( rating ).to_kind( Kind::Float );

            let embedding = Tensor::f_of_slice( &embedding ).unwrap().to_kind( Kind::Float );

            let predictedrating = self.embeddingtensortorating( &embedding);

            let loss = predictedrating.f_l1_loss( &targetrating, tch::Reduction::Mean ).unwrap();

            let predictedfloat = f32::from( predictedrating );

            let actualdifference = (rating - predictedfloat).abs();
            totalscoredifference += actualdifference;

            loss.backward();
        }

        println!("total score difference {}", totalscoredifference / totalnumber as f32);

        self.opt.step();

    }





    fn embeddingtensortorating(&self, embeddingvector: &Tensor) -> Tensor{

        //multiply the embedding by the itemvector

        let new = self.layers.forward( embeddingvector );


        return new;    
    }

}