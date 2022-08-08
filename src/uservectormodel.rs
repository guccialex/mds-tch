use tch::Tensor;
use tch::Kind;
use tch::nn::*;
use tch::nn;
use tch::Device;
pub struct UserVectorModel{

    uservector: Embedding,

    opt: Optimizer,
}

impl UserVectorModel{

    pub fn new() -> UserVectorModel{

        let vs = nn::VarStore::new(Device::Cpu);

        let mut adam = nn::Adam::default();
        // adam.beta1 = 0.8;//99;
        // adam.beta2 = 0.9;//999;
        // adam.wd = 0.1;
        let opt = adam.build(&vs, 5e-2).unwrap();

        let vs = vs.root();


        let config = EmbeddingConfig{
            .. Default::default()
        };

        let embedding = embedding(vs, 1, 20, config);


        UserVectorModel {
            uservector: embedding,
            opt,
        }

    }

    pub fn train_batch(&mut self, itemtoratings: Vec<(Vec<f32>, f32)> ){
    
        let mut totalscoredifference = 0.0;
        let totalnumber = itemtoratings.len();

        self.opt.zero_grad();

        for (embedding, rating) in itemtoratings{

            let targetrating = Tensor::from( rating ).to_kind( Kind::Float );

            let embedding = Tensor::f_of_slice( &embedding ).unwrap().to_kind( Kind::Float );

            let predictedrating = self.itemtensortorating( &embedding);

            let loss = predictedrating.f_l1_loss( &targetrating, tch::Reduction::Mean ).unwrap();


            let predictedfloat = f32::from( predictedrating );

            let actualdifference = (rating - predictedfloat).abs();
            totalscoredifference += actualdifference;


            loss.backward();
        }

        println!("total score difference {}", totalscoredifference / totalnumber as f32);

        self.opt.step();

    }


    pub fn train_step(&mut self, itemvector: Vec<f32>, rating: f32  ){

        let itemvector = Tensor::f_of_slice( &itemvector ).unwrap();

        let predictedrating = self.itemtensortorating(&itemvector);


        let floatrating = f32::from( &predictedrating );

        // println!("actual rating {:?}", rating);
        // println!("predicted rating {:?}", floatrating);

        let targetrating = Tensor::from( rating ).to_kind( Kind::Float );

        let loss = predictedrating.f_mse_loss( &targetrating, tch::Reduction::Mean ).unwrap();

        self.opt.backward_step( &loss );
    }


    pub fn train_embedding_values(&mut self, values: &Vec<f32>){

        let embeddings = self.uservector.forward( &Tensor::f_of_slice( &[0] ).unwrap() );

        let target = Tensor::f_of_slice( &values ).unwrap();

        let loss = embeddings.f_mse_loss( &target, tch::Reduction::Mean ).unwrap();

        self.opt.backward_step( &loss );
    }

    pub fn get_embedding_values(&self) -> Vec<f32>{

        let embeddings = self.uservector.forward( &Tensor::f_of_slice( &[0] ).unwrap() );

        let asvec = Vec::<f32>::from( embeddings.to_kind( Kind::Float )  );

        return asvec;
    }

    fn itemtensortorating(&self, itemvector: &Tensor) -> Tensor{

        //multiply the embedding by the itemvector

        let new = self.uservector.forward( &Tensor::f_of_slice( &[0] ).unwrap() ) * itemvector;

        let sum = new.f_sum( Kind::Float ).unwrap();

        return sum;    
    }

}