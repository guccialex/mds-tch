
use tch::nn::Embedding;

use tch::nn::embedding;
use tch::nn::Sequential;
use tch::nn;

use tch::Kind;

use tch::nn::*;
use tch::Tensor;

const WIDTH: i64 = 8;

pub struct Model{


    user_embedding: Embedding,
    item_embedding: Embedding,

    user_bias: Embedding,
    item_bias: Embedding,

    //prediction: Sequential,

    opt: Optimizer,

}

impl Model{


    pub fn new(num_users: i64, num_items: i64, layers_len: u32) -> Model{

        let varstore = nn::VarStore::new(tch::Device::Cpu);
        
        let vs = &varstore.root();
        
        //MF_Embedding_User = Embedding(input_dim = num_users, output_dim = mf_dim, name = 'mf_embedding_user',
        //init = init_normal, W_regularizer = l2(reg_mf), input_length=1)

        // pub sparse: bool,
        // pub scale_grad_by_freq: bool,
        // pub ws_init: Init,
        // pub padding_idx: i64,

        let config = EmbeddingConfig{
            sparse: true,
            ws_init: Init::Const(0.0),
            .. Default::default()
        };

        let item_embedding = embedding(vs, num_items, WIDTH, config);
        let user_embedding = embedding(vs, num_users, WIDTH, config);

        let config = EmbeddingConfig{
            sparse: true,
            ws_init: Init::Const(0.0),
            .. Default::default()
        };

        let item_bias = embedding(vs, num_items, 1, config);
        let user_bias = embedding(vs, num_users, 1, config);



        let opt = Sgd {
            momentum: 0.,
            dampening: 0.,
            wd: 0.0,
            nesterov: false,
        }.build(&varstore, 0.01 ).unwrap();

        //let opt = nn::Adam::default().build(&varstore, 0.001 ).unwrap();

        //let prediction = Sequential::  Dense(1, activation='sigmoid', init='lecun_uniform', name = "prediction")(predict_vector)


        Model{
            item_embedding,
            user_embedding,
            item_bias,
            user_bias,

            //prediction,

            opt,
        }

    }

    pub fn forward(&self, user_input: i64, item_input: i64) -> Tensor{


        let user_input = Tensor::f_of_slice( &[user_input] ).unwrap();

        let item_input = Tensor::f_of_slice( &[item_input] ).unwrap();

        let mu = Tensor::f_of_slice( &[3. as f64] ).unwrap();

        //println!("sizes {:?}, {:?}, {:?}", user_input.size(), item_input.size(), mu.size() );

        //MF part
        let mf_user_latent = self.user_embedding.forward(&user_input).f_squeeze().unwrap();
        let mf_item_latent = self.item_embedding.forward(&item_input).f_squeeze().unwrap();



        //println!("sizes 1 {:?}, {:?}, {:?}", mf_user_latent.size(), mf_item_latent.size(), mu.size() );

        let mf_vector = mf_user_latent.f_mul(&mf_item_latent).unwrap();
        let mf_sum = mf_vector.f_sum(Kind::Float).unwrap();

        let user_bias = self.user_bias.forward(&user_input).f_squeeze().unwrap();
        let item_bias = self.item_bias.forward(&item_input).f_squeeze().unwrap();

        //println!("sizes 1 {:?}, {:?}, {:?}, {:?}", user_bias.size(), item_bias.size(), mu.size(), mf_vector.size()  );

        //println!("sizes 2 {:?},", mf_vector.size() );

        //let mf_vector = Tensor::f_cat( &[ mf_vector, user_bias, item_bias, mu  ], -1  ).unwrap().to_kind( Kind::Float );


        //println!("sizes 3 {:?},", mf_vector.size() );


        //let mf_sum = mf_vector.f_sum(Kind::Float).unwrap();

        //println!("end size {:?}", mf_sum.size() );

        let rating = mf_sum + user_bias + item_bias + mu;

        //println!("rating {:?}", rating);

        return rating;
    }


    pub fn get_rating(&self, user: i64, item: i64) -> f32{

        if user > 19999{

            return 0.0;
        }
        if item > 19999{

            return 0.0;
        }


        let tensor = self.forward( user, item );

        return f32::from(tensor);
    }


    //user, item, rating
    pub fn train(&mut self, mut data: Vec<( usize, usize , f32)>) {

        let batchsize =  10;

        let mut iteminbatch = 0;

        self.opt.zero_grad();
        while !data.is_empty(){

            //println!("a");

            let (user, item, rating) = data.pop().unwrap();

            if user > 19999{
                continue;
            }
            if item > 19999{
                continue;
            }
    
            //println!("b");
            let differencetensor = self.forward( (user as i64), (item as i64) ).to_kind( Kind::Float );
   
            let label = Tensor::f_of_slice( &[rating] ).unwrap().to_kind( Kind::Float );


            //println!("diff and label {:?}, {:?}",  differencetensor.size(), label.size() );

            let loss = differencetensor.f_mse_loss( &label, tch::Reduction::Mean ).unwrap();

            //println!("loss size {:?}", loss.size() );

            //println!("c");
    
            //self.opt.backward_step( &loss );
            loss.backward();

            if iteminbatch % batchsize == 0{
                self.opt.step();
                self.opt.zero_grad();
            }

            //println!("d");
            iteminbatch += 1;
        }
        self.opt.step();

    }

}








