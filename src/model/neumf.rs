
use tch::nn::Embedding;

use tch::nn::embedding;
use tch::nn::Sequential;
use tch::nn;

use tch::Kind;

use tch::nn::*;
use tch::Tensor;

const WIDTH: i64 = 20;

pub struct Model{


    MF_Embedding_User: Embedding,
    MF_Embedding_Item: Embedding,
    MLP_Embedding_User: Embedding,
    MLP_Embedding_Item: Embedding,

    prediction: Sequential,

    layers: Sequential,

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
            .. Default::default()
        };

        let MF_Embedding_Item = embedding(vs, num_items, WIDTH, config);
        let MF_Embedding_User = embedding(vs, num_users, WIDTH, config);

        let MLP_Embedding_Item = embedding(vs, num_items, WIDTH, config);
        let MLP_Embedding_User = embedding(vs, num_users, WIDTH, config);



        let opt = Sgd {
            momentum: 0.0,
            dampening: 0.,
            wd: 0.0,
            nesterov: false,
        }.build(&varstore, 0.001 ).unwrap();

        //let opt = nn::Sgd::default().build(&varstore, 0.001 ).unwrap();

        

        //let prediction = Sequential::  Dense(1, activation='sigmoid', init='lecun_uniform', name = "prediction")(predict_vector)

        let prediction =  nn::seq()
        .add(nn::linear(
            vs,
            WIDTH + WIDTH,
            1,
            LinearConfig{
                .. Default::default()
            },
        ))
        .add_fn(|xs| xs.sigmoid());


        let mut layers = nn::seq().add(nn::linear(
            vs,
            WIDTH *2,
            WIDTH ,
            LinearConfig{
                .. Default::default()
            },
        ))
        .add_fn(|xs| xs.relu());

        for x in 0..layers_len{

            layers = layers.add(nn::linear(
                vs,
                WIDTH ,
                WIDTH ,
                LinearConfig{
                    .. Default::default()
                },
            ))
            .add_fn(|xs| xs.relu());
        }

        // layers = layers.add(nn::linear(
        //     vs,
        //     WIDTH *2,
        //     WIDTH ,
        //     LinearConfig{
        //         .. Default::default()
        //     },
        // ))
        // .add_fn(|xs| xs.relu());


        Model{
            MF_Embedding_Item,
            MF_Embedding_User,
            MLP_Embedding_Item,
            MLP_Embedding_User,

            prediction,

            layers,

            opt,
        }

    }

    pub fn forward(&self, user_input: i64, item_input: i64) -> Tensor{


        let user_input = Tensor::f_of_slice( &[user_input] ).unwrap();

        let item_input = Tensor::f_of_slice( &[item_input] ).unwrap();


        //MF part
        let mf_user_latent = self.MF_Embedding_User.forward(&user_input).f_squeeze().unwrap();
        let mf_item_latent = self.MF_Embedding_Item.forward(&item_input).f_squeeze().unwrap();


        //println!("{:?}, and {:?}", mf_user_latent.size(), mf_item_latent.size() );

        let mf_vector = mf_user_latent.f_mul(&mf_item_latent).unwrap();

        //println!("suize {:?},", mf_vector.size() );


        //MLP part
        let mlp_user_latent = self.MLP_Embedding_User.forward(&user_input).f_squeeze().unwrap();
        let mlp_item_latent = self.MLP_Embedding_Item.forward(&item_input).f_squeeze().unwrap();

        
        let mlp_vector = Tensor::f_cat( &[ mlp_user_latent, mlp_item_latent  ], -1  ).unwrap().to_kind( Kind::Float );

        //println!("a");
        
        let mlp_vector = self.layers.forward(&mlp_vector);


        //println!("{:?}, and {:?}", mf_vector.size(), mlp_vector.size() );

        //Concatenate MF and MLP parts
        let predict_vector = Tensor::f_cat( &[mf_vector, mlp_vector], -1 ).unwrap();

        //println!("b");

        //Final prediction layer
        let prediction = self.prediction.forward(&predict_vector);

        return prediction;
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

        let batchsize =  255;

        let mut iteminbatch = 0;

        self.opt.zero_grad();
        while !data.is_empty(){

            let (user, item, rating) = data.pop().unwrap();

            if user > 19999{
                continue;
            }
            if item > 19999{
                continue;
            }
    

            let differencetensor = self.forward( (user as i64), (item as i64) ).to_kind( Kind::Float );
   
            let label = Tensor::f_of_slice( &[rating] ).unwrap().to_kind( Kind::Float );

            let loss = differencetensor.f_l1_loss( &label, tch::Reduction::Mean ).unwrap();

    
            //self.opt.backward_step( &loss );
            loss.backward();

            if iteminbatch % batchsize == 0{
                self.opt.step();
                self.opt.zero_grad();
            }

            iteminbatch += 1;
        }
        self.opt.step();

    }

}








