
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

pub fn main(){
    let embeddings: HashMap<u32, Vec<f64>> = crate::readfromfile::rdd_to_embedding().into_iter().collect();

    mds::majorize( embeddings );
}










/*




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

mod mdsmodel;




pub fn get_ideal_distance( first: &Vec<f32>, second: &Vec<f32>) -> f32{


    let mut totaldifference = 0.0;
    let mut numberdiff = 0;

    
    for index in 0..first.len(){

        let this = first[index];
        let that = second[index];

        let curdiff = (this - that).abs();
        let curdiff = curdiff;

        totaldifference += curdiff;
        numberdiff += 1;
    }



    return totaldifference / (numberdiff as f32 + 0.01);
}


pub fn point_distances(first: &(f32,f32), second: &(f32,f32)) -> f32{


    return ((first.0 - second.0).powf(2.0) + (first.1 - second.1).powf(2.0)).powf(0.5);

}



//get the stress value change from going X from the left to right
fn get_gradient( positions: &HashMap<u32, (f32,f32)>, embeddings: &HashMap<u32, Vec<f32>>, targetpoint: &u32 ) -> (f32, f32){

    let targetembedding = embeddings.get(targetpoint).unwrap();

    let targetposition = positions.get(targetpoint).unwrap();

    let targetleftposition = (targetposition.0 - 0.01 , targetposition.1);
    let targetrightposition = (targetposition.0 + 0.01 , targetposition.1);

    let targetdownposition = (targetposition.0 , targetposition.1 - 0.01 );
    let targetupposition = (targetposition.0, targetposition.1 + 0.01);


    //get 100 random points in the positions map
    let samplebaseids = {

        let mut toreturn = Vec::new();
        let mut numbersamples = 1000;

        for (baseid,_) in positions{
            if baseid != targetpoint{
                numbersamples += -1;
                toreturn.push( *baseid );
            }
            if numbersamples <=0{
                break;
            }
        }
        toreturn
    };



    let mut leftstress = 0.;
    let mut rightstress = 0.;

    let mut downstress = 0.;
    let mut upstress = 0.;


    for baseid in samplebaseids{

        let otherpointposition = positions.get(&baseid).unwrap();
        let otherpointembedding = embeddings.get(&baseid).unwrap();

        let idealdistance = get_ideal_distance(targetembedding, otherpointembedding);

        let leftactualdistance = point_distances( otherpointposition, &targetleftposition);
        let rightactualdistance = point_distances( otherpointposition, &targetrightposition);
        let downactualdistance = point_distances( otherpointposition, &targetdownposition);
        let upactualdistance = point_distances( otherpointposition, &targetupposition);

        //the squared difference between ideal distance and actual distance
        leftstress += (idealdistance - leftactualdistance).powf(2.0);
        rightstress += (idealdistance - rightactualdistance).powf(2.0);

        downstress += (idealdistance - downactualdistance).powf(2.0);
        upstress += (idealdistance - upactualdistance).powf(2.0);
        
    }



    return (rightstress - leftstress,  upstress - downstress);


}



fn get_total_stress( positions: &HashMap<u32, (f32,f32)>, embeddings: &HashMap<u32, Vec<f32>> ) -> f32{

    //for each combinations of points
    let mut totalstress = 0.0;

    let mut numberofcombinations = 0;


    for (baseid, position) in positions{
        let embedding = embeddings.get(baseid).unwrap();

        for (otherbaseid, otherposition) in positions{
            let otherembedding = embeddings.get(otherbaseid).unwrap();

            if baseid != otherbaseid{

                let idealdistance = get_ideal_distance(embedding, otherembedding);

                let actualdistance = point_distances( position, otherposition);

                totalstress += (idealdistance - actualdistance).powf(2.0);
                numberofcombinations +=1;
            }
        }
    }

    return (totalstress / numberofcombinations as f32);

}


pub fn average(vecs: Vec<Vec<f32>>) -> Vec<f32>{

    let average = | vecs : Vec<Vec<f32>>| {
        let mut vecs = vecs.clone();
        let numberofembeddings = vecs.len();
        if let Some(first) = vecs.pop(){
            let mut newvector = first;
            for embedding in vecs{
                for index in 0..embedding.len(){
                    newvector[index] += embedding[index];
                }
            }
            for cur in newvector.iter_mut(){
                *cur = *cur / (numberofembeddings as f32);
            }
            return newvector;
        }
        panic!("no embeddings in the from_average thing");
    };

    return average( vecs );

}

pub fn get_my_embeddings_and_ratings() -> Vec<(Vec<f32>, f32)>{

    let embeddings: HashMap<u32, Vec<f32>> = readfromfile::rdd_to_embedding().into_iter().collect();




    for (baseid, rating) in baseidsandratings.iter_mut(){

        *rating = *rating * 1.0;
    }


    let mut embeddingandrating: Vec<(Vec<f32>, f32)> = Vec::new();

    let mut toaverage: Vec<Vec<f32>> = Vec::new();

    for (baseid, rating) in baseidsandratings{

        let embedding = embeddings.get(&baseid).unwrap();
        embeddingandrating.push((embedding.clone(), rating));

        toaverage.push(embedding.clone());
    }

    //let mut averagerating

    let toaverage = average( toaverage );

    return embeddingandrating;

}

pub fn train_rating_prediction_model(){

    let embeddingandrating = get_my_embeddings_and_ratings();


    let mut uservectormodel = ratingprediction::RatingPredictionsModel::new();


    for x in 0..1500{

        uservectormodel.train_batch( embeddingandrating.clone() );
    }

    let embeddings: HashMap<u32, Vec<f32>> = readfromfile::rdd_to_embedding().into_iter().collect();

    let mut baseidtoscores: Vec<(u32, f32)> = embeddings.into_iter()
    .map(|(baseid, embedding)| {

        let prediction = uservectormodel.predict_rating( embedding );

        return (baseid, prediction);
    } ).collect();


    baseidtoscores.sort_by(|(_,a), (_,b)|{ a.partial_cmp(b).unwrap() });

    for (baseid,score) in baseidtoscores{

        if score > 0.45 && score < 0.75{

            println!("{},", baseid);//, score);
        }
    }

    //println!("baseidtoscore {:?}", baseidtoscores);


}


fn main() {

    train_rating_prediction_model();

    panic!("done");

    let mut uservectormodel = uservectormodel::UserVectorModel::new();

    let embeddingandrating = get_my_embeddings_and_ratings();


    for x in 0..4500{

        uservectormodel.train_batch( embeddingandrating.clone() );



        println!("embedding values {:?}", uservectormodel.get_embedding_values() );



    }


    println!("embedding values {:?}", uservectormodel.get_embedding_values() );


    //prev got: total score difference 0.21169393


}






fn majorize(){

    use rand::Rng;
    let mut rng = rand::thread_rng();

    let mut model = mdsmodel::MDSModel::new();

    let embeddings: HashMap<u32, Vec<f32>> = readfromfile::rdd_to_embedding().into_iter().collect();


    let shorterembeddings = {
        let mut newembeddings = HashMap::new();

        for (baseid, embedding) in embeddings.clone(){
            if newembeddings.len() < 7_000{
                newembeddings.insert( baseid, embedding);
            }
        }
        newembeddings
    };

    
    let mut positions : HashMap<u32, (f32,f32)> = shorterembeddings.iter().map(| (baseid, _) |{ (*baseid,  (rng.gen::<f32>(), rng.gen::<f32>()))  }).collect();


    for x in 0..100{

        for (baseid, embedding) in &shorterembeddings{

            let predictedposition = model.forward( embedding.clone() );
            let predictedposition = Vec::<f32>::from( predictedposition );
            let predictedposition = (predictedposition[0], predictedposition[1]);
            positions.insert( *baseid, predictedposition );
    

            let gradient = get_gradient(&positions, &embeddings, &baseid);
            let gradient = (- gradient.0 / 500.0, -gradient.1 / 500.0);
            let targetposition = (positions.get(baseid).unwrap().0 + gradient.0, positions.get(baseid).unwrap().1 + gradient.1);

            model.train_step( embedding.clone(), targetposition);

            positions.insert( *baseid, targetposition );
        }

        let totalstress = get_total_stress(&positions, &embeddings);
        println!("total stress {}", totalstress);


        if totalstress < 0.0022{

            let towrite: HashMap<u32, (f32,f32)> = embeddings.iter().map(|(basid,embedding)|{    
                let predictedposition = model.forward( embedding.clone() );
                let predictedposition = Vec::<f32>::from( predictedposition );
                let predictedposition = (predictedposition[0], predictedposition[1]);  
                (*basid, predictedposition)
            }).collect();
    
            let mut file = std::fs::File::create("positions.json").unwrap();
    
            serde_json::to_writer( file, &towrite);
        }
    }

    // println!("embeddings {:?}", embeddings);
    panic!("a");

}



*/