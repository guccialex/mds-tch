
use std::collections::HashMap;
use std::hash::Hash;
use std::thread::current;
use serde::Serialize;
use serde::Deserialize;
use tch::Tensor;
use tch::Kind;
use tch::nn::*;
use tch::nn;
use nalgebra::geometry::Point2;
use nalgebra::base::Vector2;
use nalgebra::base::UnitVector2;

pub mod differencemdsmodel;
pub use differencemdsmodel::DifferenceMDSModel;


pub use mdsmodel::MDSModel;



fn vector_distance( user1: &Vec<f64>, user2: &Vec<f64>) -> f64{
    let mut distance = 0.0;

    for i in 0..user1.len(){
        distance += (user1[i] - user2[i]).abs();
    }

    distance = distance / user1.len() as f64;

    return distance;
}


pub fn get_ideal_distance( first: &Vec<f64>, second: &Vec<f64>) -> f64{

    return vector_distance(first, second);

    // let mut totaldifference = 0.0;
    // let mut numberdiff = 0;
    
    // for index in 0..first.len(){

    //     let this = first[index];
    //     let that = second[index];

    //     let curdiff = (this - that).abs();
    //     let curdiff = curdiff;

    //     totaldifference += curdiff;
    //     numberdiff += 1;
    // }

    // let toreturn = 0.001 + totaldifference as f64 / numberdiff as f64;

    // //println!("{}, {}, {}, {:?}, {:?}", totaldifference, numberdiff, toreturn, first, second);
    
    // if toreturn.is_finite(){
    //     return toreturn;
    // }
    // else{
    //     panic!("done" );
    // }

}


pub fn get_target_force( emittingpoint: &DVector<f64>,  curpoint: &DVector<f64>, idealdistance: f64 ) -> DVector<f64>{

    //from emittingpoint to targetpoint
    let mut tocurpoint = (curpoint - emittingpoint).normalize();

    //let emittingpoint = Vector2::new( emittingpoint.x, emittingpoint.y );
    //let curpoint = Vector2::new( curpoint.x, curpoint.y );

    let targetposition = emittingpoint + tocurpoint * idealdistance;

    let mut force = targetposition - curpoint;

    // if force.magnitude() > 10.{
    //     force = force.normalize() * 10.;
    // }
    // let force = force.normalize() * 0.002;

    let force = force * force.magnitude();

    return force;
}


use std::collections::HashSet;


//get the stress value change from going X from the left to right
fn get_gradient<K: Hash+Eq>( positionandembedding: &HashMap<K, (DVector<f64>, Vec<f64>) >, targetpoint: &K ) -> DVector<f64>{

    let (targetposition, targetembedding) = positionandembedding.get(targetpoint).unwrap();

    //get 1000 random points in the positions map
    let samplebaseids = {
        let mut toreturn = HashSet::new();
        //let mut numbersamples = 1000;
        positionandembedding.keys().for_each( |id| {
            if id != targetpoint{
                toreturn.insert( *id );
            }
        } );
        toreturn
    };


    //the list of forces each of the sample points has on the target one
    let mut forcesum = DVector::from_vec( vec![0.0; targetposition.len()] );

    for baseid in samplebaseids{

        let (otherpointposition, otherpointembedding) = positionandembedding.get(&baseid).unwrap();

        let idealdistance = get_ideal_distance(targetembedding, otherpointembedding);

        let force = get_target_force( otherpointposition, targetposition, idealdistance);

        forcesum += force;
    }


    let averageforce = forcesum / (samplebaseids.len() as f64);

    let averageforce = averageforce * 0.005;

    return averageforce;
}



mod mdsmodel;

use nalgebra::Point;
use nalgebra::DVector;


pub fn majorize<K: Hash+Eq>(  embeddings: HashMap<K, Vec<f64>>, outputsize: u32) -> HashMap<K, Vec<f32>>{

    use rand::Rng;
    let mut rng = rand::thread_rng();

    let inputsize = embeddings.iter().next().unwrap().1.len() as i64;

    let mut model = mdsmodel::MDSModel::new( inputsize, outputsize as i64 );


    //position and embedding
    let mut positionandembedding: HashMap<K, (DVector<f64>, Vec<f64>) >  = HashMap::new();

    for (k, embedding) in embeddings{
        let mut position = Vec::new();
        for x in 0..outputsize{
            position.push( rng.gen_range(-1.0, 1.0) );
        }
        let randomposition = DVector::from_vec( position );
        positionandembedding.insert( k, (randomposition, embedding) );
    }
    


    for _ in 0..10{

        print_stress(&positionandembedding );

        for (baseid, (position, embedding) ) in positionandembedding.iter_mut(){

            let newposition = model.forward( embedding.clone() );
            let newposition = DVector::from_vec( Vec::<f64>::from( newposition ) );

            *position = newposition;
    
            let gradient = get_gradient(&positionandembedding, &baseid);
            //let gradient = gradient.1 / 500.0 ;

            let targetposition = *position + gradient;
            model.train_step( embedding.clone(), targetposition.iter().map(|x|{ *x  }).collect() );


            positions.insert( *baseid, targetposition );
        }

        // if totalstress < 0.00022{
        //     let towrite: HashMap<u32, (f64,f64)> = embeddings.iter().map(|(basid,embedding)|{    
        //         let predictedposition = model.forward( embedding.clone() );
        //         let predictedposition = Vec::<f64>::from( predictedposition );
        //         let predictedposition = (predictedposition[0], predictedposition[1]);  
        //         (*basid, predictedposition)
        //     }).collect();
        //     let mut file = std::fs::File::create("positions.json").unwrap();
        //     serde_json::to_writer( file, &towrite);
        // }
    }

    print_stress(&positions, &embeddings);

    let mut toreturn = HashMap::new();

    for position in positions{
        toreturn.insert( position.0, (position.1.x as f32, position.1.y as f32) );
    }

    return toreturn;

}









fn print_stress<K: Hash+Eq>( positionandembedding: &HashMap<K, (DVector<f64>, Vec<f64>) > ) -> f64{

    //for each combinations of points
    let mut totalstress = 0.0;
    let mut totalsquaredstress = 0.0;

    let mut numberofcombinations = 0;


    use rand::Rng;
    let mut rng = rand::thread_rng();


    for (baseid, (position, embedding)) in &positionandembedding{

        for (otherbaseid, (otherposition, otherembedding)) in &positionandembedding{

            if baseid != otherbaseid{

                let idealdistance = get_ideal_distance(embedding, otherembedding);
                //println!("ideal distance {}", idealdistance);

                let actualdistance = (position - otherposition).magnitude();

                if actualdistance.is_finite(){
                    //println!("actual distance {}", actualdistance);
                    totalstress += (idealdistance - actualdistance).abs();
                    totalsquaredstress += (idealdistance - actualdistance).powf(2.0);
                    numberofcombinations +=1;
                }

            }
        }
    }


    println!("num of combinations: {}", numberofcombinations);


    let averagestress = totalstress as f64 / numberofcombinations as f64;
    println!("average stress {}", averagestress);

    let averagesquaredstress = totalsquaredstress as f64 / numberofcombinations as f64;
    println!("average squared stress {}", averagesquaredstress);


    return (totalstress as f64 / numberofcombinations as f64);


}