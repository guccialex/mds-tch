
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


fn get_ideal_distance( user1: &Vec<f64>, user2: &Vec<f64>) -> f64{
    let mut distance = 0.0;

    let squared = false;

    for i in 0..user1.len(){
        if squared{
            distance += (user1[i] - user2[i]).abs().powi(2);
        }
        else{
            distance += (user1[i] - user2[i]).abs();
        }
    }

    if squared{
        distance = distance.sqrt();
    }

    distance = distance / user1.len() as f64;
    distance = distance.powi(2);

    if ! distance.is_finite(){
        println!("distance is wrong {:?}, {:?}, {:?}", user1, user2, distance);
        panic!("wrong");
    }

    return distance;
}



pub fn get_target_force( emittingpoint: &DVector<f64>,  curpoint: &DVector<f64>, idealdistance: f64 ) -> DVector<f64>{

    //from emittingpoint to targetpoint
    let tocurpoint = (curpoint - emittingpoint).normalize();

    //let emittingpoint = Vector2::new( emittingpoint.x, emittingpoint.y );
    //let curpoint = Vector2::new( curpoint.x, curpoint.y );

    let targetposition = emittingpoint + tocurpoint * idealdistance;

    let force = targetposition - curpoint;

    // if force.magnitude() > 10.{
    //     force = force.normalize() * 10.;
    // }

    //let force = force.normalize();

    if ! force.magnitude().is_finite(){
        //println!("force is wrong {:?}, {:?}, {:?}", emittingpoint, curpoint, force);
        //this happens when curpoint == emittingpoint
        //so return that, a zero force
        return curpoint - emittingpoint;
        //panic!("wrong");
    }


    let force = force.clone();// * force.magnitude();


    // if ! force.magnitude().is_finite(){
    //     println!("wtf, {:?}, {:?}, {:?}", force, emittingpoint, curpoint);
    // }

    return force.clone();
}


use std::collections::HashSet;


//get the stress value change from going X from the left to right
fn get_gradient<K: Hash+Eq+Clone>( positionandembedding: &HashMap<K, (DVector<f64>, Vec<f64>) >, targetpoint: &K ) -> DVector<f64>{

    let (targetposition, targetembedding) = positionandembedding.get(targetpoint).unwrap();

    let numberofsamples = 2000;
    let totalitems = positionandembedding.len();

    //get 1000 random points in the positions map
    let samplebaseids = {

        use rand::Rng;
        let mut rng = rand::thread_rng();

        let mut toreturn = HashSet::new();
        //let mut numbersamples = 1000;
        positionandembedding.keys().for_each( |id| {
            if id != targetpoint{

                //random number between 0 and totalitems
                let randomid = rng.gen_range(0, totalitems);
                if randomid <= numberofsamples{
                    toreturn.insert( id.clone() );
                }
            }
        } );
        toreturn
    };

    if samplebaseids.len() == 0{
        panic!("zero samplebaseid");
    }

    let samplebaseidlen = samplebaseids.len() as f64;


    //the list of forces each of the sample points has on the target one
    let mut forcesum = DVector::from_vec( vec![0.0; targetposition.len()] );

    for baseid in samplebaseids{

        let (otherpointposition, otherpointembedding) = positionandembedding.get(&baseid).unwrap();

        let idealdistance = get_ideal_distance(targetembedding, otherpointembedding);

        let force = get_target_force( otherpointposition, targetposition, idealdistance);

        forcesum += force;
    }



    let averageforce = forcesum / samplebaseidlen;

    //let averageforce = averageforce.normalize();

    //0.01 seems good
    //well, lower seems better the more items are added. hmmm
    let mut averageforce = averageforce * 0.002;


    // if averageforce.magnitude().is_finite(){
    //     return averageforce;
    // }
    // else{
    //     return averageforce.normalize() * 0.00001;
    // }
    // if averageforce.magnitude() > 0.1{
    //     averageforce = averageforce.normalize() * 0.1;
    // }

    return averageforce;
}


pub fn set_positions_according_to_model<K: Hash+Eq+Clone>(positionandembedding: &mut HashMap<K, (DVector<f64>, Vec<f64>) >, model: &mut MDSModel){

    for (_, (position, embedding)) in positionandembedding.iter_mut(){

        let newposition = model.forward( embedding.clone() );
        let newposition = DVector::from_vec( Vec::<f64>::from( newposition ) );

        if ! newposition.magnitude().is_finite(){
            panic!("wtf {:?}", newposition);
        }
        
        *position = newposition;
    }
}


mod mdsmodel;

use nalgebra::Point;
use nalgebra::DVector;


pub fn majorize<K: Hash+Eq+Clone>(  embeddings: HashMap<K, Vec<f64>>, outputsize: u32) -> HashMap<K, Vec<f32>>{

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
    

    let mut counter = 0;

    for _ in 0..15{

        print_stress(&positionandembedding );

        let mut averagegradient = 0.0;
        let mut numbergradient = 0;

        //have to do that to the keys to iterate over the keys without borrowing them also
        for baseid in positionandembedding.keys().map(|k|{k.clone()}).collect::<Vec<K>>().into_iter(){


            let neural = true;

            if neural {

                if counter % 4000 == 0{
                    set_positions_according_to_model( &mut positionandembedding, &mut model);
                }
                counter += 1;



                let (_, embedding) = positionandembedding.get(&baseid).unwrap().clone();
                let newposition = model.forward( embedding.clone() );
                let newposition = DVector::from_vec( Vec::<f64>::from( newposition ) );
                let x = positionandembedding.get_mut(&baseid).unwrap();
                x.0 = newposition.clone();
                

                let (newposition, embedding) = positionandembedding.get(&baseid).unwrap().clone();


                let gradient = get_gradient(&positionandembedding, &baseid);
                averagegradient += gradient.magnitude();
                numbergradient += 1;

                let targetposition = newposition + gradient;

                model.train_step( embedding.clone(), targetposition.iter().map(|x|{ *x  }).collect() );

                // let newposition = model.forward( embedding.clone() );
                // let newposition = DVector::from_vec( Vec::<f64>::from( newposition ) );
                // let x = positionandembedding.get_mut(&baseid).unwrap();
                // x.0 = newposition;
            }
            else{

                let (position, _) = positionandembedding.get(&baseid).unwrap().clone();

                let gradient = get_gradient(&positionandembedding, &baseid);
                averagegradient += gradient.magnitude();
                numbergradient += 1;

                let targetposition = position + gradient;
                
                positionandembedding.get_mut(&baseid).unwrap().0 = targetposition;
            }

        }

        averagegradient = averagegradient / numbergradient as f64;
        println!("average gradient {}", averagegradient);

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

    print_stress(&positionandembedding);

    let mut toreturn = positionandembedding.into_iter().map(|(k, (position, embedding))|{
        (k, position.iter().map(|x|{ *x as f32 }).collect())
    }).collect();

    return toreturn;

}









fn print_stress<K: Hash+Eq>( positionandembedding: &HashMap<K, (DVector<f64>, Vec<f64>) > ) -> f64{

    //for each combinations of points
    let mut totalstress = 0.0;
    let mut totalsquaredstress = 0.0;

    let mut numberofcombinations = 0;



    use rand::Rng;
    let mut rng = rand::thread_rng();
    let numberofsamples = 2000;
    let totalitems = positionandembedding.len();
    //get 1000 random points in the positions map
    let samplebaseids = {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut toreturn = HashSet::new();
        //let mut numbersamples = 1000;
        positionandembedding.keys().for_each( |id| {
            //random number between 0 and totalitems
            let randomid = rng.gen_range(0, totalitems);
            if randomid <= numberofsamples{
                toreturn.insert( id.clone() );
            }
        } );
        toreturn
    };



    for baseid in samplebaseids{

        let (position, embedding) = positionandembedding.get(&baseid).unwrap();

        for (otherbaseid, (otherposition, otherembedding)) in positionandembedding{

            if baseid != otherbaseid{

                let idealdistance = get_ideal_distance(embedding, otherembedding);
                //println!("ideal distance {}", idealdistance);

                let actualdistance = (position - otherposition).magnitude();

                //if actualdistance.is_finite(){
                //println!("actual distance {}", actualdistance);
                totalstress += (idealdistance - actualdistance).abs();
                totalsquaredstress += (idealdistance - actualdistance).powf(2.0);
                numberofcombinations +=1;
                //}

            }
        }
    }


    //println!("num of combinations: {}", numberofcombinations);


    let averagestress = totalstress as f64 / numberofcombinations as f64;
    println!("                          average stress {}", averagestress);

    let averagesquaredstress = totalsquaredstress as f64 / numberofcombinations as f64;
    println!("        average squared stress {}", averagesquaredstress);


    return (totalstress as f64 / numberofcombinations as f64);


}