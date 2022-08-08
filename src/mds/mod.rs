
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


pub fn get_ideal_distance( first: &Vec<f64>, second: &Vec<f64>) -> f64{

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

    return 0.00001 + totaldifference as f64 / numberdiff as f64;// + 2.2);
}


pub fn get_target_force( emittingpoint: &Point2<f64>,  curpoint: &Point2<f64>, idealdistance: f64 ) -> Vector2<f64>{

    //from emittingpoint to targetpoint
    let mut tocurpoint = (curpoint - emittingpoint).normalize();

    let emittingpoint = Vector2::new( emittingpoint.x, emittingpoint.y );
    let curpoint = Vector2::new( curpoint.x, curpoint.y );

    let targetposition = emittingpoint + tocurpoint * idealdistance;

    let mut force = targetposition - curpoint;

    // if force.magnitude() > 10.{
    //     force = force.normalize() * 10.;
    // }

    //let force = force.normalize() * 0.002;

    let force = force * force.magnitude();

    return force;
}

use std::collections::HashSet;


//get the stress value change from going X from the left to right
fn get_gradient( positions: &HashMap<u32, Point2<f64>>, embeddings: &HashMap<u32, Vec<f64>>, targetpoint: &u32 ) -> Vector2<f64>{

    let targetembedding = embeddings.get(targetpoint).unwrap();
    let targetposition = positions.get(targetpoint).unwrap();

    //get 1000 random points in the positions map
    let samplebaseids = {
        let mut toreturn = HashSet::new();
        //let mut numbersamples = 1000;

        use rand::Rng;
        let mut rng = rand::thread_rng();

        while toreturn.len() < 990{

            let baseid = rng.gen_range(0, 350_000);
            
            if positions.contains_key( &baseid ){

                if baseid != *targetpoint{
                    toreturn.insert( baseid );
                }
            }
        }

        toreturn
        //toreturn.into_iter().collect::<Vec<u32>>()
        // for (baseid,_) in positions{

        //     if baseid != targetpoint{
                
        //         if 0 == rng.gen_range(0, 500){
        //             toreturn.push( *baseid );
        //         }            
            
        //         // numbersamples += -1;
        //         // toreturn.push( *baseid );
        //     }
        //     // if numbersamples <=0{
        //     //     break;
        //     // }
        // }
    };

    //println!("Sample baseids{:?}", samplebaseids.len() );

    //the list of forces each of the sample points has on the target one
    let mut forces = Vec::new();

    for baseid in samplebaseids{

        let otherpointposition = positions.get(&baseid).unwrap();
        let otherpointembedding = embeddings.get(&baseid).unwrap();

        let idealdistance = get_ideal_distance(targetembedding, otherpointembedding);

        let force = get_target_force( otherpointposition, targetposition, idealdistance);

        forces.push( force );
    }

    let mut sumforce = Vector2::new(0.0, 0.0);

    for force in &forces{
        sumforce += force;
    }

    let averageforce = sumforce / (forces.len() as f64);


    let averageforce = averageforce * 0.0005;

    return averageforce;
}



mod mdsmodel;



pub fn majorize(  embeddings: HashMap<u32, Vec<f64>>) -> HashMap<u32, (f32,f32)>{

    use rand::Rng;
    let mut rng = rand::thread_rng();

    let mut model = mdsmodel::MDSModel::new();

    //let embeddings: HashMap<u32, Vec<f64>> = crate::readfromfile::rdd_to_embedding().into_iter().collect();

    // let shorterembeddings = {
    //     let mut newembeddings = HashMap::new();

    //     for (baseid, embedding) in embeddings.clone(){
    //         if newembeddings.len() < 250_000{
    //             newembeddings.insert( baseid, embedding);
    //         }
    //     }
    //     newembeddings
    // };

    
    let mut positions : HashMap<u32, Point2<f64>> = embeddings.iter().map(| (baseid, _) |{ (*baseid,  Point2::new(rng.gen::<f64>(), rng.gen::<f64>()))  }).collect();


    for _ in 0..1{

        print_stress(&positions, &embeddings);

        for (baseid, embedding) in &embeddings{

            let predictedposition = model.forward( embedding.clone() );
            let predictedposition = Vec::<f64>::from( predictedposition );
            let predictedposition = Point2::new(predictedposition[0], predictedposition[1]);

            positions.insert( *baseid, predictedposition );
    

            let gradient = get_gradient(&positions, &embeddings, &baseid);
            //let gradient =gradient.1 / 500.0 ;

            let targetposition = positions.get(baseid).unwrap() + gradient;

            model.train_step( embedding.clone(), (targetposition.x, targetposition.y) );

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









fn print_stress( positions: &HashMap<u32, Point2<f64>>, embeddings: &HashMap<u32, Vec<f64>> ) -> f64{

    //for each combinations of points
    let mut totalstress = 0.0;
    let mut totalsquaredstress = 0.0;

    let mut numberofcombinations = 0;


    use rand::Rng;
    let mut rng = rand::thread_rng();


    for (baseid, position) in positions{
        let embedding = embeddings.get(baseid).unwrap();

        if 0 != rng.gen_range(0, 10000){
            continue;
        }            


        for (otherbaseid, otherposition) in positions{
            let otherembedding = embeddings.get(otherbaseid).unwrap();

            if baseid != otherbaseid{

                let idealdistance = get_ideal_distance(embedding, otherembedding);

                let actualdistance = (position - otherposition).magnitude();
                
                totalstress += (idealdistance - actualdistance).abs();
                totalsquaredstress += (idealdistance - actualdistance).powf(2.0);
                numberofcombinations +=1;
            }
        }
    }


    let averagestress = totalstress as f64 / numberofcombinations as f64;
    println!("average stress {}", averagestress);

    let averagesquaredstress = totalsquaredstress as f64 / numberofcombinations as f64;
    println!("average squared stress {}", averagesquaredstress);


    return (totalstress as f64 / numberofcombinations as f64);


}