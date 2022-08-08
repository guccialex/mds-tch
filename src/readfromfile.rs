use std::path::Path;

use serde::Deserialize;

use serde::de::DeserializeOwned;
use serde::Serialize;


pub fn read_from_csv< T: DeserializeOwned >( filename: &str ) -> Vec<T>{

    let mut toreturn = Vec::new();

    let mut rdr = csv::Reader::from_path( &filename ).unwrap();
    
    for result in rdr.deserialize() {

        if let Ok(info)  = result{

            let info: T = info;

            //println!("here  {:?}", info);

            toreturn.push( info );
        }
    }

    toreturn
}



pub fn write_to_csv< T: Serialize >( filename: &str, data: Vec<T> ) {

    let mut wtr = csv::Writer::from_path( filename ).unwrap();
    

    for datum in data {
        wtr.serialize( datum );
    }

}


//get the baseid to average rating
//then I can get the normal estimator for how good movies are





pub fn rdd_to_embedding()  -> Vec<(u32, Vec<f64>)>{

    //let mut wtr = csv::Writer::from_path( "created/itemembeddings.csv" ).unwrap();

    let mut factors: Vec<(u32, Vec<f64>)> = Vec::new();

    for path in std::fs::read_dir("rdd").unwrap() {

        let path = path.unwrap().path().clone();
        let contents = std::fs::read_to_string(path ).expect("Something went wrong reading the file");
        
        for string in contents.split("\n"){
            let string = string.replace("[", "");
            let string = string.replace("]", "");
            let string = string.replace(" ", "");
    
            if string.is_empty(){ continue; }

            let mut factor: Vec<f64> = string.split(",").map(|x|{ x.parse::<f64>().unwrap() }).collect();
            let id = factor.remove(0) as u32;

            factors.push( (id, factor)  );
        }
    }

    factors.sort_by(|(a1,a2), (b1,b2)|  {a1.cmp(b1) } );


    // for factor in &factors{
    //     wtr.serialize( factor ).unwrap();
    // }

    return factors;
}


