

pub mod neumf;

pub mod mf;




fn man()  {



    //get the training data
    //of users to the movie rating

    let ratings = read_from_file::get_raw_ratings();


    let mut normalized = Vec::new();

    for x in ratings{

        // println!("Ratings {:?}", x);

        // println!("Ratings {:?}", newratings);

        let newratings = normalize(x);
        normalized.push( newratings );
    }
    
    let ratings = normalized;



    //println!("raitng {:?}", ratings);
    println!("raitng {:?}", ratings.len());

    let mut ratings = ratings;
    ratings.truncate(6000);
    let ratings = ratings;

    // let mut newratings = Vec::new();

    // for user in ratings{
    //     newratings.push(  normalize( user ) );
    // }

    // let ratings = newratings;



    let mut questionindexes = QuestionIndexes::new();

    let mut trainingdata = Vec::new();

    for userindex in 0..ratings.len(){

        let user = &ratings[userindex];

        for (movie, rating) in user{

            let movieindex = questionindexes.newly_observed_question(movie);

            trainingdata.push( (userindex, movieindex, *rating) );
        }

    }



    /*
        
        let mut trainingdata = Vec::new();

        for userid in 0..2000{

            let mut userratings = databaseuseranswers.pop().unwrap();

            //get two random items in that user and put them together

            use rand::thread_rng;
            use rand::seq::SliceRandom;
            userratings.shuffle(&mut thread_rng());


            while userratings.len() > 2{

                let (user, item, rating) = userratings.pop().unwrap();

                trainingdata.push(  (user, item, rating) );
            }

        }
    */

    // use rand::thread_rng;
    // use rand::seq::SliceRandom;
    // trainingdata.shuffle(&mut thread_rng());

    
    let mut testdata = Vec::new();

    for x in 0..1000{
        testdata.push( trainingdata.pop().unwrap() );
    }


    // use rand::thread_rng;
    // use rand::seq::SliceRandom;
    // trainingdata.shuffle(&mut thread_rng());




    
        

    use model::mf::Model;


    let mut model = Model::new(20000, 20000, 3);


    loop{

        if trainingdata.len() < 20000{
            break;
        }

        let mut data = Vec::new();

        for x in 0..13000{
            data.push( trainingdata.pop().unwrap() );
        }

        model.train( data );



        let mut accuracy = Vec::new();

        let mut totaldiff = 0.0;

        let mut count = 0;

        for curdata in &testdata{

            let rating = model.get_rating( curdata.0 as i64, curdata.1 as i64 );

            accuracy.push(  (rating, curdata.2) );

            totaldiff += (curdata.2 - rating).abs();

            count +=1;

            // if count > 100{
            //     //break;
            // }
            // else{
            //     let rating = model.get_rating( curdata.0 as i64, curdata.1 as i64 );
            //     println!("r{}, p{}, i{}, u{}", curdata.2, rating, curdata.0, curdata.1);

            // }
        }

        //println!("test diff {:?}", accuracy  );

        for itemid in 0..4{

            for userid in 0..4{

                let rating = model.get_rating( userid, itemid );

                println!("userid{:?}, itemid{:?}, rating {:?}", userid, itemid, rating);
            }
        }

        println!("test diff {:?}", totaldiff / 1000. );//testdata.len() as f32);


    }

   
}




