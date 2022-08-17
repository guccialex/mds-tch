
pub fn difference_majorize( differencevec: Vec<Vec<f64>>, factors: usize ) -> Vec<Vec<f64>>{

    let dims = differencevec.len();
    
    //differencevec
    
    // vec![
    //     0.0,4.0,3.0,7.0,8.0,
    //     4.0,0.0,1.0,6.0,7.0,
    //     3.0,1.0,0.0,5.0,7.0,
    //     7.0,6.0,5.0,0.0,1.0,
    //     8.0,7.0,7.0,1.0,0.0
    //]

    let D = ndarray::Array2::from_shape_vec(
        (dims,dims),
        differencevec.iter().flatten().map(|x| *x as f64).collect::<Vec<f64>>()
    ).unwrap();

    
    //calculate the element-wise square of D
    let D2 = D.iter().zip(D.iter()).map(|(x,y)| x*y).collect::<Vec<f64>>();
    let D2 = ndarray::Array2::from_shape_vec((dims,dims), D2).unwrap();

    // println!("{:?}", D);
    // println!("{:?}", D2);

    //get the identity matrix
    let I = ndarray::Array2::<f64>::eye(dims);
    let S = ndarray::Array2::<f64>::ones((dims,dims));
    let C = I - S * (1.0/dims as f64);

    // println!("{:?}", C);

    //M = -0.5* C @ D2 @ C
    //print(M)
    let M = C.dot(&D2).dot(&C);
    let M = -0.5 * M;
    //println!("{:?}", M);

   
    //convert M from an ndarray to a nalgebra array
    let M = Vec::<f64>::from( M.as_slice().unwrap() );
    let M = nalgebra::base::DMatrix::from_row_slice(dims, dims, &M);
    //println!("M {}", M);

    println!("shur {}", nalgebra::linalg::Schur::new(M.clone()).complex_eigenvalues() );
    //println!("shur {}", nalgebra::linalg::Schur::new(M.clone()).unpack().1 );


    //M.complex_eigenvalues();
    // println!("----");
    // println!("----");

    //get the eigenvalues and eigenvectors of M
    let eval = M.clone().symmetric_eigen();

    
    //println!("eigen {}", M.complex_eigenvalues());

    let l = eval.eigenvalues;
    //calculate the eigenvectors given the eigenvalues l
    println!("eigenvalues l {}", l);
    
    let V = eval.eigenvectors;
    //println!("eigenvectors vee {}", V);



    let s = l.map(|x|{
        if x.is_sign_negative() {
            0.0
        } else {
            x.sqrt()
        }
    });

    //println!("s {}", s);

    //l,V = np.eig(M)
    //print(l)
    //print(V)


    let V2 = V.columns(0, factors);
    let V2 = nalgebra::base::DMatrix::from( V2 );
    //println!("v2 {}", V2 );

    let mut s2 = nalgebra::base::DMatrix::<f64>::zeros( factors, factors );
    s2.set_diagonal( &s.rows(0,factors) );
    //println!("s2 {}", s2 );


    let Q = V2 * s2;

    //println!("Q {}", Q);

    let mut toreturn = Vec::new();

    for x in Q.transpose().column_iter(){

        let x = nalgebra::base::DVector::from( x );

        let x = x.as_slice().to_vec();

        toreturn.push( x );
        //println!("{:?}", x);
    }


    return toreturn;

}

