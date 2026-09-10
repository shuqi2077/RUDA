// SPDX-License-Identifier: Apache-2.0
// Independently generated SciPy references; Rust tests still need execution.
use super::*;
#[test]fn external_svd_fixture_0(){
    let a=Matrix::new(3,2,vec![3.0,1.0,0.0,2.0,1.0,-1.0]).unwrap();
    let s=Svd::factor(a.view(),Default::default()).unwrap();
    let expected=[3.2906575520321457,2.274109248750774];
    for(g,e)in s.singular_values.iter().zip(expected){assert!((g-e).abs()<1e-10*(1.+e.abs()));}
    let expected=[0.28571428571428575,-0.07142857142857145,0.14285714285714288,0.07142857142857144,0.35714285714285715,-0.21428571428571433];let p=s.pseudo_inverse().unwrap();
    for(g,e)in p.values().iter().zip(expected){assert!((g-e).abs()<1e-9*(1.+e.abs()));}
}
#[test]fn external_svd_fixture_1(){
    let a=Matrix::new(2,3,vec![1.0,2.0,3.0,4.0,5.0,6.0]).unwrap();
    let s=Svd::factor(a.view(),Default::default()).unwrap();
    let expected=[9.508032000695724,0.7728696356734843];
    for(g,e)in s.singular_values.iter().zip(expected){assert!((g-e).abs()<1e-10*(1.+e.abs()));}
    let expected=[-0.9444444444444446,0.4444444444444444,-0.11111111111111084,0.111111111111111,0.722222222222222,-0.22222222222222202];let p=s.pseudo_inverse().unwrap();
    for(g,e)in p.values().iter().zip(expected){assert!((g-e).abs()<1e-9*(1.+e.abs()));}
}
#[test]fn external_svd_fixture_2(){
    let a=Matrix::new(3,2,vec![1.0,2.0,2.0,4.0,3.0,6.0]).unwrap();
    let s=Svd::factor(a.view(),Default::default()).unwrap();
    let expected=[8.366600265340757,7.320183254117253e-16];
    for(g,e)in s.singular_values.iter().zip(expected){assert!((g-e).abs()<1e-10*(1.+e.abs()));}
    let expected=[0.014285714285714285,0.028571428571428564,0.042857142857142844,0.028571428571428577,0.05714285714285714,0.0857142857142857];let p=s.pseudo_inverse().unwrap();
    for(g,e)in p.values().iter().zip(expected){assert!((g-e).abs()<1e-9*(1.+e.abs()));}
}
