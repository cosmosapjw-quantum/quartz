fn main(){
    for n in 0..=4096u32{
        let actual=(2.0f32*(n as f32).powf(0.5)).floor() as u32;
        let mut exact=0;while (exact+1)*(exact+1)<=4*n{exact+=1;}
        assert_eq!(actual,exact,"n={n}");
    }
    println!("4097 f32 widening widths equal exact integer floor.");
}
