// Four exclusive targets, each (quantized cost, signed expected utility).
pub type Options = [(usize, f64);4];
pub fn dp(rows:&[Options],k:usize)->Vec<usize>{
    let mut v=vec![0.;k+1];let mut take=vec![vec![0u8;k+1];rows.len()];
    for (i,r) in rows.iter().enumerate(){
        let mut w=v.clone();
        for b in 0..=k{
            for j in 1..4{
                let (c,g)=r[j];
                if c<=b && v[b-c]+g>w[b]{
                    w[b]=v[b-c]+g;take[i][b]=j as u8;
                }
            }
        }
        v=w;
    }
    let mut ans=vec![0;rows.len()];let mut b=k;
    for i in (0..rows.len()).rev(){let j=take[i][b] as usize;ans[i]=j;b-=rows[i][j].0;}
    ans
}
pub fn ratio(rows:&[Options],k:usize)->Vec<usize>{
    let mut items=Vec::new();
    for (i,r) in rows.iter().enumerate(){
        for j in 1..4{let (c,g)=r[j];if g>0. && c>0{items.push((g/c as f64,i,j,c));}}
    }
    items.sort_by(|a,b|b.0.total_cmp(&a.0).then_with(||a.1.cmp(&b.1)).then_with(||a.2.cmp(&b.2)));
    let mut ans=vec![0;rows.len()];let mut b=k;
    for (_,i,j,c) in items{if ans[i]==0 && c<=b{ans[i]=j;b-=c;}}
    ans
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn brute_force(){
        for seed in 0..300usize {
            let mut rows=[[(0,0.);4];3];
            for i in 0..3{for j in 1..4{rows[i][j]=(1+(seed*7+i*3+j*5)%8,((seed*13+i*17+j*7)%21) as f64-10.);}}
            for k in 0..=14{
                let plan=dp(&rows,k);
                let (c,g)=plan.iter().enumerate().fold((0,0.),|(c,g),(i,j)|(c+rows[i][*j].0,g+rows[i][*j].1));
                let mut exact=0f64;
                for a in 0..4{for b in 0..4{for d in 0..4{
                    let opts=[rows[0][a],rows[1][b],rows[2][d]];
                    if opts.iter().map(|v|v.0).sum::<usize>()<=k{exact=exact.max(opts.iter().map(|v|v.1).sum::<f64>());}
                }}}
                assert!(c<=k);assert_eq!(g,exact);
            }
        }
    }
    #[test]
    fn ratio_counterexample_and_stop(){
        let rows=[[(0,0.),(3,5.),(9,-1.),(9,-2.)],[(0,0.),(2,3.),(9,-1.),(9,-2.)],[(0,0.),(2,3.),(9,-1.),(9,-2.)]];
        assert_eq!(dp(&rows,4),vec![0,1,1]);
        assert_eq!(dp(&rows,0),vec![0,0,0]);
        assert_eq!(ratio(&rows,4),vec![1,0,0]);
        let neg=[[(0,0.),(1,-1.),(2,-2.),(3,-3.)]];
        assert_eq!(dp(&neg,5),vec![0]);assert_eq!(ratio(&neg,5),vec![0]);
    }
}
