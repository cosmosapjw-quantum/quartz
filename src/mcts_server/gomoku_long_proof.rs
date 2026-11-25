// Exact rule-only placement certificates; no NN and no probabilistic TT key.
pub type Key=[u64;8];
#[derive(Clone,Debug,PartialEq,Eq)]
pub struct Bound {
    pub action:u16,pub lo:i8,pub hi:i8,pub reply:Option<u16>,
}
#[derive(Clone,Debug,PartialEq,Eq)]
pub struct Proof {
    pub own:Vec<u16>,pub opponent:Vec<u16>,pub bounds:Vec<Bound>,
    pub interval:(i8,i8),pub witness:Option<u16>,pub risk2:u8,
}
pub fn key(_board:&[u8;225],_side:i8,_rule:u8)->Key{[0;8]}
pub fn wins(_board:&[u8;225],_p:u8,_a:usize,_exact:bool)->bool{false}
pub fn scan(_board:&[u8;225],_p:u8,_exact:bool)->Proof{
    Proof{own:vec![],opponent:vec![],bounds:vec![],interval:(-1,1),witness:None,risk2:2}
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]fn overline_and_distinct_identity(){
        let mut b=[0;225];for i in [106,107,108,110,111]{b[i]=1;}
        assert!(wins(&b,1,109,false));assert!(!wins(&b,1,109,true));
        assert_ne!(key(&b,1,0),key(&b,1,1));assert_ne!(key(&b,1,0),key(&b,-1,0));
        let mut c=b;c[224]=2;assert_ne!(key(&b,1,0),key(&c,1,0));
    }
    #[test]fn immediate_and_two_ply_loss(){
        let mut b=[0;225];for i in [100,101,102,103]{b[i]=2;}
        let p=scan(&b,1,false);assert_eq!(p.opponent,vec![99,104]);
        assert_eq!(p.interval,(-1,-1));assert_eq!(p.witness,Some(0));
        assert!(p.bounds.iter().all(|x|x.lo==-1&&x.hi==-1&&x.reply.is_some()));
        for i in [20,21,22,23]{b[i]=1;}
        let p=scan(&b,1,false);assert_eq!(p.own,vec![19,24]);assert_eq!(p.interval,(1,1));assert_eq!(p.witness,Some(19));
    }
    #[test]fn unknown_and_draw(){
        let b=[0;225];let p=scan(&b,1,false);
        assert_eq!(p.bounds.len(),225);assert_eq!(p.witness,None);
        let mut b=[0;225];for r in 0..15{for c in 0..15{b[15*r+c]=if (r+2*c)%4<2{1}else{2};}}
        b[0]=0;let p=scan(&b,1,false);assert_eq!(p.interval,(0,0));assert_eq!(p.witness,Some(0));
    }
}
