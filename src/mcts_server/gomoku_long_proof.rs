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
pub fn key(board:&[u8;225],side:i8,rule:u8)->Key{
    assert!(side==1||side==-1);assert!(rule<8);
    let mut out=[0u64;8];
    for (i,&v) in board.iter().enumerate(){assert!(v<=2);let bit=2*i;out[bit/64]|=(v as u64)<<(bit%64);}
    out[7]|=((side==1) as u64)<<2;out[7]|=(rule as u64)<<3;out
}
pub fn wins(board:&[u8;225],p:u8,a:usize,exact:bool)->bool{
    assert!(p==1||p==2);
    if a>=225||board[a]!=0{return false;}
    let r=(a/15) as i32;let c=(a%15) as i32;
    for (dr,dc) in [(0,1),(1,0),(1,1),(1,-1)]{
        let mut n=1;
        for sign in [-1,1]{
            let mut rr=r+sign*dr;let mut cc=c+sign*dc;
            while (0..15).contains(&rr)&&(0..15).contains(&cc)&&board[(15*rr+cc) as usize]==p{
                n+=1;rr+=sign*dr;cc+=sign*dc;
            }
        }
        if if exact{n==5}else{n>=5}{return true;}
    }
    false
}
pub fn scan(board:&[u8;225],p:u8,exact:bool)->Proof{
    let legal:Vec<u16>=(0..225).filter(|&a|board[a]==0).map(|a|a as u16).collect();
    assert!(!legal.is_empty(),"scan requires nonterminal");
    let own:Vec<_>=legal.iter().copied().filter(|&a|wins(board,p,a as usize,exact)).collect();
    let opponent:Vec<_>=legal.iter().copied().filter(|&a|wins(board,3-p,a as usize,exact)).collect();
    let bounds:Vec<_>=legal.iter().map(|&a|{
        let (lo,hi,reply)=if own.contains(&a){(1,1,None)}else if legal.len()==1{(0,0,None)}
        else if let Some(&b)=opponent.iter().find(|&&b|b!=a){(-1,-1,Some(b))}
        else{(-1,1,None)};
        Bound{action:a,lo,hi,reply}
    }).collect();
    let interval=(bounds.iter().map(|b|b.lo).max().unwrap(),bounds.iter().map(|b|b.hi).max().unwrap());
    let (risk2,witness)=if bounds.len()==1{(0,Some(bounds[0].action))}else{
        // Top two distinct action upper endpoints: linear certificate scan,
        // including all unexpanded legal actions.
        let mut max1=(-2i8,usize::MAX);let mut max2=-2i8;
        for (i,b) in bounds.iter().enumerate(){
            if b.hi>max1.0{max2=max1.0;max1=(b.hi,i);}else{max2=max2.max(b.hi);}
        }
        let mut risk=2u8;let mut w=None;
        for (i,b) in bounds.iter().enumerate(){
            let other=if i==max1.1{max2}else{max1.0};
            let d=(other-b.lo).max(0) as u8;risk=risk.min(d);
            if d==0&&w.is_none(){w=Some(b.action);}
        }
        (risk,w)
    };
    Proof{own,opponent,bounds,interval,witness,risk2}
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

// P3 root-only tactical guard; deliberately not equivalent to all WDL witnesses.
pub fn guard(_p:&Proof)->Option<u16>{None}
pub fn target_moves(p:&Proof)->Vec<u16>{
    if !p.own.is_empty(){p.own.clone()}
    else if p.opponent.len()==1{p.opponent.clone()}
    else{Vec::new()}
}
#[cfg(test)]
mod tactical_guard_tests{
    use super::*;
    #[test]fn guard_blocks_and_avoids_forced_loss_tie(){
        let mut b=[0;225];for i in [100,101,102,103]{b[i]=2;}
        let p=scan(&b,1,false);assert!(p.witness.is_some());assert_eq!(guard(&p),None);assert!(target_moves(&p).is_empty());
        b[99]=1;let p=scan(&b,1,false);assert_eq!(p.opponent,vec![104]);assert_eq!(guard(&p),Some(104));assert_eq!(target_moves(&p),vec![104]);
        for i in [20,21,22,23]{b[i]=1;}
        let p=scan(&b,1,false);assert_eq!(guard(&p),Some(19));assert_eq!(target_moves(&p),vec![19,24]);
    }
}
