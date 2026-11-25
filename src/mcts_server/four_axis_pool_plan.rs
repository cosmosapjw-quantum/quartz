// Pure bounded resident-pool scheduler; no model or solver data.
#[derive(Clone,Copy,Debug,Default,PartialEq,Eq)]
pub struct Slot{
    pub blocks:u32,pub spent_us:u64,pub retired:bool,pub risk2:u8,pub contenders:u8,
}
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum Policy{FixedWork,FixedTime(u64),Reuse,Bound}
pub fn choose(slots:&[Slot],cursor:usize,policy:Policy)->Option<usize>{
    let mut best=None;
    for k in 0..slots.len(){
        let i=(cursor+k)%slots.len();let a=slots[i];
        if a.retired||a.blocks>=63{continue;}
        if matches!(policy,Policy::FixedWork)&&a.blocks>=15{continue;}
        if let Policy::FixedTime(q)=policy{if a.spent_us>=q{continue;}}
        if policy!=Policy::Bound{return Some(i);}
        let score=(a.risk2,a.contenders,std::cmp::Reverse(a.blocks));
        if best.is_none_or(|(s,_)|score>s){best=Some((score,i));}
    }
    best.map(|(_,i)|i)
}
pub fn account(blocks:&[u32],certified:&[bool],q:u32)->(u32,u32,u32,u32){
    assert_eq!(blocks.len(),certified.len());
    let shortfall=blocks.iter().map(|&b|q.saturating_sub(b)).sum::<u32>();
    let extra=blocks.iter().map(|&b|b.saturating_sub(q)).sum::<u32>();
    let credit=blocks.iter().zip(certified).filter(|(_,c)|**c).map(|(&b,_)|q.saturating_sub(b)).sum::<u32>();
    assert!(extra<=shortfall);(credit,extra,shortfall,shortfall-extra)
}
#[cfg(test)]
mod tests{
 use super::*;
 #[test]fn real_reuse_and_fixed_quota(){
    let s=[Slot{retired:true,..Slot::default()},Slot{blocks:15,risk2:1,..Slot::default()},Slot{blocks:16,risk2:2,contenders:3,..Slot::default()},Slot{blocks:15,risk2:2,contenders:2,..Slot::default()}];
    assert_eq!(choose(&s,0,Policy::FixedWork),None);
    assert_eq!(choose(&s,0,Policy::Reuse),Some(1));
    assert_eq!(choose(&s,0,Policy::Bound),Some(2));
    assert_eq!(account(&[0,20,20,20],&[true,false,false,false],15),(15,15,15,0));
    assert_eq!(account(&[3,7,15,15],&[true,true,false,false],15),(20,0,20,20));
 }
 #[test]fn time_and_caps(){
    let s=[Slot{spent_us:100,..Slot::default()},Slot{blocks:63,..Slot::default()},Slot{retired:true,..Slot::default()},Slot{spent_us:99,..Slot::default()}];
    assert_eq!(choose(&s,0,Policy::FixedTime(100)),Some(3));
    assert_eq!(choose(&s,3,Policy::Reuse),Some(3));
    assert_eq!(choose(&[Slot{retired:true,..Slot::default()}],0,Policy::Reuse),None);
 }
}
