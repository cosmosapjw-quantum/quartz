from pathlib import Path
p=Path('src/mcts/four_axis.rs');s=p.read_text()
a=s.index('type Leaf25=');b=s.index('// K28 atomic',a)
x=s[a:b]
x=x.replace('type Leaf25=(Key,bool);','''#[derive(Clone,Copy,Debug,PartialEq,Eq,PartialOrd,Ord)]
enum ReceiptKey25{C4(Key),Freestyle15([u64;8])}
fn receipt_key25<G:GameState>(s:&G,gomoku:bool)->ReceiptKey25{
    if gomoku{
        let g=(s as &dyn std::any::Any).downcast_ref::<Gomoku15>().expect("board-only Rapfi Gomoku15 receipt");
        assert_eq!(g.rules,crate::games::gomoku15::GomokuVariant::Freestyle);
        assert_eq!(g.tt_hash(),g.hash());
        ReceiptKey25::Freestyle15(math::placement_key(g.board_as_12(),g.current_player()))
    }else{ReceiptKey25::C4(exact_key(s))}
}
type Leaf25=(ReceiptKey25,bool);''')
x=x.replace('nodes:std::collections::BTreeMap<Key,std::collections::BTreeMap<(Key,usize),Ctx25>>,\n    exact:Merge24,','nodes:std::collections::BTreeMap<ReceiptKey25,std::collections::BTreeMap<(ReceiptKey25,usize),Ctx25>>,\n    ptrs:std::collections::BTreeMap<ReceiptKey25,usize>,\n    reverse_ptrs:std::collections::BTreeMap<usize,ReceiptKey25>,')
x=x.replace('fn observe(&mut self,child:Key,ctx:(Key,usize),','fn observe(&mut self,child:ReceiptKey25,ctx:(ReceiptKey25,usize),')
s=s[:a]+x+s[b:]
s=s.replace('ledger:Mutex<Ledger25>, trace:bool, b:bool,','ledger:Mutex<Ledger25>, trace:bool, b:bool, gomoku:bool,')
s=s.replace('trace,b:false,\n    active:', 'trace,b:false,gomoku:false,\n    active:')
a=s.index('mod receipt25_tests');b=s.index('pub const S:',a)
x=s[a:b]
import re
x=re.sub(r'l\.observe\((\d+),\((\d+),(\d+)\),\((\d+),false\)',r'l.observe(ReceiptKey25::C4(\1),(ReceiptKey25::C4(\2),\3),(ReceiptKey25::C4(\4),false)',x)
s=s[:a]+x+s[b:]
a=s.index('    pub fn set_k25_mask')
s=s[:a]+'''    /// Only the dedicated fixed board-only Rapfi Freestyle15 endpoint may call this.
    pub fn with_k34_rapfi(self,trace:bool)->Self{
        let mut h=self.with_k25(trace);h.k25.as_mut().unwrap().gomoku=true;h
    }
'''+s[a:]
a=s.index('    fn backup25');b=s.index('    pub fn with_merge24',a)
x=s[a:b].replace('exact_key(&state)','receipt_key25(&state,k.gomoku)')
x=x.replace('l.exact.identity(child,&*e.child as *const _ as usize);','''let ptr=&*e.child as *const _ as usize;
            if let Some(old)=l.ptrs.insert(child,ptr){assert_eq!(old,ptr,"exact state must share actual node");}
            if let Some(old)=l.reverse_ptrs.insert(ptr,child){assert_eq!(old,child,"actual node aliases exact states");}''')
s=s[:a]+x+s[b:]
p.write_text(s)
p=Path('src/mcts_server.rs');s=p.read_text()
a='        if matches!(cmd,"research_rapfi_shared_open"|"research_rapfi_shared_search"|"research_rapfi_shared_close"){'
assert s.count(a)==1
s=s.replace(a,'''        if cmd=="research_rapfi_k34"{
            emit_stdout_json_value(&rapfi_shared_cpu_server::handle_k34(&line));
            if let Some(ring)=global_ring_buffer(){ring.set_cmd_done(true);}
            continue;
        }
'''+a)
p.write_text(s)
p=Path('src/mcts_server/rapfi_shared_cpu_server.rs');s=p.read_text()
s+='\n#[path="rapfi_k34.rs"] mod k34;\npub(super) fn handle_k34(line:&str)->Value{k34::handle(line)}\n'
p.write_text(s)
