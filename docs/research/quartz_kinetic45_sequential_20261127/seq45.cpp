#include "../quartz_kinetic44_incremental_20261127/inc44.cpp"
#include <unordered_map>
#include <functional>
#include <memory>
struct Key45{
    Bits a,b;int p;
    bool operator==(const Key45&k)const{return a==k.a&&b==k.b&&p==k.p;}
};
struct Hash45{
    size_t operator()(const Key45&k)const{
        auto a=std::hash<Bits>{}(k.a),b=std::hash<Bits>{}(k.b);
        return a^(b+size_t(0x9e3779b97f4a7c15ULL)+(a<<6)+(a>>2))^(size_t(k.p)*0x85ebca6bu);
    }
};
struct Depth45{int no=0,yes=32;};
extern "C" {
struct Receipt45 {int32_t status,gate,cert,exhausted,lo[225],hi[225],bin;uint64_t queries,scans,checks,skipped,hits,entries,positive,negative;};
}
struct Session45{
    State st;std::array<Bits,2> stones;std::vector<int>E;std::array<int,225> lo,hi;
    int hi_count[3]={0,0,0},gate=-1,cert=-1,bin=0,last_depth=0;
    uint64_t cap=0,queries=0,checks=0,skipped=0,hits=0,positive=0,negative=0;
    bool exhausted=false;
    std::unordered_map<Key45,Depth45,Hash45> memo;
    Session45(const Geometry&g,const uint8_t*b):st(g,b){
        for(int m=0;m<g.n2;++m)if(b[m])stones[b[m]-1].set(m);
        bin=int(st.s[0].pairs>0)+2*int(st.s[1].pairs>0);
        lo.fill(-2);hi.fill(-2);
        for(int m=0;m<g.n2;++m)if(st.empty[m]){
            E.push_back(m);lo[m]=-1;hi[m]=1;
            Bits o=st.s[1].imm;o.reset(m);
            if(st.s[0].imm[m])lo[m]=hi[m]=1;
            else if(st.empty.count()==1)lo[m]=hi[m]=0;
            else if(o.any())lo[m]=hi[m]=-1;
            else if(st.threats(0,m).count()>=2)lo[m]=hi[m]=1;
            ++hi_count[hi[m]+1];
        }
        if(g.n==7){
            auto rank=[&](int m){return std::make_tuple(-lo[m],!st.s[0].imm[m],std::abs(m/g.n-3)+std::abs(m%g.n-3),m);};
            for(int m:E)if(eligible(m)&&(gate<0||rank(m)<rank(gate)))gate=m;
        }else{
            if(st.s[0].imm.any())gate=st.first(st.s[0].imm);
            else{
                for(int m:E)if(lo[m]==1){gate=m;break;}
                if(gate<0){if(st.s[1].imm.count()==1)gate=st.first(st.s[1].imm);else if(E.size()==1)gate=E.front();}
            }
        }
        cert=gate>=0?gate:findcert();
    }
    bool eligible(int m)const{
        int max_other=-1;
        for(int v=1;v>=-1;--v)if(hi_count[v+1]>(hi[m]==v?1:0)){max_other=v;break;}
        return lo[m]>=max_other;
    }
    int findcert()const{for(int m:E)if(eligible(m))return m;return -1;}
    void set(int m,int v){assert(lo[m]<=v&&v<=hi[m]);--hi_count[hi[m]+1];lo[m]=hi[m]=v;++hi_count[v+1];}
    bool tick(){if(queries==cap){exhausted=true;return false;}++queries;return true;}
    void alter(int m,int p,bool make){st.alter(m,p,make);stones[p].set(m,make);}
    bool after(int p,int m,int d){
        ++checks;if(d<1)return false;
        if(st.s[p].imm[m])return true;
        if(d<3)return false;
        Bits enemy=st.s[1-p].imm;enemy.reset(m);if(enemy.any())return false;
        Bits q=st.threats(p,m);if(q.count()>=2)return true;
        if(!q.any())return false;
        int x=st.first(q);alter(m,p,true);alter(x,1-p,true);
        bool ok=attack(p,d-2);
        alter(x,1-p,false);alter(m,p,false);return ok;
    }
    bool attack(int p,int d){
        if(d<=0)return false;
        Key45 key{stones[0],stones[1],p};
        auto it=memo.find(key);
        if(it!=memo.end()){
            if(d>=it->second.yes){++hits;return true;}
            if(d<=it->second.no){++hits;return false;}
        }
        if(!tick())return false;
        bool ok=false;
        if(!st.s[1-p].wins&&!st.empty.none()){
            assert(!st.s[p].wins);
            if(st.s[p].imm.any())ok=true;
            else if(d>=3&&st.s[1-p].imm.count()<=1){
                Bits e=st.empty;if(st.s[1-p].imm.any())e&=st.s[1-p].imm;
                for(int m=0;m<st.g.n2;++m)if(e[m]&&st.s[p].adj[m].any()){
                    if(after(p,m,d)){ok=true;break;}
                    if(exhausted)break;
                }
            }
        }
        if(ok||!exhausted){
            auto&x=memo[key];
            if(ok){if(d<x.yes){x.yes=d;++positive;}}
            else if(d>x.no){x.no=d;++negative;}
            assert(x.no<x.yes);
        }
        return ok;
    }
    void advance(int d,uint64_t b){
        last_depth=d;cap=b;exhausted=false;
        if(cert>=0||bin==0)return;
        for(int m:E)if(lo[m]<hi[m]&&st.threats(0,m).any()){
            if(!tick())break;
            if(after(0,m,d)){set(m,1);cert=findcert();break;}
            if(exhausted)break;
        }
        if(cert<0&&!exhausted){
            for(int m:E)if(lo[m]<hi[m]){
                assert(st.s[1].total>=st.s[1].support[m]);
                if(st.s[1].total==st.s[1].support[m]){++skipped;continue;}
                if(!tick())break;
                alter(m,0,true);bool loss=attack(1,d-1);alter(m,0,false);
                if(loss)set(m,-1);
                cert=findcert();if(cert>=0||exhausted)break;
            }
        }
        assert((stones[0]&stones[1]).none()&&(stones[0]|stones[1])==(st.g.full&~st.empty));
    }
    void receipt(Receipt45*r)const{
        *r=Receipt45{};r->status=0;r->gate=gate;r->cert=cert;r->exhausted=exhausted;r->bin=bin;
        for(int i=0;i<225;++i){r->lo[i]=lo[i];r->hi[i]=hi[i];}
        r->queries=queries;r->scans=st.scans;r->checks=checks;r->skipped=skipped;r->hits=hits;r->entries=memo.size();r->positive=positive;r->negative=negative;
    }
};
extern "C" {
void* start45(int n,int k,const uint8_t*b){
    if(!((n==7&&k==4)||(n==15&&k==5))||!b)return nullptr;
    const Geometry&g=n==7?g7:g15;
    for(int m=0;m<g.n2;++m)if(b[m]>2)return nullptr;
    try{
        auto s=std::make_unique<Session45>(g,b);
        if(s->st.empty.none()||s->st.s[0].wins||s->st.s[1].wins)return nullptr;
        return s.release();
    }catch(...){return nullptr;}
}
int advance45(void*p,int d,uint64_t cap,Receipt45*r){
    if(!r)return 2;
    *r=Receipt45{};r->status=2;
    if(!p||d<3||d>7)return 2;
    auto&s=*static_cast<Session45*>(p);
    if(d<s.last_depth||cap<s.cap||cap>4096)return 2;
    try{s.advance(d,cap);s.receipt(r);return 0;}catch(...){return 2;}
}
void destroy45(void*p){delete static_cast<Session45*>(p);}
}
