#include <cstdint>
#include <array>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <stdexcept>
#include <cassert>
using U=uint64_t;
constexpr U FULL=(U(1)<<49)-1;
struct Geometry {
    std::vector<U> lines;
    Geometry(){for(int r=0;r<7;++r)for(int c=0;c<7;++c)
      for(auto d:{std::pair<int,int>{0,1},{1,0},{1,1},{1,-1}}){
        int rr=r+3*d.first,cc=c+3*d.second;
        if(rr<0||rr>=7||cc<0||cc>=7)continue;
        U m=0;for(int k=0;k<4;++k)m|=U(1)<<((r+k*d.first)*7+c+k*d.second);lines.push_back(m);
      }}
    bool win(U a)const{for(U m:lines)if((a&m)==m)return true;return false;}
    U immediate(U a,U b)const{
        U out=0;for(U m:lines)if(!(b&m)&&__builtin_popcountll(a&m)==3)out|=m&~a;return out;
    }
    bool dead(U a,U b)const{
        bool ax=false,bx=false;
        for(U m:lines){ax|=!(m&b);bx|=!(m&a);if(ax&&bx)return false;}
        return !ax&&!bx;
    }
};
static const Geometry geo;
struct Key{U a,b;bool operator==(const Key& k)const{return a==k.a&&b==k.b;}};
struct Hash{size_t operator()(const Key& k)const{return size_t((k.a*0x9e3779b97f4a7c15ULL)^(k.b*0xbf58476d1ce4e5b9ULL));}};
struct Bounds{int8_t lo=-1,hi=1;};
struct Limit{};
struct Search{
    U cap,nodes=0,hits=0;
    std::unordered_map<Key,Bounds,Hash> memo;
    explicit Search(U c):cap(c){}
    bool p(U a,U b,int t){
        Key key{a,b};auto it=memo.find(key);
        if(it!=memo.end()){
            if(it->second.lo>=t){++hits;return true;}
            if(it->second.hi<t){++hits;return false;}
        }
        if(nodes>=cap)throw Limit{};
        ++nodes;
        int exact=2;bool answer=false;U e=FULL^(a|b);
        if(geo.win(b))exact=-1;
        else if(!e)exact=0;
        else if(geo.immediate(a,b))exact=1;
        else{
            U threats=geo.immediate(b,a);
            if(threats&(threats-1))exact=-1;
            else if(threats)answer=!p(b,a|threats,1-t);
            else if(geo.dead(a,b))exact=0;
            else{
                while(e){U m=e&(-e);e^=m;
                    if(!p(b,a|m,1-t)){answer=true;break;}
                }
            }
        }
        auto& z=memo[key];
        if(exact!=2){assert(z.lo<=exact&&exact<=z.hi);z.lo=z.hi=exact;answer=exact>=t;}
        else if(answer)z.lo=std::max<int>(z.lo,t);
        else z.hi=std::min<int>(z.hi,t-1);
        assert(z.lo<=z.hi);return answer;
    }
};
extern "C" {
struct Receipt { int32_t status,move,value,lo[49],hi[49];uint64_t nodes,hits,states; };
int solve42(U a,U b,U budget,int census,Receipt* r){
    *r=Receipt{};r->status=2;r->move=-1;r->value=2;
    if((a&b)||((a|b)&~FULL)||geo.win(a)||geo.win(b)||!(FULL^(a|b))||__builtin_popcountll(FULL^(a|b))>16||census<0||census>1)return 2;
    Search s(budget);std::vector<int> legal;
    for(int m=0;m<49;++m){r->lo[m]=-2;r->hi[m]=-2;if(!((a|b)&(U(1)<<m))){legal.push_back(m);r->lo[m]=-1;r->hi[m]=1;}}
    auto ask=[&](int m,int t){
        if(r->lo[m]>=t)return true;
        if(r->hi[m]<t)return false;
        bool ok=!s.p(b,a|(U(1)<<m),1-t);
        if(ok)r->lo[m]=std::max(r->lo[m],t);else r->hi[m]=std::min(r->hi[m],t-1);
        return ok;
    };
    try{
        if(census){
            for(int m:legal){if(!ask(m,1))ask(m,0);}
            for(int m:legal){assert(r->lo[m]==r->hi[m]);if(r->move<0||r->lo[m]>r->value){r->move=m;r->value=r->lo[m];}}
        }else{
            for(int t:{1,0}){
                for(int m:legal)if(ask(m,t)){r->move=m;r->value=t;break;}
                if(r->move>=0)break;
            }
            if(r->move<0){r->move=legal.front();r->value=-1;}
        }
        assert(r->lo[r->move]==r->value&&r->hi[r->move]==r->value);
        for(int m:legal){assert(r->hi[m]<=r->value);if(m<r->move)assert(r->hi[m]<r->value);}
        r->status=0;
    }catch(const Limit&){r->status=1;r->move=-1;r->value=2;}
    catch(...){r->status=2;r->move=-1;r->value=2;}
    r->nodes=s.nodes;r->hits=s.hits;r->states=s.memo.size();return r->status;
}
}
