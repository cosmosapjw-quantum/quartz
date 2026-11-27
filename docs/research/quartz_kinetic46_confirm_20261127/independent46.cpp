// Independent terminal-only threshold negamax. No tactical forced-block or dead-window pruning.
#include <cstdint>
#include <array>
#include <algorithm>
#include <unordered_map>
#include <cstdlib>
using U=uint64_t;constexpr U ALL=(U(1)<<49)-1;
struct K{U a,b;int t;bool operator==(const K&o)const{return a==o.a&&b==o.b&&t==o.t;}};
struct H{size_t operator()(const K&k)const{return std::hash<U>{}(k.a)^(std::hash<U>{}(k.b*0x9e3779b97f4a7c15ULL)+size_t(k.t));}};
struct Cap{};
bool winat(U a,int m){
    int x=m%7,y=m/7;
    for(auto d:std::array<std::pair<int,int>,4>{{{1,0},{0,1},{1,1},{1,-1}}}){
        int n=1;
        for(int s:{-1,1})for(int j=1;j<=3;++j){int xx=x+s*j*d.first,yy=y+s*j*d.second;if(xx<0||xx>=7||yy<0||yy>=7||!(a&(U(1)<<(yy*7+xx))))break;++n;}
        if(n>=4)return true;
    }
    return false;
}
struct S{
    U cap,nodes=0,hits=0;std::unordered_map<K,bool,H>memo;
    explicit S(U c):cap(c){}
    bool p(U a,U b,int last,int t){
        if(winat(b,last))return false;
        if((a|b)==ALL)return t==0;
        K k{a,b,t};auto f=memo.find(k);if(f!=memo.end()){++hits;return f->second;}
        if(nodes>=cap)throw Cap{};
        ++nodes;
        // Fixed reverse legal order, not the primary search order.
        bool v=false;
        for(int m=48;m>=0;--m)if(!((a|b)&(U(1)<<m))){
            if(!p(b,a|(U(1)<<m),m,1-t)){v=true;break;}
        }
        memo.emplace(k,v);return v;
    }
};
extern "C" {
int check46(U a,U b,int m,int t,U cap,U*nodes,U*hits){
    if(m<0||m>=49||((a|b)&(U(1)<<m))||(a&b)||((a|b)&~ALL)||t<0||t>1)return -2;
    S s(cap);int v=-1;
    try{v=!s.p(b,a|(U(1)<<m),m,1-t);}catch(const Cap&){}
    *nodes=s.nodes;*hits=s.hits;return v;
}
}
