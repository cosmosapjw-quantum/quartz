// Separate coordinate geometry, terminal win and forced-response theorem.
// No K42 code, no dead-window shortcut; independent reverse ordering and threshold-key cache.
#include <cstdint>
#include <vector>
#include <array>
#include <unordered_map>
#include <algorithm>
using U=uint64_t;constexpr U ALL=(U(1)<<49)-1;
static const std::array<std::pair<int,int>,4> dirs={{{1,0},{0,1},{1,1},{1,-1}}};
static const std::vector<U> windows=[](){
    std::vector<U> z;
    for(auto [dx,dy]:dirs)for(int y=0;y<7;++y)for(int x=0;x<7;++x){
        int xx=x+3*dx,yy=y+3*dy;if(xx<0||xx>=7||yy<0||yy>=7)continue;
        U w=0;for(int j=0;j<4;++j)w|=U(1)<<((y+j*dy)*7+x+j*dx);z.push_back(w);
    }return z;
}();
bool terminal(U b,int m){
    int x=m%7,y=m/7;
    for(auto [dx,dy]:dirs){
        int n=1;
        for(int sg:{-1,1})for(int j=1;j<=3;++j){int xx=x+sg*j*dx,yy=y+sg*j*dy;if(xx<0||xx>=7||yy<0||yy>=7||!(b&(U(1)<<(yy*7+xx))))break;++n;}
        if(n>=4)return true;
    }return false;
}
U completion(U a,U b){
    U e=ALL^(a|b),out=0;
    for(U w:windows)if((w&b)==0){U x=w&e;if(x&&!(x&(x-1)))out|=x;}
    return out;
}
struct Key{U a,b;int t;bool operator==(const Key&o)const{return a==o.a&&b==o.b&&t==o.t;}};
struct Hash{size_t operator()(const Key&k)const{return (k.a*0x94d049bb133111ebULL)^(k.b*0xbf58476d1ce4e5b9ULL)^k.t;}};
struct Limit{};
struct Solver{
    U nodes=0,cap,hits=0;std::unordered_map<Key,bool,Hash> table;
    explicit Solver(U c):cap(c){}
    bool solve(U a,U b,int last,int t){
        if(terminal(b,last))return false;
        U e=ALL^(a|b);if(!e)return t==0;
        Key k{a,b,t};auto it=table.find(k);if(it!=table.end()){++hits;return it->second;}
        if(nodes==cap)throw Limit{};
        ++nodes;bool ans=false;
        if(completion(a,b))ans=true;
        else{
            U q=completion(b,a);
            if(q&&!(q&(q-1)))ans=!solve(b,a|q,__builtin_ctzll(q),1-t);
            else if(!q){
                for(int m=48;m>=0;--m)if(e&(U(1)<<m)){
                    if(!solve(b,a|(U(1)<<m),m,1-t)){ans=true;break;}
                }
            }
        }
        table.emplace(k,ans);return ans;
    }
};
extern "C" int check46t(U a,U b,int m,int t,U cap,U*nodes,U*hits){
    if(m<0||m>=49||((a|b)&(U(1)<<m))||(a&b)||((a|b)&~ALL)||t<0||t>1)return -2;
    Solver s(cap);int z=-1;try{z=!s.solve(b,a|(U(1)<<m),m,1-t);}catch(const Limit&){}
    *nodes=s.nodes;*hits=s.hits;return z;
}
