#include <cstdint>
#include <array>
#include <bitset>
#include <vector>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <tuple>
using Bits=std::bitset<225>;
struct Line{Bits mask;std::array<int,5> cells{};};
struct Geometry{
    int n,k,n2;Bits full;std::vector<Line> lines;
    Geometry(int n_,int k_):n(n_),k(k_),n2(n*n){
        for(int m=0;m<n2;++m)full.set(m);
        for(int r=0;r<n;++r)for(int c=0;c<n;++c)
          for(auto d:{std::pair<int,int>{0,1},{1,0},{1,1},{1,-1}}){
            int rr=r+(k-1)*d.first,cc=c+(k-1)*d.second;
            if(rr<0||rr>=n||cc<0||cc>=n)continue;
            Line l;for(int i=0;i<k;++i){int m=(r+i*d.first)*n+c+i*d.second;l.cells[i]=m;l.mask.set(m);}lines.push_back(l);
          }
    }
};
static const Geometry g7(7,4),g15(15,5);
struct Facts{
    Bits own,opp;
    std::array<Bits,225> adj{};
    bool ownwin=false,oppwin=false;
};
struct Search{
    const Geometry& g;
    uint64_t cap,queries=0,scans=0,checks=0;
    bool exhausted=false;
    Search(const Geometry& gg,uint64_t c):g(gg),cap(c){}
    bool tick(){if(queries==cap){exhausted=true;return false;}++queries;return true;}
    Facts facts(const Bits&a,const Bits&b){
        Facts f;Bits empty=g.full&~(a|b);
        for(const auto&l:g.lines){
            ++scans;size_t ca=(l.mask&a).count(),cb=(l.mask&b).count();
            if(ca==size_t(g.k))f.ownwin=true;
            if(cb==size_t(g.k))f.oppwin=true;
            if(!cb&&ca==size_t(g.k-1))f.own|=l.mask&empty;
            if(!ca&&cb==size_t(g.k-1))f.opp|=l.mask&empty;
            if(!cb&&ca==size_t(g.k-2)){
                int x=-1,y=-1;
                for(int i=0;i<g.k;++i)if(empty[l.cells[i]]){if(x<0)x=l.cells[i];else y=l.cells[i];}
                assert(x>=0&&y>=0);f.adj[x].set(y);f.adj[y].set(x);
            }
        }
        return f;
    }
    int first(const Bits&x)const{for(int i=0;i<g.n2;++i)if(x[i])return i;return -1;}
    Bits threats(const Facts&f,int m){Bits z=f.own|f.adj[m];z.reset(m);return z;}
    // false means no proof obtained, NOT a proof of false.
    bool after(const Bits&a,const Bits&b,const Facts&f,int m,int d){
        ++checks;if(d<1)return false;
        if(f.own[m])return true;
        if(d<3)return false;
        Bits enemy=f.opp;enemy.reset(m);if(enemy.any())return false;
        Bits q=threats(f,m);if(q.count()>=2)return true;
        if(!q.any())return false;
        Bits x=a,y=b;x.set(m);y.set(first(q));return attack(x,y,d-2);
    }
    bool attack(const Bits&a,const Bits&b,int d){
        if(d<=0||!tick())return false;
        auto f=facts(a,b);
        if(f.oppwin||(a|b)==g.full)return false;
        assert(!f.ownwin);
        if(f.own.any())return true;
        if(d<3||f.opp.count()>1)return false;
        Bits e=g.full&~(a|b);
        if(f.opp.any())e&=f.opp;
        for(int m=0;m<g.n2;++m)if(e[m]&&f.adj[m].any()){
            if(after(a,b,f,m,d))return true;
            if(exhausted)break;
        }
        return false;
    }
};
extern "C" {
struct Receipt{int32_t status,gate,cert,exhausted,lo[225],hi[225];uint64_t queries,scans,checks;};
int probe43(int n,int k,const uint8_t*board,int d,uint64_t cap,Receipt*r){
    *r=Receipt{};r->status=2;r->gate=r->cert=-1;
    if(!((n==7&&k==4)||(n==15&&k==5))||d<3||d>31||!board)return 2;
    const Geometry&g=n==7?g7:g15;Bits a,b;
    for(int m=0;m<g.n2;++m){if(board[m]==1)a.set(m);else if(board[m]==2)b.set(m);else if(board[m]!=0)return 2;}
    if((a|b)==g.full)return 2;
    Search s(g,cap);auto f=s.facts(a,b);if(f.ownwin||f.oppwin)return 2;
    std::vector<int>E;Bits e=g.full&~(a|b);
    for(int m=0;m<225;++m){
        r->lo[m]=r->hi[m]=-2;
        if(m<g.n2&&e[m]){
            E.push_back(m);r->lo[m]=-1;r->hi[m]=1;
            Bits o=f.opp;o.reset(m);
            if(f.own[m])r->lo[m]=r->hi[m]=1;
            else if(e.count()==1)r->lo[m]=r->hi[m]=0;
            else if(o.any())r->lo[m]=r->hi[m]=-1;
            else if(s.threats(f,m).count()>=2)r->lo[m]=r->hi[m]=1;
        }
    }
    auto cert=[&](){
        for(int m:E){
            bool ok=true;
            for(int j:E)if(j!=m&&r->lo[m]<r->hi[j]){ok=false;break;}
            if(ok)return m;
        }
        return -1;
    };
    if(n==7){
        auto rank=[&](int m){return std::make_tuple(-r->lo[m],!f.own[m],std::abs(m/n-3)+std::abs(m%n-3),m);};
        for(int m:E){
            bool ok=true;for(int j:E)if(j!=m&&r->lo[m]<r->hi[j]){ok=false;break;}
            if(ok&&(r->gate<0||rank(m)<rank(r->gate)))r->gate=m;
        }
    }else{
        if(f.own.any())r->gate=s.first(f.own);
        else{for(int m:E)if(r->lo[m]==1){r->gate=m;break;}
            if(r->gate<0){if(f.opp.count()==1)r->gate=s.first(f.opp);else if(E.size()==1)r->gate=E.front();}}
    }
    r->cert=r->gate>=0?r->gate:cert();
    if(r->cert<0){
        // Explore winning forcing lines first. Unknown legal actions retain intervals.
        for(int m:E)if(r->lo[m]<r->hi[m]&&s.threats(f,m).any()){
            if(!s.tick())break;
            if(s.after(a,b,f,m,d)){r->lo[m]=r->hi[m]=1;r->cert=cert();break;}
            if(s.exhausted)break;
        }
        if(r->cert<0&&!s.exhausted){
            // A sufficient opponent win proof certifies loss of THIS action.
            for(int m:E)if(r->lo[m]<r->hi[m]){
                if(!s.tick())break;
                Bits x=a;x.set(m);
                if(s.attack(b,x,d-1))r->lo[m]=r->hi[m]=-1;
                r->cert=cert();if(r->cert>=0||s.exhausted)break;
            }
        }
    }
    r->queries=s.queries;r->scans=s.scans;r->checks=s.checks;r->exhausted=s.exhausted;r->status=0;
    return 0;
}
}
