#include <cstdint>
#include <array>
#include <bitset>
#include <vector>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <tuple>
using Bits=std::bitset<225>;
struct Line{std::array<int,5> cells{};};
struct Geometry{
    int n,k,n2;Bits full;std::vector<Line> lines;std::array<std::vector<int>,225> inc;
    Geometry(int n_,int k_):n(n_),k(k_),n2(n*n){
        for(int m=0;m<n2;++m)full.set(m);
        for(int r=0;r<n;++r)for(int c=0;c<n;++c)
          for(auto d:{std::pair<int,int>{0,1},{1,0},{1,1},{1,-1}}){
            int rr=r+(k-1)*d.first,cc=c+(k-1)*d.second;
            if(rr<0||rr>=n||cc<0||cc>=n)continue;
            Line l;int id=lines.size();
            for(int i=0;i<k;++i){int m=(r+i*d.first)*n+c+i*d.second;l.cells[i]=m;inc[m].push_back(id);}lines.push_back(l);
          }
    }
};
static const Geometry g7(7,4),g15(15,5);
struct Side{
    Bits imm;std::array<Bits,225> adj{};
    std::array<uint8_t,225> ic{},support{};
    std::array<std::array<uint8_t,225>,225> pc{};
    int total=0,pairs=0,wins=0;
};
struct State{
    const Geometry&g;std::array<uint8_t,225>b{};std::array<std::array<uint8_t,2>,572> count{};
    std::array<Side,2>s{};Bits empty;uint64_t scans=0;
    State(const Geometry&gg,const uint8_t*board):g(gg){
        for(int m=0;m<g.n2;++m){b[m]=board[m];if(!b[m])empty.set(m);}
        for(size_t i=0;i<g.lines.size();++i){
            ++scans;for(int j=0;j<g.k;++j){int p=b[g.lines[i].cells[j]];if(p)++count[i][p-1];}
            contrib(i,1);
        }
    }
    static void adjust(uint8_t&x,int d){assert(int(x)+d>=0&&int(x)+d<=20);x=uint8_t(int(x)+d);}
    void contrib(int i,int d){
        auto c=count[i];if(c[0]&&c[1])return;
        int p=c[0]?0:1;int n=c[p];
        if(n==g.k){s[p].wins+=d;assert(s[p].wins>=0);return;}
        if(n<g.k-2)return;
        auto &z=s[p];int x=-1,y=-1;
        for(int j=0;j<g.k;++j){int m=g.lines[i].cells[j];if(!b[m]){if(x<0)x=m;else y=m;}}
        assert(x>=0);z.total+=d;adjust(z.support[x],d);
        if(n==g.k-1){
            assert(y<0);adjust(z.ic[x],d);z.imm.set(x,z.ic[x]!=0);
        }else{
            assert(y>=0);z.pairs+=d;adjust(z.support[y],d);adjust(z.pc[x][y],d);adjust(z.pc[y][x],d);
            z.adj[x].set(y,z.pc[x][y]!=0);z.adj[y].set(x,z.pc[y][x]!=0);
        }
        assert(z.total>=0&&z.pairs>=0);
    }
    void alter(int m,int p,bool make){
        assert(make?b[m]==0:b[m]==p+1);
        // Remove while old board is still authoritative; add after board changes.
        for(int i:g.inc[m]){++scans;contrib(i,-1);}
        b[m]=make?p+1:0;empty.set(m,!make);
        for(int i:g.inc[m]){
            if(make)++count[i][p];else{assert(count[i][p]);--count[i][p];}
            contrib(i,1);
        }
    }
    int first(const Bits&x)const{for(int i=0;i<g.n2;++i)if(x[i])return i;return -1;}
    Bits threats(int p,int m)const{Bits z=s[p].imm|s[p].adj[m];z.reset(m);return z;}
};
struct Search{
    State st;uint64_t cap,queries=0,checks=0,skipped=0;bool exhausted=false;
    Search(const Geometry&g,const uint8_t*b,uint64_t c):st(g,b),cap(c){}
    bool tick(){if(queries==cap){exhausted=true;return false;}++queries;return true;}
    bool after(int p,int m,int d){
        ++checks;if(d<1)return false;
        if(st.s[p].imm[m])return true;
        if(d<3)return false;
        Bits enemy=st.s[1-p].imm;enemy.reset(m);if(enemy.any())return false;
        Bits q=st.threats(p,m);if(q.count()>=2)return true;
        if(!q.any())return false;
        int x=st.first(q);st.alter(m,p,true);st.alter(x,1-p,true);
        bool ok=attack(p,d-2);
        st.alter(x,1-p,false);st.alter(m,p,false);return ok;
    }
    bool attack(int p,int d){
        if(d<=0||!tick())return false;
        if(st.s[1-p].wins||st.empty.none())return false;
        assert(!st.s[p].wins);
        if(st.s[p].imm.any())return true;
        if(d<3||st.s[1-p].imm.count()>1)return false;
        Bits e=st.empty;if(st.s[1-p].imm.any())e&=st.s[1-p].imm;
        for(int m=0;m<st.g.n2;++m)if(e[m]&&st.s[p].adj[m].any()){
            if(after(p,m,d))return true;
            if(exhausted)break;
        }
        return false;
    }
};
extern "C" {
struct Receipt {int32_t status,gate,cert,exhausted,lo[225],hi[225],bin;uint64_t queries,scans,checks,skipped;};
int probe44(int n,int k,const uint8_t*board,int d,uint64_t cap,int filter,Receipt*r){
    *r=Receipt{};r->status=2;r->gate=r->cert=-1;
    if(!((n==7&&k==4)||(n==15&&k==5))||d<3||d>31||!board||filter<0||filter>1)return 2;
    const Geometry&g=n==7?g7:g15;
    for(int m=0;m<g.n2;++m)if(board[m]>2)return 2;
    Search s(g,board,cap);auto&st=s.st;
    if(st.empty.none()||st.s[0].wins||st.s[1].wins)return 2;
    r->bin=int(st.s[0].pairs>0)+2*int(st.s[1].pairs>0);
    std::vector<int>E;
    int hi_count[3]={0,0,0};
    for(int m=0;m<225;++m){
        r->lo[m]=r->hi[m]=-2;
        if(m<g.n2&&st.empty[m]){
            E.push_back(m);r->lo[m]=-1;r->hi[m]=1;
            Bits o=st.s[1].imm;o.reset(m);
            if(st.s[0].imm[m])r->lo[m]=r->hi[m]=1;
            else if(st.empty.count()==1)r->lo[m]=r->hi[m]=0;
            else if(o.any())r->lo[m]=r->hi[m]=-1;
            else if(st.threats(0,m).count()>=2)r->lo[m]=r->hi[m]=1;
            ++hi_count[r->hi[m]+1];
        }
    }
    auto eligible=[&](int m){
        int max_other=-1;
        for(int v=1;v>=-1;--v)if(hi_count[v+1]>(r->hi[m]==v?1:0)){max_other=v;break;}
        return r->lo[m]>=max_other;
    };
    auto cert=[&](){for(int m:E)if(eligible(m))return m;return -1;};
    auto set=[&](int m,int v){--hi_count[r->hi[m]+1];r->lo[m]=r->hi[m]=v;++hi_count[v+1];};
    if(n==7){
        auto rank=[&](int m){return std::make_tuple(-r->lo[m],!st.s[0].imm[m],std::abs(m/n-3)+std::abs(m%n-3),m);};
        for(int m:E)if(eligible(m)&&(r->gate<0||rank(m)<rank(r->gate)))r->gate=m;
    }else{
        if(st.s[0].imm.any())r->gate=st.first(st.s[0].imm);
        else{
            for(int m:E)if(r->lo[m]==1){r->gate=m;break;}
            if(r->gate<0){if(st.s[1].imm.count()==1)r->gate=st.first(st.s[1].imm);else if(E.size()==1)r->gate=E.front();}
        }
    }
    r->cert=r->gate>=0?r->gate:cert();
    if(r->cert<0){
        for(int m:E)if(r->lo[m]<r->hi[m]&&st.threats(0,m).any()){
            if(!s.tick())break;
            if(s.after(0,m,d)){set(m,1);r->cert=cert();break;}
            if(s.exhausted)break;
        }
        if(r->cert<0&&!s.exhausted){
            for(int m:E)if(r->lo[m]<r->hi[m]){
                assert(st.s[1].total>=st.s[1].support[m]);
                if(filter&&st.s[1].total==st.s[1].support[m]){++s.skipped;continue;}
                if(!s.tick())break;
                st.alter(m,0,true);bool loss=s.attack(1,d-1);st.alter(m,0,false);
                if(loss)set(m,-1);
                r->cert=cert();if(r->cert>=0||s.exhausted)break;
            }
        }
    }
    r->queries=s.queries;r->scans=st.scans;r->checks=s.checks;r->skipped=s.skipped;r->exhausted=s.exhausted;r->status=0;
    return 0;
}
}
