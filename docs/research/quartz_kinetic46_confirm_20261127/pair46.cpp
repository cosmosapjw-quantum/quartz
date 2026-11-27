#include "../quartz_kinetic42_threshold_20261127/threshold42.cpp"
extern "C" {
struct Receipt46 {int32_t status,lo[2],hi[2],ask[4];uint64_t nodes[4],hits,states;};
int pair46(uint64_t a,uint64_t b,const int32_t*m,const int32_t*l,const int32_t*u,uint64_t cap,Receipt46*r){
    if(!r)return 2;
    *r=Receipt46{};r->status=2;
    for(int i=0;i<4;++i)r->ask[i]=-2;
    if(!m||!l||!u||(a&b)||((a|b)&~FULL)||geo.win(a)||geo.win(b)||cap>1000000)return 2;
    for(int i=0;i<2;++i){
        if(m[i]<0||m[i]>=49||((a|b)&(U(1)<<m[i]))||l[i]<-1||u[i]>1||l[i]>u[i])return 2;
        r->lo[i]=l[i];r->hi[i]=u[i];
    }
    Search s(0);
    try{
        for(int i=0;i<2;++i)for(int t=0;t<=1;++t){
            if(r->lo[i]>=t||r->hi[i]<t)continue;
            int k=2*i+t;
            U prev=s.nodes;s.cap=prev+cap;
            try{
                bool yes=!s.p(b,a|(U(1)<<m[i]),1-t);
                r->ask[k]=yes?1:0;
                if(yes)r->lo[i]=std::max(r->lo[i],t);else r->hi[i]=std::min(r->hi[i],t-1);
            }catch(const Limit&){r->ask[k]=-1;}
            r->nodes[k]=s.nodes-prev;
            assert(r->lo[i]<=r->hi[i]&&r->nodes[k]<=cap);
        }
        if(m[0]==m[1]){
            int ll=std::max(r->lo[0],r->lo[1]),uu=std::min(r->hi[0],r->hi[1]);assert(ll<=uu);
            r->lo[0]=r->lo[1]=ll;r->hi[0]=r->hi[1]=uu;
        }
        r->status=0;r->hits=s.hits;r->states=s.memo.size();
    }catch(...){return 2;}
    return r->status;
}
}
