#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <stdexcept>
using U=uint64_t;
constexpr U FULL=(U(1)<<49)-1;
struct Key { U a,b; bool operator==(const Key& k) const {return a==k.a && b==k.b;} };
struct Hash {size_t operator()(const Key& k) const {return size_t((k.a*0x9e3779b97f4a7c15ULL)^(k.b*0xbf58476d1ce4e5b9ULL));}};
bool win(U a,int m) {
    int r=m/7,c=m%7;
    for(auto d: {std::pair<int,int>{1,0},{0,1},{1,1},{1,-1}}) {
        int n=1;
        for(int s: {-1,1}) {
            int i=r+s*d.first,j=c+s*d.second;
            while(i>=0&&i<7&&j>=0&&j<7&&(a&(U(1)<<(7*i+j)))) {++n;i+=s*d.first;j+=s*d.second;}
        }
        if(n>=4)return true;
    }
    return false;
}
struct Solver {
    std::unordered_map<Key,int,Hash> memo;
    U nodes=0;
    int v(U a,U b,int last) {
        // Only terminal and existential max=1 pruning; no threat or dead-line shortcuts.
        Key k{a,b};auto it=memo.find(k);if(it!=memo.end())return it->second;
        if(++nodes>12000000)throw std::runtime_error("audit bound");
        int z=-1;
        if(win(b,last))z=-1;
        else if((a|b)==FULL)z=0;
        else {
            U e=FULL^(a|b);
            while(e) {
                int m=__builtin_ctzll(e);e&=e-1;
                int val=-v(b,a|(U(1)<<m),m);if(val>z)z=val;
                if(z==1)break;
            }
        }
        memo.emplace(k,z);return z;
    }
};
int main(){
    int id;U a,b;
    while(std::cin>>id>>a>>b){
        Solver s;s.nodes=1;auto t=std::chrono::steady_clock::now();std::vector<std::pair<int,int>> vals;
        bool complete=true;
        try {
            U e=FULL^(a|b);
            while(e){int m=__builtin_ctzll(e);e&=e-1;vals.emplace_back(m,-s.v(b,a|(U(1)<<m),m));}
        }catch(const std::runtime_error&){complete=false;}
        auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-t).count();
        std::cout<<"{\"id\":"<<id<<",\"nodes\":"<<s.nodes<<",\"wall_ns\":"<<ns<<",\"complete\":"<<(complete?"true":"false");
        if(complete){std::cout<<",\"values\":{";bool first=true;for(auto [m,v]:vals){if(!first)std::cout<<",";first=false;std::cout<<"\""<<m<<"\":"<<v;}std::cout<<"}";}
        std::cout<<"}\n";
    }
}
