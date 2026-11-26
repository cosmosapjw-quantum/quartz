#include <cstdint>
extern "C" {
struct Receipt { int32_t status,move,value,lo[49],hi[49];uint64_t nodes,hits,states; };
int solve42(uint64_t,uint64_t,uint64_t,int,Receipt* r){r->status=2;return 2;}
}
