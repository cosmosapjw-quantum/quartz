#include <cstdint>
extern "C" {
struct Receipt46 {int32_t status,lo[2],hi[2],ask[4];uint64_t nodes[4],hits,states;};
int pair46(uint64_t,uint64_t,const int32_t*,const int32_t*,const int32_t*,uint64_t,Receipt46*r){r->status=2;return 2;}
}
