#include <cstdint>
extern "C" {
struct Receipt{int32_t status,gate,cert,exhausted,lo[225],hi[225];uint64_t queries,scans,checks;};
int probe43(int,int,const uint8_t*,int,uint64_t,Receipt*r){r->status=2;return 2;}
}
