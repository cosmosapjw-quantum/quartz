#include <cstdint>
extern "C" {
struct Receipt {int32_t status,gate,cert,exhausted,lo[225],hi[225],bin;uint64_t queries,scans,checks,skipped;};
int probe44(int,int,const uint8_t*,int,uint64_t,int,Receipt*r){r->status=2;return 2;}
}
