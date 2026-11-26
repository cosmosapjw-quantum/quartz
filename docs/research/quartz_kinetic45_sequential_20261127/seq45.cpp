#include <cstdint>
extern "C" {
struct Receipt45 {int32_t status,gate,cert,exhausted,lo[225],hi[225],bin;uint64_t queries,scans,checks,skipped,hits,entries,positive,negative;};
void* start45(int,int,const uint8_t*){return nullptr;}
int advance45(void*,int,uint64_t,Receipt45*r){r->status=2;return 2;}
void destroy45(void*){}
}
