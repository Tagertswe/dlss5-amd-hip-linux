// vit_gather, extracted verbatim from the upstream deep_reference module
// (the only reference kernel the production fast graph dispatches).
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
typedef unsigned uint;
#define KERNEL extern "C" __attribute__((global))
KERNEL void vit_gather(const float* input,const uint* indices,float* output,uint count){
 uint i=__builtin_amdgcn_workgroup_id_x()*256u+__builtin_amdgcn_workitem_id_x();
 if(i<count)output[i]=input[indices[i]];
}
