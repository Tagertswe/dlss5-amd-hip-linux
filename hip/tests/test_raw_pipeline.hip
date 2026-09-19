// End-to-end bit-exactness: dlss5_run_frame_raw_gpu (fully GPU-resident, raw device
// pointers) vs the CPU dlss5_run_frame path with the identical logical input:
//   raw packed pixels (bgra8 1920x1080) -> convert -> network -> convert back.
// Both must produce byte-identical packed output (RGB replaced, alpha preserved).
#include "dlss5_capi.h"
#include "dlss5_network.hpp"
#include "dlss5_gpu_interop.hpp"
#include "../../src/native_frame_input_check.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <vector>
using namespace dlss5;
namespace fs=std::filesystem;

// Synthetic weights fixture (same as test_proxy_path.hip) so the test is self-contained.
static void table(const fs::path& d,const std::string& name,size_t n,size_t scale=0){
 std::vector<float> v(n,0);if(scale)std::fill(v.end()-scale,v.end(),1.f);
 if(name=="block0-ffn") {v[0*16+4]=.0625f;v[1*16+13]=1;std::fill(v.end()-32,v.end(),.5f);}
 if(name=="post70-scales") {std::fill(v.begin(),v.begin()+32,0.f);std::fill(v.begin()+32,v.end(),.75f);}
 if(name=="post70-head") {v[0]=.5f;v[32+1]=.5f;}
 if(name.find("-ds")!=std::string::npos||name=="head-matrix") {size_t cin=size_t(std::sqrt(double(n)/2));for(size_t j=0;j<cin*2;++j)v[j*cin+j%cin]=1;}
 if(name.find("-weights")!=std::string::npos) {size_t cout=scale,cin=(n-cout)/cout;for(size_t j=0;j<cout;++j)v[j*cin+j%cin]=1;}
 std::ofstream f(d/(name+".f32"),std::ios::binary);f.write((char*)v.data(),v.size()*4);
}
static void mh(const fs::path&d,int b,int c){std::string p=b==70?"post70":"block"+std::to_string(b);
 table(d,p+"-ffn",c==32?8736:9ull*c*c+c,c);
 table(d,p+"-attention",4ull*c*c+(c/32)*4096+c/32+c,c);
}
static void fixture(const fs::path&d){fs::create_directories(d);
 for(int b=0;b<=70;++b){if(b<=4||b>=66)mh(d,b,32);else if(b<=8||b>=62)mh(d,b,64);else if(b<=14||b>=56)mh(d,b,128);else if(b<=22||(b>=48&&b<=55))mh(d,b,256);else if((b>=23&&b<=30)||(b>=40&&b<=47)){auto p="block"+std::to_string(b);table(d,p+"-ffwd",524288);table(d,p+"-ffwd-projection",262656,512);table(d,p+"-attention",1114640,512);}else if(b>=31&&b<=38){auto p="block"+std::to_string(b);table(d,p+"-expand",4194304);table(d,p+"-contract",4195328,1024);table(d,p+"-qkv",3145760,32);table(d,p+"-projection",1049600,1024);}}
 for(auto bc: {std::pair<int,int>{4,32},{8,64},{14,128},{22,256}})table(d,"block"+std::to_string(bc.first)+"-ds",2ull*bc.second*bc.second);
 table(d,"head-matrix",1024*512);table(d,"decoder39-weights",1024*512+512,512);
 for(auto bc:{std::pair<int,int>{48,256},{56,128},{62,64},{66,32}})table(d,"block"+std::to_string(bc.first)+"-weights",2ull*bc.second*bc.second+bc.second,bc.second);
 table(d,"post70-scales",64);table(d,"post70-head",96);
 std::vector<int> map(655360);std::iota(map.begin(),map.end(),0);
 for(auto name:{"hwc-to-vit.i32","vit-to-hwc.i32"}){std::ofstream f(d/name,std::ios::binary);f.write((char*)map.data(),map.size()*4);}
}

static void NativeResizeChannels(const float*src,unsigned sw,unsigned sh,float*dst,unsigned dw,unsigned dh,unsigned ch){
 if(!src||!dst||!sw||!sh||!dw||!dh||!ch)return;
 if(sw==dw&&sh==dh){std::memcpy(dst,src,size_t(dw)*dh*ch*sizeof(float));return;}
 for(unsigned y=0;y<dh;y++){
  float fy=(y+0.5f)*float(sh)/float(dh)-0.5f;
  int y0=int(floorf(fy)),y1=y0+1;float ty=fy-float(y0);
  if(y0<0){y0=y1=0;ty=0.f;}else if(y0>=int(sh-1)){y0=y1=int(sh-1);ty=0.f;}
  for(unsigned x=0;x<dw;x++){
   float fx=(x+0.5f)*float(sw)/float(dw)-0.5f;
   int x0=int(floorf(fx)),x1=x0+1;float tx=fx-float(x0);
   if(x0<0){x0=x1=0;tx=0.f;}else if(x0>=int(sw-1)){x0=x1=int(sw-1);tx=0.f;}
   const float*p00=src+(size_t(unsigned(y0))*sw+unsigned(x0))*ch;
   const float*p10=src+(size_t(unsigned(y0))*sw+unsigned(x1))*ch;
   const float*p01=src+(size_t(unsigned(y1))*sw+unsigned(x0))*ch;
   const float*p11=src+(size_t(unsigned(y1))*sw+unsigned(x1))*ch;
   float*o=dst+(size_t(y)*dw+x)*ch;
   for(unsigned c=0;c<ch;c++)
    o[c]=(1.f-tx)*(1.f-ty)*p00[c]+tx*(1.f-ty)*p10[c]+(1.f-tx)*ty*p01[c]+tx*ty*p11[c];
  }
 }
}

int main(int argc,char**argv){
 // Weights: argv[1], or "synthetic"/absent for the self-contained fixture.
 fs::path synth="/tmp/opencode/raw_pipeline_synth";
 std::string wdir;
 bool synthetic=false;
 if(argc>1&&std::string(argv[1])!="synthetic")wdir=argv[1];
 else{wdir=synth.string();synthetic=true;}
 if(synthetic)fixture(synth);
 const int frames=argc>2?atoi(argv[2]):2;
 const unsigned W=1920,H=1080;
 const size_t pixbytes=size_t(W)*H*4;
 hipSetDevice(0);
 if(dlss5_init(wdir.c_str(),0)!=0){printf("init failed: %s\n",dlss5_last_error());return 2;}
 std::mt19937 rng(0x51ce);
 std::uniform_int_distribution<int> db(0,255);
 std::vector<unsigned char> raw(pixbytes);
 std::vector<unsigned char> out_cpu(pixbytes),out_gpu(pixbytes);
 std::vector<float> rgba(size_t(W)*H*4),rgb(size_t(W)*H*3);
 int fails=0;
 for(int fr=0;fr<frames;++fr){
  for(auto& c:raw)c=(unsigned char)db(rng);
  // CPU path: packed -> float -> run_frame -> float -> packed (alpha from original).
  NativePixelsToRgbaF32(raw.data(),NativePixelFormat::bgra8_unorm,W,H,rgba);
  NativeResizeChannels(rgba.data(),W,H,rgba.data(),1920,1080,4);
  out_cpu=raw; // NativeRgbF32ToPixels writes RGB into the original packed copy
  Dlss5Frame f{};
  f.struct_size=sizeof f;f.flags=0;f.rgba=rgba.data();f.rgb=rgb.data();
  f.seed=7u+fr;f.reset=1;f.paper_white=f.transfer=f.color=1.f;
  if(dlss5_run_frame(&f)!=0){printf("cpu frame %d failed: %s\n",fr,dlss5_last_error());return 2;}
  NativeResizeChannels(rgb.data(),1920,1080,rgb.data(),W,H,3);
  if(!NativeRgbF32ToPixels(rgb.data(),out_cpu.data(),NativePixelFormat::bgra8_unorm,W,H)){
   printf("cpu pixel convert failed\n");return 2;
  }
  // GPU path: same raw bytes on the GPU, fully GPU-resident.
  DeviceBuf grin(pixbytes),gout(pixbytes);
  grin.upload(raw.data(),pixbytes);
  gout.alloc(pixbytes);
  Dlss5FrameRaw g{};
  g.struct_size=sizeof g;
  g.in_handle=grin.p;g.out_handle=gout.p;
  g.width=W;g.height=H;g.dxgi_format=87;
  g.seed=7u+fr;g.flags=DLSS5_HIP_RAW_FLAGS_DEV_PTRS;
  g.paper_white=g.transfer=g.color=1.f;
  if(dlss5_run_frame_raw_gpu(&g)!=0){printf("gpu frame %d failed: %s\n",fr,dlss5_last_error());++fails;continue;}
   gout.download(out_gpu.data(),pixbytes);
   size_t bad=0;long long first=-1;
   for(size_t i=0;i<pixbytes;++i)if(out_gpu[i]!=out_cpu[i]){if(first<0)first=(long long)i;++bad;}
   if(bad){printf("frame %d: mismatches=%zu/%zu first=%lld cpu=%02x gpu=%02x\n",fr,bad,pixbytes,first,out_cpu[(size_t)first],out_gpu[(size_t)first]);++fails;}
   else printf("frame %d: byte-exact (%zu bytes)\n",fr,pixbytes);
   // Host-pointer mode: CPU-mapped D3D12 buffer surrogates (H2D/D2H staging).
   // Simulates the in-game flow: prefix leaves raw pixels in in and a copy in out.
   std::vector<unsigned char> hin(pixbytes),hout(pixbytes,0);
   std::memcpy(hin.data(),raw.data(),pixbytes);
   std::memcpy(hout.data(),raw.data(),pixbytes);
   Dlss5FrameRaw hp{};
   hp.struct_size=sizeof hp;
   hp.in_handle=hin.data();hp.out_handle=hout.data();
   hp.width=W;hp.height=H;hp.dxgi_format=87;
   hp.seed=7u+fr;hp.flags=DLSS5_HIP_RAW_FLAGS_HOST_PTRS;
   hp.paper_white=hp.transfer=hp.color=1.f;
   if(dlss5_run_frame_raw_gpu(&hp)!=0){printf("host frame %d failed: %s\n",fr,dlss5_last_error());++fails;continue;}
   size_t badh=0;long long firsth=-1;
   for(size_t i=0;i<pixbytes;++i)if(hout[i]!=out_cpu[i]){if(firsth<0)firsth=(long long)i;++badh;}
   if(badh){printf("host frame %d: mismatches=%zu/%zu first=%lld cpu=%02x host=%02x\n",fr,badh,pixbytes,firsth,out_cpu[(size_t)firsth],hout[(size_t)firsth]);++fails;}
   else printf("host frame %d: byte-exact (%zu bytes)\n",fr,pixbytes);
  }
 dlss5_shutdown();
 printf("raw gpu pipeline bit-exact: %s\n",fails?"FAIL":"PASS");
 return fails?1:0;
}
