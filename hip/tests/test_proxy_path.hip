// Verifies the GPU-resident proxy path (run_gpu) is bit-exact with the host path (run).
// Same input bytes -> same rgb bytes (the H2D upload / D2H download are raw copies),
// so the network output must be byte-identical.
#include "dlss5_network.hpp"
#include "dlss5_launch.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
using namespace dlss5;
namespace fs=std::filesystem;
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
int main(int argc,char**argv){try{
 fs::path d=argc>1?argv[1]:std::string("/tmp/opencode/proxy_path_synth");
 fixture(d);
 const size_t n=1920ull*1080;        // input  RGBA  n*16 bytes = n*4 floats
 const size_t N=1920ull*1152;        // output RGB   N*12 bytes = N*3 floats
 std::vector<float> proxy(n*4);
 std::mt19937 rng(0x9e3779b9u);
 std::uniform_real_distribution<float> dist(0.f,1.f);
 for(size_t i=0;i<proxy.size();++i)proxy[i]=dist(rng);
 DeviceBuf gproxy(n*16);gproxy.upload(proxy.data(),n*16);
 std::vector<float> outA(N*3),outB(N*3);
 Network70 net;net.create(d.string());
 // Path A: host input (current in-game path: H2D upload inside run).
 net.run(proxy.data(),outA.data(),7,nullptr);
 // Path B: GPU-resident input (new path: run_gpu, no round-trip).
 net.run_gpu(gproxy.as<float>(),outB.data(),7,nullptr);
 size_t mism=0;long long first=-1;
 for(size_t i=0;i<N*3;++i){if(std::memcmp(&outA[i],&outB[i],4)){if(first<0)first=(long long)i;mism++;}}
 std::printf("proxy path: run(CPU) vs run_gpu(GPU)  bytes=%zu  mismatches=%zu\n",N*3*4,mism);
 if(mism){std::printf("  first diff at output float idx %lld  A=%g B=%g\n",first,(double)outA[(size_t)first],(double)outB[(size_t)first]);return 1;}
 // Cross-check determinism: host path twice must also agree (sanity).
 net.run(proxy.data(),outB.data(),7,nullptr);
 size_t mism2=0;for(size_t i=0;i<N*3;++i)if(std::memcmp(&outA[i],&outB[i],4))++mism2;
 std::printf("proxy path: run(CPU) determinism mismatches=%zu\n",mism2);
 if(mism2)return 1;
 std::printf("PROXY PATH BIT-EXACT PASS (GPU direct == host round-trip)\n");
 return 0;
}catch(const std::exception&e){std::fprintf(stderr,"proxy path FAIL: %s\n",e.what());return 1;}}
