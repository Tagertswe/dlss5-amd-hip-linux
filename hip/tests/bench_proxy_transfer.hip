// Measures the proxy round-trip cost removed by run_gpu:
//   old = proxy.download (D2H)  +  run()  [run() does the H2D input upload internally]
//   new = run_gpu()             [input already on GPU, no H2D upload]
// gain ≈ download(D2H) + (run - run_gpu)  [= the H2D input upload]
#include "dlss5_network.hpp"
#include "dlss5_launch.hpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
using namespace dlss5;
namespace fs=std::filesystem;
static double now(){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();}
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
 fs::path d=argc>1?argv[1]:std::string("/tmp/opencode/proxy_bench_synth");
 fixture(d);
 const size_t n=1920ull*1080, N=1920ull*1152;
 std::vector<float> proxy(n*4);
 std::mt19937 rng(1u);std::uniform_real_distribution<float> dist(0.f,1.f);
 for(size_t i=0;i<proxy.size();++i)proxy[i]=dist(rng);
 DeviceBuf gproxy(n*16);gproxy.upload(proxy.data(),n*16);
 DeviceBuf gcopy(n*16);
 std::vector<float> host(n*4),out(N*3);
 Network70 net;net.create(d.string());
 // warmup
 net.run(proxy.data(),out.data(),7,nullptr);net.run_gpu(gproxy.as<float>(),out.data(),7,nullptr);
 const int R=8;
 double t0=now();for(int i=0;i<R;++i)hipMemcpy(host.data(),gproxy.p,n*16,hipMemcpyDeviceToHost);double t1=now();
 float dl_ms=(t1-t0)/R;
 double u0=now();for(int i=0;i<R;++i)hipMemcpy(gcopy.p,host.data(),n*16,hipMemcpyHostToDevice);double u1=now();
 float up_ms=(u1-u0)/R;
 double r0=now();for(int i=0;i<R;++i)net.run(proxy.data(),out.data(),7,nullptr);double r1=now();
 float run_ms=(r1-r0)/R;
 double g0=now();for(int i=0;i<R;++i)net.run_gpu(gproxy.as<float>(),out.data(),7,nullptr);double g1=now();
 float rungpu_ms=(g1-g0)/R;
 float h2d_in_upload=run_ms-rungpu_ms;
 std::printf("proxy size = %.1f MB\n",(double)n*16/1048576.0);
 std::printf("download D2H      = %7.2f ms\n",dl_ms);
 std::printf("upload   H2D      = %7.2f ms   (standalone hipMemcpy)\n",up_ms);
 std::printf("run(CPU)          = %7.2f ms   (network + H2D input upload)\n",run_ms);
 std::printf("run_gpu(GPU)      = %7.2f ms   (network, input already on GPU)\n",rungpu_ms);
 std::printf("run - run_gpu     = %7.2f ms   (= H2D input upload inside run)\n",h2d_in_upload);
 std::printf("--------------------------------------------------\n");
 std::printf("proxy round-trip removed = download + H2D upload = %.2f + %.2f = %.2f ms\n",dl_ms,h2d_in_upload,dl_ms+h2d_in_upload);
 return 0;
}catch(const std::exception&e){std::fprintf(stderr,"bench FAIL: %s\n",e.what());return 1;}}
