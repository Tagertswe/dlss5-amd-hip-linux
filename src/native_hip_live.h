#pragma once
// Record ONLY on the original FFX native list, immediately after successful
// ffxDispatch. No submission, list lifecycle changes, or graphics waits here.
#include <d3d12.h>
#include <dxgi1_4.h>
#include <atomic>
#include <cstdio>
#include <memory>
#include <mutex>
#include <thread>
#include <limits>
#include "native_hip_client.h"
#include "native_device_identity.h"
#include "../hip/include/dlss5_submit.h"

class NativeHipLive {
    template<class T> struct Ref {
        T* p{};
        Ref()=default;
        explicit Ref(T* v):p(v){if(p)p->AddRef();}
        Ref(const Ref&)=delete;
        Ref& operator=(const Ref&)=delete;
        ~Ref(){if(p)p->Release();}
        T** Out(){return &p;}
        T* operator->()const{return p;}
        void Reset(){if(p){p->Release();p=nullptr;}}
    };
    static void Log(const char* event,uint64_t frame=0,const char* detail="",size_t changed=0) noexcept {
        try {
            static std::mutex mutex;
            std::lock_guard<std::mutex> lock(mutex);
            CreateDirectoryW(NativeLabPath(L"logs").c_str(),nullptr);
            if(FILE* f=_wfopen(NativeLabPath(L"logs\\native-hip-live.txt").c_str(),L"ab")){
                fprintf(f,"pid=%lu frame_id=%llu event=%s changed_rgb_bytes=%llu detail=%s\n",
                    GetCurrentProcessId(),static_cast<unsigned long long>(frame),event,
                    static_cast<unsigned long long>(changed),detail);
                fclose(f);
            }
        }catch(...){}
    }
    struct State {
        Ref<ID3D12Device> device;
        std::shared_ptr<NativeHipClient> client;
        std::atomic<int> init{0}; // 0 cold, 1 initializing, 2 ready, -1 failed
        std::atomic<bool> disabled{false},key_down{false};
        std::atomic<uint64_t> generation{0},queue{0}; // even generation = enabled
        std::atomic<unsigned> inflight{0};
        std::mutex recording,processing;
        bool display_srgb{};
        // V3 GPU-resident frame buffers, shared by all jobs: the D3D queue and the
        // serialized HIP callbacks order every in/out use in frame order, so one
        // pair is race-free. Two backing modes, picked once per geometry:
        //  - shared_mode: DEFAULT-heap D3D12_HEAP_FLAG_SHARED buffers + Win32 shared
        //    handles (vkd3d exports OPAQUE_WIN32 -> dmabuf; the .so imports the
        //    handle into HIP: fully GPU-resident, zero staging copies);
        //  - mapped fallback: in_buf READBACK (GPU writes, CPU reads), out_buf
        //    UPLOAD (CPU writes, GPU read), mapped for the state lifetime; the host
        //    pointers travel to the .so, which stages H2D/D2H around the GPU pipeline.
        Ref<ID3D12Resource> in_buf,out_buf;
        HANDLE in_handle{},out_handle{};
        void* in_host{},* out_host{};
        bool shared_mode{};
        // Tracked D3D12 state of the persistent buffers (both are created in
        // COPY_DEST). This vkd3d build has NO implicit-barrier support: every
        // transition must use valid explicit states, and an unknown state
        // invalidates the command list and trips the device-removed guard.
        D3D12_RESOURCE_STATES in_state=D3D12_RESOURCE_STATE_COPY_DEST;
        D3D12_RESOURCE_STATES out_state=D3D12_RESOURCE_STATE_COPY_DEST;
        unsigned buf_w{},buf_h{},buf_dxgi{};
        size_t buf_bytes{};
        uint32_t buf_gen{0};
        ~State(){
            if(in_buf.p&&in_host){in_buf.p->Unmap(0,nullptr);in_host=nullptr;}
            if(out_buf.p&&out_host){out_buf.p->Unmap(0,nullptr);out_host=nullptr;}
            if(in_handle)CloseHandle(in_handle);
            if(out_handle)CloseHandle(out_handle);
        }
    };
    std::shared_ptr<State> state_=std::make_shared<State>();
    static constexpr unsigned MaxInflight=3;
    // This anchor is needed even on a runtime that ignores SetMarker: prefix
    // copy resources must not die while the recorded list can still execute.
    // Rejected markers stay anchored until LIST DESTRUCTION, conservatively
    // (D3D12 has no allocator-reset notification available to this caller).
    inline static constexpr GUID AnchorId={0xda06bd21,0x147c,0x467a,{0x89,0xd3,0x6d,0xc0,0xd8,0xa9,0x64,0xd3}};
    static GUID NextAnchor(){
        static std::atomic<uint64_t> serial{0};
        const uint64_t n=serial.fetch_add(1);GUID id=AnchorId;
        id.Data1+=uint32_t(n);id.Data2+=uint16_t(n>>32);id.Data3+=uint16_t(n>>48);return id;
    }
    struct DeviceResources {
        Ref<ID3D12Device> device;
        Ref<ID3D12Resource> color;
        void Bind(ID3D12Device* d,ID3D12Resource* c){device.p=d;d->AddRef();color.p=c;c->AddRef();}
    };
    struct Job final: IUnknown {
        std::atomic<ULONG> refs{1};
        std::atomic<bool> called{false};
        const std::shared_ptr<State> owner;
        const uint64_t frame,generation;
        const GUID anchor=NextAnchor();
        const NativePixelFormat format;
        const unsigned width,height;
        const unsigned dxgi;
        DeviceResources resources;
        // Own references to this job's buffer pair, captured at Record so an
        // in-flight job cannot observe a recreation (geometry change) racing the
        // callback. The buffer refs keep the resources (and any shared-handle
        // import built on them) alive until the job completes.
        Ref<ID3D12Resource> in_res,out_res;
        HANDLE in_handle{},out_handle{};
        void* in_host{},* out_host{};
        bool shared_mode{};
        uint32_t buf_gen{0};
        void BindBuffers(const State& s){
            in_res.p=s.in_buf.p;s.in_buf.p->AddRef();
            out_res.p=s.out_buf.p;s.out_buf.p->AddRef();
            shared_mode=s.shared_mode;
            in_handle=s.in_handle;out_handle=s.out_handle;
            in_host=s.in_host;out_host=s.out_host;
            buf_gen=s.buf_gen;
        }
        Job(std::shared_ptr<State> s,uint64_t f,uint64_t g,NativePixelFormat p,unsigned w,unsigned h,unsigned d):
            owner(std::move(s)),frame(f),generation(g),format(p),width(w),height(h),dxgi(d){}
        ~Job(){owner->inflight.fetch_sub(1);}
        HRESULT STDMETHODCALLTYPE QueryInterface(REFIID id,void** out) override {
            if(!out)return E_POINTER;
            *out=nullptr;if(!(id==IID_IUnknown))return E_NOINTERFACE;
            *out=static_cast<IUnknown*>(this);AddRef();return S_OK;
        }
        ULONG STDMETHODCALLTYPE AddRef() override{return refs.fetch_add(1)+1;}
        ULONG STDMETHODCALLTYPE Release() override{ULONG n=refs.fetch_sub(1)-1;if(!n)delete this;return n;}
        static void DLSS5_PE_ABI Retain(void* p) noexcept {static_cast<Job*>(p)->AddRef();}
        static void DLSS5_PE_ABI ReleaseContext(void* p) noexcept {static_cast<Job*>(p)->Release();}
        static int DLSS5_PE_ABI Run(void* p,uint64_t queue) noexcept {
            auto& j=*static_cast<Job*>(p);
            try{return j.Process(queue);}catch(...){Log("callback_failure",j.frame,"device must be lost; suffix not safe");return -1;}
        }
        // V3 GPU-resident callback: no CPU processing. The prefix already copied
        // color->in_buf and in_buf->out_buf (out_buf is the original-frame
        // fallback), so on ANY rejection/failure the suffix writes back the
        // original pixels exactly like the old CPU fallback path. In shared mode
        // the .so imports the shared handles into HIP (zero staging copies); in
        // mapped mode it stages the mapped in/out buffers H2D/D2H (HOST_PTRS).
        int Process(uint64_t queue){
            if(called.exchange(true)){Log("duplicate_callback",frame,"resubmitted command list rejected");return -1;}
            uint64_t expected=0;
            if(!queue||(!owner->queue.compare_exchange_strong(expected,queue)&&expected!=queue)){
                Log("queue_conflict",frame,"callback rejected before GPU frame");return -1;
            }
            std::unique_lock<std::mutex> serial(owner->processing,std::try_to_lock);
            // Prefix already copied original pixels to out_buf. A busy skip must
            // return 0 so vkd3d submits the suffix; -1 marks the device lost.
            if(!serial.owns_lock()){Log("busy",frame,"previous HIP frame still running; original input");return 0;}
            if(FAILED(resources.device->GetDeviceRemovedReason()))return -1;
            if(!owner->client||!owner->client->HasRawGpu()){
                Log("processed",frame,"V3 raw bridge missing; original input");return 0;
            }
            auto bypass=[&]{return (generation&1)||owner->disabled.load()||owner->generation.load()!=generation;};
            if(bypass()){Log("processed",frame,"F6 bypass; original input");return 0;}
            try {
                const bool use_host=!shared_mode;
                if(owner->client->RunFrameRaw(use_host?in_host:in_handle,use_host?out_host:out_handle,
                                              width,height,dxgi,unsigned(frame),owner->display_srgb,use_host,buf_gen)){
                    // out_buf still holds the prefix in->out copy (original pixels).
                    Log("processed",frame,"HIP failed; original input");return 0;
                }
                if(bypass())Log("processed",frame,"F6 changed during GPU frame; new pixels kept, bypasses next frame");
                else Log("processed",frame,"live HIP network GPU-resident; reset=1 (not temporal)");
                return 0;
            }catch(...){Log("processed",frame,"HIP exception; original input");return 0;}
        }
    };
    static bool ValidState(D3D12_RESOURCE_STATES s){
        switch(s){
        case D3D12_RESOURCE_STATE_COMMON:case D3D12_RESOURCE_STATE_RENDER_TARGET:
        case D3D12_RESOURCE_STATE_UNORDERED_ACCESS:case D3D12_RESOURCE_STATE_COPY_SOURCE:
        case D3D12_RESOURCE_STATE_COPY_DEST:case D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE:
        case D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE:
        case D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE:return true;
        default:return false;
        }
    }
    static D3D12_TEXTURE_COPY_LOCATION BufLoc(ID3D12Resource* buf,const D3D12_RESOURCE_DESC& desc,size_t bpp){
        D3D12_TEXTURE_COPY_LOCATION l{};
        l.pResource=buf;l.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        l.PlacedFootprint.Offset=0;
        l.PlacedFootprint.Footprint.Width=desc.Width;
        l.PlacedFootprint.Footprint.Height=desc.Height;
        l.PlacedFootprint.Footprint.Depth=1;
        l.PlacedFootprint.Footprint.Format=desc.Format;
        l.PlacedFootprint.Footprint.RowPitch=desc.Width*(UINT)bpp;
        return l;
    }
    // Explicit transition barrier. This vkd3d build has NO implicit-barrier
    // support: only valid D3D12 states are accepted, an unknown state invalidates
    // the command list (and then the submit extension trips device-removed).
    static D3D12_RESOURCE_BARRIER Bar(ID3D12Resource* r,D3D12_RESOURCE_STATES from,D3D12_RESOURCE_STATES to){
        D3D12_RESOURCE_BARRIER b{};b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition={r,0,from,to};
        return b;
    }
    static void ReleaseBuffers(State& s){
        if(s.in_buf.p&&s.in_host){s.in_buf.p->Unmap(0,nullptr);s.in_host=nullptr;}
        if(s.out_buf.p&&s.out_host){s.out_buf.p->Unmap(0,nullptr);s.out_host=nullptr;}
        if(s.in_handle){CloseHandle(s.in_handle);s.in_handle=nullptr;}
        if(s.out_handle){CloseHandle(s.out_handle);s.out_handle=nullptr;}
        s.in_buf.Reset();s.out_buf.Reset();s.shared_mode=false;
    }
    // Per-step failure detail; callers are serialized by the recording mutex.
    static const char* BufferFail(const char* what,HRESULT hr=0){
        static char detail[96];
        if(hr)std::snprintf(detail,sizeof detail,"%s hr=0x%08lX",what,(unsigned long)hr);
        else std::snprintf(detail,sizeof detail,"%s",what);
        return detail;
    }
    static D3D12_RESOURCE_DESC BufferDesc(const D3D12_RESOURCE_DESC& desc,size_t bpp){
        D3D12_RESOURCE_DESC buffer{};buffer.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;
        buffer.Width=UINT64(size_t(unsigned(desc.Width))*desc.Height*bpp);
        buffer.Height=1;buffer.DepthOrArraySize=buffer.MipLevels=1;
        buffer.SampleDesc.Count=1;buffer.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        return buffer;
    }
    // Zero-copy mode: DEFAULT-heap buffers created with D3D12_HEAP_FLAG_SHARED.
    // vkd3d then exports the allocation as OPAQUE_WIN32 (dma-buf under Wine) and
    // CreateSharedHandle returns a real NT handle the .so imports into HIP.
    static const char* TrySharedBuffers(State& s,ID3D12Device* d,const D3D12_RESOURCE_DESC& desc,size_t bpp){
        const D3D12_RESOURCE_DESC buffer=BufferDesc(desc,bpp);
        D3D12_HEAP_PROPERTIES heap{};heap.CreationNodeMask=heap.VisibleNodeMask=1;
        heap.Type=D3D12_HEAP_TYPE_DEFAULT;
        HRESULT hr=d->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_SHARED,&buffer,
            D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(s.in_buf.Out()));
        if(FAILED(hr))return BufferFail("shared_in_create",hr);
        hr=d->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_SHARED,&buffer,
            D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(s.out_buf.Out()));
        if(FAILED(hr)){ReleaseBuffers(s);return BufferFail("shared_out_create",hr);}
        const DWORD access=GENERIC_READ|GENERIC_WRITE;
        hr=d->CreateSharedHandle(s.in_buf.p,nullptr,access,nullptr,&s.in_handle);
        if(FAILED(hr)){ReleaseBuffers(s);return BufferFail("shared_in_handle",hr);}
        hr=d->CreateSharedHandle(s.out_buf.p,nullptr,access,nullptr,&s.out_handle);
        if(FAILED(hr)){ReleaseBuffers(s);return BufferFail("shared_out_handle",hr);}
        s.shared_mode=true;
        return nullptr;
    }
    // Fallback mode: CPU-mapped READBACK-in / UPLOAD-out buffers; the host
    // pointers go to the .so (HOST_PTRS mode, H2D/D2H staging). Initial states
    // and mapping ranges match the proven production configuration.
    static const char* TryMappedBuffers(State& s,ID3D12Device* d,const D3D12_RESOURCE_DESC& desc,size_t bpp){
        const D3D12_RESOURCE_DESC buffer=BufferDesc(desc,bpp);
        D3D12_HEAP_PROPERTIES heap{};heap.CreationNodeMask=heap.VisibleNodeMask=1;
        heap.Type=D3D12_HEAP_TYPE_READBACK;
        HRESULT hr=d->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&buffer,
            D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(s.in_buf.Out()));
        if(FAILED(hr))return BufferFail("readback_create",hr);
        heap.Type=D3D12_HEAP_TYPE_UPLOAD;
        hr=d->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&buffer,
            D3D12_RESOURCE_STATE_GENERIC_READ,nullptr,IID_PPV_ARGS(s.out_buf.Out()));
        if(FAILED(hr)){ReleaseBuffers(s);return BufferFail("upload_create",hr);}
        const D3D12_RANGE full{0,(SIZE_T)(size_t(unsigned(desc.Width))*desc.Height*bpp)},none{};
        hr=s.in_buf.p->Map(0,&full,&s.in_host);
        if(FAILED(hr)){ReleaseBuffers(s);return BufferFail("in_map",hr);}
        hr=s.out_buf.p->Map(0,&none,&s.out_host);
        if(FAILED(hr)){ReleaseBuffers(s);return BufferFail("out_map",hr);}
        return nullptr;
    }
    // V3: one in/out buffer pair per (device, geometry, format), recreated on any
    // change. Prefers the zero-copy shared-handle mode; falls back to mapped
    // buffers. Returns nullptr on success, a reason string else.
    static const char* EnsureBuffers(State& s,ID3D12Device* d,const D3D12_RESOURCE_DESC& desc,size_t bpp){
        const unsigned w=unsigned(desc.Width),h=desc.Height,dxgi=unsigned(desc.Format);
        if(s.in_buf.p&&s.buf_w==w&&s.buf_h==h&&s.buf_dxgi==dxgi)return nullptr;
        // BufLoc assumes a packed footprint (RowPitch == width*bpp). Reject any
        // row-padded layout before we allocate. Same check as the old Create().
        {
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};UINT rows=0;UINT64 rowbytes=0,total=0;
            d->GetCopyableFootprints(&desc,0,1,0,&fp,&rows,&rowbytes,&total);
            const UINT64 packed=UINT64(w)*bpp;
            if(rows!=h||rowbytes!=packed||fp.Footprint.Width!=desc.Width||
               fp.Footprint.Height!=h||fp.Footprint.Depth!=1||fp.Footprint.Format!=desc.Format||
               fp.Footprint.RowPitch<packed||!total||
               UINT64(h-1)*fp.Footprint.RowPitch+packed>total-fp.Offset)return BufferFail("packed_footprint_check");
        }
        ReleaseBuffers(s);
        const char* err=TrySharedBuffers(s,d,desc,bpp);
        if(err)err=TryMappedBuffers(s,d,desc,bpp);
        if(err)return err;
        s.in_state=s.out_state=D3D12_RESOURCE_STATE_COPY_DEST;
        s.buf_w=w;s.buf_h=h;s.buf_dxgi=dxgi;s.buf_bytes=size_t(w)*h*bpp;s.buf_gen++;
        return nullptr;
    }
    // Prefix: color -> in_buf, then in_buf -> out_buf. The second copy keeps
    // out_buf an exact original-frame fallback until the .so overwrites its RGB.
    // The color is transitioned from its KNOWN `state` and restored, matching
    // the old Copy() semantics. Our own buffers use explicit tracked states
    // (s.in_state/s.out_state) — no implicit barriers on this vkd3d build.
    static void CopyPrefix(ID3D12GraphicsCommandList* list,State& s,D3D12_RESOURCE_STATES state,
                           ID3D12Resource* color,const D3D12_RESOURCE_DESC& desc,size_t bpp){
        D3D12_RESOURCE_BARRIER b=Bar(color,state,D3D12_RESOURCE_STATE_COPY_SOURCE);
        if(state!=D3D12_RESOURCE_STATE_COPY_SOURCE)list->ResourceBarrier(1,&b);
        if(s.in_state!=D3D12_RESOURCE_STATE_COPY_DEST){
            b=Bar(s.in_buf.p,s.in_state,D3D12_RESOURCE_STATE_COPY_DEST);list->ResourceBarrier(1,&b);
            s.in_state=D3D12_RESOURCE_STATE_COPY_DEST;
        }
        D3D12_TEXTURE_COPY_LOCATION texture{},inb=BufLoc(s.in_buf.p,desc,bpp);
        texture.pResource=color;texture.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        list->CopyTextureRegion(&inb,0,0,0,&texture,nullptr);
        if(state!=D3D12_RESOURCE_STATE_COPY_SOURCE){
            b=Bar(color,D3D12_RESOURCE_STATE_COPY_SOURCE,state);list->ResourceBarrier(1,&b);
        }
        if(s.in_state!=D3D12_RESOURCE_STATE_COPY_SOURCE){
            b=Bar(s.in_buf.p,s.in_state,D3D12_RESOURCE_STATE_COPY_SOURCE);list->ResourceBarrier(1,&b);
            s.in_state=D3D12_RESOURCE_STATE_COPY_SOURCE;
        }
        if(s.out_state!=D3D12_RESOURCE_STATE_COPY_DEST){
            b=Bar(s.out_buf.p,s.out_state,D3D12_RESOURCE_STATE_COPY_DEST);list->ResourceBarrier(1,&b);
            s.out_state=D3D12_RESOURCE_STATE_COPY_DEST;
        }
        list->CopyResource(s.out_buf.p,s.in_buf.p);
    }
    // Suffix: out_buf -> color. Restores the color to its known `state`.
    static void CopySuffix(ID3D12GraphicsCommandList* list,State& s,D3D12_RESOURCE_STATES state,
                           ID3D12Resource* color,const D3D12_RESOURCE_DESC& desc,size_t bpp){
        D3D12_RESOURCE_BARRIER b=Bar(s.out_buf.p,s.out_state,D3D12_RESOURCE_STATE_COPY_SOURCE);
        if(s.out_state!=D3D12_RESOURCE_STATE_COPY_SOURCE)list->ResourceBarrier(1,&b);
        s.out_state=D3D12_RESOURCE_STATE_COPY_SOURCE;
        b=Bar(color,state,D3D12_RESOURCE_STATE_COPY_DEST);
        if(state!=D3D12_RESOURCE_STATE_COPY_DEST)list->ResourceBarrier(1,&b);
        D3D12_TEXTURE_COPY_LOCATION texture{},outb=BufLoc(s.out_buf.p,desc,bpp);
        texture.pResource=color;texture.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        list->CopyTextureRegion(&texture,0,0,0,&outb,nullptr);
        if(state!=D3D12_RESOURCE_STATE_COPY_DEST){
            b=Bar(color,D3D12_RESOURCE_STATE_COPY_DEST,state);list->ResourceBarrier(1,&b);
        }
        b=Bar(s.out_buf.p,s.out_state,D3D12_RESOURCE_STATE_COPY_DEST);
        if(s.out_state!=D3D12_RESOURCE_STATE_COPY_DEST)list->ResourceBarrier(1,&b);
        s.out_state=D3D12_RESOURCE_STATE_COPY_DEST;
    }
    static void Initialize(std::shared_ptr<State> s) noexcept {
        try {
            Ref<IDXGIFactory4> factory;Ref<IDXGIAdapter1> adapter;
            DXGI_ADAPTER_DESC1 desc{};
            if(FAILED(CreateDXGIFactory1(IID_PPV_ARGS(factory.Out())))||
               FAILED(factory->EnumAdapterByLuid(s->device->GetAdapterLuid(),IID_PPV_ARGS(adapter.Out())))||
               FAILED(adapter->GetDesc1(&desc)))throw std::runtime_error("DXGI adapter identification failed");
            char name[512]{};
            if(!WideCharToMultiByte(CP_UTF8,0,desc.Description,-1,name,sizeof name,nullptr,nullptr)||!name[0])
                throw std::runtime_error("DXGI adapter name missing");
            // DXVK appends the Vulkan driver/ISA label; HIP exposes the base
            // product name. Strip this known trailing decoration only.
            if(char* driver=std::strstr(name," (RADV ")){
                if(name[std::strlen(name)-1]==')')*driver=0;
            }
            auto client=std::make_shared<NativeHipClient>();client->Create(name);
            if(!client->Ready())throw std::runtime_error("HIP client not ready");
            s->client=std::move(client);
            Log("ready",0,name);Log("codec",0,s->display_srgb?"explicit display-sRGB; DLSS5_CODEC_SRGB=1":"linear mode=1 paper_white=1");
            Log("live_network",0,"enabled; F6 toggles bypass; reset every frame, NOT temporal; no performance claim");
            s->init.store(2,std::memory_order_release);
        }catch(const std::exception& e){Log("init_failed",0,e.what());s->init=-1;s->disabled=true;}
        catch(...){Log("init_failed");s->init=-1;s->disabled=true;}
    }
public:
    NativeHipLive()=default;
    NativeHipLive(const NativeHipLive&)=delete;
    NativeHipLive& operator=(const NativeHipLive&)=delete;
    // True means accepted marker AND suffix recorded, not GPU completion.
    // Owner destruction cannot invalidate background init or retained jobs.
    bool Record(ID3D12GraphicsCommandList* list,ID3D12Resource* color,D3D12_RESOURCE_STATES state,uint64_t frame_id) noexcept {
        auto s=state_;
        const bool down=(GetAsyncKeyState(VK_F6)&0x8000)!=0;
        const bool was=s->key_down.exchange(down);
        if(down&&!was){auto g=s->generation.fetch_add(1)+1;Log("F6",frame_id,(g&1)?"bypass; original pixels":"enabled; live network (not temporal)");}
        const uint64_t generation=s->generation.load();
        try {
            std::unique_lock<std::mutex> lock(s->recording,std::try_to_lock);
            if(!lock.owns_lock()||s->disabled||(generation&1)||!list||!color)return false;
            Ref<ID3D12GraphicsCommandList> hold_list(list);
            if(list->GetType()!=D3D12_COMMAND_LIST_TYPE_DIRECT)return false;
            Ref<ID3D12Device> device,color_device;
            if(FAILED(list->GetDevice(IID_PPV_ARGS(device.Out())))||
               FAILED(color->GetDevice(IID_PPV_ARGS(color_device.Out())))||
               !NativeSameDevice(device.p,color_device.p))return false;
            // Never unwrap the foreign resource. Only validate device identity.
            const auto desc=color->GetDesc();const auto format=NativePixelFormatFromDxgi(unsigned(desc.Format));
            if(desc.Dimension!=D3D12_RESOURCE_DIMENSION_TEXTURE2D||desc.Width>7680||
               !NativePixelGeometry(unsigned(desc.Width),desc.Height)||desc.MipLevels!=1||
               desc.DepthOrArraySize!=1||desc.SampleDesc.Count!=1||desc.SampleDesc.Quality!=0||
               (desc.Flags&D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS)||
               !NativePixelBytes(format)||!ValidState(state))return false;
            if(s->device.p&&!NativeSameDevice(s->device.p,device.p))return false;
            if(s->init==0){
                // Pin this add-on's callback code as well as its heap state. No
                // FreeLibrary race on detached initialization / allocator refs.
                HMODULE module=nullptr;
                if(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_PIN,
                    reinterpret_cast<LPCWSTR>(&Initialize),&module)){s->disabled=true;return false;}
                s->device.p=device.p;device->AddRef();
                const wchar_t* value=_wgetenv(L"DLSS5_CODEC_SRGB");s->display_srgb=value&&!wcscmp(value,L"1");
                s->init=1;Log("initializing",frame_id,"background HIP init; original frame unchanged");
                try{std::thread(&Initialize,s).detach();}catch(...){s->init=-1;s->disabled=true;throw;}
                return false;
            }
            if(s->init.load(std::memory_order_acquire)!=2)return false;
            if(s->inflight.fetch_add(1)>=MaxInflight){s->inflight.fetch_sub(1);return false;}
            Job* raw=nullptr;
            try{raw=new Job(s,frame_id,generation,format,unsigned(desc.Width),desc.Height,unsigned(desc.Format));}catch(...){s->inflight.fetch_sub(1);throw;}
            struct Caller {Job* p;~Caller(){p->Release();}} caller{raw};
            raw->resources.Bind(device.p,color);
            const size_t bpp=NativePixelBytes(format);
            const bool fresh_buffers=!s->in_buf.p;
            const char* buf_err=EnsureBuffers(*s,device.p,desc,bpp);
            if(buf_err){Log("setup_failed",frame_id,buf_err);return false;}
            if(fresh_buffers)
                Log("buffers",frame_id,s->shared_mode?"shared handles; zero-copy GPU path":"mapped READBACK/UPLOAD; H2D staging fallback");
            raw->BindBuffers(*s);
            if(FAILED(list->SetPrivateDataInterface(raw->anchor,raw))){Log("anchor_failed",frame_id);return false;}
            CopyPrefix(list,*s,state,color,desc,bpp);
            uint32_t accepted=0;
            Dlss5SubmitMarker marker{DLSS5_SUBMIT_MAGIC,DLSS5_SUBMIT_VERSION,sizeof(Dlss5SubmitMarker),raw,
                &Job::Retain,&Job::ReleaseContext,&Job::Run,&accepted};
            list->SetMarker(DLSS5_SUBMIT_METADATA,&marker,sizeof marker);
            if(accepted!=1){
                if(!s->disabled.exchange(true))Log("disabled",frame_id,"missing vkd3d HIP submit extension; no suffix writeback");
                return false;
            }
            CopySuffix(list,*s,state,color,desc,bpp);
            // Extension now owns the job through allocator GPU completion/reset.
            // Failure to remove the conservative anchor is safe, not premature release.
            list->SetPrivateDataInterface(raw->anchor,nullptr);
            Log("queued",frame_id,"prefix -> HIP callback -> suffix");return true;
        }catch(const std::exception& e){Log("record_failed",frame_id,e.what());return false;}
        catch(...){Log("record_failed",frame_id);return false;}
    }
};
