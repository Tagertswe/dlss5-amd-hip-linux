#pragma once
// Record ONLY on the original FFX native list, immediately after successful
// ffxDispatch. No submission, list lifecycle changes, or graphics waits here.
#include <d3d12.h>
#include <dxgi1_4.h>
#include <atomic>
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
        // pair is race-free. Handles are created once and kept open for the state
        // lifetime; buf_gen disambiguates the HIP import cache across recreation.
        Ref<ID3D12Resource> in_buf,out_buf;
        HANDLE in_handle{},out_handle{};
        unsigned buf_w{},buf_h{},buf_dxgi{};
        size_t buf_bytes{};
        uint32_t buf_gen{0};
        ~State(){
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
        // V3 GPU-resident callback: no CPU readback. The prefix already copied
        // color->in_buf and in_buf->out_buf (out_buf is the original-frame
        // fallback), so on ANY rejection/failure the suffix writes back the
        // original pixels exactly like the old CPU fallback path.
        int Process(uint64_t queue){
            if(called.exchange(true)){Log("duplicate_callback",frame,"resubmitted command list rejected");return -1;}
            uint64_t expected=0;
            if(!queue||(!owner->queue.compare_exchange_strong(expected,queue)&&expected!=queue)){
                Log("queue_conflict",frame,"callback rejected before GPU frame");return -1;
            }
            std::unique_lock<std::mutex> serial(owner->processing,std::try_to_lock);
            if(!serial.owns_lock())return -1;
            if(FAILED(resources.device->GetDeviceRemovedReason()))return -1;
            if(!owner->client||!owner->client->HasRawGpu()){
                Log("processed",frame,"V3 raw bridge missing; original input");return 0;
            }
            auto bypass=[&]{return (generation&1)||owner->disabled.load()||owner->generation.load()!=generation;};
            if(bypass()){Log("processed",frame,"F6 bypass; original input");return 0;}
            try {
                if(owner->client->RunFrameRaw(owner->in_handle,owner->out_handle,width,height,dxgi,
                                             unsigned(frame),owner->display_srgb,owner->buf_gen)){
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
    static D3D12_RESOURCE_BARRIER Bar(ID3D12Resource* r,int from,int to){
        D3D12_RESOURCE_BARRIER b{};b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition={r,0,static_cast<D3D12_RESOURCE_STATES>(from),static_cast<D3D12_RESOURCE_STATES>(to)};
        return b;
    }
    // Implicit-barrier state (Windows 11 SDK); missing from mingw-w64 headers
    // and outside the enum's constant range, so it travels as a plain int.
    static constexpr int StateUnknown=0x80000000;
    // V3: one DEFAULT-heap in/out buffer pair per (device, geometry, format), with
    // Win32 shared handles the .so imports into HIP. Recreated on any change.
    static bool EnsureBuffers(State& s,ID3D12Device* d,const D3D12_RESOURCE_DESC& desc,size_t bpp){
        const unsigned w=unsigned(desc.Width),h=desc.Height,dxgi=unsigned(desc.Format);
        if(s.in_buf.p&&s.buf_w==w&&s.buf_h==h&&s.buf_dxgi==dxgi)return true;
        // BufLoc assumes a packed footprint (RowPitch == width*bpp). Reject any
        // row-padded layout before we allocate. Same check as the old Create().
        {
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};UINT rows=0;UINT64 rowbytes=0,total=0;
            d->GetCopyableFootprints(&desc,0,1,0,&fp,&rows,&rowbytes,&total);
            const UINT64 packed=UINT64(w)*bpp;
            if(rows!=h||rowbytes!=packed||fp.Footprint.Width!=desc.Width||
               fp.Footprint.Height!=h||fp.Footprint.Depth!=1||fp.Footprint.Format!=desc.Format||
               fp.Footprint.RowPitch<packed||!total||
               UINT64(h-1)*fp.Footprint.RowPitch+packed>total-fp.Offset)return false;
        }
        if(s.in_handle){CloseHandle(s.in_handle);s.in_handle=nullptr;}
        if(s.out_handle){CloseHandle(s.out_handle);s.out_handle=nullptr;}
        s.in_buf.Reset();s.out_buf.Reset();
        D3D12_RESOURCE_DESC buffer{};buffer.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;
        buffer.Width=UINT64(size_t(w)*h*bpp);buffer.Height=1;buffer.DepthOrArraySize=buffer.MipLevels=1;
        buffer.SampleDesc.Count=1;buffer.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        D3D12_HEAP_PROPERTIES heap{};heap.CreationNodeMask=heap.VisibleNodeMask=1;
        heap.Type=D3D12_HEAP_TYPE_DEFAULT;
        if(FAILED(d->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&buffer,
            D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(s.in_buf.Out())))||
           FAILED(d->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&buffer,
            D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(s.out_buf.Out()))))return false;
        // R|W: HIP maps in_buf read-only and out_buf read-write.
        const DWORD access=GENERIC_READ|GENERIC_WRITE;
        if(FAILED(d->CreateSharedHandle(s.in_buf.p,nullptr,access,nullptr,&s.in_handle))||
           FAILED(d->CreateSharedHandle(s.out_buf.p,nullptr,access,nullptr,&s.out_handle))){
            if(s.in_handle){CloseHandle(s.in_handle);s.in_handle=nullptr;}
            if(s.out_handle){CloseHandle(s.out_handle);s.out_handle=nullptr;}
            s.in_buf.Reset();s.out_buf.Reset();return false;
        }
        s.buf_w=w;s.buf_h=h;s.buf_dxgi=dxgi;s.buf_bytes=size_t(w)*h*bpp;s.buf_gen++;
        return true;
    }
    // Prefix: color -> in_buf, then in_buf -> out_buf. The second copy keeps
    // out_buf an exact original-frame fallback until the .so overwrites its RGB.
    // The color is transitioned from its KNOWN `state` and restored, matching
    // the old Copy() semantics. Implicit (UNKNOWN) barriers on our own buffers:
    // we don't track them between frames, and UNKNOWN is always safe.
    static void CopyPrefix(ID3D12GraphicsCommandList* list,State& s,D3D12_RESOURCE_STATES state,
                           ID3D12Resource* color,const D3D12_RESOURCE_DESC& desc,size_t bpp){
        D3D12_RESOURCE_BARRIER b=Bar(color,state,D3D12_RESOURCE_STATE_COPY_SOURCE);
        if(state!=D3D12_RESOURCE_STATE_COPY_SOURCE)list->ResourceBarrier(1,&b);
        D3D12_TEXTURE_COPY_LOCATION texture{},inb=BufLoc(s.in_buf.p,desc,bpp);
        texture.pResource=color;texture.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        list->CopyTextureRegion(&inb,0,0,0,&texture,nullptr);
        if(state!=D3D12_RESOURCE_STATE_COPY_SOURCE){
            b=Bar(color,D3D12_RESOURCE_STATE_COPY_SOURCE,state);list->ResourceBarrier(1,&b);
        }
        D3D12_RESOURCE_BARRIER bs[2]={
            Bar(s.in_buf.p,StateUnknown,D3D12_RESOURCE_STATE_COPY_SOURCE),
            Bar(s.out_buf.p,StateUnknown,D3D12_RESOURCE_STATE_COPY_DEST)};
        list->ResourceBarrier(2,bs);
        list->CopyResource(s.out_buf.p,s.in_buf.p);
        bs[0]=Bar(s.in_buf.p,D3D12_RESOURCE_STATE_COPY_SOURCE,StateUnknown);
        bs[1]=Bar(s.out_buf.p,D3D12_RESOURCE_STATE_COPY_DEST,StateUnknown);
        list->ResourceBarrier(2,bs);
    }
    // Suffix: out_buf -> color. Restores the color to its known `state`.
    static void CopySuffix(ID3D12GraphicsCommandList* list,State& s,D3D12_RESOURCE_STATES state,
                           ID3D12Resource* color,const D3D12_RESOURCE_DESC& desc,size_t bpp){
        D3D12_RESOURCE_BARRIER bs[2]={
            Bar(s.out_buf.p,StateUnknown,D3D12_RESOURCE_STATE_COPY_SOURCE),
            Bar(color,state,D3D12_RESOURCE_STATE_COPY_DEST)};
        list->ResourceBarrier(2,bs);
        D3D12_TEXTURE_COPY_LOCATION texture{},outb=BufLoc(s.out_buf.p,desc,bpp);
        texture.pResource=color;texture.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        list->CopyTextureRegion(&texture,0,0,0,&outb,nullptr);
        bs[0]=Bar(s.out_buf.p,D3D12_RESOURCE_STATE_COPY_SOURCE,StateUnknown);
        bs[1]=Bar(color,D3D12_RESOURCE_STATE_COPY_DEST,state);
        list->ResourceBarrier(2,bs);
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
            if(!EnsureBuffers(*s,device.p,desc,bpp)){Log("setup_failed",frame_id,"buffer create/shared handle failed; original unchanged");return false;}
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
