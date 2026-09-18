// GPU-resident frame interop for the in-game path:
//  - device kernels (kernels/gpu_convert.hip) replicating the hook's CPU pixel pipeline;
//  - Win32 D3D12 shared handle -> unix fd -> hipImportExternalMemory import, cached for
//    the process lifetime (same technique as the portable LD_PRELOAD bridge);
//  - host launch helpers used by dlss5_run_frame_raw_gpu in capi.hip.
// FMA contraction must stay OFF here (bit-exact pixel math, see gpu_convert.hip).
#pragma STDC FP_CONTRACT OFF

#include "dlss5_gpu_interop.hpp"
#include "dlss5_runtime.hpp"
#include "dlss5_launch.hpp"
#include "../kernels/gpu_convert.hip"

#include <dlfcn.h>
#include <elf.h>
#include <link.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <mutex>
#include <vector>

namespace dlss5 {
namespace gpu {

Fmt format_from_dxgi(unsigned format) {
  switch (format) {
    case 28:
      return FmtRgba8;
    case 87:
      return FmtBgra8;
    case 11:
      return FmtR16Unorm;
    case 10:
      return FmtRgba16F;
    default:
      return FmtUnsupported;
  }
}

namespace {

// ---------------- Win32 handle -> dma-buf fd (ported from the portable bridge) ----------------

typedef int (*wine_h2fd_t)(void*, unsigned, int*, unsigned*);

wine_h2fd_t resolve_h2fd() {
  static wine_h2fd_t fn = nullptr;
  static bool tried = false;
  if (tried) return fn;
  tried = true;
  fn = (wine_h2fd_t)dlsym(RTLD_DEFAULT, "wine_server_handle_to_fd");
  if (fn) return fn;
  void* h = dlopen("ntdll.so", RTLD_NOW | RTLD_NOLOAD | RTLD_GLOBAL);
  if (!h) h = dlopen("ntdll.so", RTLD_NOW | RTLD_GLOBAL);
  if (h) fn = (wine_h2fd_t)dlsym(h, "wine_server_handle_to_fd");
  return fn;
}

// Local (non-dynsym) Wine unix helpers live in win32u.so; scan its .symtab directly.
void* elf_local_sym(void* handle, const char* name) {
  struct link_map* lm = nullptr;
  struct stat st;
  const ElfW(Ehdr)* eh;
  const ElfW(Shdr)* sh;
  const ElfW(Shdr)* symtab = nullptr;
  const ElfW(Shdr)* strtab = nullptr;
  const ElfW(Sym)* sym;
  const char* strs;
  void* map;
  void* found = nullptr;
  int fd, i, n;

  if (!handle || !name) return nullptr;
  if (dlinfo(handle, RTLD_DI_LINKMAP, &lm) != 0 || !lm || !lm->l_name || !lm->l_name[0]) return nullptr;
  fd = open(lm->l_name, O_RDONLY);
  if (fd < 0) return nullptr;
  if (fstat(fd, &st) != 0 || st.st_size < (off_t)sizeof(ElfW(Ehdr))) {
    close(fd);
    return nullptr;
  }
  map = mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (map == MAP_FAILED) return nullptr;
  eh = (const ElfW(Ehdr)*)map;
  if (memcmp(eh->e_ident, ELFMAG, SELFMAG) != 0) goto out;
  sh = (const ElfW(Shdr)*)((const char*)map + eh->e_shoff);
  for (i = 0; i < eh->e_shnum; i++) {
    if (sh[i].sh_type == SHT_SYMTAB) {
      symtab = &sh[i];
      if (sh[i].sh_link < eh->e_shnum) strtab = &sh[sh[i].sh_link];
      break;
    }
  }
  if (!symtab || !strtab || !symtab->sh_entsize) goto out;
  strs = (const char*)map + strtab->sh_offset;
  sym = (const ElfW(Sym)*)map + symtab->sh_offset;
  n = (int)(symtab->sh_size / symtab->sh_entsize);
  for (i = 0; i < n; i++) {
    if (!sym[i].st_name || ELF64_ST_TYPE(sym[i].st_info) != STT_FUNC) continue;
    if (strcmp(strs + sym[i].st_name, name) == 0) {
      found = (char*)lm->l_addr + sym[i].st_value;
      break;
    }
  }
out:
  munmap(map, (size_t)st.st_size);
  return found;
}

typedef unsigned int d3dkmt_handle_t;
typedef d3dkmt_handle_t (*d3dkmt_open_resource_t)(d3dkmt_handle_t, void*, d3dkmt_handle_t*, d3dkmt_handle_t*);
typedef int (*d3dkmt_object_get_fd_t)(d3dkmt_handle_t);
typedef int (*d3dkmt_destroy_resource_t)(d3dkmt_handle_t);

int d3dkmt_handle_to_fd(void* nt_handle) {
  static int resolved = 0;
  static d3dkmt_open_resource_t open_res = nullptr;
  static d3dkmt_object_get_fd_t get_fd = nullptr;
  static d3dkmt_destroy_resource_t destroy_res = nullptr;
  d3dkmt_handle_t local = 0, mutex = 0, sync = 0;
  int fd = -1;
  void* h;

  if (!resolved) {
    resolved = 1;
    h = dlopen("win32u.so", RTLD_NOW | RTLD_NOLOAD);
    if (!h) h = dlopen("win32u.so", RTLD_NOW);
    if (h) {
      open_res = (d3dkmt_open_resource_t)elf_local_sym(h, "d3dkmt_open_resource");
      get_fd = (d3dkmt_object_get_fd_t)elf_local_sym(h, "d3dkmt_object_get_fd");
      destroy_res = (d3dkmt_destroy_resource_t)elf_local_sym(h, "d3dkmt_destroy_resource");
    }
  }
  if (!open_res || !get_fd) return -1;
  local = open_res(0, nt_handle, &mutex, &sync);
  if (!local) return -1;
  fd = get_fd(local);
  if (destroy_res) destroy_res(local);
  return fd;
}

int handle_to_fd(void* handle) {
  int unix_fd = -1;
  unsigned options = 0;
  wine_h2fd_t h2fd = resolve_h2fd();
  if (h2fd) {
    int st = h2fd(handle, 0x80000000u | 0x40000000u, &unix_fd, &options);
    if (st != 0 || unix_fd < 0) st = h2fd(handle, 0x10000000u | 0x80000000u | 0x40000000u, &unix_fd, &options);
    if (st != 0 || unix_fd < 0) st = h2fd(handle, 0x80000000u, &unix_fd, &options);
    if (st != 0 || unix_fd < 0) unix_fd = -1;
  }
  if (unix_fd < 0) unix_fd = d3dkmt_handle_to_fd(handle);
  return unix_fd;
}

// ---------------- import cache (process lifetime, mirrors the portable bridge) ----------------

struct ImportEntry {
  void* handle;
  size_t size;
  int fd;
  hipExternalMemory_t ext;
  void* dev;
};
std::vector<ImportEntry> g_imports;
std::mutex g_import_mu;

}  // namespace

bool map_device(void* handle, size_t size, void** dev) {
  if (!handle || !size || !dev) return false;
  std::lock_guard<std::mutex> lk(g_import_mu);
  for (const auto& e : g_imports)
    if (e.handle == handle && e.size == size) {
      *dev = e.dev;
      return true;
    }
  int fd = handle_to_fd(handle);
  if (fd < 0) return false;
  hipExternalMemoryHandleDesc d{};
  d.type = hipExternalMemoryHandleTypeOpaqueFd;
  d.handle.fd = dup(fd);
  close(fd);
  if (d.handle.fd < 0) return false;
  d.size = size;
  hipExternalMemory_t ext = nullptr;
  if (hipImportExternalMemory(&ext, &d) != hipSuccess) {
    close(d.handle.fd);
    return false;
  }
  hipExternalMemoryBufferDesc bd{};
  bd.offset = 0;
  bd.size = size;
  void* ptr = nullptr;
  if (hipExternalMemoryGetMappedBuffer(&ptr, ext, &bd) != hipSuccess || !ptr) {
    hipDestroyExternalMemory(ext);
    close(d.handle.fd);
    return false;
  }
  g_imports.push_back({handle, size, d.handle.fd, ext, ptr});
  *dev = ptr;
  return true;
}

// ---------------- launch helpers ----------------

namespace {
constexpr int kW = 1920, kH = 1080;
constexpr int kBlock = 32;
inline dim3 grid_for(int w, int h) {
  return dim3((w + kBlock - 1) / kBlock, (h + kBlock - 1) / kBlock, 1);
}
// The raw-frame pipeline runs on the default stream (like the frame buffers in capi).
hipStream_t raw_stream() { return nullptr; }
}  // namespace

void launch_px_to_rgba(const uint8_t* dev_in, float* out, int w, int h, unsigned dxgi_format) {
  const dim3 blk(kBlock, kBlock, 1);
  switch (format_from_dxgi(dxgi_format)) {
    case FmtRgba8:
      k_gc_px_to_rgba<FmtRgba8><<<grid_for(kW, kH), blk, 0, raw_stream()>>>(dev_in, w, h, out);
      break;
    case FmtBgra8:
      k_gc_px_to_rgba<FmtBgra8><<<grid_for(kW, kH), blk, 0, raw_stream()>>>(dev_in, w, h, out);
      break;
    case FmtR16Unorm:
      k_gc_px_to_rgba<FmtR16Unorm><<<grid_for(kW, kH), blk, 0, raw_stream()>>>(dev_in, w, h, out);
      break;
    case FmtRgba16F:
      k_gc_px_to_rgba<FmtRgba16F><<<grid_for(kW, kH), blk, 0, raw_stream()>>>(dev_in, w, h, out);
      break;
    default:
      throw std::runtime_error("unsupported dxgi format for gpu conversion");
  }
  hip_check(hipGetLastError(), "launch px_to_rgba");
}

void launch_rgb_to_pix(const float* rgb, const uint8_t* alpha_src, uint8_t* out, int w, int h, unsigned dxgi_format) {
  const dim3 blk(kBlock, kBlock, 1);
  switch (format_from_dxgi(dxgi_format)) {
    case FmtRgba8:
      k_gc_rgb_to_pix<FmtRgba8><<<grid_for(w, h), blk, 0, raw_stream()>>>(rgb, w, h, alpha_src, out);
      break;
    case FmtBgra8:
      k_gc_rgb_to_pix<FmtBgra8><<<grid_for(w, h), blk, 0, raw_stream()>>>(rgb, w, h, alpha_src, out);
      break;
    case FmtR16Unorm:
      k_gc_rgb_to_pix<FmtR16Unorm><<<grid_for(w, h), blk, 0, raw_stream()>>>(rgb, w, h, alpha_src, out);
      break;
    case FmtRgba16F:
      k_gc_rgb_to_pix<FmtRgba16F><<<grid_for(w, h), blk, 0, raw_stream()>>>(rgb, w, h, alpha_src, out);
      break;
    default:
      throw std::runtime_error("unsupported dxgi format for gpu conversion");
  }
  hip_check(hipGetLastError(), "launch rgb_to_pix");
}

// Guards the 1920x1080 RGB result: non-finite (all formats) or non-representable as
// half (rgba16_float), mirroring the CPU pre-passes that bypass the whole frame.
bool guard_result_f32(const float* dev, size_t n, bool need_half_repr) {
  DeviceBuf flag(4);
  hip_check(hipMemset(flag.p, 0, 4), "guard alloc");
  if (need_half_repr) {
    k_gc_prescan_rgb16f<<<(int)((n + kBlock - 1) / kBlock), kBlock, 0, raw_stream()>>>(dev, int(n), flag.as<unsigned>());
  } else {
    k_gc_prescan_f32<<<(int)((n + kBlock - 1) / kBlock), kBlock, 0, raw_stream()>>>(dev, int(n), flag.as<unsigned>());
  }
  hip_check(hipGetLastError(), "guard launch");
  unsigned host = 0;
  hip_check(hipMemcpy(&host, flag.p, 4, hipMemcpyDeviceToHost), "guard read");
  return host != 0;
}

bool input16_bad(const uint8_t* dev, size_t n16) {
  DeviceBuf flag(4);
  hip_check(hipMemset(flag.p, 0, 4), "guard alloc");
  k_gc_prescan16<<<(int)((n16 + kBlock - 1) / kBlock), kBlock, 0, raw_stream()>>>((const uint16_t*)dev, int(n16),
                                                                            flag.as<unsigned>());
  hip_check(hipGetLastError(), "guard launch");
  unsigned host = 0;
  hip_check(hipMemcpy(&host, flag.p, 4, hipMemcpyDeviceToHost), "guard read");
  return host != 0;
}

}  // namespace gpu
}  // namespace dlss5
