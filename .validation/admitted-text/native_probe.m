#import <Foundation/Foundation.h>
#import <CoreText/CoreText.h>
#include <malloc/malloc.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <sys/mman.h>
#include <dlfcn.h>
#include <stdint.h>
#include <string.h>

struct probe_zone { uint64_t blocks, live, high, allocated; };
struct probe_event { uint64_t family, address, size, released; };
struct probe_identity { char postscript[256], version[256], path[1024], coretext[1024]; };

void *probe_pool_push(void) { return [[NSAutoreleasePool alloc] init]; }
void probe_pool_pop(void *pool) { [(NSAutoreleasePool *)pool drain]; }
void probe_zone_snapshot(struct probe_zone *output) {
    malloc_statistics_t statistics = {0};
    malloc_zone_statistics(NULL, &statistics);
    *output = (struct probe_zone){statistics.blocks_in_use, statistics.size_in_use,
        statistics.max_size_in_use, statistics.size_allocated};
}

__attribute__((noinline)) void *probe_phase_marker(size_t ordinal) {
    size_t size = 1009 + ordinal * 16;
    void *pointer = malloc(size);
    if (pointer) memset(pointer, 0x5a, size);
    return pointer;
}
void probe_release_marker(void *pointer) { free(pointer); }

__attribute__((noinline)) int probe_allocation_sentinels(struct probe_event *events, size_t capacity) {
    if (capacity < 7) return -1;
    void *pointer = malloc(1237);
    if (!pointer) return -2;
    memset(pointer, 0x11, 1237);
    events[0] = (struct probe_event){1, (uintptr_t)pointer, 1237, 1};
    free(pointer);
    pointer = calloc(1, 1877);
    if (!pointer) return -3;
    events[1] = (struct probe_event){2, (uintptr_t)pointer, 1877, 1};
    void *grown = realloc(pointer, 3251);
    if (!grown) { free(pointer); return -4; }
    ((volatile unsigned char *)grown)[3250] = 3;
    events[2] = (struct probe_event){3, (uintptr_t)grown, 3251, 1};
    free(grown);
    pointer = malloc_zone_malloc(malloc_default_zone(), 2371);
    if (!pointer) return -5;
    memset(pointer, 0x22, 2371);
    events[3] = (struct probe_event){4, (uintptr_t)pointer, 2371, 1};
    malloc_zone_free(malloc_default_zone(), pointer);
    size_t pages = (size_t)vm_page_size * 2;
    pointer = mmap(NULL, pages, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (pointer == MAP_FAILED) return -6;
    memset(pointer, 0x33, pages);
    events[4] = (struct probe_event){5, (uintptr_t)pointer, pages, 1};
    if (munmap(pointer, pages) != 0) return -7;
    mach_vm_address_t address = 0;
    if (mach_vm_allocate(mach_task_self(), &address, pages, VM_FLAGS_ANYWHERE) != KERN_SUCCESS) return -8;
    memset((void *)address, 0x44, pages);
    events[5] = (struct probe_event){6, address, pages, 1};
    if (mach_vm_deallocate(mach_task_self(), address, pages) != KERN_SUCCESS) return -9;
    char text[514]; memset(text, 'q', sizeof(text) - 1); text[513] = 0;
    CFStringRef string = CFStringCreateWithCString(NULL, text, kCFStringEncodingUTF8);
    if (!string) return -10;
    events[6] = (struct probe_event){7, (uintptr_t)string, 513, 1};
    CFRelease(string);
    return 7;
}

static void copy_string(CFStringRef string, char *output, size_t capacity) {
    if (!string || !CFStringGetCString(string, output, capacity, kCFStringEncodingUTF8)) output[0] = 0;
}
int probe_font_identity(const char *postscript, struct probe_identity *output) {
    memset(output, 0, sizeof(*output));
    CFStringRef name = CFStringCreateWithCString(NULL, postscript, kCFStringEncodingUTF8);
    if (!name) return -1;
    CTFontRef font = CTFontCreateWithName(name, 11.5, NULL);
    CFRelease(name);
    if (!font) return -2;
    CFStringRef actual = CTFontCopyPostScriptName(font);
    CFStringRef version = CTFontCopyName(font, kCTFontVersionNameKey);
    copy_string(actual, output->postscript, sizeof(output->postscript));
    copy_string(version, output->version, sizeof(output->version));
    if (actual) CFRelease(actual);
    if (version) CFRelease(version);
    CFTypeRef url = CTFontCopyAttribute(font, kCTFontURLAttribute);
    if (url && CFGetTypeID(url) == CFURLGetTypeID())
        CFURLGetFileSystemRepresentation((CFURLRef)url, true, (UInt8 *)output->path, sizeof(output->path));
    if (url) CFRelease(url);
    Dl_info info = {0};
    if (dladdr((void *)CTLineCreateWithAttributedString, &info) && info.dli_fname)
        strlcpy(output->coretext, info.dli_fname, sizeof(output->coretext));
    CFRelease(font);
    return 0;
}
