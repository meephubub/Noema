/*
 * shim.c — Minimal C file that references the Needle 2 C API symbols.
 *
 * Compiling this and linking with libneedle.a forces the static linker to
 * pull in the object files from the archive, making the symbols available
 * to Rust FFI declarations.
 */

#include "needle.h"

/*
 * A single non-inlined function that touches every exported symbol.
 * The compiler cannot optimise these away because the function is
 * exported (and called from the Rust build via cc).
 */
void __needle_shim_keep_alive(void) {
    volatile int x = 0;
    x += needle_init(0, 0, 0);
    x += needle_complete(0, 0, 0, 0);
    needle_reset();
    x += needle_load(0, 0);
    (void)x;
}
