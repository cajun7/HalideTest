#include <gtest/gtest.h>
#include <cstdio>
#include <cstdint>
#include <sys/auxv.h>

// HWCAP bit definitions for ARM64 (from Linux kernel asm/hwcap.h)
#ifndef HWCAP_ASIMDHP
#define HWCAP_ASIMDHP (1UL << 10)
#endif
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1UL << 20)
#endif
#ifndef HWCAP_SVE
#define HWCAP_SVE (1UL << 22)
#endif

TEST(TargetDispatch, DetectArmFeatures) {
    unsigned long hwcap = getauxval(AT_HWCAP);

    bool has_fp16 = (hwcap & HWCAP_ASIMDHP) != 0;
    bool has_dot_prod = (hwcap & HWCAP_ASIMDDP) != 0;
    bool has_sve = (hwcap & HWCAP_SVE) != 0;

    printf("=== ARM Feature Detection ===\n");
    printf("AT_HWCAP     = 0x%lx\n", hwcap);
    printf("ASIMDHP (fp16)     : %s\n", has_fp16 ? "YES" : "no");
    printf("ASIMDDP (dot_prod) : %s\n", has_dot_prod ? "YES" : "no");
    printf("SVE                : %s\n", has_sve ? "YES" : "no");

    if (has_dot_prod && has_fp16) {
        printf("=> Halide will select HIGH-feature variant (armv82a+dot_prod+fp16)\n");
    } else {
        printf("=> Halide will select BASELINE variant (arm-64-android)\n");
    }

    // This test always passes -- its purpose is diagnostic output
    SUCCEED();
}
