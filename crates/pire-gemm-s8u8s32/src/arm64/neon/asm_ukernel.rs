use seq_macro::seq;
use crate::{TA, TB, TC, TC_SIZE};
use pire_base::{
    prefetch_0, def_ukernel_neon_i8mm, mem,
    acc_2, acc_1,
    store_2, store_1,
};

type TS = f32;

const ONE_SCALAR: TS = 1f32;
const ZERO_SCALAR: TS = 0f32;

macro_rules! unzip_tuple {
    ($r1:tt, $r2:tt,$rt1:tt,$rt2:tt) => {
        concat!(
            "uzp1 v", $rt1, ".2d, v", $r1, ".2d, v", $r2, ".2d\n",
            "uzp2 v", $rt2, ".2d, v", $r1, ".2d, v", $r2, ".2d\n",
            // copy uzp1 to z8 and uzp2 to v11
            "orr v", $r1, ".16b, v", $rt1, ".16b, v", $rt1, ".16b\n",
            "orr v", $r2, ".16b, v", $rt2, ".16b, v", $rt2, ".16b\n",
        )
    };
}

macro_rules! v_i {
    ($m0:tt, $ni:tt) => {
        concat!("[", $m0, ", #", $ni, "*0x10]")
    }
}

macro_rules! unzip_c {
    () => {
        concat!(
            unzip_tuple!(8, 9, 1, 2),
            unzip_tuple!(10, 11, 3, 4),

            unzip_tuple!(12, 13, 5, 6),
            unzip_tuple!(14, 15, 7, 1),

            unzip_tuple!(16, 17, 2, 3),
            unzip_tuple!(18, 19, 4, 5),

            unzip_tuple!(20, 21, 6, 7),
            unzip_tuple!(22, 23, 1, 2),
            
            unzip_tuple!(24, 25, 3, 4),
            unzip_tuple!(26, 27, 5, 6),
            
            unzip_tuple!(28, 29, 7, 1),
            unzip_tuple!(30, 31, 2, 3),
        )
    }
}

macro_rules! beta_fmadd {
    (C, $m0:expr, $r1:expr,1) => {
        concat!(
            "ldr q1, ", $m0, "\n",
            "add v", $r1, ".4s, v", $r1, ".4s, v1.4s\n",
        ) 
    };
    (M, $m0:expr, $r1:expr,1) => {
        concat!(
            "ldr q1, ", $m0, "\n",
            "add v", $r1, ".4s, v", $r1, ".4s, v1.4s\n",
        ) 
    };

    (C, $m0:expr, $r1:expr,2) => {
        concat!(
            "ldr q1, ", $m0, "\n",
            "scvtf v1.4s, v1.4s\n",
            "scvtf v", $r1, ".4s, v", $r1, ".4s\n",
            "fmla v", $r1, ".4s, v1.4s, v0.s[0]\n",
            "fcvtzs v", $r1, ".4s, v", $r1, ".4s\n",
        ) 
    };
    (M, $m0:expr, $r1:expr,2) => {
        concat!(
            "ldr q1, ", $m0, "\n",
            "scvtf v1.4s, v1.4s\n",
            "scvtf v", $r1, ".4s, v", $r1, ".4s\n",
            "fmla v", $r1, ".4s, v1.4s, v0.s[0]\n",
            "fcvtzs v", $r1, ".4s, v", $r1, ".4s\n",
        ) 
    };
}

macro_rules! vzeroall {
    ($r0:tt, $r1:tt) => {
        seq!(r in $r0..=$r1 {
            concat!(#("dup v", r, ".4s, wzr \n",)*)
        })
    }
}

macro_rules! vfmadd {
    ($r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            "usmmla v", $r3, ".4s", ", v", $r2,".16b, v", $r1, ".16b\n",
        ) 
    };
}

macro_rules! loadp_unit {
    ($m0:expr, $r1:expr) => {
        concat!(
            "ldr q", $r1, ", ", $m0, "\n",
        )
    };
}

macro_rules! storep_unit {
    ($l:tt, $r1:expr, $m0:expr) => {
        concat!(
            "str q", $r1, ", ", $m0,  "\n",
        )
    };
}

macro_rules! alpha_scale_0 {
    ($r0:tt, $r1:tt) => {
        seq!(r in $r0..=$r1 {
            concat!(
                "ldr s1, [{alphax}]", "\n",
                #(
                    "scvtf v", r, ".4s, v", r, ".4s\n",
                    "fmul  v", r, ".4s, v", r, ".4s, v1.s[0]\n",
                    "fcvtzs v", r, ".4s, v", r, ".4s\n",
                )*
            )
        })
    }
}

macro_rules! load_beta {
    () => {
        concat!(
            "ldr s0, [{betax}]", "\n",
            "/* {betax} */", "\n",
        )
    }
}


macro_rules! loadp {
    (2, $m0:expr) => {
        concat!(
            loadp_unit!(mem!($m0), 0),
            loadp_unit!(mem!($m0, "0x10"), 1),
            loadp_unit!(mem!($m0, "0x20"), 2),
            loadp_unit!(mem!($m0, "0x30"), 3),
        )
    };

    (1, $m0:expr) => {
        concat!(
            loadp_unit!(mem!($m0), 0),
            loadp_unit!(mem!($m0, "0x10"), 1),
        )
    };
}

// only non contigous along m and n direction which is not changed frequently during iteration along k direction
/*

x1 -> cs_a
x2 -> cs_b
x4 -> cx + 3*cs_b

*/


macro_rules! init_ab {
    (B) => {
        concat!(
            "/* {x11} */", "\n",
            "/* {x10} */", "\n",
            "/* {x9} */", "\n",
            "/* {x8} */", "\n",
            "/* {x7} */", "\n",
            "/* {x6} */", "\n",
            "/* {x5} */", "\n",
            "/* {x4} */", "\n",
            "/* {x3} */", "\n",
            "/* {x2} */", "\n",
            "/* {x1} */", "\n",

            "ldr {x0}, [{dim_arrx}, #24]", "\n",
        )
    };
    (S) => {
        concat!(
            // mov cs_b to reg
            "mov ({dim_arrx}), {x1}", "\n",
            // "mov 8({dim_arrx}), {x2}", "\n",
            "ldr {x0}, [{dim_arrx}, #24]", "\n",
        )
    };
}


macro_rules! c_load {
    () => {
        concat!(
            unzip_c!(),
            "ldr {x0}, [{dim_arrx}, #16]\n",
            "add {x1}, {cx}, {x0} \n",
            "add {x2}, {x1}, {x0} \n",
            "add {x3}, {x2}, {x0} \n",
            "add {x4}, {x3}, {x0} \n",
            "add {x5}, {x4}, {x0} \n",
            "add {x6}, {x5}, {x0} \n",
            "add {x7}, {x6}, {x0} \n",
            "add {x8}, {x7}, {x0} \n",
            "add {x9}, {x8}, {x0} \n",
            "add {x10}, {x9}, {x0} \n",
            "add {x11}, {x10}, {x0} \n",
        )
    };
    (11) => {
        concat!(
            "ldr {x0}, [{dim_arrx}, #16]\n",
            "add {x1}, {cx}, {x0} \n",
            "add {x2}, {x1}, {x0} \n",
            "add {x3}, {x2}, {x0} \n",
            "add {x4}, {x3}, {x0} \n",
            "add {x5}, {x4}, {x0} \n",
            "add {x6}, {x5}, {x0} \n",
            "add {x7}, {x6}, {x0} \n",
            "add {x8}, {x7}, {x0} \n",
            "add {x9}, {x8}, {x0} \n",
            "add {x10}, {x9}, {x0} \n",
        )
    };
    (10) => {
        concat!(
            "ldr {x0}, [{dim_arrx}, #16]\n",
            "add {x1}, {cx}, {x0} \n",
            "add {x2}, {x1}, {x0} \n",
            "add {x3}, {x2}, {x0} \n",
            "add {x4}, {x3}, {x0} \n",
            "add {x5}, {x4}, {x0} \n",
            "add {x6}, {x5}, {x0} \n",
            "add {x7}, {x6}, {x0} \n",
            "add {x8}, {x7}, {x0} \n",
            "add {x9}, {x8}, {x0} \n",
        )
    };
    (9) => {
        concat!(
            "ldr {x0}, [{dim_arrx}, #16]\n",
            "add {x1}, {cx}, {x0} \n",
            "add {x2}, {x1}, {x0} \n",
            "add {x3}, {x2}, {x0} \n",
            "add {x4}, {x3}, {x0} \n",
            "add {x5}, {x4}, {x0} \n",
            "add {x6}, {x5}, {x0} \n",
            "add {x7}, {x6}, {x0} \n",
            "add {x8}, {x7}, {x0} \n",
        )
    };
    (8) => {
        concat!(
            "ldr {x0}, [{dim_arrx}, #16]\n",
            "add {x1}, {cx}, {x0} \n",
            "add {x2}, {x1}, {x0} \n",
            "add {x3}, {x2}, {x0} \n",
            "add {x4}, {x3}, {x0} \n",
            "add {x5}, {x4}, {x0} \n",
            "add {x6}, {x5}, {x0} \n",
            "add {x7}, {x6}, {x0} \n",
        )
    };
    (7) => {
        concat!(
            "ldr {x0}, [{dim_arrx}, #16]\n",
            "add {x1}, {cx}, {x0} \n",
            "add {x2}, {x1}, {x0} \n",
            "add {x3}, {x2}, {x0} \n",
            "add {x4}, {x3}, {x0} \n",
            "add {x5}, {x4}, {x0} \n",
            "add {x6}, {x5}, {x0} \n",
        )
    };
    (6) => {
        concat!(
            "ldr {x0}, [{dim_arrx}, #16]\n",
            "add {x1}, {cx}, {x0} \n",
            "add {x2}, {x1}, {x0} \n",
            "add {x3}, {x2}, {x0} \n",
            "add {x4}, {x3}, {x0} \n",
            "add {x5}, {x4}, {x0} \n",
        )
    };
    (5) => {
        concat!(
            "ldr {x0}, [{dim_arrx}, #16]\n",
            "add {x1}, {cx}, {x0} \n",
            "add {x2}, {x1}, {x0} \n",
            "add {x3}, {x2}, {x0} \n",
            "add {x4}, {x3}, {x0} \n",
        )
    };
    (4) => {
        concat!(
            "ldr {x0}, [{dim_arrx}, #16]\n",
            "add {x1}, {cx}, {x0} \n",
            "add {x2}, {x1}, {x0} \n",
            "add {x3}, {x2}, {x0} \n",
        )
    };
    (3) => {
        concat!(
            "ldr {x0}, [{dim_arrx}, #16]\n",
            "add {x1}, {cx}, {x0} \n",
            "add {x2}, {x1}, {x0} \n",
        )
    };
    (2) => {
        concat!(
            "ldr {x0}, [{dim_arrx}, #16]\n",
            "add {x1}, {cx}, {x0} \n",
        )
    };
    (1) => {
        concat!(
            "ldr {x0}, [{dim_arrx}, #16]\n",
            "add {x1}, {cx}, {x0} \n",
        )
    };
}


macro_rules! vzero_kernel {
    () => {vzeroall!(8,31)};
}

macro_rules! inc_b {
    (S,$nr:tt) => {
        "add {x1},{cx} \n"
    };
    (S,1) => {
        "add {x1},{cx} \n"
    };
    (B,$nr:tt) => {
        ""
    };
}


macro_rules! alpha_scale {
    () => {
        alpha_scale_0!(8,31)
    };
}

macro_rules! cr_2 {
    (0,0) => { 8 };
    (0,1) => { 9 };

    (1,0) => { 10 };
    (1,1) => { 11 };

    (0,2) => { 12 };
    (0,3) => { 13 };

    (1,2) => { 14 };
    (1,3) => { 15 };

    (0,4) => { 16 };
    (0,5) => { 17 };

    (1,4) => { 18 };
    (1,5) => { 19 };

    (0,6) => { 20 };
    (0,7) => { 21 };

    (1,6) => { 22 };
    (1,7) => { 23 };

    (0,8) => { 24 };
    (0,9) => { 25 };

    (1,8) => { 26 };
    (1,9) => { 27 };

    (0,10) => { 28 };
    (0,11) => { 29 };

    (1,10) => { 30 };
    (1,11) => { 31 };
}

macro_rules! cr_1 {
    (0,0) => { 8 };
    (0,1) => { 9 };

    (0,2) => { 12 };
    (0,3) => { 13 };

    (0,4) => { 16 };
    (0,5) => { 17 };

    (0,6) => { 20 };
    (0,7) => { 21 };

    (0,8) => { 24 };
    (0,9) => { 25 };

    (0,10) => { 28 };
    (0,11) => { 29 };
}

macro_rules! load_b {
    (B, 0) => {
        concat!(
            "ld1 {{v4.2d}}, [{bx}]", "\n",
            "add {bx}, {bx}, #8 \n",
        )
    };
    (B, 2) => {
        concat!(
            "ld1 {{v5.2d}}, [{bx}]", "\n",
            "add {bx}, {bx}, #8 \n",
        )
    };
    (B, 4) => {
        concat!(
            "ld1 {{v6.2d}}, [{bx}]", "\n",
            "add {bx}, {bx}, #8 \n",
        )
    };
    (B, 6) => {
        concat!(
            "ld1 {{v7.2d}}, [{bx}]", "\n",
            "add {bx}, {bx}, #8 \n",
        )
    };
    (B, 8) => {
        concat!(
            "ld1 {{v4.2d}}, [{bx}]", "\n",
            "add {bx}, {bx}, #8 \n",
        )
    };
    (B, 10) => {
        concat!(
            "ld1 {{v5.2d}}, [{bx}]", "\n",
            "add {bx}, {bx}, #8 \n",
        )
    };

    (B, $nr:tt) => {
        "add {bx}, {bx}, #8 \n"
    };
}


macro_rules! load_a {
    ($mr:tt) => {
        loadp!($mr, "{ax}")
    };
}

macro_rules! fmadd_3v2 {
    (0) => {
        concat!(
            vfmadd!(0, 4, 8),
            vfmadd!(1, 4, 9),
            vfmadd!(2, 4, 10),
            vfmadd!(3, 4, 11),
        )
    };
    (2) => {
        concat!(
            vfmadd!(0, 5, 12),
            vfmadd!(1, 5, 13),
            vfmadd!(2, 5, 14),
            vfmadd!(3, 5, 15),
        )
    };
    (4) => {
        concat!(
            vfmadd!(0, 6, 16),
            vfmadd!(1, 6, 17),
            vfmadd!(2, 6, 18),
            vfmadd!(3, 6, 19),
        )
    };
    (6) => {
        concat!(
            vfmadd!(0, 7, 20),
            vfmadd!(1, 7, 21),
            vfmadd!(2, 7, 22),
            vfmadd!(3, 7, 23),
        )
    };
    (8) => {
        concat!(
            vfmadd!(0, 4, 24),
            vfmadd!(1, 4, 25),
            vfmadd!(2, 4, 26),
            vfmadd!(3, 4, 27),
        )
    };
    (10) => {
        concat!(
            vfmadd!(0, 5, 28),
            vfmadd!(1, 5, 29),
            vfmadd!(2, 5, 30),
            vfmadd!(3, 5, 31),
        )
    };
    ($nr:tt) => {""};
}

macro_rules! step_2 {
    ($b_layout:tt, $nr:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(2),
                "add {ax}, {ax}, #64 \n",
                #(
                    load_b!($b_layout, n),
                    fmadd_3v2!(n),
                )*
                // "add {bx}, {bx}, #256 \n",
                inc_b!($b_layout,$nr), 
            )
        })
    };
}


macro_rules! fmadd_1v2 {
    (0) => {
        concat!(
            vfmadd!(0, 4, 8),
            vfmadd!(1, 4, 9),
        )
    };
    (2) => {
        concat!(
            vfmadd!(0, 5, 12),
            vfmadd!(1, 5, 13),
        )
    };
    (4) => {
        concat!(
            vfmadd!(0, 6, 16),
            vfmadd!(1, 6, 17),
        )
    };
    (6) => {
        concat!(
            vfmadd!(0, 7, 20),
            vfmadd!(1, 7, 21),
        )
    };
    (8) => {
        concat!(
            vfmadd!(0, 4, 24),
            vfmadd!(1, 4, 25),
        )
    };
    (10) => {
        concat!(
            vfmadd!(0, 5, 28),
            vfmadd!(1, 5, 29),
        )
    };
    ($nr:tt) => {""};
}

macro_rules! step_1 {
    ($b_layout:tt, $nr:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(1),
                "add {ax}, {ax}, #32 \n",
                #(
                    load_b!($b_layout, n),
                    fmadd_1v2!(n),
                )*
                // "add {bx}, {bx}, #256 \n",
                inc_b!($b_layout,$nr), 
            )
        })
    };
}

macro_rules! prefetch_c {
    () => {
        concat!(
            "ldr {x0}, [{dim_arrx}, #16]\n",
            "add {x1}, {cx}, {x0}\n ",
            "add {x2}, {x1}, {x0} \n",
            "add {x3}, {x2}, {x0} \n",
            "add {x4}, {x3}, {x0} \n",
            "add {x5}, {x4}, {x0} \n",
            "add {x6}, {x5}, {x0} \n",
            "add {x7}, {x6}, {x0} \n",
            "add {x8}, {x7}, {x0} \n",
            "add {x9}, {x8}, {x0} \n",
            "add {x10}, {x9}, {x0} \n",
            "add {x11}, {x10}, {x0} \n",
            "prfm pldl1keep, [{cx}] \n",
            "prfm pldl1keep, [{cx},#56]\n",
            "prfm pldl1keep, [{x1}] \n",
            "prfm pldl1keep, [{x1},#56]\n",
            "prfm pldl1keep, [{x2}] \n",
            "prfm pldl1keep, [{x2},#56]\n",
            "prfm pldl1keep, [{x3}] \n",
            "prfm pldl1keep, [{x3},#56]\n",
            "prfm pldl1keep, [{x4}] \n",
            "prfm pldl1keep, [{x4},#56]\n",
            "prfm pldl1keep, [{x5}] \n",
            "prfm pldl1keep, [{x5},#56]\n",
            "prfm pldl1keep, [{x6}] \n",
            "prfm pldl1keep, [{x6},#56]\n",
            "prfm pldl1keep, [{x7}] \n",
            "prfm pldl1keep, [{x7},#56]\n",
            "prfm pldl1keep, [{x8}] \n",
            "prfm pldl1keep, [{x8},#56]\n",
            "prfm pldl1keep, [{x9}] \n",
            "prfm pldl1keep, [{x9},#56]\n",
            "prfm pldl1keep, [{x10}] \n",
            "prfm pldl1keep, [{x10},#56]\n",
            "prfm pldl1keep, [{x11}] \n",
            "prfm pldl1keep, [{x11},#56]\n",
        )
    };
}


def_ukernel_neon_i8mm!(step_1, acc_1, store_1, 1, 12, B, M, ukernel_1_bbp);
def_ukernel_neon_i8mm!(step_2, acc_2, store_2, 2, 12, B, M, ukernel_2_bbp);


def_ukernel_neon_i8mm!(step_2, acc_2, store_2, 2, 12, B, C, ukernel_bbc);
