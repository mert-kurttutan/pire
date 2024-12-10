use seq_macro::seq;
use crate::{TA, TB, TC, TC_SIZE};
use pire_base::{
    prefetch_0, def_ukernel_neon, mem,
    acc_3, acc_2, acc_1,
    store_3, store_2, store_1,
};

const ONE_SCALAR: TC = 1f64;
const ZERO_SCALAR: TC = 0f64;

macro_rules! v_i {
    ($m0:tt, $ni:tt) => {
        concat!("[", $m0, ", #", $ni, "*0x10]")
    }
}

macro_rules! br_3 {
    (0) => { "v3.d[0]" };
    (1) => { "v3.d[1]" };
    (2) => { "v4.d[0]" };
    (3) => { "v4.d[1]" };
    (4) => { "v5.d[0]" };
    (5) => { "v5.d[1]" };
    (6) => { "v6.d[0]" };
    (7) => { "v6.d[1]" };
}

macro_rules! br_2 {
    (0) => { "v3.d[0]" };
    (1) => { "v3.d[1]" };
    (2) => { "v4.d[0]" };
    (3) => { "v4.d[1]" };
    (4) => { "v5.d[0]" };
    (5) => { "v5.d[1]" };
    (6) => { "v6.d[0]" };
    (7) => { "v6.d[1]" };
}
macro_rules! br_1 {
    (0) => { "v3.d[0]" };
    (1) => { "v3.d[1]" };
    (2) => { "v4.d[0]" };
    (3) => { "v4.d[1]" };
    (4) => { "v5.d[0]" };
    (5) => { "v5.d[1]" };
    (6) => { "v6.d[0]" };
    (7) => { "v6.d[1]" };
}

macro_rules! beta_fmadd {
    (C, $m0:expr, $r1:expr, 2) => {
        concat!(
            "ldr q1, ", $m0, "\n",
            "fmla v", $r1, ".2d, v1.2d, v0.d[0]\n",
        ) 
    };
    (C, $m0:expr, $r1:expr, 1) => {
        concat!(
            "ldr q1, ", $m0, "\n",
            "fadd v", $r1, ".2d, v", $r1, ".2d, v1.2d\n",
        ) 
    };
}

macro_rules! vzeroall {
    ($r0:tt, $r1:tt) => {
        seq!(r in $r0..=$r1 {
            concat!(#("dup v", r, ".2d, xzr \n",)*)
        })
    }
}

macro_rules! vfmadd {
    ($r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            "fmla v", $r3, ".2d", ", v", $r1,".2d, ", $r2, "\n",
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
    ($layout:tt, $r1:expr, $m0:expr) => {
        concat!(
            "str q", $r1, ", ", $m0,  "\n",
        )
    };
}

macro_rules! alpha_scale_0 {
    ($r0:tt, $r1:tt) => {
        seq!(r in $r0..=$r1 {
            concat!(
                "ldr d1, [{alphax}]", "\n",
                #(
                    "fmul  v", r, ".2d, v", r, ".2d, v1.d[0]\n",
                )*
            )
        })
    }
}

macro_rules! load_beta {
    () => {
        concat!(
            "ldr d0, [{betax}]", "\n",
            "/* {betax} */", "\n",

            "fcmp d0,#0.0", "\n",
        )
    }
}


macro_rules! loadp {
    (3, $m0:expr) => {
        concat!(
            loadp_unit!(mem!($m0), 0),
            loadp_unit!(mem!($m0, "0x10"), 1),
            loadp_unit!(mem!($m0, "0x20"), 2),
            loadp_unit!(mem!($m0, "0x30"), 3),
            loadp_unit!(mem!($m0, "0x40"), 4),
            loadp_unit!(mem!($m0, "0x50"), 5),
        )
    };
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
            "ldr {x0}, [{dim_arrx}, #16]\n",
            "add {x1}, {cx}, {x0} \n",
            "add {x2}, {x1}, {x0} \n",
            "add {x3}, {x2}, {x0} \n",
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
    (B,$nr:tt) => {
        concat!("add {bx}, {bx}, #8*", $nr, " \n")
    };
}

macro_rules! alpha_scale {
    () => {
        alpha_scale_0!(8,31)
    };
}

macro_rules! cr_3 {
    (0,0) => { 8 };
    (1,0) => { 9 };
    (2,0) => { 10 };

    (0,1) => { 11 };
    (1,1) => { 12 };
    (2,1) => { 13 };

    (0,2) => { 14 };
    (1,2) => { 15 };
    (2,2) => { 16 };

    (0,3) => { 17 };
    (1,3) => { 18 };
    (2,3) => { 19 };

    (0,4) => { 20 };
    (1,4) => { 21 };
    (2,4) => { 22 };

    (0,5) => { 23 };
    (1,5) => { 24 };
    (2,5) => { 25 };

    (0,6) => { 26 };
    (1,6) => { 27 };
    (2,6) => { 28 };

    (0,7) => { 29 };
    (1,7) => { 30 };
    (2,7) => { 31 };
}

macro_rules! cr_2 {
    (0,0) => { 8 };
    (1,0) => { 9 };

    (0,1) => { 10 };
    (1,1) => { 11 };

    (0,2) => { 12 };
    (1,2) => { 13 };

    (0,3) => { 14 };
    (1,3) => { 15 };

    (0,4) => { 16 };
    (1,4) => { 17 };

    (0,5) => { 18 };
    (1,5) => { 19 };

    (0,6) => { 20 };
    (1,6) => { 21 };

    (0,7) => { 22 };
    (1,7) => { 23 };
}

macro_rules! cr_1 {
    (0,0) => { 8 };

    (0,1) => { 9 };

    (0,2) => { 10 };

    (0,3) => { 11 };

    (0,4) => { 12 };

    (0,5) => { 13 };

    (0,6) => { 14 };

    (0,7) => { 15 };
}

macro_rules! load_b {
    (B, $r:expr) => {
        concat!(
            "ldr q", $r, ", [{bx}]", "\n",
        )
    };
    (B, $r1:expr, $r2:expr) => {
        concat!(
            "ldr q", $r1, ", [{bx}]", "\n",
            "ldr q", $r2, ", [{bx}, #0x10]", "\n",
        )
    };
    (B, $r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            "ldr q", $r1, ", [{bx}]", "\n",
            "ldr q", $r2, ", [{bx}, #0x10]", "\n",
            "ldr q", $r3, ", [{bx}, #0x20]", "\n",
        )
    };
    (B, $r1:expr, $r2:expr, $r3:expr, $r4:expr) => {
        concat!(
            "ldr q", $r1, ", [{bx}]", "\n",
            "ldr q", $r2, ", [{bx}, #0x10]", "\n",
            "ldr q", $r3, ", [{bx}, #0x20]", "\n",
            "ldr q", $r4, ", [{bx}, #0x30]", "\n",
        )
    };
}


macro_rules! load_a {
    ($mr:tt) => {
        loadp!($mr, "{ax}")
    };
}


macro_rules! fmadd_3 {
    ($ni:tt) => {
        concat!(
            vfmadd!(0, br_3!($ni), cr_3!(0, $ni)),
            vfmadd!(1, br_3!($ni), cr_3!(1, $ni)),
            vfmadd!(2, br_3!($ni), cr_3!(2, $ni)),
        )
    };
}

macro_rules! fmadd_2 {
    ($ni:tt) => {
        concat!(
            vfmadd!(0, br_2!($ni), cr_2!(0, $ni)),
            vfmadd!(1, br_2!($ni), cr_2!(1, $ni)),
        )
    };
}

macro_rules! fmadd_1 {
    ($ni:tt) => {
        concat!(
            vfmadd!(0, br_1!($ni), cr_1!(0, $ni)),
        )
    };
}

// ***************************** 3 ******************************* //
macro_rules! step_3 {
    ($b_layout:tt, $nr:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(3),
                "add {ax}, {ax}, #4*3*4 \n",
                load_b!($b_layout, 4, 5, 6, 7),
                #(
                    fmadd_3!(n),
                )*
                inc_b!($b_layout,$nr), 
            )
        })
    };
}

// ***************************** 2 ******************************* //
macro_rules! step_2 {
    ($b_layout:tt, $nr:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(2),
                "add {ax}, {ax}, #4*2*4 \n",
                load_b!($b_layout, 4, 5),
                #(
                    fmadd_2!(n),
                )*
                inc_b!($b_layout,$nr), 
            )
        })
    };
}

// ***************************** 2 ******************************* //
macro_rules! step_1 {
    ($b_layout:tt, $nr:tt) => {
        seq!(n in 0..$nr {
            concat!(
                load_a!(1),
                "add {ax}, {ax}, #4*1*4 \n",
                load_b!($b_layout, 2, 3),
                #(
                    fmadd_1!(n),
                )*
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
            "prfm pldl1keep, [{cx}] \n",
            "prfm pldl1keep, [{cx},#64]\n",
            "prfm pldl1keep, [{cx},#96]\n",
            "prfm pldl1keep, [{x1}] \n",
            "prfm pldl1keep, [{x1},#64]\n",
            "prfm pldl1keep, [{x1},#96]\n",
            "prfm pldl1keep, [{x2}] \n",
            "prfm pldl1keep, [{x2},#64]\n",
            "prfm pldl1keep, [{x2},#96]\n",
            "prfm pldl1keep, [{x3}] \n",
            "prfm pldl1keep, [{x3},#64]\n",
            "prfm pldl1keep, [{x3},#96]\n",
        )
    };
}

def_ukernel_neon!(step_3, acc_3, store_3, 3, 8, B, C, ukernel_bbc);

def_ukernel_neon!(step_3, acc_3, store_3, 3, 8, B, C, ukernel_3_bbp);
def_ukernel_neon!(step_2, acc_2, store_2, 2, 8, B, C, ukernel_2_bbp);
def_ukernel_neon!(step_1, acc_1, store_1, 1, 8, B, C, ukernel_1_bbp);
