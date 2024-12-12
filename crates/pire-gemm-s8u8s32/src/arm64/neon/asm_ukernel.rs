use seq_macro::seq;
use crate::{TA, TB, TC, TC_SIZE};
use pire_base::{
    prefetch_0, def_ukernel_neon_i8mm,
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

macro_rules! inc_a {
    ($mr:tt) => {
        concat!("add {ax}, {ax}, #32*", $mr, " \n")
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
    ($m0:expr, 0) => {
        concat!(
            "ldr q0, [", $m0, "]\n",
            "ldr q1, [", $m0, ", #16]\n",
        )
    };
    ($m0:expr, 1) => {
        concat!(
            "ldr q2, [", $m0, ", #32]\n",
            "ldr q3, [", $m0, ", #48]\n",
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
}


macro_rules! vzero_kernel {
    () => {vzeroall!(8,31)};
}

macro_rules! inc_b {
    (S,$nr:tt) => {
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

macro_rules! br_2 {
    (0) => { 4 };
    (2) => { 5 };
    (4) => { 6 };
    (6) => { 7 };
    (8) => { 4 };
    (10) => { 5 };
    ($nr:tt) => { 4 };
}

macro_rules! br_1 {
    (0) => { 4 };
    (2) => { 5 };
    (4) => { 6 };
    (6) => { 7 };
    (8) => { 4 };
    (10) => { 5 };
    ($nr:tt) => { 4 };
}

macro_rules! cr {
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

macro_rules! dr {
    (0,0) => { 8 };
    (1,0) => { 9 };
    (2,0) => { 10 };
    (3,0) => { 11 };

    (0,2) => { 12 };
    (1,2) => { 13 };
    (2,2) => { 14 };
    (3,2) => { 15 };

    (0,4) => { 16 };
    (1,4) => { 17 };
    (2,4) => { 18 };
    (3,4) => { 19 };

    (0,6) => { 20 };
    (1,6) => { 21 };
    (2,6) => { 22 };
    (3,6) => { 23 };

    (0,8) => { 24 };
    (1,8) => { 25 };
    (2,8) => { 26 };
    (3,8) => { 27 };

    (0,10) => { 28 };
    (1,10) => { 29 };
    (2,10) => { 30 };
    (3,10) => { 31 };
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

macro_rules! fmadd_2 {
    (1) => {""};
    (3) => {""};
    (5) => {""};
    (7) => {""};
    (9) => {""};
    (11) => {""};
    ($ni:tt) => {
        concat!(
            vfmadd!(0, br_2!($ni), dr!(0,$ni)),
            vfmadd!(1, br_2!($ni), dr!(1,$ni)),
            vfmadd!(2, br_2!($ni), dr!(2,$ni)),
            vfmadd!(3, br_2!($ni), dr!(3,$ni)),
        )
    };
}

macro_rules! step_2 {
    ($b_layout:tt, $nr:tt) => {
        seq!(n in 0..$nr {
            concat!(
                #(
                    load_b!($b_layout, n),
                    fmadd_2!(n),
                )*
                inc_b!($b_layout,$nr), 
            )
        })
    };
}


macro_rules! fmadd_1 {
    (1) => {""};
    (3) => {""};
    (5) => {""};
    (7) => {""};
    (9) => {""};
    (11) => {""};
    ($ni:tt) => {
        concat!(
            vfmadd!(0, br_1!($ni), dr!(0,$ni)),
            vfmadd!(1, br_1!($ni), dr!(1,$ni)),
        )
    };
}

macro_rules! step_1 {
    ($b_layout:tt, $nr:tt) => {
        seq!(n in 0..$nr {
            concat!(
                #(
                    load_b!($b_layout, n),
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
