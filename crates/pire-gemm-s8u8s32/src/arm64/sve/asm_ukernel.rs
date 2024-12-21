use seq_macro::seq;
use crate::{TA, TB, TC, TC_SIZE};
use pire_base::{
    def_ukernel_sve_i8mm,
    acc_2, acc_1,
    store_2, store_1,
};
use super::super::sve_vs;

const ONE_SCALAR: f32 = 0f32;
const ZERO_SCALAR: f32 = 0f32;
type TS = f32;

macro_rules! v_i {
    ($m0:tt, $ni:tt) => {
        concat!("[", $m0, ", #", $ni, ", MUL VL]")
    }
}

macro_rules! set_predicate {
    (M) => { "mov {m_s}, #0 \n whilelo p1.s, {m_s}, {m_e} \n" };
    (C) => { "/* {m_s}, {m_e} */" }
}

macro_rules! inc_a {
    ($mr:tt) => {
        concat!("add {ax}, {ax}, {incax} \n")
    };
}

macro_rules! unzip_tuple {
    ($r1:tt, $r2:tt,$rt1:tt,$rt2:tt) => {
        concat!(
            "uzp1 z", $rt1, ".d, z", $r1, ".d, z", $r2, ".d\n",
            "uzp2 z", $rt2, ".d, z", $r1, ".d, z", $r2, ".d\n",
            // copy uzp1 to z8 and uzp2 to z11
            "orr z", $r1, ".b, z", $rt1, ".b, z", $rt1, ".b\n",
            "orr z", $r2, ".b, z", $rt2, ".b, z", $rt2, ".b\n",
        )
    };
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
            "ld1w {{ z1.s }}, p0/z, ", $m0, "\n",
            "add z", $r1, ".s, p0/m, z", $r1, ".s, z1.s\n",
        ) 
    };
    (M, $m0:expr, $r1:expr,1) => {
        concat!(
            "ld1w {{ z1.s }}, p1/z, ", $m0, "\n",
            "add z", $r1, ".s, p1/m, z", $r1, ".s, z1.s\n",
        ) 
    };

    (C, $m0:expr, $r1:expr,2) => {
        concat!(
            "ld1w {{ z1.s }}, p0/z, ", $m0, "\n",
            "scvtf z1.s, p0/m, z1.s\n",
            "scvtf z", $r1, ".s, p0/m, z", $r1, ".s\n",
            "fmla z", $r1, ".s, z1.s, z0.s[0]\n",
            "fcvtzs z", $r1, ".s, p0/m, z", $r1, ".s\n",
        ) 
    };
    (M, $m0:expr, $r1:expr,2) => {
        concat!(
            "ld1w {{ z1.s }}, p1/z, ", $m0, "\n",
            "scvtf z1.s, p0/m, z1.s\n",
            "scvtf z", $r1, ".s, p0/m, z", $r1, ".s\n",
            "fmla z", $r1, ".s, z1.s, z0.s[0]\n",
            "fcvtzs z", $r1, ".s, p0/m, z", $r1, ".s\n",
        ) 
    };
}

macro_rules! vzeroall {
    ($r0:tt, $r1:tt) => {
        seq!(r in $r0..=$r1 {
            concat!(#("dup z", r, ".s, #0 \n",)*)
        })
    }
}

macro_rules! vfmadd {
    ($r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            "usmmla z", $r3, ".s", ", z", $r2,".b, z", $r1, ".b\n",
        ) 
    };
}

macro_rules! loadp_unit {
    ($m0:expr, 0) => {
        concat!(
            "ld1w {{ z0.s }}, p0/z, [", $m0, "]\n",
            "ld1w {{ z1.s }}, p0/z, [", $m0, ", #1, MUL VL]\n",
        )
    };
    ($m0:expr, 1) => {
        concat!(
            "ld1w {{ z2.s }}, p0/z, [", $m0, ", #2, MUL VL]\n",
            "ld1w {{ z3.s }}, p0/z, [", $m0, ", #3, MUL VL]\n",
        )
    };
}

macro_rules! storep_unit {
    (C, $r1:expr, $m0:expr) => {
        concat!(
            "st1w {{ z", $r1, ".s }}, p0, ", $m0, "\n",
        )
    };
    (M, $r1:expr, $m0:expr) => {
        concat!(
            "st1w {{ z", $r1, ".s }}, p1, ", $m0, "\n",
        )
    };
}

macro_rules! alpha_scale_0 {
    ($r0:tt, $r1:tt) => {
        seq!(r in $r0..=$r1 {
            concat!(
                "ld1rqw {{ z1.s }}, p0/z, [{alphax}]", "\n",

                #(
                    "scvtf z", r, ".s, p0/m, z", r, ".s\n",
                    "fmul  z", r, ".s, z", r, ".s, z1.s[0]\n",
                    "fcvtzs z", r, ".s, p0/m, z", r, ".s\n",
                )*
            )
        })
    }
}

macro_rules! load_beta {
    () => {
        concat!(
            "ld1rqw {{ z0.s }}, p0/z, [{betax}]", "\n",
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
        concat!(
            "add {bx}, {bx}, #", $nr, "*8 \n",
        )
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
            "ld1rqd {{ z4.d }}, p0/z, [{bx}]", "\n",
        )
    };
    (B, 2) => {
        concat!(
            "ld1rqd {{ z5.d }}, p0/z, [{bx}, #0x10]", "\n",
        )
    };
    (B, 4) => {
        concat!(
            "ld1rqd {{ z6.d }}, p0/z, [{bx}, #0x20]", "\n",
        )
    };
    (B, 6) => {
        concat!(
            "ld1rqd {{ z7.d }}, p0/z, [{bx}, #0x30]", "\n",
        )
    };
    (B, 8) => {
        concat!(
            "ld1rqd {{ z4.d }}, p0/z, [{bx}, #0x40]", "\n",
        )
    };
    (B, 10) => {
        concat!(
            "ld1rqd {{ z5.d }}, p0/z, [{bx}, #0x50]", "\n",
        )
    };

    (B, $nr:tt) => {
        ""
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

macro_rules! step_2 {
    ($b_layout:tt, $nr:tt) => {
        seq!(n in 0..$nr {
            concat!(
                #(
                    load_b!($b_layout, n),
                    fmadd_2!(n),
                )*
            )
        })
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
            "prfm pldl1keep, [{cx},#64]\n",
            "prfm pldl1keep, [{x1}] \n",
            "prfm pldl1keep, [{x1},#64]\n",
            "prfm pldl1keep, [{x2}] \n",
            "prfm pldl1keep, [{x2},#64]\n",
            "prfm pldl1keep, [{x3}] \n",
            "prfm pldl1keep, [{x3},#64]\n",
            "prfm pldl1keep, [{x4}] \n",
            "prfm pldl1keep, [{x4},#64]\n",
            "prfm pldl1keep, [{x5}] \n",
            "prfm pldl1keep, [{x5},#64]\n",
            "prfm pldl1keep, [{x6}] \n",
            "prfm pldl1keep, [{x6},#64]\n",
            "prfm pldl1keep, [{x7}] \n",
            "prfm pldl1keep, [{x7},#64]\n",
            "prfm pldl1keep, [{x8}] \n",
            "prfm pldl1keep, [{x8},#64]\n",
            "prfm pldl1keep, [{x9}] \n",
            "prfm pldl1keep, [{x9},#64]\n",
            "prfm pldl1keep, [{x10}] \n",
            "prfm pldl1keep, [{x10},#64]\n",
            "prfm pldl1keep, [{x11}] \n",
            "prfm pldl1keep, [{x11},#64]\n",
        )
    };
}

def_ukernel_sve_i8mm!(step_1, acc_1, store_1, 1, 12, B, M, ukernel_1_bbp);
def_ukernel_sve_i8mm!(step_2, acc_2, store_2, 2, 12, B, M, ukernel_2_bbp);

def_ukernel_sve_i8mm!(step_2, acc_2, store_2, 2, 12, B, C, ukernel_bbc);
