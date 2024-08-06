use seq_macro::seq;
use std::arch::asm;

use corenum_base::asm_macro::*;
use crate::{TA, TB, TC};

use paste::paste;

use std::arch::x86_64::*;


macro_rules! inc_a {
	(C) => {
    	"add {x1}, {ax}"
	};
	(B) => {
    	""
	};
}

macro_rules! inc_b {
	(C,$nr:tt) => {
    	""
	};
	(R,$nr:tt) => {
    	"add {x2},{bx}"
	};
	(B,$nr:tt) => {
    	""
	};
}


macro_rules! inc_a_k_unroll {
	($a:tt, C, $X:tt, $K:tt) => {
        $a = $a.wrapping_add($X*$K);
	};
	($a:tt, B, $X:tt, $K:tt) => {
        $a = $a.wrapping_add($K*$X);
	};
}

macro_rules! inc_b_k_unroll {
	($b:tt, $x3:tt, C, 6, $K:tt) => {
        $b = $b.wrapping_add($K);
        $x3 = $x3.wrapping_add($K);
	};
	($b:tt, $x3:tt, C, 5, $K:tt) => {
        $b = $b.wrapping_add($K);
        $x3 = $x3.wrapping_add($K);
	};
	($b:tt, $x3:tt, C, 4, $K:tt) => {
        $b = $b.wrapping_add($K);
        $x3 = $x3.wrapping_add($K);
	};
	($b:tt, $x3:tt, C, $nr:tt, $K:tt) => {
        $b = $b.wrapping_add($K);
	};

	($b:tt, $x3:tt, R, $X:tt, $K:tt) => {
	};
	($b:tt, $x3:tt, B, $X:tt, $K:tt) => {
        $b = $b.wrapping_add($K*$X);
	};
}



macro_rules! load_b {
	(C, 0, $K:tt, $X:tt, $r:expr) => {
    	concat!(
        	"vbroadcastss ", $K, "*4({bx}),%ymm", $r, "\n",
    	)
	};
	(C, 1, $K:tt, $X:tt, $r:expr) => {
    	concat!(
        	"vbroadcastss ", $K, "*4({bx},{x2},1),%ymm", $r, "\n",
    	)
	};
	(C, 2, $K:tt, $X:tt, $r:expr) => {
    	concat!(
        	"vbroadcastss ", $K, "*4({bx},{x2},2),%ymm", $r, "\n",
    	)
	};
	(C, 3, $K:tt, $X:tt, $r:expr) => {
    	concat!(
        	"vbroadcastss ", $K, "*4({x3}),%ymm", $r, "\n",
    	)
	};
	(C, 4, $K:tt, $X:tt, $r:expr) => {
    	concat!(
        	"vbroadcastss ", $K, "*4({x3},{x2},1),%ymm", $r, "\n",
    	)
	};
	(C, 5, $K:tt, $X:tt, $r:expr) => {
    	concat!(
        	"vbroadcastss ", $K, "*4({x3},{x2},2),%ymm", $r, "\n",
    	)
	};
	(R, $N:tt, $K:tt, $X:tt, $r:expr) => {
    	concat!(
        	"vbroadcastss ", $N, "*4({bx}),%ymm", $r, "\n",
    	)
	};
	(B, $N:tt, $K:tt, $X:tt, $r:expr) => {
    	concat!(
        	"vbroadcastss ", $K, "*", $X, "*4+", $N, "*4({bx}), %ymm", $r, "\n",
    	)
	};
}


macro_rules! load_a {
	(24, B, $K:tt) => {
    	loadps!(24, B, concat!("96*",$K,"({ax})"), 0, 1, 2)
	};
	(24, C, $K:tt) => {
    	loadps!($mr, C, "0({ax})", 0, 1, 2)
	};
	(16, B, $K:tt) => {
    	loadps!(16, B, concat!("64*",$K,"({ax})"), 0, 1)
	};
	(16, C, $K:tt) => {
    	loadps!($mr, C, "0({ax})", $r1, $r2)
	};
	(8, B, $K:tt) => {
    	loadps!(8, B, concat!("32*",$K,"({ax})"), 0)
	};
	(8, C, $K:tt) => {
    	loadps!($mr, C, "0({ax})", 0)
	};
}


macro_rules! asm_24x4_step {
	($N:tt, 4, $a_layout:tt, $b_layout:tt, $K:tt) => {
    	concat!(
        	load_a!($N, $a_layout, $K), "\n",
        	load_b!($b_layout, 0, $K, 4, 3), "\n",
			vfmadd231ps!(0, 3, 4),
			vfmadd231ps!(1, 3, 5),
			vfmadd231ps!(2, 3, 6),
        	load_b!($b_layout, 1, $K, 4, 3), "\n",
			vfmadd231ps!(0, 3, 7),
			vfmadd231ps!(1, 3, 8),
			vfmadd231ps!(2, 3, 9),
        	load_b!($b_layout, 2, $K, 4, 3), "\n",
			vfmadd231ps!(0, 3, 10),
			vfmadd231ps!(1, 3, 11),
			vfmadd231ps!(2, 3, 12),
        	load_b!($b_layout, 3, $K, 4, 3), "\n",
			vfmadd231ps!(0, 3, 13),
			vfmadd231ps!(1, 3, 14),
			vfmadd231ps!(2, 3, 15),
    	)
	};
	($N:tt, 3, $a_layout:tt, $b_layout:tt, $K:tt) => {
    	concat!(
        	load_a!($N, $a_layout, $K), "\n",
        	load_b!($b_layout, 0, $K, 3, 3), "\n",
			vfmadd231ps!(0, 3, 4),
			vfmadd231ps!(1, 3, 5),
			vfmadd231ps!(2, 3, 6),
        	load_b!($b_layout, 1, $K, 3, 3), "\n",
			vfmadd231ps!(0, 3, 7),
			vfmadd231ps!(1, 3, 8),
			vfmadd231ps!(2, 3, 9),
        	load_b!($b_layout, 2, $K, 3, 3), "\n",
			vfmadd231ps!(0, 3, 10),
			vfmadd231ps!(1, 3, 11),
			vfmadd231ps!(2, 3, 12),
    	)
	};
	($N:tt, 2, $a_layout:tt, $b_layout:tt, $K:tt) => {
    	concat!(
        	load_a!($N, $a_layout, $K), "\n",
        	load_b!($b_layout, 0, $K, 2, 3), "\n",
			vfmadd231ps!(0, 3, 4),
			vfmadd231ps!(1, 3, 5),
			vfmadd231ps!(2, 3, 6),
        	load_b!($b_layout, 1, $K, 2, 3), "\n",
			vfmadd231ps!(0, 3, 7),
			vfmadd231ps!(1, 3, 8),
			vfmadd231ps!(2, 3, 9),
    	)
	};
	($N:tt, 1, $a_layout:tt, $b_layout:tt, $K:tt) => {
    	concat!(
        	load_a!($N, $a_layout, $K), "\n",
        	load_b!($b_layout, 0, $K, 1, 3), "\n",
			vfmadd231ps!(0, 3, 4),
			vfmadd231ps!(1, 3, 5),
			vfmadd231ps!(2, 3, 6),
    	)
	};
}



macro_rules! asm_16x6_step {
	($N:tt, 6, $a_layout:tt, $b_layout:tt, $K:tt) => {
    	concat!(
        	load_a!($N, $a_layout, $K), "\n",
        	load_b!($b_layout, 0, $K, 6, 2), "\n",
			vfmadd231ps!(0, 2, 4),
			vfmadd231ps!(1, 2, 5),
        	load_b!($b_layout, 1, $K, 6, 3), "\n",
			vfmadd231ps!(0, 3, 6),
			vfmadd231ps!(1, 3, 7),
        	load_b!($b_layout, 2, $K, 6, 2), "\n",
			vfmadd231ps!(0, 2, 8),
			vfmadd231ps!(1, 2, 9),
        	load_b!($b_layout, 3, $K, 6, 3), "\n",
			vfmadd231ps!(0, 3, 10),
			vfmadd231ps!(1, 3, 11),

        	load_b!($b_layout, 4, $K, 6, 2), "\n",
			vfmadd231ps!(0, 2, 12),
			vfmadd231ps!(1, 2, 13),
        	load_b!($b_layout, 5, $K, 6, 3), "\n",
			vfmadd231ps!(0, 3, 14),
			vfmadd231ps!(1, 3, 15),
    	)
	};
	($N:tt, 5, $a_layout:tt, $b_layout:tt, $K:tt) => {
    	concat!(
        	load_a!($N, $a_layout, $K), "\n",

        	load_b!($b_layout, 0, $K, 5, 2), "\n",
			vfmadd231ps!(0, 2, 4),
			vfmadd231ps!(1, 2, 5),
        	load_b!($b_layout, 1, $K, 5, 3), "\n",
			vfmadd231ps!(0, 3, 6),
			vfmadd231ps!(1, 3, 7),

        	load_b!($b_layout, 2, $K, 5, 14), "\n",
			vfmadd231ps!(0, 14, 8),
			vfmadd231ps!(1, 14, 9),
        	load_b!($b_layout, 3, $K, 5, 15), "\n",
			vfmadd231ps!(0, 15, 10),
			vfmadd231ps!(1, 15, 11),

        	load_b!($b_layout, 4, $K, 5, 2), "\n",
			vfmadd231ps!(0, 2, 12),
			vfmadd231ps!(1, 2, 13)
    	)
	};
	($N:tt, 4, $a_layout:tt, $b_layout:tt, $K:tt) => {
    	concat!(
        	load_a!($N, $a_layout, $K), "\n",

        	load_b!($b_layout, 0, $K, 4, 2), "\n",
			vfmadd231ps!(0, 2, 4),
			vfmadd231ps!(1, 2, 5),
        	load_b!($b_layout, 1, $K, 4, 3), "\n",
			vfmadd231ps!(0, 3, 6),
			vfmadd231ps!(1, 3, 7),

        	load_b!($b_layout, 2, $K, 4, 12), "\n",
			vfmadd231ps!(0, 12, 8),
			vfmadd231ps!(1, 12, 9),
        	load_b!($b_layout, 3, $K, 4, 13), "\n",
			vfmadd231ps!(0, 13, 10),
			vfmadd231ps!(1, 13, 11),
    	)
	};
	($N:tt, 3, $a_layout:tt, $b_layout:tt, $K:tt) => {
    	concat!(
        	load_a!($N, $a_layout, $K), "\n",

        	load_b!($b_layout, 0, $K, 3, 2), "\n",
			vfmadd231ps!(0, 2, 4),
			vfmadd231ps!(1, 2, 5),
        	load_b!($b_layout, 1, $K, 3, 3), "\n",
			vfmadd231ps!(0, 3, 6),
			vfmadd231ps!(1, 3, 7),

        	load_b!($b_layout, 2, $K, 3, 10), "\n",
			vfmadd231ps!(0, 10, 8),
			vfmadd231ps!(1, 10, 9),
    	)
	};
	($N:tt, 2, $a_layout:tt, $b_layout:tt, $K:tt) => {
    	concat!(
        	load_a!($N, $a_layout, $K), "\n",

        	load_b!($b_layout, 0, $K, 2, 2), "\n",
			vfmadd231ps!(0, 2, 4),
			vfmadd231ps!(1, 2, 5),
        	load_b!($b_layout, 1, $K, 2, 3), "\n",
			vfmadd231ps!(0, 3, 6),
			vfmadd231ps!(1, 3, 7),

    	)
	};
	($N:tt, 1, $a_layout:tt, $b_layout:tt, $K:tt) => {
    	concat!(
        	load_a!($N, $a_layout, $K), "\n",

        	load_b!($b_layout, 0, $K, 1, 2), "\n",
			vfmadd231ps!(0, 2, 4),
			vfmadd231ps!(1, 2, 5),
    	)
	};
}


macro_rules! asm_8x6_step {
	($N:tt, 6, $a_layout:tt, $b_layout:tt, $K:tt) => {
    	concat!(
        	load_a!($N, $a_layout, $K), "\n",
        	load_b!($b_layout, 0, $K, 6, 1), "\n",
			vfmadd231ps!(0, 1, 4),
        	load_b!($b_layout, 1, $K, 6, 2), "\n",
			vfmadd231ps!(0, 2, 5),
        	load_b!($b_layout, 2, $K, 6, 3), "\n",
			vfmadd231ps!(0, 3, 6),
        	load_b!($b_layout, 3, $K, 6, 10), "\n",
			vfmadd231ps!(0, 10, 7),
        	load_b!($b_layout, 4, $K, 6, 11), "\n",
			vfmadd231ps!(0, 11, 8),
        	load_b!($b_layout, 5, $K, 6, 12), "\n",
			vfmadd231ps!(0, 12, 9),
    	)
	};
	($N:tt, 5, $a_layout:tt, $b_layout:tt, $K:tt) => {
    	concat!(
        	load_a!($N, $a_layout, $K), "\n",
        	load_b!($b_layout, 0, $K, 5, 1), "\n",
			vfmadd231ps!(0, 1, 4),
        	load_b!($b_layout, 1, $K, 5, 2), "\n",
			vfmadd231ps!(0, 2, 5),
        	load_b!($b_layout, 2, $K, 5, 3), "\n",
			vfmadd231ps!(0, 3, 6),
        	load_b!($b_layout, 3, $K, 5, 10), "\n",
			vfmadd231ps!(0, 10, 7),
        	load_b!($b_layout, 4, $K, 5, 11), "\n",
			vfmadd231ps!(0, 11, 8),
    	)
	};
	($N:tt, 4, $a_layout:tt, $b_layout:tt, $K:tt) => {
    	concat!(
        	load_a!($N, $a_layout, $K), "\n",
        	load_b!($b_layout, 0, $K, 4, 1), "\n",
			vfmadd231ps!(0, 1, 4),
        	load_b!($b_layout, 1, $K, 4, 2), "\n",
			vfmadd231ps!(0, 2, 5),
        	load_b!($b_layout, 2, $K, 4, 3), "\n",
			vfmadd231ps!(0, 3, 6),
        	load_b!($b_layout, 3, $K, 4, 10), "\n",
			vfmadd231ps!(0, 10, 7),
    	)
	};
	($N:tt, 3, $a_layout:tt, $b_layout:tt, $K:tt) => {
    	concat!(
        	load_a!($N, $a_layout, $K), "\n",
        	load_b!($b_layout, 0, $K, 3, 1), "\n",
			vfmadd231ps!(0, 1, 4),
        	load_b!($b_layout, 1, $K, 3, 2), "\n",
			vfmadd231ps!(0, 2, 5),
        	load_b!($b_layout, 2, $K, 3, 3), "\n",
			vfmadd231ps!(0, 3, 6),
    	)
	};
	($N:tt, 2, $a_layout:tt, $b_layout:tt, $K:tt) => {
    	concat!(
        	load_a!($N, $a_layout, $K), "\n",
        	load_b!($b_layout, 0, $K, 2, 1), "\n",
			vfmadd231ps!(0, 1, 4),
        	load_b!($b_layout, 1, $K, 2, 2), "\n",
			vfmadd231ps!(0, 2, 5),
    	)
	};
	($N:tt, 1, $a_layout:tt, $b_layout:tt, $K:tt) => {
    	concat!(
        	load_a!($N, $a_layout, $K), "\n",
        	load_b!($b_layout, 0, $K, 1, 1), "\n",
			vfmadd231ps!(0, 1, 4),
    	)
	};
}

macro_rules! prefetch_0 {
	($dist:tt, $reg:tt, $k_i:tt) => {
		concat!(
			"prefetcht0 ", $dist, "+", $k_i, "*64(", $reg, ")"
		)
	};
}

macro_rules! prefetch_a {
	($mr:tt, $layout:tt, 0, $unroll:tt, $k_i:tt) => {
		""
	};
	(24, B, $dist:tt, $unroll:tt, 0) => {
		prefetch_0!($dist, "{ax}", 0)
	};

	(24, B, $dist:tt, $unroll:tt, 1) => {
		concat!(
			prefetch_0!($dist, "{ax}", 1), "\n",
			prefetch_0!($dist, "{ax}", 2)
		)
	};

	(24, B, $dist:tt, $unroll:tt, 2) => {
		prefetch_0!($dist, "{ax}", 3)
	};

	(24, B, $dist:tt, $unroll:tt, 3) => {
		concat!(
			prefetch_0!($dist, "{ax}", 4), "\n",
			prefetch_0!($dist, "{ax}", 5)
		)
	};

	(24, B, $dist:tt, $unroll:tt, 4) => {
		prefetch_0!($dist, "{ax}", 6)
	};
	(24, B, $dist:tt, $unroll:tt, 5) => {
		concat!(
			prefetch_0!($dist, "{ax}", 7), "\n",
			prefetch_0!($dist, "{ax}", 8)
		)
	};
	(24, B, $dist:tt, $unroll:tt, 6) => {
		prefetch_0!($dist, "{ax}", 9)
	};

	(24, B, $dist:tt, $unroll:tt, 7) => {
		concat!(
			prefetch_0!($dist, "{ax}", 10), "\n",
			prefetch_0!($dist, "{ax}", 11)
		)
	};

	(16, B, $dist:tt, $unroll:tt, $i:tt) => {
		prefetch_0!($dist, "{ax}", $i)
	};

	(8, B, $dist:tt, $unroll:tt, 0) => {
		prefetch_0!($dist, "{ax}", 0)
	};
	(8, B, $dist:tt, $unroll:tt, 2) => {
		prefetch_0!($dist, "{ax}", 1)
	};
	(8, B, $dist:tt, $unroll:tt, 4) => {
		prefetch_0!($dist, "{ax}", 2)
	};
	(8, B, $dist:tt, $unroll:tt, 6) => {
		prefetch_0!($dist, "{ax}", 3)
	};


	(8, B, $dist:tt, $unroll:tt, $k_i:tt) => {
		""
	};
}

// Only R layout has prefetch for eack k_i
macro_rules! prefetch_b {
	($nr:tt, R, $dist:tt, $unroll:tt, $i:tt) => {
		prefetch_0!(0, "{bx},{x2},8", 0)
	};
	($nr:tt, $layout:tt, $dist:tt, $unroll:tt, 0) => {
		prefetch_0!($dist, "{bx}", 0)
	};
	($nr:tt, $layout:tt, $dist:tt, $unroll:tt, 4) => {
		prefetch_0!($dist, "{bx}", 1)
	};

	($nr:tt, $layout:tt, $dist:tt, $unroll:tt, $k_i:tt) => {
		""
	};
}

macro_rules! asm_str {
    (
        $mr:expr, $nr:expr, $a_layout:expr, $b_layout:expr, $unroll:expr,
        $pfa_dist:expr, $pfb_dist:expr,
        $asm_step_macro:tt
    ) => {
            seq!( i in 0..$unroll{
                concat!(
                    unused_var_asm_b!($nr, $b_layout), "\n",
                    unused_var_asm_a!($mr, $a_layout), "\n",
                    #(
                        prefetch_a!($mr, $a_layout, $pfa_dist, $unroll, i), "\n",
                        prefetch_b!($nr, $b_layout, $pfb_dist, $unroll, i), "\n",
                        $asm_step_macro!($mr, $nr, $a_layout, $b_layout, i), "\n",
                        inc_a!($a_layout), "\n",
                        inc_b!($b_layout,$nr),  "\n",
                    )*
                )
            })
    };
}


macro_rules! unused_var_asm_b {
    (6,B) => {
        "/* {x2} {x3} */"
    };
    (5,B) => {
        "/* {x2} {x3} */"
    };
    (4,B) => {
        "/* {x2} {x3} */"
    };
    (1,B) => {
        "/* {x2} */"
    };
    (1,C) => {
        "/* {x2} */"
    };
    ($nr:tt,B) => {
        "/* {x2} */"
    };
    (6,R) => {
        "/* {x3} */"
    };
    (5,R) => {
        "/* {x3} */"
    };
    (4,R) => {
        "/* {x3} */"
    };
    ($nr:tt,R) => {
        ""
    };
    ($nr:tt,C) => {
        ""
    };
}

macro_rules! unused_var_asm_a {
    ($nr:tt,B) => {
        "/* {x1} */"
    };
    // (3,C) => {
    //     "/* {x3} */"
    // };
    // (2,B) => {
    //     "/* {x3} */"
    // };
    // (1,B) => {
    //     "/* {x3} */"
    // };
}


macro_rules! asm_wrapped {
    (
        $x:expr,
        $a:expr, $b:expr, $x1:expr, $x2:expr, $x3:expr,
        $x_arr:expr,
        8, 6,
    ) => {
        asm!(
            $x,
            ax = inout(reg) $a,
            bx = inout(reg) $b,
            x1 = in(reg) $x1,
            x2 = in(reg) $x2,
            x3 = inout(reg) $x3,
            out("ymm0") _, out("ymm1") _, out("ymm2") _, out("ymm3") _,
            inout("ymm4") $x_arr[0],
            inout("ymm5") $x_arr[1], 
            inout("ymm6") $x_arr[2],
            inout("ymm7") $x_arr[3], 
            inout("ymm8") $x_arr[4],
            inout("ymm9") $x_arr[5],
            options(att_syntax)
        )
    };
    (
        $x:expr,
        $a:expr, $b:expr, $x1:expr, $x2:expr, $x3:expr,
        $x_arr:expr,
        8, 5,
    ) => {
        asm!(
            $x,
            ax = inout(reg) $a,
            bx = inout(reg) $b,
            x1 = in(reg) $x1,
            x2 = in(reg) $x2,
            x3 = inout(reg) $x3,
            out("ymm0") _, out("ymm1") _, out("ymm2") _, out("ymm3") _,
            inout("ymm4") $x_arr[0],
            inout("ymm5") $x_arr[1], 
            inout("ymm6") $x_arr[2],
            inout("ymm7") $x_arr[3], 
            inout("ymm8") $x_arr[4],
            options(att_syntax)
        )
    };
    (
        $x:expr,
        $a:expr, $b:expr, $x1:expr, $x2:expr, $x3:expr,
        $x_arr:expr,
        8, 4,
    ) => {
        asm!(
            $x,
            ax = inout(reg) $a,
            bx = inout(reg) $b,
            x1 = in(reg) $x1,
            x2 = in(reg) $x2,
            x3 = inout(reg) $x3,
            out("ymm0") _, out("ymm1") _, out("ymm2") _, out("ymm3") _,
            inout("ymm4") $x_arr[0],
            inout("ymm5") $x_arr[1], 
            inout("ymm6") $x_arr[2],
            inout("ymm7") $x_arr[3], 
            options(att_syntax)
        )
    };
    (
        $x:expr,
        $a:expr, $b:expr, $x1:expr, $x2:expr, $x3:expr,
        $x_arr:expr,
        8, 3,
    ) => {
        asm!(
            $x,
            ax = inout(reg) $a,
            bx = inout(reg) $b,
            x1 = in(reg) $x1,
            x2 = in(reg) $x2,
            out("ymm0") _, out("ymm1") _, out("ymm2") _, out("ymm3") _,
            inout("ymm4") $x_arr[0],
            inout("ymm5") $x_arr[1], 
            inout("ymm6") $x_arr[2],
            options(att_syntax)
        )
    };
    (
        $x:expr,
        $a:expr, $b:expr, $x1:expr, $x2:expr, $x3:expr,
        $x_arr:expr,
        8, 2,
    ) => {
        asm!(
            $x,
            ax = inout(reg) $a,
            bx = inout(reg) $b,
            x1 = in(reg) $x1,
            x2 = in(reg) $x2,
            out("ymm0") _, out("ymm1") _, out("ymm2") _, out("ymm3") _,
            inout("ymm4") $x_arr[0],
            inout("ymm5") $x_arr[1],
            options(att_syntax)
        )
    };
    (
        $x:expr,
        $a:expr, $b:expr, $x1:expr, $x2:expr, $x3:expr,
        $x_arr:expr,
        8, 1,
    ) => {
        asm!(
            $x,
            ax = inout(reg) $a,
            bx = inout(reg) $b,
            x1 = in(reg) $x1,
            x2 = in(reg) $x2,
            out("ymm0") _, out("ymm1") _, out("ymm2") _, out("ymm3") _,
            inout("ymm4") $x_arr[0], 
            options(att_syntax)
        )
    };
    (
        $x:expr,
        $a:expr, $b:expr, $x1:expr, $x2:expr, $x3:expr,
        $x_arr:expr,
        16, 6,
    ) => {
        asm!(
            $x,
            ax = inout(reg) $a,
            bx = inout(reg) $b,
            x1 = in(reg) $x1,
            x2 = in(reg) $x2,
            x3 = inout(reg) $x3,
            out("ymm0") _, out("ymm1") _, out("ymm2") _, out("ymm3") _,
            inout("ymm4") $x_arr[0], inout("ymm5") $x_arr[1], 
            inout("ymm6") $x_arr[2],  inout("ymm7") $x_arr[3], 
            inout("ymm8") $x_arr[4], inout("ymm9") $x_arr[5],
            inout("ymm10") $x_arr[6], inout("ymm11") $x_arr[7], 
            inout("ymm12") $x_arr[8], inout("ymm13") $x_arr[9], 
            inout("ymm14") $x_arr[10], inout("ymm15") $x_arr[11],
            options(att_syntax)
        )
    };
    (
        $x:expr,
        $a:expr, $b:expr, $x1:expr, $x2:expr, $x3:expr,
        $x_arr:expr,
        16, 5,
    ) => {
        asm!(
            $x,
            ax = inout(reg) $a,
            bx = inout(reg) $b,
            x1 = in(reg) $x1,
            x2 = in(reg) $x2,
            x3 = inout(reg) $x3,
            out("ymm0") _, out("ymm1") _, out("ymm2") _, out("ymm3") _,
            inout("ymm4") $x_arr[0], inout("ymm5") $x_arr[1], 
            inout("ymm6") $x_arr[2],  inout("ymm7") $x_arr[3], 
            inout("ymm8") $x_arr[4], inout("ymm9") $x_arr[5],
            inout("ymm10") $x_arr[6], inout("ymm11") $x_arr[7], 
            inout("ymm12") $x_arr[8], inout("ymm13") $x_arr[9],
            options(att_syntax)
        )
    };
    (
        $x:expr,
        $a:expr, $b:expr, $x1:expr, $x2:expr, $x3:expr,
        $x_arr:expr,
        16, 4,
    ) => {
        asm!(
            $x,
            ax = inout(reg) $a,
            bx = inout(reg) $b,
            x1 = in(reg) $x1,
            x2 = in(reg) $x2,
            x3 = inout(reg) $x3,
            out("ymm0") _, out("ymm1") _, out("ymm2") _, out("ymm3") _,
            inout("ymm4") $x_arr[0], inout("ymm5") $x_arr[1], 
            inout("ymm6") $x_arr[2],  inout("ymm7") $x_arr[3], 
            inout("ymm8") $x_arr[4], inout("ymm9") $x_arr[5],
            inout("ymm10") $x_arr[6], inout("ymm11") $x_arr[7], 
            options(att_syntax)
        )
    };
    (
        $x:expr,
        $a:expr, $b:expr, $x1:expr, $x2:expr, $x3:expr,
        $x_arr:expr,
        16, 3,
    ) => {
        asm!(
            $x,
            ax = inout(reg) $a,
            bx = inout(reg) $b,
            x1 = in(reg) $x1,
            x2 = in(reg) $x2,
            out("ymm0") _, out("ymm1") _, out("ymm2") _, out("ymm3") _,
            inout("ymm4") $x_arr[0], inout("ymm5") $x_arr[1], 
            inout("ymm6") $x_arr[2],  inout("ymm7") $x_arr[3], 
            inout("ymm8") $x_arr[4], inout("ymm9") $x_arr[5],
            options(att_syntax)
        )
    };
    (
        $x:expr,
        $a:expr, $b:expr, $x1:expr, $x2:expr, $x3:expr,
        $x_arr:expr,
        16, 2,
    ) => {
        asm!(
            $x,
            ax = inout(reg) $a,
            bx = inout(reg) $b,
            x1 = in(reg) $x1,
            x2 = in(reg) $x2,
            out("ymm0") _, out("ymm1") _, out("ymm2") _, out("ymm3") _,
            inout("ymm4") $x_arr[0], inout("ymm5") $x_arr[1], 
            inout("ymm6") $x_arr[2],  inout("ymm7") $x_arr[3], 
            options(att_syntax)
        )
    };
    (
        $x:expr,
        $a:expr, $b:expr, $x1:expr, $x2:expr, $x3:expr,
        $x_arr:expr,
        16, 1,
    ) => {
        asm!(
            $x,
            ax = inout(reg) $a,
            bx = inout(reg) $b,
            x1 = in(reg) $x1,
            x2 = in(reg) $x2,
            out("ymm0") _, out("ymm1") _, out("ymm2") _, out("ymm3") _,
            inout("ymm4") $x_arr[0], inout("ymm5") $x_arr[1], 
            options(att_syntax)
        )
    };
    (
        $x:expr,
        $a:expr, $b:expr, $x1:expr, $x2:expr, $x3:expr,
        $x_arr:expr,
        24, 4,
    ) => {
        asm!(
            $x,
            ax = inout(reg) $a,
            bx = inout(reg) $b,
            x1 = in(reg) $x1,
            x2 = in(reg) $x2,
            x3 = inout(reg) $x3,
            out("ymm0") _, out("ymm1") _, out("ymm2") _, out("ymm3") _,
            inout("ymm4") $x_arr[0], inout("ymm5") $x_arr[1], inout("ymm6") $x_arr[2],
            inout("ymm7") $x_arr[3], inout("ymm8") $x_arr[4], inout("ymm9") $x_arr[5],
            inout("ymm10") $x_arr[6], inout("ymm11") $x_arr[7], inout("ymm12") $x_arr[8],
            inout("ymm13") $x_arr[9], inout("ymm14") $x_arr[10], inout("ymm15") $x_arr[11],
            options(att_syntax)
        )
    };
    (
        $x:expr, 
        $a:expr, $b:expr, $x1:expr, $x2:expr, $x3:expr,
        $x_arr:expr,
        24, 3,
    ) => {
        asm!(
            $x,
            ax = inout(reg) $a,
            bx = inout(reg) $b,
            x1 = in(reg) $x1,
            x2 = in(reg) $x2,
            out("ymm0") _, out("ymm1") _, out("ymm2") _, out("ymm3") _,
            inout("ymm4") $x_arr[0], inout("ymm5") $x_arr[1], inout("ymm6") $x_arr[2],
            inout("ymm7") $x_arr[3], inout("ymm8") $x_arr[4], inout("ymm9") $x_arr[5],
            inout("ymm10") $x_arr[6], inout("ymm11") $x_arr[7], inout("ymm12") $x_arr[8],
            options(att_syntax)
        )
    };
    (
        $x:expr, 
        $a:expr, $b:expr, $x1:expr, $x2:expr, $x3:expr,
        $x_arr:expr,
        24, 2,
    ) => {
        asm!(
            $x,
            ax = inout(reg) $a,
            bx = inout(reg) $b,
            x1 = in(reg) $x1,
            x2 = in(reg) $x2,
            out("ymm0") _, out("ymm1") _, out("ymm2") _, out("ymm3") _,
            inout("ymm4") $x_arr[0], inout("ymm5") $x_arr[1], inout("ymm6") $x_arr[2],
            inout("ymm7") $x_arr[3], inout("ymm8") $x_arr[4], inout("ymm9") $x_arr[5],
            options(att_syntax)
        )
    };
    (
        $x:expr, 
        $a:expr, $b:expr, $x1:expr, $x2:expr, $x3:expr,
        $x_arr:expr,
        24, 1,
    ) => {
        asm!(
            $x,
            ax = inout(reg) $a,
            bx = inout(reg) $b,
            x1 = in(reg) $x1,
            x2 = in(reg) $x2,
            out("ymm0") _, out("ymm1") _, out("ymm2") _, out("ymm3") _,
            inout("ymm4") $x_arr[0], inout("ymm5") $x_arr[1], inout("ymm6") $x_arr[2],
            options(att_syntax)
        )
    };
}
// 12, 10, 9, 8, 6, 5 4, 3 2 1


macro_rules! def_x3 {
    ($x3:ident,$b:tt,$x2:tt,$b_layout:tt,1) => {
    };
    ($x3:ident,$b:tt,$x2:tt,$b_layout:tt,2) => {
    };
    ($x3:ident,$b:tt,$x2:tt,$b_layout:tt,3) => {
    };
    ($x3:ident,$b:tt,$x2:tt,$b_layout:tt,$nr:tt) => {
        let mut $x3 = $b.add($x2/4*3);
    }
}

macro_rules! prefetch_c {
    (24, $nr:tt, $c:tt, $ldc:tt) => {
        seq!(j in 0..$nr {
            _mm_prefetch($c.add(4+j*$ldc) as *mut i8, 3);
            _mm_prefetch($c.add(20+j*$ldc) as *mut i8, 3);
        });
    };
    (16, $nr:tt, $c:tt, $ldc:tt) => {
        seq!(j in 0..$nr {
            _mm_prefetch($c.add(16+j*$ldc) as *mut i8, 3);
        });
    };
    (8, $nr:tt, $c:tt, $ldc:tt) => {
        seq!(j in 0..$nr {
            _mm_prefetch($c.add(8+j*$ldc) as *mut i8, 3);
        });
    }
}

macro_rules! def_ukernel {
    (
        $mr:tt, $nr:tt,
        $mr_unit:tt,
        $a_layout:tt, $b_layout:tt,
        $al:tt, $bl:tt,
        $asm_step_macro:tt
    ) => {
        paste! {
            #[target_feature(enable = "avx,fma")]
            pub unsafe fn [<ukernel_$mr x$nr _$al $bl>](
                a: *const TA, b: *const TB, c: *mut TC,
                alpha: *const TA, beta: *const TB,
                k: usize,
                ldc: usize,
                ld_arr: [usize; 2]
            ) {
                let mut a = a;
                let mut b = b;
                let mut x_arr = [_mm256_setzero_ps(); $mr_unit*$nr];
                let x1 = ld_arr[0];
                let x2 = ld_arr[1];
                def_x3!(x3,b,x2,$b_layout,$nr);
                
                prefetch_c!($mr,$nr,c,ldc);

                let (k_iter, k_left) = (k / 4, k % 4);
                let mut i = 0;
                while i < k_iter {
                    asm_wrapped!(
                        asm_str!(
                            $mr, $nr, $a_layout, $b_layout, 
                            4,
                            0, 128,
                            $asm_step_macro
                        ),
                        a, b, x1, x2, x3,
                        x_arr,
                        $mr, $nr,
                    );
                    inc_a_k_unroll!(a, $a_layout, $mr, 4);
                    inc_b_k_unroll!(b, x3, $b_layout, $nr, 4);
                    i += 1;
                }
        
                i = 0;
                while i < k_left {
                    asm_wrapped!(
                        asm_str!(
                            $mr, $nr, $a_layout, $b_layout,
                            1,
                            0, 128,
                            $asm_step_macro
                        ),
                        a, b, x1, x2, x3,
                        x_arr,
                        $mr, $nr,
                    );
                    inc_a_k_unroll!(a, $a_layout, $mr, 1);
                    inc_b_k_unroll!(b, x3, $b_layout, $nr, 1);
                    i += 1;
                }
                let alpha_v = _mm256_broadcast_ss(&*alpha);
                seq!(j in 0..$nr {
                    seq!(i in 0..$mr_unit {
                        x_arr[i+j*$mr_unit] = _mm256_mul_ps(alpha_v, x_arr[i+j*$mr_unit]);
                    });
                });
                if *beta != 0.0 {
                    let beta_v = _mm256_broadcast_ss(&*beta);
                    seq!(j in 0..$nr {
                        seq!(i in 0..$mr_unit {
                            x_arr[i+j*$mr_unit] = _mm256_fmadd_ps(
                                _mm256_loadu_ps(c.add(i*8+j*ldc)),
                                beta_v,
                                x_arr[i+j*$mr_unit]
                            );
                        });
                    });
                }
                seq!(j in 0..$nr {
                    seq!(i in 0..$mr_unit {
                        _mm256_storeu_ps(c.add(i*8+j*ldc), x_arr[i+j*$mr_unit]);
                    });
                });
            }
        }
    }
}

def_ukernel!(24, 4, 3, B, B, b, b, asm_24x4_step);
def_ukernel!(24, 3, 3, B, B, b, b, asm_24x4_step);
def_ukernel!(24, 2, 3, B, B, b, b, asm_24x4_step);
def_ukernel!(24, 1, 3, B, B, b, b, asm_24x4_step);

def_ukernel!(24, 4, 3, B, C, b, c, asm_24x4_step);
def_ukernel!(24, 3, 3, B, C, b, c, asm_24x4_step);
def_ukernel!(24, 2, 3, B, C, b, c, asm_24x4_step);
def_ukernel!(24, 1, 3, B, C, b, c, asm_24x4_step);

def_ukernel!(24, 4, 3, B, R, b, r, asm_24x4_step);
def_ukernel!(24, 3, 3, B, R, b, r, asm_24x4_step);
def_ukernel!(24, 2, 3, B, R, b, r, asm_24x4_step);
def_ukernel!(24, 1, 3, B, R, b, r, asm_24x4_step);

def_ukernel!(16, 6, 2, B, B, b, b, asm_16x6_step);
def_ukernel!(16, 5, 2, B, B, b, b, asm_16x6_step);
def_ukernel!(16, 4, 2, B, B, b, b, asm_16x6_step);
def_ukernel!(16, 3, 2, B, B, b, b, asm_16x6_step);
def_ukernel!(16, 2, 2, B, B, b, b, asm_16x6_step);
def_ukernel!(16, 1, 2, B, B, b, b, asm_16x6_step);

// def_ukernel!(16, 6, 2, B, C, b, c, asm_16x6_step);
// def_ukernel!(16, 5, 2, B, C, b, c, asm_16x6_step);
def_ukernel!(16, 4, 2, B, C, b, c, asm_16x6_step);
def_ukernel!(16, 3, 2, B, C, b, c, asm_16x6_step);
def_ukernel!(16, 2, 2, B, C, b, c, asm_16x6_step);
def_ukernel!(16, 1, 2, B, C, b, c, asm_16x6_step);

// def_ukernel!(16, 6, 2, B, R, b, r, asm_16x6_step);
// def_ukernel!(16, 5, 2, B, R, b, r, asm_16x6_step);
def_ukernel!(16, 4, 2, B, R, b, r, asm_16x6_step);
def_ukernel!(16, 3, 2, B, R, b, r, asm_16x6_step);
def_ukernel!(16, 2, 2, B, R, b, r, asm_16x6_step);
def_ukernel!(16, 1, 2, B, R, b, r, asm_16x6_step);


def_ukernel!(8, 6, 1, B, B, b, b, asm_8x6_step);
def_ukernel!(8, 5, 1, B, B, b, b, asm_8x6_step);
def_ukernel!(8, 4, 1, B, B, b, b, asm_8x6_step);
def_ukernel!(8, 3, 1, B, B, b, b, asm_8x6_step);
def_ukernel!(8, 2, 1, B, B, b, b, asm_8x6_step);
def_ukernel!(8, 1, 1, B, B, b, b, asm_8x6_step);

// def_ukernel!(8, 6, 1, B, C, b, c, asm_8x6_step);
// def_ukernel!(8, 5, 1, B, C, b, c, asm_8x6_step);
def_ukernel!(8, 4, 1, B, C, b, c, asm_8x6_step);
def_ukernel!(8, 3, 1, B, C, b, c, asm_8x6_step);
def_ukernel!(8, 2, 1, B, C, b, c, asm_8x6_step);
def_ukernel!(8, 1, 1, B, C, b, c, asm_8x6_step);

// def_ukernel!(8, 6, 1, B, R, b, r, asm_8x6_step);
// def_ukernel!(8, 5, 1, B, R, b, r, asm_8x6_step);
def_ukernel!(8, 4, 1, B, R, b, r, asm_8x6_step);
def_ukernel!(8, 3, 1, B, R, b, r, asm_8x6_step);
def_ukernel!(8, 2, 1, B, R, b, r, asm_8x6_step);
def_ukernel!(8, 1, 1, B, R, b, r, asm_8x6_step);



macro_rules! def_ukernel_partial {
    (
        $mr:tt, $nr:tt,
        $mr_unit:tt,
        $a_layout:tt, $b_layout:tt,
        $al:tt, $bl:tt,
        $asm_step_macro:tt
    ) => {
        paste! {
            #[target_feature(enable = "avx,fma")]
            pub unsafe fn [<ukernel_$mr x$nr _$al $bl _partial>](
                a: *const TA, b: *const TB, c: *mut TC,
                alpha: *const TA, beta: *const TB,
                k: usize,
                ldc: usize,
                ld_arr: [usize; 2],
                mask: *const u32,
            ) {
                let mut a = a;
                let mut b = b;
                let mut x_arr = [_mm256_setzero_ps(); $mr_unit*$nr];
                let x1 = ld_arr[0];
                let x2 = ld_arr[1];
                def_x3!(x3,b,x2,$b_layout,$nr);
            
                prefetch_c!($mr,$nr,c,ldc);
        
                let (k_iter, k_left) = (k / 4, k % 4);
                let mut i = 0;
                while i < k_iter {
                    asm_wrapped!(
                        asm_str!(
                            $mr, $nr, $a_layout, $b_layout, 
                            4,
                            0, 128,
                            $asm_step_macro
                        ),
                        a, b, x1, x2, x3,
                        x_arr,
                        $mr, $nr,
                    );
                    inc_a_k_unroll!(a, $a_layout, $mr, 4);
                    inc_b_k_unroll!(b, x3, $b_layout, $nr, 4);
                    i += 1;
                }
        
                i = 0;
                while i < k_left {
                    asm_wrapped!(
                        asm_str!(
                            $mr, $nr, $a_layout, $b_layout,
                            1,
                            0, 128,
                            $asm_step_macro
                        ),
                        a, b, x1, x2, x3,
                        x_arr,
                        $mr, $nr,
                    );
                    inc_a_k_unroll!(a, $a_layout, $mr, 1);
                    inc_b_k_unroll!(b, x3, $b_layout, $nr, 1);
                    i += 1;
                }
                let alpha_v = _mm256_broadcast_ss(&*alpha);
                seq!(j in 0..$nr {
                    seq!(i in 0..$mr_unit {
                        x_arr[i+j*$mr_unit] = _mm256_mul_ps(alpha_v, x_arr[i+j*$mr_unit]);
                    });
                });
                let mask = _mm256_loadu_si256(mask as *const __m256i);
                if *beta != 0.0 {
                    let beta_v = _mm256_broadcast_ss(&*beta);
                    seq!(j in 0..$nr {
                        seq!(i in 1..$mr_unit {
                            x_arr[i-1+j*$mr_unit] = _mm256_fmadd_ps(
                                _mm256_loadu_ps(c.add((i-1)*8+j*ldc)),
                                beta_v,
                                x_arr[i-1+j*$mr_unit]
                            );
                        });
                        x_arr[$mr_unit-1+j*$mr_unit] = _mm256_fmadd_ps(
                            _mm256_maskload_ps(c.add(($mr_unit-1)*8+j*ldc), mask),
                            beta_v,
                            x_arr[$mr_unit-1+j*$mr_unit]
                        );
                    });
        
                }
                seq!(j in 0..$nr {
                    seq!(i in 1..$mr_unit {
                        _mm256_storeu_ps(c.add((i-1)*8+j*ldc), x_arr[i-1+j*$mr_unit]);
                    });
                    _mm256_maskstore_ps(c.add(($mr_unit-1)*8+j*ldc), mask, x_arr[$mr_unit-1+j*$mr_unit]);
                    
                });
            }
        }
    }
}

def_ukernel_partial!(24, 4, 3, B, B, b, b, asm_24x4_step);
def_ukernel_partial!(24, 3, 3, B, B, b, b, asm_24x4_step);
def_ukernel_partial!(24, 2, 3, B, B, b, b, asm_24x4_step);
def_ukernel_partial!(24, 1, 3, B, B, b, b, asm_24x4_step);

def_ukernel_partial!(16, 6, 2, B, B, b, b, asm_16x6_step);
def_ukernel_partial!(16, 5, 2, B, B, b, b, asm_16x6_step);
def_ukernel_partial!(16, 4, 2, B, B, b, b, asm_16x6_step);
def_ukernel_partial!(16, 3, 2, B, B, b, b, asm_16x6_step);
def_ukernel_partial!(16, 2, 2, B, B, b, b, asm_16x6_step);
def_ukernel_partial!(16, 1, 2, B, B, b, b, asm_16x6_step);

def_ukernel_partial!(8, 6, 1, B, B, b, b, asm_8x6_step);
def_ukernel_partial!(8, 5, 1, B, B, b, b, asm_8x6_step);
def_ukernel_partial!(8, 4, 1, B, B, b, b, asm_8x6_step);
def_ukernel_partial!(8, 3, 1, B, B, b, b, asm_8x6_step);
def_ukernel_partial!(8, 2, 1, B, B, b, b, asm_8x6_step);
def_ukernel_partial!(8, 1, 1, B, B, b, b, asm_8x6_step);


macro_rules! def_ukernel {
    (
        $mr:tt, $nr:tt,
        $mr_unit:tt,
        $a_layout:tt, $b_layout:tt,
        $al:tt,
        $asm_step_macro:tt
    ) => {
        paste! {
            #[target_feature(enable = "avx,fma")]
            pub unsafe fn [<ukernel_$mr x$nr _$al>](
                a: *const TA, b: *const TB, c: *mut TC,
                alpha: *const TA, beta: *const TB,
                k: usize,
                ldc: usize,
                ld_arr: [usize; 2]
            ) {
                let mut a = a;
                let mut b = b;
                let mut x_arr = [_mm256_setzero_ps(); $mr_unit*$nr];
                let x1 = ld_arr[0];
                let x2 = ld_arr[1];
                def_x3!(x3,b,x2,$b_layout,$nr);
            
                prefetch_c!($mr,$nr,c,ldc);
        
                let (k_iter, k_left) = (k / 4, k % 4);
                let mut i = 0;
                while i < k_iter {
                    asm_wrapped!(
                        asm_str!(
                            $mr, $nr, $a_layout, $b_layout, 
                            4,
                            0, 128,
                            $asm_step_macro
                        ),
                        a, b, x1, x2, x3,
                        x_arr,
                        $mr, $nr,
                    );
                    inc_a_k_unroll!(a, $a_layout, $mr, 4);
                    inc_b_k_unroll!(b, x3, $b_layout, $nr, 4);
                    i += 1;
                }
        
                i = 0;
                while i < k_left {
                    asm_wrapped!(
                        asm_str!(
                            $mr, $nr, $a_layout, $b_layout,
                            1,
                            0, 128,
                            $asm_step_macro
                        ),
                        a, b, x1, x2, x3,
                        x_arr,
                        $mr, $nr,
                    );
                    inc_a_k_unroll!(a, $a_layout, $mr, 1);
                    inc_b_k_unroll!(b, x3, $b_layout, $nr, 1);
                    i += 1;
                }
                let alpha_v = _mm256_broadcast_ss(&*alpha);
                seq!(j in 0..$nr {
                    seq!(i in 0..$mr_unit {
                        x_arr[i+j*$mr_unit] = _mm256_mul_ps(alpha_v, x_arr[i+j*$mr_unit]);
                    });
                });
                // tranpose
                seq!(i in 0..$mr_unit {
                    let t0 = _mm256_unpacklo_ps(x_arr[i], x_arr[$mr_unit+i]);
                    let t1 = _mm256_unpackhi_ps(x_arr[i], x_arr[$mr_unit+i]);
                    let t2 = _mm256_unpacklo_ps(x_arr[$mr_unit*2+i], x_arr[$mr_unit*3+i]);
                    let t3 = _mm256_unpackhi_ps(x_arr[$mr_unit*2+i], x_arr[$mr_unit*3+i]);
                    x_arr[i] = _mm256_shuffle_ps(t0, t2, 0b01000100);
                    x_arr[$mr_unit+i] = _mm256_shuffle_ps(t0, t2, 0b11101110);
                    x_arr[$mr_unit*2+i] = _mm256_shuffle_ps(t1, t3, 0b01000100);
                    x_arr[$mr_unit*3+i] = _mm256_shuffle_ps(t1, t3, 0b11101110);
                });
                
                if *beta != 0.0 {
                    let beta_v = _mm256_broadcast_ss(&*beta);
                    seq!(i in 0..$mr_unit {
                        seq!(j in 0..4 {
                            x_arr[j*$mr_unit+i] = _mm256_fmadd_ps(
                                _mm256_loadu2_m128(c.add((4+j+i*8)*ldc), c.add((j+i*8)*ldc)),
                                beta_v,
                                x_arr[j*$mr_unit+i]
                            );
                            _mm256_storeu2_m128(c.add((j+4+i*8)*ldc), c.add((j+i*8)*ldc), x_arr[j*$mr_unit+i]);
                        });
                    });
                    return;
                }
                // store tranposed
                seq!(i in 0..$mr_unit {
                    seq!(j in 0..4 {
                        _mm256_storeu2_m128(c.add((j+4+i*8)*ldc), c.add((j+i*8)*ldc), x_arr[j*$mr_unit+i]);
                    });
                });
            }
        }
    }
}

def_ukernel!(24, 4, 3, B, C, rb_t, asm_24x4_step);
def_ukernel!(24, 4, 3, B, R, cb_t, asm_24x4_step);


#[target_feature(enable = "avx,fma")]
unsafe fn scale_vec(x: &mut[__m256], alpha: *const f32) {
    let alpha_v = _mm256_broadcast_ss(&*alpha);
    // for i in 0..x.len() {
    //     x[i] = _mm256_mul_ps(alpha_v, x[i]);
    // }

    asm!(
        "vmulps %ymm4, {0}, %ymm4",
        "vmulps %ymm5, {0}, %ymm5",
        "vmulps %ymm6, {0}, %ymm6",
        "vmulps %ymm7, {0}, %ymm7",
        "vmulps %ymm8, {0}, %ymm8",
        "vmulps %ymm9, {0}, %ymm9",
        "vmulps %ymm10, {0}, %ymm10",
        "vmulps %ymm11, {0}, %ymm11",
        "vmulps %ymm12, {0}, %ymm12",
        "vmulps %ymm13, {0}, %ymm13",
        "vmulps %ymm14, {0}, %ymm14",
        "vmulps %ymm15, {0}, %ymm15",
        in(reg) alpha,
        inout("ymm4") x[0], inout("ymm5") x[1], inout("ymm6") x[2], inout("ymm7") x[3],
        inout("ymm8") x[4], inout("ymm9") x[5], inout("ymm10") x[6], inout("ymm11") x[7],
        inout("ymm12") x[8], inout("ymm13") x[9], inout("ymm14") x[10], inout("ymm15") x[11],
    );
}


macro_rules! def_store_kernel {
    ($mr:tt,4,$mr_unit:tt,$x_arr:tt,$c:tt,$beta:tt,$n_left:tt,$ldc:tt) => {
        let beta_v = _mm256_broadcast_ss(&*$beta);
        seq!(i in 1..$mr_unit {
            seq!(j in 0..4 {
                $x_arr[j*$mr_unit+i-1] = _mm256_fmadd_ps(
                    _mm256_loadu2_m128($c.add((4+j+i*8-8)*$ldc), $c.add((j+i*8-8)*$ldc)),
                    beta_v,
                    $x_arr[j*$mr_unit+i-1]
                );
                _mm256_storeu2_m128($c.add((j+4+i*8-8)*$ldc), $c.add((j+i*8-8)*$ldc), $x_arr[j*$mr_unit+i-1]);
            });
        });
        let beta_v = _mm256_castps256_ps128(beta_v);
        seq!(j in 0..4 {
            _mm_storeu_ps($c.add((($mr_unit-1)*8+j)*$ldc), _mm_fmadd_ps(
                _mm_loadu_ps($c.add((($mr_unit-1)*8+j)*$ldc)),
                beta_v,
                _mm256_castps256_ps128($x_arr[$mr_unit-1+j*$mr_unit])
            ));
            if $n_left == j+1 {return;}
        });
        seq!(j in 0..4 {
            _mm_storeu_ps($c.add((($mr_unit-1)*8+4+j)*$ldc), _mm_fmadd_ps(
                _mm_loadu_ps($c.add((($mr_unit-1)*8+4+j)*$ldc)),
                beta_v,
                _mm256_extractf128_ps($x_arr[$mr_unit-1+j*$mr_unit], 1)
            ));
            if $n_left == j+5 {return;}
        });
        return;
    };
    ($mr:tt,3,$mr_unit:tt,$x_arr:tt,$c:tt,$beta:tt,$n_left:tt,$ldc:tt) => {
        let beta_v = _mm_broadcast_ss(&*$beta);
        seq!(i in 0..$mr_unit {
            seq!(j in 0..4 {
                let mut x = [0_f32;8];
                std::ptr::copy_nonoverlapping($c.add((j+i*8)*$ldc), x.as_mut_ptr() as *mut f32, 3);
                _mm_storeu_ps(
                    x.as_mut_ptr() as *mut f32,
                    _mm_fmadd_ps(
                        _mm_loadu_ps(x.as_mut_ptr() as *mut f32),
                        beta_v,
                        _mm256_castps256_ps128($x_arr[j*$mr_unit+i])
                    )
                );
                std::ptr::copy_nonoverlapping(x.as_ptr() as *const f32, $c.add((j+i*8)*$ldc), 3);

                if $n_left == j+1 && i +1 == $mr_unit {return;}
            });
            seq!(j in 0..4 {
                let mut x = [0_f32;8];
                std::ptr::copy_nonoverlapping($c.add((j+i*8+4)*$ldc), x.as_mut_ptr() as *mut f32, 3);
                _mm_storeu_ps(
                    x.as_mut_ptr() as *mut f32,
                    _mm_fmadd_ps(
                        _mm_loadu_ps(x.as_mut_ptr() as *mut f32),
                        beta_v,
                        _mm256_extractf128_ps($x_arr[j*$mr_unit+i], 1)
                    )
                );
                std::ptr::copy_nonoverlapping(x.as_ptr() as *const f32, $c.add((j+i*8+4)*$ldc), 3);
                if $n_left == j+5 && i +1 == $mr_unit {return;}
            });
        });
        
        return;
    };
    ($mr:tt,2,$mr_unit:tt,$x_arr:tt,$c:tt,$beta:tt,$n_left:tt,$ldc:tt) => {
        let beta_v = _mm_broadcast_ss(&*$beta);
        seq!(i in 0..$mr_unit {
            seq!(j in 0..4 {
                _mm_store_sd($c.add((j+i*8)*$ldc) as *mut f64, _mm_castps_pd(_mm_fmadd_ps(
                    _mm_castpd_ps(_mm_load_sd($c.add((j+i*8)*$ldc) as *mut f64)),
                    beta_v,
                    _mm256_castps256_ps128($x_arr[j*$mr_unit+i])
                )));
                if $n_left == j+1 && i +1 == $mr_unit {return;}
            });
            seq!(j in 0..4 {
                _mm_store_sd($c.add((j+i*8+4)*$ldc) as *mut f64, _mm_castps_pd(_mm_fmadd_ps(
                    _mm_castpd_ps(_mm_load_sd($c.add((j+i*8+4)*$ldc) as *mut f64)),
                    beta_v,
                    _mm256_extractf128_ps($x_arr[j*$mr_unit+i], 1)
                )));
                if $n_left == j+5 && i +1 == $mr_unit {return;}
            });
        });
        
        return;
    };
    ($mr:tt,1,$mr_unit:tt,$x_arr:tt,$c:tt,$beta:tt,$n_left:tt,$ldc:tt) => {
        let beta_v = _mm_broadcast_ss(&*$beta);
        seq!(i in 0..$mr_unit {
            seq!(j in 0..4 {
                _mm_store_ss($c.add((j+i*8)*$ldc), _mm_fmadd_ss(
                    _mm_load_ss($c.add((j+i*8)*$ldc)),
                    beta_v,
                    _mm256_castps256_ps128($x_arr[j*$mr_unit+i])
                ));
                if $n_left == j+1 && i +1 == $mr_unit {return;}
            });
            seq!(j in 0..4 {
                _mm_store_ss($c.add((j+i*8+4)*$ldc), _mm_fmadd_ss(
                    _mm_load_ss($c.add((j+i*8+4)*$ldc)),
                    beta_v,
                    _mm256_extractf128_ps($x_arr[j*$mr_unit+i], 1)
                ));
                if $n_left == j+5 && i +1 == $mr_unit {return;}
            });
        });
        
        return;
    };
}

macro_rules! def_store2_kernel {
    ($mr:tt,4,$mr_unit:tt,$x_arr:tt,$c:tt,$beta:tt,$n_left:tt,$ldc:tt) => {
        // store tranposed
        seq!(i in 1..$mr_unit {
            seq!(j in 0..4 {
                _mm256_storeu2_m128($c.add((j+4+i*8-8)*$ldc), $c.add((j+i*8-8)*$ldc), $x_arr[j*$mr_unit+i-1]);
            });
        });
        seq!(j in 0..4 {
            _mm_storeu_ps($c.add((($mr_unit-1)*8+j)*$ldc), _mm256_castps256_ps128($x_arr[$mr_unit-1+j*$mr_unit]));
            if $n_left == j+1 {return;}
        });
        seq!(j in 0..4 {
            _mm_storeu_ps($c.add((($mr_unit-1)*8+4+j)*$ldc), _mm256_extractf128_ps($x_arr[$mr_unit-1+j*$mr_unit], 1));
            if $n_left == j+5 {return;}
        });
        return;
    };

    ($mr:tt,2,$mr_unit:tt,$x_arr:tt,$c:tt,$beta:tt,$n_left:tt,$ldc:tt) => {
        seq!(i in 0..$mr_unit {
            seq!(j in 0..4 {
                _mm_store_sd($c.add((j+i*8)*$ldc) as *mut f64, _mm_castps_pd(_mm256_castps256_ps128($x_arr[j*$mr_unit+i])));
                if $n_left == j+1 && i +1 == $mr_unit {return;}
            });
            seq!(j in 0..4 {
                _mm_store_sd($c.add((j+i*8+4)*$ldc) as *mut f64, _mm_castps_pd(_mm256_extractf128_ps($x_arr[j*$mr_unit+i], 1)));
                if $n_left == j+5 && i +1 == $mr_unit {return;}
            });
        });
        
        return;
    };

    ($mr:tt,3,$mr_unit:tt,$x_arr:tt,$c:tt,$beta:tt,$n_left:tt,$ldc:tt) => {
        seq!(i in 0..$mr_unit {
            seq!(j in 0..4 {
                let mut x = [0_f32;8];
                _mm256_storeu_ps(x.as_mut_ptr(), $x_arr[j*$mr_unit+i]);
                std::ptr::copy_nonoverlapping(x.as_ptr(), $c.add((j+i*8)*$ldc), 3);
                if $n_left == j+1 && i +1 == $mr_unit {return;}
            });
            seq!(j in 0..4 {
                let mut x = [0_f32;8];
                _mm_storeu_ps(x.as_mut_ptr(), _mm256_extractf128_ps($x_arr[j*$mr_unit+i], 1));
                std::ptr::copy_nonoverlapping(x.as_ptr(), $c.add((j+i*8+4)*$ldc), 3);
                if $n_left == j+5 && i +1 == $mr_unit {return;}
            });
        });
        
        return;
    };

    ($mr:tt,1,$mr_unit:tt,$x_arr:tt,$c:tt,$beta:tt,$n_left:tt,$ldc:tt) => {
        seq!(i in 0..$mr_unit {
            seq!(j in 0..4 {
                _mm_store_ss($c.add((j+i*8)*$ldc), _mm256_castps256_ps128($x_arr[j*$mr_unit+i]));
                if $n_left == j+1 && i +1 == $mr_unit {return;}
            });
            seq!(j in 0..4 {
                _mm_store_ss($c.add((j+i*8+4)*$ldc), _mm256_extractf128_ps($x_arr[j*$mr_unit+i], 1));
                if $n_left == j+5 && i +1 == $mr_unit {return;}
            });
        });
        
        return;
    }
}

macro_rules! def_ukernel {
    (
        $mr:tt, $nr:tt,
        $mr_unit:tt,
        $a_layout:tt, $b_layout:tt,
        $al:tt,
        $asm_step_macro:tt
    ) => {
        paste! {
            #[target_feature(enable = "avx,fma")]
            pub unsafe fn [<ukernel_$mr x$nr _$al>](
                a: *const TA, b: *const TB, c: *mut TC,
                alpha: *const TA, beta: *const TB,
                k: usize,
                ldc: usize,
                ld_arr: [usize; 2],
                n_left: usize,
            ) {
                let mut a = a;
                let mut b = b;
                let mut x_arr = [_mm256_setzero_ps(); 4*3];
                let x1 = ld_arr[0];
                let x2 = ld_arr[1];
                def_x3!(x3,b,x2,$b_layout,$nr);
            
                prefetch_c!($mr,$nr,c,ldc);
        
                let (k_iter, k_left) = (k / 4, k % 4);
                let mut i = 0;
                while i < k_iter {
                    asm_wrapped!(
                        asm_str!(
                            $mr, $nr, $a_layout, $b_layout, 
                            4,
                            0, 128,
                            $asm_step_macro
                        ),
                        a, b, x1, x2, x3,
                        x_arr,
                        $mr, $nr,
                    );
                    inc_a_k_unroll!(a, $a_layout, $mr, 4);
                    inc_b_k_unroll!(b, x3, $b_layout, $nr, 4);
                    i += 1;
                }
        
                i = 0;
                while i < k_left {
                    asm_wrapped!(
                        asm_str!(
                            $mr, $nr, $a_layout, $b_layout,
                            1,
                            0, 128,
                            $asm_step_macro
                        ),
                        a, b, x1, x2, x3,
                        x_arr,
                        $mr, $nr,
                    );
                    inc_a_k_unroll!(a, $a_layout, $mr, 1);
                    inc_b_k_unroll!(b, x3, $b_layout, $nr, 1);
                    i += 1;
                }
                let alpha_v = _mm256_broadcast_ss(&*alpha);
                seq!(j in 0..$nr {
                    seq!(i in 0..$mr_unit {
                        x_arr[i+j*$mr_unit] = _mm256_mul_ps(alpha_v, x_arr[i+j*$mr_unit]);
                    });
                });
                // tranpose
                seq!(i in 0..$mr_unit {
                    let t0 = _mm256_unpacklo_ps(x_arr[i], x_arr[$mr_unit+i]);
                    let t1 = _mm256_unpackhi_ps(x_arr[i], x_arr[$mr_unit+i]);
                    let t2 = _mm256_unpacklo_ps(x_arr[$mr_unit*2+i], x_arr[$mr_unit*3+i]);
                    let t3 = _mm256_unpackhi_ps(x_arr[$mr_unit*2+i], x_arr[$mr_unit*3+i]);
                    x_arr[i] = _mm256_shuffle_ps(t0, t2, 0b01000100);
                    x_arr[$mr_unit+i] = _mm256_shuffle_ps(t0, t2, 0b11101110);
                    x_arr[$mr_unit*2+i] = _mm256_shuffle_ps(t1, t3, 0b01000100);
                    x_arr[$mr_unit*3+i] = _mm256_shuffle_ps(t1, t3, 0b11101110);
                });
                
                if *beta != 0.0 {
                    def_store_kernel!($mr,$nr,$mr_unit,x_arr,c,beta,n_left,ldc);
                }
                def_store2_kernel!($mr,$nr,$mr_unit,x_arr,c,beta,n_left,ldc);
            }
        }
    }
}

def_ukernel!(24, 4, 3, B, C, rb_t_partial, asm_24x4_step);
def_ukernel!(24, 4, 3, B, R, cb_t_partial, asm_24x4_step);

def_ukernel!(24, 1, 3, B, C, rb_t_partial, asm_24x4_step);
def_ukernel!(24, 1, 3, B, R, cb_t_partial, asm_24x4_step);

def_ukernel!(24, 2, 3, B, C, rb_t_partial, asm_24x4_step);
def_ukernel!(24, 2, 3, B, R, cb_t_partial, asm_24x4_step);

def_ukernel!(24, 3, 3, B, C, rb_t_partial, asm_24x4_step);
def_ukernel!(24, 3, 3, B, R, cb_t_partial, asm_24x4_step);


def_ukernel!(16, 4, 2, B, C, rb_t_partial, asm_16x6_step);
def_ukernel!(16, 4, 2, B, R, cb_t_partial, asm_16x6_step);

def_ukernel!(16, 3, 2, B, C, rb_t_partial, asm_16x6_step);
def_ukernel!(16, 3, 2, B, R, cb_t_partial, asm_16x6_step);

def_ukernel!(16, 2, 2, B, C, rb_t_partial, asm_16x6_step);
def_ukernel!(16, 2, 2, B, R, cb_t_partial, asm_16x6_step);

def_ukernel!(16, 1, 2, B, C, rb_t_partial, asm_16x6_step);
def_ukernel!(16, 1, 2, B, R, cb_t_partial, asm_16x6_step);


def_ukernel!(8, 4, 1, B, C, rb_t_partial, asm_8x6_step);
def_ukernel!(8, 4, 1, B, R, cb_t_partial, asm_8x6_step);

def_ukernel!(8, 3, 1, B, C, rb_t_partial, asm_8x6_step);
def_ukernel!(8, 3, 1, B, R, cb_t_partial, asm_8x6_step);

def_ukernel!(8, 2, 1, B, C, rb_t_partial, asm_8x6_step);
def_ukernel!(8, 2, 1, B, R, cb_t_partial, asm_8x6_step);

def_ukernel!(8, 1, 1, B, C, rb_t_partial, asm_8x6_step);
def_ukernel!(8, 1, 1, B, R, cb_t_partial, asm_8x6_step);
