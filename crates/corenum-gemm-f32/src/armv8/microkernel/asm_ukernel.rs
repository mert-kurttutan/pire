use seq_macro::seq;
use std::arch::asm;


use crate::{TA, TB, TC};

use paste::paste;
macro_rules! beta_fmaddps {
    (C, $m0:expr, $r1:expr) => {
        concat!(
			"ldr q1, ", $m0, "\n",
			"fmla v", $r1, ".4s, v1.4s, v0.s[0]\n",
        ) 
    };
    (M, $m0:expr, $r1:expr) => {
        concat!(
            // "vmaskmovps ", $m0, ", %ymm1", ", %ymm2",  "\n",
            // "vfmadd231ps %ymm2, %ymm0,%ymm", $r1, "\n",
        ) 
    };
 }

macro_rules! mem {
    ($m0:tt, $b0:tt, $b1:tt) => {
		concat!("[", $m0, ", #", $b0, "+", $b1, "]")
	};
	($m0:tt, $b0:tt) => {
		concat!("[", $m0, ", #", $b0, "]")
	};
	($m0:tt) => {
		concat!("[", $m0, "]")
	};
 }
macro_rules! vmovps {
    (B) => {
        ""
    };
    ($layout:tt) => {
        " "
    };
 }
 macro_rules! acc_ps {
    (24, $layout:tt, $m0:expr, $r1:expr, $r2:expr, $r3:expr, $r4:expr, $r5:expr, $r6:expr) => {
        concat!(
            beta_fmaddps!(C, mem!($m0), $r1),
            beta_fmaddps!(C, mem!($m0, "0x10"), $r2),
            beta_fmaddps!(C, mem!($m0, "0x20"), $r3),
			beta_fmaddps!(C, mem!($m0, "0x30"), $r4),
			beta_fmaddps!(C, mem!($m0, "0x40"), $r5),
			beta_fmaddps!(C, mem!($m0, "0x50"), $r6),
        )
    };
    (16, $layout:tt, $m0:expr, $r1:expr, $r2:expr, $r3:expr, $r4:expr) => {
        concat!(
            beta_fmaddps!(C, mem!($m0), $r1),
            beta_fmaddps!(C, mem!($m0, "0x10"), $r2),
            beta_fmaddps!(C, mem!($m0, "0x20"), $r3),
			beta_fmaddps!(C, mem!($m0, "0x30"), $r4),
        )
    };
    ($N:tt, $layout:tt, $m0:expr, $r1:expr, $r2:expr) => {
        concat!(
            beta_fmaddps!(C, mem!($m0), $r1),
            beta_fmaddps!(C, mem!($m0, "0x10"), $r2),
        )
    };
 }

macro_rules! vzeroall {
	($r0:tt, $r1:tt) => {
		seq!(r in $r0..=$r1 {
			concat!(
				#(
					"dup v", r, ".4s, wzr \n",
				)*
			)
		})
	}
}

macro_rules! loadps_unit {
    ($layout:tt, $m0:expr, $r1:expr) => {
        concat!(
			"ldr q", $r1, ", ", $m0, "\n",
        )
    };
 }
 macro_rules! loadps {
    (24, $layout:tt, $m0:expr, $b0:expr) => {
        concat!(
            loadps_unit!($layout, mem!($m0, $b0), 0),
            loadps_unit!($layout, mem!($m0, $b0, "0x10"), 1),
            loadps_unit!($layout, mem!($m0, $b0, "0x20"), 2),
			loadps_unit!($layout, mem!($m0, $b0, "0x30"), 3),
			loadps_unit!($layout, mem!($m0, $b0, "0x40"), 4),
			loadps_unit!($layout, mem!($m0, $b0, "0x50"), 5),
        )
    };
    (16, $layout:tt, $m0:expr, $b0:expr) => {
        concat!(
            loadps_unit!($layout, mem!($m0, $b0), 0),
            loadps_unit!($layout, mem!($m0, $b0, "0x10"), 1),
            loadps_unit!($layout, mem!($m0, $b0, "0x20"), 2),
			loadps_unit!($layout, mem!($m0, $b0, "0x30"), 3),
        )
    };
    (8, $layout:tt, $m0:expr, $b0:expr) => {
        concat!(
            loadps_unit!($layout, mem!($m0, $b0), 0),
            loadps_unit!($layout, mem!($m0, $b0, "0x10"), 1),
        )
    };
 }

 macro_rules! storeps_unit {
    (C, $r1:expr, $m0:expr) => {
        concat!(
			"str q", $r1, ", ", $m0,  "\n",
        )
    };
    (B, $r1:expr, $m0:expr) => {
        concat!(
			"str q", $r1, ", ", $m0,  "\n",
		)
    };
    (M, $r1:expr, $m0:expr) => {
        concat!(
            // "vmaskmovps %ymm", $r1, ", %ymm1, ", $m0,  "\n",
        )
    };
    (4, M, $r1:expr, $m0:expr) => {
        concat!(
            // "vmaskmovps %xmm", $r1, ", %xmm1, ", $m0,  "\n",
        )
    };
    (4, $layout:tt, $r1:expr, $m0:expr) => {
        concat!(
			"str q", $r1, ", ", $m0,  "\n",
		)
    };
}
macro_rules! storeps {
	(24, $layout:tt, $m0:expr, $r1:expr, $r2:expr, $r3:expr, $r4:expr, $r5:expr, $r6:expr) => {
		concat!(
			storeps_unit!(C, $r1, mem!($m0)),
			storeps_unit!(C, $r2, mem!($m0, "0x10")),
			storeps_unit!(C, $r3, mem!($m0, "0x20")),
			storeps_unit!(C, $r4, mem!($m0, "0x30")),
			storeps_unit!(C, $r5, mem!($m0, "0x40")),
			storeps_unit!(C, $r6, mem!($m0, "0x50")),
		)
	};
	(16, $layout:tt, $m0:expr, $r1:expr, $r2:expr, $r3:expr, $r4:expr) => {
		concat!(
			storeps_unit!(C, $r1, mem!($m0)),
			storeps_unit!(C, $r2, mem!($m0, "0x10")),
			storeps_unit!(C, $r3, mem!($m0, "0x20")),
			storeps_unit!(C, $r4, mem!($m0, "0x30")),
		)
	};
	(8, $layout:tt, $m0:expr, $r1:expr, $r2:expr) => {
		concat!(
			storeps_unit!(C, $r1, mem!($m0)),
			storeps_unit!(C, $r2, mem!($m0, "0x10")),
		)
	};
 }

macro_rules! vfmadd231ps {
    ($r1:expr, $r2:expr, $r3:expr, $s0:expr) => {
        concat!(
            "fmla v", $r3, ".4s", ", v", $r1,".4s, v", $r2, ".s[", $s0, "]\n",
        ) 
    };
}

// only non contigous along m and n direction which is not changed frequently during iteration along k direction
/*

x1 -> cs_a
x2 -> cs_b
x3 -> ax + 3*cs_a
x4 -> bx + 3*cs_b

*/


macro_rules! asm_init_ab {
	($KER:tt,B,B) => {
    	concat!(
			"/* {x3} */", "\n",
			"/* {x2} */", "\n",
			"/* {x1} */", "\n",
			"/* {x0} */", "\n",
			"ldr {x0}, [{int_arrx}, #24]", "\n",
        	"cmp {x0}, #0",
    	)
	};
	($ker:tt,B,S) => {
    	concat!(
        	// mov cs_b to reg
			"/* {x3} */", "\n",
			"/* {x2} */", "\n",
			"/* {x1} */", "\n",
			"/* {x0} */", "\n",

			"ldr {x1}, [{int_arrx}]", "\n",
			"ldr {x2}, [{int_arrx}, #8]", "\n",

			// "add {x3}, {x2}, {x2}, lsl #1 \n",
			// "lsl {x3}, {x2}, #1 \n",
			"add {x3}, {bx}, {x2} \n",
			"add {x4}, {x3}, {x2} \n",
			"add {x5}, {x4}, {x2} \n",
			"ldr {x0}, [{int_arrx}, #24]", "\n",
        	"cmp {x0}, #0",
    	)
	};
}


macro_rules! asm_c_load {
	(6) => {
    	concat!(
			"ldr {x0}, [{int_arrx}, #16]\n",
			"lsl {x0}, {x0}, #2\n",
        	"add {x1}, {cx}, {x0} \n",
			"add {x2}, {x1}, {x0} \n",
        	"add {x3}, {x2}, {x0} \n",
    	)
	};
	(5) => {
    	concat!(
			"ldr {x0}, [{int_arrx}, #16]\n",
			"lsl {x0}, {x0}, #2\n",
        	"add {x1}, {cx}, {x0} \n",
			"add {x2}, {x1}, {x0} \n",
        	"add {x3}, {x2}, {x0} \n",
    	)
	};
	(4) => {
    	concat!(
			"ldr {x0}, [{int_arrx}, #16]\n",
			"lsl {x0}, {x0}, #2\n",
        	"add {x1}, {cx}, {x0} \n",
			"add {x2}, {x1}, {x0} \n",
        	"add {x3}, {x2}, {x0} \n",
    	)
	};
	(3) => {
    	concat!(
			"ldr {x0}, [{int_arrx}, #16]\n",
			"lsl {x0}, {x0}, #2\n",
        	"add {x1}, {cx}, {x0} \n",
			"add {x2}, {x1}, {x0} \n",
    	)
	};
	(2) => {
    	concat!(
			"ldr {x0}, [{int_arrx}, #16]\n",
			"lsl {x0}, {x0}, #2\n",
        	"add {x1}, {cx}, {x0} \n",
    	)
	};
	(1) => {
    	concat!(
    	)
	};
}


macro_rules! asm_vzeroall {

	(VER24,4) => {vzeroall!(8,31)};
	(VER24,3) => {vzeroall!(8,25)};
	(VER24,2) => {vzeroall!(8,19)};
	(VER24,1) => {vzeroall!(8,13)};

	(VER16,6) => {vzeroall!(8,31)};
	(VER16,5) => {vzeroall!(8,27)};
	(VER16,4) => {vzeroall!(8,23)};
	(VER16,3) => {vzeroall!(8,19)};
	(VER16,2) => {vzeroall!(8,15)};
	(VER16,1) => {vzeroall!(8,11)};

	(VER8,6) => {vzeroall!(20,31)};
	(VER8,5) => {vzeroall!(20,29)};
	(VER8,4) => {vzeroall!(20,27)};
	(VER8,3) => {vzeroall!(20,25)};
	(VER8,2) => {vzeroall!(20,23)};
	(VER8,1) => {vzeroall!(20,21)};
}

macro_rules! inc_a {
	(C) => {
    	"add {ax}, {ax}, {x1} \n"
	};
	(B) => {
    	""
	};
}

macro_rules! inc_b {
	(S,6) => {
    	"add {bx},{bx},{x1} \n add {x3},{x3},{x1} \n"
	};
	(S,5) => {
    	"add {bx},{bx},{x1} \n add {x3},{x3},{x1} \n"
	};
	(S,4) => {
    	"add {bx},{bx},{x1} \n add {x3},{x3},{x1} \n add {x4},{x4},{x1} \n add {x5},{x5},{x1} \n"
	};
	(S,3) => {
    	"add {bx},{bx},{x1} \n add {x3},{x3},{x1} \n add {x4},{x4},{x1} \n"
	};
	(S,2) => {
    	"add {bx},{bx},{x1} \n add {x3},{x3},{x1} \n"
	};
	(S,1) => {
    	"add {bx},{bx},{x1} \n"
	};
	(B,$nr:tt) => {
    	""
	};
}

macro_rules! inc_a_k_unroll {
	(C, $X:tt, $K:tt) => {
    	""
	};
	(B, $X:tt, $K:tt) => {
    	concat!(
        	"add {ax}, {ax}, #4*", $K, "*", $X, " \n",
    	)
	};
}

macro_rules! inc_b_k_unroll {
	(S, $X:tt, $K:tt) => {
    	""
	};
	(B, $X:tt, $K:tt) => {
    	concat!(
        	"add {bx}, {bx}, #4*", $K, "*", $X, " \n",
    	)
	};
}

macro_rules! asm_alpha_scale_0 {
	($r0:tt, $r1:tt) => {
		seq!(r in $r0..=$r1 {
			concat!(
				"/* {cx} */", "\n",
				"ldr s1, [{alphax}]", "\n",
				#(
					"fmul  v", r, ".4s, v", r, ".4s, v1.s[0]\n",
				)*
			)
		})
	}
}
macro_rules! asm_alpha_scale {
	(VER16, 6) => {
		asm_alpha_scale_0!(4,15)
	};
	(VER16, 5) => {
    	asm_alpha_scale_0!(4,13)
	};
	(VER16, 4) => {
    	asm_alpha_scale_0!(4,11)
	};
	(VER16, 3) => {
    	asm_alpha_scale_0!(4,9)
	};
	(VER16, 2) => {
    	asm_alpha_scale_0!(4,7)
	};
	(VER16, 1) => {
    	asm_alpha_scale_0!(4,5)
	};

	(VER24, 4) => {
    	asm_alpha_scale_0!(4,15)
	};
	(VER24, 3) => {
    	asm_alpha_scale_0!(4,12)
	};
	(VER24, 2) => {
    	asm_alpha_scale_0!(4,9)
	};
	(VER24, 1) => {
    	asm_alpha_scale_0!(4,6)
	};

	(VER8, 6) => {
    	asm_alpha_scale_0!(7,12)
	};
	(VER8, 5) => {
		asm_alpha_scale_0!(7,11)
	};
	(VER8, 4) => {
    	asm_alpha_scale_0!(7,10)
	};
	(VER8, 3) => {
    	asm_alpha_scale_0!(7,9)
	};
	(VER8, 2) => {
    	asm_alpha_scale_0!(7,8)
	};
	(VER8, 1) => {
    	asm_alpha_scale_0!(7,7)
	};
}


macro_rules! asm_24x4_acc_seq {
	($mr:tt, 0, $layout:tt) => {
		acc_ps!($mr, $layout, "{cx}", 8, 9, 10, 11, 12, 13)
	};
	($mr:tt, 1, $layout:tt) => {
		acc_ps!($mr, $layout, "{x1}", 14, 15, 16, 17, 18, 19)
	};
	($mr:tt, 2, $layout:tt) => {
		acc_ps!($mr, $layout, "{x2}", 20, 21, 22, 23, 24, 25)
	}; 
	($mr:tt, 3, $layout:tt) => {
		acc_ps!($mr, $layout, "{x3}", 26, 27, 28, 29, 30, 31)
	};
}

macro_rules! asm_24x4_store_seq {
	($mr:tt, 0, $layout:tt) => {
		storeps!($mr, $layout, "{cx}", 8, 9, 10, 11, 12, 13)
	};
	($mr:tt, 1, $layout:tt) => {
		storeps!($mr, $layout, "{x1}", 14, 15, 16, 17, 18, 19)
	};
	($mr:tt, 2, $layout:tt) => {
		storeps!($mr, $layout, "{x2}", 20, 21, 22, 23, 24, 25)
	}; 
	($mr:tt, 3, $layout:tt) => {
		storeps!($mr, $layout, "{x3}", 26, 27, 28, 29, 30, 31)
	};
}

macro_rules! asm_16x6_acc_seq {
	($mr:tt, 0, $layout:tt) => {
		acc_ps!($mr, $layout, "{cx}", 8, 9, 10, 11)
	};
	($mr:tt, 1, $layout:tt) => {
		acc_ps!($mr, $layout, "{x1}", 12, 13, 14, 15)
	};
	($mr:tt, 2, $layout:tt) => {
		acc_ps!($mr, $layout, "{x2}", 16, 17, 18, 19)
	}; 
	($mr:tt, 3, $layout:tt) => {
		acc_ps!($mr, $layout, "{x3}", 20, 21, 22, 23)
	};
	($mr:tt, 4, $layout:tt) => {
        acc_ps!($mr, $layout, "{x3}", 24, 25, 26, 27)
	};
	($mr:tt, 5, $layout:tt) => {
		acc_ps!($mr, $layout, "{x3}", 28, 29, 30, 31)
	};
}

macro_rules! asm_16x6_store_seq {
	($mr:tt, 0, $layout:tt) => {
		storeps!($mr, $layout, "{cx}", 8, 9, 10, 11)
	};
	($mr:tt, 1, $layout:tt) => {
		storeps!($mr, $layout, "{x1}", 12, 13, 14, 15)
	};
	($mr:tt, 2, $layout:tt) => {
		storeps!($mr, $layout, "{x2}", 16, 17, 18, 19)
	}; 
	($mr:tt, 3, $layout:tt) => {
		storeps!($mr, $layout, "{x3}", 20, 21, 22, 23)
	};
	($mr:tt, 4, $layout:tt) => {
    	storeps!($mr, $layout, "{x3}", 24, 25, 26, 27)
	};
	($mr:tt, 5, $layout:tt) => {
		storeps!($mr, $layout, "{x3}", 28, 29, 30, 31)
	};
}

macro_rules! asm_8x6_acc_seq {
	($mr:tt, 0, $layout:tt) => {
		acc_ps!($mr, $layout, "{cx}", 20, 21)
	};
	($mr:tt, 1, $layout:tt) => {
		acc_ps!($mr, $layout, "{x1}", 22, 23)
	};
	($mr:tt, 2, $layout:tt) => {
		acc_ps!($mr, $layout, "{x2}", 24, 25)
	}; 
	($mr:tt, 3, $layout:tt) => {
		acc_ps!($mr, $layout, "{x3}", 26, 27)
	};
	($mr:tt, 4, $layout:tt) => {
    	acc_ps!($mr, $layout, "{x3}", 28, 29)
	};
	($mr:tt, 5, $layout:tt) => {
		acc_ps!($mr, $layout, "{x3}", 30, 31)
	};
}

macro_rules! asm_8x6_store_seq {
	($mr:tt, 0, $layout:tt) => {
		storeps!($mr, $layout, "{cx}", 20, 21)
	};
	($mr:tt, 1, $layout:tt) => {
		storeps!($mr, $layout, "{x1}", 22, 23)
	};
	($mr:tt, 2, $layout:tt) => {
		storeps!($mr, $layout, "{x2}", 24, 25)
	}; 
	($mr:tt, 3, $layout:tt) => {
		storeps!($mr, $layout, "{x3}", 26, 27)
	};
	($mr:tt, 4, $layout:tt) => {
    	storeps!($mr, $layout, "{x3}", 28, 29)
	};
	($mr:tt, 5, $layout:tt) => {
		storeps!($mr, $layout, "{x3}", 30, 31)
	};
}

macro_rules! asm_24x4_acc {
	($mr:tt, $nr:tt, $layout:tt) => {
		seq!(n in 0..$nr {
			concat!(
				#(
					asm_24x4_acc_seq!($mr, n, $layout), 
				)*
			)
		})
	};
}


macro_rules! asm_24x4_store {
	($mr:tt, $nr:tt, $layout:tt) => {
		seq!(n in 0..$nr {
			concat!(
				#(
					asm_24x4_store_seq!($mr, n, $layout), 
				)*
			)
		})
	};
}


macro_rules! asm_16x6_acc {
	($mr:tt, $nr:tt, $layout:tt) => {
		seq!(n in 0..$nr {
			concat!(
				#(
					asm_16x6_acc_seq!($mr, n, $layout), 
				)*
			)
		})
	};
}

macro_rules! asm_16x6_store {
	($mr:tt, $nr:tt, $layout:tt) => {
		seq!(n in 0..$nr {
			concat!(
				#(
					asm_16x6_store_seq!($mr, n, $layout), 
				)*
			)
		})
	};
}


macro_rules! asm_8x6_acc {
	($mr:tt, $nr:tt, $layout:tt) => {
		seq!(n in 0..$nr {
			concat!(
				#(
					asm_8x6_acc_seq!($mr, n, $layout), 
				)*
			)
		})
	};
}


macro_rules! asm_8x6_store {
	($mr:tt, $nr:tt, $layout:tt) => {
		seq!(n in 0..$nr {
			concat!(
				#(
					asm_8x6_store_seq!($mr, n, $layout), 
				)*
			)
		})
	};
}

macro_rules! load_b {
	(S, 0, $K:tt, $X:tt, $r:expr) => {
    	concat!(
			"ld1 {{v", $r, ".s}}[0], [{bx}] \n",
			// "ldr s", $r, ", [{bx}]", "\n",	
    	)
	};
	(S, 1, $K:tt, $X:tt, $r:expr) => {
    	concat!(
			// "ldr s", $r, ", [{bx},{x2}]", "\n",	
			"ld1 {{v", $r, ".s}}[1], [{x3}] \n",
    	)
	};
	(S, 2, $K:tt, $X:tt, $r:expr) => {
    	concat!(
			// "ldr s", $r, ", [{x3}]", "\n",	
			"ld1 {{v", $r, ".s}}[2], [{x4}] \n",
    	)
	};
	(S, 3, $K:tt, $X:tt, $r:expr) => {
    	concat!(
			// "ldr s", $r, ", [{x3},{x2}]", "\n",	
			"ld1 {{v", $r, ".s}}[3], [{x5}] \n",
    	)
	};
	(S, 4, $K:tt, $X:tt, $r:expr) => {
    	concat!(
			"ldr s", $r, ", [{x3},{x2}]", "\n",	
    	)
	};
	(S, 5, $K:tt, $X:tt, $r:expr) => {
    	concat!(
			"ldr s", $r, ", [{x3},{x2}, LSL #2]", "\n",
    	)
	};
	(B, 0, $K:tt, $X:tt, $r:expr) => {
    	concat!(
			"ldr q", $r, ", [{bx}, #", $K, "*", $X, "*4]", "\n",
    	)
	};
	(B, $N:tt, $K:tt, $X:tt, $r:expr) => {
    	""
	};
}


macro_rules! load_a {
	($mr:tt, B, $K:tt) => {
    	loadps!($mr, B, "{ax}", concat!($mr,"*4*",$K))
	};
	($mr:tt, C, $K:tt) => {
    	loadps!($mr, C, "[{ax}]")
	};
}

macro_rules! fmadd_3v {
	(0) => {
		concat!(
			vfmadd231ps!(0, 6, 8, 0),
			vfmadd231ps!(1, 6, 9, 0),
			vfmadd231ps!(2, 6, 10, 0),
		    vfmadd231ps!(3, 6, 11, 0),
			vfmadd231ps!(4, 6, 12, 0),
			vfmadd231ps!(5, 6, 13, 0),
		)
	};
	(1) => {
		concat!(
			vfmadd231ps!(0, 6, 14, 1),
			vfmadd231ps!(1, 6, 15, 1),
			vfmadd231ps!(2, 6, 16, 1),
		    vfmadd231ps!(3, 6, 17, 1),
			vfmadd231ps!(4, 6, 18, 1),
			vfmadd231ps!(5, 6, 19, 1),
		)
	};
	(2) => {
		concat!(
			vfmadd231ps!(0, 6, 20, 2),
			vfmadd231ps!(1, 6, 21, 2),
			vfmadd231ps!(2, 6, 22, 2),
		    vfmadd231ps!(3, 6, 23, 2),
			vfmadd231ps!(4, 6, 24, 2),
			vfmadd231ps!(5, 6, 25, 2),
		)
	};
	(3) => {
		concat!(
			vfmadd231ps!(0, 6, 26, 3),
			vfmadd231ps!(1, 6, 27, 3),
			vfmadd231ps!(2, 6, 28, 3),
		    vfmadd231ps!(3, 6, 29, 3),
			vfmadd231ps!(4, 6, 30, 3),
			vfmadd231ps!(5, 6, 31, 3),
		)
	};
}

macro_rules! fmadd_2v {
	(0) => {
		concat!(
			vfmadd231ps!(0, 4, 8, 0),
			vfmadd231ps!(1, 4, 9, 0),
			vfmadd231ps!(2, 4, 10, 0),
		    vfmadd231ps!(3, 4, 11, 0),
		)
	};
	(1) => {
		concat!(
			vfmadd231ps!(0, 4, 12, 1),
			vfmadd231ps!(1, 4, 13, 1),
			vfmadd231ps!(2, 4, 14, 1),
		    vfmadd231ps!(3, 4, 15, 1),
		)
	};
	(2) => {
		concat!(
			vfmadd231ps!(0, 4, 16, 2),
			vfmadd231ps!(1, 4, 17, 2),
			vfmadd231ps!(2, 4, 18, 2),
		    vfmadd231ps!(3, 4, 19, 2),
		)
	};
	(3) => {
		concat!(
			vfmadd231ps!(0, 4, 20, 3),
			vfmadd231ps!(1, 4, 21, 3),
			vfmadd231ps!(2, 4, 22, 3),
		    vfmadd231ps!(3, 4, 23, 3),
		)
	};
	(4) => {
		concat!(
			vfmadd231ps!(0, 5, 24, 0),
			vfmadd231ps!(1, 5, 25, 0),
			vfmadd231ps!(2, 5, 26, 0),
		    vfmadd231ps!(3, 5, 27, 0),
		)
	};
	(5) => {
		concat!(
			vfmadd231ps!(0, 5, 28, 1),
			vfmadd231ps!(1, 5, 29, 1),
			vfmadd231ps!(2, 5, 30, 1),
		    vfmadd231ps!(3, 5, 31, 1),
		)
	};
}

macro_rules! fmadd_1v {
	(0) => {
		concat!(
			vfmadd231ps!(0, 2, 20, 0),
			vfmadd231ps!(1, 2, 21, 0),
		)
	};
	(1) => {
		concat!(
			vfmadd231ps!(0, 2, 22, 1),
			vfmadd231ps!(1, 2, 23, 1),
		)
	};
	(2) => {
		concat!(
			vfmadd231ps!(0, 2, 24, 2),
			vfmadd231ps!(1, 2, 25, 2),
		)
	};
	(3) => {
		concat!(
			vfmadd231ps!(0, 2, 26, 3),
			vfmadd231ps!(1, 2, 27, 3),
		)
	};
	(4) => {
		concat!(
			vfmadd231ps!(0, 3, 28, 0),
			vfmadd231ps!(1, 3, 29, 0),
		)
	};
	(5) => {
		concat!(
			vfmadd231ps!(0, 3, 30, 1),
			vfmadd231ps!(1, 3, 31, 1),
		)
	};
}

macro_rules! b_num_24x4 {
	(0,S) => {6};
	(1,S) => {6};
	(2,S) => {6};
	(3,S) => {6};
	($nr:tt,B) => {6};
}

macro_rules! b_num_16x6 {
	(0) => {2};
	(1) => {3};
	(2) => {2};
	(3) => {3};
	(4) => {2};
	(5) => {3};
}

macro_rules! b_num_8x6 {
	(0) => {1};
	(1) => {2};
	(2) => {3};
	(3) => {4};
	(4) => {5};
	(5) => {6};
}

// ***************************** 24x4 ******************************* //
macro_rules! asm_24x4_step {
	($N:tt, $nr:tt, $a_layout:tt, $b_layout:tt, $K:tt) => {
		seq!(n in 0..$nr {
			concat!(
				load_a!($N, $a_layout, $K),
				#(
					load_b!($b_layout, n, $K, $nr, b_num_24x4!(n,$b_layout)),
					fmadd_3v!(n),
				)*
			)
		})
	};
}

// ***************************** 16x6 ******************************* //
macro_rules! asm_16x6_step {
	($N:tt, $nr:tt, $a_layout:tt, $b_layout:tt, $K:tt) => {
		seq!(n in 0..$nr {
			concat!(
				load_a!($N, $a_layout, $K),
				#(
					load_b!($b_layout, n, $K, $nr, 4),
					fmadd_2v!(n),
				)*
			)
		})
	};
}

// ***************************** 8x6 ******************************* //
macro_rules! asm_8x6_step {
	($N:tt, $nr:tt, $a_layout:tt, $b_layout:tt, $K:tt) => {
		seq!(n in 0..$nr {
			concat!(
				load_a!($N, $a_layout, $K),
				#(
					load_b!($b_layout, n, $K, $nr, 2),
					fmadd_1v!(n),
				)*
			)
		})
	};
}

macro_rules! prefetch_0 {
	(0, $reg:tt, $k_i:tt) => {
		concat!(
			"",
		)
	};
	($dist:tt, $reg:tt, $k_i:tt) => {
		concat!(
			"prfm pldl1keep, [", $reg, ", #", $k_i, "*64+", $dist, "] \n",
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
			prefetch_0!($dist, "{ax}", 1),
			prefetch_0!($dist, "{ax}", 2)
		)
	};

	(24, B, $dist:tt, $unroll:tt, 2) => {
		prefetch_0!($dist, "{ax}", 3)
	};

	(24, B, $dist:tt, $unroll:tt, 3) => {
		concat!(
			prefetch_0!($dist, "{ax}", 4),
			prefetch_0!($dist, "{ax}", 5)
		)
	};

	(24, B, $dist:tt, $unroll:tt, 4) => {
		prefetch_0!($dist, "{ax}", 6)
	};
	(24, B, $dist:tt, $unroll:tt, 5) => {
		concat!(
			prefetch_0!($dist, "{ax}", 7),
			prefetch_0!($dist, "{ax}", 8)
		)
	};
	(24, B, $dist:tt, $unroll:tt, 6) => {
		prefetch_0!($dist, "{ax}", 9)
	};

	(24, B, $dist:tt, $unroll:tt, 7) => {
		concat!(
			prefetch_0!($dist, "{ax}", 10),
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

macro_rules! prefetch_b {
	($nr:tt, S, $dist:tt, $unroll:tt, $i:tt) => {
		prefetch_0!(0, "{bx},{x1}, lsl #3", 0)
	};
	($nr:tt, B, $dist:tt, $unroll:tt, 0) => {
		prefetch_0!($dist, "{bx}", 0)
	};
	($nr:tt, B, $dist:tt, $unroll:tt, 4) => {
		prefetch_0!($dist, "{bx}", 1)
	};
	($nr:tt, B, $dist:tt, $unroll:tt, $k_i:tt) => {
		""
	};
}

macro_rules! prefetch_c {
	(24, 4) => {
		concat!(
			"ldr {x0}, [{int_arrx}, #16]\n",
			"lsl {x0}, {x0}, #2\n",
			"add {x1}, {cx}, {x0}\n ",
			"add {x2}, {x1}, {x0} \n",
			"add {x3}, {x2}, {x0} \n",
			"prfm pldl1keep, [{cx}] \n",
			"prfm pldl1keep, [{cx},#64]\n",
			"prfm pldl1keep, [{x1}] \n",
			"prfm pldl1keep, [{x1},#64]\n",
			"prfm pldl1keep, [{x2}] \n",
			"prfm pldl1keep, [{x2},#64]\n",
			"prfm pldl1keep, [{x3}] \n",
			"prfm pldl1keep, [{x3},#64]\n",
		)
    };
    (24, $nr:tt) => {
		""
        // seq!(j in 0..$nr {
        //     concat!(
		// 		#(

		// 		)*
		// 	)
        // });
    };
    (16, $nr:tt) => {
			""
        // seq!(j in 0..$nr {
        //     _mm_prefetch($c.add(16+j*$ldc) as *const i8, 3);
        // });
    };
    (8, $nr:tt) => {
			""
        // seq!(j in 0..$nr {
        //     _mm_prefetch($c.add(8+j*$ldc) as *const i8, 3);
        // });
    }
}

macro_rules! pre_temp_buf {
	(T, $mr:tt, $nr:tt, $c_buf:ident,$c_ptr:ident,$c0_ptr:tt,$u64_arr:ident,$u64_ptr:ident,$ld_arr:tt,$ldc:tt,$k_iter:tt,$k_left:tt) => {
		let mut $c_buf = [0_f32;$mr*$nr];
		// process c_buf
		let $u64_arr = [$ld_arr[0], $ld_arr[1], $mr, $k_iter, $k_left];
		let $u64_ptr = $u64_arr.as_ptr();
		let $c_ptr = $c_buf.as_mut_ptr();
		for i in 0..$nr {
			std::ptr::copy_nonoverlapping($c0_ptr.add(i*$ldc), $c_ptr.add(i*$mr),$ld_arr[2])
		}
	};
	(F, $mr:tt, $nr:tt, $c_buf:ident,$c_ptr:ident,$c0_ptr:tt,$u64_arr:ident,$u64_ptr:ident,$ld_arr:tt,$ldc:tt,$k_iter:tt,$k_left:tt) => {
		let $c_ptr = $c0_ptr;
		let $u64_arr = [$ld_arr[0], $ld_arr[1], $ldc, $k_iter, $k_left];
		let $u64_ptr = $u64_arr.as_ptr();
	};
}

macro_rules! post_temp_buf {
	(T, $mr:tt, $nr:tt, $c_buf:ident,$c_ptr:ident,$c0_ptr:tt,$u64_arr:ident,$u64_ptr:ident,$ld_arr:tt,$ldc:tt,$k_iter:tt,$k_left:tt) => {
		// process original c_ptr
		for i in 0..$nr {
			std::ptr::copy_nonoverlapping($c_buf.as_mut_ptr().add(i*$mr), $c0_ptr.add(i*$ldc),$ld_arr[2])
		}
	};
	(F, $mr:tt, $nr:tt, $c_buf:ident,$c_ptr:ident,$c0_ptr:tt,$u64_arr:ident,$u64_ptr:ident,$ld_arr:tt,$ldc:tt,$k_iter:tt,$k_left:tt) => {
	};
}
 
 
macro_rules! def_ukernel {
	(
		$VER:tt,
		$asm_step_macro:tt,
		$asm_acc_macro:tt,
		$asm_store_macro:tt,
    	$mr:tt, $nr:tt,
    	$a_layout:tt, $b_layout:tt,
		$pfa_dist:tt, $pfb_dist:tt,
		$unroll:tt,
		$temp_buf:tt,
    	$func_name:ident
	) => {
		#[target_feature(enable = "neon")]
    	pub(crate) unsafe fn $func_name(
        	a: *const TA, b: *const TB, c: *mut TC,
        	alpha: *const TA, beta: *const TB,
        	k: usize,
        	ldc: usize,
			ld_arr: [usize; 3]
    	) {
        	let k_iter = k / $unroll;
        	let k_left = k % $unroll;
			// let cf = c;
			pre_temp_buf!($temp_buf,$mr,$nr,arr_buf,cf,c,u64_arr,u64_ptr,ld_arr,ldc,k_iter,k_left);
			// prefetch for c
        	asm!(
				prefetch_c!($mr,$nr),

            	asm_vzeroall!($VER,$nr),
   	 
            	asm_init_ab!($VER,$a_layout,$b_layout),
           	 
            	// 3 -> CONSIDKLEFT
            	"BEQ 3f",
           	 
            	// 2 -> KITER
            	"2:",
				seq!( i in 0..$unroll{
					concat!(
						#(
							prefetch_a!($mr, $a_layout, $pfa_dist, $unroll, i),
							prefetch_b!($nr, $b_layout, $pfb_dist, $unroll, i),
							$asm_step_macro!($mr, $nr, $a_layout, $b_layout, i),
							inc_a!($a_layout),
							inc_b!($b_layout,$nr), 
						)*
					)
				}),

            	inc_a_k_unroll!($a_layout, $mr, $unroll),
            	inc_b_k_unroll!($b_layout, $nr, $unroll),
   	 
            	"sub {x0}, {x0}, #1",
            	// 2 -> KITER
				"cmp {x0}, 0",
				"BNE 2b",

            	// 3 -> CONSIDKLEFT
            	"3:",
            	"ldr {x0}, [{int_arrx}, #32]",
				"cmp {x0}, #0",

            	// 5 -> POSTACCUM
            	"BEQ 5f",
   	 
            	// 4 -> KLEFT
            	"4:",
            	$asm_step_macro!($mr, $nr, $a_layout, $b_layout, 0),
            	inc_a!($a_layout),
            	inc_b!($b_layout,$nr),
            	inc_a_k_unroll!($a_layout, $mr, 1),
            	inc_b_k_unroll!($b_layout, $nr, 1),

            	"sub {x0}, {x0}, #1",
   	 
            	// 4 -> KLEFT
				"cmp {x0}, 0",
				"BNE 4b",
   	 
            	// 5 -> POSTACCUM
            	"5:",
            	asm_c_load!($nr),
            	// scale by alpha
            	asm_alpha_scale!($VER, $nr),

				"ldr s0, [{betax}]",
				"/* {betax} */",

            	"fcmp s0,#0.0",

            	// 6 -> BETAZERO
            	"BEQ 6f",
            	$asm_acc_macro!($mr,$nr,C),

            	// 6 -> BETAZERO
            	"6:",
            	$asm_store_macro!($mr,$nr,C),
   	 
            	// 7 -> DDONE
            	"7:",
				"/* {x4} */",
				"/* {x5} */",
            	ax = inout(reg) a => _,
            	bx = inout(reg) b => _,
            	cx = inout(reg) cf => _,
            	int_arrx = inout(reg) u64_ptr => _,
            	alphax = inout(reg) alpha => _,
            	betax = inout(reg) beta => _,
            	x0 = out(reg) _,
            	x1 = out(reg) _,
            	x2 = out(reg) _,
            	x3 = out(reg) _,
				x4 = out(reg) _,
				x5 = out(reg) _,
            	out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            	out("v8") _, out("v9") _, out("v10") _, out("v11") _, out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            	out("v16") _, out("v17") _, out("v18") _, out("v19") _, out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            	out("v24") _, out("v25") _, out("v26") _, out("v27") _, out("v28") _, out("v29") _, out("v30") _, out("v31") _,

        	);
			post_temp_buf!($temp_buf,$mr,$nr,arr_buf,cf,c,u64_arr,u64_ptr,ld_arr,ldc,k_iter,k_left);
    	}
	};
}

macro_rules! def_ukernel_partial {
	(
		$VER:tt,
		$asm_step_macro:tt,
		$asm_acc_macro:tt,
		$asm_store_macro:tt,
    	$mr:tt, $nr:tt,
    	$a_layout:tt, $b_layout:tt,
		$pfa_dist:tt, $pfb_dist:tt,
		$unroll:tt,
		$temp_buf:tt,
    	$func_name:ident
	) => {
		#[target_feature(enable = "neon")]
    	pub(crate) unsafe fn $func_name(
        	a: *const TA, b: *const TB, c: *mut TC,
        	alpha: *const TA, beta: *const TB,
        	k: usize,
        	ldc: usize,
			ld_arr: [usize; 3],
			mask: *const u32
    	) {
        	let k_iter = k / $unroll;
        	let k_left = k % $unroll;
            let u64_arr = [ld_arr[0], ld_arr[1], ldc, k_iter, k_left];
        	let u64_ptr = u64_arr.as_ptr();
			let cf = c;
			// prefetch for c
        	asm!(
				prefetch_c!($mr,$nr),

            	asm_vzeroall!($VER,$nr),
   	 
            	asm_init_ab!($VER,$a_layout,$b_layout),
           	 
            	// 3 -> CONSIDKLEFT
            	"BEQ 3f",
           	 
            	// 2 -> KITER
            	"2:",
				seq!( i in 0..$unroll{
					concat!(
						#(
							prefetch_a!($mr, $a_layout, $pfa_dist, $unroll, i),
							prefetch_b!($nr, $b_layout, $pfb_dist, $unroll, i),
							$asm_step_macro!($mr, $nr, $a_layout, $b_layout, i),
							inc_a!($a_layout),
							inc_b!($b_layout,$nr), 
						)*
					)
				}),

            	inc_a_k_unroll!($a_layout, $mr, $unroll),
            	inc_b_k_unroll!($b_layout, $nr, $unroll),
   	 
            	"sub {x0}, {x0}, #1",
            	// 2 -> KITER
				"cmp {x0}, 0",
				"BNE 2b",

            	// 3 -> CONSIDKLEFT
            	"3:",
				"ldr {x0}, [{int_arrx}, #32]",
            	"cmp {x0}, #0",

            	// 5 -> POSTACCUM
            	"BEQ 5f",
   	 
            	// 4 -> KLEFT
            	"4:",
            	$asm_step_macro!($mr, $nr, $a_layout, $b_layout, 0),
            	inc_a!($a_layout),
            	inc_b!($b_layout,$nr),
            	inc_a_k_unroll!($a_layout, $mr, 1),
            	inc_b_k_unroll!($b_layout, $nr, 1),

            	"sub {x0}, {x0}, #1",
   	 
            	// 4 -> KLEFT
				"cmp {x0}, 0",
				"BNE 4b",
   	 
            	// 5 -> POSTACCUM
            	"5:",
            	asm_c_load!($nr),
            	// scale by alpha
            	asm_alpha_scale!($VER, $nr),

				"ldr s0, [{betax}]",
				"/* {maskx} */",

				"fcmp s7,#0.0",
				// "vmovdqu ({maskx}), %ymm1",
				// "/* {maskx}*/",

            	// 6 -> BETAZERO
            	"BEQ 6f",
            	$asm_acc_macro!($mr,$nr,M),

            	// 6 -> BETAZERO
            	"6:",
            	$asm_store_macro!($mr,$nr,M),
   	 
            	// 7 -> DDONE
            	"7:",
				"/* {x4} */",
				"/* {x5} */",
            	ax = inout(reg) a => _,
            	bx = inout(reg) b => _,
            	cx = inout(reg) cf => _,
            	int_arrx = inout(reg) u64_ptr => _,
            	alphax = inout(reg) alpha => _,
            	betax = inout(reg) beta => _,
				maskx = inout(reg) mask => _,
            	x0 = out(reg) _,
            	x1 = out(reg) _,
            	x2 = out(reg) _,
            	x3 = out(reg) _,
				x4 = out(reg) _,
				x5 = out(reg) _,
            	out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            	out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            	out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            	out("v12") _, out("v13") _, out("v14") _, out("v15") _,
        	);
    	}
	};
}


macro_rules! group_def_ukernel {
	(
    	$mr:tt,
    	$n0:tt, $n1:tt,
    	$a_layout:tt, $b_layout:tt,
    	$func_name:ident, $def_macro:tt,
		$asm_step_macro:tt,
		$asm_acc_macro:tt,
		$asm_store_macro:tt,
		$temp_buf:tt,
		$VER:tt
	) => {
		seq!(nr in $n0..=$n1 {
			paste! {
				[<$def_macro>]!(
					$VER,
					$asm_step_macro,
					$asm_acc_macro,
					$asm_store_macro,
					$mr, nr,
					$a_layout, $b_layout,
					0, 128,
					4,
					$temp_buf,
					[<ukernel_ $mr x nr _ $func_name>]
				);
			}
		});
	};
}

group_def_ukernel!(24, 1, 4, B, B, bb, def_ukernel, asm_24x4_step, asm_24x4_acc, asm_24x4_store, F, VER24);
group_def_ukernel!(24, 1, 4, B, S, bs, def_ukernel, asm_24x4_step, asm_24x4_acc, asm_24x4_store, F, VER24);
// group_def_ukernel!(24, 1, 4, B, B, bb_partial, def_ukernel_partial, asm_24x4_step, asm_24x4_acc, asm_24x4_store, F, VER24);
group_def_ukernel!(24, 1, 4, B, B, bb_partial, def_ukernel, asm_24x4_step, asm_24x4_acc, asm_24x4_store, T, VER24);
// group_def_ukernel!(24, 1, 4, B, S, bs_partial, def_ukernel_partial, asm_24x4_step, asm_24x4_acc, asm_24x4_store, F, VER24);
group_def_ukernel!(24, 1, 4, B, S, bs_partial, def_ukernel, asm_24x4_step, asm_24x4_acc, asm_24x4_store, T, VER24);



group_def_ukernel!(16, 1, 6, B, B, bb, def_ukernel, asm_16x6_step, asm_16x6_acc, asm_16x6_store, F, VER16);
// group_def_ukernel!(16, 1, 4, B, S, bs, def_ukernel, asm_16x6_step, asm_16x6_acc, asm_16x6_store, VER16);
// group_def_ukernel!(16, 1, 6, B, B, bb_partial, def_ukernel_partial, asm_16x6_step, asm_16x6_acc, asm_16x6_store, F, VER16);
// group_def_ukernel!(16, 1, 4, B, S, bs_partial, def_ukernel_partial, asm_16x6_step, asm_16x6_acc, asm_16x6_store, F, VER16);
group_def_ukernel!(16, 1, 6, B, B, bb_partial, def_ukernel, asm_16x6_step, asm_16x6_acc, asm_16x6_store, T, VER16);
group_def_ukernel!(16, 1, 4, B, S, bs_partial, def_ukernel, asm_16x6_step, asm_16x6_acc, asm_16x6_store, T, VER16);


group_def_ukernel!(8, 1, 6, B, B, bb, def_ukernel, asm_8x6_step, asm_8x6_acc, asm_8x6_store, F, VER8);
// group_def_ukernel!(8, 1, 4, B, S, bs, def_ukernel, asm_8x6_step, asm_8x6_acc, asm_8x6_store, VER8);
// group_def_ukernel!(8, 1, 6, B, B, bb_partial, def_ukernel_partial, asm_8x6_step, asm_8x6_acc, asm_8x6_store, F, VER8);
// group_def_ukernel!(8, 1, 4, B, S, bs_partial, def_ukernel_partial, asm_8x6_step, asm_8x6_acc, asm_8x6_store, F, VER8);
group_def_ukernel!(8, 1, 6, B, B, bb_partial, def_ukernel, asm_8x6_step, asm_8x6_acc, asm_8x6_store, T, VER8);
group_def_ukernel!(8, 1, 4, B, S, bs_partial, def_ukernel, asm_8x6_step, asm_8x6_acc, asm_8x6_store, T, VER8);



#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn ukernel_24x4_bsold(
    a: *const TA,
    b: *const TB,
    c: *mut TC,
    alpha: *const TA,
    beta: *const TB,
    k: usize,
    ldc: usize,
    ld_arr: [usize; 3],
) {
    let k_iter = k / 4;
    let k_left = k % 4;
    let cf = c;
    let u64_arr = [ld_arr[0], ld_arr[1], ldc, k_iter, k_left];
    let u64_ptr = u64_arr.as_ptr();
    asm!(
        "ldr {6}, [{3}, #16]",
"lsl {6}, {6}, #2",
"add {7}, {2}, {6}",
" add {8}, {7}, {6} ",
"add {9}, {8}, {6} ",
"prfm pldl1keep, [{2}] ",
"prfm pldl1keep, [{2},#64]",
"prfm pldl1keep, [{7}] ",
"prfm pldl1keep, [{7},#64]",
"prfm pldl1keep, [{8}] ",
"prfm pldl1keep, [{8},#64]",
"prfm pldl1keep, [{9}] ",
"prfm pldl1keep, [{9},#64]",
"dup v8.4s, wzr ",
"dup v9.4s, wzr ",
"dup v10.4s, wzr ",
"dup v11.4s, wzr ",
"dup v12.4s, wzr ",
"dup v13.4s, wzr ",
"dup v14.4s, wzr ",
"dup v15.4s, wzr ",
"dup v16.4s, wzr ",
"dup v17.4s, wzr ",
"dup v18.4s, wzr ",
"dup v19.4s, wzr ",
"dup v20.4s, wzr ",
"dup v21.4s, wzr ",
"dup v22.4s, wzr ",
"dup v23.4s, wzr ",
"dup v24.4s, wzr ",
"dup v25.4s, wzr ",
"dup v26.4s, wzr ",
"dup v27.4s, wzr ",
"dup v28.4s, wzr ",
"dup v29.4s, wzr ",
"dup v30.4s, wzr ",
"dup v31.4s, wzr ",
"/* {9} */",
"/* {8} */",
"/* {7} */",
"/* {6} */",
"ldr {7}, [{3}]",
"ldr {8}, [{3}, #8]",
"lsl {9}, {8}, #1 ",
"add {9}, {1}, {9} ",
"ldr {6}, [{3}, #24]",
"cmp {6}, #0",
"BEQ 3f",
"2:",
"ldr q0, [{0}, #24*4*0]",
"ldr q1, [{0}, #24*4*0+0x10]",
"ldr q2, [{0}, #24*4*0+0x20]",
"ldr q3, [{0}, #24*4*0+0x30]",
"ldr q4, [{0}, #24*4*0+0x40]",
"ldr q5, [{0}, #24*4*0+0x50]",
"ldr q6, [{1}]",
"fmla v8.4s, v0.4s, v6.s[0]",
"fmla v9.4s, v1.4s, v6.s[0]",
"fmla v10.4s, v2.4s, v6.s[0]",
"fmla v11.4s, v3.4s, v6.s[0]",
"fmla v12.4s, v4.4s, v6.s[0]",
"fmla v13.4s, v5.4s, v6.s[0]",
"ldr q6, [{1},{8}]",
"fmla v14.4s, v0.4s, v6.s[0]",
"fmla v15.4s, v1.4s, v6.s[0]",
"fmla v16.4s, v2.4s, v6.s[0]",
"fmla v17.4s, v3.4s, v6.s[0]",
"fmla v18.4s, v4.4s, v6.s[0]",
"fmla v19.4s, v5.4s, v6.s[0]",
"ldr q6, [{9}]",
"fmla v20.4s, v0.4s, v6.s[0]",
"fmla v21.4s, v1.4s, v6.s[0]",
"fmla v22.4s, v2.4s, v6.s[0]",
"fmla v23.4s, v3.4s, v6.s[0]",
"fmla v24.4s, v4.4s, v6.s[0]",
"fmla v25.4s, v5.4s, v6.s[0]",
"ldr q6, [{9},{8}]",
"fmla v26.4s, v0.4s, v6.s[0]",
"fmla v27.4s, v1.4s, v6.s[0]",
"fmla v28.4s, v2.4s, v6.s[0]",
"fmla v29.4s, v3.4s, v6.s[0]",
"fmla v30.4s, v4.4s, v6.s[0]",
"fmla v31.4s, v5.4s, v6.s[0]",
"add {1},{1},{7} ",
" add {9},{9},{7} ",
"ldr q0, [{0}, #24*4*1]",
"ldr q1, [{0}, #24*4*1+0x10]",
"ldr q2, [{0}, #24*4*1+0x20]",
"ldr q3, [{0}, #24*4*1+0x30]",
"ldr q4, [{0}, #24*4*1+0x40]",
"ldr q5, [{0}, #24*4*1+0x50]",
"ldr q6, [{1}]",
"fmla v8.4s, v0.4s, v6.s[0]",
"fmla v9.4s, v1.4s, v6.s[0]",
"fmla v10.4s, v2.4s, v6.s[0]",
"fmla v11.4s, v3.4s, v6.s[0]",
"fmla v12.4s, v4.4s, v6.s[0]",
"fmla v13.4s, v5.4s, v6.s[0]",
"ldr q6, [{1},{8}]",
"fmla v14.4s, v0.4s, v6.s[0]",
"fmla v15.4s, v1.4s, v6.s[0]",
"fmla v16.4s, v2.4s, v6.s[0]",
"fmla v17.4s, v3.4s, v6.s[0]",
"fmla v18.4s, v4.4s, v6.s[0]",
"fmla v19.4s, v5.4s, v6.s[0]",
"ldr q6, [{9}]",
"fmla v20.4s, v0.4s, v6.s[0]",
"fmla v21.4s, v1.4s, v6.s[0]",
"fmla v22.4s, v2.4s, v6.s[0]",
"fmla v23.4s, v3.4s, v6.s[0]",
"fmla v24.4s, v4.4s, v6.s[0]",
"fmla v25.4s, v5.4s, v6.s[0]",
"ldr q6, [{9},{8}]",
"fmla v26.4s, v0.4s, v6.s[0]",
"fmla v27.4s, v1.4s, v6.s[0]",
"fmla v28.4s, v2.4s, v6.s[0]",
"fmla v29.4s, v3.4s, v6.s[0]",
"fmla v30.4s, v4.4s, v6.s[0]",
"fmla v31.4s, v5.4s, v6.s[0]",
"add {1},{1},{7} ",
" add {9},{9},{7} ",
"ldr q0, [{0}, #24*4*2]",
"ldr q1, [{0}, #24*4*2+0x10]",
"ldr q2, [{0}, #24*4*2+0x20]",
"ldr q3, [{0}, #24*4*2+0x30]",
"ldr q4, [{0}, #24*4*2+0x40]",
"ldr q5, [{0}, #24*4*2+0x50]",
"ldr q6, [{1}]",
"fmla v8.4s, v0.4s, v6.s[0]",
"fmla v9.4s, v1.4s, v6.s[0]",
"fmla v10.4s, v2.4s, v6.s[0]",
"fmla v11.4s, v3.4s, v6.s[0]",
"fmla v12.4s, v4.4s, v6.s[0]",
"fmla v13.4s, v5.4s, v6.s[0]",
"ldr q6, [{1},{8}]",
"fmla v14.4s, v0.4s, v6.s[0]",
"fmla v15.4s, v1.4s, v6.s[0]",
"fmla v16.4s, v2.4s, v6.s[0]",
"fmla v17.4s, v3.4s, v6.s[0]",
"fmla v18.4s, v4.4s, v6.s[0]",
"fmla v19.4s, v5.4s, v6.s[0]",
"ldr q6, [{9}]",
"fmla v20.4s, v0.4s, v6.s[0]",
"fmla v21.4s, v1.4s, v6.s[0]",
"fmla v22.4s, v2.4s, v6.s[0]",
"fmla v23.4s, v3.4s, v6.s[0]",
"fmla v24.4s, v4.4s, v6.s[0]",
"fmla v25.4s, v5.4s, v6.s[0]",
"ldr q6, [{9},{8}]",
"fmla v26.4s, v0.4s, v6.s[0]",
"fmla v27.4s, v1.4s, v6.s[0]",
"fmla v28.4s, v2.4s, v6.s[0]",
"fmla v29.4s, v3.4s, v6.s[0]",
"fmla v30.4s, v4.4s, v6.s[0]",
"fmla v31.4s, v5.4s, v6.s[0]",
"add {1},{1},{7} ",
" add {9},{9},{7} ",
"ldr q0, [{0}, #24*4*3]",
"ldr q1, [{0}, #24*4*3+0x10]",
"ldr q2, [{0}, #24*4*3+0x20]",
"ldr q3, [{0}, #24*4*3+0x30]",
"ldr q4, [{0}, #24*4*3+0x40]",
"ldr q5, [{0}, #24*4*3+0x50]",
"ldr q6, [{1}]",
"fmla v8.4s, v0.4s, v6.s[0]",
"fmla v9.4s, v1.4s, v6.s[0]",
"fmla v10.4s, v2.4s, v6.s[0]",
"fmla v11.4s, v3.4s, v6.s[0]",
"fmla v12.4s, v4.4s, v6.s[0]",
"fmla v13.4s, v5.4s, v6.s[0]",
"ldr q6, [{1},{8}]",
"fmla v14.4s, v0.4s, v6.s[0]",
"fmla v15.4s, v1.4s, v6.s[0]",
"fmla v16.4s, v2.4s, v6.s[0]",
"fmla v17.4s, v3.4s, v6.s[0]",
"fmla v18.4s, v4.4s, v6.s[0]",
"fmla v19.4s, v5.4s, v6.s[0]",
"ldr s6, [{9}]",
"fmla v20.4s, v0.4s, v6.s[0]",
"fmla v21.4s, v1.4s, v6.s[0]",
"fmla v22.4s, v2.4s, v6.s[0]",
"fmla v23.4s, v3.4s, v6.s[0]",
"fmla v24.4s, v4.4s, v6.s[0]",
"fmla v25.4s, v5.4s, v6.s[0]",
"ldr s6, [{9},{8}]",
"fmla v26.4s, v0.4s, v6.s[0]",
"fmla v27.4s, v1.4s, v6.s[0]",
"fmla v28.4s, v2.4s, v6.s[0]",
"fmla v29.4s, v3.4s, v6.s[0]",
"fmla v30.4s, v4.4s, v6.s[0]",
"fmla v31.4s, v5.4s, v6.s[0]",
"add {1},{1},{7} ",
" add {9},{9},{7} ",
"add {0}, {0}, #4*4*24 ",
"sub {6}, {6}, #1",
"cmp {6}, 0",
"BNE 2b",
"3:",
"ldr {6}, [{3}, #32]",
"cmp {6}, #0",
"BEQ 5f",
"4:",
"ldr q0, [{0}, #24*4*0]",
"ldr q1, [{0}, #24*4*0+0x10]",
"ldr q2, [{0}, #24*4*0+0x20]",
"ldr q3, [{0}, #24*4*0+0x30]",
"ldr q4, [{0}, #24*4*0+0x40]",
"ldr q5, [{0}, #24*4*0+0x50]",
"ldr s24, [{1}]",
"fmla v8.4s, v0.4s, v6.s[0]",
"fmla v9.4s, v1.4s, v6.s[0]",
"fmla v10.4s, v2.4s, v6.s[0]",
"fmla v11.4s, v3.4s, v6.s[0]",
"fmla v12.4s, v4.4s, v6.s[0]",
"fmla v13.4s, v5.4s, v6.s[0]",
"ldr s25, [{1},{8}]",
"fmla v14.4s, v0.4s, v6.s[1]",
"fmla v15.4s, v1.4s, v6.s[1]",
"fmla v16.4s, v2.4s, v6.s[1]",
"fmla v17.4s, v3.4s, v6.s[1]",
"fmla v18.4s, v4.4s, v6.s[1]",
"fmla v19.4s, v5.4s, v6.s[1]",
"ldr s26, [{9}]",
"fmla v20.4s, v0.4s, v6.s[2]",
"fmla v21.4s, v1.4s, v6.s[2]",
"fmla v22.4s, v2.4s, v6.s[2]",
"fmla v23.4s, v3.4s, v6.s[2]",
"fmla v24.4s, v4.4s, v6.s[2]",
"fmla v25.4s, v5.4s, v6.s[2]",
"ldr s27, [{9},{8}]",
"fmla v26.4s, v0.4s, v6.s[3]",
"fmla v27.4s, v1.4s, v6.s[3]",
"fmla v28.4s, v2.4s, v6.s[3]",
"fmla v29.4s, v3.4s, v6.s[3]",
"fmla v30.4s, v4.4s, v6.s[3]",
"fmla v31.4s, v5.4s, v6.s[3]",
"add {1},{1},{7} ",
" add {9},{9},{7} ",
"add {0}, {0}, #4*1*24 ",
"sub {6}, {6}, #1",
"cmp {6}, 0",
"BNE 4b",
"5:",
"ldr {6}, [{3}, #16]",
"lsl {6}, {6}, #2",
"add {7}, {2}, {6} ",
"add {8}, {7}, {6} ",
"add {9}, {8}, {6} ",
"/* {2} */",
"ldr s1, [{4}]",
// "fmul  v4.4s, v4.4s, v1.s[0]",
// "fmul  v5.4s, v5.4s, v1.s[0]",
// "fmul  v6.4s, v6.4s, v1.s[0]",
// "fmul  v7.4s, v7.4s, v1.s[0]",
// "fmul  v8.4s, v8.4s, v1.s[0]",
// "fmul  v9.4s, v9.4s, v1.s[0]",
// "fmul  v10.4s, v10.4s, v1.s[0]",
// "fmul  v11.4s, v11.4s, v1.s[0]",
// "fmul  v12.4s, v12.4s, v1.s[0]",
// "fmul  v13.4s, v13.4s, v1.s[0]",
// "fmul  v14.4s, v14.4s, v1.s[0]",
// "fmul  v15.4s, v15.4s, v1.s[0]",
"ldr s0, [{5}]",
"/* {5} */",
"fcmp s0,#0.0",
"BEQ 6f",
"ldr q1, [{2}]",
"fmla v8.4s, v1.4s, v0.s[0]",
"ldr q1, [{2}, #0x10]",
"fmla v9.4s, v1.4s, v0.s[0]",
"ldr q1, [{2}, #0x20]",
"fmla v10.4s, v1.4s, v0.s[0]",
"ldr q1, [{2}, #0x30]",
"fmla v11.4s, v1.4s, v0.s[0]",
"ldr q1, [{2}, #0x40]",
"fmla v12.4s, v1.4s, v0.s[0]",
"ldr q1, [{2}, #0x50]",
"fmla v13.4s, v1.4s, v0.s[0]",
"ldr q1, [{7}]",
"fmla v14.4s, v1.4s, v0.s[0]",
"ldr q1, [{7}, #0x10]",
"fmla v15.4s, v1.4s, v0.s[0]",
"ldr q1, [{7}, #0x20]",
"fmla v16.4s, v1.4s, v0.s[0]",
"ldr q1, [{7}, #0x30]",
"fmla v17.4s, v1.4s, v0.s[0]",
"ldr q1, [{7}, #0x40]",
"fmla v18.4s, v1.4s, v0.s[0]",
"ldr q1, [{7}, #0x50]",
"fmla v19.4s, v1.4s, v0.s[0]",
"ldr q1, [{8}]",
"fmla v20.4s, v1.4s, v0.s[0]",
"ldr q1, [{8}, #0x10]",
"fmla v21.4s, v1.4s, v0.s[0]",
"ldr q1, [{8}, #0x20]",
"fmla v22.4s, v1.4s, v0.s[0]",
"ldr q1, [{8}, #0x30]",
"fmla v23.4s, v1.4s, v0.s[0]",
"ldr q1, [{8}, #0x40]",
"fmla v24.4s, v1.4s, v0.s[0]",
"ldr q1, [{8}, #0x50]",
"fmla v25.4s, v1.4s, v0.s[0]",
"ldr q1, [{9}]",
"fmla v26.4s, v1.4s, v0.s[0]",
"ldr q1, [{9}, #0x10]",
"fmla v27.4s, v1.4s, v0.s[0]",
"ldr q1, [{9}, #0x20]",
"fmla v28.4s, v1.4s, v0.s[0]",
"ldr q1, [{9}, #0x30]",
"fmla v29.4s, v1.4s, v0.s[0]",
"ldr q1, [{9}, #0x40]",
"fmla v30.4s, v1.4s, v0.s[0]",
"ldr q1, [{9}, #0x50]",
"fmla v31.4s, v1.4s, v0.s[0]",
"6:",
"str q8, [{2}]",
"str q9, [{2}, #0x10]",
"str q10, [{2}, #0x20]",
"str q11, [{2}, #0x30]",
"str q12, [{2}, #0x40]",
"str q13, [{2}, #0x50]",
"str q14, [{7}]",
"str q15, [{7}, #0x10]",
"str q16, [{7}, #0x20]",
"str q17, [{7}, #0x30]",
"str q18, [{7}, #0x40]",
"str q19, [{7}, #0x50]",
"str q20, [{8}]",
"str q21, [{8}, #0x10]",
"str q22, [{8}, #0x20]",
"str q23, [{8}, #0x30]",
"str q24, [{8}, #0x40]",
"str q25, [{8}, #0x50]",
"str q26, [{9}]",
"str q27, [{9}, #0x10]",
"str q28, [{9}, #0x20]",
"str q29, [{9}, #0x30]",
"str q30, [{9}, #0x40]",
"str q31, [{9}, #0x50]",
"7:",
        inout(reg) a => _, inout(reg) b => _, inout(reg) cf => _, inout(reg) u64_ptr =>
        _, inout(reg) alpha => _, inout(reg) beta => _, out(reg) _, out(reg) _, out(reg)
        _, out(reg) _, out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _,
        out("v5") _, out("v6") _, out("v7") _, out("v8") _, out("v9") _, out("v10") _,
        out("v11") _, out("v12") _, out("v13") _, out("v14") _, out("v15") _, out("v16")
        _, out("v17") _, out("v18") _, out("v19") _, out("v20") _, out("v21") _,
        out("v22") _, out("v23") _, out("v24") _, out("v25") _, out("v26") _, out("v27")
        _, out("v28") _, out("v29") _, out("v30") _, out("v31") _
    );
}
