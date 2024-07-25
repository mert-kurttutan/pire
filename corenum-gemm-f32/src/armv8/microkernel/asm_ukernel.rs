use seq_macro::seq;
use std::arch::asm;


use crate::{TA, TB, TC};

use paste::paste;
macro_rules! beta_fmaddps {
    (C, $m0:expr, $r1:expr) => {
        concat!(
			"ldr q", $r1, ", ", $m0, "\n",
			"fmla v", $r1, ".4s, v0.4s, v1.4s\n",
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
    (24, $layout:tt, $m0:expr, $r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            beta_fmaddps!(C, mem!($m0), $r1),
            beta_fmaddps!(C, mem!($m0, "0x20"), $r2),
            beta_fmaddps!($layout, mem!($m0, "0x40"), $r3),
        )
    };
    (16, $layout:tt, $m0:expr, $r1:expr, $r2:expr) => {
        concat!(
            beta_fmaddps!(C, mem!($m0), $r1),
            beta_fmaddps!($layout, mem!($m0, "0x20"), $r2),
        )
    };
    ($N:tt, $layout:tt, $m0:expr, $r1:expr) => {
        concat!(
            beta_fmaddps!($layout, mem!($m0), $r1),
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
	(24, $layout:tt, $m0:expr, $r1:expr, $r2:expr, $r3:expr) => {
		concat!(
			storeps_unit!(C, $r1, mem!($m0)),
			storeps_unit!(C, $r2, mem!($m0, "0x20")),
			storeps_unit!($layout, $r3, mem!($m0, "0x40")),
		)
	};
	(16, $layout:tt, $m0:expr, $r1:expr, $r2:expr) => {
		concat!(
			storeps_unit!(C, $r1, mem!($m0)),
			storeps_unit!($layout, $r2, mem!($m0, "0x20")),
		)
	};
	(8, $layout:tt, $m0:expr, $r1:expr) => {
		concat!(
			storeps_unit!($layout, $r1, mem!($m0)),
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
    	)
	};
}


macro_rules! asm_c_load {
	(6) => {
    	concat!(
			"ldr {x0}, [{int_arrx}, #16]\n",
			"lsl {x0}, {x0}, #4\n",
        	"add {x1}, {cx}, {x0} \n",
			"add {x2}, {x1}, {x0} \n",
        	"add {x3}, {x2}, {x0} \n",
    	)
	};
	(5) => {
    	concat!(
			"ldr {x0}, [{int_arrx}, #16]\n",
			"lsl {x0}, {x0}, #4\n",
        	"add {x1}, {cx}, {x0} \n",
			"add {x2}, {x1}, {x0} \n",
        	"add {x3}, {x2}, {x0} \n",
    	)
	};
	(4) => {
    	concat!(
			"ldr {x0}, [{int_arrx}, #16]\n",
			"lsl {x0}, {x0}, #4\n",
        	"add {x1}, {cx}, {x0} \n",
			"add {x2}, {x1}, {x0} \n",
        	"add {x3}, {x2}, {x0} \n",
    	)
	};
	(3) => {
    	concat!(
			"ldr {x0}, [{int_arrx}, #16]\n",
			"lsl {x0}, {x0}, #4\n",
        	"add {x1}, {cx}, {x0} \n",
			"add {x2}, {x1}, {x0} \n",
    	)
	};
	(2) => {
    	concat!(
			"ldr {x0}, [{int_arrx}, #16]\n",
			"lsl {x0}, {x0}, #4\n",
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
	(VER24,3) => {vzeroall!(4,12)};
	(VER24,2) => {vzeroall!(4,9)};
	(VER24,1) => {vzeroall!(4,6)};

	(VER16,6) => {vzeroall!(4,15)};
	(VER16,5) => {vzeroall!(4,13)};
	(VER16,4) => {vzeroall!(4,11)};
	(VER16,3) => {vzeroall!(4,9)};
	(VER16,2) => {vzeroall!(4,7)};
	(VER16,1) => {vzeroall!(4,5)};

	(VER8,6) => {vzeroall!(7,12)};
	(VER8,5) => {vzeroall!(7,11)};
	(VER8,4) => {vzeroall!(7,10)};
	(VER8,3) => {vzeroall!(7,9)};
	(VER8,2) => {vzeroall!(7,8)};
	(VER8,1) => {vzeroall!(7,7)};
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
    	"add {bx},{bx},{x1} \n add {x3},{x3},{x1} \n"
	};
	(S,3) => {
    	"add {bx},{bx},{x1} \n"
	};
	(S,2) => {
    	"add {bx},{bx},{x1} \n"
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
		acc_ps!($mr, $layout, "{cx}", 4, 5, 6)
	};
	($mr:tt, 1, $layout:tt) => {
		acc_ps!($mr, $layout, "{x1}", 7, 8, 9)
	};
	($mr:tt, 2, $layout:tt) => {
		acc_ps!($mr, $layout, "{x2}",  10, 11, 12)
	}; 
	($mr:tt, 3, $layout:tt) => {
		acc_ps!($mr, $layout, "{x3}",  13, 14, 15)
	};
}

macro_rules! asm_24x4_store_seq {
	($mr:tt, 0, $layout:tt) => {
		storeps!($mr, $layout, "{cx}", 4, 5, 6)
	};
	($mr:tt, 1, $layout:tt) => {
		storeps!($mr, $layout, "{x1}", 7, 8, 9)
	};
	($mr:tt, 2, $layout:tt) => {
		storeps!($mr, $layout, "{x2}",  10, 11, 12)
	}; 
	($mr:tt, 3, $layout:tt) => {
		storeps!($mr, $layout, "{x3}",  13, 14, 15)
	};
}

macro_rules! asm_16x6_acc_seq {
	($mr:tt, 0, $layout:tt) => {
		acc_ps!($mr, $layout, "{cx}", 4, 5)
	};
	($mr:tt, 1, $layout:tt) => {
		acc_ps!($mr, $layout, "{x1}", 6, 7)
	};
	($mr:tt, 2, $layout:tt) => {
		acc_ps!($mr, $layout, "{x2}", 8, 9)
	}; 
	($mr:tt, 3, $layout:tt) => {
		acc_ps!($mr, $layout, "{x3}", 10, 11)
	};
	($mr:tt, 4, $layout:tt) => {
        acc_ps!($mr, $layout, "{x3}", 12, 13)
	};
	($mr:tt, 5, $layout:tt) => {
		acc_ps!($mr, $layout, "{x3}", 14, 15)
	};
}

macro_rules! asm_16x6_store_seq {
	($mr:tt, 0, $layout:tt) => {
		storeps!($mr, $layout, "{cx}", 4, 5)
	};
	($mr:tt, 1, $layout:tt) => {
		storeps!($mr, $layout, "{x1}", 6, 7)
	};
	($mr:tt, 2, $layout:tt) => {
		storeps!($mr, $layout, "{x2}", 8, 9)
	}; 
	($mr:tt, 3, $layout:tt) => {
		storeps!($mr, $layout, "{x3}", 10, 11)
	};
	($mr:tt, 4, $layout:tt) => {
    	storeps!($mr, $layout, "{x3}", 12, 13)
	};
	($mr:tt, 5, $layout:tt) => {
		storeps!($mr, $layout, "{x3}", 14, 15)
	};
}

macro_rules! asm_8x6_acc_seq {
	($mr:tt, 0, $layout:tt) => {
		acc_ps!($mr, $layout, "{cx}", 7)
	};
	($mr:tt, 1, $layout:tt) => {
		acc_ps!($mr, $layout, "{x1}", 8)
	};
	($mr:tt, 2, $layout:tt) => {
		acc_ps!($mr, $layout, "{x2}", 9)
	}; 
	($mr:tt, 3, $layout:tt) => {
		acc_ps!($mr, $layout, "{x3}", 10)
	};
	($mr:tt, 4, $layout:tt) => {
        acc_ps!($mr, $layout, "{x3}", 11)
	};
	($mr:tt, 5, $layout:tt) => {
		acc_ps!($mr, $layout, "{x3}", 12)
	};
}

macro_rules! asm_8x6_store_seq {
	($mr:tt, 0, $layout:tt) => {
		storeps!($mr, $layout, "{cx}", 7)
	};
	($mr:tt, 1, $layout:tt) => {
		storeps!($mr, $layout, "{x1}", 8)
	};
	($mr:tt, 2, $layout:tt) => {
		storeps!($mr, $layout, "{x2}", 9)
	}; 
	($mr:tt, 3, $layout:tt) => {
		storeps!($mr, $layout, "{x3}", 10)
	};
	($mr:tt, 4, $layout:tt) => {
        storeps!($mr, $layout, "{x3}", 11)
	};
	($mr:tt, 5, $layout:tt) => {
		storeps!($mr, $layout, "{x3}", 12)
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
			"ldr s", $r, ", [{bx}]", "\n",			
    	)
	};
	(S, 1, $K:tt, $X:tt, $r:expr) => {
    	concat!(
			"ldr s", $r, ", [{bx},{x2}]", "\n",			
    	)
	};
	(S, 2, $K:tt, $X:tt, $r:expr) => {
    	concat!(
			"ldr s", $r, ", [{bx},{x2}, LSL #2]", "\n",	
    	)
	};
	(S, 3, $K:tt, $X:tt, $r:expr) => {
    	concat!(
			"ldr s", $r, ", [{x3}]", "\n",	
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
	(B, $N:tt, $K:tt, $X:tt, $r:expr) => {
    	concat!(
			"ldr q", $r, ", [{bx}, #", $K, "*", $X, "*4+", $N, "*4]", "\n",
    	)
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
			vfmadd231ps!(0, 2, 4, 0),
			vfmadd231ps!(1, 2, 5, 0),
		)
	};
	(1) => {
		concat!(
			vfmadd231ps!(0, 3, 6, 0),
			vfmadd231ps!(1, 3, 7, 0),
		)
	};
	(2) => {
		concat!(
			vfmadd231ps!(0, 2, 8, 0),
			vfmadd231ps!(1, 2, 9, 0),
		)
	};
	(3) => {
		concat!(
			vfmadd231ps!(0, 3, 10, 0),
			vfmadd231ps!(1, 3, 11, 0),
		)
	};
	(4) => {
		concat!(
			vfmadd231ps!(0, 2, 12, 0),
			vfmadd231ps!(1, 2, 13, 0),
		)
	};
	(5) => {
		concat!(
			vfmadd231ps!(0, 3, 14, 0),
			vfmadd231ps!(1, 3, 15, 0),
		)
	};
}

macro_rules! fmadd_1v {
	(0) => {
		concat!(
			vfmadd231ps!(0, 1, 7, 0),
		)
	};
	(1) => {
		concat!(
			vfmadd231ps!(0, 2, 8, 0),
		)
	};
	(2) => {
		concat!(
			vfmadd231ps!(0, 3, 9, 0),
		)
	};
	(3) => {
		concat!(
			vfmadd231ps!(0, 4, 10, 0),
		)
	};
	(4) => {
		concat!(
			vfmadd231ps!(0, 5, 11, 0),
		)
	};
	(5) => {
		concat!(
			vfmadd231ps!(0, 6, 12, 0),
		)
	};
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
				load_b!($b_layout, 0, $K, $nr, 6),
				#(
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
				load_b!($b_layout, 0, $K, $nr, 6),
				#(
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
				load_b!($b_layout, 0, $K, $nr, 6),
				#(
					fmadd_1v!(n),
				)*
			)
		})
	};
}

macro_rules! prefetch_0 {
	($dist:tt, $reg:tt, $k_i:tt) => {
		concat!(
			// "prefetcht0 ", $dist, "+", $k_i, "*64(", $reg, ")", "\n"
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
		prefetch_0!(0, "{bx},{x1},8", 0)
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
    (24, $nr:tt, $c:tt, $ldc:tt) => {
        seq!(j in 0..$nr {
            // _mm_prefetch($c.add(0+j*$ldc) as *const i8, 3);
            // _mm_prefetch($c.add(12+j*$ldc) as *const i8, 3);
			// _mm_prefetch($c.add(23+j*$ldc) as *const i8, 3);
        });
    };
    (16, $nr:tt, $c:tt, $ldc:tt) => {
        // seq!(j in 0..$nr {
        //     _mm_prefetch($c.add(16+j*$ldc) as *const i8, 3);
        // });
    };
    (8, $nr:tt, $c:tt, $ldc:tt) => {
        // seq!(j in 0..$nr {
        //     _mm_prefetch($c.add(8+j*$ldc) as *const i8, 3);
        // });
    }
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
    	$func_name:ident
	) => {
		#[target_feature(enable = "neon")]
    	pub(crate) unsafe fn $func_name(
        	a: *const TA, b: *const TB, c: *mut TC,
        	alpha: *const TA, beta: *const TB,
        	k: usize,
        	ldc: usize,
			ld_arr: [usize; 2]
    	) {
        	let k_iter = k / $unroll;
        	let k_left = k % $unroll;
            let u64_arr = [ld_arr[0], ld_arr[1], ldc, k_iter, k_left];
        	let u64_ptr = u64_arr.as_ptr();
			let cf = c;
			// prefetch for c
			prefetch_c!($mr,$nr,c,ldc);
        	asm!(
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

				"ldr q0, [{betax}]",
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
            	out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            	out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            	out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            	out("v12") _, out("v13") _, out("v14") _, out("v15") _,
        	);
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
    	$func_name:ident
	) => {
		#[target_feature(enable = "neon")]
    	pub(crate) unsafe fn $func_name(
        	a: *const TA, b: *const TB, c: *mut TC,
        	alpha: *const TA, beta: *const TB,
        	k: usize,
        	ldc: usize,
			ld_arr: [usize; 2],
			mask: *const u32
    	) {
        	let k_iter = k / $unroll;
        	let k_left = k % $unroll;
            let u64_arr = [ld_arr[0], ld_arr[1], ldc, k_iter, k_left];
        	let u64_ptr = u64_arr.as_ptr();
			let cf = c;
			// prefetch for c
			prefetch_c!($mr,$nr,c,ldc);
        	asm!(
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

				"ldr q0, [{betax}]",
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
					[<ukernel_ $mr x nr _ $func_name>]
				);
			}
		});
	};
}

group_def_ukernel!(24, 1, 4, B, B, bb, def_ukernel, asm_24x4_step, asm_24x4_acc, asm_24x4_store, VER24);
group_def_ukernel!(24, 1, 4, B, S, bs, def_ukernel, asm_24x4_step, asm_24x4_acc, asm_24x4_store, VER24);
group_def_ukernel!(24, 1, 4, B, B, bb_partial, def_ukernel_partial, asm_24x4_step, asm_24x4_acc, asm_24x4_store, VER24);
group_def_ukernel!(24, 1, 4, B, S, bs_partial, def_ukernel_partial, asm_24x4_step, asm_24x4_acc, asm_24x4_store, VER24);


group_def_ukernel!(16, 1, 6, B, B, bb, def_ukernel, asm_16x6_step, asm_16x6_acc, asm_16x6_store, VER16);
// group_def_ukernel!(16, 1, 4, B, S, bs, def_ukernel, asm_16x6_step, asm_16x6_acc, asm_16x6_store, VER16);
group_def_ukernel!(16, 1, 6, B, B, bb_partial, def_ukernel_partial, asm_16x6_step, asm_16x6_acc, asm_16x6_store, VER16);
group_def_ukernel!(16, 1, 4, B, S, bs_partial, def_ukernel_partial, asm_16x6_step, asm_16x6_acc, asm_16x6_store, VER16);

group_def_ukernel!(8, 1, 6, B, B, bb, def_ukernel, asm_8x6_step, asm_8x6_acc, asm_8x6_store, VER8);
// group_def_ukernel!(8, 1, 4, B, S, bs, def_ukernel, asm_8x6_step, asm_8x6_acc, asm_8x6_store, VER8);
group_def_ukernel!(8, 1, 6, B, B, bb_partial, def_ukernel_partial, asm_8x6_step, asm_8x6_acc, asm_8x6_store, VER8);
group_def_ukernel!(8, 1, 4, B, S, bs_partial, def_ukernel_partial, asm_8x6_step, asm_8x6_acc, asm_8x6_store, VER8);

