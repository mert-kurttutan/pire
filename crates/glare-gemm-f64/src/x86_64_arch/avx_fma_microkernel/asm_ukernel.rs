use seq_macro::seq;
use std::arch::asm;

use crate::{TA, TB, TC};

use paste::paste;
macro_rules! beta_fmaddpd {
    (C, $m0:expr, $r1:expr) => {
        concat!(
            "vfmadd231pd ", $m0, ",%ymm0,%ymm", $r1, "\n",
        ) 
    };
    (M, $m0:expr, $r1:expr) => {
        concat!(
            "vmaskmovpd ", $m0, ", %ymm1", ", %ymm2",  "\n",
            "vfmadd231pd %ymm2, %ymm0,%ymm", $r1, "\n",
        ) 
    };
 }

macro_rules! mem {
    ($m0:tt, $b0:tt) => {
        concat!($b0, "+", $m0)
    };
 }
macro_rules! vmovpd {
    (B) => {
        "vmovapd "
    };
    ($layout:tt) => {
        "vmovupd "
    };
 }
 macro_rules! acc_pd {
    (12, $layout:tt, $m0:expr, $r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            beta_fmaddpd!(C, $m0, $r1),
            beta_fmaddpd!(C, mem!($m0, "0x20"), $r2),
            beta_fmaddpd!($layout, mem!($m0, "0x40"), $r3),
        )
    };
    (8, $layout:tt, $m0:expr, $r1:expr, $r2:expr) => {
        concat!(
            beta_fmaddpd!(C, $m0, $r1),
            beta_fmaddpd!($layout, mem!($m0, "0x20"), $r2),
        )
    };
    ($N:tt, $layout:tt, $m0:expr, $r1:expr) => {
        concat!(
            beta_fmaddpd!($layout, $m0, $r1),
        )
    };
 }

macro_rules! vzeroall {
	($r0:tt, $r1:tt) => {
		seq!(r in $r0..=$r1 {
			concat!(
				#(
					"vpxor %ymm",r,",%ymm",r,",%ymm",r,"\n",
				)*
			)
		})
	}
}

macro_rules! loadpd_unit {
    ($layout:tt, $m0:expr, $r1:expr) => {
        concat!(
            vmovpd!($layout), $m0, ",%ymm", $r1, "\n",
        )
    };
 }
 macro_rules! loadpd {
    (12, $layout:tt, $m0:expr, $r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            loadpd_unit!($layout, $m0, $r1),
            loadpd_unit!($layout, mem!($m0, "0x20"), $r2),
            loadpd_unit!($layout, mem!($m0, "0x40"), $r3),
        )
    };
    (12, $layout:tt, $m0:expr) => {
        concat!(
            loadpd_unit!($layout, $m0, 0),
            loadpd_unit!($layout, mem!($m0, "0x20"), 1),
            loadpd_unit!($layout, mem!($m0, "0x40"), 2),
        )
    };
    (8, $layout:tt, $m0:expr, $r1:expr, $r2:expr) => {
        concat!(
            loadpd_unit!($layout, $m0, $r1),
            loadpd_unit!($layout, mem!($m0, "0x20"), $r2),
        )
    };
    (8, $layout:tt, $m0:expr) => {
        concat!(
            loadpd_unit!($layout, $m0, 0),
            loadpd_unit!($layout, mem!($m0, "0x20"), 1),
        )
    };
    (4, $layout:tt, $m0:expr) => {
        concat!(
            loadpd_unit!($layout, $m0, 0),
        )
    };
    (4, $layout:tt, $m0:expr, $r1:expr) => {
        concat!(
            loadpd_unit!($layout, $m0, $r1),
        )
    };
 }

 macro_rules! storepd_unit {
    (C, $r1:expr, $m0:expr) => {
        concat!(
            "vmovupd %ymm", $r1, ", ", $m0,  "\n",
        )
    };
    (B, $r1:expr, $m0:expr) => {
        concat!(
            "vmovapd %ymm", $r1, ", ", $m0,  "\n",
        )
    };
    (M, $r1:expr, $m0:expr) => {
        concat!(
            "vmaskmovpd %ymm", $r1, ", %ymm1, ", $m0,  "\n",
        )
    };
}
macro_rules! storepd {
	(12, $layout:tt, $m0:expr, $r1:expr, $r2:expr, $r3:expr) => {
		concat!(
			storepd_unit!(C, $r1, $m0),
			storepd_unit!(C, $r2, mem!($m0, "0x20")),
			storepd_unit!($layout, $r3, mem!($m0, "0x40")),
		)
	};
	(8, $layout:tt, $m0:expr, $r1:expr, $r2:expr) => {
		concat!(
			storepd_unit!(C, $r1, $m0),
			storepd_unit!($layout, $r2, mem!($m0, "0x20")),
		)
	};
	(4, $layout:tt, $m0:expr, $r1:expr) => {
		concat!(
			storepd_unit!($layout, $r1, $m0),
		)
	};
 }

macro_rules! vfmadd231pd {
    ($r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            "vfmadd231pd %ymm", $r1, ", %ymm", $r2,", %ymm", $r3, "\n",
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
        	"mov 24({int_arrx}),{x0}", "\n",
        	"test {x0},{x0}", "\n",
    	)
	};
	($ker:tt,B,S) => {
    	concat!(
        	// mov cs_b to reg
			"mov ({int_arrx}), {x1}", "\n",
        	"mov 8({int_arrx}), {x2}", "\n",
        	"lea ({x2}, {x2}, 2), {x3}", "\n",
        	"lea ({bx}, {x3}, 1), {x3}", "\n",

        	"mov 24({int_arrx}),{x0}", "\n",
        	"test {x0},{x0}", "\n",
    	)
	};
}


macro_rules! asm_c_load {
	(6) => {
    	concat!(
        	"mov 16({int_arrx}),{x0}", "\n",
        	"lea ({x0}, {x0}, 2), {x1}", "\n",
        	"lea ({cx}, {x1},), {x1}", "\n",
    	)
	};
	(5) => {
    	concat!(
        	"mov 16({int_arrx}),{x0}", "\n",
        	"lea ({x0}, {x0}, 2), {x1}", "\n",
        	"lea ({cx}, {x1},), {x1}", "\n",
    	)
	};
	(4) => {
    	concat!(
        	"mov 16({int_arrx}),{x0}", "\n",
        	"lea ({x0}, {x0}, 2), {x1}", "\n",
        	"lea ({cx}, {x1},), {x1}", "\n",
    	)
	};
	(3) => {
    	concat!(
        	"mov 16({int_arrx}),{x0}", "\n",
    	)
	};
	(2) => {
    	concat!(
        	"mov 16({int_arrx}),{x0}", "\n",
    	)
	};
	(1) => {
    	concat!(
        	"mov 16({int_arrx}),{x0}", "\n",
    	)
	};
}


macro_rules! asm_vzeroall {

	(VER24,4) => {vzeroall!(4,15)};
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
    	"add {x1}, {ax} \n"
	};
	(B) => {
    	""
	};
}

macro_rules! inc_b {
	(S,6) => {
    	"add {x1},{bx} \n add {x1},{x3} \n"
	};
	(S,5) => {
    	"add {x1},{bx} \n add {x1},{x3} \n"
	};
	(S,4) => {
    	"add {x1},{bx} \n add {x1},{x3} \n"
	};
	(S,3) => {
    	"add {x1},{bx} \n"
	};
	(S,2) => {
    	"add {x1},{bx} \n"
	};
	(S,1) => {
    	"add {x1},{bx} \n"
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
        	"add $8*", $K, "*", $X, ",{ax}", "\n",
    	)
	};
}

macro_rules! inc_b_k_unroll {
	(S, $X:tt, $K:tt) => {
    	""
	};
	(B, $X:tt, $K:tt) => {
    	concat!(
        	"add $8*", $K, "*", $X, ", {bx}", "\n",
    	)
	};
}

macro_rules! asm_alpha_scale_0 {
	($r0:tt, $r1:tt) => {
		seq!(r in $r0..=$r1 {
			concat!(
				"vbroadcastsd ({alphax}),%ymm1", "\n",
				#(
					"vmulpd %ymm1, %ymm", r, ",%ymm", r, "\n",
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


macro_rules! asm_12x4_acc_seq {
	($mr:tt, 0, $layout:tt) => {
		acc_pd!($mr, $layout, "0({cx})", 4, 5, 6)
	};
	($mr:tt, 1, $layout:tt) => {
		acc_pd!($mr, $layout, "0({cx}, {x0})", 7, 8, 9)
	};
	($mr:tt, 2, $layout:tt) => {
		acc_pd!($mr, $layout, "0({cx}, {x0}, 2)",  10, 11, 12)
	}; 
	($mr:tt, 3, $layout:tt) => {
		acc_pd!($mr, $layout, "0({x1})",  13, 14, 15)
	};
}

macro_rules! asm_12x4_store_seq {
	($mr:tt, 0, $layout:tt) => {
		storepd!($mr, $layout, "0({cx})", 4, 5, 6)
	};
	($mr:tt, 1, $layout:tt) => {
		storepd!($mr, $layout, "0({cx}, {x0})", 7, 8, 9)
	};
	($mr:tt, 2, $layout:tt) => {
		storepd!($mr, $layout, "0({cx}, {x0}, 2)",  10, 11, 12)
	}; 
	($mr:tt, 3, $layout:tt) => {
		storepd!($mr, $layout, "0({x1})",  13, 14, 15)
	};
}

macro_rules! asm_8x6_acc_seq {
	($mr:tt, 0, $layout:tt) => {
		acc_pd!($mr, $layout, "0({cx})", 4, 5)
	};
	($mr:tt, 1, $layout:tt) => {
		acc_pd!($mr, $layout, "0({cx}, {x0})", 6, 7)
	};
	($mr:tt, 2, $layout:tt) => {
		acc_pd!($mr, $layout, "0({cx}, {x0}, 2)", 8, 9)
	}; 
	($mr:tt, 3, $layout:tt) => {
		acc_pd!($mr, $layout, "0({x1})", 10, 11)
	};
	($mr:tt, 4, $layout:tt) => {
        acc_pd!($mr, $layout, "0({x1}, {x0})", 12, 13)
	};
	($mr:tt, 5, $layout:tt) => {
		acc_pd!($mr, $layout, "0({x1}, {x0}, 2)", 14, 15)
	};
}

macro_rules! asm_8x6_store_seq {
	($mr:tt, 0, $layout:tt) => {
		storepd!($mr, $layout, "0({cx})", 4, 5)
	};
	($mr:tt, 1, $layout:tt) => {
		storepd!($mr, $layout, "0({cx}, {x0})", 6, 7)
	};
	($mr:tt, 2, $layout:tt) => {
		storepd!($mr, $layout, "0({cx}, {x0}, 2)", 8, 9)
	}; 
	($mr:tt, 3, $layout:tt) => {
		storepd!($mr, $layout, "0({x1})", 10, 11)
	};
	($mr:tt, 4, $layout:tt) => {
    	storepd!($mr, $layout, "0({x1}, {x0})", 12, 13)
	};
	($mr:tt, 5, $layout:tt) => {
		storepd!($mr, $layout, "0({x1}, {x0}, 2)", 14, 15)
	};
}

macro_rules! asm_4x6_acc_seq {
	($mr:tt, 0, $layout:tt) => {
		acc_pd!($mr, $layout, "0({cx})", 7)
	};
	($mr:tt, 1, $layout:tt) => {
		acc_pd!($mr, $layout, "0({cx}, {x0})", 8)
	};
	($mr:tt, 2, $layout:tt) => {
		acc_pd!($mr, $layout, "0({cx}, {x0}, 2)", 9)
	}; 
	($mr:tt, 3, $layout:tt) => {
		acc_pd!($mr, $layout, "0({x1})", 10)
	};
	($mr:tt, 4, $layout:tt) => {
        acc_pd!($mr, $layout, "0({x1}, {x0})", 11)
	};
	($mr:tt, 5, $layout:tt) => {
		acc_pd!($mr, $layout, "0({x1}, {x0}, 2)", 12)
	};
}

macro_rules! asm_4x6_store_seq {
	($mr:tt, 0, $layout:tt) => {
		storepd!($mr, $layout, "0({cx})", 7)
	};
	($mr:tt, 1, $layout:tt) => {
		storepd!($mr, $layout, "0({cx}, {x0})", 8)
	};
	($mr:tt, 2, $layout:tt) => {
		storepd!($mr, $layout, "0({cx}, {x0}, 2)", 9)
	}; 
	($mr:tt, 3, $layout:tt) => {
		storepd!($mr, $layout, "0({x1})", 10)
	};
	($mr:tt, 4, $layout:tt) => {
        storepd!($mr, $layout, "0({x1}, {x0})", 11)
	};
	($mr:tt, 5, $layout:tt) => {
		storepd!($mr, $layout, "0({x1}, {x0}, 2)", 12)
	};
}

macro_rules! asm_12x4_acc {
	($mr:tt, $nr:tt, $layout:tt) => {
		seq!(n in 0..$nr {
			concat!(
				#(
					asm_12x4_acc_seq!($mr, n, $layout), 
				)*
			)
		})
	};
}


macro_rules! asm_12x4_store {
	($mr:tt, $nr:tt, $layout:tt) => {
		seq!(n in 0..$nr {
			concat!(
				#(
					asm_12x4_store_seq!($mr, n, $layout), 
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


macro_rules! asm_4x6_acc {
	($mr:tt, $nr:tt, $layout:tt) => {
		seq!(n in 0..$nr {
			concat!(
				#(
					asm_4x6_acc_seq!($mr, n, $layout), 
				)*
			)
		})
	};
}


macro_rules! asm_4x6_store {
	($mr:tt, $nr:tt, $layout:tt) => {
		seq!(n in 0..$nr {
			concat!(
				#(
					asm_4x6_store_seq!($mr, n, $layout), 
				)*
			)
		})
	};
}

macro_rules! load_b {
	(S, 0, $K:tt, $X:tt, $r:expr) => {
    	concat!(
        	"vbroadcastsd ({bx}),%ymm", $r, "\n",
    	)
	};
	(S, 1, $K:tt, $X:tt, $r:expr) => {
    	concat!(
        	"vbroadcastsd ({bx},{x2},1),%ymm", $r, "\n",
    	)
	};
	(S, 2, $K:tt, $X:tt, $r:expr) => {
    	concat!(
        	"vbroadcastsd ({bx},{x2},2),%ymm", $r, "\n",
    	)
	};
	(S, 3, $K:tt, $X:tt, $r:expr) => {
    	concat!(
        	"vbroadcastsd ({x3}),%ymm", $r, "\n",
    	)
	};
	(S, 4, $K:tt, $X:tt, $r:expr) => {
    	concat!(
        	"vbroadcastsd ({x3},{x2},1),%ymm", $r, "\n",
    	)
	};
	(S, 5, $K:tt, $X:tt, $r:expr) => {
    	concat!(
        	"vbroadcastsd ({x3},{x2},2),%ymm", $r, "\n",
    	)
	};
	(B, $N:tt, $K:tt, $X:tt, $r:expr) => {
    	concat!(
        	"vbroadcastsd ", $K, "*", $X, "*8+", $N, "*8({bx}), %ymm", $r, "\n",
    	)
	};
}


macro_rules! load_a {
	($mr:tt, B, $K:tt) => {
    	loadpd!($mr, B, concat!($mr,"*8*",$K,"({ax})"))
	};
	($mr:tt, C, $K:tt) => {
    	loadpd!($mr, C, "0({ax})")
	};
}

macro_rules! fmadd_3v {
	(0) => {
		concat!(
			vfmadd231pd!(0, 3, 4),
			vfmadd231pd!(1, 3, 5),
			vfmadd231pd!(2, 3, 6),
		)
	};
	(1) => {
		concat!(
			vfmadd231pd!(0, 3, 7),
			vfmadd231pd!(1, 3, 8),
			vfmadd231pd!(2, 3, 9),
		)
	};
	(2) => {
		concat!(
			vfmadd231pd!(0, 3, 10),
			vfmadd231pd!(1, 3, 11),
			vfmadd231pd!(2, 3, 12),
		)
	};
	(3) => {
		concat!(
			vfmadd231pd!(0, 3, 13),
			vfmadd231pd!(1, 3, 14),
			vfmadd231pd!(2, 3, 15),
		)
	};
}

macro_rules! fmadd_2v {
	(0) => {
		concat!(
			vfmadd231pd!(0, 2, 4),
			vfmadd231pd!(1, 2, 5),
		)
	};
	(1) => {
		concat!(
			vfmadd231pd!(0, 3, 6),
			vfmadd231pd!(1, 3, 7),
		)
	};
	(2) => {
		concat!(
			vfmadd231pd!(0, 2, 8),
			vfmadd231pd!(1, 2, 9),
		)
	};
	(3) => {
		concat!(
			vfmadd231pd!(0, 3, 10),
			vfmadd231pd!(1, 3, 11),
		)
	};
	(4) => {
		concat!(
			vfmadd231pd!(0, 2, 12),
			vfmadd231pd!(1, 2, 13),
		)
	};
	(5) => {
		concat!(
			vfmadd231pd!(0, 3, 14),
			vfmadd231pd!(1, 3, 15),
		)
	};
}

macro_rules! fmadd_1v {
	(0) => {
		concat!(
			vfmadd231pd!(0, 1, 7),
		)
	};
	(1) => {
		concat!(
			vfmadd231pd!(0, 2, 8),
		)
	};
	(2) => {
		concat!(
			vfmadd231pd!(0, 3, 9),
		)
	};
	(3) => {
		concat!(
			vfmadd231pd!(0, 4, 10),
		)
	};
	(4) => {
		concat!(
			vfmadd231pd!(0, 5, 11),
		)
	};
	(5) => {
		concat!(
			vfmadd231pd!(0, 6, 12),
		)
	};
}

macro_rules! b_num_8x6 {
	(0) => {2};
	(1) => {3};
	(2) => {2};
	(3) => {3};
	(4) => {2};
	(5) => {3};
}

macro_rules! b_num_4x6 {
	(0) => {1};
	(1) => {2};
	(2) => {3};
	(3) => {4};
	(4) => {5};
	(5) => {6};
}

// ***************************** 12x4 ******************************* //
macro_rules! asm_12x4_step {
	($N:tt, $nr:tt, $a_layout:tt, $b_layout:tt, $K:tt) => {
		seq!(n in 0..$nr {
			concat!(
				load_a!($N, $a_layout, $K),
				#(
					load_b!($b_layout, n, $K, $nr, 3),
					fmadd_3v!(n),
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
					load_b!($b_layout, n, $K, $nr, b_num_8x6!(n)),
					fmadd_2v!(n),
				)*
			)
		})
	};
}

// ***************************** 4x6 ******************************* //
macro_rules! asm_4x6_step {
	($N:tt, $nr:tt, $a_layout:tt, $b_layout:tt, $K:tt) => {
		seq!(n in 0..$nr {
			concat!(
				load_a!($N, $a_layout, $K),
				#(
					load_b!($b_layout, n, $K, $nr, b_num_4x6!(n)),
					fmadd_1v!(n),
				)*
			)
		})
	};
}

macro_rules! prefetch_0 {
	($dist:tt, $reg:tt, $k_i:tt) => {
		concat!(
			"prefetcht0 ", $dist, "+", $k_i, "*64(", $reg, ")", "\n"
		)
	};
}

macro_rules! prefetch_a {
	($mr:tt, $layout:tt, 0, $unroll:tt, $k_i:tt) => {
		""
	};
	(12, B, $dist:tt, $unroll:tt, 0) => {
		prefetch_0!($dist, "{ax}", 0)
	};

	(12, B, $dist:tt, $unroll:tt, 1) => {
		concat!(
			prefetch_0!($dist, "{ax}", 1),
			prefetch_0!($dist, "{ax}", 2)
		)
	};

	(12, B, $dist:tt, $unroll:tt, 2) => {
		prefetch_0!($dist, "{ax}", 3)
	};

	(12, B, $dist:tt, $unroll:tt, 3) => {
		concat!(
			prefetch_0!($dist, "{ax}", 4),
			prefetch_0!($dist, "{ax}", 5)
		)
	};

	(12, B, $dist:tt, $unroll:tt, 4) => {
		prefetch_0!($dist, "{ax}", 6)
	};
	(12, B, $dist:tt, $unroll:tt, 5) => {
		concat!(
			prefetch_0!($dist, "{ax}", 7),
			prefetch_0!($dist, "{ax}", 8)
		)
	};
	(12, B, $dist:tt, $unroll:tt, 6) => {
		prefetch_0!($dist, "{ax}", 9)
	};

	(12, B, $dist:tt, $unroll:tt, 7) => {
		concat!(
			prefetch_0!($dist, "{ax}", 10),
			prefetch_0!($dist, "{ax}", 11)
		)
	};

	(8, B, $dist:tt, $unroll:tt, $i:tt) => {
		prefetch_0!($dist, "{ax}", $i)
	};

	(4, B, $dist:tt, $unroll:tt, 0) => {
		prefetch_0!($dist, "{ax}", 0)
	};
	(4, B, $dist:tt, $unroll:tt, 2) => {
		prefetch_0!($dist, "{ax}", 1)
	};
	(4, B, $dist:tt, $unroll:tt, 4) => {
		prefetch_0!($dist, "{ax}", 2)
	};
	(4, B, $dist:tt, $unroll:tt, 6) => {
		prefetch_0!($dist, "{ax}", 3)
	};


	(4, B, $dist:tt, $unroll:tt, $k_i:tt) => {
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
	($nr:tt, B, $dist:tt, $unroll:tt, 2) => {
		prefetch_0!($dist, "{bx}", 1)
	};
	// ($nr:tt, B, $dist:tt, $unroll:tt, 4) => {
	// 	prefetch_0!($dist, "{bx}", 2)
	// };
	($nr:tt, B, $dist:tt, $unroll:tt, $k_i:tt) => {
		""
	};
}

macro_rules! prefetch_c {
    (12, $nr:tt, $c:tt, $ldc:tt) => {
        seq!(j in 0..$nr {
            _mm_prefetch($c.add(0+j*$ldc) as *const i8, 3);
            _mm_prefetch($c.add(6+j*$ldc) as *const i8, 3);
			_mm_prefetch($c.add(12+j*$ldc) as *const i8, 3);
        });
    };
    (8, $nr:tt, $c:tt, $ldc:tt) => {
        seq!(j in 0..$nr {
            _mm_prefetch($c.add(4+j*$ldc) as *const i8, 3);
        });
    };
    (4, $nr:tt, $c:tt, $ldc:tt) => {
        seq!(j in 0..$nr {
            _mm_prefetch($c.add(0+j*$ldc) as *const i8, 3);
        });
    }
}

use crate::MyFn;
 
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
		#[target_feature(enable = "avx,fma")]
    	pub(crate) unsafe fn $func_name<F: MyFn>(
        	a: *const TA, b: *const TB, c: *mut TC,
        	alpha: *const TA, beta: *const TB,
        	k: usize,
        	ldc: usize,
			ld_arr: [usize; 2],
			f: F,
    	) {
        	let k_iter = k / $unroll;
        	let k_left = k % $unroll;
            let u64_arr = [ld_arr[0], ld_arr[1], ldc*8, k_iter, k_left];
        	let u64_ptr = u64_arr.as_ptr();
			let cf = c;
			// prefetch for c
			use std::arch::x86_64::_mm_prefetch;
			prefetch_c!($mr,$nr,c,ldc);
        	asm!(
            	asm_vzeroall!($VER,$nr),
   	 
            	asm_init_ab!($VER,$a_layout,$b_layout),
           	 
            	// 3 -> CONSIDKLEFT
            	"je 3f",
           	 
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
   	 
            	"dec {x0}",
            	// 2 -> KITER
            	"jne 2b",

            	// 3 -> CONSIDKLEFT
            	"3:",
            	"mov 32({int_arrx}),{x0}",
            	"test {x0},{x0}",

            	// 5 -> POSTACCUM
            	"je 5f",
   	 
            	// 4 -> KLEFT
            	"4:",
            	$asm_step_macro!($mr, $nr, $a_layout, $b_layout, 0),
            	inc_a!($a_layout),
            	inc_b!($b_layout,$nr),
            	inc_a_k_unroll!($a_layout, $mr, 1),
            	inc_b_k_unroll!($b_layout, $nr, 1),

            	"dec {x0}",
   	 
            	// 4 -> KLEFT
            	"jne 4b",
   	 
            	// 5 -> POSTACCUM
            	"5:",
            	asm_c_load!($nr),
            	// scale by alpha
            	asm_alpha_scale!($VER, $nr),

            	"vbroadcastsd ({betax}), %ymm0",

            	"vxorpd %ymm3,%ymm3,%ymm3",
            	"vucomisd %xmm3,%xmm0",

            	// 6 -> BETAZERO
            	"je 6f",
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
            	out("xmm0") _, out("xmm1") _, out("xmm2") _, out("xmm3") _,
            	out("xmm4") _, out("xmm5") _, out("xmm6") _, out("xmm7") _,
            	out("xmm8") _, out("xmm9") _, out("xmm10") _, out("xmm11") _,
            	out("xmm12") _, out("xmm13") _, out("xmm14") _, out("xmm15") _,
            	options(att_syntax)
        	);

			for j in 0..$nr {
				for i in 0..$mr/8 {
					f.call(c.add(i*8+j*ldc), 8);
				}
			}
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
		#[target_feature(enable = "avx,fma")]
    	pub(crate) unsafe fn $func_name<F: MyFn>(
        	a: *const TA, b: *const TB, c: *mut TC,
        	alpha: *const TA, beta: *const TB,
        	k: usize,
        	ldc: usize,
			ld_arr: [usize; 2],
			mask: *const u64,
			f: F,
    	) {
        	let k_iter = k / $unroll;
        	let k_left = k % $unroll;
            let u64_arr = [ld_arr[0], ld_arr[1], ldc*8, k_iter, k_left];
        	let u64_ptr = u64_arr.as_ptr();
			let cf = c;
			// prefetch for c
			use std::arch::x86_64::_mm_prefetch;
			prefetch_c!($mr,$nr,c,ldc);
        	asm!(
            	asm_vzeroall!($VER,$nr),
   	 
            	asm_init_ab!($VER,$a_layout,$b_layout),
           	 
            	// 3 -> CONSIDKLEFT
            	"je 3f",
           	 
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
   	 
            	"dec {x0}",
            	// 2 -> KITER
            	"jne 2b",

            	// 3 -> CONSIDKLEFT
            	"3:",
            	"mov 32({int_arrx}),{x0}",
            	"test {x0},{x0}",

            	// 5 -> POSTACCUM
            	"je 5f",
   	 
            	// 4 -> KLEFT
            	"4:",
            	$asm_step_macro!($mr, $nr, $a_layout, $b_layout, 0),
            	inc_a!($a_layout),
            	inc_b!($b_layout,$nr),
            	inc_a_k_unroll!($a_layout, $mr, 1),
            	inc_b_k_unroll!($b_layout, $nr, 1),

            	"dec {x0}",
   	 
            	// 4 -> KLEFT
            	"jne 4b",
   	 
            	// 5 -> POSTACCUM
            	"5:",
            	asm_c_load!($nr),
            	// scale by alpha
            	asm_alpha_scale!($VER, $nr),

            	"vbroadcastsd ({betax}), %ymm0",

            	"vxorpd %ymm3,%ymm3,%ymm3",
            	"vucomisd %xmm3,%xmm0",
				"vmovdqu ({maskx}), %ymm1",
				// "/* {maskx}*/",

            	// 6 -> BETAZERO
            	"je 6f",
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
            	out("xmm0") _, out("xmm1") _, out("xmm2") _, out("xmm3") _,
            	out("xmm4") _, out("xmm5") _, out("xmm6") _, out("xmm7") _,
            	out("xmm8") _, out("xmm9") _, out("xmm10") _, out("xmm11") _,
            	out("xmm12") _, out("xmm13") _, out("xmm14") _, out("xmm15") _,
            	options(att_syntax)
        	);

			for j in 0..$nr {
				for i in 0..$mr/8 {
					f.call(c.add(i*8+j*ldc), 8);
				}
			}
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
					0, 256,
					4,
					[<ukernel_ $mr x nr _ $func_name>]
				);
			}
		});
	};
}

group_def_ukernel!(12, 1, 4, B, B, bb, def_ukernel, asm_12x4_step, asm_12x4_acc, asm_12x4_store, VER24);
group_def_ukernel!(12, 1, 4, B, S, bs, def_ukernel, asm_12x4_step, asm_12x4_acc, asm_12x4_store, VER24);
group_def_ukernel!(12, 1, 4, B, B, bb_partial, def_ukernel_partial, asm_12x4_step, asm_12x4_acc, asm_12x4_store, VER24);
group_def_ukernel!(12, 1, 4, B, S, bs_partial, def_ukernel_partial, asm_12x4_step, asm_12x4_acc, asm_12x4_store, VER24);


group_def_ukernel!(8, 1, 4, B, B, bb, def_ukernel, asm_8x6_step, asm_8x6_acc, asm_8x6_store, VER16);
group_def_ukernel!(8, 1, 4, B, B, bb_partial, def_ukernel_partial, asm_8x6_step, asm_8x6_acc, asm_8x6_store, VER16);
group_def_ukernel!(8, 1, 4, B, S, bs_partial, def_ukernel_partial, asm_8x6_step, asm_8x6_acc, asm_8x6_store, VER16);

group_def_ukernel!(4, 1, 4, B, B, bb, def_ukernel, asm_4x6_step, asm_4x6_acc, asm_4x6_store, VER8);
group_def_ukernel!(4, 1, 4, B, B, bb_partial, def_ukernel_partial, asm_4x6_step, asm_4x6_acc, asm_4x6_store, VER8);
group_def_ukernel!(4, 1, 4, B, S, bs_partial, def_ukernel_partial, asm_4x6_step, asm_4x6_acc, asm_4x6_store, VER8);

