use seq_macro::seq;
use std::arch::asm;
use super::VS;
use crate::{TA, TB, TC};

macro_rules! beta_fmadd {
    (C, $m0:expr, $r1:expr) => {
        concat!(
            "vfmadd231ps ", $m0, ",%ymm0,%ymm", $r1, "\n",
        ) 
    };
    (M, $m0:expr, $r1:expr) => {
        concat!(
            "vmaskmovps ", $m0, ", %ymm1", ", %ymm2",  "\n",
            "vfmadd231ps %ymm2, %ymm0,%ymm", $r1, "\n",
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
macro_rules! vmovp {
    (B) => {
        "vmovaps "
    };
    ($layout:tt) => {
        "vmovups "
    };
 }

 macro_rules! vbroadcast {
	() => {
		"vbroadcastss"
	};
}

macro_rules! vfmadd {
    ($r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            "vfmadd231ps %ymm", $r1, ", %ymm", $r2,", %ymm", $r3, "\n",
        ) 
    };
}

macro_rules! loadp_unit {
    ($layout:tt, $m0:expr, $r1:expr) => {
        concat!(
            vmovp!($layout), $m0, ",%ymm", $r1, "\n",
        )
    };
    (4, $layout:tt, $m0:expr, $r1:expr) => {
        concat!(
            vmovp!($layout), $m0, ",", "%xmm", $r1, "\n",
        )
    };
 }

 macro_rules! storep_unit {
    (C, $r1:expr, $m0:expr) => {
        concat!(
            "vmovups %ymm", $r1, ", ", $m0,  "\n",
        )
    };
    (B, $r1:expr, $m0:expr) => {
        concat!(
            "vmovaps %ymm", $r1, ", ", $m0,  "\n",
        )
    };
    (M, $r1:expr, $m0:expr) => {
        concat!(
            "vmaskmovps %ymm", $r1, ", %ymm1, ", $m0,  "\n",
        )
    };
}

macro_rules! asm_alpha_scale_0 {
	($r0:tt, $r1:tt) => {
		seq!(r in $r0..=$r1 {
			concat!(
				vbroadcast!(), " ({alphax}),%ymm1", "\n",
				#(
					"vmulps %ymm1, %ymm", r, ",%ymm", r, "\n",
				)*
			)
		})
	}
}

macro_rules! load_beta {
	() => {
		concat!(
			vbroadcast!(), " ({betax}), %ymm0\n",
			"vxorps %ymm3,%ymm3,%ymm3\n",
			"vucomiss %xmm3,%xmm0\n",
		)
	}
}

macro_rules! mem {
    ($m0:tt, $b0:tt) => {
        concat!($b0, "+", $m0)
    };
}

 macro_rules! acc_p {
    (24, $layout:tt, $m0:expr, $r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            beta_fmadd!(C, $m0, $r1),
            beta_fmadd!(C, mem!($m0, "0x20"), $r2),
            beta_fmadd!($layout, mem!($m0, "0x40"), $r3),
        )
    };
    (20, $layout:tt, $m0:expr, $r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            beta_fmadd!(C, $m0, $r1),
            beta_fmadd!(C, mem!($m0, "0x20"), $r2),
            beta_fmadd!($layout, mem!($m0, "0x40"), $r3),
        )
    };
    (16, $layout:tt, $m0:expr, $r1:expr, $r2:expr) => {
        concat!(
            beta_fmadd!(C, $m0, $r1),
            beta_fmadd!($layout, mem!($m0, "0x20"), $r2),
        )
    };
    (12, $layout:tt, $m0:expr, $r1:expr, $r2:expr) => {
        concat!(
            beta_fmadd!(C, $m0, $r1),
            beta_fmadd!($layout, mem!($m0, "0x20"), $r2),
        )
    };
    ($N:tt, $layout:tt, $m0:expr, $r1:expr) => {
        concat!(
            beta_fmadd!($layout, $m0, $r1),
        )
    };
 }



 macro_rules! loadp {
    (24, $layout:tt, $m0:expr, $r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            loadp_unit!($layout, $m0, $r1),
            loadp_unit!($layout, mem!($m0, "0x20"), $r2),
            loadp_unit!($layout, mem!($m0, "0x40"), $r3),
        )
    };
    (24, $layout:tt, $m0:expr) => {
        concat!(
            loadp_unit!($layout, $m0, 0),
            loadp_unit!($layout, mem!($m0, "0x20"), 1),
            loadp_unit!($layout, mem!($m0, "0x40"), 2),
        )
    };
    (16, $layout:tt, $m0:expr, $r1:expr, $r2:expr) => {
        concat!(
            loadp_unit!($layout, $m0, $r1),
            loadp_unit!($layout, mem!($m0, "0x20"), $r2),
        )
    };
    (16, $layout:tt, $m0:expr) => {
        concat!(
            loadp_unit!($layout, $m0, 0),
            loadp_unit!($layout, mem!($m0, "0x20"), 1),
        )
    };
    (8, $layout:tt, $m0:expr) => {
        concat!(
            loadp_unit!($layout, $m0, 0),
        )
    };
    (8, $layout:tt, $m0:expr, $r1:expr) => {
        concat!(
            loadp_unit!($layout, $m0, $r1),
        )
    };
 }


macro_rules! storep {
	(24, $layout:tt, $m0:expr, $r1:expr, $r2:expr, $r3:expr) => {
		concat!(
			storep_unit!(C, $r1, $m0),
			storep_unit!(C, $r2, mem!($m0, "0x20")),
			storep_unit!($layout, $r3, mem!($m0, "0x40")),
		)
	};
	(16, $layout:tt, $m0:expr, $r1:expr, $r2:expr) => {
		concat!(
			storep_unit!(C, $r1, $m0),
			storep_unit!($layout, $r2, mem!($m0, "0x20")),
		)
	};
	(8, $layout:tt, $m0:expr, $r1:expr) => {
		concat!(
			storep_unit!($layout, $r1, $m0),
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
        	"mov 24({dim_arrx}),{x0}", "\n",
        	"test {x0},{x0}", "\n",
    	)
	};
	($ker:tt,B,S) => {
    	concat!(
        	// mov cs_b to reg
			"mov ({dim_arrx}), {x1}", "\n",
        	"mov 8({dim_arrx}), {x2}", "\n",
        	"lea ({x2}, {x2}, 2), {x3}", "\n",
        	"lea ({bx}, {x3}, 1), {x3}", "\n",

        	"mov 24({dim_arrx}),{x0}", "\n",
        	"test {x0},{x0}", "\n",
    	)
	};
}


macro_rules! asm_c_load {
	(6) => {
    	concat!(
        	"mov 16({dim_arrx}),{x0}", "\n",
        	"lea ({x0}, {x0}, 2), {x1}", "\n",
        	"lea ({cx}, {x1},), {x1}", "\n",
    	)
	};
	(5) => {
    	concat!(
        	"mov 16({dim_arrx}),{x0}", "\n",
        	"lea ({x0}, {x0}, 2), {x1}", "\n",
        	"lea ({cx}, {x1},), {x1}", "\n",
    	)
	};
	(4) => {
    	concat!(
        	"mov 16({dim_arrx}),{x0}", "\n",
        	"lea ({x0}, {x0}, 2), {x1}", "\n",
        	"lea ({cx}, {x1},), {x1}", "\n",
    	)
	};
	(3) => {
    	concat!(
        	"mov 16({dim_arrx}),{x0}", "\n",
    	)
	};
	(2) => {
    	concat!(
        	"mov 16({dim_arrx}),{x0}", "\n",
    	)
	};
	(1) => {
    	concat!(
        	"mov 16({dim_arrx}),{x0}", "\n",
    	)
	};
}


macro_rules! asm_vzeroall {

	(VER3,4) => {vzeroall!(4,15)};
	(VER3,3) => {vzeroall!(4,12)};
	(VER3,2) => {vzeroall!(4,9)};
	(VER3,1) => {vzeroall!(4,6)};

	(VER2,6) => {vzeroall!(4,15)};
	(VER2,5) => {vzeroall!(4,13)};
	(VER2,4) => {vzeroall!(4,11)};
	(VER2,3) => {vzeroall!(4,9)};
	(VER2,2) => {vzeroall!(4,7)};
	(VER2,1) => {vzeroall!(4,5)};

	(VER1,6) => {vzeroall!(7,12)};
	(VER1,5) => {vzeroall!(7,11)};
	(VER1,4) => {vzeroall!(7,10)};
	(VER1,3) => {vzeroall!(7,9)};
	(VER1,2) => {vzeroall!(7,8)};
	(VER1,1) => {vzeroall!(7,7)};
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
        	"add $4*", $K, "*", $X, ",{ax}", "\n",
    	)
	};
}

macro_rules! inc_b_k_unroll {
	(S, $X:tt, $K:tt) => {
    	""
	};
	(B, $X:tt, $K:tt) => {
    	concat!(
        	"add $4*", $K, "*", $X, ", {bx}", "\n",
    	)
	};
}


macro_rules! asm_alpha_scale {
	(VER2, 6) => {
		asm_alpha_scale_0!(4,15)
	};
	(VER2, 5) => {
    	asm_alpha_scale_0!(4,13)
	};
	(VER2, 4) => {
    	asm_alpha_scale_0!(4,11)
	};
	(VER2, 3) => {
    	asm_alpha_scale_0!(4,9)
	};
	(VER2, 2) => {
    	asm_alpha_scale_0!(4,7)
	};
	(VER2, 1) => {
    	asm_alpha_scale_0!(4,5)
	};

	(VER3, 4) => {
    	asm_alpha_scale_0!(4,15)
	};
	(VER3, 3) => {
    	asm_alpha_scale_0!(4,12)
	};
	(VER3, 2) => {
    	asm_alpha_scale_0!(4,9)
	};
	(VER3, 1) => {
    	asm_alpha_scale_0!(4,6)
	};

	(VER1, 6) => {
    	asm_alpha_scale_0!(7,12)
	};
	(VER1, 5) => {
		asm_alpha_scale_0!(7,11)
	};
	(VER1, 4) => {
    	asm_alpha_scale_0!(7,10)
	};
	(VER1, 3) => {
    	asm_alpha_scale_0!(7,9)
	};
	(VER1, 2) => {
    	asm_alpha_scale_0!(7,8)
	};
	(VER1, 1) => {
    	asm_alpha_scale_0!(7,7)
	};
}


macro_rules! acc_24x4_seq {
	($mr:tt, 0, $layout:tt) => {
		acc_p!($mr, $layout, "0({cx})", 4, 5, 6)
	};
	($mr:tt, 1, $layout:tt) => {
		acc_p!($mr, $layout, "0({cx}, {x0})", 7, 8, 9)
	};
	($mr:tt, 2, $layout:tt) => {
		acc_p!($mr, $layout, "0({cx}, {x0}, 2)",  10, 11, 12)
	}; 
	($mr:tt, 3, $layout:tt) => {
		acc_p!($mr, $layout, "0({x1})",  13, 14, 15)
	};
}

macro_rules! store_24x4_seq {
	($mr:tt, 0, $layout:tt) => {
		storep!($mr, $layout, "0({cx})", 4, 5, 6)
	};
	($mr:tt, 1, $layout:tt) => {
		storep!($mr, $layout, "0({cx}, {x0})", 7, 8, 9)
	};
	($mr:tt, 2, $layout:tt) => {
		storep!($mr, $layout, "0({cx}, {x0}, 2)",  10, 11, 12)
	}; 
	($mr:tt, 3, $layout:tt) => {
		storep!($mr, $layout, "0({x1})",  13, 14, 15)
	};
}

macro_rules! acc_16x6_seq {
	($mr:tt, 0, $layout:tt) => {
		acc_p!($mr, $layout, "0({cx})", 4, 5)
	};
	($mr:tt, 1, $layout:tt) => {
		acc_p!($mr, $layout, "0({cx}, {x0})", 6, 7)
	};
	($mr:tt, 2, $layout:tt) => {
		acc_p!($mr, $layout, "0({cx}, {x0}, 2)", 8, 9)
	}; 
	($mr:tt, 3, $layout:tt) => {
		acc_p!($mr, $layout, "0({x1})", 10, 11)
	};
	($mr:tt, 4, $layout:tt) => {
        acc_p!($mr, $layout, "0({x1}, {x0})", 12, 13)
	};
	($mr:tt, 5, $layout:tt) => {
		acc_p!($mr, $layout, "0({x1}, {x0}, 2)", 14, 15)
	};
}

macro_rules! store_16x6_seq {
	($mr:tt, 0, $layout:tt) => {
		storep!($mr, $layout, "0({cx})", 4, 5)
	};
	($mr:tt, 1, $layout:tt) => {
		storep!($mr, $layout, "0({cx}, {x0})", 6, 7)
	};
	($mr:tt, 2, $layout:tt) => {
		storep!($mr, $layout, "0({cx}, {x0}, 2)", 8, 9)
	}; 
	($mr:tt, 3, $layout:tt) => {
		storep!($mr, $layout, "0({x1})", 10, 11)
	};
	($mr:tt, 4, $layout:tt) => {
    	storep!($mr, $layout, "0({x1}, {x0})", 12, 13)
	};
	($mr:tt, 5, $layout:tt) => {
		storep!($mr, $layout, "0({x1}, {x0}, 2)", 14, 15)
	};
}

macro_rules! acc_8x6_seq {
	($mr:tt, 0, $layout:tt) => {
		acc_p!($mr, $layout, "0({cx})", 7)
	};
	($mr:tt, 1, $layout:tt) => {
		acc_p!($mr, $layout, "0({cx}, {x0})", 8)
	};
	($mr:tt, 2, $layout:tt) => {
		acc_p!($mr, $layout, "0({cx}, {x0}, 2)", 9)
	}; 
	($mr:tt, 3, $layout:tt) => {
		acc_p!($mr, $layout, "0({x1})", 10)
	};
	($mr:tt, 4, $layout:tt) => {
        acc_p!($mr, $layout, "0({x1}, {x0})", 11)
	};
	($mr:tt, 5, $layout:tt) => {
		acc_p!($mr, $layout, "0({x1}, {x0}, 2)", 12)
	};
}

macro_rules! store_8x6_seq {
	($mr:tt, 0, $layout:tt) => {
		storep!($mr, $layout, "0({cx})", 7)
	};
	($mr:tt, 1, $layout:tt) => {
		storep!($mr, $layout, "0({cx}, {x0})", 8)
	};
	($mr:tt, 2, $layout:tt) => {
		storep!($mr, $layout, "0({cx}, {x0}, 2)", 9)
	}; 
	($mr:tt, 3, $layout:tt) => {
		storep!($mr, $layout, "0({x1})", 10)
	};
	($mr:tt, 4, $layout:tt) => {
        storep!($mr, $layout, "0({x1}, {x0})", 11)
	};
	($mr:tt, 5, $layout:tt) => {
		storep!($mr, $layout, "0({x1}, {x0}, 2)", 12)
	};
}

macro_rules! acc_24x4 {
	($mr:tt, $nr:tt, $layout:tt) => {
		seq!(n in 0..$nr {
			concat!(
				#(
					acc_24x4_seq!($mr, n, $layout), 
				)*
			)
		})
	};
}


macro_rules! store_24x4 {
	($mr:tt, $nr:tt, $layout:tt) => {
		seq!(n in 0..$nr {
			concat!(
				#(
					store_24x4_seq!($mr, n, $layout), 
				)*
			)
		})
	};
}


macro_rules! acc_16x6 {
	($mr:tt, $nr:tt, $layout:tt) => {
		seq!(n in 0..$nr {
			concat!(
				#(
					acc_16x6_seq!($mr, n, $layout), 
				)*
			)
		})
	};
}

macro_rules! store_16x6 {
	($mr:tt, $nr:tt, $layout:tt) => {
		seq!(n in 0..$nr {
			concat!(
				#(
					store_16x6_seq!($mr, n, $layout), 
				)*
			)
		})
	};
}


macro_rules! acc_8x6 {
	($mr:tt, $nr:tt, $layout:tt) => {
		seq!(n in 0..$nr {
			concat!(
				#(
					acc_8x6_seq!($mr, n, $layout), 
				)*
			)
		})
	};
}


macro_rules! store_8x6 {
	($mr:tt, $nr:tt, $layout:tt) => {
		seq!(n in 0..$nr {
			concat!(
				#(
					store_8x6_seq!($mr, n, $layout), 
				)*
			)
		})
	};
}

macro_rules! load_b {
	(S, 0, $K:tt, $X:tt, $r:expr) => {
    	concat!(
        	vbroadcast!(), " ({bx}),%ymm", $r, "\n",
    	)
	};
	(S, 1, $K:tt, $X:tt, $r:expr) => {
    	concat!(
			"prefetcht0 64({bx},{x2},1) \n",
        	vbroadcast!(), " ({bx},{x2},1),%ymm", $r, "\n",
    	)
	};
	(S, 2, $K:tt, $X:tt, $r:expr) => {
    	concat!(
        	vbroadcast!(), " ({bx},{x2},2),%ymm", $r, "\n",
    	)
	};
	(S, 3, $K:tt, $X:tt, $r:expr) => {
    	concat!(
			"prefetcht0 64({x3}) \n",
        	vbroadcast!(), " ({x3}),%ymm", $r, "\n",
    	)
	};
	(S, 4, $K:tt, $X:tt, $r:expr) => {
    	concat!(
        	vbroadcast!(), " ({x3},{x2},1),%ymm", $r, "\n",
    	)
	};
	(S, 5, $K:tt, $X:tt, $r:expr) => {
    	concat!(
        	vbroadcast!(), " ({x3},{x2},2),%ymm", $r, "\n",
    	)
	};
	(B, $N:tt, $K:tt, $X:tt, $r:expr) => {
    	concat!(
        	vbroadcast!(), " ", $K, "*", $X, "*4+", $N, "*4({bx}), %ymm", $r, "\n",
    	)
	};
}


macro_rules! load_a {
	($mr:tt, B, $K:tt) => {
    	loadp!($mr, B, concat!($mr,"*4*",$K,"({ax})"))
	};
	($mr:tt, C, $K:tt) => {
    	loadp!($mr, C, "0({ax})")
	};
}

macro_rules! fmadd_3v {
	(0) => {
		concat!(
			vfmadd!(0, 3, 4),
			vfmadd!(1, 3, 5),
			vfmadd!(2, 3, 6),
		)
	};
	(1) => {
		concat!(
			vfmadd!(0, 3, 7),
			vfmadd!(1, 3, 8),
			vfmadd!(2, 3, 9),
		)
	};
	(2) => {
		concat!(
			vfmadd!(0, 3, 10),
			vfmadd!(1, 3, 11),
			vfmadd!(2, 3, 12),
		)
	};
	(3) => {
		concat!(
			vfmadd!(0, 3, 13),
			vfmadd!(1, 3, 14),
			vfmadd!(2, 3, 15),
		)
	};
}

macro_rules! fmadd_2v {
	(0) => {
		concat!(
			vfmadd!(0, 2, 4),
			vfmadd!(1, 2, 5),
		)
	};
	(1) => {
		concat!(
			vfmadd!(0, 3, 6),
			vfmadd!(1, 3, 7),
		)
	};
	(2) => {
		concat!(
			vfmadd!(0, 2, 8),
			vfmadd!(1, 2, 9),
		)
	};
	(3) => {
		concat!(
			vfmadd!(0, 3, 10),
			vfmadd!(1, 3, 11),
		)
	};
	(4) => {
		concat!(
			vfmadd!(0, 2, 12),
			vfmadd!(1, 2, 13),
		)
	};
	(5) => {
		concat!(
			vfmadd!(0, 3, 14),
			vfmadd!(1, 3, 15),
		)
	};
}

macro_rules! fmadd_1v {
	(0) => {
		concat!(vfmadd!(0, 1, 7))
	};
	(1) => {
		concat!(vfmadd!(0, 2, 8))
	};
	(2) => {
		concat!(vfmadd!(0, 3, 9))
	};
	(3) => {
		concat!(vfmadd!(0, 4, 10))
	};
	(4) => {
		concat!(vfmadd!(0, 5, 11))
	};
	(5) => {
		concat!(vfmadd!(0, 6, 12))
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
macro_rules! step_24x4 {
	($N:tt, $nr:tt, $a_layout:tt, $b_layout:tt, $K:tt) => {
		seq!(n in 0..$nr {
			concat!(
				load_a!($N, $a_layout, $K),
				#(
					load_b!($b_layout, n, $K, $nr, 3),
					fmadd_3v!(n),
				)*
				inc_b!($b_layout,$nr), 
			)
		})
	};
}

// ***************************** 16x6 ******************************* //
macro_rules! step_16x6 {
	($N:tt, $nr:tt, $a_layout:tt, $b_layout:tt, $K:tt) => {
		seq!(n in 0..$nr {
			concat!(
				load_a!($N, $a_layout, $K),
				#(
					load_b!($b_layout, n, $K, $nr, b_num_16x6!(n)),
					fmadd_2v!(n),
				)*
				inc_b!($b_layout,$nr), 
			)
		})
	};
}

// ***************************** 8x6 ******************************* //
macro_rules! step_8x6 {
	($N:tt, $nr:tt, $a_layout:tt, $b_layout:tt, $K:tt) => {
		seq!(n in 0..$nr {
			concat!(
				load_a!($N, $a_layout, $K),
				#(
					load_b!($b_layout, n, $K, $nr, b_num_8x6!(n)),
					fmadd_1v!(n),
				)*
				inc_b!($b_layout,$nr), 
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

macro_rules! prefetch_c {
    (24, $nr:tt, $c:tt, $ldc:tt) => {
        seq!(j in 0..$nr {
            _mm_prefetch($c.add(0+j*$ldc) as *const i8, 3);
            _mm_prefetch($c.add(12+j*$ldc) as *const i8, 3);
			_mm_prefetch($c.add(23+j*$ldc) as *const i8, 3);
        });
    };
    (16, $nr:tt, $c:tt, $ldc:tt) => {
        seq!(j in 0..$nr {
            _mm_prefetch($c.add(16+j*$ldc) as *const i8, 3);
        });
    };
    (8, $nr:tt, $c:tt, $ldc:tt) => {
        seq!(j in 0..$nr {
            _mm_prefetch($c.add(8+j*$ldc) as *const i8, 3);
        });
    }
}

use crate::MyFn;

#[inline(always)]
fn mask_and_offset(m: usize) -> ([u32;16], usize) {
	let mask: [u32; 16] = [
		u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX,
		0, 0, 0, 0, 0, 0, 0, 0,
	];
	let mask_offset = if m % VS == 0 { 0 } else { VS - (m %VS)};

	(mask, mask_offset)
}



macro_rules! mask_ptr {
	(M, $m:tt, $nm:ident) => {
		let (mask, mask_offset) = mask_and_offset($m);
		let $nm = mask.as_ptr().add(mask_offset);
	};
	(C, $m:tt, $nm:ident) => {
		let mask = [0xFFFF_u32];
		let $nm = mask.as_ptr();
	};
}

macro_rules! load_mask_ptr_asm {
	(M) => {
		"vmovdqu ({maskx}), %ymm1"
	};
	(C) => {
		"/* {maskx} */"
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
		$is_partial:tt,
		$unroll:tt,
    	$func_name:ident
	) => {
    	pub(crate) unsafe fn $func_name<F: MyFn, const BUF: bool>(
        	a: *const TA, b: *const TB, c: *mut TC,
        	alpha: *const TA, beta: *const TB,
        	k: usize,
			ld_arr: [usize; 4],
			m: usize, _n: usize,
			f: F,
    	) {
			mask_ptr!($is_partial, m, x);
			let mask_ptr = x;
			let k_iter = k / $unroll;
        	let k_left = k % $unroll;
            let mut dim_arr = [ld_arr[0]*4, ld_arr[1]*4, ld_arr[3]*4, k_iter, k_left];
			let mut cf = c;
			let mut c_buf = [0f32;$mr*$nr];
			let c_cs = ld_arr[3];
			if BUF {
				let c_rs = ld_arr[2];
				if m != $mr || c_rs != 1 {
					for j in 0..$nr {
						for i in 0..m {
							c_buf[j*$mr+i] = *c.add(i*c_rs+j*c_cs);
						}
					}
					cf = c_buf.as_mut_ptr();
					dim_arr[2] = $mr*4;
				}
			}
			// prefetch for c
			use std::arch::x86_64::_mm_prefetch;
			prefetch_c!($mr,$nr,c,c_cs);
        	asm!(
            	asm_vzeroall!($VER,$nr),
   	 
            	asm_init_ab!($VER,$a_layout,$b_layout),
           	 
            	// 3 -> CONSIDKLEFT
            	"je 3f",
           	 
            	// 2 -> KITER
            	"2:",
				prefetch_0!(128, "{bx}", 0),
				$asm_step_macro!($mr, $nr, $a_layout, $b_layout, 0),
				$asm_step_macro!($mr, $nr, $a_layout, $b_layout, 1),
				$asm_step_macro!($mr, $nr, $a_layout, $b_layout, 2),
				$asm_step_macro!($mr, $nr, $a_layout, $b_layout, 3),

            	inc_a_k_unroll!($a_layout, $mr, $unroll),
            	inc_b_k_unroll!($b_layout, $nr, $unroll),
   	 
            	"dec {x0}",
            	// 2 -> KITER
            	"jne 2b",

            	// 3 -> CONSIDKLEFT
            	"3:",
            	"mov 32({dim_arrx}),{x0}",
            	"test {x0},{x0}",

            	// 5 -> POSTACCUM
            	"je 5f",
            	// 4 -> KLEFT
            	"4:",
            	$asm_step_macro!($mr, $nr, $a_layout, $b_layout, 0),
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

            	load_beta!(),

				load_mask_ptr_asm!($is_partial),				
            	// 6 -> BETAZERO
            	"je 6f",
            	$asm_acc_macro!($mr,$nr,$is_partial),

            	// 6 -> BETAZERO
            	"6:",
            	$asm_store_macro!($mr,$nr,$is_partial),
				
            	// 7 -> DDONE
            	"7:",
				// "vzeroupper",
            	ax = inout(reg) a => _,
            	bx = inout(reg) b => _,
            	cx = inout(reg) cf => _,
            	dim_arrx = inout(reg) dim_arr.as_ptr() => _,
            	alphax = inout(reg) alpha => _,
            	betax = inout(reg) beta => _,
				maskx = inout(reg) mask_ptr => _,
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
			if BUF {
				let c_rs = ld_arr[2];
				if m != $mr || c_rs != 1 {
					for j in 0..$nr {
						for i in 0..m {
							*c.add(i*c_rs+j*c_cs) = c_buf[j*$mr+i];
						}
					}
				}
			}
			for j in 0..$nr {
				for i in 0..$mr/8 {
					f.call(c.add(i*8+j*c_cs), 8);
				}
			}
    	}
	};
}

macro_rules! def_ukernelxn {
	(
		$VER:tt,
		$asm_step_macro:tt,
		$asm_acc_macro:tt,
		$asm_store_macro:tt,
    	$mr:tt, $nr:tt,
    	$a_layout:tt, $b_layout:tt,
		$is_partial:tt,
		$unroll:tt,
    	$func_name:ident
	) => {
    	pub(crate) unsafe fn $func_name<F: MyFn, const BUF: bool>(
        	a: *const TA, b: *const TB, c: *mut TC,
        	alpha: *const TA, beta: *const TB,
        	k: usize,
			d_arr: [usize; 4],
			m: usize, n: usize,
			f: F,
    	) {
			mask_ptr!($is_partial, m, x);
			let mask_ptr = x;
			let k_iter = k / $unroll;
        	let k_left = k % $unroll;
            let mut dim_arr = [d_arr[0]*4, d_arr[1]*4, d_arr[3]*4, k_iter, k_left];
			let mut cf = c;
			let mut c_buf = [0f32;$mr*$nr];
			let c_cs = d_arr[3];
			if BUF {
				let c_rs = d_arr[2];
				if m != $mr || c_rs != 1 {
					for j in 0..n {
						for i in 0..m {
							c_buf[j*$mr+i] = *c.add(i*c_rs+j*c_cs);
						}
					}
					cf = c_buf.as_mut_ptr();
					dim_arr[2] = $mr*4;
				}
			}
			use std::arch::x86_64::_mm_prefetch;
			let _ = 'blk: {
				seq!(ni in 1..$nr {
					if ni == n {
						// prefetch for c
						prefetch_c!($mr,ni,c,c_cs);
						asm!(
							asm_vzeroall!($VER,ni),
				
							asm_init_ab!($VER,$a_layout,$b_layout),
						
							// 3 -> CONSIDKLEFT
							"je 3f",
						
							// 2 -> KITER
							"2:",
							prefetch_0!(128, "{bx}", 0),
							$asm_step_macro!($mr, ni, $a_layout, $b_layout, 0),
							$asm_step_macro!($mr, ni, $a_layout, $b_layout, 1),
							$asm_step_macro!($mr, ni, $a_layout, $b_layout, 2),
							$asm_step_macro!($mr, ni, $a_layout, $b_layout, 3),
			
							inc_a_k_unroll!($a_layout, $mr, $unroll),
							inc_b_k_unroll!($b_layout, ni, $unroll),
				
							"dec {x0}",
							// 2 -> KITER
							"jne 2b",

							// 3 -> CONSIDKLEFT
							"3:",
							"mov 32({dim_arrx}),{x0}",
							"test {x0},{x0}",

							// 5 -> POSTACCUM
							"je 5f",
							// 4 -> KLEFT
							"4:",
							$asm_step_macro!($mr, ni, $a_layout, $b_layout, 0),
							inc_a_k_unroll!($a_layout, $mr, 1),
							inc_b_k_unroll!($b_layout, ni, 1),

							"dec {x0}",
				
							// 4 -> KLEFT
							"jne 4b",
				
							// 5 -> POSTACCUM
							"5:",
							asm_c_load!(ni),
							// scale by alpha
							asm_alpha_scale!($VER, ni),

							load_beta!(),

							load_mask_ptr_asm!($is_partial),				
							// 6 -> BETAZERO
							"je 6f",
							$asm_acc_macro!($mr,ni,$is_partial),

							// 6 -> BETAZERO
							"6:",
							$asm_store_macro!($mr,ni,$is_partial),
							
							// 7 -> DDONE
							"7:",
							// "vzeroupper",
							ax = inout(reg) a => _,
							bx = inout(reg) b => _,
							cx = inout(reg) cf => _,
							dim_arrx = inout(reg) dim_arr.as_ptr() => _,
							alphax = inout(reg) alpha => _,
							betax = inout(reg) beta => _,
							maskx = inout(reg) mask_ptr => _,
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
						break 'blk;
					}
				});
			};
			if BUF {
				let c_rs = d_arr[2];
				if m != $mr || c_rs != 1 {
					for j in 0..n {
						for i in 0..m {
							*c.add(i*c_rs+j*c_cs) = c_buf[j*$mr+i];
						}
					}
				}
			}
			for j in 0..n {
				for i in 0..$mr/8 {
					f.call(c.add(i*8+j*c_cs), 8);
				}
			}
    	}
	};
}

def_ukernel!(VER3, step_24x4, acc_24x4, store_24x4, 24, 4, B, B, C, 4, ukernel_24x4_bb);
// def_ukernel!(VER2, step_16x6, acc_16x6, store_16x6, 16, 4, B, B, C, 4, ukernel_32x8_bb);
// def_ukernel!(VER1, step_8x6, acc_8x6, store_8x6, 8, 4, B, B, C, 4, ukernel_16x8_bb);

def_ukernel!(VER3, step_24x4, acc_24x4, store_24x4, 24, 4, B, B, M, 4, ukernel_24x4_bb_partial);
def_ukernel!(VER2, step_16x6, acc_16x6, store_16x6, 16, 4, B, B, M, 4, ukernel_16x4_bb_partial);
def_ukernel!(VER1, step_8x6, acc_8x6, store_8x6, 8, 4, B, B, M, 4, ukernel_8x4_bb_partial);

def_ukernel!(VER3, step_24x4, acc_24x4, store_24x4, 24, 4, B, S, C, 4, ukernel_24x4_bs);

def_ukernel!(VER3, step_24x4, acc_24x4, store_24x4, 24, 4, B, S, M, 4, ukernel_24x4_bs_partial);
def_ukernel!(VER2, step_16x6, acc_16x6, store_16x6, 16, 4, B, S, M, 4, ukernel_16x4_bs_partial);
def_ukernel!(VER1, step_8x6, acc_8x6, store_8x6, 8, 4, B, S, M, 4, ukernel_8x4_bs_partial);


def_ukernelxn!(VER3, step_24x4, acc_24x4, store_24x4, 24, 4, B, B, C, 4, ukernel_24xn_bb);
// def_ukernelxn!(VER2, step_16x6, acc_16x6, store_16x6, 16, 4, B, B, C, 4, ukernel_16xn_bb);
// def_ukernelxn!(VER1, step_8x6, acc_8x6, store_8x6, 8, 4, B, B, C, 4, ukernel_16xn_bb);

def_ukernelxn!(VER3, step_24x4, acc_24x4, store_24x4, 24, 4, B, B, M, 4, ukernel_24xn_bb_partial);
def_ukernelxn!(VER2, step_16x6, acc_16x6, store_16x6, 16, 4, B, B, M, 4, ukernel_16xn_bb_partial);
def_ukernelxn!(VER1, step_8x6, acc_8x6, store_8x6, 8, 4, B, B, M, 4, ukernel_8xn_bb_partial);

def_ukernelxn!(VER3, step_24x4, acc_24x4, store_24x4, 24, 4, B, S, C, 4, ukernel_24xn_bs);

def_ukernelxn!(VER3, step_24x4, acc_24x4, store_24x4, 24, 4, B, S, M, 4, ukernel_24xn_bs_partial);
def_ukernelxn!(VER2, step_16x6, acc_16x6, store_16x6, 16, 4, B, S, M, 4, ukernel_16xn_bs_partial);
def_ukernelxn!(VER1, step_8x6, acc_8x6, store_8x6, 8, 4, B, S, M, 4, ukernel_8xn_bs_partial);


// group_def_ukernel!(48, 1, 7, B, B, C, bb, step_24x4, acc_24x4, store_24x4, VER3);
// group_def_ukernel!(48, 1, 1, B, S, C, bs, step_24x4, acc_24x4, store_24x4, VER3);
// group_def_ukernel!(48, 1, 8, B, B, M, bb_partial, step_24x4, acc_24x4, store_24x4, VER3);
// group_def_ukernel!(48, 1, 8, B, S, M, bs_partial, step_24x4, acc_24x4, store_24x4, VER3);

// // group_def_ukernel!(32, 1, 8, B, B, bb, def_ukernel, step_16x6, acc_16x6, store_16x6, VER2);
// // group_def_ukernel!(32, 1, 8, B, S, bs, def_ukernel, step_16x6, acc_16x6, store_16x6, VER2);
// group_def_ukernel!(32, 1, 8, B, B, M, bb_partial, step_16x6, acc_16x6, store_16x6, VER2);
// group_def_ukernel!(32, 1, 8, B, S, M, bs_partial, step_16x6, acc_16x6, store_16x6, VER2);


// // group_def_ukernel!(16, 1, 8, B, B, bb, def_ukernel, step_8x6, acc_8x6, store_8x6, VER1);
// // group_def_ukernel!(16, 1, 8, B, S, bs, def_ukernel, step_8x6, acc_8x6, store_8x6, VER1);
// group_def_ukernel!(16, 1, 8, B, B, M, bb_partial, step_8x6, acc_8x6, store_8x6, VER1);
// group_def_ukernel!(16, 1, 8, B, S, M, bs_partial, step_8x6, acc_8x6, store_8x6, VER1);
