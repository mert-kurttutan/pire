use seq_macro::seq;
use std::arch::asm;

use crate::{TA, TB, TC};

use paste::paste;
macro_rules! beta_fmaddps {
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

macro_rules! mem {
    ($m0:tt, $b0:tt) => {
        concat!($b0, "+", $m0)
    };
 }
macro_rules! vmovps {
    (B) => {
        "vmovaps "
    };
    ($layout:tt) => {
        "vmovups "
    };
 }
 macro_rules! acc_ps {
    (24, $layout:tt, $m0:expr, $r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            beta_fmaddps!(C, $m0, $r1),
            beta_fmaddps!(C, mem!($m0, "0x20"), $r2),
            beta_fmaddps!($layout, mem!($m0, "0x40"), $r3),
        )
    };
    (20, $layout:tt, $m0:expr, $r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            beta_fmaddps!(C, $m0, $r1),
            beta_fmaddps!(C, mem!($m0, "0x20"), $r2),
            beta_fmaddps!($layout, mem!($m0, "0x40"), $r3),
        )
    };
    (16, $layout:tt, $m0:expr, $r1:expr, $r2:expr) => {
        concat!(
            beta_fmaddps!(C, $m0, $r1),
            beta_fmaddps!($layout, mem!($m0, "0x20"), $r2),
        )
    };
    (12, $layout:tt, $m0:expr, $r1:expr, $r2:expr) => {
        concat!(
            beta_fmaddps!(C, $m0, $r1),
            beta_fmaddps!($layout, mem!($m0, "0x20"), $r2),
        )
    };
    ($N:tt, $layout:tt, $m0:expr, $r1:expr) => {
        concat!(
            beta_fmaddps!($layout, $m0, $r1),
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

macro_rules! loadps_unit {
    ($layout:tt, $m0:expr, $r1:expr) => {
        concat!(
            vmovps!($layout), $m0, ",%ymm", $r1, "\n",
        )
    };
    (4, $layout:tt, $m0:expr, $r1:expr) => {
        concat!(
            vmovps!($layout), $m0, ",", "%xmm", $r1, "\n",
        )
    };
 }
 macro_rules! loadps {
    (24, $layout:tt, $m0:expr, $r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            loadps_unit!($layout, $m0, $r1),
            loadps_unit!($layout, mem!($m0, "0x20"), $r2),
            loadps_unit!($layout, mem!($m0, "0x40"), $r3),
        )
    };
    (24, $layout:tt, $m0:expr) => {
        concat!(
            loadps_unit!($layout, $m0, 0),
            loadps_unit!($layout, mem!($m0, "0x20"), 1),
            loadps_unit!($layout, mem!($m0, "0x40"), 2),
        )
    };
    (20, $layout:tt, $m0:expr,  $r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            loadps_unit!($layout, $m0, $r1),
            loadps_unit!($layout, mem!($m0, "0x20"), $r2),
            loadps_unit!(4, $layout, mem!($m0, "0x40"), $r3),
        )
    };
    (16, $layout:tt, $m0:expr, $r1:expr, $r2:expr) => {
        concat!(
            loadps_unit!($layout, $m0, $r1),
            loadps_unit!($layout, mem!($m0, "0x20"), $r2),
        )
    };
    (16, $layout:tt, $m0:expr) => {
        concat!(
            loadps_unit!($layout, $m0, 0),
            loadps_unit!($layout, mem!($m0, "0x20"), 1),
        )
    };
    (12, $layout:tt, $m0:expr, $r1:expr, $r2:expr) => {
        concat!(
            loadps_unit!($layout, $m0, 0),
            loadps_unit!(4, $layout, mem!($m0, "0x20"), 1),
        )
    };
    (8, $layout:tt, $m0:expr) => {
        concat!(
            loadps_unit!($layout, $m0, 0),
        )
    };
    (8, $layout:tt, $m0:expr, $r1:expr) => {
        concat!(
            loadps_unit!($layout, $m0, $r1),
        )
    };
    (4, $layout:tt, $m0:expr) => {
        concat!(
            loadps_unit!(4, $layout, $m0, 0),
        )
    };
 }

 macro_rules! storeps_unit {
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
    (4, M, $r1:expr, $m0:expr) => {
        concat!(
            "vmaskmovps %xmm", $r1, ", %xmm1, ", $m0,  "\n",
        )
    };
    (4, $layout:tt, $r1:expr, $m0:expr) => {
        concat!(
            "vmovups ", "%xmm", $r1, ", ", $m0, "\n",
        )
    };
}
macro_rules! storeps {
	(24, $layout:tt, $m0:expr, $r1:expr, $r2:expr, $r3:expr) => {
		concat!(
			storeps_unit!(C, $r1, $m0),
			storeps_unit!(C, $r2, mem!($m0, "0x20")),
			storeps_unit!($layout, $r3, mem!($m0, "0x40")),
		)
	};
	(20, $layout:tt, $m0:expr, $r1:expr, $r2:expr, $r3:expr) => {
		concat!(
			storeps_unit!(C, $r1, $m0),
			storeps_unit!(C, $r2, mem!($m0, "0x20")),
			storeps_unit!(4, $layout, $r3, mem!($m0, "0x40")),
		)
	};
	(16, $layout:tt, $m0:expr, $r1:expr, $r2:expr) => {
		concat!(
			storeps_unit!(C, $r1, $m0),
			storeps_unit!($layout, $r2, mem!($m0, "0x20")),
		)
	};
	(12, $layout:tt, $m0:expr, $r1:expr, $r2:expr) => {
		concat!(
			storeps_unit!(C, $r1, $m0),
			storeps_unit!(4, $layout, $r2, mem!($m0, "0x20")),
		)
	};
	(8, $layout:tt, $m0:expr, $r1:expr) => {
		concat!(
			storeps_unit!($layout, $r1, $m0),
		)
	};
 }

macro_rules! vfmadd231ps {
    ($r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            "vfmadd231ps %ymm", $r1, ", %ymm", $r2,", %ymm", $r3, "\n",
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
	($KER:tt,C,R) => {
    	concat!(
        	// mov cs_a to reg
        	"mov ({int_arrx}), {x1}", "\n",
        	// mov cs_b to reg
        	"mov 8({int_arrx}), {x2}", "\n",
        	"mov 24({int_arrx}),{x0}", "\n",
        	"test {x0},{x0}", "\n",
    	)
	};
	($KER:tt,B,R) => {
    	concat!(
			"/* {x3} */", "\n",
			"/* {x1} */", "\n",
        	// mov cs_b to reg
        	"mov 8({int_arrx}), {x2}", "\n",
        	"mov 24({int_arrx}),{x0}", "\n",
        	"test {x0},{x0}", "\n",
    	)
	};
	(VER24,C,C) => {
    	concat!(
        	// mov cs_a to reg
        	"mov ({int_arrx}), {x1}", "\n",
        	// mov cs_b to reg
        	"mov 8({int_arrx}), {x2}", "\n",
        	"lea ({x2}, {x2}, 2), {x0}", "\n",

        	"lea ({bx}, {x0}, 1), {x3}", "\n",
        	"mov 24({int_arrx}),{x0}", "\n",
        	"test {x0},{x0}", "\n",
    	)
	};
	(VER16,C,C) => {
    	concat!(
        	// mov cs_a to reg
        	"mov ({int_arrx}), {x1}", "\n",
        	// mov cs_b to reg
        	"mov 8({int_arrx}), {x2}", "\n",
        	"lea ({x2}, {x2}, 2), {x0}", "\n",

        	"lea ({bx}, {x0}, 1), {x3}", "\n",
        	"mov 24({int_arrx}),{x0}", "\n",
        	"test {x0},{x0}", "\n",
    	)
	};
	(VER8,C,C) => {
    	concat!(
        	// mov cs_a to reg
        	"mov ({int_arrx}), {x1}", "\n",
        	// mov cs_b to reg
        	"mov 8({int_arrx}), {x2}", "\n",
        	"lea ({x2}, {x2}, 2), {x0}", "\n",

        	"lea ({bx}, {x0}, 1), {x3}", "\n",
        	"mov 24({int_arrx}),{x0}", "\n",
        	"test {x0},{x0}", "\n",
    	)
	};
	(VER24,B,C) => {
    	concat!(
        	// mov cs_b to reg
        	"mov 8({int_arrx}), {x2}", "\n",
        	"lea ({x2}, {x2}, 2), {x1}", "\n",

        	"lea ({bx}, {x1}, 1), {x3}", "\n",
        	"mov 24({int_arrx}),{x0}", "\n",
        	"test {x0},{x0}", "\n",
    	)
	};
	(VER16,B,C) => {
    	concat!(
        	// mov cs_b to reg
        	"mov 8({int_arrx}), {x2}", "\n",
        	"lea ({x2}, {x2}, 2), {x1}", "\n",

        	"lea ({bx}, {x1}, 1), {x3}", "\n",
        	"mov 24({int_arrx}),{x0}", "\n",
        	"test {x0},{x0}", "\n",
    	)
	};
	(VER8,B,C) => {
    	concat!(
        	// mov cs_b to reg
        	"mov 8({int_arrx}), {x2}", "\n",
        	"lea ({x2}, {x2}, 2), {x1}", "\n",

        	"lea ({bx}, {x1}, 1), {x3}", "\n",
        	"mov 24({int_arrx}),{x0}", "\n",
        	"test {x0},{x0}", "\n",
    	)
	};
}


macro_rules! asm_c_load {
	(6) => {
    	concat!(
        	"mov 16({int_arrx}),{x0}", "\n",
        	"lea (,{x0}, 4), {x0}","\n",
        	"lea ({x0}, {x0}, 2), {x1}", "\n",
        	"lea ({cx}, {x1},), {x1}", "\n",
    	)
	};
	(5) => {
    	concat!(
        	"mov 16({int_arrx}),{x0}", "\n",
        	"lea (,{x0}, 4), {x0}","\n",
        	"lea ({x0}, {x0}, 2), {x1}", "\n",
        	"lea ({cx}, {x1},), {x1}", "\n",
    	)
	};
	(4) => {
    	concat!(
        	"mov 16({int_arrx}),{x0}", "\n",
        	"lea (,{x0}, 4), {x0}", "\n",
        	"lea ({x0}, {x0}, 2), {x1}", "\n",
        	"lea ({cx}, {x1},), {x1}", "\n",
    	)
	};
	(3) => {
    	concat!(
        	"mov 16({int_arrx}),{x0}", "\n",
        	"lea (,{x0}, 4), {x0}", "\n",
    	)
	};
	(2) => {
    	concat!(
        	"mov 16({int_arrx}),{x0}", "\n",
        	"lea (,{x0}, 4), {x0}","\n",
    	)
	};
	(1) => {
    	concat!(
        	"mov 16({int_arrx}),{x0}", "\n",
        	"lea (,{x0}, 4), {x0}","\n",
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
	(C,$nr:tt) => {
    	""
	};
	(R,$nr:tt) => {
    	"add {x2},{bx} \n"
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
	(C, $X:tt, $K:tt ) => {
    	concat!(
        	"add $4*", $K, ",{bx}", "\n",
        	"add $4*", $K, ",{x3}", "\n",
    	)
	};
	(R, $X:tt, $K:tt) => {
    	""
	};
	(B, $X:tt, $K:tt) => {
    	concat!(
        	"add $4*", $K, "*", $X, ", {bx}", "\n",
    	)
	};
}

macro_rules! asm_alpha_scale_0 {
	($r0:tt, $r1:tt) => {
		seq!(r in $r0..=$r1 {
			concat!(
				"vbroadcastss ({alphax}),%ymm1", "\n",
				#(
					"vmulps %ymm1, %ymm", r, ",%ymm", r, "\n",
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
		acc_ps!($mr, $layout, "0({cx})", 4, 5, 6)
	};
	($mr:tt, 1, $layout:tt) => {
		acc_ps!($mr, $layout, "0({cx}, {x0})", 7, 8, 9)
	};
	($mr:tt, 2, $layout:tt) => {
		acc_ps!($mr, $layout, "0({cx}, {x0}, 2)",  10, 11, 12)
	}; 
	($mr:tt, 3, $layout:tt) => {
		acc_ps!($mr, $layout, "0({x1})",  13, 14, 15)
	};
}

macro_rules! acc_ps_t {
	($r0:tt,T) => {
		concat!(
			"vmaskmovps 0({cx}), %xmm1, %xmm3", "\n",
			"vmaskmovps 0({cx}, {x0}, 4), %xmm1, %xmm2", "\n",
			"vinsertf128 $1, %xmm2, %ymm3, %ymm3", "\n",
			"vfmadd231ps %ymm3, %ymm0, %ymm", $r0, "\n",
		)
	};
	($r0:tt,F) => {
		concat!(
			"vmovups 0({cx}) , %xmm3", "\n",
			"vinsertf128 $1, 0({cx}, {x0}, 4), %ymm3, %ymm3", "\n",
			"vfmadd231ps %ymm3, %ymm0, %ymm", $r0, "\n",
		)
	}
}

macro_rules! acc_ps_t2 {
	($r0:tt, $m_th:tt) => {
		concat!(
			"vmaskmovps 0({cx}), %xmm1, %xmm3", "\n",
			"vfmadd231ps %ymm3, %ymm0, %ymm", $r0, "\n",
			"cmp $", $m_th, ", {m}", "\n",
			"jl 8f", "\n",
		)
	};
}

// macro_rules! store_ps_t2 {
// 	($r0:tt, $m_th:tt) => {
// 		concat!(
// 			"vmaskmovps %xmm", $r0, ", %xmm1, 0({cx})", "\n",
// 			"cmp $", $m_th, ", {m}", "\n",
// 			"jl 7f", "\n",
// 		)
// 	};
// }

macro_rules! acc_ps_t3 {
	($r0:tt, $m_th:tt) => {
		concat!(
			"vmaskmovps 0({cx}), %xmm1, %xmm2", "\n",
			"vinsertf128 $1, %xmm2, %ymm3, %ymm3", "\n",
			"vfmadd231ps %ymm3, %ymm0, %ymm", $r0, "\n",
			"cmp $", $m_th, ", {m}", "\n",
			"jl 8f", "\n",
		)
	};
	($r0:tt) => {
		concat!(
			"vmaskmovps 0({cx}), %xmm1, %xmm2", "\n",
			"vinsertf128 $1, %xmm2, %ymm3, %ymm3", "\n",
			"vfmadd231ps %ymm3, %ymm0, %ymm", $r0, "\n",
		)
	};
}


macro_rules! store_t_4x4 {
	($r0:tt, $r1:tt, $r2:tt, $r3:tt) => {
		concat!(
			"vmovups %xmm", $r0, ", 0({cx})", "\n",
			"vmovups %xmm", $r1, ", 0({cx}, {x0})", "\n",
			"vmovups %xmm", $r2, ", 0({cx}, {x0}, 2)", "\n",
			"vmovups %xmm", $r3, ", 0({x1})", "\n",
 
			"lea ({cx}, {x0}, 4), {cx}", "\n",
			"lea ({x1}, {x0}, 4), {x1}", "\n",
 
			"vextractf128 $1, %ymm", $r0, ", %xmm", $r0, "\n",
			"vmovups %xmm", $r0, ", 0({cx})", "\n",
			"vextractf128 $1, %ymm", $r1, ", %xmm", $r1, "\n",
			"vmovups %xmm", $r1, ", 0({cx}, {x0})", "\n",
			"vextractf128 $1, %ymm", $r2, ", %xmm", $r2, "\n",
			"vmovups %xmm", $r2, ", 0({cx}, {x0}, 2)", "\n",
			"vextractf128 $1, %ymm", $r3, ", %xmm", $r3, "\n",
			"vmovups %xmm", $r3, ", 0({x1})", "\n",
 
		)
	};

	($r0:tt, $r1:tt, $r2:tt, $r3:tt, T) => {
		concat!(
			"vmaskmovps %xmm", $r0, ", %xmm1, ({cx})",  "\n",
			"vextractf128 $1, %ymm", $r0, ", %xmm", $r0, "\n",
			"vmaskmovps %xmm", $r0, ", %xmm1, 0({cx}, {x0}, 4)", "\n",
			"add {x0}, {cx}", "\n",

			"vmaskmovps %xmm", $r1, ", %xmm1, ({cx})",  "\n",
			"vextractf128 $1, %ymm", $r1, ", %xmm", $r1, "\n",
			"vmaskmovps %xmm", $r1, ", %xmm1, 0({cx}, {x0}, 4)", "\n",
			"add {x0}, {cx}", "\n",
 
			"vmaskmovps %xmm", $r2, ", %xmm1, ({cx})",  "\n",
			"vextractf128 $1, %ymm", $r2, ", %xmm", $r2, "\n",
			"vmaskmovps %xmm", $r2, ", %xmm1, 0({cx}, {x0}, 4)", "\n",
			"add {x0}, {cx}", "\n",
 
			"vmaskmovps %xmm", $r3, ", %xmm1, ({cx})",  "\n",
			"vextractf128 $1, %ymm", $r3, ", %xmm", $r3, "\n",
			"vmaskmovps %xmm", $r3, ", %xmm1, 0({cx}, {x0}, 4)", "\n",
			"add {x0}, {cx}", "\n",
 
		)
	}
}

macro_rules! store_t_4x4_2 {
	($r0:tt, $r1:tt, $r2:tt, $r3:tt) => {
		concat!(
			"vmaskmovps %xmm", $r0, ", %xmm1, 0({cx})", "\n",
			"cmp $2, {m}", "\n",
			"jl 7f", "\n",
			"add {x0}, {cx}", "\n",
 
			"vmaskmovps %xmm", $r1, ", %xmm1, 0({cx})", "\n",
			"cmp $3, {m}", "\n",
			"jl 7f", "\n",
			"add {x0}, {cx}", "\n",
 
			"vmaskmovps %xmm", $r2, ", %xmm1, 0({cx})", "\n",
			"cmp $4, {m}", "\n",
			"jl 7f", "\n",
			"add {x0}, {cx}", "\n",
 
			"vmaskmovps %xmm", $r3, ", %xmm1, 0({cx})", "\n",
			"cmp $5, {m}", "\n",
			"jl 7f", "\n",
			"add {x0}, {cx}", "\n",
 
			"vextractf128 $1, %ymm", $r0, ", %xmm", $r0, "\n",
			"vmaskmovps %xmm", $r0, ", %xmm1, 0({cx})", "\n",
			"cmp $6, {m}", "\n",
			"jl 7f", "\n",
			"add {x0}, {cx}", "\n",
 
			"vextractf128 $1, %ymm", $r1, ", %xmm", $r1, "\n",
			"vmaskmovps %xmm", $r1, ", %xmm1, 0({cx})", "\n",
			"cmp $7, {m}", "\n",
			"jl 7f", "\n",
			"add {x0}, {cx}", "\n",
 
			"vextractf128 $1, %ymm", $r2, ", %xmm", $r2, "\n",
			"vmaskmovps %xmm", $r2, ", %xmm1, 0({cx})", "\n",
			"cmp $8, {m}", "\n",
			"jl 7f", "\n",
			"add {x0}, {cx}", "\n",
 
			"vextractf128 $1, %ymm", $r3, ", %xmm", $r3, "\n",
			"vmaskmovps %xmm", $r3, ", %xmm1, 0({cx})", "\n",
		)
	}
}

macro_rules! asm_24x4_acc {
	($mr:tt, $nr:tt, $layout:tt, F) => {
		seq!(n in 0..$nr {
			concat!(
				#(
					asm_24x4_acc_seq!($mr, n, $layout), 
				)*
			)
		})
	};
	(24, $nr:tt, M, T) => {
		concat!(
			"mov {cx}, {x3}", "\n",

			acc_ps_t!(4,T),
			"lea ({cx}, {x0}), {cx}", "\n",
 
			acc_ps_t!(7,T),
			"lea ({cx}, {x0}), {cx}", "\n",
 
			acc_ps_t!(10,T),
			"lea ({cx}, {x0}), {cx}", "\n",
 
			acc_ps_t!(13,T),
			"lea ({cx}, {x0}), {cx}", "\n",
 
			"lea ({cx}, {x0}, 4), {cx}", "\n",
 
			acc_ps_t!(5,T),
			"lea ({cx}, {x0}), {cx}", "\n",
 
			acc_ps_t!(8,T),
			"lea ({cx}, {x0}), {cx}", "\n",
 
			acc_ps_t!(11,T),
			"lea ({cx}, {x0}), {cx}", "\n",
 
			acc_ps_t!(14,T),
			"lea ({cx}, {x0}), {cx}", "\n",
 
			"lea ({cx}, {x0}, 4), {cx}", "\n",

			acc_ps_t2!(6,2),
			"lea ({cx}, {x0}), {cx}", "\n",

			acc_ps_t2!(9,3),
			"lea ({cx}, {x0}), {cx}", "\n",
 
			acc_ps_t2!(12,4),
			"lea ({cx}, {x0}), {cx}", "\n",
 
			acc_ps_t2!(15,5),
			"lea ({cx}, {x0}), {cx}", "\n",

			"vpxor %ymm3, %ymm3, %ymm3", "\n",

			acc_ps_t3!(6,6),
			"lea ({cx}, {x0}), {cx}", "\n",

			acc_ps_t3!(9,7),
			"lea ({cx}, {x0}), {cx}", "\n",

			acc_ps_t3!(12,8),
			"lea ({cx}, {x0}), {cx}", "\n",

			acc_ps_t3!(15),

			"8:", "\n",

			"mov {x3}, {cx}", "\n",

		)
	}; 
	(24, $nr:tt, C, T) => {
		concat!(
			"mov {cx}, {x3}", "\n",

			acc_ps_t!(4,F),
			"lea ({cx}, {x0}), {cx}", "\n",
			acc_ps_t!(7,F),
			"lea ({cx}, {x0}), {cx}", "\n",
			acc_ps_t!(10,F),
			"lea ({cx}, {x0}), {cx}", "\n",
			acc_ps_t!(13,F),
			"lea ({cx}, {x0}), {cx}", "\n",
 
			"lea ({cx}, {x0}, 4), {cx}", "\n",
 
			acc_ps_t!(5,F),
			"lea ({cx}, {x0}), {cx}", "\n",
			acc_ps_t!(8,F),
			"lea ({cx}, {x0}), {cx}", "\n",
			acc_ps_t!(11,F),
			"lea ({cx}, {x0}), {cx}", "\n",
			acc_ps_t!(14,F),
			"lea ({cx}, {x0}), {cx}", "\n",
 
			"lea ({cx}, {x0}, 4), {cx}", "\n",
 
			acc_ps_t!(6,F),
			"lea ({cx}, {x0}), {cx}", "\n",
			acc_ps_t!(9,F),
			"lea ({cx}, {x0}), {cx}", "\n",
			acc_ps_t!(12,F),
			"lea ({cx}, {x0}), {cx}", "\n",
			acc_ps_t!(15,F),
			"lea ({cx}, {x0}), {cx}", "\n",

			"mov {x3}, {cx}", "\n",
		)
	}; 
}


macro_rules! asm_24x4_store_seq {
	($mr:tt, 0, $layout:tt) => {
		storeps!($mr, $layout, "0({cx})", 4, 5, 6)
	};
	($mr:tt, 1, $layout:tt) => {
		storeps!($mr, $layout, "0({cx}, {x0})", 7, 8, 9)
	};
	($mr:tt, 2, $layout:tt) => {
		storeps!($mr, $layout, "0({cx}, {x0}, 2)",  10, 11, 12)
	}; 
	($mr:tt, 3, $layout:tt) => {
		storeps!($mr, $layout, "0({x1})",  13, 14, 15)
	};
}


macro_rules! asm_24x4_store {
	($mr:tt, $nr:tt, $layout:tt, F) => {
		seq!(n in 0..$nr {
			concat!(
				#(
					asm_24x4_store_seq!($mr, n, $layout), 
				)*
			)
		})
	};
	(24, $nr:tt, C, T) => {
		concat!(

			store_t_4x4!(4,7,10,13),
 
			"lea ({cx}, {x0}, 4), {cx}", "\n",
			"lea ({x1}, {x0}, 4), {x1}", "\n",
 
			store_t_4x4!(5,8,11,14),
 
			"lea ({cx}, {x0}, 4), {cx}", "\n",
			"lea ({x1}, {x0}, 4), {x1}", "\n",
 
			store_t_4x4!(6,9,12,15),
		)
	};
	(24, $nr:tt, M, T) => {
		concat!(

			store_t_4x4!(4,7,10,13,T),

			"lea ({cx}, {x0}, 4), {cx}", "\n",
 
			store_t_4x4!(5,8,11,14,T),

			"lea ({cx}, {x0}, 4), {cx}", "\n",

			store_t_4x4_2!(6,9,12,15),
		)
	};
  
}


macro_rules! asm_16x6_acc_seq {
	($mr:tt, 0, $layout:tt) => {
		acc_ps!($mr, $layout, "0({cx})", 4, 5)
	};
	($mr:tt, 1, $layout:tt) => {
		acc_ps!($mr, $layout, "0({cx}, {x0})", 6, 7)
	};
	($mr:tt, 2, $layout:tt) => {
		acc_ps!($mr, $layout, "0({cx}, {x0}, 2)", 8, 9)
	}; 
	($mr:tt, 3, $layout:tt) => {
		acc_ps!($mr, $layout, "0({x1})", 10, 11)
	};
	($mr:tt, 4, $layout:tt) => {
        acc_ps!($mr, $layout, "0({x1}, {x0})", 12, 13)
	};
	($mr:tt, 5, $layout:tt) => {
		acc_ps!($mr, $layout, "0({x1}, {x0}, 2)", 14, 15)
	};
}

macro_rules! asm_16x6_acc {
	($mr:tt, $nr:tt, $layout:tt,F) => {
		seq!(n in 0..$nr {
			concat!(
				#(
					asm_16x6_acc_seq!($mr, n, $layout), 
				)*
			)
		})
	};
	(16, $nr:tt, M, T) => {
		concat!(
			"mov {cx}, {x3}", "\n",

			acc_ps_t!(4,T),
			"lea ({cx}, {x0}), {cx}", "\n",
			acc_ps_t!(6,T),
			"lea ({cx}, {x0}), {cx}", "\n",
			acc_ps_t!(8,T),
			"lea ({cx}, {x0}), {cx}", "\n",
			acc_ps_t!(10,T),
			"lea ({cx}, {x0}), {cx}", "\n",
 
			"lea ({cx}, {x0}, 4), {cx}", "\n",

			acc_ps_t2!(5,2),
			"lea ({cx}, {x0}), {cx}", "\n",
			
			acc_ps_t2!(7,3),
			"lea ({cx}, {x0}), {cx}", "\n",

			acc_ps_t2!(9,4),
			"lea ({cx}, {x0}), {cx}", "\n",
 
			acc_ps_t2!(11,5),
			"lea ({cx}, {x0}), {cx}", "\n",

			"vpxor %ymm3, %ymm3, %ymm3", "\n",

			acc_ps_t3!(5,6),
			"lea ({cx}, {x0}), {cx}", "\n",

			acc_ps_t3!(7,7),
			"lea ({cx}, {x0}), {cx}", "\n",

			acc_ps_t3!(9,8),
			"lea ({cx}, {x0}), {cx}", "\n",

			acc_ps_t3!(11),

			"8:", "\n",

			"mov {x3}, {cx}", "\n",
		)
	};
 
}

macro_rules! asm_16x6_store_seq {
	($mr:tt, 0, $layout:tt) => {
		storeps!($mr, $layout, "0({cx})", 4, 5)
	};
	($mr:tt, 1, $layout:tt) => {
		storeps!($mr, $layout, "0({cx}, {x0})", 6, 7)
	};
	($mr:tt, 2, $layout:tt) => {
		storeps!($mr, $layout, "0({cx}, {x0}, 2)", 8, 9)
	}; 
	($mr:tt, 3, $layout:tt) => {
		storeps!($mr, $layout, "0({x1})", 10, 11)
	};
	($mr:tt, 4, $layout:tt) => {
    	storeps!($mr, $layout, "0({x1}, {x0})", 12, 13)
	};
	($mr:tt, 5, $layout:tt) => {
		storeps!($mr, $layout, "0({x1}, {x0}, 2)", 14, 15)
	};
}

macro_rules! asm_16x6_store {
	($mr:tt, $nr:tt, $layout:tt,F) => {
		seq!(n in 0..$nr {
			concat!(
				#(
					asm_16x6_store_seq!($mr, n, $layout), 
				)*
			)
		})
	};
	(16, $nr:tt, M, T) => {
		concat!(

			store_t_4x4!(4,6,8,10,T),

			"lea ({cx}, {x0}, 4), {cx}", "\n",
 
			store_t_4x4_2!(5,7,9,11),
 
		)
	}; 
}

macro_rules! asm_8x6_acc_seq {
	($mr:tt, 0, $layout:tt) => {
		acc_ps!($mr, $layout, "0({cx})", 7)
	};
	($mr:tt, 1, $layout:tt) => {
		acc_ps!($mr, $layout, "0({cx}, {x0})", 8)
	};
	($mr:tt, 2, $layout:tt) => {
		acc_ps!($mr, $layout, "0({cx}, {x0}, 2)", 9)
	}; 
	($mr:tt, 3, $layout:tt) => {
		acc_ps!($mr, $layout, "0({x1})", 10)
	};
	($mr:tt, 4, $layout:tt) => {
        acc_ps!($mr, $layout, "0({x1}, {x0})", 11)
	};
	($mr:tt, 5, $layout:tt) => {
		acc_ps!($mr, $layout, "0({x1}, {x0}, 2)", 12)
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
	($mr:tt, $nr:tt, $layout:tt, F) => {
		seq!(n in 0..$nr {
			concat!(
				#(
					asm_8x6_acc_seq!($mr, n, $layout), 
				)*
			)
		})
	};
	(8, $nr:tt, M, T) => {
		concat!(
			"mov {cx}, {x3}", "\n",
			
			acc_ps_t2!(7,2),
			"lea ({cx}, {x0}), {cx}", "\n",

			acc_ps_t2!(8,3),
			"lea ({cx}, {x0}), {cx}", "\n",
 
			acc_ps_t2!(9,4),
			"lea ({cx}, {x0}), {cx}", "\n",
 
			acc_ps_t2!(10,5),
			"lea ({cx}, {x0}), {cx}", "\n",

			"vpxor %ymm3, %ymm3, %ymm3", "\n",

			acc_ps_t3!(7,6),
			"lea ({cx}, {x0}), {cx}", "\n",

			acc_ps_t3!(8,7),
			"lea ({cx}, {x0}), {cx}", "\n",

			acc_ps_t3!(9,8),
			"lea ({cx}, {x0}), {cx}", "\n",

			acc_ps_t3!(10),

			"8:", "\n",

			"mov {x3}, {cx}", "\n",
 
		)
	};
 
}


macro_rules! asm_8x6_store_seq {
	($mr:tt, 0, $layout:tt) => {
		storeps!($mr, $layout, "0({cx})", 7)
	};
	($mr:tt, 1, $layout:tt) => {
		storeps!($mr, $layout, "0({cx}, {x0})", 8)
	};
	($mr:tt, 2, $layout:tt) => {
		storeps!($mr, $layout, "0({cx}, {x0}, 2)", 9)
	}; 
	($mr:tt, 3, $layout:tt) => {
		storeps!($mr, $layout, "0({x1})", 10)
	};
	($mr:tt, 4, $layout:tt) => {
        storeps!($mr, $layout, "0({x1}, {x0})", 11)
	};
	($mr:tt, 5, $layout:tt) => {
		storeps!($mr, $layout, "0({x1}, {x0}, 2)", 12)
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
	($mr:tt, $nr:tt, $layout:tt, F) => {
		seq!(n in 0..$nr {
			concat!(
				#(
					asm_8x6_store_seq!($mr, n, $layout), 
				)*
			)
		})
	};
	(8, $nr:tt, M, T) => {
		concat!(
			store_t_4x4_2!(7,8,9,10),
		)
	}; 
}

macro_rules! transpose_4x4 {
	($r0:tt, $r1:tt, $r2:tt, $r3:tt) => {
		concat!(
			"vunpcklps %ymm", $r1, ", %ymm", $r0, ", %ymm1", "\n",
			"vunpckhps %ymm", $r1, ", %ymm", $r0, ", %ymm0", "\n",
			"vunpcklps %ymm", $r3, ", %ymm", $r2, ", %ymm3", "\n",
			"vunpckhps %ymm", $r3, ", %ymm", $r2, ", %ymm2", "\n",

			"vunpcklpd %ymm3, %ymm1, %ymm", $r0, "\n",
			"vunpckhpd %ymm3, %ymm1, %ymm", $r1, "\n",
			"vunpcklpd %ymm2, %ymm0, %ymm", $r2, "\n",
			"vunpckhpd %ymm2, %ymm0, %ymm", $r3, "\n",
		)
	};
}

macro_rules! transpose {
	(F, $mr:tt, $nr:tt) => {
		""
	};
	(T, 24, $nr:tt) => {
		concat!(
			transpose_4x4!(4, 7, 10, 13),
			transpose_4x4!(5, 8, 11, 14),
			transpose_4x4!(6, 9, 12, 15),
		)
	};
	(T, 16, $nr:tt) => {
		concat!(
			transpose_4x4!(4, 6, 8, 10),
			transpose_4x4!(5, 7, 9, 11),
		)
	};
	(T, 8, $nr:tt) => {
		concat!(
			transpose_4x4!(7, 8, 9, 10),
		)
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
	($mr:tt, B, $K:tt) => {
    	loadps!($mr, B, concat!($mr,"*4*",$K,"({ax})"))
	};
	($mr:tt, C, $K:tt) => {
    	loadps!($mr, C, "0({ax})")
	};
}

macro_rules! fmadd_3v {
	(0) => {
		concat!(
			vfmadd231ps!(0, 3, 4),
			vfmadd231ps!(1, 3, 5),
			vfmadd231ps!(2, 3, 6),
		)
	};
	(1) => {
		concat!(
			vfmadd231ps!(0, 3, 7),
			vfmadd231ps!(1, 3, 8),
			vfmadd231ps!(2, 3, 9),
		)
	};
	(2) => {
		concat!(
			vfmadd231ps!(0, 3, 10),
			vfmadd231ps!(1, 3, 11),
			vfmadd231ps!(2, 3, 12),
		)
	};
	(3) => {
		concat!(
			vfmadd231ps!(0, 3, 13),
			vfmadd231ps!(1, 3, 14),
			vfmadd231ps!(2, 3, 15),
		)
	};
}

macro_rules! fmadd_2v {
	(0) => {
		concat!(
			vfmadd231ps!(0, 2, 4),
			vfmadd231ps!(1, 2, 5),
		)
	};
	(1) => {
		concat!(
			vfmadd231ps!(0, 3, 6),
			vfmadd231ps!(1, 3, 7),
		)
	};
	(2) => {
		concat!(
			vfmadd231ps!(0, 2, 8),
			vfmadd231ps!(1, 2, 9),
		)
	};
	(3) => {
		concat!(
			vfmadd231ps!(0, 3, 10),
			vfmadd231ps!(1, 3, 11),
		)
	};
	(4) => {
		concat!(
			vfmadd231ps!(0, 2, 12),
			vfmadd231ps!(1, 2, 13),
		)
	};
	(5) => {
		concat!(
			vfmadd231ps!(0, 3, 14),
			vfmadd231ps!(1, 3, 15),
		)
	};
}

macro_rules! fmadd_1v {
	(0) => {
		concat!(
			vfmadd231ps!(0, 1, 7),
		)
	};
	(1) => {
		concat!(
			vfmadd231ps!(0, 2, 8),
		)
	};
	(2) => {
		concat!(
			vfmadd231ps!(0, 3, 9),
		)
	};
	(3) => {
		concat!(
			vfmadd231ps!(0, 4, 10),
		)
	};
	(4) => {
		concat!(
			vfmadd231ps!(0, 5, 11),
		)
	};
	(5) => {
		concat!(
			vfmadd231ps!(0, 6, 12),
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
				#(
					load_b!($b_layout, n, $K, $nr, 3),
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
					load_b!($b_layout, n, $K, $nr, b_num_16x6!(n)),
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
					load_b!($b_layout, n, $K, $nr, b_num_8x6!(n)),
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
 
macro_rules! def_ukernel {
	(
		$VER:tt,
		$asm_step_macro:tt,
		$asm_acc_macro:tt,
		$asm_store_macro:tt,
    	$mr:tt, $nr:tt, $transpose:tt,
    	$a_layout:tt, $b_layout:tt,
		$pfa_dist:tt, $pfb_dist:tt,
		$unroll:tt,
    	$func_name:ident
	) => {
		#[target_feature(enable = "avx,fma")]
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

				transpose!($transpose, $mr, $nr),

            	"vbroadcastss ({betax}), %ymm0",

            	"vxorps %ymm3,%ymm3,%ymm3",
            	"vucomiss %xmm3,%xmm0",

            	// 6 -> BETAZERO
            	"je 6f",
            	$asm_acc_macro!($mr,$nr,C,$transpose),

            	// 6 -> BETAZERO
            	"6:",
            	$asm_store_macro!($mr,$nr,C,$transpose),
   	 
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
    	}
	};
}

macro_rules! def_ukernel_partial {
	(
		$VER:tt,
		$asm_step_macro:tt,
		$asm_acc_macro:tt,
		$asm_store_macro:tt,
    	$mr:tt, $nr:tt, $transpose:tt,
    	$a_layout:tt, $b_layout:tt,
		$pfa_dist:tt, $pfb_dist:tt,
		$unroll:tt,
    	$func_name:ident
	) => {
		#[target_feature(enable = "avx,fma")]
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

				transpose!($transpose, $mr, $nr),

            	"vbroadcastss ({betax}), %ymm0",

            	"vxorps %ymm3,%ymm3,%ymm3",
            	"vucomiss %xmm3,%xmm0",
				"vmovdqu ({maskx}), %ymm1",
				// "/* {maskx}*/",

            	// 6 -> BETAZERO
            	"je 6f",
            	$asm_acc_macro!($mr,$nr,M,$transpose),

            	// 6 -> BETAZERO
            	"6:",
            	$asm_store_macro!($mr,$nr,M,$transpose),
   	 
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
    	}
	};
}

macro_rules! def_ukernel_partial_new {
	(
		$VER:tt,
		$asm_step_macro:tt,
		$asm_acc_macro:tt,
		$asm_store_macro:tt,
    	$mr:tt, $nr:tt, $transpose:tt,
    	$a_layout:tt, $b_layout:tt,
		$pfa_dist:tt, $pfb_dist:tt,
		$unroll:tt,
    	$func_name:ident
	) => {
		#[target_feature(enable = "avx,fma")]
    	pub(crate) unsafe fn $func_name(
        	a: *const TA, b: *const TB, c: *mut TC,
        	alpha: *const TA, beta: *const TB,
        	k: usize,
        	ldc: usize,
			ld_arr: [usize; 2],
			// mask: *const u32,
			m: usize
    	) {
        	let k_iter = k / $unroll;
        	let k_left = k % $unroll;
            let u64_arr = [ld_arr[0], ld_arr[1], ldc, k_iter, k_left];
        	let u64_ptr = u64_arr.as_ptr();
			let cf = c;
			let mask_arr = [
				u32::MAX,
				u32::MAX,
				u32::MAX,
				u32::MAX,
				u32::MAX,
				u32::MAX,
				u32::MAX,
				u32::MAX,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
				0,
			];
			let mask = mask_arr.as_ptr().add(8-$nr);
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

				transpose!($transpose, $mr, $nr),

            	"vbroadcastss ({betax}), %ymm0",

            	"vxorps %ymm3,%ymm3,%ymm3",
            	"vucomiss %xmm3,%xmm0",
				"vmovdqu ({maskx}), %ymm1",

            	// 6 -> BETAZERO
            	"je 6f",
            	$asm_acc_macro!($mr,$nr,M,$transpose),

            	// 6 -> BETAZERO
            	"6:",
            	$asm_store_macro!($mr,$nr,M,$transpose),
   	 
            	// 7 -> DDONE
            	"7:",
            	ax = inout(reg) a => _,
            	bx = inout(reg) b => _,
            	cx = inout(reg) cf => _,
            	int_arrx = inout(reg) u64_ptr => _,
            	alphax = inout(reg) alpha => _,
            	betax = inout(reg) beta => _,
				maskx = inout(reg) mask => _,
				m = inout(reg) m => _,
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
    	}
	};
}

macro_rules! group_def_ukernel {
	(
    	$mr:tt,
    	$n0:tt, $n1:tt,
		$transpose:tt,
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
					$mr, nr, $transpose,
					$a_layout, $b_layout,
					0, 128,
					4,
					[<ukernel_ $mr x nr _ $func_name>]
				);
			}
		});
	};
}

group_def_ukernel!(24, 1, 4, F, B, B, bb, def_ukernel, asm_24x4_step, asm_24x4_acc, asm_24x4_store, VER24);
group_def_ukernel!(24, 1, 4, F, B, R, br, def_ukernel, asm_24x4_step, asm_24x4_acc, asm_24x4_store, VER24);
group_def_ukernel!(24, 1, 4, F, B, C, bc, def_ukernel, asm_24x4_step, asm_24x4_acc, asm_24x4_store, VER24);
group_def_ukernel!(24, 4, 4, T, B, C, rb_t, def_ukernel, asm_24x4_step, asm_24x4_acc, asm_24x4_store, VER24);
group_def_ukernel!(24, 4, 4, T, B, R, cb_t, def_ukernel, asm_24x4_step, asm_24x4_acc, asm_24x4_store, VER24);
group_def_ukernel!(24, 1, 4, F, B, B, bb_partial, def_ukernel_partial, asm_24x4_step, asm_24x4_acc, asm_24x4_store, VER24);
group_def_ukernel!(24, 1, 4, T, B, C, rb_t_partial, def_ukernel_partial_new, asm_24x4_step, asm_24x4_acc, asm_24x4_store, VER24);
group_def_ukernel!(24, 1, 4, T, B, R, cb_t_partial, def_ukernel_partial_new, asm_24x4_step, asm_24x4_acc, asm_24x4_store, VER24);

group_def_ukernel!(16, 1, 6, F, B, B, bb, def_ukernel, asm_16x6_step, asm_16x6_acc, asm_16x6_store, VER16);
group_def_ukernel!(16, 1, 4, F, B, R, br, def_ukernel, asm_16x6_step, asm_16x6_acc, asm_16x6_store, VER16);
group_def_ukernel!(16, 1, 4, F, B, C, bc, def_ukernel, asm_16x6_step, asm_16x6_acc, asm_16x6_store, VER16);
group_def_ukernel!(16, 1, 6, F, B, B, bb_partial, def_ukernel_partial, asm_16x6_step, asm_16x6_acc, asm_16x6_store, VER16);
group_def_ukernel!(16, 1, 4, T, B, C, rb_t_partial, def_ukernel_partial_new, asm_16x6_step, asm_16x6_acc, asm_16x6_store, VER16);
group_def_ukernel!(16, 1, 4, T, B, R, cb_t_partial, def_ukernel_partial_new, asm_16x6_step, asm_16x6_acc, asm_16x6_store, VER16);

group_def_ukernel!(8, 1, 6, F, B, B, bb, def_ukernel, asm_8x6_step, asm_8x6_acc, asm_8x6_store, VER8);
group_def_ukernel!(8, 1, 4, F, B, R, br, def_ukernel, asm_8x6_step, asm_8x6_acc, asm_8x6_store, VER8);
group_def_ukernel!(8, 1, 4, F, B, C, bc, def_ukernel, asm_8x6_step, asm_8x6_acc, asm_8x6_store, VER8);
group_def_ukernel!(8, 1, 6, F, B, B, bb_partial, def_ukernel_partial, asm_8x6_step, asm_8x6_acc, asm_8x6_store, VER8);
group_def_ukernel!(8, 1, 4, T, B, C, rb_t_partial, def_ukernel_partial_new, asm_8x6_step, asm_8x6_acc, asm_8x6_store, VER8);
group_def_ukernel!(8, 1, 4, T, B, R, cb_t_partial, def_ukernel_partial_new, asm_8x6_step, asm_8x6_acc, asm_8x6_store, VER8);

