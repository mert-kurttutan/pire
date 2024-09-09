use seq_macro::seq;
use std::arch::asm;
use super::VS;
use crate::{TA, TB, TC};
use crate::MyFn;
use half::f16;


macro_rules! beta_fmadd {
    (C, $m0:expr, $r1:expr) => {
        concat!(
            "vfmadd231ph ", $m0, ",%zmm0,%zmm", $r1, "\n",
        ) 
    };
    (M, $m0:expr, $r1:expr) => {
        concat!(
            // "vmovups ", $m0, ", %zmm1", ", %zmm2",  "\n",
			// "vmovups ", $m0, ", %zmm1 {{%k1}}", "\n",
			"vmovdqu16 ", $m0, ", %zmm1 {{%k1}}", "\n",
			
			"vfmadd231ph %zmm1,%zmm0,%zmm", $r1, "\n",
        )
    };
 }

macro_rules! vzeroall {
	($r0:tt, $r1:tt) => {
		seq!(r in $r0..=$r1 {
			concat!(
				#(
					"vpxorq %zmm",r,",%zmm",r,",%zmm",r,"\n",
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
		"vpbroadcastw"
	};
}

 macro_rules! vfmadd {
    // ($r1:expr, $r2:expr, $r3:expr) => {
    //     concat!(
    //         "vfmadd231ph %zmm", $r1, ", %zmm", $r2,", %zmm", $r3, "\n",
    //     ) 
    // };
    ($r1:expr, $m2:expr, $r3:expr) => {
        concat!(
			// "vfmadd231ph {bx}{{1to32}}", ", %zmm", $r1,", %zmm", $r3, "\n",
            // "vfmadd231ph ", $m2, "*2({bx}){{1to32}}", ", %zmm", $r1,", %zmm", $r3, "\n",
			"vfmadd231ph ", $m2, "{{1to32}}", ", %zmm", $r1,", %zmm", $r3, "\n",
        ) 
    };
}

macro_rules! bd {
	(B, $i:tt) => {
		concat!($i, "*2({bx})")
	};
	(S, 0) => {
		"0({bx})"
	};
	(S, 1) => {
		"0({bx}, {x2})"
	};
	(S, 2) => {
		"0({bx}, {x2}, 2)"
	};
	(S, 3) => {
		"0({x3})"
	};
	(S, 4) => {
		"0({x3}, {x2})"
	};
	(S, 5) => {
		"0({x3}, {x2}, 2)"
	};
	(S, 6) => {
		"0({x4})"
	};
	(S, 7) => {
		"0({x4}, {x2})"
	};
	(S, 8) => {
		"0({x4}, {x2}, 2)"
	};
	(S, 9) => {
		"0({x5})"
	};
	(S, 10) => {
		"0({x5}, {x2})"
	};
	(S, 11) => {
		"0({x5}, {x2}, 2)"
	};
	(S, 12) => {
		"0({x6})"
	};
	(S, 13) => {
		"0({x6}, {x2})"
	};
	(S, 14) => {
		"0({x6}, {x2}, 2)"
	};
}

macro_rules! loadp_unit {
    ($layout:tt, $m0:expr, $r1:expr) => {
        concat!(
            vmovp!($layout), $m0, ",%zmm", $r1, "\n",
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
            "vmovups %zmm", $r1, ", ", $m0, "\n",
        )
    };
    (B, $r1:expr, $m0:expr) => {
        concat!(
            "vmovaps %zmm", $r1, ", ", $m0, "\n",
        )
    };
    (M, $r1:expr, $m0:expr) => {
        concat!(
			"vmovdqu16 %zmm", $r1, ", ", $m0, " {{%k1}}\n",
		)
    };
}


macro_rules! asm_alpha_scale_0 {
	($r0:tt, $r1:tt) => {
		seq!(r in $r0..=$r1 {
			concat!(
				vbroadcast!(), " ({alphax}),%zmm1", "\n",
				#(
					"vmulph %zmm1, %zmm", r, ",%zmm", r, "\n",
				)*
			)
		})
	}
}

macro_rules! load_beta {
	() => {
		concat!(
			vbroadcast!(), " ({betax}), %zmm0\n",
			"vxorps %ymm1,%ymm1,%ymm1\n",
			"vcomish %xmm1,%xmm0\n",
		)
	}
}

macro_rules! mem {
    ($m0:tt, $b0:tt) => {
        concat!($b0, "+", $m0)
    };
 }

 macro_rules! acc_p {
    ($layout:tt, $m0:expr, $r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            beta_fmadd!(C, $m0, $r1),
            beta_fmadd!(C, mem!($m0, "0x40"), $r2),
            beta_fmadd!($layout, mem!($m0, "0x80"), $r3),
        )
    };
    ($layout:tt, $m0:expr, $r1:expr, $r2:expr) => {
        concat!(
            beta_fmadd!(C, $m0, $r1),
            beta_fmadd!($layout, mem!($m0, "0x40"), $r2),
        )
    };
    ($layout:tt, $m0:expr, $r1:expr) => {
        concat!(
            beta_fmadd!($layout, $m0, $r1),
        )
    };
 }

 macro_rules! loadp {
    (96, $layout:tt, $m0:expr) => {
        concat!(
            loadp_unit!($layout, $m0, 0),
            loadp_unit!($layout, mem!($m0, "0x40"), 1),
            loadp_unit!($layout, mem!($m0, "0x80"), 2),
        )
    };
    (64, $layout:tt, $m0:expr) => {
        concat!(
            loadp_unit!($layout, $m0, 0),
            loadp_unit!($layout, mem!($m0, "0x40"), 1),
        )
    };
    (32, $layout:tt, $m0:expr) => {
        concat!(
            loadp_unit!($layout, $m0, 0),
        )
    };
 }


macro_rules! storep {
	($layout:tt, $m0:expr, $r1:expr, $r2:expr, $r3:expr) => {
		concat!(
			storep_unit!(C, $r1, $m0),
			storep_unit!(C, $r2, mem!($m0, "0x40")),
			storep_unit!($layout, $r3, mem!($m0, "0x80")),
		)
	};
	($layout:tt, $m0:expr, $r1:expr, $r2:expr) => {
		concat!(
			storep_unit!(C, $r1, $m0),
			storep_unit!($layout, $r2, mem!($m0, "0x40")),
		)
	};
	($layout:tt, $m0:expr, $r1:expr) => {
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
			"/* {x5} */", "\n",
			"/* {x4} */", "\n",
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
        	"lea ({x2}, {x2}, 2), {x6}", "\n",
        	"lea ({bx}, {x6}, 1), {x3}", "\n",
			"lea ({x3}, {x6}, 1), {x4}", "\n",
			"lea ({x4}, {x6}, 1), {x5}", "\n",
			"lea ({x5}, {x6}, 1), {x6}", "\n",

        	"mov 24({dim_arrx}),{x0}", "\n",
        	"test {x0},{x0}", "\n",
    	)
	};
}


macro_rules! asm_c_load {
	(15) => {
    	concat!(
        	"mov 16({dim_arrx}),{x0}", "\n",
        	"lea ({x0}, {x0}, 2), {x4}", "\n",
        	"lea ({cx}, {x4},), {x1}", "\n",
			"lea ({x1}, {x4},), {x2}", "\n",
			"lea ({x2}, {x4},), {x3}", "\n",
			"lea ({x3}, {x4},), {x4}", "\n",
    	)
	};
	(14) => {
    	concat!(
        	"mov 16({dim_arrx}),{x0}", "\n",
        	"lea ({x0}, {x0}, 2), {x4}", "\n",
        	"lea ({cx}, {x4},), {x1}", "\n",
			"lea ({x1}, {x4},), {x2}", "\n",
			"lea ({x2}, {x4},), {x3}", "\n",
			"lea ({x3}, {x4},), {x4}", "\n",
    	)
	};
	(13) => {
    	concat!(
        	"mov 16({dim_arrx}),{x0}", "\n",
        	"lea ({x0}, {x0}, 2), {x4}", "\n",
        	"lea ({cx}, {x4},), {x1}", "\n",
			"lea ({x1}, {x4},), {x2}", "\n",
			"lea ({x2}, {x4},), {x3}", "\n",
			"lea ({x3}, {x4},), {x4}", "\n",
    	)
	};
	(12) => {
    	concat!(
        	"mov 16({dim_arrx}),{x0}", "\n",
        	"lea ({x0}, {x0}, 2), {x3}", "\n",
        	"lea ({cx}, {x3},), {x1}", "\n",
			"lea ({x1}, {x3},), {x2}", "\n",
			"lea ({x2}, {x3},), {x3}", "\n",
    	)
	};
	(11) => {
    	concat!(
        	"mov 16({dim_arrx}),{x0}", "\n",
        	"lea ({x0}, {x0}, 2), {x3}", "\n",
        	"lea ({cx}, {x3},), {x1}", "\n",
			"lea ({x1}, {x3},), {x2}", "\n",
			"lea ({x2}, {x3},), {x3}", "\n",
    	)
	};
	(10) => {
    	concat!(
        	"mov 16({dim_arrx}),{x0}", "\n",
        	"lea ({x0}, {x0}, 2), {x3}", "\n",
        	"lea ({cx}, {x3},), {x1}", "\n",
			"lea ({x1}, {x3},), {x2}", "\n",
			"lea ({x2}, {x3},), {x3}", "\n",
    	)
	};
	(9) => {
    	concat!(
        	"mov 16({dim_arrx}),{x0}", "\n",
        	"lea ({x0}, {x0}, 2), {x3}", "\n",
        	"lea ({cx}, {x3},), {x1}", "\n",
			"lea ({x1}, {x3},), {x2}", "\n",
    	)
	};
	(8) => {
    	concat!(
        	"mov 16({dim_arrx}),{x0}", "\n",
        	"lea ({x0}, {x0}, 2), {x3}", "\n",
        	"lea ({cx}, {x3},), {x1}", "\n",
			"lea ({x1}, {x3},), {x2}", "\n",
    	)
	};
	(7) => {
    	concat!(
        	"mov 16({dim_arrx}),{x0}", "\n",
        	"lea ({x0}, {x0}, 2), {x3}", "\n",
        	"lea ({cx}, {x3},), {x1}", "\n",
			"lea ({x1}, {x3},), {x2}", "\n",
    	)
	};
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
	(96,9) => {vzeroall!(5,31)};
	(96,8) => {vzeroall!(5,28)};
	(96,7) => {vzeroall!(5,25)};
	(96,6) => {vzeroall!(5,22)};
	(96,5) => {vzeroall!(5,19)};
	(96,4) => {vzeroall!(5,16)};
	(96,3) => {vzeroall!(5,13)};
	(96,2) => {vzeroall!(5,10)};
	(96,1) => {vzeroall!(5,7)};

	(64,15) => {vzeroall!(2,31)};
	(64,14) => {vzeroall!(2,29)};
	(64,13) => {vzeroall!(2,27)};
	(64,12) => {vzeroall!(2,25)};
	(64,11) => {vzeroall!(2,23)};
	(64,10) => {vzeroall!(2,21)};
	(64,9) => {vzeroall!(2,19)};
	(64,8) => {vzeroall!(2,17)};
	(64,7) => {vzeroall!(2,15)};
	(64,6) => {vzeroall!(2,13)};
	(64,5) => {vzeroall!(2,11)};
	(64,4) => {vzeroall!(2,9)};
	(64,3) => {vzeroall!(2,7)};
	(64,2) => {vzeroall!(2,5)};
	(64,1) => {vzeroall!(2,3)};

	(32,15) => {vzeroall!(17,31)};
	(32,14) => {vzeroall!(17,30)};
	(32,13) => {vzeroall!(17,29)};
	(32,12) => {vzeroall!(17,28)};
	(32,11) => {vzeroall!(17,27)};
	(32,10) => {vzeroall!(17,26)};
	(32,9) => {vzeroall!(17,25)};
	(32,8) => {vzeroall!(17,24)};
	(32,7) => {vzeroall!(17,23)};
	(32,6) => {vzeroall!(17,22)};
	(32,5) => {vzeroall!(17,21)};
	(32,4) => {vzeroall!(17,20)};
	(32,3) => {vzeroall!(17,19)};
	(32,2) => {vzeroall!(17,18)};
	(32,1) => {vzeroall!(17,17)};
}

macro_rules! asm_alpha_scale {
	(96,9) => {asm_alpha_scale_0!(5,31)};
	(96,8) => {asm_alpha_scale_0!(5,28)};
	(96,7) => {asm_alpha_scale_0!(5,25)};
	(96,6) => {asm_alpha_scale_0!(5,22)};
	(96,5) => {asm_alpha_scale_0!(5,19)};
	(96,4) => {asm_alpha_scale_0!(5,16)};
	(96,3) => {asm_alpha_scale_0!(5,13)};
	(96,2) => {asm_alpha_scale_0!(5,10)};
	(96,1) => {asm_alpha_scale_0!(5,7)};

	(64,15) => {asm_alpha_scale_0!(2,31)};
	(64,14) => {asm_alpha_scale_0!(2,29)};
	(64,13) => {asm_alpha_scale_0!(2,27)};
	(64,12) => {asm_alpha_scale_0!(2,25)};
	(64,11) => {asm_alpha_scale_0!(2,23)};
	(64,10) => {asm_alpha_scale_0!(2,21)};
	(64,9) => {asm_alpha_scale_0!(2,19)};
	(64,8) => {asm_alpha_scale_0!(2,17)};
	(64,7) => {asm_alpha_scale_0!(2,15)};
	(64,6) => {asm_alpha_scale_0!(2,13)};
	(64,5) => {asm_alpha_scale_0!(2,11)};
	(64,4) => {asm_alpha_scale_0!(2,9)};
	(64,3) => {asm_alpha_scale_0!(2,7)};
	(64,2) => {asm_alpha_scale_0!(2,5)};
	(64,1) => {asm_alpha_scale_0!(2,3)};

	(32,15) => {asm_alpha_scale_0!(17,31)};
	(32,14) => {asm_alpha_scale_0!(17,30)};
	(32,13) => {asm_alpha_scale_0!(17,29)};
	(32,12) => {asm_alpha_scale_0!(17,28)};
	(32,11) => {asm_alpha_scale_0!(17,27)};
	(32,10) => {asm_alpha_scale_0!(17,26)};
	(32,9) => {asm_alpha_scale_0!(17,25)};
	(32,8) => {asm_alpha_scale_0!(17,24)};
	(32,7) => {asm_alpha_scale_0!(17,23)};
	(32,6) => {asm_alpha_scale_0!(17,22)};
	(32,5) => {asm_alpha_scale_0!(17,21)};
	(32,4) => {asm_alpha_scale_0!(17,20)};
	(32,3) => {asm_alpha_scale_0!(17,19)};
	(32,2) => {asm_alpha_scale_0!(17,18)};
	(32,1) => {asm_alpha_scale_0!(17,17)};
}

macro_rules! inc_b {
	(S,15) => {
    	"add {x1},{bx} \n add {x1},{x3} \n add {x1},{x4} \n add {x1},{x5} \n add {x1},{x6} \n"
	};
	(S,14) => {
    	"add {x1},{bx} \n add {x1},{x3} \n add {x1},{x4} \n add {x1},{x5} \n add {x1},{x6} \n"
	};
	(S,13) => {
    	"add {x1},{bx} \n add {x1},{x3} \n add {x1},{x4} \n add {x1},{x5} \n add {x1},{x6} \n"
	};
	(S,12) => {
    	"add {x1},{bx} \n add {x1},{x3} \n add {x1},{x4} \n add {x1},{x5} \n"
	};
	(S,11) => {
    	"add {x1},{bx} \n add {x1},{x3} \n add {x1},{x4} \n add {x1},{x5} \n"
	};
	(S,10) => {
    	"add {x1},{bx} \n add {x1},{x3} \n add {x1},{x4} \n add {x1},{x5} \n"
	};
	(S,9) => {
    	"add {x1},{bx} \n add {x1},{x3} \n add {x1},{x4} \n"
	};
	(S,8) => {
    	"add {x1},{bx} \n add {x1},{x3} \n add {x1},{x4} \n"
	};
	(S,7) => {
    	"add {x1},{bx} \n add {x1},{x3} \n add {x1},{x4} \n"
	};
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
    	concat!(
        	"add $2*", $nr, ", {bx}", "\n",
    	)
	};
}

macro_rules! cum_seq {
	($step_macro:tt, $nr:tt, $layout:tt) => {
		seq!(n in 0..$nr {
			concat!(#($step_macro!(n, $layout),)*)
		})
	};
}

// macro_rules! acc_96x9 {
// 	(0, $layout:tt) => {
// 		acc_p!($layout, "0({cx})", 5, 6, 7)
// 	};
// 	(1, $layout:tt) => {
// 		acc_p!($layout, "0({cx}, {x0})", 8, 9, 10)
// 	};
// 	(2, $layout:tt) => {
// 		acc_p!($layout, "0({cx}, {x0}, 2)", 11, 12, 13)
// 	};
// 	(3, $layout:tt) => {
// 		acc_p!($layout, "0({x1})",  14, 15, 16)
// 	}; 
// 	(4, $layout:tt) => {
// 		acc_p!($layout, "0({x1}, {x0})",  17, 18, 19)
// 	};
// 	(5, $layout:tt) => {
// 		acc_p!($layout, "0({x1}, {x0}, 2)", 20, 21, 22)
// 	};
// 	(6, $layout:tt) => {
// 		acc_p!($layout, "0({x2})", 23, 24, 25)
// 	};
// 	(7, $layout:tt) => {
// 		acc_p!($layout, "0({x2}, {x0})", 26, 27, 28)
// 	};
// 	(8, $layout:tt) => {
// 		acc_p!($layout, "0({x2}, {x0}, 2)", 29, 30, 31)
// 	};
// }

// macro_rules! store_96x9 {
// 	(0, $layout:tt) => {
// 		storep!($layout, "0({cx})", 5, 6, 7)
// 	};
// 	(1, $layout:tt) => {
// 		storep!($layout, "0({cx}, {x0})", 8, 9, 10)
// 	};
// 	(2, $layout:tt) => {
// 		storep!($layout, "0({cx}, {x0}, 2)",  11, 12, 13)
// 	}; 
// 	(3, $layout:tt) => {
// 		storep!($layout, "0({x1})",  14, 15, 16)
// 	};
// 	(4, $layout:tt) => {
// 		storep!($layout, "0({x1}, {x0})", 17, 18, 19)
// 	};
// 	(5, $layout:tt) => {
// 		storep!($layout, "0({x1}, {x0}, 2)", 20, 21, 22)
// 	};
// 	(6, $layout:tt) => {
// 		storep!($layout, "0({x2})", 23, 24, 25)
// 	};
// 	(7, $layout:tt) => {
// 		storep!($layout, "0({x2}, {x0})", 26, 27, 28)
// 	};
// 	(8, $layout:tt) => {
// 		storep!($layout, "0({x2}, {x0}, 2)", 29, 30, 31)
// 	};
// }

macro_rules! acc_64x15 {
	(0, $layout:tt) => {
		acc_p!($layout, "0({cx})", 2, 3)
	};
	(1, $layout:tt) => {
		acc_p!($layout, "0({cx}, {x0})", 4, 5)
	};
	(2, $layout:tt) => {
		acc_p!($layout, "0({cx}, {x0}, 2)", 6, 7)
	}; 
	(3, $layout:tt) => {
		acc_p!($layout, "0({x1})", 8, 9)
	};
	(4, $layout:tt) => {
        acc_p!($layout, "0({x1}, {x0})", 10, 11)
	};
	(5, $layout:tt) => {
		acc_p!($layout, "0({x1}, {x0}, 2)", 12, 13)
	};
	(6, $layout:tt) => {
		acc_p!($layout, "0({x2})", 14, 15)
	};
	(7, $layout:tt) => {
		acc_p!($layout, "0({x2}, {x0})", 16, 17)
	};
	(8, $layout:tt) => {
		acc_p!($layout, "0({x2}, {x0}, 2)", 18, 19)
	};
	(9, $layout:tt) => {
		acc_p!($layout, "0({x3})", 20, 21)
	};
	(10, $layout:tt) => {
		acc_p!($layout, "0({x3}, {x0})", 22, 23)
	};
	(11, $layout:tt) => {
		acc_p!($layout, "0({x3}, {x0}, 2)", 24, 25)
	};
	(12, $layout:tt) => {
		acc_p!($layout, "0({x4})", 26, 27)
	};
	(13, $layout:tt) => {
		acc_p!($layout, "0({x4}, {x0})", 28, 29)
	};
	(14, $layout:tt) => {
		acc_p!($layout, "0({x4}, {x0}, 2)", 30, 31)
	};
}

macro_rules! store_64x15 {
	(0, $layout:tt) => {
		storep!($layout, "0({cx})", 2, 3)
	};
	(1, $layout:tt) => {
		storep!($layout, "0({cx}, {x0})", 4, 5)
	};
	(2, $layout:tt) => {
		storep!($layout, "0({cx}, {x0}, 2)", 6, 7)
	}; 
	(3, $layout:tt) => {
		storep!($layout, "0({x1})", 8, 9)
	};
	(4, $layout:tt) => {
    	storep!($layout, "0({x1}, {x0})", 10, 11)
	};
	(5, $layout:tt) => {
		storep!($layout, "0({x1}, {x0}, 2)", 12, 13)
	};
	(6, $layout:tt) => {
		storep!($layout, "0({x2})", 14, 15)
	};
	(7, $layout:tt) => {
		storep!($layout, "0({x2}, {x0})", 16, 17)
	};
	(8, $layout:tt) => {
		storep!($layout, "0({x2}, {x0}, 2)", 18, 19)
	};
	(9, $layout:tt) => {
		storep!($layout, "0({x3})", 20, 21)
	};
	(10, $layout:tt) => {
		storep!($layout, "0({x3}, {x0})", 22, 23)
	};
	(11, $layout:tt) => {
		storep!($layout, "0({x3}, {x0}, 2)", 24, 25)
	};
	(12, $layout:tt) => {
		storep!($layout, "0({x4})", 26, 27)
	};
	(13, $layout:tt) => {
		storep!($layout, "0({x4}, {x0})", 28, 29)
	};
	(14, $layout:tt) => {
		storep!($layout, "0({x4}, {x0}, 2)", 30, 31)
	};
}

macro_rules! acc_32x15 {
	(0, $layout:tt) => {
		acc_p!($layout, "0({cx})", 17)
	};
	(1, $layout:tt) => {
		acc_p!($layout, "0({cx}, {x0})", 18)
	};
	(2, $layout:tt) => {
		acc_p!($layout, "0({cx}, {x0}, 2)", 19)
	}; 
	(3, $layout:tt) => {
		acc_p!($layout, "0({x1})", 20)
	};
	(4, $layout:tt) => {
        acc_p!($layout, "0({x1}, {x0})", 21)
	};
	(5, $layout:tt) => {
		acc_p!($layout, "0({x1}, {x0}, 2)", 22)
	};
	(6, $layout:tt) => {
		acc_p!($layout, "0({x2})", 23)
	};
	(7, $layout:tt) => {
    	acc_p!($layout, "0({x2}, {x0})", 24)
	};
	(8, $layout:tt) => {
		acc_p!($layout, "0({x2}, {x0}, 2)", 25)
	};
	(9, $layout:tt) => {
		acc_p!($layout, "0({x3})", 26)
	};
	(10, $layout:tt) => {
        acc_p!($layout, "0({x3}, {x0})", 27)
	};
	(11, $layout:tt) => {
		acc_p!($layout, "0({x3}, {x0}, 2)", 28)
	};
	(12, $layout:tt) => {
		acc_p!($layout, "0({x4})", 29)
	};
	(13, $layout:tt) => {
        acc_p!($layout, "0({x4}, {x0})", 30)
	};
	(14, $layout:tt) => {
		acc_p!($layout, "0({x4}, {x0}, 2)", 31)
	};
}

macro_rules! store_32x15 {
	(0, $layout:tt) => {
		storep!($layout, "0({cx})", 17)
	};
	(1, $layout:tt) => {
		storep!($layout, "0({cx}, {x0})", 18)
	};
	(2, $layout:tt) => {
		storep!($layout, "0({cx}, {x0}, 2)", 19)
	}; 
	(3, $layout:tt) => {
		storep!($layout, "0({x1})", 20)
	};
	(4, $layout:tt) => {
        storep!($layout, "0({x1}, {x0})", 21)
	};
	(5, $layout:tt) => {
		storep!($layout, "0({x1}, {x0}, 2)", 22)
	};
	(6, $layout:tt) => {
		storep!($layout, "0({x2})", 23)
	};
	(7, $layout:tt) => {
        storep!($layout, "0({x2}, {x0})", 24)
	};
	(8, $layout:tt) => {
		storep!($layout, "0({x2}, {x0}, 2)", 25)
	};
	(9, $layout:tt) => {
		storep!($layout, "0({x3})", 26)
	};
	(10, $layout:tt) => {
        storep!($layout, "0({x3}, {x0})", 27)
	};
	(11, $layout:tt) => {
		storep!($layout, "0({x3}, {x0}, 2)", 28)
	};
	(12, $layout:tt) => {
		storep!($layout, "0({x4})", 29)
	};
	(13, $layout:tt) => {
        storep!($layout, "0({x4}, {x0})", 30)
	};
	(14, $layout:tt) => {
		storep!($layout, "0({x4}, {x0}, 2)", 31)
	};
}


macro_rules! load_a {
	($mr:tt, B) => {
    	loadp!($mr, B, "0({ax})")
	};
	($mr:tt, C) => {
    	loadp!($mr, C, "0({ax})")
	};
}

// macro_rules! fmadd_3v {
// 	(0, $m:expr) => {
// 		concat!(
// 			vfmadd!(0, $m, 5),
// 			vfmadd!(1, $m, 6),
// 			vfmadd!(2, $m, 7),
// 		)
// 	};
// 	(1, $m:expr) => {
// 		concat!(
// 			vfmadd!(0, $m, 8),
// 			vfmadd!(1, $m, 9),
// 			vfmadd!(2, $m, 10),
// 		)
// 	};
// 	(2, $m:expr) => {
// 		concat!(
// 			vfmadd!(0, $m, 11),
// 			vfmadd!(1, $m, 12),
// 			vfmadd!(2, $m, 13),
// 		)
// 	};
// 	(3, $m:expr) => {
// 		concat!(
// 			vfmadd!(0, $m, 14),
// 			vfmadd!(1, $m, 15),
// 			vfmadd!(2, $m, 16),
// 		)
// 	};
// 	(4, $m:expr) => {
// 		concat!(
// 			vfmadd!(0, $m, 17),
// 			vfmadd!(1, $m, 18),
// 			vfmadd!(2, $m, 19),
// 		)
// 	};
// 	(5, $m:expr) => {
// 		concat!(
// 			vfmadd!(0, $m, 20),
// 			vfmadd!(1, $m, 21),
// 			vfmadd!(2, $m, 22),
// 		)
// 	};
// 	(6, $m:expr) => {
// 		concat!(
// 			vfmadd!(0, $m, 23),
// 			vfmadd!(1, $m, 24),
// 			vfmadd!(2, $m, 25),
// 		)
// 	};
// 	(7, $m:expr) => {
// 		concat!(
// 			vfmadd!(0, $m, 26),
// 			vfmadd!(1, $m, 27),
// 			vfmadd!(2, $m, 28),
// 		)
// 	};
// 	(8, $m:expr) => {
// 		concat!(
// 			vfmadd!(0, $m, 29),
// 			vfmadd!(1, $m, 30),
// 			vfmadd!(2, $m, 31),
// 		)
// 	};
// }

macro_rules! fmadd_2v {
	(0, $m:expr) => {
		concat!(
			vfmadd!(0, $m, 2),
			vfmadd!(1, $m, 3),
		)
	};
	(1, $m:expr) => {
		concat!(
			vfmadd!(0, $m, 4),
			vfmadd!(1, $m, 5),
		)
	};
	(2, $m:expr) => {
		concat!(
			vfmadd!(0, $m, 6),
			vfmadd!(1, $m, 7),
		)
	};
	(3, $m:expr) => {
		concat!(
			vfmadd!(0, $m, 8),
			vfmadd!(1, $m, 9),
		)
	};
	(4, $m:expr) => {
		concat!(
			vfmadd!(0, $m, 10),
			vfmadd!(1, $m, 11),
		)
	};
	(5, $m:expr) => {
		concat!(
			vfmadd!(0, $m, 12),
			vfmadd!(1, $m, 13),
		)
	};
	(6, $m:expr) => {
		concat!(
			vfmadd!(0, $m, 14),
			vfmadd!(1, $m, 15),
		)
	};
	(7, $m:expr) => {
		concat!(
			vfmadd!(0, $m, 16),
			vfmadd!(1, $m, 17),
		)
	};
	(8, $m:expr) => {
		concat!(
			vfmadd!(0, $m, 18),
			vfmadd!(1, $m, 19),
		)
	};
	(9, $m:expr) => {
		concat!(
			vfmadd!(0, $m, 20),
			vfmadd!(1, $m, 21),
		)
	};
	(10, $m:expr) => {
		concat!(
			vfmadd!(0, $m, 22),
			vfmadd!(1, $m, 23),
		)
	};
	(11, $m:expr) => {
		concat!(
			vfmadd!(0, $m, 24),
			vfmadd!(1, $m, 25),
		)
	};
	(12, $m:expr) => {
		concat!(
			vfmadd!(0, $m, 26),
			vfmadd!(1, $m, 27),
		)
	};
	(13, $m:expr) => {
		concat!(
			vfmadd!(0, $m, 28),
			vfmadd!(1, $m, 29),
		)
	};
	(14, $m:expr) => {
		concat!(
			vfmadd!(0, $m, 30),
			vfmadd!(1, $m, 31),
		)
	};
}

macro_rules! fmadd_1v {
	(0, $m:expr) => {
		concat!(
			vfmadd!(0, $m, 17),
		)
	};
	(1, $m:expr) => {
		concat!(
			vfmadd!(0, $m, 18),
		)
	};
	(2, $m:expr) => {
		concat!(
			vfmadd!(0, $m, 19),
		)
	};
	(3, $m:expr) => {
		concat!(
			vfmadd!(0, $m, 20),
		)
	};
	(4, $m:expr) => {
		concat!(
			vfmadd!(0, $m, 21),
		)
	};
	(5, $m:expr) => {
		concat!(
			vfmadd!(0, $m, 22),
		)
	};
	(6, $m:expr) => {
		concat!(
			vfmadd!(0, $m, 23),
		)
	};
	(7, $m:expr) => {
		concat!(
			vfmadd!(0, $m, 24),
		)
	};
	(8, $m:expr) => {
		concat!(
			vfmadd!(0, $m, 25),
		)
	};
	(9, $m:expr) => {
		concat!(
			vfmadd!(0, $m, 26),
		)
	};
	(10, $m:expr) => {
		concat!(
			vfmadd!(0, $m, 27),
		)
	};
	(11, $m:expr) => {
		concat!(
			vfmadd!(0, $m, 28),
		)
	};
	(12, $m:expr) => {
		concat!(
			vfmadd!(0, $m, 29),
		)
	};
	(13, $m:expr) => {
		concat!(
			vfmadd!(0, $m, 30),
		)
	};
	(14, $m:expr) => {
		concat!(
			vfmadd!(0, $m, 31),
		)
	};
}

// // ***************************** 96x9 ******************************* //
// macro_rules! step_96x9 {
// 	(9, B, B) => {
// 		concat!(
// 			load_a!(96, B),
// 			"addq $192, {ax} \n",
// 			fmadd_3v!(0, bd!(B, 0)),
// 			fmadd_3v!(1, bd!(B, 1)),
// 			"prefetcht0 384({ax}) \n",
// 			"prefetcht0 32({bx}) \n",
// 			fmadd_3v!(2, bd!(B, 2)),
// 			fmadd_3v!(3, bd!(B, 3)),
// 			"prefetcht0 448({ax}) \n",
// 			fmadd_3v!(4, bd!(B, 4)),
// 			fmadd_3v!(5, bd!(B, 5)),
// 			"prefetcht0 512({ax}) \n",
// 			fmadd_3v!(6, bd!(B, 6)),
// 			fmadd_3v!(7, bd!(B, 7)),
// 			fmadd_3v!(8, bd!(B, 8)),
// 			"addq $18, {bx} \n",
// 		)
		
// 	};
// 	($nr:tt, $a_layout:tt, $b_layout:tt) => {
// 		seq!(n in 0..$nr {
// 			concat!(
// 				load_a!($mr, $a_layout),
// 				inc_a!($a_layout,96),
// 				"prefetcht0 64({bx}) \n",
// 				#(
// 					fmadd_3v!(n, bd!($b_layout, n)),
// 				)*
// 				inc_b!($b_layout,$nr), 
// 			)
// 		})
// 	};
// }

// ***************************** 64x15 ******************************* //
macro_rules! step_64x15 {
	(15, B, B) => {
		concat!(
			load_a!(64, B),
			"addq $128, {ax} \n",
			fmadd_2v!(0, bd!(B, 0)),
			fmadd_2v!(1, bd!(B, 1)),
			"prefetcht0 256({ax}) \n",
			fmadd_2v!(2, bd!(B, 2)),
			"prefetcht0 64({bx}) \n",
			fmadd_2v!(3, bd!(B, 3)),
			fmadd_2v!(4, bd!(B, 4)),
			fmadd_2v!(5, bd!(B, 5)),
			fmadd_2v!(6, bd!(B, 6)),
			fmadd_2v!(7, bd!(B, 7)),
			fmadd_2v!(8, bd!(B, 8)),
			fmadd_2v!(9, bd!(B, 9)),
			fmadd_2v!(10, bd!(B, 10)),
			"prefetcht0 320({ax}) \n",
			fmadd_2v!(11, bd!(B, 11)),
			fmadd_2v!(12, bd!(B, 12)),
			fmadd_2v!(13, bd!(B, 13)),
			fmadd_2v!(14, bd!(B, 14)),
			"addq $30, {bx} \n",
		)
		
	};
	($nr:tt, $a_layout:tt, $b_layout:tt) => {
		seq!(n in 0..$nr {
			concat!(
				load_a!(64, $a_layout),
				"addq $128, {ax} \n",
				"prefetcht0 64({bx}) \n",
				#(
					fmadd_2v!(n, bd!($b_layout, n)),
				)*
				inc_b!($b_layout,$nr), 
			)
		})
	};
}
// ***************************** 32x15 ******************************* //
macro_rules! step_32x15 {
	(15, B, B) => {
		concat!(
			load_a!(32, B),
			"addq $64, {ax} \n",
			fmadd_1v!(0, bd!(B, 0)),
			fmadd_1v!(1, bd!(B, 1)),
			"prefetcht0 256({ax}) \n",
			fmadd_1v!(2, bd!(B, 2)),
			"prefetcht0 64({bx}) \n",
			fmadd_1v!(3, bd!(B, 3)),
			fmadd_1v!(4, bd!(B, 4)),
			fmadd_1v!(5, bd!(B, 5)),
			fmadd_1v!(6, bd!(B, 6)),
			fmadd_1v!(7, bd!(B, 7)),
			fmadd_1v!(8, bd!(B, 8)),
			fmadd_1v!(9, bd!(B, 9)),
			fmadd_1v!(10, bd!(B, 10)),
			"prefetcht0 320({ax}) \n",
			fmadd_1v!(11, bd!(B, 11)),
			fmadd_1v!(12, bd!(B, 12)),
			fmadd_1v!(13, bd!(B, 13)),
			fmadd_1v!(14, bd!(B, 14)),
			"addq $30, {bx} \n",
		)
		
	};
	($nr:tt, $a_layout:tt, $b_layout:tt) => {
		seq!(n in 0..$nr {
			concat!(
				load_a!(32, $a_layout),
				"addq $64, {ax} \n",
				"prefetcht0 64({bx}) \n",
				#(
					fmadd_1v!(n, bd!($b_layout, n)),
				)*
				inc_b!($b_layout,$nr), 
			)
		})
	};
}


macro_rules! prefetch_c {
    (96, $nr:tt, $c:tt, $ldc:tt) => {
        seq!(j in 0..$nr {
            _mm_prefetch($c.add(0+j*$ldc) as *const i8, 3);
            _mm_prefetch($c.add(16+j*$ldc) as *const i8, 3);
			_mm_prefetch($c.add(32+j*$ldc) as *const i8, 3);
        });
    };
    (64, $nr:tt, $c:tt, $ldc:tt) => {
        seq!(j in 0..$nr {
            _mm_prefetch($c.add(0+j*$ldc) as *const i8, 3);
            _mm_prefetch($c.add(16+j*$ldc) as *const i8, 3);
			_mm_prefetch($c.add(32+j*$ldc) as *const i8, 3);
        });
    };
    (32, $nr:tt, $c:tt, $ldc:tt) => {
        seq!(j in 0..$nr {
            _mm_prefetch($c.add(16+j*$ldc) as *const i8, 3);
        });
    };
}

macro_rules! mask_ptr {
	(M, $m:expr, $nm:ident) => {
		let $nm = if $m % VS == 0 && $m > 0 { 0xFFFFFFFF } else { (1_u32 << ($m % VS)) - 1 };
	};
	(C, $m:expr, $nm:ident) => {
		let $nm = 0xFFFFFFFF_u32;
	};
}

macro_rules! load_mask_ptr_asm {
	(M) => {
		"kmovd ({maskx}), %k1"
	};
	(C) => {
		"/* {maskx} */"
	}
}

macro_rules! def_ukernel {
	(
		$step_macro:tt,
		$acc_macro:tt,
		$store_macro:tt,
    	$mr:tt, $nr:tt,
    	$a_layout:tt, $b_layout:tt,
		$is_partial:tt,
    	$func_name:ident
	) => {
    	pub(crate) unsafe fn $func_name<F: MyFn, const BUF: bool>(
        	a: *const TA, b: *const TB, c: *mut TC,
        	alpha: *const TA, beta: *const TB,
        	k: usize,
			d_arr: [usize; 4],
			m: usize,
			f: F,
    	) {
			mask_ptr!($is_partial, m, x);
			let mask_ptr = (&x) as *const u32;
			let k_iter = k / 4;
        	let k_left = k % 4;
            let mut dim_arr = [d_arr[0]*2, d_arr[1]*2, d_arr[3]*2, k_iter, k_left];
			let mut cf = c;
			let mut c_buf = [f16::ZERO;$mr*$nr];
			let c_cs = d_arr[3];
			if BUF {
				let c_rs = d_arr[2];
				if m != $mr || c_rs != 1 {
					for j in 0..$nr {
						for i in 0..m {
							c_buf[j*$mr+i] = *c.add(i*c_rs+j*c_cs);
						}
					}
					cf = c_buf.as_mut_ptr();
					dim_arr[2] = $mr*2;
				}
			}
			// prefetch for c
			use std::arch::x86_64::_mm_prefetch;
			prefetch_c!($mr,$nr,c,c_cs);
        	asm!(
				"/* {x6} */",
            	asm_vzeroall!($mr,$nr),
   	 
            	asm_init_ab!($mr,$a_layout,$b_layout),
           	 
            	// 3 -> CONSIDKLEFT
            	"je 3f",
           	 
            	// 2 -> KITER
            	"2:",
				$step_macro!($nr, $a_layout, $b_layout),
				$step_macro!($nr, $a_layout, $b_layout),
				$step_macro!($nr, $a_layout, $b_layout),
				$step_macro!($nr, $a_layout, $b_layout),
   	 
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
            	$step_macro!($nr, $a_layout, $b_layout),

            	"dec {x0}",
   	 
            	// 4 -> KLEFT
            	"jne 4b",
   	 
            	// 5 -> POSTACCUM
            	"5:",
            	asm_c_load!($nr),
            	// scale by alpha
            	asm_alpha_scale!($mr, $nr),

				load_mask_ptr_asm!($is_partial),
				load_beta!(),
            	// 6 -> BETAZERO
            	"je 6f",
				cum_seq!($acc_macro,$nr,$is_partial),

            	// 6 -> BETAZERO
            	"6:",
				cum_seq!($store_macro,$nr,$is_partial),
				
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
				x4 = out(reg) _,
            	x5 = out(reg) _,
				x6 = out(reg) _,
            	out("xmm0") _, out("xmm1") _, out("xmm2") _, out("xmm3") _,
            	out("xmm4") _, out("xmm5") _, out("xmm6") _, out("xmm7") _,
            	out("xmm8") _, out("xmm9") _, out("xmm10") _, out("xmm11") _,
            	out("xmm12") _, out("xmm13") _, out("xmm14") _, out("xmm15") _,
				out("k1") _,
            	options(att_syntax)
        	);
			if BUF {
				let c_rs = d_arr[2];
				if m != $mr || c_rs != 1 {
					for j in 0..$nr {
						for i in 0..m {
							*c.add(i*c_rs+j*c_cs) = c_buf[j*$mr+i];
						}
					}
				}
			}
			for j in 0..$nr {
				for i in 0..$mr/16 {
					f.call(c.add(i*16+j*c_cs), 8);
				}
			}
    	}
	};
}


macro_rules! def_ukernelxn {
	(
		$step_macro:tt,
		$acc_macro:tt,
		$store_macro:tt,
    	$mr:tt, $nr:tt,
    	$a_layout:tt, $b_layout:tt,
		$is_partial:tt,
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
			let mask_ptr = (&x) as *const u32;
			let k_iter = k / 4;
        	let k_left = k % 4;
            let mut dim_arr = [d_arr[0]*2, d_arr[1]*2, d_arr[3]*2, k_iter, k_left];
			let mut cf = c;
			let mut c_buf = [f16::ZERO;$mr*$nr];
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
					dim_arr[2] = $mr*2;
				}
			}
			use std::arch::x86_64::_mm_prefetch;
			let _ = 'blk: {
				seq!(ni in 1..$nr {
					if ni == n {
						// prefetch for c
						prefetch_c!($mr,ni,c,c_cs);
						asm!(
							"/* {x6} */",
							asm_vzeroall!($mr,ni),
				
							asm_init_ab!($mr,$a_layout,$b_layout),
						
							// 3 -> CONSIDKLEFT
							"je 3f",
						
							// 2 -> KITER
							"2:",
							$step_macro!(ni, $a_layout, $b_layout),
							$step_macro!(ni, $a_layout, $b_layout),
							$step_macro!(ni, $a_layout, $b_layout),
							$step_macro!(ni, $a_layout, $b_layout),
				
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
							$step_macro!(ni, $a_layout, $b_layout),

							"dec {x0}",
				
							// 4 -> KLEFT
							"jne 4b",
				
							// 5 -> POSTACCUM
							"5:",
							asm_c_load!(ni),
							// scale by alpha
							asm_alpha_scale!($mr, ni),

							load_beta!(),

							load_mask_ptr_asm!($is_partial),				
							// 6 -> BETAZERO
							"je 6f",
							cum_seq!($acc_macro,ni,$is_partial),

							// 6 -> BETAZERO
							"6:",
							cum_seq!($store_macro,ni,$is_partial),
							
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
							x4 = out(reg) _,
							x5 = out(reg) _,
							x6 = out(reg) _,
							out("xmm0") _, out("xmm1") _, out("xmm2") _, out("xmm3") _,
							out("xmm4") _, out("xmm5") _, out("xmm6") _, out("xmm7") _,
							out("xmm8") _, out("xmm9") _, out("xmm10") _, out("xmm11") _,
							out("xmm12") _, out("xmm13") _, out("xmm14") _, out("xmm15") _,
							out("k1") _,
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
				for i in 0..$mr/16 {
					f.call(c.add(i*16+j*c_cs), 16);
				}
			}
    	}
	};
}

// def_ukernel!(step_96x9, acc_96x9, store_96x9, 96, 9, B, B, C, ukernel_96x9_bb);
// def_ukernel!(step_64x15, acc_64x15, store_64x15, 32, 8, B, B, C, ukernel_32x8_bb);
// def_ukernel!(step_32x15, acc_32x15, store_32x15, 16, 8, B, B, C, ukernel_16x8_bb);

// def_ukernel!(step_96x9, acc_96x9, store_96x9, 96, 9, B, B, M, ukernel_96x9_bb_partial);
def_ukernel!(step_64x15, acc_64x15, store_64x15, 64, 15, B, B, M, ukernel_64x15_bb_partial);
def_ukernel!(step_32x15, acc_32x15, store_32x15, 32, 15, B, B, M, ukernel_32x15_bb_partial);

// def_ukernel!(96, step_96x9, acc_96x9, store_96x9, 96, 9, B, S, C, ukernel_96x9_bs);
def_ukernel!(step_64x15, acc_64x15, store_64x15, 64, 15, B, S, C, ukernel_64x15_bs);

// def_ukernel!(96, step_96x9, acc_96x9, store_96x9, 96, 9, B, S, M, ukernel_96x9_bs_partial);
def_ukernel!(step_64x15, acc_64x15, store_64x15, 64, 15, B, S, M, ukernel_64x15_bs_partial);
def_ukernel!(step_32x15, acc_32x15, store_32x15, 32, 15, B, S, M, ukernel_32x15_bs_partial);


// def_ukernelxn!(step_96x9, acc_96x9, store_96x9, 96, 9, B, B, C, ukernel_96xn_bb);
def_ukernelxn!(step_64x15, acc_64x15, store_64x15, 64, 15, B, B, C, ukernel_64xn_bb);
// def_ukernelxn!(step_64x15, acc_64x15, store_64x15, 32, 7, B, B, C, ukernel_32xn_bb);
// def_ukernelxn!(step_32x15, acc_32x15, store_32x15, 16, 7, B, B, C, ukernel_16xn_bb);

// def_ukernelxn!(step_96x9, acc_96x9, store_96x9, 96, 9, B, B, M, ukernel_96xn_bb_partial);
def_ukernelxn!(step_64x15, acc_64x15, store_64x15, 64, 15, B, B, M, ukernel_64xn_bb_partial);
def_ukernelxn!(step_32x15, acc_32x15, store_32x15, 32, 15, B, B, M, ukernel_32xn_bb_partial);

// def_ukernelxn!(step_96x9, acc_96x9, store_96x9, 96, 9, B, S, C, ukernel_96xn_bs);
def_ukernelxn!(step_64x15, acc_64x15, store_64x15, 64, 15, B, S, C, ukernel_64xn_bs);

// def_ukernelxn!(step_96x9, acc_96x9, store_96x9, 96, 9, B, S, M, ukernel_96xn_bs_partial);
def_ukernelxn!(step_64x15, acc_64x15, store_64x15, 64, 15, B, S, M, ukernel_64xn_bs_partial);
def_ukernelxn!(step_32x15, acc_32x15, store_32x15, 32, 15, B, S, M, ukernel_32xn_bs_partial);



// pub(crate) unsafe fn ukernel_96x9_bb<F: MyFn, const BUF: bool>(
//     a: *const TA, b: *const TB, c: *mut TC,
//     alpha: *const TA, beta: *const TB,
//     k: usize,
//     d_arr: [usize; 4],
//     a_pft1_offset: usize,
//     f: F,
// ) {
// 	let k_left0 = k % 8;
// 	let k_left = if k_left0 == 0 {8} else {k_left0};
// 	let k_iter = (k - k_left) / 4;
//     let mut dim_arr = [d_arr[3]*2, k_iter, k_left, a_pft1_offset, a_pft1_offset];
//     let mut cf = c;
//     let mut c_buf = [f16::ZERO; 96 * 8];
// 	let c_cs = d_arr[3];
//     if BUF {
//         let c_rs = d_arr[2];
//         if c_rs != 1 {
//             for j in 0..8 {
//                 for i in 0..96 {
//                     c_buf[j * 96 + i] = *c.add(i * c_rs + j * c_cs);
//                 }
//             }
//             cf = c_buf.as_mut_ptr();
//             dim_arr[2] = 96 * 2;
//         }
//     }
//     asm!(
// 		asm_vzeroall!(96,9),
// 		"mov 8({dim_arrx}),{x0}",
// 		"test {x0},{x0}",
// 		"je 3f",
// 		// "je 3f",
// 		"mov {cx}, {x2}",
// 		"mov {ax}, {x5}",
// 		"mov 24({dim_arrx}),{x1}",
// 		"add {x1}, {x5}",

// 		// "mov {bx}, {x3}",
// 		// "mov 24({dim_arrx}),{x1}",
// 		// "add {x1}, {x3}",
// 		"mov ({dim_arrx}),{x1}",
// 		"2:",
// 		// loadp_unit!(C, "0({bx})", 3),
// 		step_96x9!(9, B, B),

// 		// "prefetcht1 ({x3})",
// 		// "addq $32, {x3}",

// 		"movq $64*4, {x4}",
// 		// divisiblity by 4
// 		"testq $3, {x0}",
// 		"cmovz {x1},{x4}",

// 		step_96x9!(9, B, B),

// 		"prefetcht1 ({x2})",

// 		"subq $64*3, {x2}",
// 		"addq {x4}, {x2}",

// 		step_96x9!(9, B, B),

// 		"prefetcht1 ({x5})",
// 		"addq $32, {x5}",

// 		"testq $63, {x0}",
// 		"cmovz {cx},{x2}",

// 		step_96x9!(9, B, B),
// 		// "addq $64, {bx} \n",


// 		"dec {x0}",
// 		"jne 2b",
// 		"3:",
// 		"mov 16({dim_arrx}),{x0}",
// 		"test {x0},{x0}",

// 		// 5 -> POSTACCUM
// 		"je 5f",
// 		"mov {cx}, {x2}",
// 		"mov ({dim_arrx}),{x1}",
// 		"4:",
// 		"prefetcht0 ({x2})",
// 		"prefetcht0 64({x2})",
// 		"prefetcht0 128({x2})",
// 		step_96x9!(9, B, B),

// 		"add {x1}, {x2}",
// 		"dec {x0}",
// 		"jne 4b",
// 		"5:",
// 		"mov ({dim_arrx}),{x0}",
// 		"lea ({x0}, {x0}, 2), {x3}",
// 		"lea ({cx}, {x3},), {x1}",
// 		"lea ({x1}, {x3},), {x2}",
// 		// scale by alpha
// 		asm_alpha_scale!(96, 9),
// 		load_beta!(),

// 		// 6 -> BETAZERO
// 		"je 6f",
//		cum_seq!(acc_96x9,9,C),

// 		// 6 -> BETAZERO
// 		"6:",
//		cum_seq!(store_96x9,9,C),

// 		"7:",
//         ax = inout(reg) a => _, 
// 		bx = inout(reg) b => _, 
// 		cx = inout(reg) cf => _,
// 		dim_arrx = inout(reg) dim_arr.as_ptr() => _, 
// 		alphax = inout(reg) alpha => _, 
// 		betax = inout(reg) beta => _, 
// 		x0 = out(reg) _, 
// 		x1 = out(reg)_, 
// 		x2 = out(reg) _, 
// 		x3 = out(reg) _, 
// 		x4 = out(reg) _,
// 		x5 = out(reg) _, 
// 		out("xmm0") _, out("xmm1") _,
//         out("xmm2") _, out("xmm3") _, out("xmm4") _, out("xmm5") _, out("xmm6") _,
//         out("xmm7") _, out("xmm8") _, out("xmm9") _, out("xmm10") _, out("xmm11") _,
//         out("xmm12") _, out("xmm13") _, out("xmm14") _, out("xmm15") _,
//         options(att_syntax)
//     );
//     if BUF {
//         let c_rs = d_arr[2];
//         if c_rs != 1 {
//             for j in 0..8 {
//                 for i in 0..96 {
//                     *c.add(i * c_rs + j * c_cs) = c_buf[j * 96 + i];
//                 }
//             }
//         }
//     }
//     for j in 0..8 {
//         for i in 0..96 / 16 {
//             f.call(c.add(i * 16 + j * c_cs), 16);
//         }
//     }
// }


pub(crate) unsafe fn ukernel_64x15_bb<F: MyFn, const BUF: bool>(
    a: *const TA, b: *const TB, c: *mut TC,
    alpha: *const TA, beta: *const TB,
    k: usize,
    d_arr: [usize; 4],
    a_pft1_offset: usize,
    f: F,
) {
	let k_left0 = k % 12;
	let k_left = if k_left0 == 0 {12} else {k_left0};
	let k_iter = (k - k_left) / 4;
    let mut dim_arr = [d_arr[3]*2, k_iter, k_left, a_pft1_offset];
    let mut cf = c;
    let mut c_buf = [f16::ZERO; 96 * 8];
	let c_cs = d_arr[3];
    if BUF {
        let c_rs = d_arr[2];
        if c_rs != 1 {
            for j in 0..8 {
                for i in 0..96 {
                    c_buf[j * 96 + i] = *c.add(i * c_rs + j * c_cs);
                }
            }
            cf = c_buf.as_mut_ptr();
            dim_arr[2] = 96 * 2;
        }
    }
    asm!(
		asm_vzeroall!(64,15),
		"mov 8({dim_arrx}),{x0}",
		"test {x0},{x0}",
		"je 3f",
		// "je 3f",
		"mov {cx}, {x2}",
		"mov {ax}, {x5}",
		"mov 24({dim_arrx}),{x1}",
		"add {x1}, {x5}",
		"mov ({dim_arrx}),{x1}",
		"2:",
		step_64x15!(15, B, B),

		// "prefetcht1 ({x5})",
		// "addq $64, {x5}",

		"movq $64*4, {x4}",
		// divisiblity by 4
		"testq $3, {x0}",
		"cmovz {x1},{x4}",

		step_64x15!(15, B, B),

		"prefetcht1 ({x2})",

		"subq $64*3, {x2}",
		"addq {x4}, {x2}",

		step_64x15!(15, B, B),

		"prefetcht1 ({x5})",
		"addq $64, {x5}",

		"testq $63, {x0}",
		"cmovz {cx},{x2}",

		step_64x15!(15, B, B),

		"dec {x0}",
		"jne 2b",
		"3:",
		"mov 16({dim_arrx}),{x0}",
		"test {x0},{x0}",

		// 5 -> POSTACCUM
		"je 5f",
		"mov {cx}, {x2}",
		"mov ({dim_arrx}),{x1}",
		"4:",
		"prefetcht0 ({x2})",
		"prefetcht0 64({x2})",
		// "prefetcht0 128({x2})",
		step_64x15!(15, B, B),

		"add {x1}, {x2}",
		"dec {x0}",
		"jne 4b",
		"5:",
		"mov ({dim_arrx}),{x0}",
		"lea ({x0}, {x0}, 2), {x4}",
		"lea ({cx}, {x4},), {x1}",
		"lea ({x1}, {x4},), {x2}",
		"lea ({x2}, {x4},), {x3}",
		"lea ({x3}, {x4},), {x4}",
		// scale by alpha
		asm_alpha_scale!(64, 15),
		// "/* {alphax} */",
		load_beta!(),

		// 6 -> BETAZERO
		"je 6f",
		cum_seq!(acc_64x15,15,C),

		// 6 -> BETAZERO
		"6:",
		cum_seq!(store_64x15,15,C),

		"7:",
        ax = inout(reg) a => _, 
		bx = inout(reg) b => _, 
		cx = inout(reg) cf => _,
		dim_arrx = inout(reg) dim_arr.as_ptr() => _, 
		alphax = inout(reg) alpha => _, 
		betax = inout(reg) beta => _, 
		x0 = out(reg) _, 
		x1 = out(reg)_, 
		x2 = out(reg) _, 
		x3 = out(reg) _, 
		x4 = out(reg) _,
		x5 = out(reg) _, 
		out("xmm0") _, out("xmm1") _,
        out("xmm2") _, out("xmm3") _, out("xmm4") _, out("xmm5") _, out("xmm6") _,
        out("xmm7") _, out("xmm8") _, out("xmm9") _, out("xmm10") _, out("xmm11") _,
        out("xmm12") _, out("xmm13") _, out("xmm14") _, out("xmm15") _,
        options(att_syntax)
    );
    if BUF {
        let c_rs = d_arr[2];
        if c_rs != 1 {
            for j in 0..8 {
                for i in 0..96 {
                    *c.add(i * c_rs + j * c_cs) = c_buf[j * 96 + i];
                }
            }
        }
    }
    for j in 0..8 {
        for i in 0..96 / 16 {
            f.call(c.add(i * 16 + j * c_cs), 16);
        }
    }
}
