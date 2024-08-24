use seq_macro::seq;
use std::arch::asm;
use super::VS;
use crate::{TA, TB, TC};
use crate::MyFn;


macro_rules! beta_fmadd {
    (C, $m0:expr, $r1:expr) => {
        concat!(
            "vfmadd231ps ", $m0, ",%zmm0,%zmm", $r1, "\n",
        ) 
    };
    (M, $m0:expr, $r1:expr) => {
        concat!(
            // "vmovups ", $m0, ", %zmm1", ", %zmm2",  "\n",
			"vmovups ", $m0, ", %zmm1 {{%k1}}", "\n",
			"vfmadd231ps %zmm1,%zmm0,%zmm", $r1, "\n",
        )
    };
	(C, $m0:expr) => {
		concat!(
			"vfmadd231ps ", $m0, ",%zmm0,%zmm", 0, "\n",
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
		"vbroadcastss"
	};
}

 macro_rules! vfmadd {
    ($r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            "vfmadd231ps %zmm", $r1, ", %zmm", $r2,", %zmm", $r3, "\n",
        ) 
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
			"vmovups %zmm", $r1, ", ", $m0, " {{%k1}}\n",
		)
    };
}


macro_rules! asm_alpha_scale_0 {
	($r0:tt, $r1:tt) => {
		seq!(r in $r0..=$r1 {
			concat!(
				vbroadcast!(), " ({alphax}),%zmm1", "\n",
				#(
					"vmulps %zmm1, %zmm", r, ",%zmm", r, "\n",
				)*
			)
		})
	}
}

macro_rules! load_beta {
	() => {
		concat!(
			vbroadcast!(), " ({betax}), %zmm0\n",
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
    (48, $layout:tt, $m0:expr, $r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            beta_fmadd!(C, $m0, $r1),
            beta_fmadd!(C, mem!($m0, "0x40"), $r2),
            beta_fmadd!($layout, mem!($m0, "0x80"), $r3),
        )
    };
    (32, $layout:tt, $m0:expr, $r1:expr, $r2:expr) => {
        concat!(
            beta_fmadd!(C, $m0, $r1),
            beta_fmadd!($layout, mem!($m0, "0x40"), $r2),
        )
    };
    ($N:tt, $layout:tt, $m0:expr, $r1:expr) => {
        concat!(
            beta_fmadd!($layout, $m0, $r1),
        )
    };
 }

 macro_rules! loadp {
    (48, $layout:tt, $m0:expr) => {
        concat!(
            loadp_unit!($layout, $m0, 0),
            loadp_unit!($layout, mem!($m0, "0x40"), 1),
            loadp_unit!($layout, mem!($m0, "0x80"), 2),
        )
    };
    (32, $layout:tt, $m0:expr) => {
        concat!(
            loadp_unit!($layout, $m0, 0),
            loadp_unit!($layout, mem!($m0, "0x40"), 1),
        )
    };
    (16, $layout:tt, $m0:expr) => {
        concat!(
            loadp_unit!($layout, $m0, 0),
        )
    };
    (8, $layout:tt, $m0:expr, $r1:expr) => {
        concat!(
            loadp_unit!($layout, $m0, $r1),
        )
    };
    (4, $layout:tt, $m0:expr) => {
        concat!(
            loadp_unit!(4, $layout, $m0, 0),
        )
    };
 }


macro_rules! storep {
	(48, $layout:tt, $m0:expr, $r1:expr, $r2:expr, $r3:expr) => {
		concat!(
			storep_unit!(C, $r1, $m0),
			storep_unit!(C, $r2, mem!($m0, "0x40")),
			storep_unit!($layout, $r3, mem!($m0, "0x80")),
		)
	};
	(32, $layout:tt, $m0:expr, $r1:expr, $r2:expr) => {
		concat!(
			storep_unit!(C, $r1, $m0),
			storep_unit!($layout, $r2, mem!($m0, "0x40")),
		)
	};
	(16, $layout:tt, $m0:expr, $r1:expr) => {
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
        	"lea ({x2}, {x2}, 2), {x5}", "\n",
        	"lea ({bx}, {x5}, 1), {x3}", "\n",
			"lea ({x3}, {x5}, 1), {x4}", "\n",
			"lea ({x4}, {x5}, 1), {x5}", "\n",

        	"mov 24({dim_arrx}),{x0}", "\n",
        	"test {x0},{x0}", "\n",
    	)
	};
}


macro_rules! asm_c_load {
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

	(VER3,8) => {vzeroall!(8,31)};
	(VER3,7) => {vzeroall!(8,28)};
	(VER3,6) => {vzeroall!(8,25)};
	(VER3,5) => {vzeroall!(8,22)};
	(VER3,4) => {vzeroall!(8,19)};
	(VER3,3) => {vzeroall!(8,16)};
	(VER3,2) => {vzeroall!(8,13)};
	(VER3,1) => {vzeroall!(8,10)};

	(VER2,12) => {vzeroall!(8,31)};
	(VER2,11) => {vzeroall!(8,29)};
	(VER2,10) => {vzeroall!(8,27)};
	(VER2,9) => {vzeroall!(8,25)};
	(VER2,8) => {vzeroall!(8,23)};
	(VER2,7) => {vzeroall!(8,21)};
	(VER2,6) => {vzeroall!(8,19)};
	(VER2,5) => {vzeroall!(8,17)};
	(VER2,4) => {vzeroall!(8,15)};
	(VER2,3) => {vzeroall!(8,13)};
	(VER2,2) => {vzeroall!(8,11)};
	(VER2,1) => {vzeroall!(8,9)};

	(VER1,12) => {vzeroall!(20,31)};
	(VER1,11) => {vzeroall!(20,30)};
	(VER1,10) => {vzeroall!(20,29)};
	(VER1,9) => {vzeroall!(20,28)};
	(VER1,8) => {vzeroall!(20,27)};
	(VER1,7) => {vzeroall!(20,26)};
	(VER1,6) => {vzeroall!(20,25)};
	(VER1,5) => {vzeroall!(20,24)};
	(VER1,4) => {vzeroall!(20,23)};
	(VER1,3) => {vzeroall!(20,22)};
	(VER1,2) => {vzeroall!(20,21)};
	(VER1,1) => {vzeroall!(20,20)};
}

macro_rules! asm_alpha_scale {
	(VER3, 8) => {asm_alpha_scale_0!(8,31)};
	(VER3, 7) => {asm_alpha_scale_0!(8,28)};
	(VER3, 6) => {asm_alpha_scale_0!(8,25)};
	(VER3, 5) => {asm_alpha_scale_0!(8,22)};
	(VER3, 4) => {asm_alpha_scale_0!(8,19)};
	(VER3, 3) => {asm_alpha_scale_0!(8,16)};
	(VER3, 2) => {asm_alpha_scale_0!(8,13)};
	(VER3, 1) => {asm_alpha_scale_0!(8,10)};

	(VER2, 12) => {asm_alpha_scale_0!(8,31)};
	(VER2, 11) => {asm_alpha_scale_0!(8,29)};
	(VER2, 10) => {asm_alpha_scale_0!(8,27)};
	(VER2, 9) => {asm_alpha_scale_0!(8,25)};
	(VER2, 8) => {asm_alpha_scale_0!(8,23)};
	(VER2, 7) => {asm_alpha_scale_0!(8,21)};
	(VER2, 6) => {asm_alpha_scale_0!(8,19)};
	(VER2, 5) => {asm_alpha_scale_0!(8,17)};
	(VER2, 4) => {asm_alpha_scale_0!(8,15)};
	(VER2, 3) => {asm_alpha_scale_0!(8,13)};
	(VER2, 2) => {asm_alpha_scale_0!(8,11)};
	(VER2, 1) => {asm_alpha_scale_0!(8,9)};

	(VER1, 12) => {asm_alpha_scale_0!(20,31)};
	(VER1, 11) => {asm_alpha_scale_0!(20,30)};
	(VER1, 10) => {asm_alpha_scale_0!(20,29)};
	(VER1, 9) => {asm_alpha_scale_0!(20,28)};
	(VER1, 8) => {asm_alpha_scale_0!(20,27)};
	(VER1, 7) => {asm_alpha_scale_0!(20,26)};
	(VER1, 6) => {asm_alpha_scale_0!(20,25)};
	(VER1, 5) => {asm_alpha_scale_0!(20,24)};
	(VER1, 4) => {asm_alpha_scale_0!(20,23)};
	(VER1, 3) => {asm_alpha_scale_0!(20,22)};
	(VER1, 2) => {asm_alpha_scale_0!(20,21)};
	(VER1, 1) => {asm_alpha_scale_0!(20,20)};
}

macro_rules! inc_a {
	(C) => {
    	"add {x1}, {ax} \n"
	};
	(B, $mr:tt) => {
    	concat!(
        	"add $4*", $mr, ", {ax}", "\n",
    	)
	};
}

macro_rules! inc_b {
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
        	"add $4*", $nr, ", {bx}", "\n",
    	)
	};
}

macro_rules! acc_48x8_seq {
	($mr:tt, 0, $layout:tt) => {
		acc_p!($mr, $layout, "0({cx})", 8, 9, 10)
	};
	($mr:tt, 1, $layout:tt) => {
		acc_p!($mr, $layout, "0({cx}, {x0})", 11, 12, 13)
	};
	($mr:tt, 2, $layout:tt) => {
		acc_p!($mr, $layout, "0({cx}, {x0}, 2)",  14, 15, 16)
	}; 
	($mr:tt, 3, $layout:tt) => {
		acc_p!($mr, $layout, "0({x1})",  17, 18, 19)
	};
	($mr:tt, 4, $layout:tt) => {
		acc_p!($mr, $layout, "0({x1}, {x0})", 20, 21, 22)
	};
	($mr:tt, 5, $layout:tt) => {
		acc_p!($mr, $layout, "0({x1}, {x0}, 2)", 23, 24, 25)
	};
	($mr:tt, 6, $layout:tt) => {
		acc_p!($mr, $layout, "0({x2})", 26, 27, 28)
	};
	($mr:tt, 7, $layout:tt) => {
		acc_p!($mr, $layout, "0({x2}, {x0})", 29, 30, 31)
	};
}

macro_rules! store_48x8_seq {
	($mr:tt, 0, $layout:tt) => {
		storep!($mr, $layout, "0({cx})", 8, 9, 10)
	};
	($mr:tt, 1, $layout:tt) => {
		storep!($mr, $layout, "0({cx}, {x0})", 11, 12, 13)
	};
	($mr:tt, 2, $layout:tt) => {
		storep!($mr, $layout, "0({cx}, {x0}, 2)",  14, 15, 16)
	}; 
	($mr:tt, 3, $layout:tt) => {
		storep!($mr, $layout, "0({x1})",  17, 18, 19)
	};
	($mr:tt, 4, $layout:tt) => {
		storep!($mr, $layout, "0({x1}, {x0})", 20, 21, 22)
	};
	($mr:tt, 5, $layout:tt) => {
		storep!($mr, $layout, "0({x1}, {x0}, 2)", 23, 24, 25)
	};
	($mr:tt, 6, $layout:tt) => {
		storep!($mr, $layout, "0({x2})", 26, 27, 28)
	};
	($mr:tt, 7, $layout:tt) => {
		storep!($mr, $layout, "0({x2}, {x0})", 29, 30, 31)
	};
}

macro_rules! acc_32x12_seq {
	($mr:tt, 0, $layout:tt) => {
		acc_p!($mr, $layout, "0({cx})", 8, 9)
	};
	($mr:tt, 1, $layout:tt) => {
		acc_p!($mr, $layout, "0({cx}, {x0})", 10, 11)
	};
	($mr:tt, 2, $layout:tt) => {
		acc_p!($mr, $layout, "0({cx}, {x0}, 2)", 12, 13)
	}; 
	($mr:tt, 3, $layout:tt) => {
		acc_p!($mr, $layout, "0({x1})", 14, 15)
	};
	($mr:tt, 4, $layout:tt) => {
        acc_p!($mr, $layout, "0({x1}, {x0})", 16, 17)
	};
	($mr:tt, 5, $layout:tt) => {
		acc_p!($mr, $layout, "0({x1}, {x0}, 2)", 18, 19)
	};
	($mr:tt, 6, $layout:tt) => {
		acc_p!($mr, $layout, "0({x2})", 20, 21)
	};
	($mr:tt, 7, $layout:tt) => {
		acc_p!($mr, $layout, "0({x2}, {x0})", 22, 23)
	};
	($mr:tt, 8, $layout:tt) => {
		acc_p!($mr, $layout, "0({x2}, {x0}, 2)", 24, 25)
	};
	($mr:tt, 9, $layout:tt) => {
		acc_p!($mr, $layout, "0({x3})", 26, 27)
	};
	($mr:tt, 10, $layout:tt) => {
		acc_p!($mr, $layout, "0({x3}, {x0})", 28, 29)
	};
	($mr:tt, 11, $layout:tt) => {
		acc_p!($mr, $layout, "0({x3}, {x0}, 2)", 30, 31)
	};
}

macro_rules! store_32x12_seq {
	($mr:tt, 0, $layout:tt) => {
		storep!($mr, $layout, "0({cx})", 8, 9)
	};
	($mr:tt, 1, $layout:tt) => {
		storep!($mr, $layout, "0({cx}, {x0})", 10, 11)
	};
	($mr:tt, 2, $layout:tt) => {
		storep!($mr, $layout, "0({cx}, {x0}, 2)", 12, 13)
	}; 
	($mr:tt, 3, $layout:tt) => {
		storep!($mr, $layout, "0({x1})", 14, 15)
	};
	($mr:tt, 4, $layout:tt) => {
    	storep!($mr, $layout, "0({x1}, {x0})", 16, 17)
	};
	($mr:tt, 5, $layout:tt) => {
		storep!($mr, $layout, "0({x1}, {x0}, 2)", 18, 19)
	};
	($mr:tt, 6, $layout:tt) => {
		storep!($mr, $layout, "0({x2})", 20, 21)
	};
	($mr:tt, 7, $layout:tt) => {
		storep!($mr, $layout, "0({x2}, {x0})", 22, 23)
	};
	($mr:tt, 8, $layout:tt) => {
		storep!($mr, $layout, "0({x2}, {x0}, 2)", 24, 25)
	};
	($mr:tt, 9, $layout:tt) => {
		storep!($mr, $layout, "0({x3})", 26, 27)
	};
	($mr:tt, 10, $layout:tt) => {
		storep!($mr, $layout, "0({x3}, {x0})", 28, 29)
	};
	($mr:tt, 11, $layout:tt) => {
		storep!($mr, $layout, "0({x3}, {x0}, 2)", 30, 31)
	};
}

macro_rules! acc_16x12_seq {
	($mr:tt, 0, $layout:tt) => {
		acc_p!($mr, $layout, "0({cx})", 20)
	};
	($mr:tt, 1, $layout:tt) => {
		acc_p!($mr, $layout, "0({cx}, {x0})", 21)
	};
	($mr:tt, 2, $layout:tt) => {
		acc_p!($mr, $layout, "0({cx}, {x0}, 2)", 22)
	}; 
	($mr:tt, 3, $layout:tt) => {
		acc_p!($mr, $layout, "0({x1})", 23)
	};
	($mr:tt, 4, $layout:tt) => {
        acc_p!($mr, $layout, "0({x1}, {x0})", 24)
	};
	($mr:tt, 5, $layout:tt) => {
		acc_p!($mr, $layout, "0({x1}, {x0}, 2)", 25)
	};
	($mr:tt, 6, $layout:tt) => {
		acc_p!($mr, $layout, "0({x2})", 26)
	};
	($mr:tt, 7, $layout:tt) => {
    	acc_p!($mr, $layout, "0({x2}, {x0})", 27)
	};
	($mr:tt, 8, $layout:tt) => {
		acc_p!($mr, $layout, "0({x2}, {x0}, 2)", 28)
	};
	($mr:tt, 9, $layout:tt) => {
		acc_p!($mr, $layout, "0({x3})", 29)
	};
	($mr:tt, 10, $layout:tt) => {
        acc_p!($mr, $layout, "0({x3}, {x0})", 30)
	};
	($mr:tt, 11, $layout:tt) => {
		acc_p!($mr, $layout, "0({x3}, {x0}, 2)", 31)
	};
}

macro_rules! store_16x12_seq {
	($mr:tt, 0, $layout:tt) => {
		storep!($mr, $layout, "0({cx})", 20)
	};
	($mr:tt, 1, $layout:tt) => {
		storep!($mr, $layout, "0({cx}, {x0})", 21)
	};
	($mr:tt, 2, $layout:tt) => {
		storep!($mr, $layout, "0({cx}, {x0}, 2)", 22)
	}; 
	($mr:tt, 3, $layout:tt) => {
		storep!($mr, $layout, "0({x1})", 23)
	};
	($mr:tt, 4, $layout:tt) => {
        storep!($mr, $layout, "0({x1}, {x0})", 24)
	};
	($mr:tt, 5, $layout:tt) => {
		storep!($mr, $layout, "0({x1}, {x0}, 2)", 25)
	};
	($mr:tt, 6, $layout:tt) => {
		storep!($mr, $layout, "0({x2})", 26)
	};
	($mr:tt, 7, $layout:tt) => {
        storep!($mr, $layout, "0({x2}, {x0})", 27)
	};
	($mr:tt, 8, $layout:tt) => {
		storep!($mr, $layout, "0({x2}, {x0}, 2)", 28)
	};
	($mr:tt, 9, $layout:tt) => {
		storep!($mr, $layout, "0({x3})", 29)
	};
	($mr:tt, 10, $layout:tt) => {
        storep!($mr, $layout, "0({x3}, {x0})", 30)
	};
	($mr:tt, 11, $layout:tt) => {
		storep!($mr, $layout, "0({x3}, {x0}, 2)", 31)
	};
}

macro_rules! acc_48x8 {
	($mr:tt, $nr:tt, $layout:tt) => {
		seq!(n in 0..$nr {
			concat!(
				#(
					acc_48x8_seq!($mr, n, $layout), 
				)*
			)
		})
	};
}

macro_rules! store_48x8 {
	($mr:tt, $nr:tt, $layout:tt) => {
		seq!(n in 0..$nr {
			concat!(
				#(
					store_48x8_seq!($mr, n, $layout), 
				)*
			)
		})
	};
}

macro_rules! acc_32x12 {
	($mr:tt, $nr:tt, $layout:tt) => {
		seq!(n in 0..$nr {
			concat!(
				#(
					acc_32x12_seq!($mr, n, $layout), 
				)*
			)
		})
	};
}

macro_rules! store_32x12 {
	($mr:tt, $nr:tt, $layout:tt) => {
		seq!(n in 0..$nr {
			concat!(
				#(
					store_32x12_seq!($mr, n, $layout), 
				)*
			)
		})
	};
}


macro_rules! acc_16x12 {
	($mr:tt, $nr:tt, $layout:tt) => {
		seq!(n in 0..$nr {
			concat!(
				#(
					acc_16x12_seq!($mr, n, $layout), 
				)*
			)
		})
	};
}


macro_rules! store_16x12 {
	($mr:tt, $nr:tt, $layout:tt) => {
		seq!(n in 0..$nr {
			concat!(
				#(
					store_16x12_seq!($mr, n, $layout), 
				)*
			)
		})
	};
}

macro_rules! load_b {
	(S, 0, $X:tt, $r:expr) => {
    	concat!(
        	vbroadcast!(), " ({bx}),%zmm", $r, "\n",
    	)
	};
	(S, 1, $X:tt, $r:expr) => {
    	concat!(
        	vbroadcast!(), " ({bx},{x2},1),%zmm", $r, "\n",
    	)
	};
	(S, 2, $X:tt, $r:expr) => {
    	concat!(
        	vbroadcast!(), " ({bx},{x2},2),%zmm", $r, "\n",
    	)
	};
	(S, 3, $X:tt, $r:expr) => {
    	concat!(
			"prefetcht0 64({x3}) \n",
        	vbroadcast!(), " ({x3}),%zmm", $r, "\n",
    	)
	};
	(S, 4, $X:tt, $r:expr) => {
    	concat!(
        	vbroadcast!(), " ({x3},{x2},1),%zmm", $r, "\n",
    	)
	};
	(S, 5, $X:tt, $r:expr) => {
    	concat!(
        	vbroadcast!(), " ({x3},{x2},2),%zmm", $r, "\n",
    	)
	};
	(S, 6, $X:tt, $r:expr) => {
    	concat!(
			"prefetcht0 64({x4}) \n",
        	vbroadcast!(), " ({x4}),%zmm", $r, "\n",
    	)
	};
	(S, 7, $X:tt, $r:expr) => {
    	concat!(
        	vbroadcast!(), " ({x4},{x2},1),%zmm", $r, "\n",
    	)
	};
	(S, 8, $X:tt, $r:expr) => {
    	concat!(
        	vbroadcast!(), " ({x4},{x2},2),%zmm", $r, "\n",
    	)
	};
	(S, 9, $X:tt, $r:expr) => {
    	concat!(
        	vbroadcast!(), " ({x5}),%zmm", $r, "\n",
    	)
	};
	(S, 10, $X:tt, $r:expr) => {
    	concat!(
        	vbroadcast!(), " ({x5},{x2},1),%zmm", $r, "\n",
    	)
	};
	(S, 11, $X:tt, $r:expr) => {
    	concat!(
        	vbroadcast!(), " ({x5},{x2},2),%zmm", $r, "\n",
    	)
	};
	(B, $N:tt, $X:tt, $r:expr) => {
    	concat!(
        	vbroadcast!(), " ", $N, "*4({bx}), %zmm", $r, "\n",
    	)
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

macro_rules! fmadd_3v {
	(0) => {
		concat!(
			vfmadd!(0, 3, 8),
			vfmadd!(1, 3, 9),
			vfmadd!(2, 3, 10),
		)
	};
	(1) => {
		concat!(
			vfmadd!(0, 4, 11),
			vfmadd!(1, 4, 12),
			vfmadd!(2, 4, 13),
		)
	};
	(2) => {
		concat!(
			vfmadd!(0, 5, 14),
			vfmadd!(1, 5, 15),
			vfmadd!(2, 5, 16),
		)
	};
	(3) => {
		concat!(
			vfmadd!(0, 6, 17),
			vfmadd!(1, 6, 18),
			vfmadd!(2, 6, 19),
		)
	};
	(4) => {
		concat!(
			vfmadd!(0, 7, 20),
			vfmadd!(1, 7, 21),
			vfmadd!(2, 7, 22),
		)
	};
	(5) => {
		concat!(
			vfmadd!(0, 3, 23),
			vfmadd!(1, 3, 24),
			vfmadd!(2, 3, 25),
		)
	};
	(6) => {
		concat!(
			vfmadd!(0, 4, 26),
			vfmadd!(1, 4, 27),
			vfmadd!(2, 4, 28),
		)
	};
	(7) => {
		concat!(
			vfmadd!(0, 5, 29),
			vfmadd!(1, 5, 30),
			vfmadd!(2, 5, 31),
		)
	};
}

macro_rules! fmadd_2v {
	(0) => {
		concat!(
			vfmadd!(0, 2, 8),
			vfmadd!(1, 2, 9),
		)
	};
	(1) => {
		concat!(
			vfmadd!(0, 3, 10),
			vfmadd!(1, 3, 11),
		)
	};
	(2) => {
		concat!(
			vfmadd!(0, 4, 12),
			vfmadd!(1, 4, 13),
		)
	};
	(3) => {
		concat!(
			vfmadd!(0, 5, 14),
			vfmadd!(1, 5, 15),
		)
	};
	(4) => {
		concat!(
			vfmadd!(0, 6, 16),
			vfmadd!(1, 6, 17),
		)
	};
	(5) => {
		concat!(
			vfmadd!(0, 7, 18),
			vfmadd!(1, 7, 19),
		)
	};
	(6) => {
		concat!(
			vfmadd!(0, 2, 20),
			vfmadd!(1, 2, 21),
		)
	};
	(7) => {
		concat!(
			vfmadd!(0, 3, 22),
			vfmadd!(1, 3, 23),
		)
	};
	(8) => {
		concat!(
			vfmadd!(0, 4, 24),
			vfmadd!(1, 4, 25),
		)
	};
	(9) => {
		concat!(
			vfmadd!(0, 5, 26),
			vfmadd!(1, 5, 27),
		)
	};
	(10) => {
		concat!(
			vfmadd!(0, 6, 28),
			vfmadd!(1, 6, 29),
		)
	};
	(11) => {
		concat!(
			vfmadd!(0, 7, 30),
			vfmadd!(1, 7, 31),
		)
	};
}

macro_rules! fmadd_1v {
	(0) => {
		concat!(
			vfmadd!(0, 1, 20),
		)
	};
	(1) => {
		concat!(
			vfmadd!(0, 2, 21),
		)
	};
	(2) => {
		concat!(
			vfmadd!(0, 3, 22),
		)
	};
	(3) => {
		concat!(
			vfmadd!(0, 4, 23),
		)
	};
	(4) => {
		concat!(
			vfmadd!(0, 5, 24),
		)
	};
	(5) => {
		concat!(
			vfmadd!(0, 6, 25),
		)
	};
	(6) => {
		concat!(
			vfmadd!(0, 7, 26),
		)
	};
	(7) => {
		concat!(
			vfmadd!(0, 8, 27),
		)
	};
	(8) => {
		concat!(
			vfmadd!(0, 9, 28),
		)
	};
	(9) => {
		concat!(
			vfmadd!(0, 10, 29),
		)
	};
	(10) => {
		concat!(
			vfmadd!(0, 11, 30),
		)
	};
	(11) => {
		concat!(
			vfmadd!(0, 12, 31),
		)
	};
}

macro_rules! b_num_48x8 {
	(0) => {3};
	(1) => {4};
	(2) => {5};
	(3) => {6};
	(4) => {7};
	(5) => {3};
	(6) => {4};
	(7) => {5};
}

macro_rules! b_num_32x12 {
	(0) => {2};
	(1) => {3};
	(2) => {4};
	(3) => {5};
	(4) => {6};
	(5) => {7};
	(6) => {2};
	(7) => {3};
	(8) => {4};
	(9) => {5};
	(10) => {6};
	(11) => {7};
}

macro_rules! b_num_16x12 {
	(0) => {1};
	(1) => {2};
	(2) => {3};
	(3) => {4};
	(4) => {5};
	(5) => {6};
	(6) => {7};
	(7) => {8};
	(8) => {9};
	(9) => {10};
	(10) => {11};
	(11) => {12};
}

// ***************************** 48x8 ******************************* //
macro_rules! step_48x8 {
	(48, 8, B, B) => {
		concat!(
			load_a!(48, B),
			"addq $192, {ax} \n",
			load_b!(B, 0, 8, 3),
			fmadd_3v!(0),
			load_b!(B, 1, 8, 4),
			fmadd_3v!(1),
			"prefetcht0 384({ax}) \n",
			"prefetcht0 64({bx}) \n",
			load_b!(B, 2, 8, 5),
			fmadd_3v!(2),
			load_b!(B, 3, 8, 6),
			fmadd_3v!(3),
			"prefetcht0 448({ax}) \n",
			load_b!(B, 4, 8, 7),
			fmadd_3v!(4),
			load_b!(B, 5, 8, 3),
			fmadd_3v!(5),
			"prefetcht0 512({ax}) \n",
			load_b!(B, 6, 8, 4),
			fmadd_3v!(6),
			load_b!(B, 7, 8, 5),
			fmadd_3v!(7),
			"addq $32, {bx} \n",
		)
		
	};
	($mr:tt, $nr:tt, $a_layout:tt, $b_layout:tt) => {
		seq!(n in 0..$nr {
			concat!(
				load_a!($mr, $a_layout),
				inc_a!($a_layout,$mr),
				"prefetcht0 64({bx}) \n",
				#(
					load_b!($b_layout, n, $nr, b_num_48x8!(n)),
					fmadd_3v!(n),
				)*
				inc_b!($b_layout,$nr), 
			)
		})
	};
}

// ***************************** 32x12 ******************************* //
macro_rules! step_32x12 {
	($mr:tt, $nr:tt, $a_layout:tt, $b_layout:tt) => {
		seq!(n in 0..$nr {
			concat!(
				load_a!($mr, $a_layout),
				inc_a!($a_layout,$mr),
				"prefetcht0 64({bx}) \n",
				#(
					load_b!($b_layout, n, $nr, b_num_32x12!(n)),
					fmadd_2v!(n),
				)*
				inc_b!($b_layout,$nr), 
			)
		})
	};
}

// ***************************** 16x12 ******************************* //
macro_rules! step_16x12 {
	($mr:tt, $nr:tt, $a_layout:tt, $b_layout:tt) => {
		seq!(n in 0..$nr {
			concat!(
				load_a!($mr, $a_layout),
				inc_a!($a_layout,$mr),
				"prefetcht0 64({bx}) \n",
				#(
					load_b!($b_layout, n, $nr, b_num_16x12!(n)),
					fmadd_1v!(n),
				)*
				inc_b!($b_layout,$nr), 
			)
		})
	};
}

macro_rules! prefetch_c {
    (48, $nr:tt, $c:tt, $ldc:tt) => {
        seq!(j in 0..$nr {
            _mm_prefetch($c.add(0+j*$ldc) as *const i8, 3);
            _mm_prefetch($c.add(16+j*$ldc) as *const i8, 3);
			_mm_prefetch($c.add(32+j*$ldc) as *const i8, 3);
        });
    };
    (32, $nr:tt, $c:tt, $ldc:tt) => {
        seq!(j in 0..$nr {
            _mm_prefetch($c.add(0+j*$ldc) as *const i8, 3);
            _mm_prefetch($c.add(16+j*$ldc) as *const i8, 3);
			_mm_prefetch($c.add(32+j*$ldc) as *const i8, 3);
        });
    };
    (16, $nr:tt, $c:tt, $ldc:tt) => {
        seq!(j in 0..$nr {
            _mm_prefetch($c.add(16+j*$ldc) as *const i8, 3);
        });
    };
}

macro_rules! mask_ptr {
	(M, $m:tt, $nm:ident) => {
		let $nm = if $m % VS == 0 && $m > 0 { 0xFFFF } else { (1_u16 << ($m % VS)) - 1 };
	};
	(C, $m:tt, $nm:ident) => {
		let $nm = 0xFFFF_u16;
	};
}

macro_rules! load_mask_ptr_asm {
	(M) => {
		"kmovw ({maskx}), %k1"
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
			d_arr: [usize; 4],
			m: usize, _n: usize,
			f: F,
    	) {
			mask_ptr!($is_partial, m, x);
			let mask_ptr = (&x) as *const u16;
			let k_iter = k / $unroll;
        	let k_left = k % $unroll;
            let mut dim_arr = [d_arr[0]*4, d_arr[1]*4, d_arr[3]*4, k_iter, k_left];
			let mut cf = c;
			let mut c_buf = [0f32;$mr*$nr];
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
				$asm_step_macro!($mr, $nr, $a_layout, $b_layout),
				$asm_step_macro!($mr, $nr, $a_layout, $b_layout),
				$asm_step_macro!($mr, $nr, $a_layout, $b_layout),
				$asm_step_macro!($mr, $nr, $a_layout, $b_layout),
   	 
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
            	$asm_step_macro!($mr, $nr, $a_layout, $b_layout),

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
				x4 = out(reg) _,
            	x5 = out(reg) _,
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
			let mask_ptr = (&x) as *const u16;
			let k_iter = k / $unroll;
        	let k_left = k % $unroll;
            let mut dim_arr = [d_arr[0]*4, d_arr[1]*4, d_arr[3]*4, k_iter, k_left];
			let mut cf = c;
			let mut c_buf = [0f32;$mr*$nr];
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
							$asm_step_macro!($mr, ni, $a_layout, $b_layout),
							$asm_step_macro!($mr, ni, $a_layout, $b_layout),
							$asm_step_macro!($mr, ni, $a_layout, $b_layout),
							$asm_step_macro!($mr, ni, $a_layout, $b_layout),
				
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
							$asm_step_macro!($mr, ni, $a_layout, $b_layout),

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
							x4 = out(reg) _,
							x5 = out(reg) _,
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
				for i in 0..$mr/8 {
					f.call(c.add(i*8+j*c_cs), 8);
				}
			}
    	}
	};
}

// def_ukernel!(VER3, step_48x8, acc_48x8, store_48x8, 48, 8, B, B, C, 4, ukernel_48x8_bb);
// def_ukernel!(VER2, step_32x12, acc_32x12, store_32x12, 32, 8, B, B, C, 4, ukernel_32x8_bb);
// def_ukernel!(VER1, step_16x12, acc_16x12, store_16x12, 16, 8, B, B, C, 4, ukernel_16x8_bb);

def_ukernel!(VER3, step_48x8, acc_48x8, store_48x8, 48, 8, B, B, M, 4, ukernel_48x8_bb_partial);
def_ukernel!(VER2, step_32x12, acc_32x12, store_32x12, 32, 8, B, B, M, 4, ukernel_32x8_bb_partial);
def_ukernel!(VER1, step_16x12, acc_16x12, store_16x12, 16, 8, B, B, M, 4, ukernel_16x8_bb_partial);

def_ukernel!(VER3, step_48x8, acc_48x8, store_48x8, 48, 8, B, S, C, 4, ukernel_48x8_bs);

def_ukernel!(VER3, step_48x8, acc_48x8, store_48x8, 48, 8, B, S, M, 4, ukernel_48x8_bs_partial);
def_ukernel!(VER2, step_32x12, acc_32x12, store_32x12, 32, 8, B, S, M, 4, ukernel_32x8_bs_partial);
def_ukernel!(VER1, step_16x12, acc_16x12, store_16x12, 16, 8, B, S, M, 4, ukernel_16x8_bs_partial);


def_ukernelxn!(VER3, step_48x8, acc_48x8, store_48x8, 48, 8, B, B, C, 4, ukernel_48xn_bb);
// def_ukernelxn!(VER2, step_32x12, acc_32x12, store_32x12, 32, 7, B, B, C, 4, ukernel_32xn_bb);
// def_ukernelxn!(VER1, step_16x12, acc_16x12, store_16x12, 16, 7, B, B, C, 4, ukernel_16xn_bb);

def_ukernelxn!(VER3, step_48x8, acc_48x8, store_48x8, 48, 8, B, B, M, 4, ukernel_48xn_bb_partial);
def_ukernelxn!(VER2, step_32x12, acc_32x12, store_32x12, 32, 8, B, B, M, 4, ukernel_32xn_bb_partial);
def_ukernelxn!(VER1, step_16x12, acc_16x12, store_16x12, 16, 8, B, B, M, 4, ukernel_16xn_bb_partial);

def_ukernelxn!(VER3, step_48x8, acc_48x8, store_48x8, 48, 8, B, S, C, 4, ukernel_48xn_bs);

def_ukernelxn!(VER3, step_48x8, acc_48x8, store_48x8, 48, 8, B, S, M, 4, ukernel_48xn_bs_partial);
def_ukernelxn!(VER2, step_32x12, acc_32x12, store_32x12, 32, 8, B, S, M, 4, ukernel_32xn_bs_partial);
def_ukernelxn!(VER1, step_16x12, acc_16x12, store_16x12, 16, 8, B, S, M, 4, ukernel_16xn_bs_partial);


// group_def_ukernel!(48, 1, 7, B, B, C, bb, step_48x8, acc_48x8, store_48x8, VER3);
// group_def_ukernel!(48, 1, 1, B, S, C, bs, step_48x8, acc_48x8, store_48x8, VER3);
// group_def_ukernel!(48, 1, 8, B, B, M, bb_partial, step_48x8, acc_48x8, store_48x8, VER3);
// group_def_ukernel!(48, 1, 8, B, S, M, bs_partial, step_48x8, acc_48x8, store_48x8, VER3);

// // group_def_ukernel!(32, 1, 8, B, B, bb, def_ukernel, step_32x12, acc_32x12, store_32x12, VER2);
// // group_def_ukernel!(32, 1, 8, B, S, bs, def_ukernel, step_32x12, acc_32x12, store_32x12, VER2);
// group_def_ukernel!(32, 1, 8, B, B, M, bb_partial, step_32x12, acc_32x12, store_32x12, VER2);
// group_def_ukernel!(32, 1, 8, B, S, M, bs_partial, step_32x12, acc_32x12, store_32x12, VER2);


// // group_def_ukernel!(16, 1, 8, B, B, bb, def_ukernel, step_16x12, acc_16x12, store_16x12, VER1);
// // group_def_ukernel!(16, 1, 8, B, S, bs, def_ukernel, step_16x12, acc_16x12, store_16x12, VER1);
// group_def_ukernel!(16, 1, 8, B, B, M, bb_partial, step_16x12, acc_16x12, store_16x12, VER1);
// group_def_ukernel!(16, 1, 8, B, S, M, bs_partial, step_16x12, acc_16x12, store_16x12, VER1);


// based on l1 prefetching scheme is from openblas impl for skylax
// see: https://github.com/OpenMathLib/OpenBLAS/pull/2300
// this is adapted to our ukernel of 48x8
// seems to stem from high bandwith of l1 cache (compared to other uarch e.g. haswell
// where the same l1 prefetching does not benefit as much)
pub(crate) unsafe fn ukernel_48x8_bb<F: MyFn, const BUF: bool>(
    a: *const TA, b: *const TB, c: *mut TC,
    alpha: *const TA, beta: *const TB,
    k: usize,
    d_arr: [usize; 4],
    a_pft1_offset: usize,
    f: F,
) {
	let k_left0 = k % 8;
	let k_left = if k_left0 == 0 {8} else {k_left0};
	let k_iter = (k - k_left) / 4;
    let mut dim_arr = [d_arr[3]*4, k_iter, k_left, a_pft1_offset];
    let mut cf = c;
    let mut c_buf = [0f32; 48 * 8];
	let c_cs = d_arr[3];
    if BUF {
        let c_rs = d_arr[2];
        if c_rs != 1 {
            for j in 0..8 {
                for i in 0..48 {
                    c_buf[j * 48 + i] = *c.add(i * c_rs + j * c_cs);
                }
            }
            cf = c_buf.as_mut_ptr();
            dim_arr[2] = 48 * 4;
        }
    }
    asm!(
		asm_vzeroall!(VER3,8),
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
		step_48x8!(48, 8, B, B),

		"movq $64*4, {x4}",
		// divisiblity by 4
		"testq $3, {x0}",
		"cmovz {x1},{x4}",

		step_48x8!(48, 8, B, B),

		"prefetcht1 ({x2})",

		"subq $64*3, {x2}",
		"addq {x4}, {x2}",

		step_48x8!(48, 8, B, B),

		"prefetcht1 ({x5})",
		"addq $32, {x5}",

		"testq $63, {x0}",
		"cmovz {cx},{x2}",

		step_48x8!(48, 8, B, B),

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
		"prefetcht0 128({x2})",
		step_48x8!(48, 8, B, B),

		"add {x1}, {x2}",
		"dec {x0}",
		"jne 4b",
		"5:",
		"mov ({dim_arrx}),{x0}",
		"lea ({x0}, {x0}, 2), {x3}",
		"lea ({cx}, {x3},), {x1}",
		"lea ({x1}, {x3},), {x2}",
		// scale by alpha
		asm_alpha_scale!(VER3, 8),
		load_beta!(),

		// 6 -> BETAZERO
		"je 6f",
		acc_48x8!(48,8,C),

		// 6 -> BETAZERO
		"6:",
		store_48x8!(48,8,C),

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
                for i in 0..48 {
                    *c.add(i * c_rs + j * c_cs) = c_buf[j * 48 + i];
                }
            }
        }
    }
    for j in 0..8 {
        for i in 0..48 / 8 {
            f.call(c.add(i * 8 + j * c_cs), 8);
        }
    }
}