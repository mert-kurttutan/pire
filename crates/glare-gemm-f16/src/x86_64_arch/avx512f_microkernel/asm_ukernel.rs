use seq_macro::seq;
use std::arch::asm;

use half::f16;


use paste::paste;
macro_rules! beta_fmaddps {
    (C, $m0:expr, $r1:expr) => {
        concat!(
			"vcvtph2ps ", $m0, ", %zmm2", "\n",
            "vfmadd231ps %zmm2,%zmm0,%zmm", $r1, "\n",
        ) 
    };
    (M, $m0:expr, $r1:expr) => {
        concat!(
            // "vmovups ", $m0, ", %zmm1", ", %zmm2",  "\n",
			// "vmovups ", $m0, ", %zmm1 {{%k1}}", "\n",
			"vcvtph2ps ", $m0, ", %zmm2{{%k1}}", "\n",
			"vfmadd231ps %zmm2,%zmm0,%zmm", $r1, "\n",
        )
    };
	(C, $m0:expr) => {
		concat!(
			"vfmadd231ps ", $m0, ",%zmm0,%zmm", 0, "\n",
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
    (48, $layout:tt, $m0:expr, $r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            beta_fmaddps!(C, $m0, $r1),
            beta_fmaddps!(C, mem!($m0, "0x20"), $r2),
            beta_fmaddps!($layout, mem!($m0, "0x40"), $r3),
        )
    };
    (32, $layout:tt, $m0:expr, $r1:expr, $r2:expr) => {
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
					"vpxorq %zmm",r,",%zmm",r,",%zmm",r,"\n",
				)*
			)
		})
	}
}

macro_rules! loadps_unit {
    ($layout:tt, $m0:expr, $r1:expr) => {
        concat!(
            vmovps!($layout), $m0, ",%zmm", $r1, "\n",
        )
    };
    (4, $layout:tt, $m0:expr, $r1:expr) => {
        concat!(
            vmovps!($layout), $m0, ",", "%xmm", $r1, "\n",
        )
    };
 }
 macro_rules! loadps {
    (48, $layout:tt, $m0:expr) => {
        concat!(
            loadps_unit!($layout, $m0, 0),
            loadps_unit!($layout, mem!($m0, "0x40"), 1),
            loadps_unit!($layout, mem!($m0, "0x80"), 2),
        )
    };
    (32, $layout:tt, $m0:expr) => {
        concat!(
            loadps_unit!($layout, $m0, 0),
            loadps_unit!($layout, mem!($m0, "0x40"), 1),
        )
    };
    (16, $layout:tt, $m0:expr) => {
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
			"vcvtps2ph $0x00, %zmm", $r1, ", ", $m0, "\n",
            // "vmovups %zmm", $r1, ", ", $m0, "\n",
        )
    };
    (B, $r1:expr, $m0:expr) => {
        concat!(
            "vmovaps %zmm", $r1, ", ", $m0, "\n",
        )
    };
    (M, $r1:expr, $m0:expr) => {
        concat!(
			// "vmovups %zmm", $r1, ", ", $m0, " {{%k1}}\n",
			"vcvtps2ph $0x00, %zmm", $r1, ", ", $m0, " {{%k1}}\n",
		)
    };
}
macro_rules! storeps {
	(48, $layout:tt, $m0:expr, $r1:expr, $r2:expr, $r3:expr) => {
		concat!(
			storeps_unit!(C, $r1, $m0),
			storeps_unit!(C, $r2, mem!($m0, "0x20")),
			storeps_unit!($layout, $r3, mem!($m0, "0x40")),
		)
	};
	(32, $layout:tt, $m0:expr, $r1:expr, $r2:expr) => {
		concat!(
			storeps_unit!(C, $r1, $m0),
			storeps_unit!($layout, $r2, mem!($m0, "0x20")),
		)
	};
	(16, $layout:tt, $m0:expr, $r1:expr) => {
		concat!(
			storeps_unit!($layout, $r1, $m0),
		)
	};
 }

macro_rules! vfmadd231ps {
    ($r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            "vfmadd231ps %zmm", $r1, ", %zmm", $r2,", %zmm", $r3, "\n",
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
        	"mov 24({int_arrx}),{x0}", "\n",
        	"test {x0},{x0}", "\n",
    	)
	};
	($ker:tt,B,S) => {
    	concat!(
        	// mov cs_b to reg
			"mov ({int_arrx}), {x1}", "\n",
        	"mov 8({int_arrx}), {x2}", "\n",
        	"lea ({x2}, {x2}, 2), {x5}", "\n",
        	"lea ({bx}, {x5}, 1), {x3}", "\n",
			"lea ({x3}, {x5}, 1), {x4}", "\n",
			"lea ({x4}, {x5}, 1), {x5}", "\n",

        	"mov 24({int_arrx}),{x0}", "\n",
        	"test {x0},{x0}", "\n",
    	)
	};
}


macro_rules! asm_c_load {
	(12) => {
    	concat!(
        	"mov 16({int_arrx}),{x0}", "\n",
        	"lea ({x0}, {x0}, 2), {x3}", "\n",
        	"lea ({cx}, {x3},), {x1}", "\n",
			"lea ({x1}, {x3},), {x2}", "\n",
			"lea ({x2}, {x3},), {x3}", "\n",
    	)
	};
	(11) => {
    	concat!(
        	"mov 16({int_arrx}),{x0}", "\n",
        	"lea ({x0}, {x0}, 2), {x3}", "\n",
        	"lea ({cx}, {x3},), {x1}", "\n",
			"lea ({x1}, {x3},), {x2}", "\n",
			"lea ({x2}, {x3},), {x3}", "\n",
    	)
	};
	(10) => {
    	concat!(
        	"mov 16({int_arrx}),{x0}", "\n",
        	"lea ({x0}, {x0}, 2), {x3}", "\n",
        	"lea ({cx}, {x3},), {x1}", "\n",
			"lea ({x1}, {x3},), {x2}", "\n",
			"lea ({x2}, {x3},), {x3}", "\n",
    	)
	};
	(9) => {
    	concat!(
        	"mov 16({int_arrx}),{x0}", "\n",
        	"lea ({x0}, {x0}, 2), {x3}", "\n",
        	"lea ({cx}, {x3},), {x1}", "\n",
			"lea ({x1}, {x3},), {x2}", "\n",
    	)
	};
	(8) => {
    	concat!(
        	"mov 16({int_arrx}),{x0}", "\n",
        	"lea ({x0}, {x0}, 2), {x3}", "\n",
        	"lea ({cx}, {x3},), {x1}", "\n",
			"lea ({x1}, {x3},), {x2}", "\n",
    	)
	};
	(7) => {
    	concat!(
        	"mov 16({int_arrx}),{x0}", "\n",
        	"lea ({x0}, {x0}, 2), {x3}", "\n",
        	"lea ({cx}, {x3},), {x1}", "\n",
			"lea ({x1}, {x3},), {x2}", "\n",
    	)
	};
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

	(VER48,8) => {vzeroall!(8,31)};
	(VER48,7) => {vzeroall!(8,28)};
	(VER48,6) => {vzeroall!(8,25)};
	(VER48,5) => {vzeroall!(8,22)};
	(VER48,4) => {vzeroall!(8,19)};
	(VER48,3) => {vzeroall!(8,16)};
	(VER48,2) => {vzeroall!(8,13)};
	(VER48,1) => {vzeroall!(8,10)};

	(VER32,12) => {vzeroall!(8,31)};
	(VER32,11) => {vzeroall!(8,29)};
	(VER32,10) => {vzeroall!(8,27)};
	(VER32,9) => {vzeroall!(8,25)};
	(VER32,8) => {vzeroall!(8,23)};
	(VER32,7) => {vzeroall!(8,21)};
	(VER32,6) => {vzeroall!(8,19)};
	(VER32,5) => {vzeroall!(8,17)};
	(VER32,4) => {vzeroall!(8,15)};
	(VER32,3) => {vzeroall!(8,13)};
	(VER32,2) => {vzeroall!(8,11)};
	(VER32,1) => {vzeroall!(8,9)};

	(VER16,12) => {vzeroall!(20,31)};
	(VER16,11) => {vzeroall!(20,30)};
	(VER16,10) => {vzeroall!(20,29)};
	(VER16,9) => {vzeroall!(20,28)};
	(VER16,8) => {vzeroall!(20,27)};
	(VER16,7) => {vzeroall!(20,26)};
	(VER16,6) => {vzeroall!(20,25)};
	(VER16,5) => {vzeroall!(20,24)};
	(VER16,4) => {vzeroall!(20,23)};
	(VER16,3) => {vzeroall!(20,22)};
	(VER16,2) => {vzeroall!(20,21)};
	(VER16,1) => {vzeroall!(20,20)};
}

macro_rules! asm_alpha_scale {
	(VER48, 8) => {asm_alpha_scale_0!(8,31)};
	(VER48, 7) => {asm_alpha_scale_0!(8,28)};
	(VER48, 6) => {asm_alpha_scale_0!(8,25)};
	(VER48, 5) => {asm_alpha_scale_0!(8,22)};
	(VER48, 4) => {asm_alpha_scale_0!(8,19)};
	(VER48, 3) => {asm_alpha_scale_0!(8,16)};
	(VER48, 2) => {asm_alpha_scale_0!(8,13)};
	(VER48, 1) => {asm_alpha_scale_0!(8,10)};

	(VER32, 12) => {asm_alpha_scale_0!(8,31)};
	(VER32, 11) => {asm_alpha_scale_0!(8,29)};
	(VER32, 10) => {asm_alpha_scale_0!(8,27)};
	(VER32, 9) => {asm_alpha_scale_0!(8,25)};
	(VER32, 8) => {asm_alpha_scale_0!(8,23)};
	(VER32, 7) => {asm_alpha_scale_0!(8,21)};
	(VER32, 6) => {asm_alpha_scale_0!(8,19)};
	(VER32, 5) => {asm_alpha_scale_0!(8,17)};
	(VER32, 4) => {asm_alpha_scale_0!(8,15)};
	(VER32, 3) => {asm_alpha_scale_0!(8,13)};
	(VER32, 2) => {asm_alpha_scale_0!(8,11)};
	(VER32, 1) => {asm_alpha_scale_0!(8,9)};

	(VER16, 12) => {asm_alpha_scale_0!(20,31)};
	(VER16, 11) => {asm_alpha_scale_0!(20,30)};
	(VER16, 10) => {asm_alpha_scale_0!(20,29)};
	(VER16, 9) => {asm_alpha_scale_0!(20,28)};
	(VER16, 8) => {asm_alpha_scale_0!(20,27)};
	(VER16, 7) => {asm_alpha_scale_0!(20,26)};
	(VER16, 6) => {asm_alpha_scale_0!(20,25)};
	(VER16, 5) => {asm_alpha_scale_0!(20,24)};
	(VER16, 4) => {asm_alpha_scale_0!(20,23)};
	(VER16, 3) => {asm_alpha_scale_0!(20,22)};
	(VER16, 2) => {asm_alpha_scale_0!(20,21)};
	(VER16, 1) => {asm_alpha_scale_0!(20,20)};
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
				"vbroadcastss ({alphax}),%zmm1", "\n",
				#(
					"vmulps %zmm1, %zmm", r, ",%zmm", r, "\n",
				)*
			)
		})
	}
}


macro_rules! asm_48x8_acc_seq {
	($mr:tt, 0, $layout:tt) => {
		acc_ps!($mr, $layout, "0({cx})", 8, 9, 10)
	};
	($mr:tt, 1, $layout:tt) => {
		acc_ps!($mr, $layout, "0({cx}, {x0})", 11, 12, 13)
	};
	($mr:tt, 2, $layout:tt) => {
		acc_ps!($mr, $layout, "0({cx}, {x0}, 2)",  14, 15, 16)
	}; 
	($mr:tt, 3, $layout:tt) => {
		acc_ps!($mr, $layout, "0({x1})",  17, 18, 19)
	};
	($mr:tt, 4, $layout:tt) => {
		acc_ps!($mr, $layout, "0({x1}, {x0})", 20, 21, 22)
	};
	($mr:tt, 5, $layout:tt) => {
		acc_ps!($mr, $layout, "0({x1}, {x0}, 2)", 23, 24, 25)
	};
	($mr:tt, 6, $layout:tt) => {
		acc_ps!($mr, $layout, "0({x2})", 26, 27, 28)
	};
	($mr:tt, 7, $layout:tt) => {
		acc_ps!($mr, $layout, "0({x2}, {x0})", 29, 30, 31)
	};
}

macro_rules! asm_48x8_store_seq {
	($mr:tt, 0, $layout:tt) => {
		storeps!($mr, $layout, "0({cx})", 8, 9, 10)
	};
	($mr:tt, 1, $layout:tt) => {
		storeps!($mr, $layout, "0({cx}, {x0})", 11, 12, 13)
	};
	($mr:tt, 2, $layout:tt) => {
		storeps!($mr, $layout, "0({cx}, {x0}, 2)",  14, 15, 16)
	}; 
	($mr:tt, 3, $layout:tt) => {
		storeps!($mr, $layout, "0({x1})",  17, 18, 19)
	};
	($mr:tt, 4, $layout:tt) => {
		storeps!($mr, $layout, "0({x1}, {x0})", 20, 21, 22)
	};
	($mr:tt, 5, $layout:tt) => {
		storeps!($mr, $layout, "0({x1}, {x0}, 2)", 23, 24, 25)
	};
	($mr:tt, 6, $layout:tt) => {
		storeps!($mr, $layout, "0({x2})", 26, 27, 28)
	};
	($mr:tt, 7, $layout:tt) => {
		storeps!($mr, $layout, "0({x2}, {x0})", 29, 30, 31)
	};
}

macro_rules! asm_32x12_acc_seq {
	($mr:tt, 0, $layout:tt) => {
		acc_ps!($mr, $layout, "0({cx})", 8, 9)
	};
	($mr:tt, 1, $layout:tt) => {
		acc_ps!($mr, $layout, "0({cx}, {x0})", 10, 11)
	};
	($mr:tt, 2, $layout:tt) => {
		acc_ps!($mr, $layout, "0({cx}, {x0}, 2)", 12, 13)
	}; 
	($mr:tt, 3, $layout:tt) => {
		acc_ps!($mr, $layout, "0({x1})", 14, 15)
	};
	($mr:tt, 4, $layout:tt) => {
        acc_ps!($mr, $layout, "0({x1}, {x0})", 16, 17)
	};
	($mr:tt, 5, $layout:tt) => {
		acc_ps!($mr, $layout, "0({x1}, {x0}, 2)", 18, 19)
	};
	($mr:tt, 6, $layout:tt) => {
		acc_ps!($mr, $layout, "0({x2})", 20, 21)
	};
	($mr:tt, 7, $layout:tt) => {
		acc_ps!($mr, $layout, "0({x2}, {x0})", 22, 23)
	};
	($mr:tt, 8, $layout:tt) => {
		acc_ps!($mr, $layout, "0({x2}, {x0}, 2)", 24, 25)
	};
	($mr:tt, 9, $layout:tt) => {
		acc_ps!($mr, $layout, "0({x3})", 26, 27)
	};
	($mr:tt, 10, $layout:tt) => {
		acc_ps!($mr, $layout, "0({x3}, {x0})", 28, 29)
	};
	($mr:tt, 11, $layout:tt) => {
		acc_ps!($mr, $layout, "0({x3}, {x0}, 2)", 30, 31)
	};
}

macro_rules! asm_32x12_store_seq {
	($mr:tt, 0, $layout:tt) => {
		storeps!($mr, $layout, "0({cx})", 8, 9)
	};
	($mr:tt, 1, $layout:tt) => {
		storeps!($mr, $layout, "0({cx}, {x0})", 10, 11)
	};
	($mr:tt, 2, $layout:tt) => {
		storeps!($mr, $layout, "0({cx}, {x0}, 2)", 12, 13)
	}; 
	($mr:tt, 3, $layout:tt) => {
		storeps!($mr, $layout, "0({x1})", 14, 15)
	};
	($mr:tt, 4, $layout:tt) => {
    	storeps!($mr, $layout, "0({x1}, {x0})", 16, 17)
	};
	($mr:tt, 5, $layout:tt) => {
		storeps!($mr, $layout, "0({x1}, {x0}, 2)", 18, 19)
	};
	($mr:tt, 6, $layout:tt) => {
		storeps!($mr, $layout, "0({x2})", 20, 21)
	};
	($mr:tt, 7, $layout:tt) => {
		storeps!($mr, $layout, "0({x2}, {x0})", 22, 23)
	};
	($mr:tt, 8, $layout:tt) => {
		storeps!($mr, $layout, "0({x2}, {x0}, 2)", 24, 25)
	};
	($mr:tt, 9, $layout:tt) => {
		storeps!($mr, $layout, "0({x3})", 26, 27)
	};
	($mr:tt, 10, $layout:tt) => {
		storeps!($mr, $layout, "0({x3}, {x0})", 28, 29)
	};
	($mr:tt, 11, $layout:tt) => {
		storeps!($mr, $layout, "0({x3}, {x0}, 2)", 30, 31)
	};
}

macro_rules! asm_16x12_acc_seq {
	($mr:tt, 0, $layout:tt) => {
		acc_ps!($mr, $layout, "0({cx})", 20)
	};
	($mr:tt, 1, $layout:tt) => {
		acc_ps!($mr, $layout, "0({cx}, {x0})", 21)
	};
	($mr:tt, 2, $layout:tt) => {
		acc_ps!($mr, $layout, "0({cx}, {x0}, 2)", 22)
	}; 
	($mr:tt, 3, $layout:tt) => {
		acc_ps!($mr, $layout, "0({x1})", 23)
	};
	($mr:tt, 4, $layout:tt) => {
        acc_ps!($mr, $layout, "0({x1}, {x0})", 24)
	};
	($mr:tt, 5, $layout:tt) => {
		acc_ps!($mr, $layout, "0({x1}, {x0}, 2)", 25)
	};
	($mr:tt, 6, $layout:tt) => {
		acc_ps!($mr, $layout, "0({x2})", 26)
	};
	($mr:tt, 7, $layout:tt) => {
    	acc_ps!($mr, $layout, "0({x2}, {x0})", 27)
	};
	($mr:tt, 8, $layout:tt) => {
		acc_ps!($mr, $layout, "0({x2}, {x0}, 2)", 28)
	};
	($mr:tt, 9, $layout:tt) => {
		acc_ps!($mr, $layout, "0({x3})", 29)
	};
	($mr:tt, 10, $layout:tt) => {
        acc_ps!($mr, $layout, "0({x3}, {x0})", 30)
	};
	($mr:tt, 11, $layout:tt) => {
		acc_ps!($mr, $layout, "0({x3}, {x0}, 2)", 31)
	};
}

macro_rules! asm_16x12_store_seq {
	($mr:tt, 0, $layout:tt) => {
		storeps!($mr, $layout, "0({cx})", 20)
	};
	($mr:tt, 1, $layout:tt) => {
		storeps!($mr, $layout, "0({cx}, {x0})", 21)
	};
	($mr:tt, 2, $layout:tt) => {
		storeps!($mr, $layout, "0({cx}, {x0}, 2)", 22)
	}; 
	($mr:tt, 3, $layout:tt) => {
		storeps!($mr, $layout, "0({x1})", 23)
	};
	($mr:tt, 4, $layout:tt) => {
        storeps!($mr, $layout, "0({x1}, {x0})", 24)
	};
	($mr:tt, 5, $layout:tt) => {
		storeps!($mr, $layout, "0({x1}, {x0}, 2)", 25)
	};
	($mr:tt, 6, $layout:tt) => {
		storeps!($mr, $layout, "0({x2})", 26)
	};
	($mr:tt, 7, $layout:tt) => {
        storeps!($mr, $layout, "0({x2}, {x0})", 27)
	};
	($mr:tt, 8, $layout:tt) => {
		storeps!($mr, $layout, "0({x2}, {x0}, 2)", 28)
	};
	($mr:tt, 9, $layout:tt) => {
		storeps!($mr, $layout, "0({x3})", 29)
	};
	($mr:tt, 10, $layout:tt) => {
        storeps!($mr, $layout, "0({x3}, {x0})", 30)
	};
	($mr:tt, 11, $layout:tt) => {
		storeps!($mr, $layout, "0({x3}, {x0}, 2)", 31)
	};
}

macro_rules! asm_48x8_acc {
	($mr:tt, $nr:tt, $layout:tt) => {
		seq!(n in 0..$nr {
			concat!(
				#(
					asm_48x8_acc_seq!($mr, n, $layout), 
				)*
			)
		})
	};
}

macro_rules! asm_48x8_store {
	($mr:tt, $nr:tt, $layout:tt) => {
		seq!(n in 0..$nr {
			concat!(
				#(
					asm_48x8_store_seq!($mr, n, $layout), 
				)*
			)
		})
	};
}

macro_rules! asm_32x12_acc {
	($mr:tt, $nr:tt, $layout:tt) => {
		seq!(n in 0..$nr {
			concat!(
				#(
					asm_32x12_acc_seq!($mr, n, $layout), 
				)*
			)
		})
	};
}

macro_rules! asm_32x12_store {
	($mr:tt, $nr:tt, $layout:tt) => {
		seq!(n in 0..$nr {
			concat!(
				#(
					asm_32x12_store_seq!($mr, n, $layout), 
				)*
			)
		})
	};
}


macro_rules! asm_16x12_acc {
	($mr:tt, $nr:tt, $layout:tt) => {
		seq!(n in 0..$nr {
			concat!(
				#(
					asm_16x12_acc_seq!($mr, n, $layout), 
				)*
			)
		})
	};
}


macro_rules! asm_16x12_store {
	($mr:tt, $nr:tt, $layout:tt) => {
		seq!(n in 0..$nr {
			concat!(
				#(
					asm_16x12_store_seq!($mr, n, $layout), 
				)*
			)
		})
	};
}

macro_rules! load_b {
	(S, 0, $K:tt, $X:tt, $r:expr) => {
    	concat!(
        	"vbroadcastss ({bx}),%zmm", $r, "\n",
    	)
	};
	(S, 1, $K:tt, $X:tt, $r:expr) => {
    	concat!(
        	"vbroadcastss ({bx},{x2},1),%zmm", $r, "\n",
    	)
	};
	(S, 2, $K:tt, $X:tt, $r:expr) => {
    	concat!(
        	"vbroadcastss ({bx},{x2},2),%zmm", $r, "\n",
    	)
	};
	(S, 3, $K:tt, $X:tt, $r:expr) => {
    	concat!(
        	"vbroadcastss ({x3}),%zmm", $r, "\n",
    	)
	};
	(S, 4, $K:tt, $X:tt, $r:expr) => {
    	concat!(
        	"vbroadcastss ({x3},{x2},1),%zmm", $r, "\n",
    	)
	};
	(S, 5, $K:tt, $X:tt, $r:expr) => {
    	concat!(
        	"vbroadcastss ({x3},{x2},2),%zmm", $r, "\n",
    	)
	};
	(S, 6, $K:tt, $X:tt, $r:expr) => {
    	concat!(
        	"vbroadcastss ({x4}),%zmm", $r, "\n",
    	)
	};
	(S, 7, $K:tt, $X:tt, $r:expr) => {
    	concat!(
        	"vbroadcastss ({x4},{x2},1),%zmm", $r, "\n",
    	)
	};
	(S, 8, $K:tt, $X:tt, $r:expr) => {
    	concat!(
        	"vbroadcastss ({x4},{x2},2),%zmm", $r, "\n",
    	)
	};
	(S, 9, $K:tt, $X:tt, $r:expr) => {
    	concat!(
        	"vbroadcastss ({x5}),%zmm", $r, "\n",
    	)
	};
	(S, 10, $K:tt, $X:tt, $r:expr) => {
    	concat!(
        	"vbroadcastss ({x5},{x2},1),%zmm", $r, "\n",
    	)
	};
	(S, 11, $K:tt, $X:tt, $r:expr) => {
    	concat!(
        	"vbroadcastss ({x5},{x2},2),%zmm", $r, "\n",
    	)
	};
	(B, $N:tt, $K:tt, $X:tt, $r:expr) => {
    	concat!(
        	"vbroadcastss ", $K, "*", $X, "*4+", $N, "*4({bx}), %zmm", $r, "\n",
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
			vfmadd231ps!(0, 3, 8),
			vfmadd231ps!(1, 3, 9),
			vfmadd231ps!(2, 3, 10),
		)
	};
	(1) => {
		concat!(
			vfmadd231ps!(0, 4, 11),
			vfmadd231ps!(1, 4, 12),
			vfmadd231ps!(2, 4, 13),
		)
	};
	(2) => {
		concat!(
			vfmadd231ps!(0, 5, 14),
			vfmadd231ps!(1, 5, 15),
			vfmadd231ps!(2, 5, 16),
		)
	};
	(3) => {
		concat!(
			vfmadd231ps!(0, 6, 17),
			vfmadd231ps!(1, 6, 18),
			vfmadd231ps!(2, 6, 19),
		)
	};
	(4) => {
		concat!(
			vfmadd231ps!(0, 7, 20),
			vfmadd231ps!(1, 7, 21),
			vfmadd231ps!(2, 7, 22),
		)
	};
	(5) => {
		concat!(
			vfmadd231ps!(0, 3, 23),
			vfmadd231ps!(1, 3, 24),
			vfmadd231ps!(2, 3, 25),
		)
	};
	(6) => {
		concat!(
			vfmadd231ps!(0, 4, 26),
			vfmadd231ps!(1, 4, 27),
			vfmadd231ps!(2, 4, 28),
		)
	};
	(7) => {
		concat!(
			vfmadd231ps!(0, 5, 29),
			vfmadd231ps!(1, 5, 30),
			vfmadd231ps!(2, 5, 31),
		)
	};
}

macro_rules! fmadd_2v {
	(0) => {
		concat!(
			vfmadd231ps!(0, 2, 8),
			vfmadd231ps!(1, 2, 9),
		)
	};
	(1) => {
		concat!(
			vfmadd231ps!(0, 3, 10),
			vfmadd231ps!(1, 3, 11),
		)
	};
	(2) => {
		concat!(
			vfmadd231ps!(0, 4, 12),
			vfmadd231ps!(1, 4, 13),
		)
	};
	(3) => {
		concat!(
			vfmadd231ps!(0, 5, 14),
			vfmadd231ps!(1, 5, 15),
		)
	};
	(4) => {
		concat!(
			vfmadd231ps!(0, 6, 16),
			vfmadd231ps!(1, 6, 17),
		)
	};
	(5) => {
		concat!(
			vfmadd231ps!(0, 7, 18),
			vfmadd231ps!(1, 7, 19),
		)
	};
	(6) => {
		concat!(
			vfmadd231ps!(0, 2, 20),
			vfmadd231ps!(1, 2, 21),
		)
	};
	(7) => {
		concat!(
			vfmadd231ps!(0, 3, 22),
			vfmadd231ps!(1, 3, 23),
		)
	};
	(8) => {
		concat!(
			vfmadd231ps!(0, 4, 24),
			vfmadd231ps!(1, 4, 25),
		)
	};
	(9) => {
		concat!(
			vfmadd231ps!(0, 5, 26),
			vfmadd231ps!(1, 5, 27),
		)
	};
	(10) => {
		concat!(
			vfmadd231ps!(0, 6, 28),
			vfmadd231ps!(1, 6, 29),
		)
	};
	(11) => {
		concat!(
			vfmadd231ps!(0, 7, 30),
			vfmadd231ps!(1, 7, 31),
		)
	};
}

macro_rules! fmadd_1v {
	(0) => {
		concat!(
			vfmadd231ps!(0, 1, 20),
		)
	};
	(1) => {
		concat!(
			vfmadd231ps!(0, 2, 21),
		)
	};
	(2) => {
		concat!(
			vfmadd231ps!(0, 3, 22),
		)
	};
	(3) => {
		concat!(
			vfmadd231ps!(0, 4, 23),
		)
	};
	(4) => {
		concat!(
			vfmadd231ps!(0, 5, 24),
		)
	};
	(5) => {
		concat!(
			vfmadd231ps!(0, 6, 25),
		)
	};
	(6) => {
		concat!(
			vfmadd231ps!(0, 7, 26),
		)
	};
	(7) => {
		concat!(
			vfmadd231ps!(0, 8, 27),
		)
	};
	(8) => {
		concat!(
			vfmadd231ps!(0, 9, 28),
		)
	};
	(9) => {
		concat!(
			vfmadd231ps!(0, 10, 29),
		)
	};
	(10) => {
		concat!(
			vfmadd231ps!(0, 11, 30),
		)
	};
	(11) => {
		concat!(
			vfmadd231ps!(0, 12, 31),
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
macro_rules! asm_48x8_step {
	($N:tt, $nr:tt, $a_layout:tt, $b_layout:tt, $K:tt) => {
		seq!(n in 0..$nr {
			concat!(
				load_a!($N, $a_layout, $K),
				#(
					load_b!($b_layout, n, $K, $nr, b_num_48x8!(n)),
					fmadd_3v!(n),
				)*
			)
		})
	};
}

// ***************************** 32x12 ******************************* //
macro_rules! asm_32x12_step {
	($N:tt, $nr:tt, $a_layout:tt, $b_layout:tt, $K:tt) => {
		seq!(n in 0..$nr {
			concat!(
				load_a!($N, $a_layout, $K),
				#(
					load_b!($b_layout, n, $K, $nr, b_num_32x12!(n)),
					fmadd_2v!(n),
				)*
			)
		})
	};
}

// ***************************** 16x12 ******************************* //
macro_rules! asm_16x12_step {
	($N:tt, $nr:tt, $a_layout:tt, $b_layout:tt, $K:tt) => {
		seq!(n in 0..$nr {
			concat!(
				load_a!($N, $a_layout, $K),
				#(
					load_b!($b_layout, n, $K, $nr, b_num_16x12!(n)),
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
    	pub(crate) unsafe fn $func_name<F:MyFn>(
        	a: *const f32, b: *const f32, c: *mut f16,
        	alpha: *const f32, beta: *const f32,
        	k: usize,
        	ldc: usize,
			ld_arr: [usize; 2],
			f: F
    	) {
        	let k_iter = k / $unroll;
        	let k_left = k % $unroll;
            let u64_arr = [ld_arr[0], ld_arr[1], ldc*2, k_iter, k_left];
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

            	"vbroadcastss ({betax}), %zmm0",

            	"vxorps %ymm3,%ymm3,%ymm3",
            	"vucomiss %xmm3,%xmm0",

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
				x4 = out(reg) _,
            	x5 = out(reg) _,
            	out("xmm0") _, out("xmm1") _, out("xmm2") _, out("xmm3") _,
            	out("xmm4") _, out("xmm5") _, out("xmm6") _, out("xmm7") _,
            	out("xmm8") _, out("xmm9") _, out("xmm10") _, out("xmm11") _,
            	out("xmm12") _, out("xmm13") _, out("xmm14") _, out("xmm15") _,
            	options(att_syntax)
        	);

			for j in 0..$nr {
				for i in 0..$mr/16 {
					f.call(c.add(i*16+j*ldc), 16);
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
        	a: *const f32, b: *const f32, c: *mut f16,
        	alpha: *const f32, beta: *const f32,
        	k: usize,
        	ldc: usize,
			ld_arr: [usize; 2],
			mask: *const u16,
			f: F
    	) {
        	let k_iter = k / $unroll;
        	let k_left = k % $unroll;
            let u64_arr = [ld_arr[0], ld_arr[1], ldc*2, k_iter, k_left];
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

            	"vbroadcastss ({betax}), %zmm0",

            	"vxorps %ymm3,%ymm3,%ymm3",
            	"vucomiss %xmm3,%xmm0",
				"kmovw ({maskx}), %k1",
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
				x4 = out(reg) _,
            	x5 = out(reg) _,
            	out("xmm0") _, out("xmm1") _, out("xmm2") _, out("xmm3") _,
            	out("xmm4") _, out("xmm5") _, out("xmm6") _, out("xmm7") _,
            	out("xmm8") _, out("xmm9") _, out("xmm10") _, out("xmm11") _,
            	out("xmm12") _, out("xmm13") _, out("xmm14") _, out("xmm15") _,
				out("k1") _,
            	options(att_syntax)
        	);
			for j in 0..$nr {
				for i in 0..$mr/16 {
					f.call(c.add(i*16+j*ldc), 16);
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
					0, 128,
					4,
					[<ukernel_ $mr x nr _ $func_name>]
				);
			}
		});
	};
}

group_def_ukernel!(48, 1, 8, B, B, bb, def_ukernel, asm_48x8_step, asm_48x8_acc, asm_48x8_store, VER48);
group_def_ukernel!(48, 1, 8, B, B, bb_partial, def_ukernel_partial, asm_48x8_step, asm_48x8_acc, asm_48x8_store, VER48);

group_def_ukernel!(32, 1, 8, B, B, bb, def_ukernel, asm_32x12_step, asm_32x12_acc, asm_32x12_store, VER32);
group_def_ukernel!(32, 1, 8, B, B, bb_partial, def_ukernel_partial, asm_32x12_step, asm_32x12_acc, asm_32x12_store, VER32);

group_def_ukernel!(16, 1, 8, B, B, bb, def_ukernel, asm_16x12_step, asm_16x12_acc, asm_16x12_store, VER16);
group_def_ukernel!(16, 1, 8, B, B, bb_partial, def_ukernel_partial, asm_16x12_step, asm_16x12_acc, asm_16x12_store, VER16);

