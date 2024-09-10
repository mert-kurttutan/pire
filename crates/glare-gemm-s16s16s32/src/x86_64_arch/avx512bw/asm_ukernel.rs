use seq_macro::seq;
use std::arch::asm;
use crate::MyFn;
use super::VS;

use crate::{TA, TB, TC};

macro_rules! beta_fmadd {
    (C, $m0:expr, $r:expr, 1) => {
        concat!(
			"vpaddd ", $m0, ", %zmm", $r, ", %zmm", $r, "\n",
        ) 
    };
    (C, $m0:expr, $r:expr, 2) => {
        concat!(
			"vcvtdq2ps %zmm", $r, ",%zmm", $r, "\n",
			"vcvtdq2ps ", $m0, ",%zmm30", "\n",
			"vfmadd231ps %zmm30,%zmm0,%zmm", $r, "\n",
			"vcvtps2dq %zmm", $r, ",%zmm", $r, "\n",
        ) 
    };
    (M, $m0:expr, $r:expr, 1) => {
        concat!(
			"vmovups ", $m0, ", %zmm30 {{%k1}}", "\n",
			"vpaddd %zmm30, %zmm", $r, ", %zmm", $r, "\n",
        ) 
    };

    (M, $m0:expr, $r:expr, 2) => {
        concat!(
			"vmovups ", $m0, ", %zmm30 {{%k1}}", "\n",
			"vcvtdq2ps %zmm30,%zmm30", "\n",
			"vcvtdq2ps %zmm", $r, ",%zmm", $r, "\n",
			"vfmadd231ps %zmm30,%zmm0,%zmm", $r, "\n",
			"vcvtps2dq %zmm", $r, ",%zmm", $r, "\n",
        ) 
    };
}

macro_rules! vzeroall {
	($r0:tt, $r1:tt) => {
		seq!(r in $r0..=$r1 {
			concat!(#("vpxorq %zmm",r,",%zmm",r,",%zmm",r,"\n",)*)
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
    ($r1:expr, $r2:expr, $r3:expr, $r4:expr) => {
        concat!(
			"vpmaddwd %zmm", $r1, ", %zmm", $r2, ", %zmm", $r4, "\n",
			"vpaddd %zmm", $r4, ", %zmm", $r3, ", %zmm", $r3, "\n",
        ) 
    };
}

macro_rules! loadp_unit {
    ($layout:tt, $m0:expr, $r1:expr) => {
        concat!(
            vmovp!($layout), $m0, ",%zmm", $r1, "\n",
        )
    };
 }

 macro_rules! storep_unit {
    (C, $r1:expr, $m0:expr) => {
        concat!(
            "vmovups %zmm", $r1, ", ", $m0,  "\n",
        )
    };
    (B, $r1:expr, $m0:expr) => {
        concat!(
            "vmovaps %zmm", $r1, ", ", $m0,  "\n",
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
				// jmp to 8 if alpha is equal to onex
				"vbroadcastss ({alphax}),%ymm1", "\n",
				"vucomiss ({onex}), %xmm1 \n",
				"vbroadcastss ({alphax}),%zmm1", "\n",
				#(
					"vcvtdq2ps %zmm", r, ",%zmm", r, "\n",
					"vmulps %zmm1, %zmm", r, ",%zmm", r, "\n",
					"vcvtps2dq %zmm", r, ",%zmm", r, "\n",
				)*
			)
		})
	}
}

macro_rules! load_beta {
	() => {
		concat!(
			vbroadcast!(), " ({betax}), %zmm0\n",
			"vxorps %zmm31,%zmm31,%zmm31\n",
			"vucomiss %xmm31,%xmm0\n",
		)
	}
}

macro_rules! mem {
    ($m0:tt, $b0:tt) => {
        concat!($b0, "+", $m0)
    };
 }

 macro_rules! acc_p {
    ($layout:tt, $m0:expr, $r1:expr, $r2:expr, $b:tt) => {
        concat!(
            beta_fmadd!(C, $m0, $r1, $b),
            beta_fmadd!($layout, mem!($m0, "0x40"), $r2, $b),
        )
    };
    ($layout:tt, $m0:expr, $r1:expr, $b:tt) => {
        concat!(
            beta_fmadd!($layout, $m0, $r1, $b),
        )
    };
 }


 macro_rules! loadp {
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
 }

macro_rules! storep {
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
        	"lea ({x2}, {x2}, 2), {x3}", "\n",
        	"lea ({bx}, {x3}, 1), {x3}", "\n",
			"lea ({bx}, {x2}, 1), {x4}", "\n",
			"lea ({bx}, {x2}, 2), {x5}", "\n",

        	"mov 24({dim_arrx}),{x0}", "\n",
        	"test {x0},{x0}", "\n",
    	)
	};
}


macro_rules! asm_c_load {
	(8) => {
    	concat!(
        	"mov 16({dim_arrx}),{x0}", "\n",
        	"lea ({x0}, {x0}, 2), {x2}", "\n",
        	"lea ({cx}, {x2},), {x1}", "\n",
			"lea ({x1}, {x2},), {x2}", "\n",
    	)
	};
	(7) => {
    	concat!(
        	"mov 16({dim_arrx}),{x0}", "\n",
        	"lea ({x0}, {x0}, 2), {x2}", "\n",
        	"lea ({cx}, {x2},), {x1}", "\n",
			"lea ({x1}, {x2},), {x2}", "\n",
    	)
	};
	(6) => {
    	concat!(
        	"mov 16({dim_arrx}),{x0}", "\n",
        	"lea ({x0}, {x0}, 2), {x2}", "\n",
        	"lea ({cx}, {x2},), {x1}", "\n",
			"lea ({x1}, {x2},), {x2}", "\n",
    	)
	};
	(5) => {
    	concat!(
        	"mov 16({dim_arrx}),{x0}", "\n",
        	"lea ({x0}, {x0}, 2), {x2}", "\n",
        	"lea ({cx}, {x2},), {x1}", "\n",
			"lea ({x1}, {x2},), {x2}", "\n",
    	)
	};
	(4) => {
    	concat!(
        	"mov 16({dim_arrx}),{x0}", "\n",
        	"lea ({x0}, {x0}, 2), {x2}", "\n",
        	"lea ({cx}, {x2},), {x1}", "\n",
			"lea ({x1}, {x2},), {x2}", "\n",
    	)
	};
	(3) => {
    	concat!(
        	"mov 16({dim_arrx}),{x0}", "\n",
        	"lea ({x0}, {x0}, 2), {x2}", "\n",
        	"lea ({cx}, {x2},), {x1}", "\n",
			"lea ({x1}, {x2},), {x2}", "\n",
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

	(32,8) => {vzeroall!(2,17)};
	(32,7) => {vzeroall!(2,15)};
	(32,6) => {vzeroall!(2,13)};
	(32,5) => {vzeroall!(2,11)};
	(32,4) => {vzeroall!(2,9)};
	(32,3) => {vzeroall!(2,7)};
	(32,2) => {vzeroall!(2,5)};
	(32,1) => {vzeroall!(2,3)};

	(16,8) => {vzeroall!(2,9)};
	(16,7) => {vzeroall!(2,8)};
	(16,6) => {vzeroall!(2,7)};
	(16,5) => {vzeroall!(2,6)};
	(16,4) => {vzeroall!(2,5)};
	(16,3) => {vzeroall!(2,4)};
	(16,2) => {vzeroall!(2,3)};
	(16,1) => {vzeroall!(2,2)};
}


macro_rules! asm_alpha_scale {
	(32, 8) => {asm_alpha_scale_0!(2,17)};
	(32, 7) => {asm_alpha_scale_0!(2,15)};
	(32, 6) => {asm_alpha_scale_0!(2,13)};
	(32, 5) => {asm_alpha_scale_0!(2,11)};
	(32, 4) => {asm_alpha_scale_0!(2,9)};
	(32, 3) => {asm_alpha_scale_0!(2,7)};
	(32, 2) => {asm_alpha_scale_0!(2,5)};
	(32, 1) => {asm_alpha_scale_0!(2,3)};

	(16, 8) => {asm_alpha_scale_0!(2,9)};
	(16, 7) => {asm_alpha_scale_0!(2,8)};
	(16, 6) => {asm_alpha_scale_0!(2,7)};
	(16, 5) => {asm_alpha_scale_0!(2,6)};
	(16, 4) => {asm_alpha_scale_0!(2,5)};
	(16, 3) => {asm_alpha_scale_0!(2,4)};
	(16, 2) => {asm_alpha_scale_0!(2,3)};
	(16, 1) => {asm_alpha_scale_0!(2,2)};
}

macro_rules! acc_32x8 {
	(0, $layout:tt, $b:tt) => {
		acc_p!($layout, "0({cx})", 2, 3, $b)
	};
	(1, $layout:tt, $b:tt) => {
		acc_p!($layout, "0({cx}, {x0})", 4, 5, $b)
	};
	(2, $layout:tt, $b:tt) => {
		acc_p!($layout, "0({cx}, {x0}, 2)", 6, 7, $b)
	}; 
	(3, $layout:tt, $b:tt) => {
		acc_p!($layout, "0({x1})", 8, 9, $b)
	};
	(4, $layout:tt, $b:tt) => {
		acc_p!($layout, "0({x1}, {x0})", 10, 11, $b)
	};
	(5, $layout:tt, $b:tt) => {
		acc_p!($layout, "0({x1}, {x0}, 2)", 12, 13, $b)
	};
	(6, $layout:tt, $b:tt) => {
		acc_p!($layout, "0({x2})", 14, 15, $b)
	};
	(7, $layout:tt, $b:tt) => {
		acc_p!($layout, "0({x2}, {x0})", 16, 17, $b)
	};
}

macro_rules! store_32x8 {
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
}

macro_rules! acc_16x8 {
	(0, $layout:tt, $b:tt) => {
		acc_p!($layout, "0({cx})", 2, $b)
	};
	(1, $layout:tt, $b:tt) => {
		acc_p!($layout, "0({cx}, {x0})", 3, $b)
	};
	(2, $layout:tt, $b:tt) => {
		acc_p!($layout, "0({cx}, {x0}, 2)", 4, $b)
	}; 
	(3, $layout:tt, $b:tt) => {
		acc_p!($layout, "0({x1})", 5, $b)
	};
	(4, $layout:tt, $b:tt) => {
		acc_p!($layout, "0({x1}, {x0})", 6, $b)
	};
	(5, $layout:tt, $b:tt) => {
		acc_p!($layout, "0({x1}, {x0}, 2)", 7, $b)
	};
	(6, $layout:tt, $b:tt) => {
		acc_p!($layout, "0({x2})", 8, $b)
	};
	(7, $layout:tt, $b:tt) => {
		acc_p!($layout, "0({x2}, {x0})", 9, $b)
	};

}

macro_rules! store_16x8 {
	(0, $layout:tt) => {
		storep!($layout, "0({cx})", 2)
	};
	(1, $layout:tt) => {
		storep!($layout, "0({cx}, {x0})", 3)
	};
	(2, $layout:tt) => {
		storep!($layout, "0({cx}, {x0}, 2)", 4)
	}; 
	(3, $layout:tt) => {
		storep!($layout, "0({x1})", 5)
	};
	(4, $layout:tt) => {
		storep!($layout, "0({x1}, {x0})", 6)
	};
	(5, $layout:tt) => {
		storep!($layout, "0({x1}, {x0}, 2)", 7)
	};
	(6, $layout:tt) => {
		storep!($layout, "0({x2})", 8)
	};
	(7, $layout:tt) => {
		storep!($layout, "0({x2}, {x0})", 9)
	};
}


macro_rules! cum_seq {
	($step_macro:tt, $nr:tt, $layout:tt, $b:tt) => {
		seq!(n in 0..$nr {
			concat!(#($step_macro!(n, $layout, $b),)*)
		})
	};
	($step_macro:tt, $nr:tt, $layout:tt) => {
		seq!(n in 0..$nr {
			concat!(#($step_macro!(n, $layout),)*)
		})
	};
}

macro_rules! load_b {
	(B, $N:tt, $r:expr) => {
    	concat!(
        	vbroadcast!(), "  ", $N, "*4({bx}), %zmm", $r, "\n",
    	)
	};
}


macro_rules! load_a {
	($mr:tt, B) => {
    	loadp!($mr, B, "0({ax})")
	};
	($mr:tt, C, $K:tt) => {
    	loadp!($mr, C, "0({ax})")
	};
}

macro_rules! fmadd_2v {
	(0) => {
		concat!(
			vfmadd!(0, 18, 2, 22),
			vfmadd!(1, 18, 3, 23),
		)
	};
	(1) => {
		concat!(
			vfmadd!(0, 19, 4, 24),
			vfmadd!(1, 19, 5, 25),
		)
	};
	(2) => {
		concat!(
			vfmadd!(0, 20, 6, 26),
			vfmadd!(1, 20, 7, 27),
		)
	};
	(3) => {
		concat!(
			vfmadd!(0, 21, 8, 28),
			vfmadd!(1, 21, 9, 29),
		)
	};
	(4) => {
		concat!(
			vfmadd!(0, 18, 10, 30),
			vfmadd!(1, 18, 11, 31),
		)
	};
	(5) => {
		concat!(
			vfmadd!(0, 19, 12, 22),
			vfmadd!(1, 19, 13, 23),
		)
	};
	(6) => {
		concat!(
			vfmadd!(0, 20, 14, 24),
			vfmadd!(1, 20, 15, 25),
		)
	};
	(7) => {
		concat!(
			vfmadd!(0, 21, 16, 26),
			vfmadd!(1, 21, 17, 27),
		)
	};
}

macro_rules! fmadd_1v {
	(0) => {
		concat!(
			vfmadd!(0, 10, 2, 18),
		)
	};
	(1) => {
		concat!(
			vfmadd!(0, 11, 3, 19),
		)
	};
	(2) => {
		concat!(
			vfmadd!(0, 12, 4, 20),
		)
	};
	(3) => {
		concat!(
			vfmadd!(0, 13, 5, 21),
		)
	};
	(4) => {
		concat!(
			vfmadd!(0, 14, 6, 22),
		)
	};
	(5) => {
		concat!(
			vfmadd!(0, 15, 7, 23),
		)
	};
	(6) => {
		concat!(
			vfmadd!(0, 16, 8, 24),
		)
	};
	(7) => {
		concat!(
			vfmadd!(0, 17, 9, 25),
		)
	};
}

macro_rules! b_num_32x8 {
	(0) => {18};
	(1) => {19};
	(2) => {20};
	(3) => {21};
	(4) => {18};
	(5) => {19};
	(6) => {20};
	(7) => {21};
}

macro_rules! b_num_16x8 {
	(0) => {10};
	(1) => {11};
	(2) => {12};
	(3) => {13};
	(4) => {14};
	(5) => {15};
	(6) => {16};
	(7) => {17};
}

// ***************************** 32x8 ******************************* //
macro_rules! step_32x8 {
	(8, B, B) => {
		concat!(

			load_a!(32, B),
			"add $128, {ax}\n",
			load_b!(B, 0, 18),
			fmadd_2v!(0),

			load_b!(B, 1, 19),
			fmadd_2v!(1),

			load_b!(B, 2, 20),
			"prefetcht0 256({ax}) \n",
			fmadd_2v!(2),

			load_b!(B, 3, 21),
			fmadd_2v!(3),

			load_b!(B, 4, 18),
			"prefetcht0 320({ax}) \n",
			fmadd_2v!(4),

			load_b!(B, 5, 19),
			"prefetcht0 64({bx}) \n",
			fmadd_2v!(5),

			load_b!(B, 6, 20),
			fmadd_2v!(6),

			load_b!(B, 7, 21),
			fmadd_2v!(7),

			"add $32, {bx}\n",	
		)
	};
	($nr:tt, $a_layout:tt, $b_layout:tt) => {
		seq!(n in 0..$nr {
			concat!(
				load_a!(32, $a_layout),
				"add $128, {ax}\n",
				#(
					load_b!($b_layout, n, b_num_32x8!(n)),
					fmadd_2v!(n),
				)*
				"add $4*", $nr, ", {bx}\n",
			)
		})
	};
}

// ***************************** 16x8 ******************************* //
macro_rules! step_16x8 {
	($nr:tt, $a_layout:tt, $b_layout:tt) => {
		seq!(n in 0..$nr {
			concat!(
				load_a!(16, $a_layout),
				"add $64, {ax}\n",
				#(
					load_b!($b_layout, n, b_num_16x8!(n)),
					fmadd_1v!(n),
				)*
				"add $4*", $nr, ", {bx}\n",
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
    (32, $nr:tt, $c:tt, $ldc:tt) => {
        seq!(j in 0..$nr {
            _mm_prefetch($c.add(j*$ldc) as *const i8, 3);
			_mm_prefetch($c.add(16+j*$ldc) as *const i8, 3);
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
        	alpha: *const f32, beta: *const f32,
        	k: usize,
			ld_arr: [usize; 4],
			m: usize,
			f: F,
    	) {
			mask_ptr!($is_partial, m, x);
			let mask_ptr = (&x) as *const u16;
			let k = (k+1) / 2 *2;
        	let k_iter = k / 8;
        	let k_left = (k % 8) / 2;
            let mut dim_arr = [ld_arr[0]*4, ld_arr[1]*4, ld_arr[3]*4, k_iter, k_left];
			let mut cf = c;
			let mut c_buf = [0i32;$mr*$nr];
			let c_cs = ld_arr[3];
			let one = 1_f32;
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
            	asm_vzeroall!($mr,$nr),
   	 
            	asm_init_ab!($mr,$a_layout,$b_layout),
           	 
            	// 3 -> CONSIDKLEFT
            	"je 3f",
           	 
            	// 2 -> KITER
            	"2:",
				prefetch_0!(128, "{bx}", 0),
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

            	asm_alpha_scale!($mr, $nr),

				"8:",

				load_beta!(),
				load_mask_ptr_asm!($is_partial),				


            	// 6 -> BETAZERO
            	"je 6f",

				// check if beta is equal to 1
				"vucomiss ({onex}), %xmm0",
				"je 9f",

				cum_seq!($acc_macro,$nr,$is_partial,2),
				"jmp 6f",

				"9:",
				// 9 -> BETA ONE
				cum_seq!($acc_macro,$nr,$is_partial,1),

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
				onex = inout(reg) &one => _,
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
				for i in 0..$mr/8 {
					f.call(c.add(i*8+j*c_cs), 8);
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
        	alpha: *const f32, beta: *const f32,
        	k: usize,
			d_arr: [usize; 4],
			m: usize, n: usize,
			f: F,
    	) {
			mask_ptr!($is_partial, m, x);
			let mask_ptr = (&x) as *const u16;
			let k = (k+1) / 2 *2;
        	let k_iter = k / 8;
        	let k_left = (k % 8) / 2;
            let mut dim_arr = [d_arr[0]*4, d_arr[1]*4, d_arr[3]*4, k_iter, k_left];
			let mut cf = c;
			let mut c_buf = [0i32;$mr*$nr];
			let c_cs = d_arr[3];
			let one = 1_f32;
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
							asm_vzeroall!($mr,ni),
				
							asm_init_ab!($mr,$a_layout,$b_layout),
						
							// 3 -> CONSIDKLEFT
							"je 3f",
						
							// 2 -> KITER
							"2:",
							prefetch_0!(128, "{bx}", 0),
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
							// jmp to 8 if alpha is equal to onex
							"vmovss ({alphax}), %xmm0",
							"vucomiss ({onex}), %xmm0",
							"je 8f",
							// scale by alpha
							asm_alpha_scale!($mr, ni),

							"8:",
							load_mask_ptr_asm!($is_partial),
							load_beta!(),

							// 6 -> BETAZERO
							"je 6f",

							// check if beta is equal to 1
							"vucomiss ({onex}), %xmm0",
							"je 9f",

							cum_seq!($acc_macro,ni,$is_partial,2),
							"jmp 6f",

							"9:",
							// 9 -> BETA ONE
							cum_seq!($acc_macro,ni,$is_partial,1),

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
							onex = inout(reg) &one => _,
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

// def_ukernel!(step_32x8, acc_32x8, store_32x8, 32, 8, B, B, C, ukernel_32x8_bb);
// def_ukernel!(step_16x8, acc_16x8, store_16x8, 8, 4, B, B, C, 4, ukernel_16x8_bb);

def_ukernel!(step_32x8, acc_32x8, store_32x8, 32, 8, B, B, M, ukernel_32x8_bb_partial);
def_ukernel!(step_16x8, acc_16x8, store_16x8, 16, 8, B, B, M, ukernel_16x8_bb_partial);


def_ukernelxn!(step_32x8, acc_32x8, store_32x8, 32, 8, B, B, C, ukernel_32xn_bb);
// def_ukernelxn!(32, step_32x8, acc_32x8, store_32x8, 16, 4, B, B, C, 4, ukernel_16xn_bb);
// def_ukernelxn!(16, step_16x8, acc_16x8, store_16x8, 8, 4, B, B, C, 4, ukernel_16xn_bb);

def_ukernelxn!(step_32x8, acc_32x8, store_32x8, 32, 8, B, B, M, ukernel_32xn_bb_partial);
def_ukernelxn!(step_16x8, acc_16x8, store_16x8, 16, 8, B, B, M, ukernel_16xn_bb_partial);



pub(crate) unsafe fn ukernel_32x8_bb<F: MyFn, const BUF: bool>(
    a: *const TA, b: *const TB, c: *mut TC,
    alpha: *const f32, beta: *const f32,
    k: usize,
    d_arr: [usize; 4],
    a_pft1_offset: usize,
    f: F,
) {
	let k_left0 = k % 16;
	let k_left = if k_left0 == 0 {8} else {k_left0 / 2};
	let k_iter = (k - k_left*2) / 8;

	// let k = (k+1) / 2 *2;
	// let k_iter = k / 8;
	// let k_left = (k % 8) / 2;
	let one = 1_f32;
    let mut dim_arr = [d_arr[3]*4, k_iter, k_left, a_pft1_offset];
    let mut cf = c;
    let mut c_buf = [0i32; 48 * 8];
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
		asm_vzeroall!(32,8),
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
		step_32x8!(8, B, B),

		"movq $64*4, {x4}",
		// divisiblity by 4
		"testq $3, {x0}",
		"cmovz {x1},{x4}",

		step_32x8!(8, B, B),

		"prefetcht1 ({x2})",

		"subq $64*3, {x2}",
		"addq {x4}, {x2}",

		step_32x8!(8, B, B),

		"prefetcht1 ({x5})",
		"addq $32, {x5}",

		"testq $63, {x0}",
		"cmovz {cx},{x2}",

		step_32x8!(8, B, B),

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
		step_32x8!(8, B, B),

		"add {x1}, {x2}",
		"dec {x0}",
		"jne 4b",
		
		"5:",
		"mov ({dim_arrx}),{x0}",
		"lea ({x0}, {x0}, 2), {x4}",
		"lea ({cx}, {x4},), {x1}",
		"lea ({x1}, {x4},), {x2}",
		asm_alpha_scale!(32, 8),
		"8:",

		load_beta!(),

		// 6 -> BETAZERO
		"je 6f",

		// check if beta is equal to 1
		"vucomiss ({onex}), %xmm0",
		"je 9f",

		cum_seq!(acc_32x8,8,C,2),
		"jmp 6f",

		"9:",
		// 9 -> BETA ONE
		cum_seq!(acc_32x8,8,C,1),

		// 6 -> BETAZERO
		"6:",
		cum_seq!(store_32x8,8,C),

		"7:",
        ax = inout(reg) a => _, 
		bx = inout(reg) b => _, 
		cx = inout(reg) cf => _,
		dim_arrx = inout(reg) dim_arr.as_ptr() => _, 
		alphax = inout(reg) alpha => _, 
		betax = inout(reg) beta => _, 
		onex = inout(reg) &one => _,
		x0 = out(reg) _, 
		x1 = out(reg)_, 
		x2 = out(reg) _, 
		// x3 = out(reg) _, 
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