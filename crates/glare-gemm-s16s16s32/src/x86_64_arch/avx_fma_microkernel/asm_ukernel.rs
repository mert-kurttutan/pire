use seq_macro::seq;
use std::arch::asm;

use crate::{TA, TB, TC};

use paste::paste;
macro_rules! beta_fmaddps {
    (C, $m0:expr, $r:expr, 1) => {
        concat!(
			"vpaddd ", $m0, ", %ymm", $r, ", %ymm", $r, "\n",
            // "vfmadd231ps ", $m0, ",%ymm0,%ymm", $r1, "\n",
        ) 
    };
    (C, $m0:expr, $r:expr, 2) => {
        concat!(
			"vcvtdq2ps %ymm", $r, ",%ymm", $r, "\n",
			"vfmadd231ps ", $m0, ",%ymm0,%ymm", $r, "\n",
			"vcvtps2dq %ymm", $r, ",%ymm", $r, "\n",
        ) 
    };
    (M, $m0:expr, $r:expr, 1) => {
        concat!(
            "vmaskmovps ", $m0, ", %ymm1", ", %ymm2",  "\n",
            // "vfmadd231ps %ymm2, %ymm0,%ymm", $r1, "\n",
			"vpaddd %ymm2, %ymm", $r, ", %ymm", $r, "\n",
        ) 
    };

    (M, $m0:expr, $r:expr, 2) => {
        concat!(
			"vmaskmovps ", $m0, ", %ymm1", ", %ymm2",  "\n",
			"vcvtdq2ps %ymm", $r, ",%ymm", $r, "\n",
			"vfmadd231ps %ymm2,%ymm0,%ymm", $r, "\n",
			"vcvtps2dq %ymm", $r, ",%ymm", $r, "\n",
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
    (16, $layout:tt, $m0:expr, $r1:expr, $r2:expr, $b:tt) => {
        concat!(
            beta_fmaddps!(C, $m0, $r1, $b),
            beta_fmaddps!($layout, mem!($m0, "0x20"), $r2, $b),
        )
    };
    ($N:tt, $layout:tt, $m0:expr, $r1:expr, $b:tt) => {
        concat!(
            beta_fmaddps!($layout, $m0, $r1, $b),
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
    ($r1:expr, $r2:expr, $r3:expr, $r4:expr) => {
        concat!(
            // "vfmadd231ps %ymm", $r1, ", %ymm", $r2,", %ymm", $r3, "\n",
			"vpmaddwd %ymm", $r1, ", %ymm", $r2, ", %ymm", $r4, "\n",
			"vpaddd %ymm", $r4, ", %ymm", $r3, ", %ymm", $r3, "\n",
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
        	"lea ({x2}, {x2}, 2), {x3}", "\n",
        	"lea ({bx}, {x3}, 1), {x3}", "\n",
			"lea ({bx}, {x2}, 1), {x4}", "\n",
			"lea ({bx}, {x2}, 2), {x5}", "\n",

        	"mov 24({int_arrx}),{x0}", "\n",
        	"test {x0},{x0}", "\n",
    	)
	};
}


macro_rules! asm_c_load {
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

	(VER16,4) => {vzeroall!(4,11)};
	(VER16,3) => {vzeroall!(4,9)};
	(VER16,2) => {vzeroall!(4,7)};
	(VER16,1) => {vzeroall!(4,5)};

	(VER8,4) => {vzeroall!(5,8)};
	(VER8,3) => {vzeroall!(5,7)};
	(VER8,2) => {vzeroall!(5,6)};
	(VER8,1) => {vzeroall!(5,5)};
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
	(S, $X:tt, $K:tt) => {
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
					"vcvtdq2ps %ymm", r, ",%ymm", r, "\n",
					"vmulps %ymm1, %ymm", r, ",%ymm", r, "\n",
					"vcvtps2dq %ymm", r, ",%ymm", r, "\n",
				)*
			)
		})
	}
}
macro_rules! asm_alpha_scale {
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

	(VER8, 4) => {
    	asm_alpha_scale_0!(5,8)
	};
	(VER8, 3) => {
    	asm_alpha_scale_0!(5,7)
	};
	(VER8, 2) => {
    	asm_alpha_scale_0!(5,6)
	};
	(VER8, 1) => {
    	asm_alpha_scale_0!(5,5)
	};
}

macro_rules! asm_16x4_acc_seq {
	($mr:tt, 0, $layout:tt, $b:tt) => {
		acc_ps!($mr, $layout, "0({cx})", 4, 5, $b)
	};
	($mr:tt, 1, $layout:tt, $b:tt) => {
		acc_ps!($mr, $layout, "0({cx}, {x0})", 6, 7, $b)
	};
	($mr:tt, 2, $layout:tt, $b:tt) => {
		acc_ps!($mr, $layout, "0({cx}, {x0}, 2)", 8, 9, $b)
	}; 
	($mr:tt, 3, $layout:tt, $b:tt) => {
		acc_ps!($mr, $layout, "0({x1})", 10, 11, $b)
	};
}

macro_rules! asm_16x4_store_seq {
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
}

macro_rules! asm_8x4_acc_seq {
	($mr:tt, 0, $layout:tt, $b:tt) => {
		acc_ps!($mr, $layout, "0({cx})", 5, $b)
	};
	($mr:tt, 1, $layout:tt, $b:tt) => {
		acc_ps!($mr, $layout, "0({cx}, {x0})", 6, $b)
	};
	($mr:tt, 2, $layout:tt, $b:tt) => {
		acc_ps!($mr, $layout, "0({cx}, {x0}, 2)", 7, $b)
	}; 
	($mr:tt, 3, $layout:tt, $b:tt) => {
		acc_ps!($mr, $layout, "0({x1})", 8, $b)
	};
}

macro_rules! asm_8x4_store_seq {
	($mr:tt, 0, $layout:tt) => {
		storeps!($mr, $layout, "0({cx})", 5)
	};
	($mr:tt, 1, $layout:tt) => {
		storeps!($mr, $layout, "0({cx}, {x0})", 6)
	};
	($mr:tt, 2, $layout:tt) => {
		storeps!($mr, $layout, "0({cx}, {x0}, 2)", 7)
	}; 
	($mr:tt, 3, $layout:tt) => {
		storeps!($mr, $layout, "0({x1})", 8)
	};
}

macro_rules! asm_16x4_acc {
	($mr:tt, $nr:tt, $layout:tt, $b:tt) => {
		seq!(n in 0..$nr {
			concat!(
				#(
					asm_16x4_acc_seq!($mr, n, $layout, $b), 
				)*
			)
		})
	};
}

macro_rules! asm_16x4_store {
	($mr:tt, $nr:tt, $layout:tt) => {
		seq!(n in 0..$nr {
			concat!(
				#(
					asm_16x4_store_seq!($mr, n, $layout), 
				)*
			)
		})
	};
}


macro_rules! asm_8x4_acc {
	($mr:tt, $nr:tt, $layout:tt, $b:tt) => {
		seq!(n in 0..$nr {
			concat!(
				#(
					asm_8x4_acc_seq!($mr, n, $layout, $b), 
				)*
			)
		})
	};
}


macro_rules! asm_8x4_store {
	($mr:tt, $nr:tt, $layout:tt) => {
		seq!(n in 0..$nr {
			concat!(
				#(
					asm_8x4_store_seq!($mr, n, $layout), 
				)*
			)
		})
	};
}

macro_rules! load_b {
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

macro_rules! fmadd_2v {
	(0) => {
		concat!(
			vfmadd231ps!(0, 2, 4, 12),
			vfmadd231ps!(1, 2, 5, 13),
		)
	};
	(1) => {
		concat!(
			vfmadd231ps!(0, 3, 6, 14),
			vfmadd231ps!(1, 3, 7, 15),
		)
	};
	(2) => {
		concat!(
			vfmadd231ps!(0, 2, 8, 12),
			vfmadd231ps!(1, 2, 9, 13),
		)
	};
	(3) => {
		concat!(
			vfmadd231ps!(0, 3, 10, 14),
			vfmadd231ps!(1, 3, 11, 15),
		)
	};
}

macro_rules! fmadd_1v {
	(0) => {
		concat!(
			vfmadd231ps!(0, 1, 5, 9),
		)
	};
	(1) => {
		concat!(
			vfmadd231ps!(0, 2, 6, 10),
		)
	};
	(2) => {
		concat!(
			vfmadd231ps!(0, 3, 7, 11),
		)
	};
	(3) => {
		concat!(
			vfmadd231ps!(0, 4, 8, 12),
		)
	};
}

macro_rules! b_num_16x4 {
	(0) => {2};
	(1) => {3};
	(2) => {2};
	(3) => {3};
}

macro_rules! b_num_8x4 {
	(0) => {1};
	(1) => {2};
	(2) => {3};
	(3) => {4};
}

// ***************************** 16x4 ******************************* //
macro_rules! asm_16x4_step {
	($N:tt, $nr:tt, $a_layout:tt, $b_layout:tt, $K:tt) => {
		seq!(n in 0..$nr {
			concat!(
				load_a!($N, $a_layout, $K),
				#(
					load_b!($b_layout, n, $K, $nr, b_num_16x4!(n)),
					fmadd_2v!(n),
				)*
			)
		})
	};
}

// ***************************** 8x4 ******************************* //
macro_rules! asm_8x4_step {
	($N:tt, $nr:tt, $a_layout:tt, $b_layout:tt, $K:tt) => {
		seq!(n in 0..$nr {
			concat!(
				load_a!($N, $a_layout, $K),
				#(
					load_b!($b_layout, n, $K, $nr, b_num_8x4!(n)),
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
    	pub(crate) unsafe fn $func_name<F: MyFn>(
        	a: *const TA, b: *const TB, c: *mut TC,
        	alpha: *const f32, beta: *const f32,
        	k: usize,
        	ldc: usize,
			ld_arr: [usize; 2],
			f: F,
    	) {
			// let beta_val = *beta;
			// let beta_val = 1.0;
			// let alpha = &beta_val;
			// println!("alpha, beta: {}, {}", *alpha, *beta);
			// let c_slice = std::slice::from_raw_parts_mut(c, $mr);
			// println!("c: {:?}", c_slice);
			let k = (k+1) / 2 *2;
        	let k_iter = k / 8;
        	let k_left = (k % 8) / 2;
            let u64_arr = [ld_arr[0], ld_arr[1], ldc, k_iter, k_left];
        	let u64_ptr = u64_arr.as_ptr();
			let one = 1_f32;
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
				// jmp to 8 if alpha is equal to onex
				"vmovss ({alphax}), %xmm0",
				"vucomiss ({onex}), %xmm0",
				"je 8f",
            	// scale by alpha
				// "/* {alphax} */",
            	asm_alpha_scale!($VER, $nr),

				"8:",

            	"vbroadcastss ({betax}), %ymm0",

            	"vxorps %ymm3,%ymm3,%ymm3",
            	"vucomiss %xmm3,%xmm0",

            	// 6 -> BETAZERO
            	"je 6f",

				// check if beta is equal to 1
				// "vucomiss ({onex}), %xmm0",
				// "je 9f",

            	// $asm_acc_macro!($mr,$nr,C,2),

				"9:",
				// 9 -> BETA ONE
				$asm_acc_macro!($mr,$nr,C,1),

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
			// println!("c: {:?}", c_slice);

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
    	pub(crate) unsafe fn $func_name<F: MyFn>(
        	a: *const TA, b: *const TB, c: *mut TC,
        	alpha: *const f32, beta: *const f32,
        	k: usize,
        	ldc: usize,
			ld_arr: [usize; 2],
			mask: *const u32,
			f: F,
    	) {
			let k = (k+1) / 2 *2;
        	let k_iter = k / 8;
        	let k_left = (k % 8) / 2;
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
				"/* {alphax} */",
            	// asm_alpha_scale!($VER, $nr),

            	"vbroadcastss ({betax}), %ymm0",

            	"vxorps %ymm3,%ymm3,%ymm3",
            	"vucomiss %xmm3,%xmm0",
				"vmovdqu ({maskx}), %ymm1",
				// "/* {maskx}*/",

            	// 6 -> BETAZERO
            	"je 6f",
            	$asm_acc_macro!($mr,$nr,M,1),

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
					0, 128,
					4,
					[<ukernel_ $mr x nr _ $func_name>]
				);
			}
		});
	};
}

group_def_ukernel!(16, 1, 4, B, B, bb, def_ukernel, asm_16x4_step, asm_16x4_acc, asm_16x4_store, VER16);
// group_def_ukernel!(16, 1, 4, B, S, bs, def_ukernel, asm_16x4_step, asm_16x4_acc, asm_16x4_store, VER16);
group_def_ukernel!(16, 1, 4, B, B, bb_partial, def_ukernel_partial, asm_16x4_step, asm_16x4_acc, asm_16x4_store, VER16);
// group_def_ukernel!(16, 1, 4, B, S, bs_partial, def_ukernel_partial, asm_16x4_step, asm_16x4_acc, asm_16x4_store, VER16);

group_def_ukernel!(8, 1, 4, B, B, bb, def_ukernel, asm_8x4_step, asm_8x4_acc, asm_8x4_store, VER8);
// group_def_ukernel!(8, 1, 4, B, S, bs, def_ukernel, asm_8x4_step, asm_8x4_acc, asm_8x4_store, VER8);
group_def_ukernel!(8, 1, 4, B, B, bb_partial, def_ukernel_partial, asm_8x4_step, asm_8x4_acc, asm_8x4_store, VER8);
// group_def_ukernel!(8, 1, 4, B, S, bs_partial, def_ukernel_partial, asm_8x4_step, asm_8x4_acc, asm_8x4_store, VER8);

