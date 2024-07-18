#[macro_export]
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

#[macro_export]
macro_rules! vmovps {
    (B) => {
        "vmovaps "
    };
    ($layout:tt) => {
        "vmovups "
    };
 }



 #[macro_export]
 macro_rules! mem {
    ($m0:tt, $b0:tt) => {
        concat!($b0, "+", $m0)
    };
 }
 
 
 
#[macro_export]
 macro_rules! loadps_unit {
    (8, $layout:tt, $m0:expr, $r1:expr) => {
        concat!(
            vmovps!($layout), $m0, ",%ymm", $r1, "\n",
        )
    };
    (4, $layout:tt, $m0:expr, $r1:expr) => {
        concat!(
            vmovps!($layout), $m0, ",", "%xmm", $r1, "\n",
        )
    };
    (2, $layout:tt, $m0:expr, $r1:expr) => {
        concat!(
            "vmovsd ", $m0, ",", "%xmm", $r1, "\n",
        )
    };
    (1, $layout:tt, $m0:expr, $r1:expr) => {
        concat!(
            "vmovss ", $m0, ",", "%xmm", $r1, "\n",
        )
    };
 }
 
 
 
 

 // using clang 17.0.1
 #[macro_export]
 macro_rules! loadps {
    (24, $layout:tt, $m0:expr, $r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            loadps_unit!(8, $layout, $m0, $r1), "\n",
            loadps_unit!(8, $layout, mem!($m0, "0x20"), $r2), "\n",
            loadps_unit!(8, $layout, mem!($m0, "0x40"), $r3), "\n",
        )
    };
    (24, $layout:tt, $m0:expr) => {
        concat!(
            loadps_unit!(8, $layout, $m0, 0), "\n",
            loadps_unit!(8, $layout, mem!($m0, "0x20"), 1), "\n",
            loadps_unit!(8, $layout, mem!($m0, "0x40"), 2), "\n",
        )
    };
    (20, $layout:tt, $m0:expr,  $r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            loadps_unit!(8, $layout, $m0, $r1), "\n",
            loadps_unit!(8, $layout, mem!($m0, "0x20"), $r2), "\n",
            loadps_unit!(4, $layout, mem!($m0, "0x40"), $r3),
        )
    };
    (16, $layout:tt, $m0:expr, $r1:expr, $r2:expr) => {
        concat!(
            loadps_unit!(8, $layout, $m0, $r1), "\n",
            loadps_unit!(8, $layout, mem!($m0, "0x20"), $r2), "\n",
        )
    };
    (16, $layout:tt, $m0:expr) => {
        concat!(
            loadps_unit!(8, $layout, $m0, 0), "\n",
            loadps_unit!(8, $layout, mem!($m0, "0x20"), 1), "\n",
        )
    };
    (12, $layout:tt, $m0:expr, $r1:expr, $r2:expr) => {
        concat!(
            loadps_unit!(8, $layout, $m0, 0), "\n",
            loadps_unit!(4, $layout, mem!($m0, "0x20"), 1), "\n",
        )
    };
    ($N:tt, $layout:tt, $m0:expr, $r1:expr) => {
        concat!(
            loadps_unit!($N, $layout, $m0, $r1),
        )
    };
    ($N:tt, $layout:tt, $m0:expr) => {
        concat!(
            loadps_unit!($N, $layout, $m0, 0),
        )
    };
 }

#[macro_export]
 macro_rules! storeps_unit {
    (8, C, $r1:expr, $m0:expr) => {
        concat!(
            "vmovups %ymm", $r1, ", ", $m0,  "\n",
        )
    };
    (8, B, $r1:expr, $m0:expr) => {
        concat!(
            "vmovaps %ymm", $r1, ", ", $m0,  "\n",
        )
    };
    (8, M, $r1:expr, $m0:expr) => {
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
    (2, $layout:tt, $r1:expr, $m0:expr) => {
        concat!(
            "vmovlps %xmm", $r1, ",", $m0, "\n",
        )
    };
    (1, $layout:tt, $r1:expr, $m0:expr) => {
        concat!(
            "vmovss %xmm", $r1, ",", $m0, "\n",
        )
    };
 }
 
 
 
 
#[macro_export]
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
 

 
 
#[macro_export]
macro_rules! storeps {
   (24, $layout:tt, $m0:expr, $r1:expr, $r2:expr, $r3:expr) => {
       concat!(
           storeps_unit!(8, C, $r1, $m0),
           storeps_unit!(8, C, $r2, mem!($m0, "0x20")),
           storeps_unit!(8, $layout, $r3, mem!($m0, "0x40")),
       )
   };
   (20, $layout:tt, $m0:expr, $r1:expr, $r2:expr, $r3:expr) => {
       concat!(
           storeps_unit!(8, C, $r1, $m0),
           storeps_unit!(8, C, $r2, mem!($m0, "0x20")),
           storeps_unit!(4, $layout, $r3, mem!($m0, "0x40")),
       )
   };
   (16, $layout:tt, $m0:expr, $r1:expr, $r2:expr) => {
       concat!(
           storeps_unit!(8, C, $r1, $m0),
           storeps_unit!(8, $layout, $r2, mem!($m0, "0x20")),
       )
   };
   (12, $layout:tt, $m0:expr, $r1:expr, $r2:expr) => {
       concat!(
           storeps_unit!(8, C, $r1, $m0),
           storeps_unit!(4, $layout, $r2, mem!($m0, "0x20")),
       )
   };
   ($N:tt, $layout:tt, $m0:expr, $r1:expr) => {
       concat!(
           storeps_unit!($N, $layout, $r1, $m0),
       )
   };
}


 
#[macro_export]
 macro_rules! beta_fmaddps {
    (C, $m0:expr, $r1:expr) => {
        concat!(
            "vfmadd231ps ", $m0, ",%ymm0,%ymm", $r1, "\n",
            // "vaddps ", "%ymm", $r1, ", %ymm", $r1, ", ", $m0, "\n",
            // "vaddps ", $m0, ",%ymm", $r1, ",%ymm", $r1, "\n",
        ) 
    };
    (M, $m0:expr, $r1:expr) => {
        concat!(
            "vmaskmovps ", $m0, ", %ymm1", ", %ymm2",  "\n",
            "vfmadd231ps %ymm2, %ymm0,%ymm", $r1, "\n",
            // "vaddps ", "%ymm", $r1, ", %ymm", $r1, ", ", $m0, "\n",
            // "vaddps ", $m0, ",%ymm", $r1, ",%ymm", $r1, "\n",
        ) 
    };
 }

 #[macro_export]
 macro_rules! vfmadd231ps {
    ($r1:expr, $r2:expr, $r3:expr) => {
        concat!(
            "vfmadd231ps %ymm", $r1, ", %ymm", $r2,", %ymm", $r3, "\n",
            // "vaddps ", "%ymm", $r1, ", %ymm", $r1, ", ", $m0, "\n",
            // "vaddps ", $m0, ",%ymm", $r1, ",%ymm", $r1, "\n",
        ) 
    };
 }
 
 
 

pub use vzeroall;
pub use vmovps;
pub use loadps;
pub use loadps_unit;
pub use storeps;
pub use acc_ps;
pub use storeps_unit;
pub use beta_fmaddps;
pub use mem;
pub use vfmadd231ps;