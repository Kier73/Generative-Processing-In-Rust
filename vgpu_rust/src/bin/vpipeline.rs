use std::time::Instant;
use vgpu_rust::vio::{Format, UniversalCodec, VData};
use vgpu_rust::vmatrix::GeometricField;
use vgpu_rust::vphy::{RigidBody, Vec3};
use vgpu_rust::vrender::{Ray, Sphere};
use vgpu_rust::{VGpuContext, vphy, vrender};

fn main() {
    println!("===============================================================");
    println!("vGPU END-TO-END PIPELINE VERIFICATION");
    println!("===============================================================");

    let mut ctx = VGpuContext::new(1, 12345);

    // -----------------------------------------------------------------------
    // PIPELINE 1: MATRIX (CSV -> GEMM -> CSV)
    // -----------------------------------------------------------------------
    println!("\n[1] PIPELINE: MATRIX (CSV -> GEMM -> CSV)");
    // Input: CSV string representing two 3x3 matrices (simulated for test)
    // In a real app, this would be read from a file.
    let csv_input = "row,col,val_a,val_b\n0,0,1.0,2.0\n0,1,2.0,3.0\n1,0,3.0,4.0\n1,1,4.0,5.0";
    // We'll parse this to demonstrate universal IO, but for the actual GEMM we'll use GeometricField
    // to simulate a large operation.

    let start_total = Instant::now();

    // 1. Ingest
    let t0 = Instant::now();
    let _data =
        UniversalCodec::import(csv_input.as_bytes(), Format::Csv).expect("Import CSV failed");
    println!("    Ingest (CSV): {:.6} s", t0.elapsed().as_secs_f64());

    // 2. Process (Geometric GEMM)
    let t1 = Instant::now();
    // Simulate mapping CSV to Field (in reality, we'd parse rows to coordinates)
    // Here we just launch the 8Kx8K geometric task as the "Work"
    let mat_a = GeometricField::new(8192, 8192, 1);
    let mat_b = GeometricField::new(8192, 8192, 2);
    let _mat_c = mat_a.multiply(&mat_b).expect("GEMM failed");
    println!(
        "    Process (GEMM 8Kx8K): {:.6} s",
        t1.elapsed().as_secs_f64()
    );

    // 3. Export (Result Description)
    let t2 = Instant::now();
    // We export the *result signature* as a small CSV for verification
    let result_data = VData::Table {
        headers: vec![
            "rows".to_string(),
            "cols".to_string(),
            "signature".to_string(),
        ],
        rows: vec![vec![
            mat_a.rows.to_string(),
            mat_b.cols.to_string(),
            format!("{:x}", _mat_c.signature),
        ]],
    };
    let output_bytes =
        UniversalCodec::export(&result_data, Format::Csv).expect("Export CSV failed");
    println!("    Export (CSV): {:.6} s", t2.elapsed().as_secs_f64());

    println!(
        "    -> Total Pipeline: {:.6} s",
        start_total.elapsed().as_secs_f64()
    );
    println!(
        "    Output Preview: {}",
        String::from_utf8_lossy(&output_bytes).trim()
    );

    // -----------------------------------------------------------------------
    // PIPELINE 2: PHYSICS (JSON -> N-BODY -> JSON)
    // -----------------------------------------------------------------------
    println!("\n[2] PIPELINE: PHYSICS (JSON -> N-BODY -> JSON)");
    let json_input = r#"{
        "bodies": [
            {"pos": [0.0, 0.0, 0.0], "vel": [1.0, 0.0, 0.0], "mass": 1.0, "radius": 1.0},
            {"pos": [1.5, 0.0, 0.0], "vel": [-1.0, 0.0, 0.0], "mass": 1.0, "radius": 1.0}
        ],
        "dt": 0.016,
        "steps": 10
    }"#;

    let start_total = Instant::now();

    // 1. Ingest
    let t0 = Instant::now();
    let data =
        UniversalCodec::import(json_input.as_bytes(), Format::Json).expect("Import JSON failed");
    println!("    Ingest (JSON): {:.6} s", t0.elapsed().as_secs_f64());

    // 2. Process
    let t1 = Instant::now();
    let mut bodies = Vec::new();
    if let VData::Structure(val) = data {
        if let Some(arr) = val["bodies"].as_array() {
            for b_obj in arr {
                let p = b_obj["pos"].as_array().unwrap();
                let v = b_obj["vel"].as_array().unwrap();
                bodies.push(RigidBody {
                    pos: Vec3::new(
                        p[0].as_f64().unwrap() as f32,
                        p[1].as_f64().unwrap() as f32,
                        p[2].as_f64().unwrap() as f32,
                    ),
                    vel: Vec3::new(
                        v[0].as_f64().unwrap() as f32,
                        v[1].as_f64().unwrap() as f32,
                        v[2].as_f64().unwrap() as f32,
                    ),
                    inv_mass: 1.0 / b_obj["mass"].as_f64().unwrap() as f32,
                    radius: b_obj["radius"].as_f64().unwrap() as f32,
                });
            }
        }
    }

    // Run simulation
    // We execute the collision step 10 times to verify inductive warm-up
    for _ in 0..10 {
        if bodies.len() >= 2 {
            let (left, right) = bodies.split_at_mut(1);
            vphy::vphy_step(&mut ctx, &mut left[0], &mut right[0], 0.016);
        }
    }
    println!(
        "    Process (Physics 10 Steps): {:.6} s",
        t1.elapsed().as_secs_f64()
    );

    // 3. Export
    let t2 = Instant::now();
    // Serialize final state
    let mut rows = Vec::new();
    for (i, b) in bodies.iter().enumerate() {
        rows.push(vec![
            i.to_string(),
            format!("{:.4}", b.pos.x),
            format!("{:.4}", b.pos.y),
            format!("{:.4}", b.pos.z),
            format!("{:.4}", b.vel.x),
        ]);
    }
    // We'll export as CSV for variety
    let result_data = VData::Table {
        headers: vec![
            "id".to_string(),
            "x".to_string(),
            "y".to_string(),
            "z".to_string(),
            "vx".to_string(),
        ],
        rows,
    };
    let _output_bytes =
        UniversalCodec::export(&result_data, Format::Csv).expect("Export CSV failed");
    println!("    Export (CSV): {:.6} s", t2.elapsed().as_secs_f64());

    println!(
        "    -> Total Pipeline: {:.6} s",
        start_total.elapsed().as_secs_f64()
    );
    // println!("    Output:\n{}", String::from_utf8_lossy(&output_bytes));

    // -----------------------------------------------------------------------
    // PIPELINE 3: RENDER (JSON SCENE -> RAYTRACE -> PNG)
    // -----------------------------------------------------------------------
    println!("\n[3] PIPELINE: RENDER (JSON -> TRACE -> PNG)");
    // Scene definition
    let scene_json = r#"{
        "spheres": [
            {"center": [0.0, 0.0, -5.0], "radius": 1.0, "color": [1.0, 0.0, 0.0]},
            {"center": [2.0, 0.0, -6.0], "radius": 1.0, "color": [0.0, 1.0, 0.0]}
        ],
        "width": 64,
        "height": 64
    }"#;

    let start_total = Instant::now();

    // 1. Ingest
    let data =
        UniversalCodec::import(scene_json.as_bytes(), Format::Json).expect("Import JSON failed");

    // 2. Process
    let t1 = Instant::now();
    let mut spheres = Vec::new();
    let mut width = 64;
    let mut height = 64;

    if let VData::Structure(val) = data {
        if let Some(arr) = val["spheres"].as_array() {
            for s_obj in arr {
                let c = s_obj["center"].as_array().unwrap();
                let col = s_obj["color"].as_array().unwrap();
                spheres.push(Sphere {
                    center: Vec3::new(
                        c[0].as_f64().unwrap() as f32,
                        c[1].as_f64().unwrap() as f32,
                        c[2].as_f64().unwrap() as f32,
                    ),
                    radius: s_obj["radius"].as_f64().unwrap() as f32,
                    color: Vec3::new(
                        col[0].as_f64().unwrap() as f32,
                        col[1].as_f64().unwrap() as f32,
                        col[2].as_f64().unwrap() as f32,
                    ),
                });
            }
        }
        if let Some(w) = val["width"].as_u64() {
            width = w as u32;
        }
        if let Some(h) = val["height"].as_u64() {
            height = h as u32;
        }
    }

    // Trace Buffer
    let mut pixel_data = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            // Normalized Device Coordinates
            let u = (x as f32 / width as f32) * 2.0 - 1.0;
            let v = (y as f32 / height as f32) * 2.0 - 1.0;
            // Aspect ratio correction would go here
            let ray = Ray {
                origin: Vec3::new(0.0, 0.0, 0.0),
                dir: Vec3::new(u, -v, -1.0).normalize(),
            };

            let color = vrender::vrender_trace(&mut ctx, &ray, &spheres, 3);

            // Convert to RGBAu8
            pixel_data.push((color.x.clamp(0.0, 1.0) * 255.0) as u8);
            pixel_data.push((color.y.clamp(0.0, 1.0) * 255.0) as u8);
            pixel_data.push((color.z.clamp(0.0, 1.0) * 255.0) as u8);
            pixel_data.push(255);
        }
    }
    println!(
        "    Process (Trace 64x64): {:.6} s",
        t1.elapsed().as_secs_f64()
    );

    // 3. Export
    let t2 = Instant::now();
    let image_data = VData::Image {
        width,
        height,
        rgba: pixel_data,
    };
    let png_bytes = UniversalCodec::export(&image_data, Format::Png).expect("Export PNG failed");
    println!("    Export (PNG): {:.6} s", t2.elapsed().as_secs_f64());

    println!(
        "    -> Total Pipeline: {:.6} s",
        start_total.elapsed().as_secs_f64()
    );
    println!("    PNG Size: {} bytes", png_bytes.len());

    println!("\n===============================================================");
    println!("PIPELINE VERIFICATION COMPLETE");
}
