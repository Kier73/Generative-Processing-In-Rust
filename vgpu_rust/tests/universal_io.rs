use vgpu_rust::VGpuContext;
use vgpu_rust::vio::{Format, UniversalCodec, VData};
use vgpu_rust::vmatrix::GeometricField;

#[test]
fn test_json_roundtrip() {
    let json_str = r#"{"name": "vGPU", "version": 1.0, "active": true}"#;
    let data =
        UniversalCodec::import(json_str.as_bytes(), Format::Json).expect("Import JSON failed");

    if let VData::Structure(v) = &data {
        assert_eq!(v["name"], "vGPU");
    } else {
        panic!("Wrong VData type");
    }

    let exported = UniversalCodec::export(&data, Format::Json).expect("Export JSON failed");
    let imported_again =
        UniversalCodec::import(&exported, Format::Json).expect("Re-import JSON failed");

    if let VData::Structure(v) = imported_again {
        assert_eq!(v["version"], 1.0);
    }
}

#[test]
fn test_csv_roundtrip() {
    let csv_data = "col1,col2\nval1,val2\nval3,val4";
    let data = UniversalCodec::import(csv_data.as_bytes(), Format::Csv).expect("Import CSV failed");

    if let VData::Table { headers, rows } = &data {
        assert_eq!(headers[0], "col1");
        assert_eq!(rows[0][1], "val2");
    } else {
        panic!("Wrong VData type");
    }

    let exported = UniversalCodec::export(&data, Format::Csv).expect("Export CSV failed");
    let csv_string = String::from_utf8(exported).unwrap();
    assert!(csv_string.contains("val3,val4"));
}

#[test]
fn test_image_roundtrip() {
    // Create a 2x2 image
    let rgba = vec![
        255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
    ];
    let data = VData::Image {
        width: 2,
        height: 2,
        rgba: rgba.clone(),
    };

    // Export to PNG
    let png_bytes = UniversalCodec::export(&data, Format::Png).expect("Export PNG failed");

    // Import back
    let imported = UniversalCodec::import(&png_bytes, Format::Png).expect("Import PNG failed");

    if let VData::Image {
        width,
        height,
        rgba: imported_rgba,
    } = imported
    {
        assert_eq!(width, 2);
        assert_eq!(height, 2);
        assert_eq!(imported_rgba, rgba);
    } else {
        panic!("Wrong VData type");
    }
}

#[test]
fn test_tensor_geometric_roundtrip() {
    // 1. Generate Matrix Data using vGPU
    let mut ctx = VGpuContext::new(1, 42);
    let field = GeometricField::new(16, 16, 123);
    let mut buffer = vec![0.0; 256];
    field.resolve_bulk(&mut ctx, &mut buffer);

    let data = VData::Tensor {
        shape: vec![16, 16],
        data: buffer.clone(),
    };

    // 2. Export to NPY
    let npy_bytes = UniversalCodec::export(&data, Format::Npy).expect("Export NPY failed");

    // 3. Import back
    let imported = UniversalCodec::import(&npy_bytes, Format::Npy).expect("Import NPY failed");

    if let VData::Tensor {
        shape,
        data: imported_data,
    } = imported
    {
        // Limitation: npy codec manually written should preserve shape!
        assert_eq!(shape, vec![16, 16]);
        assert_eq!(imported_data.len(), 256);
        assert_eq!(imported_data, buffer);
    } else {
        panic!("Wrong VData type");
    }
}
