use image::{DynamicImage, ImageFormat};
use serde::{Deserialize, Serialize};
use std::io::{Cursor, Read, Write};

/// Universal Data Container
/// Capable of holding any format the vGPU might encounter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VData {
    /// Structural Data (JSON-compatible)
    Structure(serde_json::Value),
    /// Tabular Data (Matrix/CSV) - Row-major
    Table {
        headers: Vec<String>,
        rows: Vec<Vec<String>>,
    },
    /// Dense Numerical Data (Tensor)
    Tensor { shape: Vec<u64>, data: Vec<f32> },
    /// Visual Data (Image)
    Image {
        width: u32,
        height: u32,
        rgba: Vec<u8>,
    },
    /// Raw Bytes (Fallback)
    Binary(Vec<u8>),
}

#[derive(Debug)]
pub enum Format {
    Json,
    Csv,
    Png,
    Npy, // Numpy
    Raw,
}

#[derive(Debug)]
pub enum VioError {
    Io(std::io::Error),
    Json(serde_json::Error),
    Csv(csv::Error),
    Image(image::ImageError),
    Npy(String),
    DimensionMismatch,
    UnsupportedFormat,
}

impl From<std::io::Error> for VioError {
    fn from(e: std::io::Error) -> Self {
        VioError::Io(e)
    }
}
impl From<serde_json::Error> for VioError {
    fn from(e: serde_json::Error) -> Self {
        VioError::Json(e)
    }
}
impl From<csv::Error> for VioError {
    fn from(e: csv::Error) -> Self {
        VioError::Csv(e)
    }
}
impl From<image::ImageError> for VioError {
    fn from(e: image::ImageError) -> Self {
        VioError::Image(e)
    }
}

/// The Universal Codec
pub struct UniversalCodec;

impl UniversalCodec {
    /// Ingest data from a raw byte stream into VData
    pub fn import(data: &[u8], format: Format) -> Result<VData, VioError> {
        match format {
            Format::Json => {
                let v: serde_json::Value = serde_json::from_slice(data)?;
                Ok(VData::Structure(v))
            }
            Format::Csv => {
                let mut rdr = csv::Reader::from_reader(data);
                let headers = rdr.headers()?.iter().map(|s| s.to_string()).collect();
                let mut rows = Vec::new();
                for result in rdr.records() {
                    let record = result?;
                    rows.push(record.iter().map(|s| s.to_string()).collect());
                }
                Ok(VData::Table { headers, rows })
            }
            Format::Png => {
                let img = image::load_from_memory_with_format(data, ImageFormat::Png)?;
                let rgba = img.to_rgba8();
                Ok(VData::Image {
                    width: rgba.width(),
                    height: rgba.height(),
                    rgba: rgba.into_raw(),
                })
            }
            Format::Npy => {
                // Manual NPY Reader (Version 1.0)
                let mut cursor = Cursor::new(data);
                let mut magic = [0u8; 6];
                if cursor.read_exact(&mut magic).is_err() || &magic != b"\x93NUMPY" {
                    return Err(VioError::Npy("Invalid Magic".to_string()));
                }

                // Version
                let mut ver = [0u8; 2];
                cursor.read_exact(&mut ver)?;

                // Header Len
                let mut len_bytes = [0u8; 2];
                cursor.read_exact(&mut len_bytes)?;
                let header_len = u16::from_le_bytes(len_bytes) as usize;

                // Header String
                let mut header_buf = vec![0u8; header_len];
                cursor.read_exact(&mut header_buf)?;
                let header_str = String::from_utf8_lossy(&header_buf);

                // Parse 'shape': (16, 16),
                // Find "shape" key
                let shape_key = "'shape':";
                let shape_start = header_str
                    .find(shape_key)
                    .ok_or(VioError::Npy("Missing shape key".to_string()))?
                    + shape_key.len();

                // Find start of tuple '('
                let tuple_start = header_str[shape_start..]
                    .find('(')
                    .ok_or(VioError::Npy("Missing shape tuple start".to_string()))?
                    + shape_start
                    + 1;

                // Find end of tuple ')'
                let tuple_end = header_str[tuple_start..]
                    .find(')')
                    .ok_or(VioError::Npy("Missing shape tuple end".to_string()))?
                    + tuple_start;

                let shape_content = &header_str[tuple_start..tuple_end];

                let mut shape = Vec::new();
                for dim in shape_content.split(',') {
                    let d = dim.trim();
                    if !d.is_empty() {
                        if let Ok(val) = d.parse::<u64>() {
                            shape.push(val);
                        }
                    }
                }

                // Data (Assume Little Endian f32)
                let mut data_vec = Vec::new();
                let mut f32_buf = [0u8; 4];
                while cursor.read_exact(&mut f32_buf).is_ok() {
                    data_vec.push(f32::from_le_bytes(f32_buf));
                }

                Ok(VData::Tensor {
                    shape,
                    data: data_vec,
                })
            }
            Format::Raw => Ok(VData::Binary(data.to_vec())),
        }
    }

    /// Export VData back to a raw byte stream
    pub fn export(data: &VData, format: Format) -> Result<Vec<u8>, VioError> {
        match (data, format) {
            (VData::Structure(v), Format::Json) => {
                let bytes = serde_json::to_vec(v)?;
                Ok(bytes)
            }
            (VData::Table { headers, rows }, Format::Csv) => {
                let mut wtr = csv::Writer::from_writer(Vec::new());
                wtr.write_record(headers)?;
                for row in rows {
                    wtr.write_record(row)?;
                }
                Ok(wtr
                    .into_inner()
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?)
            }
            (
                VData::Image {
                    width,
                    height,
                    rgba,
                },
                Format::Png,
            ) => {
                let mut bytes = Vec::new();
                let img_buffer = image::RgbaImage::from_raw(*width, *height, rgba.clone())
                    .ok_or(VioError::DimensionMismatch)?;
                let dyn_img = DynamicImage::ImageRgba8(img_buffer);
                let mut cursor = Cursor::new(&mut bytes);
                dyn_img.write_to(&mut cursor, ImageFormat::Png)?;
                Ok(bytes)
            }
            (VData::Tensor { shape, data }, Format::Npy) => {
                // Manual NPY Writing to support Arbitrary Dimensions
                let mut bytes = Vec::new();
                bytes.write_all(b"\x93NUMPY\x01\x00")?;

                let shape_str = if shape.len() == 1 {
                    format!("({},)", shape[0])
                } else {
                    format!(
                        "({})",
                        shape
                            .iter()
                            .map(|d| d.to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                };

                // Check if we need comma for single element > 1?? No, handled above.
                // Re-check single-element tuple logic: `(16)` is not a tuple in Python, `(16,)` is.
                // My logic handles 1 element correctly. Multi-element doesn't need trailing comma necessarily but Python allows it.

                let header_dict = format!(
                    "{{'descr': '<f4', 'fortran_order': False, 'shape': {}, }}",
                    shape_str
                );

                let mut header_bytes = header_dict.into_bytes();
                // Padding: Total length (10 + len) must be divisible by 64? Or just aligned?
                // Standard: 10 + 2 + len divisible by 16. (Wait, let's use 64 to be safe)
                while (10 + 2 + header_bytes.len()) % 64 != 0 {
                    header_bytes.push(b' ');
                }
                if let Some(last) = header_bytes.last_mut() {
                    *last = b'\n';
                }

                let header_len = header_bytes.len() as u16;
                bytes.write_all(&header_len.to_le_bytes())?;
                bytes.write_all(&header_bytes)?;

                for val in data {
                    bytes.write_all(&val.to_le_bytes())?;
                }

                Ok(bytes)
            }
            (VData::Binary(b), Format::Raw) => Ok(b.clone()),
            _ => Err(VioError::UnsupportedFormat),
        }
    }
}
